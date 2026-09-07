"""Board representation, move generation, make/unmake.

Everything here is jitted and operates on plain NumPy arrays rather than objects. Numba can pass
arrays into nopython code at near-zero cost; a class instance would either force object mode or
need a jitclass, which compiles slowly and inlines poorly. So a position is a handful of arrays
threaded through every function, and the search allocates them once.

Layout
------
``bb``       uint64[15]  twelve piece bitboards, then white / black / all occupancy
``mailbox``  int8[64]    piece index on each square, or EMPTY -- makes captures O(1)
``state``    int64[5]    side to move, castling rights, en-passant square, halfmove, fullmove
``key``      uint64[1]   the running Zobrist hash

Squares are 0 = a1 to 63 = h8, so ``rank = square >> 3`` and ``file = square & 7``, and white
pawns advance by +8. Moves pack into an int32 with the conventional layout: from in bits 0-5, to
in bits 6-11, and a four-bit flag in 12-15 chosen so that bit 14 means capture and bit 15 means
promotion, which makes the two hottest tests in move ordering a single mask each.

Generation is pseudo-legal: moves that leave the king in check are made, detected and unmade.
That costs a little against fully legal generation but removes the pin and discovered-check logic
that is where movegen bugs actually live, and perft is the arbiter of whether it is right.

Beware of one numba trap throughout: mixing uint64 with a signed integer promotes to float64 and
silently destroys the low bits. Every shift and mask below keeps both operands uint64, and every
array index is int64.
"""

import numpy as np
from numba import njit

from engine.bitboard import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    bishop_attacks,
    rook_attacks,
)

U64 = np.uint64
I64 = np.int64

# Numba works in machine integers: every jitted helper here returns np.int64, not a Python int,
# and those values then flow straight back in as arguments. Spelling that out as `int | np.int64`
# on each parameter would be pure noise about a distinction numba erases at compile time, so the
# concept gets one name and the signatures stay readable.
Int = int | np.int64

# piece indices: white pawn..king 0-5, black pawn..king 6-11
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
EMPTY = 12

WHITE_OCC, BLACK_OCC, ALL_OCC = 12, 13, 14

# state slots
STM, CASTLE, EP, HALFMOVE, FULLMOVE = 0, 1, 2, 3, 4

# castling rights bits
CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

# move flags
QUIET = 0
DOUBLE_PUSH = 1
CASTLE_KING = 2
CASTLE_QUEEN = 3
CAPTURE = 4
EP_CAPTURE = 5
PROMO_N, PROMO_B, PROMO_R, PROMO_Q = 8, 9, 10, 11
PROMO_CAPTURE_N, PROMO_CAPTURE_B, PROMO_CAPTURE_R, PROMO_CAPTURE_Q = 12, 13, 14, 15

CAPTURE_BIT = 4
PROMO_BIT = 8

MAX_MOVES = 256
MAX_PLY = 256

PIECE_CHARS = "PNBRQKpnbrqk"

# Castling rights survive a move unless it touches one of the six squares that matter. Masking
# with the from-square and the to-square handles every case at once, including a rook captured
# on its home square, which is the one that gets forgotten.
_CASTLE_MASK = np.full(64, 15, dtype=I64)
_CASTLE_MASK[4] = 15 & ~(CASTLE_WK | CASTLE_WQ)
_CASTLE_MASK[0] = 15 & ~CASTLE_WQ
_CASTLE_MASK[7] = 15 & ~CASTLE_WK
_CASTLE_MASK[60] = 15 & ~(CASTLE_BK | CASTLE_BQ)
_CASTLE_MASK[56] = 15 & ~CASTLE_BQ
_CASTLE_MASK[63] = 15 & ~CASTLE_BK
CASTLE_MASK = _CASTLE_MASK

# Zobrist keys. Fixed seed so a hash is reproducible across runs, which matters when a search bug
# only shows up on one position and the transposition table has to be inspected.
_rng = np.random.default_rng(0x5EED_C4E5)
PIECE_KEYS = _rng.integers(0, 1 << 64, size=(12, 64), dtype=np.uint64)
CASTLE_KEYS = _rng.integers(0, 1 << 64, size=16, dtype=np.uint64)
EP_FILE_KEYS = _rng.integers(0, 1 << 64, size=8, dtype=np.uint64)
SIDE_KEY = U64(_rng.integers(0, 1 << 64, dtype=np.uint64))

# De Bruijn sequence for a branch-free bit scan. Numba has no ctz intrinsic exposed, and a loop
# would cost more than the multiply-and-lookup.
DEBRUIJN = U64(0x03F7_9D71_B4CB_0A89)
DEBRUIJN_INDEX = np.zeros(64, dtype=I64)
# The multiply is meant to wrap; numpy would otherwise warn on every import.
with np.errstate(over="ignore"):
    for _i in range(64):
        DEBRUIJN_INDEX[int((DEBRUIJN * (U64(1) << U64(_i))) >> U64(58))] = _i


@njit("uint64(int64)", cache=False)
def bit(square: Int) -> U64:
    return U64(1) << U64(square)


@njit("int64(uint64)", cache=False)
def lsb(bitboard: U64) -> Int:
    """Index of the least significant set bit. Undefined for zero, so never call it on zero."""
    isolated = bitboard & (~bitboard + U64(1))
    return I64(DEBRUIJN_INDEX[I64((isolated * DEBRUIJN) >> U64(58))])


@njit("int64(uint64)", cache=False)
def popcount(bitboard: U64) -> Int:
    """SWAR population count."""
    x = bitboard - ((bitboard >> U64(1)) & U64(0x5555555555555555))
    x = (x & U64(0x3333333333333333)) + ((x >> U64(2)) & U64(0x3333333333333333))
    x = (x + (x >> U64(4))) & U64(0x0F0F0F0F0F0F0F0F)
    return I64((x * U64(0x0101010101010101)) >> U64(56))


@njit("int32(int64, int64, int64)", cache=False)
def encode(from_square: Int, to_square: Int, flag: Int) -> np.int32:
    return np.int32(from_square | (to_square << 6) | (flag << 12))


@njit("int64(int32)", cache=False)
def move_from(move: np.int32) -> Int:
    return I64(move & 63)


@njit("int64(int32)", cache=False)
def move_to(move: np.int32) -> Int:
    return I64((move >> 6) & 63)


@njit("int64(int32)", cache=False)
def move_flag(move: np.int32) -> Int:
    return I64((move >> 12) & 15)


def new_position() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate the arrays for one position. The search reuses these; nothing here is per-move."""
    bb = np.zeros(15, dtype=U64)
    mailbox = np.full(64, EMPTY, dtype=np.int8)
    state = np.zeros(5, dtype=I64)
    key = np.zeros(1, dtype=U64)
    return bb, mailbox, state, key


def new_undo() -> tuple[np.ndarray, np.ndarray]:
    """Per-ply save slots: captured piece, castling, en passant, halfmove -- plus the old key."""
    return np.zeros((MAX_PLY, 4), dtype=I64), np.zeros(MAX_PLY, dtype=U64)


def compute_key(bb: np.ndarray, state: np.ndarray) -> U64:
    """Hash from scratch. Only used at set_fen and to assert the incremental update in tests."""
    key = U64(0)
    for piece in range(12):
        pieces = int(bb[piece])
        while pieces:
            square = (pieces & -pieces).bit_length() - 1
            key ^= U64(PIECE_KEYS[piece, square])
            pieces &= pieces - 1
    key ^= U64(CASTLE_KEYS[int(state[CASTLE])])
    if state[EP] >= 0:
        key ^= U64(EP_FILE_KEYS[int(state[EP]) & 7])
    if state[STM] == 1:
        key ^= SIDE_KEY
    return key


def set_fen(
    bb: np.ndarray, mailbox: np.ndarray, state: np.ndarray, key: np.ndarray, fen: str
) -> None:
    """Parse a FEN into the arrays. Plain Python: called once per search, never in the tree."""
    bb[:] = 0
    mailbox[:] = EMPTY
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"malformed FEN, need at least four fields: {fen!r}")
    placement, side, castling, ep_field = fields[:4]

    rank = 7
    file = 0
    for char in placement:
        if char == "/":
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            piece = PIECE_CHARS.index(char)
            square = rank * 8 + file
            bb[piece] |= U64(1) << U64(square)
            mailbox[square] = piece
            file += 1

    state[STM] = 0 if side == "w" else 1
    rights = 0
    for char, flag in (("K", CASTLE_WK), ("Q", CASTLE_WQ), ("k", CASTLE_BK), ("q", CASTLE_BQ)):
        if char in castling:
            rights |= flag
    state[CASTLE] = rights
    if ep_field == "-":
        state[EP] = -1
    else:
        square = (int(ep_field[1]) - 1) * 8 + (ord(ep_field[0]) - ord("a"))
        # The pawn that just double-pushed sits behind the marked square. Drop the marker unless a
        # capture is really available, so a FEN and a `make_move` agree on the key -- and so movegen
        # is never offered an en-passant capture nobody can make.
        pawn = square - 8 if state[STM] == 0 else square + 8
        state[EP] = square if _ep_available(bb, pawn, 1 - state[STM]) else -1
    state[HALFMOVE] = int(fields[4]) if len(fields) > 4 else 0
    state[FULLMOVE] = int(fields[5]) if len(fields) > 5 else 1

    refresh_occupancy(bb)
    key[0] = compute_key(bb, state)


def refresh_occupancy(bb: np.ndarray) -> None:
    white = U64(0)
    black = U64(0)
    for piece in range(6):
        white |= bb[piece]
        black |= bb[piece + 6]
    bb[WHITE_OCC] = white
    bb[BLACK_OCC] = black
    bb[ALL_OCC] = white | black


def move_to_uci(move: np.int32) -> str:
    """UCI string for a move. Used at the root only, so it stays in Python."""
    from_square = int(move) & 63
    to_square = (int(move) >> 6) & 63
    flag = (int(move) >> 12) & 15
    text = _square_name(from_square) + _square_name(to_square)
    if flag & PROMO_BIT:
        text += "nbrq"[flag & 3]
    return text


def _square_name(square: Int) -> str:
    return chr(ord("a") + (square & 7)) + str((square >> 3) + 1)


@njit("boolean(uint64[:], int64, int64)", cache=False)
def is_attacked(bb: np.ndarray, square: Int, by_black: Int) -> bool:
    """Is `square` attacked by the given side?

    Pawns use the reverse-colour trick, and the colour index is the one that inverts: `square` is
    attacked by a black pawn exactly when a *white* pawn standing on `square` would attack that
    pawn. So the table is indexed by the defending colour, `1 - by_black`, not the attacking one.
    """
    occupancy = bb[ALL_OCC]
    offset = 6 if by_black == 1 else 0
    if PAWN_ATTACKS[1 - by_black, square] & bb[WP + offset]:
        return True
    if KNIGHT_ATTACKS[square] & bb[WN + offset]:
        return True
    if KING_ATTACKS[square] & bb[WK + offset]:
        return True
    if bishop_attacks(square, occupancy) & (bb[WB + offset] | bb[WQ + offset]):
        return True
    return bool(rook_attacks(square, occupancy) & (bb[WR + offset] | bb[WQ + offset]))


@njit("boolean(uint64[:], int64[:])", cache=False)
def in_check(bb: np.ndarray, state: np.ndarray) -> bool:
    side = state[STM]
    king_square = lsb(bb[WK + 6 * side])
    return is_attacked(bb, king_square, 1 - side)


@njit("int64(uint64[:], int8[:], int64[:], int32[:], int64)", cache=False)
def generate_moves(
    bb: np.ndarray, mailbox: np.ndarray, state: np.ndarray, moves: np.ndarray, offset: Int
) -> Int:
    """Append every pseudo-legal move for the side to move; return how many were written."""
    count = 0
    side = state[STM]
    us = 6 * side
    own = bb[WHITE_OCC + side]
    enemy = bb[BLACK_OCC - side]
    occupancy = bb[ALL_OCC]
    empty = ~occupancy
    ep_square = state[EP]

    # --- pawns -----------------------------------------------------------------------------
    forward = 8 if side == 0 else -8
    start_rank = 1 if side == 0 else 6
    promo_rank = 7 if side == 0 else 0
    pawns = bb[WP + us]
    while pawns:
        square = lsb(pawns)
        pawns &= pawns - U64(1)

        push = square + forward
        if empty & bit(push):
            if (push >> 3) == promo_rank:
                moves[offset + count] = encode(square, push, PROMO_N)
                moves[offset + count + 1] = encode(square, push, PROMO_B)
                moves[offset + count + 2] = encode(square, push, PROMO_R)
                moves[offset + count + 3] = encode(square, push, PROMO_Q)
                count += 4
            else:
                moves[offset + count] = encode(square, push, QUIET)
                count += 1
                double = push + forward
                if (square >> 3) == start_rank and (empty & bit(double)):
                    moves[offset + count] = encode(square, double, DOUBLE_PUSH)
                    count += 1

        attacks = PAWN_ATTACKS[side, square]
        targets = attacks & enemy
        while targets:
            target = lsb(targets)
            targets &= targets - U64(1)
            if (target >> 3) == promo_rank:
                moves[offset + count] = encode(square, target, PROMO_CAPTURE_N)
                moves[offset + count + 1] = encode(square, target, PROMO_CAPTURE_B)
                moves[offset + count + 2] = encode(square, target, PROMO_CAPTURE_R)
                moves[offset + count + 3] = encode(square, target, PROMO_CAPTURE_Q)
                count += 4
            else:
                moves[offset + count] = encode(square, target, CAPTURE)
                count += 1

        if ep_square >= 0 and (attacks & bit(ep_square)):
            moves[offset + count] = encode(square, ep_square, EP_CAPTURE)
            count += 1

    # --- knights ---------------------------------------------------------------------------
    knights = bb[WN + us]
    while knights:
        square = lsb(knights)
        knights &= knights - U64(1)
        targets = KNIGHT_ATTACKS[square] & ~own
        while targets:
            target = lsb(targets)
            targets &= targets - U64(1)
            flag = CAPTURE if (enemy & bit(target)) else QUIET
            moves[offset + count] = encode(square, target, flag)
            count += 1

    # --- bishops and queens (diagonal) -----------------------------------------------------
    diagonal = bb[WB + us] | bb[WQ + us]
    while diagonal:
        square = lsb(diagonal)
        diagonal &= diagonal - U64(1)
        targets = bishop_attacks(square, occupancy) & ~own
        while targets:
            target = lsb(targets)
            targets &= targets - U64(1)
            flag = CAPTURE if (enemy & bit(target)) else QUIET
            moves[offset + count] = encode(square, target, flag)
            count += 1

    # --- rooks and queens (orthogonal) -----------------------------------------------------
    orthogonal = bb[WR + us] | bb[WQ + us]
    while orthogonal:
        square = lsb(orthogonal)
        orthogonal &= orthogonal - U64(1)
        targets = rook_attacks(square, occupancy) & ~own
        while targets:
            target = lsb(targets)
            targets &= targets - U64(1)
            flag = CAPTURE if (enemy & bit(target)) else QUIET
            moves[offset + count] = encode(square, target, flag)
            count += 1

    # --- king --------------------------------------------------------------------------------
    king_square = lsb(bb[WK + us])
    targets = KING_ATTACKS[king_square] & ~own
    while targets:
        target = lsb(targets)
        targets &= targets - U64(1)
        flag = CAPTURE if (enemy & bit(target)) else QUIET
        moves[offset + count] = encode(king_square, target, flag)
        count += 1

    # Castling. The king may not start in check, pass through an attacked square, or land on one;
    # the rook is allowed to pass over an attacked square, which is why b1/b8 is only tested for
    # emptiness. Squares are checked before generation so an illegal castle never enters the list.
    rights = state[CASTLE]
    enemy_is_black = 1 - side
    if side == 0:
        if (
            (rights & CASTLE_WK)
            and mailbox[5] == EMPTY
            and mailbox[6] == EMPTY
            and not is_attacked(bb, 4, enemy_is_black)
            and not is_attacked(bb, 5, enemy_is_black)
            and not is_attacked(bb, 6, enemy_is_black)
        ):
            moves[offset + count] = encode(4, 6, CASTLE_KING)
            count += 1
        if (
            (rights & CASTLE_WQ)
            and mailbox[1] == EMPTY
            and mailbox[2] == EMPTY
            and mailbox[3] == EMPTY
            and not is_attacked(bb, 4, enemy_is_black)
            and not is_attacked(bb, 3, enemy_is_black)
            and not is_attacked(bb, 2, enemy_is_black)
        ):
            moves[offset + count] = encode(4, 2, CASTLE_QUEEN)
            count += 1
    else:
        if (
            (rights & CASTLE_BK)
            and mailbox[61] == EMPTY
            and mailbox[62] == EMPTY
            and not is_attacked(bb, 60, enemy_is_black)
            and not is_attacked(bb, 61, enemy_is_black)
            and not is_attacked(bb, 62, enemy_is_black)
        ):
            moves[offset + count] = encode(60, 62, CASTLE_KING)
            count += 1
        if (
            (rights & CASTLE_BQ)
            and mailbox[57] == EMPTY
            and mailbox[58] == EMPTY
            and mailbox[59] == EMPTY
            and not is_attacked(bb, 60, enemy_is_black)
            and not is_attacked(bb, 59, enemy_is_black)
            and not is_attacked(bb, 58, enemy_is_black)
        ):
            moves[offset + count] = encode(60, 58, CASTLE_QUEEN)
            count += 1

    return count


@njit("void(uint64[:])", cache=False)
def refresh_occupancy_jit(bb: np.ndarray) -> None:
    white = U64(0)
    black = U64(0)
    for piece in range(6):
        white |= bb[piece]
        black |= bb[piece + 6]
    bb[WHITE_OCC] = white
    bb[BLACK_OCC] = black
    bb[ALL_OCC] = white | black


@njit("boolean(uint64[:], int64, int64)", cache=False, nogil=True)
def _ep_available(bb: np.ndarray, to_square: Int, side: Int) -> bool:
    """Is there actually an enemy pawn placed to play the en-passant capture?

    Only then does the en-passant square belong in the position, and only then may it enter the
    Zobrist key. `board.fen()` in python-chess omits the square when no capture is on offer, so
    hashing it unconditionally made one position hash two different ways depending on whether it
    was reached by `make_move` or parsed by `set_fen`. Roughly one position in ten after a double
    push was affected, and the cost was silent: transposition entries never matched across a move
    boundary, and the repetition check never matched the game history at all.

    FIDE agrees, for what it is worth -- two positions differ only if an en-passant capture is
    genuinely available, not merely if a pawn happened to move two squares.
    """
    enemy_pawns = bb[WP + 6 * (1 - side)]
    adjacent = U64(0)
    if to_square & 7 > 0:
        adjacent |= U64(1) << U64(to_square - 1)
    if to_square & 7 < 7:
        adjacent |= U64(1) << U64(to_square + 1)
    return bool(enemy_pawns & adjacent)


@njit("void(uint64[:], int8[:], int64[:], uint64[:], int64[:, :], uint64[:], int64, int32)")
def make_move(
    bb: np.ndarray,
    mailbox: np.ndarray,
    state: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    keys: np.ndarray,
    ply: Int,
    move: np.int32,
) -> None:
    """Apply a pseudo-legal move, saving everything unmake needs at `ply`."""
    from_square = move_from(move)
    to_square = move_to(move)
    flag = move_flag(move)
    side = state[STM]
    us = 6 * side
    piece = I64(mailbox[from_square])

    hash_key = key[0]
    keys[ply] = hash_key
    undo[ply, 1] = state[CASTLE]
    undo[ply, 2] = state[EP]
    undo[ply, 3] = state[HALFMOVE]

    # clear the old en-passant file from the hash before anything else touches state
    if state[EP] >= 0:
        hash_key ^= EP_FILE_KEYS[state[EP] & 7]

    captured = I64(EMPTY)
    if flag == EP_CAPTURE:
        captured_square = to_square - 8 if side == 0 else to_square + 8
        captured = I64(mailbox[captured_square])
        bb[captured] ^= bit(captured_square)
        mailbox[captured_square] = EMPTY
        hash_key ^= PIECE_KEYS[captured, captured_square]
    elif flag & CAPTURE_BIT:
        captured = I64(mailbox[to_square])
        bb[captured] ^= bit(to_square)
        hash_key ^= PIECE_KEYS[captured, to_square]
    undo[ply, 0] = captured

    bb[piece] ^= bit(from_square)
    mailbox[from_square] = EMPTY
    hash_key ^= PIECE_KEYS[piece, from_square]

    if flag & PROMO_BIT:
        # flag & 3 orders the promotion pieces knight, bishop, rook, queen -- the same order as
        # the piece indices offset by one, so the arithmetic is a straight add.
        promoted = us + WN + (flag & 3)
        bb[promoted] |= bit(to_square)
        mailbox[to_square] = promoted
        hash_key ^= PIECE_KEYS[promoted, to_square]
    else:
        bb[piece] |= bit(to_square)
        mailbox[to_square] = piece
        hash_key ^= PIECE_KEYS[piece, to_square]

    if flag == CASTLE_KING:
        rook_from = 7 if side == 0 else 63
        rook_to = 5 if side == 0 else 61
        rook = us + WR
        bb[rook] ^= bit(rook_from) | bit(rook_to)
        mailbox[rook_from] = EMPTY
        mailbox[rook_to] = rook
        hash_key ^= PIECE_KEYS[rook, rook_from] ^ PIECE_KEYS[rook, rook_to]
    elif flag == CASTLE_QUEEN:
        rook_from = 0 if side == 0 else 56
        rook_to = 3 if side == 0 else 59
        rook = us + WR
        bb[rook] ^= bit(rook_from) | bit(rook_to)
        mailbox[rook_from] = EMPTY
        mailbox[rook_to] = rook
        hash_key ^= PIECE_KEYS[rook, rook_from] ^ PIECE_KEYS[rook, rook_to]

    hash_key ^= CASTLE_KEYS[state[CASTLE]]
    state[CASTLE] = state[CASTLE] & CASTLE_MASK[from_square] & CASTLE_MASK[to_square]
    hash_key ^= CASTLE_KEYS[state[CASTLE]]

    if flag == DOUBLE_PUSH and _ep_available(bb, to_square, side):
        state[EP] = from_square + (8 if side == 0 else -8)
        hash_key ^= EP_FILE_KEYS[state[EP] & 7]
    else:
        state[EP] = -1

    if piece == WP + us or captured != EMPTY:
        state[HALFMOVE] = 0
    else:
        state[HALFMOVE] += 1
    if side == 1:
        state[FULLMOVE] += 1
    state[STM] = 1 - side
    hash_key ^= SIDE_KEY

    refresh_occupancy_jit(bb)
    key[0] = hash_key


@njit("void(uint64[:], int8[:], int64[:], uint64[:], int64[:, :], uint64[:], int64, int32)")
def unmake_move(
    bb: np.ndarray,
    mailbox: np.ndarray,
    state: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    keys: np.ndarray,
    ply: Int,
    move: np.int32,
) -> None:
    """Exact inverse of make_move. Restores from the slots saved at `ply`."""
    from_square = move_from(move)
    to_square = move_to(move)
    flag = move_flag(move)
    side = 1 - state[STM]  # the side that moved
    us = 6 * side

    state[STM] = side
    if side == 1:
        state[FULLMOVE] -= 1
    captured = undo[ply, 0]
    state[CASTLE] = undo[ply, 1]
    state[EP] = undo[ply, 2]
    state[HALFMOVE] = undo[ply, 3]
    key[0] = keys[ply]

    if flag & PROMO_BIT:
        promoted = us + WN + (flag & 3)
        bb[promoted] ^= bit(to_square)
        bb[WP + us] |= bit(from_square)
        mailbox[from_square] = WP + us
    else:
        piece = I64(mailbox[to_square])
        bb[piece] ^= bit(to_square)
        bb[piece] |= bit(from_square)
        mailbox[from_square] = piece
    mailbox[to_square] = EMPTY

    if flag == EP_CAPTURE:
        captured_square = to_square - 8 if side == 0 else to_square + 8
        bb[captured] |= bit(captured_square)
        mailbox[captured_square] = captured
    elif flag & CAPTURE_BIT:
        bb[captured] |= bit(to_square)
        mailbox[to_square] = captured

    if flag == CASTLE_KING:
        rook_from = 7 if side == 0 else 63
        rook_to = 5 if side == 0 else 61
        bb[us + WR] ^= bit(rook_from) | bit(rook_to)
        mailbox[rook_to] = EMPTY
        mailbox[rook_from] = us + WR
    elif flag == CASTLE_QUEEN:
        rook_from = 0 if side == 0 else 56
        rook_to = 3 if side == 0 else 59
        bb[us + WR] ^= bit(rook_from) | bit(rook_to)
        mailbox[rook_to] = EMPTY
        mailbox[rook_from] = us + WR

    refresh_occupancy_jit(bb)


@njit(
    "int64(uint64[:], int8[:], int64[:], uint64[:], int64[:, :], uint64[:], int32[:], int64,"
    " int64)",
    cache=False,
)
def perft(
    bb: np.ndarray,
    mailbox: np.ndarray,
    state: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    keys: np.ndarray,
    buffer: np.ndarray,
    ply: Int,
    depth: Int,
) -> Int:
    """Count legal leaf nodes. The gate on every claim this file makes about being correct.

    There is no bulk counting at depth 1: it would skip the make/unmake path for the last ply,
    which is exactly where the subtle bugs are, and perft would then pass while the search
    corrupted the board.
    """
    if depth == 0:
        return 1
    offset = ply * MAX_MOVES
    count = generate_moves(bb, mailbox, state, buffer, offset)
    side = state[STM]
    nodes = I64(0)
    for index in range(count):
        move = buffer[offset + index]
        make_move(bb, mailbox, state, key, undo, keys, ply, move)
        king_square = lsb(bb[WK + 6 * side])
        if not is_attacked(bb, king_square, 1 - side):
            nodes += perft(bb, mailbox, state, key, undo, keys, buffer, ply + 1, depth - 1)
        unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
    return nodes

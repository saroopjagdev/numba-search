"""Hand-crafted evaluation.

This exists for two reasons. It is a shippable evaluator in its own right, and it is the
*reference* the NNUE has to beat under SPRT before the net is allowed anywhere near a submission.
Without it there is no way to answer "is the net actually better", only "the net is what we have".

Piece-square tables are generated from stated principles rather than copied from a published
engine. That is a deliberate choice: the rules ban shipping a third-party engine, and while a
table of constants is not an engine, a set of tables we can explain from first principles is
defensible in a finalist walkthrough and a set we cannot is an argument waiting to happen. It also
means the tables can be retuned by changing one coefficient instead of by hand-editing 768 numbers.

Everything is tapered between a midgame and an endgame score by a phase count, so the same table
can say "keep the king in the corner" early and "march the king to the centre" late.
"""

import numpy as np
from numba import njit

from engine.bitboard import KING_ATTACKS, KNIGHT_ATTACKS, bishop_attacks, rook_attacks
from engine.position import (
    ALL_OCC,
    STM,
    WB,
    WHITE_OCC,
    WK,
    WN,
    WP,
    WQ,
    WR,
    Int,
    lsb,
    popcount,
)

I32 = np.int32

# Material, midgame and endgame. Pawns are worth more in the endgame because they promote; minor
# pieces are worth slightly less because there is less to coordinate against.
MATERIAL_MG = (100, 320, 330, 500, 950, 0)
MATERIAL_EG = (120, 310, 340, 550, 950, 0)

# Phase: how far from an endgame we are. 24 is a full board, 0 is bare kings.
PHASE_WEIGHTS = (0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24


def _centrality() -> np.ndarray:
    """0 at the rim, 3 in the four central squares. The backbone of most of the tables below."""
    table = np.zeros(64, dtype=np.int32)
    for square in range(64):
        file, rank = square & 7, square >> 3
        table[square] = min(file, 7 - file, rank, 7 - rank)
    return table


def _build_psqt() -> tuple[np.ndarray, np.ndarray]:
    """Piece-square tables for white, midgame and endgame, from explicit principles.

    Each block below states the idea in one line and then applies it. Nothing here is tuned yet;
    these are starting values, and SPRT decides whether any later change to them is an improvement.
    """
    mg = np.zeros((6, 64), dtype=np.int32)
    eg = np.zeros((6, 64), dtype=np.int32)
    centre = _centrality()

    for square in range(64):
        file, rank = square & 7, square >> 3

        # Pawns: advancing is worth little early and a great deal late, and central pawns that
        # control the centre are worth more than wing pawns that do not.
        mg[0, square] = 2 * (rank - 1) + 3 * min(centre[square], 2)
        eg[0, square] = 12 * (rank - 1) * (rank - 1) // 5
        if rank == 1 and file in (3, 4):
            mg[0, square] -= 12  # an unmoved d/e pawn blocks its own bishop

        # Knights: short-range, so proximity to the centre is almost the whole story, and the rim
        # is genuinely bad rather than merely unhelpful.
        mg[1, square] = 12 * centre[square] - 18
        eg[1, square] = 10 * centre[square] - 15

        # Bishops: reward the long diagonals and mild centralisation; penalise the back rank,
        # where a bishop is usually still undeveloped.
        long_diagonal = file == rank or file + rank == 7
        mg[2, square] = 5 * centre[square] + (10 if long_diagonal else 0) - (8 if rank == 0 else 0)
        eg[2, square] = 5 * centre[square] + (6 if long_diagonal else 0)

        # Rooks: the seventh rank and central files. Open-file detection needs the pawn structure,
        # so it lives in the evaluation proper rather than in a static table.
        mg[3, square] = (25 if rank == 6 else 0) + (5 if file in (3, 4) else 0)
        eg[3, square] = 10 if rank == 6 else 0

        # Queens: barely any positional preference. Early sorties are punished by losing tempo,
        # which the search sees perfectly well on its own, so the table stays nearly flat.
        mg[4, square] = 2 * centre[square] - (6 if rank == 0 else 0)
        eg[4, square] = 4 * centre[square]

        # Kings: the two halves of the game disagree completely. Hide behind the pawns in the
        # midgame, walk to the centre in the endgame. This is the single largest use of tapering.
        shelter = 0
        if rank == 0:
            shelter = 25 if file in (0, 1, 2, 6, 7) else -10
        mg[5, square] = shelter - 12 * rank
        eg[5, square] = 14 * centre[square] - 20

    return mg, eg


_MG, _EG = _build_psqt()

# Tables for both colours. Black's are the white tables mirrored vertically -- rank 1 for white is
# rank 8 for black -- so evaluation never has to branch on colour.
PSQT_MG = np.zeros((12, 64), dtype=np.int32)
PSQT_EG = np.zeros((12, 64), dtype=np.int32)
for _piece in range(6):
    for _square in range(64):
        _mirrored = _square ^ 56
        PSQT_MG[_piece, _square] = _MG[_piece, _square] + MATERIAL_MG[_piece]
        PSQT_EG[_piece, _square] = _EG[_piece, _square] + MATERIAL_EG[_piece]
        PSQT_MG[_piece + 6, _mirrored] = _MG[_piece, _square] + MATERIAL_MG[_piece]
        PSQT_EG[_piece + 6, _mirrored] = _EG[_piece, _square] + MATERIAL_EG[_piece]

PHASE_TABLE = np.array(PHASE_WEIGHTS * 2, dtype=np.int32)

# File and rank masks, used by the pawn-structure and open-file terms.
FILE_MASKS = np.zeros(8, dtype=np.uint64)
for _file in range(8):
    FILE_MASKS[_file] = np.uint64(0x0101010101010101) << np.uint64(_file)

# Squares in front of a pawn on its own file and the two adjacent ones. A pawn with none of the
# enemy's pawns in this region is passed.
PASSED_MASKS = np.zeros((2, 64), dtype=np.uint64)
for _square in range(64):
    _f, _r = _square & 7, _square >> 3
    _span = np.uint64(0)
    for _adjacent in range(max(0, _f - 1), min(7, _f + 1) + 1):
        _span |= FILE_MASKS[_adjacent]
    _ahead_white = np.uint64(0)
    _ahead_black = np.uint64(0)
    for _rr in range(_r + 1, 8):
        _ahead_white |= np.uint64(0xFF) << np.uint64(8 * _rr)
    for _rr in range(0, _r):
        _ahead_black |= np.uint64(0xFF) << np.uint64(8 * _rr)
    PASSED_MASKS[0, _square] = _span & _ahead_white
    PASSED_MASKS[1, _square] = _span & _ahead_black

# Passed pawns by rank, from the mover's point of view. Superlinear: a pawn on the seventh is not
# six times a pawn on the second, it is very nearly a queen.
PASSED_BONUS_MG = np.array([0, 5, 10, 20, 35, 60, 100, 0], dtype=np.int32)
PASSED_BONUS_EG = np.array([0, 10, 20, 40, 70, 120, 190, 0], dtype=np.int32)

ISOLATED_PENALTY = 14
DOUBLED_PENALTY = 12
BISHOP_PAIR_MG = 30
BISHOP_PAIR_EG = 45
ROOK_OPEN_FILE = 22
ROOK_SEMI_OPEN_FILE = 10

# Mobility is scored per attacked square, which is a crude proxy for "this piece has options".
# Knights get more per square than bishops because they have fewer squares to begin with.
KNIGHT_MOBILITY = 4
BISHOP_MOBILITY = 3
ROOK_MOBILITY_MG = 2
ROOK_MOBILITY_EG = 4
QUEEN_MOBILITY = 1

# King safety: count enemy attacks landing in the ring around the king and charge for them on a
# rising scale, so three attackers cost far more than three times one.
KING_ATTACK_WEIGHT = np.array(
    [0, 4, 12, 24, 40, 60, 84, 110, 140, 140, 140, 140, 140, 140, 140, 140], dtype=np.int32
)

MATE_SCORE = 30000
DRAW_SCORE = 0


@njit("boolean(uint64[:], int64, int64, uint64)", cache=False)
def _attack_count(bb: np.ndarray, square: Int, offset: Int, occupancy: np.uint64) -> bool:
    """Does the side at `offset` attack `square`? Sliders only plus knights -- pawns and the king
    contribute little to the king-safety term and cost a table lookup each."""
    if KNIGHT_ATTACKS[square] & bb[WN + offset]:
        return True
    if bishop_attacks(square, occupancy) & (bb[WB + offset] | bb[WQ + offset]):
        return True
    return bool(rook_attacks(square, occupancy) & (bb[WR + offset] | bb[WQ + offset]))


@njit("int32(uint64[:], int8[:], int64[:])", cache=False)
def evaluate(bb: np.ndarray, mailbox: np.ndarray, state: np.ndarray) -> np.int32:
    """Score the position in centipawns, positive for the side to move.

    Returning from the mover's point of view rather than white's is what lets the search negate
    and recurse without ever asking whose turn it is.
    """
    mg = I32(0)
    eg = I32(0)
    phase = I32(0)

    white_pawns = bb[WP]
    black_pawns = bb[WP + 6]
    occupancy = bb[ALL_OCC]

    for piece in range(12):
        pieces = bb[piece]
        sign = I32(1) if piece < 6 else I32(-1)
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - np.uint64(1)
            mg += sign * PSQT_MG[piece, square]
            eg += sign * PSQT_EG[piece, square]
            phase += PHASE_TABLE[piece]

    # --- mobility ------------------------------------------------------------------------------
    for colour in range(2):
        sign = I32(1) if colour == 0 else I32(-1)
        offset = 6 * colour
        own = bb[WHITE_OCC + colour]

        knights = bb[WN + offset]
        while knights:
            square = lsb(knights)
            knights &= knights - np.uint64(1)
            moves = I32(popcount(KNIGHT_ATTACKS[square] & ~own))
            mg += sign * I32(KNIGHT_MOBILITY) * moves
            eg += sign * I32(KNIGHT_MOBILITY) * moves

        bishops = bb[WB + offset]
        while bishops:
            square = lsb(bishops)
            bishops &= bishops - np.uint64(1)
            moves = I32(popcount(bishop_attacks(square, occupancy) & ~own))
            mg += sign * I32(BISHOP_MOBILITY) * moves
            eg += sign * I32(BISHOP_MOBILITY) * moves

        rooks = bb[WR + offset]
        while rooks:
            square = lsb(rooks)
            rooks &= rooks - np.uint64(1)
            moves = I32(popcount(rook_attacks(square, occupancy) & ~own))
            mg += sign * I32(ROOK_MOBILITY_MG) * moves
            eg += sign * I32(ROOK_MOBILITY_EG) * moves
            # Open and semi-open files. A rook behind its own pawn is doing very little.
            file_mask = FILE_MASKS[square & 7]
            own_pawns = white_pawns if colour == 0 else black_pawns
            enemy_pawns = black_pawns if colour == 0 else white_pawns
            if not (file_mask & own_pawns):
                if not (file_mask & enemy_pawns):
                    mg += sign * I32(ROOK_OPEN_FILE)
                    eg += sign * I32(ROOK_OPEN_FILE)
                else:
                    mg += sign * I32(ROOK_SEMI_OPEN_FILE)
                    eg += sign * I32(ROOK_SEMI_OPEN_FILE)

        queens = bb[WQ + offset]
        while queens:
            square = lsb(queens)
            queens &= queens - np.uint64(1)
            moves = I32(
                popcount(
                    (rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)) & ~own
                )
            )
            mg += sign * I32(QUEEN_MOBILITY) * moves
            eg += sign * I32(QUEEN_MOBILITY) * moves

        # Bishop pair: two bishops cover both colour complexes, which is worth more than the sum.
        if popcount(bb[WB + offset]) >= 2:
            mg += sign * I32(BISHOP_PAIR_MG)
            eg += sign * I32(BISHOP_PAIR_EG)

    # --- pawn structure ------------------------------------------------------------------------
    for colour in range(2):
        sign = I32(1) if colour == 0 else I32(-1)
        own_pawns = white_pawns if colour == 0 else black_pawns
        enemy_pawns = black_pawns if colour == 0 else white_pawns
        pawns = own_pawns
        while pawns:
            square = lsb(pawns)
            pawns &= pawns - np.uint64(1)
            file = square & 7
            relative_rank = (square >> 3) if colour == 0 else 7 - (square >> 3)

            if not (PASSED_MASKS[colour, square] & enemy_pawns):
                mg += sign * PASSED_BONUS_MG[relative_rank]
                eg += sign * PASSED_BONUS_EG[relative_rank]

            # Isolated: no friendly pawn on either adjacent file to defend it, ever.
            neighbours = np.uint64(0)
            if file > 0:
                neighbours |= FILE_MASKS[file - 1]
            if file < 7:
                neighbours |= FILE_MASKS[file + 1]
            if not (neighbours & own_pawns):
                mg -= sign * I32(ISOLATED_PENALTY)
                eg -= sign * I32(ISOLATED_PENALTY)

        # Doubled: charge once per extra pawn on a file, not once per pawn.
        for file in range(8):
            extra = I32(popcount(FILE_MASKS[file] & own_pawns))
            if extra > 1:
                mg -= sign * I32(DOUBLED_PENALTY) * (extra - I32(1))
                eg -= sign * I32(DOUBLED_PENALTY) * (extra - I32(1))

    # --- king safety, midgame only -------------------------------------------------------------
    # In the endgame an exposed king is an asset, not a liability, so this term taper to nothing.
    for colour in range(2):
        sign = I32(1) if colour == 0 else I32(-1)
        king_square = lsb(bb[WK + 6 * colour])
        ring = KING_ATTACKS[king_square]
        enemy_offset = 6 - 6 * colour
        attackers = I32(0)
        ring_copy = ring
        while ring_copy:
            square = lsb(ring_copy)
            ring_copy &= ring_copy - np.uint64(1)
            if _attack_count(bb, square, enemy_offset, occupancy):
                attackers += I32(1)
        if attackers > I32(15):
            attackers = I32(15)
        mg -= sign * KING_ATTACK_WEIGHT[attackers]

    if phase > I32(TOTAL_PHASE):
        phase = I32(TOTAL_PHASE)
    score = (mg * phase + eg * (I32(TOTAL_PHASE) - phase)) // I32(TOTAL_PHASE)
    return score if state[STM] == 0 else -score


@njit("int32(uint64[:])", cache=False)
def material_balance(bb: np.ndarray) -> np.int32:
    """Pure material, white minus black, on the referee's scale.

    `harness/referee.py` adjudicates at ply 300 on exactly this -- P=1 N=3 B=3 R=5 Q=9, no
    positional term at all. Search uses it near the cap to bank material rather than shuffle.
    """
    values = (1, 3, 3, 5, 9, 0)
    total = I32(0)
    for piece in range(6):
        total += I32(values[piece]) * I32(popcount(bb[piece]))
        total -= I32(values[piece]) * I32(popcount(bb[piece + 6]))
    return total


# a1 is dark, so the low byte is a1/c1/e1/g1 and the pattern alternates up the board.
DARK_SQUARES = np.uint64(0xAA55AA55AA55AA55)


@njit("boolean(uint64[:])", cache=False)
def no_mating_material(bb: np.ndarray) -> bool:
    """True when neither side can force mate, whatever the piece values say.

    Without this the engine happily trades its last pawn for the opponent's last piece: a lone
    bishop still scores as a bishop, so K+B vs K read as +372 and three won ladder games were
    liquidated into dead draws. `tools/audit_truth.py` asserts each case below.

    Only configurations that are drawn are listed. Two knights against a bare king cannot be
    *forced*, and a knight each cannot either; a helpmate exists in both, but treating them as
    playable is the mistake that costs games, not the other way round.
    """
    # Any pawn, rook or queen and mate is on the table. Pawns matter most -- they promote.
    for piece in (WP, WR, WQ):
        if bb[piece] | bb[piece + 6]:
            return False

    white_knights = popcount(bb[WN])
    black_knights = popcount(bb[WN + 6])
    white_bishops = popcount(bb[WB])
    black_bishops = popcount(bb[WB + 6])

    # K vs K, and a single minor against a bare king.
    if white_knights + black_knights + white_bishops + black_bishops <= 1:
        return True

    white_minors = white_knights + white_bishops
    black_minors = black_knights + black_bishops

    # One bishop each: drawn when they share a colour complex, since neither can ever attack the
    # squares the other defends. On opposite colours it stays a normal position.
    if white_bishops == 1 and black_bishops == 1 and white_minors == 1 and black_minors == 1:
        white_dark = (bb[WB] & DARK_SQUARES) != np.uint64(0)
        black_dark = (bb[WB + 6] & DARK_SQUARES) != np.uint64(0)
        return bool(white_dark == black_dark)

    # A knight each: no forced mate for either side.
    if white_minors == 1 and black_minors == 1 and white_knights == 1 and black_knights == 1:
        return True

    # Two knights against a bare king cannot be forced.
    if white_knights == 2 and white_minors == 2 and black_minors == 0:
        return True
    return bool(black_knights == 2 and black_minors == 2 and white_minors == 0)

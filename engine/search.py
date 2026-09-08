"""Alpha-beta search: iterative deepening, transposition table, quiescence, and the usual pruning.

Clock handling lives in Python, not here
----------------------------------------
Numba's nopython mode has no `time` module, and the usual workarounds (`objmode` around a clock
read) cost more than the check saves when it runs at every node. So the division of labour is:
**one jitted call per depth**, with Python owning the clock between iterations.

Inside an iteration the only budget is a node limit. When it is exceeded the search sets an abort
flag and unwinds immediately, and the caller throws that iteration away and keeps the last
completed depth. That is exactly the right semantics anyway — a partial depth is not trustworthy
unless it improved on the previous one, and iterative deepening already guarantees a usable move
from the depth before.

Python converts remaining milliseconds into a node budget using measured nps, deliberately
over-provisioned: the node limit is a safety valve against a single pathological iteration, while
the real time control is the decision, taken between iterations, of whether to start another one.

Scores are centipawns from the side to move. Mate scores are `MATE - ply`, so a shorter mate beats
a longer one and the search prefers to deliver mate rather than merely keep it available.

Two evaluations, chosen at run time
-----------------------------------
`use_nnue` selects the network over the hand-crafted evaluation. It is a parameter rather than a
compile-time constant on purpose: the whole question of whether the net is worth its cost is an
SPRT question, and a runtime flag lets one binary play both sides of that match. Both functions
return int32 centipawns relative to the side to move, so the call sites differ only in the branch.

When the net is on, the accumulator travels with the search on a per-ply stack. Every `make_move`
that survives the legality check is followed by a `push`, and `unmake_move` needs no counterpart
because the parent ply's accumulator was never written. The pairing is load-bearing and unchecked
at run time: a `make_move` without its `push` leaves every evaluation below it scoring a position
the search is not in.
"""

import time

import numpy as np
from numba import njit

from engine.bitboard import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    bishop_attacks,
    rook_attacks,
)
from engine.eval import evaluate, no_mating_material
from engine.nnue import Network, evaluate_at, new_stack, push, push_null, refresh
from engine.position import (
    ALL_OCC,
    CAPTURE_BIT,
    EMPTY,
    EP,
    EP_CAPTURE,
    EP_FILE_KEYS,
    HALFMOVE,
    MAX_MOVES,
    MAX_PLY,
    PROMO_BIT,
    SIDE_KEY,
    STM,
    WB,
    WHITE_OCC,
    WK,
    WN,
    WP,
    WQ,
    WR,
    Int,
    generate_moves,
    is_attacked,
    lsb,
    make_move,
    move_flag,
    move_from,
    move_to,
    new_position,
    new_undo,
    popcount,
    set_fen,
    unmake_move,
)

I32 = np.int32
U64 = np.uint64

MATE = 30000
# Anything at or above this is a forced mate, not an evaluation. The margin is wide enough that a
# mate score can never be confused with a material advantage, however lopsided.
MATE_THRESHOLD = MATE - 1000
INFINITY = 32000

# Transposition table bound types.
EXACT, LOWER, UPPER = 0, 1, 2

# Move ordering scores. The gaps are large so the categories can never interleave: a losing
# capture must always be tried after every killer, regardless of how the history table has grown.
SCORE_TT = 1 << 24
SCORE_GOOD_CAPTURE = 1 << 22
SCORE_KILLER_1 = 1 << 21
SCORE_KILLER_2 = (1 << 21) - 1
SCORE_BAD_CAPTURE = -(1 << 22)

# MVV-LVA: capturing a queen with a pawn is the best possible ordering guess, and the table is
# indexed [victim, attacker] so bigger victims and smaller attackers both push the score up.
PIECE_VALUES = np.array([100, 320, 330, 500, 950, 10000], dtype=np.int32)

# Hard ceiling on search ply, well inside MAX_PLY. Two things need it: the check extension can
# lengthen a line indefinitely when checks keep coming, and every buffer here is indexed by ply
# with no bounds checking inside nopython code -- so an unbounded line does not raise, it
# segfaults. The headroom below MAX_PLY covers the quiescence tail that hangs off the deepest node.
MAX_SEARCH_PLY = 120

TT_BITS = 22  # 4M entries, ~40 MB across the five arrays


TTArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def new_tt(bits: int = TT_BITS) -> TTArrays:
    """Allocate a transposition table. Kept out of the jitted code so the size stays tunable."""
    size = 1 << bits
    return (
        np.zeros(size, dtype=np.uint64),  # key
        np.zeros(size, dtype=np.int32),  # move
        np.zeros(size, dtype=np.int32),  # score
        np.zeros(size, dtype=np.int16),  # depth
        np.zeros(size, dtype=np.int8),  # bound
    )


def new_search_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Killers, history, the per-ply key path, and the abort/counter cell."""
    killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
    history = np.zeros((12, 64), dtype=np.int32)
    path = np.zeros(MAX_PLY + 1, dtype=np.uint64)
    # nodes, node limit, aborted, root best move. The root move travels out through `control`
    # rather than being read back from the transposition table: a collision could replace the root
    # entry mid-search, and a move from the wrong position is an instant loss.
    control = np.zeros(4, dtype=np.int64)
    return killers, history, path, control


@njit("int32(int8[:], int32)", cache=False)
def _mvv_lva(mailbox: np.ndarray, move: np.int32) -> np.int32:
    """Most valuable victim, least valuable attacker."""
    attacker = I32(mailbox[move_from(move)]) % I32(6)
    target = mailbox[move_to(move)]
    victim = I32(0) if target == EMPTY else I32(target) % I32(6)
    # As in engine/bitboard.py, the I32() is free inside numba and exists so njit's erased
    # signature does not leak Any into the ordering code.
    return I32(PIECE_VALUES[victim] * I32(16) - PIECE_VALUES[attacker] // I32(16))


@njit("uint64(uint64[:], int64, uint64)", cache=False)
def _attackers_to(bb: np.ndarray, square: Int, occ: U64) -> U64:
    """Every piece of either colour attacking `square`, under the given occupancy.

    Occupancy is a parameter rather than `bb[ALL_OCC]` because the swap-off below removes pieces as
    they are captured, and each removal can uncover an x-ray attacker behind it. Recomputing the
    slider attacks against the shrinking occupancy is what makes those appear.

    The pawn table is indexed by the *defending* colour, as in `is_attacked`: a white pawn attacks
    `square` exactly when a black pawn standing there would attack the white pawn.
    """
    attackers = PAWN_ATTACKS[1, square] & bb[WP]
    attackers |= PAWN_ATTACKS[0, square] & bb[WP + 6]
    attackers |= KNIGHT_ATTACKS[square] & (bb[WN] | bb[WN + 6])
    attackers |= KING_ATTACKS[square] & (bb[WK] | bb[WK + 6])
    diagonal = bb[WB] | bb[WQ] | bb[WB + 6] | bb[WQ + 6]
    attackers |= bishop_attacks(square, occ) & diagonal
    straight = bb[WR] | bb[WQ] | bb[WR + 6] | bb[WQ + 6]
    attackers |= rook_attacks(square, occ) & straight
    return U64(attackers)


@njit("boolean(uint64[:], int8[:], int64[:], int32, int32)", cache=False)
def see_ge(
    bb: np.ndarray, mailbox: np.ndarray, state: np.ndarray, move: np.int32, threshold: np.int32
) -> bool:
    """Is the static exchange on this capture worth at least `threshold`?

    MVV-LVA orders captures by what they win; it cannot tell that the win is illusory. QxP looks
    excellent right up to the moment the pawn is defended, and quiescence then searches the whole
    losing chain to find out. SEE settles it without making a single move, by playing out the
    exchange on the target square with the least valuable attacker each time.

    Answering "at least `threshold`?" rather than "how much?" is what keeps this cheap: it needs no
    gain array, so nothing is allocated per call -- which matters, because this runs on every
    capture in the quiescence tree.

    `swap` carries the running balance, negated at each step so it is always from the point of view
    of the side about to capture, and `result` flips with it. A side that would come out behind
    simply declines to continue the exchange, which is what the early `break` represents.

    Two things this deliberately does not model: pins, so a pinned defender is counted as a real
    one, and promotions, whose value change mid-exchange the caller avoids by not asking. Both make
    it conservative rather than wrong.
    """
    frm = move_from(move)
    to = move_to(move)

    victim = mailbox[to]
    captured = PIECE_VALUES[I32(victim) % I32(6)] if victim != EMPTY else I32(0)
    swap = captured - threshold
    if swap < 0:
        # Even winning the piece for free falls short of the threshold.
        return False

    swap = PIECE_VALUES[I32(mailbox[frm]) % I32(6)] - swap
    if swap <= 0:
        # Losing our own attacker outright still clears the threshold, so nothing can go wrong.
        return True

    # The target square is emptied as well as the origin: the victim is gone, and leaving it in
    # would block the x-rays that recapture through it.
    occ = (bb[ALL_OCC] ^ (U64(1) << U64(frm))) & ~(U64(1) << U64(to))
    stm = I32(state[STM])
    attackers = _attackers_to(bb, to, occ)
    result = 1

    while True:
        stm = I32(1) - stm
        attackers &= occ
        own = attackers & bb[WHITE_OCC + stm]
        if own == 0:
            break
        result ^= 1
        offset = 6 * stm
        diagonal = bb[WB] | bb[WQ] | bb[WB + 6] | bb[WQ + 6]
        straight = bb[WR] | bb[WQ] | bb[WR + 6] | bb[WQ + 6]

        piece = own & bb[WP + offset]
        if piece:
            swap = PIECE_VALUES[0] - swap
            if swap < result:
                break
            occ ^= U64(1) << U64(lsb(piece))
            attackers |= bishop_attacks(to, occ) & diagonal
            continue
        piece = own & bb[WN + offset]
        if piece:
            swap = PIECE_VALUES[1] - swap
            if swap < result:
                break
            occ ^= U64(1) << U64(lsb(piece))
            continue
        piece = own & bb[WB + offset]
        if piece:
            swap = PIECE_VALUES[2] - swap
            if swap < result:
                break
            occ ^= U64(1) << U64(lsb(piece))
            attackers |= bishop_attacks(to, occ) & diagonal
            continue
        piece = own & bb[WR + offset]
        if piece:
            swap = PIECE_VALUES[3] - swap
            if swap < result:
                break
            occ ^= U64(1) << U64(lsb(piece))
            attackers |= rook_attacks(to, occ) & straight
            continue
        piece = own & bb[WQ + offset]
        if piece:
            swap = PIECE_VALUES[4] - swap
            if swap < result:
                break
            occ ^= U64(1) << U64(lsb(piece))
            attackers |= (bishop_attacks(to, occ) & diagonal) | (rook_attacks(to, occ) & straight)
            continue
        # Only the king is left. Capturing into a square the opponent still attacks is illegal, so
        # the exchange ends here and the side to move forfeits the last swap rather than making it.
        if attackers & bb[WHITE_OCC + (I32(1) - stm)]:
            result ^= 1
        break

    return result != 0


@njit(
    "void(uint64[:], int8[:], int64[:], int32[:], int32[:], int64, int64, int32, int32[:, :],"
    " int32[:, :], int64)",
    cache=False,
)
def _score_moves(
    bb: np.ndarray,
    mailbox: np.ndarray,
    state: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    offset: Int,
    count: Int,
    tt_move: np.int32,
    killers: np.ndarray,
    history: np.ndarray,
    ply: Int,
) -> None:
    """Assign an ordering score to each move. Sorting happens lazily, one pick at a time.

    Captures split into winning and losing by the swap-off rather than all sitting above the
    killers. MVV-LVA sorts captures by what they *win* and is silent about what they cost, so
    before this the first move tried at a node was routinely a queen grabbing a defended pawn --
    a whole subtree spent proving what `see_ge` answers in a few dozen instructions.

    The swap-off is not asked about every capture. When the victim is worth at least as much as
    the attacker the capture cannot lose material outright, which covers most of them; only the
    ambiguous ones pay for a SEE call. En passant and promotions skip it for the same reason
    quiescence does -- the swap-off assumes a stationary victim and a fixed attacker value.
    """
    for index in range(count):
        move = moves[offset + index]
        if move == tt_move:
            scores[offset + index] = I32(SCORE_TT)
        elif move_flag(move) & CAPTURE_BIT:
            base = I32(SCORE_GOOD_CAPTURE) + _mvv_lva(mailbox, move)
            attacker = I32(mailbox[move_from(move)]) % I32(6)
            target = mailbox[move_to(move)]
            ambiguous = (
                move_flag(move) != EP_CAPTURE
                and not (move_flag(move) & PROMO_BIT)
                and target != EMPTY
                and PIECE_VALUES[I32(target) % I32(6)] < PIECE_VALUES[attacker]
            )
            if ambiguous and not see_ge(bb, mailbox, state, move, I32(0)):
                # Demoted below the killers, not discarded. A losing capture is still sometimes
                # the move -- it can be the only way out of a fork, or a sacrifice the search
                # needs to see -- so it is tried last rather than pruned here.
                scores[offset + index] = I32(SCORE_BAD_CAPTURE) + _mvv_lva(mailbox, move)
            else:
                scores[offset + index] = base
        elif move_flag(move) & PROMO_BIT:
            scores[offset + index] = I32(SCORE_GOOD_CAPTURE) + PIECE_VALUES[move_flag(move) & 3]
        elif move == killers[ply, 0]:
            scores[offset + index] = I32(SCORE_KILLER_1)
        elif move == killers[ply, 1]:
            scores[offset + index] = I32(SCORE_KILLER_2)
        else:
            scores[offset + index] = history[mailbox[move_from(move)], move_to(move)]


@njit("int64(int32[:], int32[:], int64, int64, int64)", cache=False)
def _pick_best(moves: np.ndarray, scores: np.ndarray, offset: Int, count: Int, start: Int) -> Int:
    """Selection sort, one step at a time.

    Sorting the whole list up front wastes work: most nodes cut after two or three moves, so the
    remaining scores are never looked at. This swaps the best remaining move into position and
    returns, which costs O(n) per move actually tried rather than O(n log n) per node.
    """
    best = start
    for index in range(start + 1, count):
        if scores[offset + index] > scores[offset + best]:
            best = index
    if best != start:
        moves[offset + start], moves[offset + best] = moves[offset + best], moves[offset + start]
        scores[offset + start], scores[offset + best] = (
            scores[offset + best],
            scores[offset + start],
        )
    return start


@njit("int64(uint64[:], int64[:], uint64, int64, uint64[:], int64)", cache=False, nogil=True)
def _occurrences(
    path: np.ndarray,
    state: np.ndarray,
    candidate: np.uint64,
    ply: Int,
    game_keys: np.ndarray,
    game_count: Int,
) -> Int:
    """How many times `candidate` has already appeared, in the search path and then the game.

    Distinct from `_is_repetition` in that the exact number matters: the referee's claim needs a
    position to have occurred *twice* already, where a draw score needs only once. Counts the same
    positions in the same order; only the stopping condition differs.
    """
    total = 0
    limit = state[HALFMOVE]
    back = 2
    while back <= limit:
        if back <= ply:
            probe = path[ply - back]
        else:
            index = game_count - (back - ply)
            if index < 0:
                return total
            probe = game_keys[index]
        if probe == candidate:
            total += 1
        back += 2
    return total


@njit("boolean(uint64[:], int64[:], uint64[:], int64, uint64[:], int64)", cache=False)
def _is_repetition(
    path: np.ndarray,
    state: np.ndarray,
    key: np.ndarray,
    ply: Int,
    game_keys: np.ndarray,
    game_count: Int,
) -> bool:
    """A single repetition counts as a draw, scanning the search path and then the game itself.

    Waiting for the third occurrence would let the search walk into a repetition it cannot avoid
    and only notice one ply too late. Requiring two is the standard trade: it occasionally scores
    a position as drawn that could still be won, and it never misses a real repetition.

    The scan continues past the root into `game_keys`, every position of the game so far, one per
    ply. That matters because `harness/play.py` calls `board.outcome(claim_draw=True)`, and
    python-chess claims on `can_claim_threefold_repetition()` -- which fires on the *second*
    occurrence the moment a legal move exists that would produce a third. The draw therefore
    arrives a full move earlier than counting to three would suggest, and the referee claims it
    for us whether we wanted it or not.

    `game_keys` used to hold only the positions we were asked to move in, one every two plies, and
    the walk gave up at odd plies because the opponent's positions had never been observed. That
    left us blind to repetitions of opponent-to-move positions, which is precisely the class an
    opponent uses to draw against us: rounds 27 and 30 were both drawn that way, from +202 and
    +650, by an opponent move we never saw coming. `Searcher.record_move` now supplies the missing
    plies, so there is no parity case left.
    """
    current = key[0]
    # Only positions since the last irreversible move can repeat, so the halfmove clock bounds it.
    limit = state[HALFMOVE]
    back = 2
    while back <= limit:
        if back <= ply:
            candidate = path[ply - back]
        else:
            index = game_count - (back - ply)
            if index < 0:
                return False
            candidate = game_keys[index]
        if candidate == current:
            return True
        back += 2
    return False


@njit(
    "int32(uint64[:], int8[:], int64[:], uint64[:], int64[:, :], uint64[:], int32[:], int32[:],"
    " int16[:, :, :], int16[:, :], int16[:, :], int32[:],"
    " int64, int64, int32, int32, boolean, int64[:])",
    cache=False,
)
def quiescence(
    bb: np.ndarray,
    mailbox: np.ndarray,
    state: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    keys: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    acc: np.ndarray,
    transformer: np.ndarray,
    output: np.ndarray,
    output_bias: np.ndarray,
    ply: Int,
    depth: Int,
    alpha: np.int32,
    beta: np.int32,
    use_nnue: bool,
    control: np.ndarray,
) -> np.int32:
    """Search only captures and promotions until the position is quiet.

    Without this the evaluation is called in the middle of an exchange and reports whatever the
    material happens to be halfway through it, which is the single largest source of nonsense in a
    naive alpha-beta. `depth` counts *down* from zero and bounds how deep the capture chain may go.

    In check is the one case where none of that applies. Standing pat means "I decline to move
    further and the static score stands" -- not a legal option when in check, since the side to
    move has no pass. A quiet king step or a block can be the only way to survive, and skipping
    quiet moves here would never look at either, so a check found at the horizon would be scored on
    whatever captures happen to be lying around rather than on whether the king actually escapes.
    When checked, every legal reply is generated and searched, not just captures, `depth` does not
    count down (an evasion is forced, not a choice to keep hunting for exchanges), and a position
    with no legal reply is checkmate rather than a quiet score of zero.
    """
    control[0] += 1
    if control[0] > control[1]:
        control[2] = 1
        return I32(0)

    # Checked here as well as in the main search because this is where the liquidation actually
    # happens: the capture that leaves K+B vs K is a capture, so quiescence is what scores it.
    if no_mating_material(bb):
        return I32(0)

    side = state[STM]
    king_square = lsb(bb[WK + 6 * side])
    checked = is_attacked(bb, king_square, 1 - side)

    if use_nnue:
        stand_pat = evaluate_at(acc, output, output_bias, mailbox, ply, state[STM])
    else:
        stand_pat = evaluate(bb, mailbox, state)
    if ply >= MAX_SEARCH_PLY:
        return stand_pat

    if not checked:
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        if depth <= -8:
            return stand_pat

    offset = ply * MAX_MOVES
    count = generate_moves(bb, mailbox, state, moves, offset)

    # Score in place, then pick lazily. When not in check, quiet moves are skipped entirely below;
    # when in check every legal reply is a candidate, captures still tried first.
    for index in range(count):
        move = moves[offset + index]
        flag = move_flag(move)
        if (flag & CAPTURE_BIT) or (flag & PROMO_BIT):
            scores[offset + index] = _mvv_lva(mailbox, move)
        elif checked:
            scores[offset + index] = I32(0)
        else:
            scores[offset + index] = I32(-(1 << 30))

    # A mate score, not stand_pat: if nothing below finds a legal reply, this is checkmate. Only
    # reachable when checked, since generate_moves always has at least one move otherwise (the
    # position was reached by a legal move from the other side).
    best = I32(-MATE + ply) if checked else stand_pat
    found_legal = False
    for index in range(count):
        _pick_best(moves, scores, offset, count, index)
        if not checked and scores[offset + index] == I32(-(1 << 30)):
            break  # everything remaining is quiet, and we are not obliged to look at it
        move = moves[offset + index]

        # Delta and SEE pruning only apply to the "hunt for a good exchange" case. When in check
        # every reply is forced, not optional, so pruning one on the grounds that it looks bad would
        # be discarding the only way to survive.
        if not checked and move_flag(move) != EP_CAPTURE and not (move_flag(move) & PROMO_BIT):
            victim = mailbox[move_to(move)]
            if victim != EMPTY:
                gain = PIECE_VALUES[I32(victim) % I32(6)]
                if stand_pat + gain + I32(200) < alpha and popcount(bb[ALL_OCC]) > 6:
                    continue
                # Static exchange pruning. MVV-LVA has already sorted these by what they *win*,
                # which is exactly the ordering that puts QxP-defended-by-a-pawn near the front.
                # Playing it out to discover the loss costs a subtree; asking the swap-off costs a
                # few dozen instructions and no nodes at all.
                #
                # Only plain captures, matching the guard above: the swap-off assumes the victim
                # stands on the target square and that the attacker's value does not change
                # mid-exchange, and en passant breaks the first while promotions break the second.
                if not see_ge(bb, mailbox, state, move, I32(0)):
                    continue

        make_move(bb, mailbox, state, key, undo, keys, ply, move)
        moved_king_square = lsb(bb[WK + 6 * side])
        if is_attacked(bb, moved_king_square, 1 - side):
            unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
            continue
        found_legal = True
        # After the legality check, not before: an illegal move is about to be taken back, and
        # pushing the accumulator for it would be the most expensive part of discovering that.
        if use_nnue:
            push(acc, transformer, mailbox, ply, move, undo[ply, 0], side)
        # A check evasion does not spend the capture-chain budget: it was not a choice to keep
        # hunting for exchanges, so it should not count against the depth that bounds that hunt.
        next_depth = depth if checked else depth - 1
        # fmt: off
        score = -quiescence(
            bb, mailbox, state, key, undo, keys, moves, scores,
            acc, transformer, output, output_bias,
            ply + 1, next_depth, -beta, -alpha, use_nnue, control,
        )
        # fmt: on
        unmake_move(bb, mailbox, state, key, undo, keys, ply, move)

        if control[2]:
            return I32(0)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    if checked and not found_legal:
        return I32(-MATE + ply)
    return best


@njit(
    "int32(uint64[:], int8[:], int64[:], uint64[:], int64[:, :], uint64[:], int32[:], int32[:],"
    " int16[:, :, :], int16[:, :], int16[:, :], int32[:],"
    " uint64[:], int32[:], int32[:], int16[:], int8[:], int32[:, :], int32[:, :], uint64[:],"
    " uint64[:], int64, uint64, int64, int64, int32, int32, boolean, boolean, int64[:])",
    cache=False,
    # Releases the GIL for the whole call, which is what makes pondering possible: the main thread
    # has to be able to run Python -- to accept the next `get_move` and stop us -- while a ponder
    # thread is inside this function. Without it the ponder thread would hold the interpreter for
    # the entire search and the agent would simply stop responding.
    #
    # Only the outermost dispatcher needs this. `quiescence` and the movegen helpers are called
    # from compiled code, never from Python, so they never take the GIL in the first place.
    #
    # Nothing here is made thread-safe by it. Two searches must not share mutable state, and the
    # only thing they do share is the transposition table -- see `Searcher.__init__`.
    nogil=True,
)
def negamax(
    bb: np.ndarray,
    mailbox: np.ndarray,
    state: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    keys: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    acc: np.ndarray,
    transformer: np.ndarray,
    output: np.ndarray,
    output_bias: np.ndarray,
    tt_key: np.ndarray,
    tt_move: np.ndarray,
    tt_score: np.ndarray,
    tt_depth: np.ndarray,
    tt_bound: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    path: np.ndarray,
    game_keys: np.ndarray,
    game_count: Int,
    claim_mask: np.uint64,
    ply: Int,
    depth: Int,
    alpha: np.int32,
    beta: np.int32,
    allow_null: bool,
    use_nnue: bool,
    control: np.ndarray,
) -> np.int32:
    control[0] += 1
    if control[0] > control[1]:
        control[2] = 1
        return I32(0)

    if ply >= MAX_SEARCH_PLY:
        if use_nnue:
            return evaluate_at(acc, output, output_bias, mailbox, ply, state[STM])
        return evaluate(bb, mailbox, state)

    root = ply == 0
    path[ply] = key[0]

    if not root:
        # Draw by repetition or the fifty-move rule. Checked before anything else so a repetition
        # is never masked by a transposition table hit from a different path.
        if _is_repetition(path, state, key, ply, game_keys, game_count) or state[HALFMOVE] >= 100:
            return I32(0)
        # Neither side can force mate, so the piece count is irrelevant and the game is drawn.
        if no_mating_material(bb):
            return I32(0)
        # Mate-distance pruning: if we already have a mate at this ply, a longer one cannot help.
        alpha = max(alpha, I32(-MATE + ply))
        beta = min(beta, I32(MATE - ply - 1))
        if alpha >= beta:
            return alpha

    side = state[STM]
    king_square = lsb(bb[WK + 6 * side])
    checked = is_attacked(bb, king_square, 1 - side)
    if checked:
        depth += 1  # check extension: forced lines are cheap and easy to get wrong shallow

    if depth <= 0:
        # fmt: off
        return quiescence(
            bb, mailbox, state, key, undo, keys, moves, scores,
            acc, transformer, output, output_bias,
            ply, 0, alpha, beta, use_nnue, control,
        )
        # fmt: on

    # The mask comes from the array, not from a module constant. A constant would be baked in at
    # compile time, and a table allocated at any other size would then be indexed past its end --
    # which inside nopython code is not an IndexError, it is a segfault.
    index_tt = np.int64(key[0] & np.uint64(tt_key.shape[0] - 1))
    stored_move = I32(0)
    if tt_key[index_tt] == key[0]:
        stored_move = tt_move[index_tt]
        if not root and tt_depth[index_tt] >= depth:
            stored = I32(tt_score[index_tt])
            # Mate scores are stored relative to the node they were found at, so they have to be
            # re-based on the way out. Storing them absolute is the classic silent TT bug.
            if stored > MATE_THRESHOLD:
                stored -= I32(ply)
            elif stored < -MATE_THRESHOLD:
                stored += I32(ply)
            stored_bound = tt_bound[index_tt]
            if stored_bound == EXACT:
                return stored
            if stored_bound == LOWER and stored >= beta:
                return stored
            if stored_bound == UPPER and stored <= alpha:
                return stored

    if use_nnue:
        static = evaluate_at(acc, output, output_bias, mailbox, ply, side)
    else:
        static = evaluate(bb, mailbox, state)

    # Reverse futility: far enough above beta that giving away a piece would still cut. Only when
    # not in check and not in a mate-scored window, where the margin means nothing.
    if (
        not checked
        and depth <= 6
        and beta < MATE_THRESHOLD
        and static - I32(85) * I32(depth) >= beta
    ):
        return static

    # Null move: give the opponent a free move; if we are still winning, this node is not worth
    # searching properly. Disabled in check, at low depth, and with only pawns left, where
    # zugzwang makes "pass" strictly better than any legal move and the assumption fails.
    non_pawn = popcount(bb[WHITE_OCC + side] & ~bb[WP + 6 * side])
    if allow_null and not checked and depth >= 3 and static >= beta and non_pawn > 1:
        reduction = 2 + depth // 6
        saved_ep = state[EP]
        saved_key = key[0]
        # A null move flips the side to move and clears en passant, and nothing else. Both have to
        # be reflected in the hash or the TT entries stored below this node key on the wrong
        # position -- the classic way null move quietly poisons a table.
        state[STM] = 1 - side
        state[EP] = -1
        key[0] = key[0] ^ SIDE_KEY
        if saved_ep >= 0:
            key[0] = key[0] ^ EP_FILE_KEYS[saved_ep & 7]
        if use_nnue:
            push_null(acc, ply)
        # fmt: off
        # Numba wants every array as its own argument, so the recursive calls run to 27 of them.
        # The formatter would put each on its own line, turning five call sites into a hundred
        # lines in which the only thing that varies -- ply, depth, window -- is invisible.
        score = -negamax(
            bb, mailbox, state, key, undo, keys, moves, scores,
            acc, transformer, output, output_bias, tt_key, tt_move, tt_score,
            tt_depth, tt_bound, killers, history, path,
            game_keys, game_count, claim_mask, ply + 1, depth - 1 - reduction,
            -beta, -beta + I32(1), False, use_nnue, control,
        )
        # fmt: on
        state[STM] = side
        state[EP] = saved_ep
        key[0] = saved_key
        if control[2]:
            return I32(0)
        if score >= beta:
            # Do not return a mate score from a null-move search: the mate is an artefact of the
            # opponent having been made to pass.
            return beta if score > MATE_THRESHOLD else score

    offset = ply * MAX_MOVES
    count = generate_moves(bb, mailbox, state, moves, offset)
    _score_moves(
        bb, mailbox, state, moves, scores, offset, count, stored_move, killers, history, ply
    )

    best_score = I32(-INFINITY)
    best_move = I32(0)
    legal = 0
    original_alpha = alpha

    for index in range(count):
        _pick_best(moves, scores, offset, count, index)
        move = moves[offset + index]
        flag = move_flag(move)
        quiet = not ((flag & CAPTURE_BIT) or (flag & PROMO_BIT))

        # Futility: a quiet move at shallow depth that cannot lift a hopeless static score into
        # the window. Never applied to the first move, so a node always searches something.
        if (
            quiet
            and legal > 0
            and not checked
            and depth <= 3
            and best_score > -MATE_THRESHOLD
            and static + I32(120) * I32(depth) <= alpha
        ):
            continue

        make_move(bb, mailbox, state, key, undo, keys, ply, move)
        king_square = lsb(bb[WK + 6 * side])
        if is_attacked(bb, king_square, 1 - side):
            unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
            continue

        # The referee does not wait for a third occurrence. `referee.py:59` calls
        # `board.outcome(claim_draw=True)`, and python-chess claims as soon as the side to move
        # *has a legal move* reaching a position that has already occurred twice -- whether or not
        # they would ever play it. Merely standing here ends the game, and the fact that we are
        # winning and would obviously choose something else does not matter. Round 27 was drawn
        # from +202 exactly this way, on a board holding no threefold at all.
        #
        # So if any move reaches a twice-seen position, the whole node is a draw, however good the
        # alternatives look. `claim_mask` is a 64-bit membership sketch of the game history keyed
        # on the low six bits; it is wrong only in the direction of a false positive, and a miss
        # costs one shift and one test, which is what keeps this out of the hot path. The exact
        # count runs on the roughly one node in sixty-four that gets through.
        if (
            ply > 0
            and (claim_mask >> (key[0] & np.uint64(63))) & np.uint64(1)
            and _occurrences(path, state, key[0], ply + 1, game_keys, game_count) >= 2
        ):
            unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
            return I32(0)

        legal += 1
        if use_nnue:
            push(acc, transformer, mailbox, ply, move, undo[ply, 0], side)

        # Late move reductions. Moves ordered this far down are rarely best, so search them
        # shallower first and only re-search at full depth if one surprises us.
        reduction = 0
        if quiet and depth >= 3 and legal > 3 and not checked:
            reduction = 1
            if legal > 8:
                reduction += 1
            if depth >= 6:
                reduction += 1

        # fmt: off
        if legal == 1:
            score = -negamax(
                bb, mailbox, state, key, undo, keys, moves, scores,
                acc, transformer, output, output_bias, tt_key, tt_move, tt_score,
                tt_depth, tt_bound, killers, history, path,
                game_keys, game_count, claim_mask, ply + 1, depth - 1,
                -beta, -alpha, True, use_nnue, control,
            )
        else:
            # Zero-window probe, reduced. Two things can send us back for a full search: the probe
            # beating alpha despite the reduction, or it landing inside a real window at the root.
            score = -negamax(
                bb, mailbox, state, key, undo, keys, moves, scores,
                acc, transformer, output, output_bias, tt_key, tt_move, tt_score,
                tt_depth, tt_bound, killers, history, path,
                game_keys, game_count, claim_mask, ply + 1, depth - 1 - reduction,
                -alpha - I32(1), -alpha, True, use_nnue, control,
            )
            if score > alpha and reduction > 0:
                score = -negamax(
                    bb, mailbox, state, key, undo, keys, moves, scores,
                    acc, transformer, output, output_bias, tt_key, tt_move, tt_score,
                    tt_depth, tt_bound, killers, history, path,
                    game_keys, game_count, claim_mask, ply + 1, depth - 1,
                    -alpha - I32(1), -alpha, True, use_nnue, control,
                )
            if score > alpha and score < beta:
                score = -negamax(
                    bb, mailbox, state, key, undo, keys, moves, scores,
                    acc, transformer, output, output_bias, tt_key, tt_move, tt_score,
                    tt_depth, tt_bound, killers, history, path,
                    game_keys, game_count, claim_mask, ply + 1, depth - 1,
                    -beta, -alpha, True, use_nnue, control,
                )
        # fmt: on

        unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
        if control[2]:
            return I32(0)

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if quiet:
                # Killers and history both record "this quiet move caused a cutoff", at different
                # scopes: killers are per-ply and forgotten quickly, history is global and grows.
                if killers[ply, 0] != move:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = move
                bonus = I32(depth) * I32(depth)
                piece = mailbox[move_from(move)]
                history[piece, move_to(move)] += bonus
                if history[piece, move_to(move)] > I32(1 << 20):
                    # Halve everything rather than clamp one entry, so the *relative* ordering
                    # the table has learned survives the rescale.
                    for p in range(12):
                        for s in range(64):
                            history[p, s] //= I32(2)
            break

    if legal == 0:
        # No legal move: mate if in check, stalemate if not. Distance-to-mate keeps shorter mates
        # scoring higher, which is what makes the engine actually finish games.
        return I32(-MATE + ply) if checked else I32(0)

    if root:
        control[3] = best_move

    bound = EXACT
    if best_score <= original_alpha:
        bound = UPPER
    elif best_score >= beta:
        bound = LOWER

    stored = best_score
    if stored > MATE_THRESHOLD:
        stored += I32(ply)
    elif stored < -MATE_THRESHOLD:
        stored -= I32(ply)
    # Depth-preferred replacement, with the current position always allowed to overwrite itself.
    if tt_depth[index_tt] <= depth or tt_key[index_tt] != key[0]:
        tt_key[index_tt] = key[0]
        tt_move[index_tt] = best_move
        tt_score[index_tt] = stored
        tt_depth[index_tt] = np.int16(depth)
        tt_bound[index_tt] = np.int8(bound)

    return best_score


class Searcher:
    """Owns the search's persistent state and drives iterative deepening.

    Everything expensive is allocated once, in `__init__`: the transposition table survives across
    moves (the whole point of it -- a bullet game is one long search that happens to be interrupted
    by the opponent), and the move buffers are sized for the deepest legal ply so no allocation
    ever happens mid-search.
    """

    def __init__(self, tt_bits: int = TT_BITS, share: "Searcher | None" = None) -> None:
        """Allocate a searcher. `share` makes this a second searcher onto the same table.

        Pondering needs two searchers, because a search owns mutable scratch -- the board, the move
        buffers, the accumulator stack, the abort cell -- and two searches running over one set of
        those would corrupt each other immediately. What they must share is the transposition
        table, since filling it on the opponent's clock is the entire point of pondering.

        The table is shared without locking. That is safe here only because of how a stored move is
        used: `_score_moves` compares it against moves the generator has already produced, so an
        entry torn by a concurrent write matches nothing and costs a little move ordering. It can
        never introduce a move that was not legally generated. A torn *score* is possible and can
        cost a cutoff; that is the same trade lazy-SMP engines make, and it is bounded by the fact
        that the root move never comes from the table (see `new_search_state`).

        The network is shared too, and unconditionally safe: it is read-only after loading.
        """
        self.bb, self.mailbox, self.state, self.key = new_position()
        self.undo, self.keys = new_undo()
        self.moves = np.zeros(MAX_MOVES * MAX_PLY, dtype=np.int32)
        self.scores = np.zeros(MAX_MOVES * MAX_PLY, dtype=np.int32)
        # A second searcher must not allocate a second 40 MB table, and would defeat its own
        # purpose by filling one nobody reads.
        # Annotated because `share` is a `Searcher`, so reading `share.tt` here is mypy resolving
        # the very attribute this line defines. The same applies to `network` below.
        self.tt: TTArrays = share.tt if share is not None else new_tt(tt_bits)
        self.killers, self.history, self.path, self.control = new_search_state()
        # The presence of `weights/nnue.npz` *is* the decision to use the net. Nothing else gates
        # it, because the gate happens earlier: the file only enters the repository once SPRT has
        # said the net beats the hand-crafted evaluation, and until then `Network` hands back a
        # zero net that would score every position as a draw. Defaulting to "on if available" and
        # keeping the file out is safer than defaulting to "off" and forgetting to turn it on.
        self.network: Network = share.network if share is not None else Network()
        self.use_nnue = self.network.available
        # One accumulator per ply, plus one: the deepest node still pushes a child before the ply
        # cap turns it back.
        self.acc = new_stack(self.network.hidden, MAX_PLY + 1)
        # Every position we have already been asked to move in, oldest first. Sized past the ply
        # cap so it cannot overflow in a legal game.
        self.game_keys = np.zeros(512, dtype=np.uint64)
        self.game_count = 0
        # Membership sketch of every position this game has visited, keyed on the low six bits
        # of the Zobrist key. The search consults it before paying for an exact repetition count.
        self.claim_mask = np.uint64(0)
        # Measured nps, used only to convert remaining time into a node budget. Starts pessimistic
        # so the first move of a game cannot overshoot; one iteration replaces it with the truth.
        self.nps = 200_000.0
        # Set by `stop()` from another thread to end a ponder search early. See `stop()` for why
        # this is a plain Python attribute and not a cell the compiled code polls.
        self._stop = False

    def stop(self) -> None:
        """Ask an in-flight search to abort. Safe to call from another thread.

        The compiled search already aborts when its node count passes `control[1]`, so driving that
        cell negative ends the current depth within a node or two -- no extra check in the hot path,
        which matters because the hot path runs a few hundred thousand times a second and pondering
        does not.

        The ordering is what makes this race-free, and it only works in this order. `_stop` is set
        *first*; `_iterate` re-asserts the abort after it writes its own budget, so an interleaving
        that loses this write to `control[1]` still leaves `_stop` visible to the check that
        follows it. Both threads hold the GIL for every Python-level access here -- the ponder
        thread gives it up only inside `negamax` -- so there is no visibility question to answer.
        """
        self._stop = True
        self.control[1] = -1

    def resume(self) -> None:
        """Clear a previous `stop()` so this searcher can be used again."""
        self._stop = False

    def set_position(self, fen: str) -> None:
        set_fen(self.bb, self.mailbox, self.state, self.key, fen)

    def record_position(self) -> None:
        """Append the current position to the game history the repetition check reads.

        Called before searching, and again through `record_move` for the position our reply leads
        to, so the history holds every ply rather than every other one. A halfmove clock of zero
        means the last move was irreversible, so nothing before it can repeat and the history is
        dropped -- which also keeps the array from filling up in a long game.
        """
        if self.state[HALFMOVE] == 0:
            self.game_count = 0
        if self.game_count < self.game_keys.shape[0]:
            self.game_keys[self.game_count] = self.key[0]
            self.game_count += 1
        self._refresh_claim_mask()

    def _refresh_claim_mask(self) -> None:
        """Rebuild the membership sketch the search tests before counting occurrences.

        Rebuilt rather than updated because the history is dropped on every irreversible move, and
        a mask can have bits set but never cleared. It is at most a hundred cheap operations once
        per ply.
        """
        mask = np.uint64(0)
        for i in range(self.game_count):
            mask |= np.uint64(1) << (self.game_keys[i] & np.uint64(63))
        self.claim_mask = mask

    def record_move(self, move: np.int32) -> None:
        """Append the position our own move leads to -- one with the opponent to move.

        These plies are never handed to us: we see the position before our move and the position
        after the opponent's reply, so without this the history skips every odd ply and the
        repetition check cannot see a repetition the opponent is steering towards. Ply slot 0 of
        the undo stack is free here because this runs between searches, never inside one.
        """
        make_move(self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, 0, move)
        self.record_position()
        unmake_move(self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, 0, move)

    def forget_history(self) -> None:
        """Drop the game history after a move we could not record.

        A gap in `game_keys` is worse than no history at all: the walk counts back one entry per
        ply, so a missing ply shifts every older entry and the check compares against positions
        that never occurred. Forgetting only risks missing a repetition; keeping a gap risks
        inventing one and throwing away a won game.
        """
        self.game_count = 0
        self.claim_mask = np.uint64(0)

    def new_game(self) -> None:
        """Clear everything that could carry over between games."""
        for array in self.tt:
            array.fill(0)
        self.killers.fill(0)
        self.history.fill(0)
        self.game_count = 0

    def _iterate(self, depth: int, alpha: int, beta: int, budget: int) -> tuple[int, int, int]:
        """One depth. Returns (score, move, nodes); `self.control[2]` says whether it aborted."""
        self.control[0] = 0
        self.control[1] = budget
        self.control[2] = 0
        self.control[3] = 0
        # Re-assert an abort that `stop()` may have raised while we were setting up. Without this
        # the line above would quietly restore a full budget to a search that has been cancelled,
        # and the depth would run to completion on the opponent's move.
        if self._stop:
            self.control[1] = -1
        # The last entry in game_keys is the root itself, and the search reaches the root through
        # `path`, so it is handed only the positions strictly before it.
        history_count = max(0, self.game_count - 1)
        if self.use_nnue:
            # Rebuilt every iteration rather than once per move. It costs ~20 us against an
            # iteration measured in milliseconds, and it makes the root accumulator unconditionally
            # correct -- including after an aborted iteration, which is the case where reasoning
            # about whether the stack unwound cleanly would be doing real work for no gain.
            refresh(
                self.mailbox,
                self.network.transformer,
                self.network.transformer_bias,
                self.acc[0],
            )
        # fmt: off
        score = negamax(
            self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, self.moves,
            self.scores, self.acc, self.network.transformer, self.network.output,
            self.network.output_bias, *self.tt, self.killers, self.history, self.path,
            self.game_keys, history_count, self.claim_mask,
            0, depth, I32(alpha), I32(beta), True,
            self.use_nnue, self.control,
        )
        # fmt: on
        return int(score), int(self.control[3]), int(self.control[0])

    def search(
        self, budget_ms: float, max_depth: int = 64, node_limit: int = 1 << 62
    ) -> tuple[int, int, int]:
        """Iteratively deepen until the budget runs out. Returns (move, score, depth).

        The clock is read between jitted calls, never inside one. A depth is either finished and
        trusted or abandoned and discarded -- there is no partial result, because a partial depth
        has searched the moves in the order the *previous* depth suggested and so is biased toward
        them.

        The decision to start depth N+1 uses a fraction of the remaining time: iterations grow by
        roughly 2-4x, so if less than a third of the budget is left the next one will not finish
        and the time is better banked.

        Note that an aspiration re-search is a *separate* jitted call, and so gets its own clock
        read and its own node budget. Treating a depth as one indivisible unit is what let this
        overrun a 1000 ms budget by 69% on WAC.008: three widening re-searches at the same depth,
        with the clock consulted before the first of them and not again.
        """
        started = time.perf_counter()
        deadline = started + budget_ms / 1000.0

        # History fades between moves rather than resetting: the ordering knowledge is still mostly
        # valid one ply later, but stale entries should not outweigh what this search learns.
        self.history //= 8
        self.killers.fill(0)

        best_move, best_score, completed = 0, 0, 0
        window = 30
        depth = 1
        last_nodes = 0
        while depth <= max_depth:
            if self._stop:
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if completed >= 1 and remaining < (deadline - started) * 0.35:
                break

            if completed >= 4:
                alpha, beta = best_score - window, best_score + window
            else:
                alpha, beta = -INFINITY, INFINITY

            # Multiplier on the branching-factor cap, reset at every depth. See the retry below.
            scale = 1
            while True:
                # Recomputed for every call, including each aspiration re-search. The margin is
                # 1.2x rather than 3x: the node limit is what actually bounds the overrun, since
                # nothing else can interrupt a call once it has started.
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    self.control[2] = 1
                    break
                # Two independent caps, because the nps estimate can be wrong in the dangerous
                # direction: a shallow TT-saturated call reports millions of nodes per second, and
                # the next real iteration would then be handed a budget it takes several seconds
                # to spend. The branching-factor cap does not depend on nps at all -- a depth
                # costs roughly 3x the one before it, so 6x the last completed call is generous
                # and still bounds the overrun when the timing estimate is nonsense.
                budget = min(
                    node_limit,
                    int(remaining * self.nps * 1.2) + 5_000,
                    scale * (6 * last_nodes + 50_000),
                )

                call_started = time.perf_counter()
                score, move, nodes = self._iterate(depth, alpha, beta, budget)
                call_elapsed = time.perf_counter() - call_started
                last_nodes = max(last_nodes, nodes)
                if call_elapsed > 0.01 and nodes > 20_000:
                    # Per call, not cumulative. Dividing one iteration's nodes by the whole
                    # search's elapsed time understates nps by the branching factor and makes
                    # every subsequent budget too small.
                    # Clamped: one anomalous sample must not be able to hand the next call a
                    # budget large enough to blow the clock outright.
                    self.nps = min(5e6, max(5e4, 0.7 * self.nps + 0.3 * (nodes / call_elapsed)))
                if self.control[2]:
                    # Aborted -- but on nodes or on the clock? The branching-factor cap is sized
                    # from the last completed call, and after the opponent plays the move we
                    # predicted, the transposition table answers the early depths in a few hundred
                    # nodes. The cap then sits near its 50k floor while the position itself needs
                    # millions, so the first depth that has to do real work trips it. Treating that
                    # as "time is up" ended the whole search and played a depth-12 move with three
                    # and a half seconds still on the clock -- eight times in one rated game.
                    #
                    # So when the call died well inside the time left, it was the cap and not the
                    # clock: widen it and search the same depth again. The clock read at the top of
                    # this loop is what actually terminates us, and the quarter-of-remaining test
                    # keeps a retry from being the call that overruns.
                    if call_elapsed < 0.25 * remaining and scale < 4096:
                        scale *= 4
                        continue
                    break
                if score <= alpha:
                    # Fail low: the position is worse than we thought. Widen downward and retry;
                    # keeping the narrow window would return a bound, not a move worth playing.
                    alpha = max(-INFINITY, alpha - window * 4)
                    window *= 4
                    continue
                if score >= beta:
                    beta = min(INFINITY, beta + window * 4)
                    window *= 4
                    continue
                break

            if self.control[2]:
                break
            if move != 0:
                best_move, best_score, completed = move, score, depth
            window = 30
            depth += 1

        if best_move == 0:
            # Nothing completed -- a budget so small that even depth 1 was cut off. Return the
            # first legal move rather than nothing; a legal move always beats a forfeit.
            best_move = self._first_legal()
        return best_move, best_score, completed

    def _first_legal(self) -> int:
        count = generate_moves(self.bb, self.mailbox, self.state, self.moves, 0)
        side = int(self.state[STM])
        for index in range(count):
            move = self.moves[index]
            make_move(self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, 0, move)
            king_square = lsb(self.bb[WK + 6 * side])
            legal = not is_attacked(self.bb, king_square, 1 - side)
            unmake_move(self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, 0, move)
            if legal:
                return int(move)
        return 0

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
"""

import time

import numpy as np
from numba import njit

from engine.eval import evaluate
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
    WHITE_OCC,
    WK,
    WP,
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


@njit(
    "void(int8[:], int32[:], int32[:], int64, int64, int32, int32[:, :], int32[:, :], int64)",
    cache=False,
)
def _score_moves(
    mailbox: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    offset: Int,
    count: Int,
    tt_move: np.int32,
    killers: np.ndarray,
    history: np.ndarray,
    ply: Int,
) -> None:
    """Assign an ordering score to each move. Sorting happens lazily, one pick at a time."""
    for index in range(count):
        move = moves[offset + index]
        if move == tt_move:
            scores[offset + index] = I32(SCORE_TT)
        elif move_flag(move) & CAPTURE_BIT:
            scores[offset + index] = I32(SCORE_GOOD_CAPTURE) + _mvv_lva(mailbox, move)
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

    The scan continues past the root into `game_keys`, the positions the agent has already been
    asked to move in. That matters because `harness/play.py` calls `board.outcome(claim_draw=True)`
    -- the referee claims threefold *for* us, so an engine that only remembers its own search can
    shuffle a won game into a draw and never see it coming. Those positions all have us to move,
    i.e. they sit at even offsets from the root, which is why the walk only reaches for them when
    `ply` is even; at odd plies the repeating positions are the opponent's and were never observed.
    """
    current = key[0]
    # Only positions since the last irreversible move can repeat, so the halfmove clock bounds it.
    limit = state[HALFMOVE]
    back = 2
    while back <= limit:
        if back <= ply:
            candidate = path[ply - back]
        else:
            if ply % 2 == 1:
                return False
            index = game_count - (back - ply) // 2
            if index < 0:
                return False
            candidate = game_keys[index]
        if candidate == current:
            return True
        back += 2
    return False


@njit(
    "int32(uint64[:], int8[:], int64[:], uint64[:], int64[:, :], uint64[:], int32[:], int32[:],"
    " int64, int64, int32, int32, int64[:])",
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
    ply: Int,
    depth: Int,
    alpha: np.int32,
    beta: np.int32,
    control: np.ndarray,
) -> np.int32:
    """Search only captures and promotions until the position is quiet.

    Without this the evaluation is called in the middle of an exchange and reports whatever the
    material happens to be halfway through it, which is the single largest source of nonsense in a
    naive alpha-beta. `depth` counts *down* from zero and bounds how deep the capture chain may go.
    """
    control[0] += 1
    if control[0] > control[1]:
        control[2] = 1
        return I32(0)

    stand_pat = evaluate(bb, mailbox, state)
    if ply >= MAX_SEARCH_PLY:
        return stand_pat
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat
    if depth <= -8:
        return stand_pat

    offset = ply * MAX_MOVES
    count = generate_moves(bb, mailbox, state, moves, offset)
    side = state[STM]

    # Score in place, then pick lazily. Quiet moves are skipped entirely below.
    for index in range(count):
        move = moves[offset + index]
        flag = move_flag(move)
        if (flag & CAPTURE_BIT) or (flag & PROMO_BIT):
            scores[offset + index] = _mvv_lva(mailbox, move)
        else:
            scores[offset + index] = I32(-(1 << 30))

    best = stand_pat
    for index in range(count):
        _pick_best(moves, scores, offset, count, index)
        if scores[offset + index] == I32(-(1 << 30)):
            break  # everything remaining is quiet
        move = moves[offset + index]

        # Delta pruning: if winning this piece outright still leaves us far below alpha, the whole
        # capture chain is hopeless. Skipped when the position is nearly bare, where zugzwang-ish
        # material swings make the margin unreliable.
        if move_flag(move) != EP_CAPTURE and not (move_flag(move) & PROMO_BIT):
            victim = mailbox[move_to(move)]
            if victim != EMPTY:
                gain = PIECE_VALUES[I32(victim) % I32(6)]
                if stand_pat + gain + I32(200) < alpha and popcount(bb[ALL_OCC]) > 6:
                    continue

        make_move(bb, mailbox, state, key, undo, keys, ply, move)
        king_square = lsb(bb[WK + 6 * side])
        if is_attacked(bb, king_square, 1 - side):
            unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
            continue
        score = -quiescence(
            bb,
            mailbox,
            state,
            key,
            undo,
            keys,
            moves,
            scores,
            ply + 1,
            depth - 1,
            -beta,
            -alpha,
            control,
        )
        unmake_move(bb, mailbox, state, key, undo, keys, ply, move)

        if control[2]:
            return I32(0)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


@njit(
    "int32(uint64[:], int8[:], int64[:], uint64[:], int64[:, :], uint64[:], int32[:], int32[:],"
    " uint64[:], int32[:], int32[:], int16[:], int8[:], int32[:, :], int32[:, :], uint64[:],"
    " uint64[:], int64, int64, int64, int32, int32, boolean, int64[:])",
    cache=False,
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
    ply: Int,
    depth: Int,
    alpha: np.int32,
    beta: np.int32,
    allow_null: bool,
    control: np.ndarray,
) -> np.int32:
    control[0] += 1
    if control[0] > control[1]:
        control[2] = 1
        return I32(0)

    if ply >= MAX_SEARCH_PLY:
        return evaluate(bb, mailbox, state)

    root = ply == 0
    path[ply] = key[0]

    if not root:
        # Draw by repetition or the fifty-move rule. Checked before anything else so a repetition
        # is never masked by a transposition table hit from a different path.
        if _is_repetition(path, state, key, ply, game_keys, game_count) or state[HALFMOVE] >= 100:
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
        return quiescence(
            bb, mailbox, state, key, undo, keys, moves, scores, ply, 0, alpha, beta, control
        )

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
        # fmt: off
        # Numba wants every array as its own argument, so the recursive calls run to 22 of them.
        # The formatter would put each on its own line, turning five call sites into a hundred
        # lines in which the only thing that varies -- ply, depth, window -- is invisible.
        score = -negamax(
            bb, mailbox, state, key, undo, keys, moves, scores, tt_key, tt_move, tt_score,
            tt_depth, tt_bound, killers, history, path,
            game_keys, game_count, ply + 1, depth - 1 - reduction,
            -beta, -beta + I32(1), False, control,
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
    _score_moves(mailbox, moves, scores, offset, count, stored_move, killers, history, ply)

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
        legal += 1

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
                bb, mailbox, state, key, undo, keys, moves, scores, tt_key, tt_move, tt_score,
                tt_depth, tt_bound, killers, history, path,
                game_keys, game_count, ply + 1, depth - 1,
                -beta, -alpha, True, control,
            )
        else:
            # Zero-window probe, reduced. Two things can send us back for a full search: the probe
            # beating alpha despite the reduction, or it landing inside a real window at the root.
            score = -negamax(
                bb, mailbox, state, key, undo, keys, moves, scores, tt_key, tt_move, tt_score,
                tt_depth, tt_bound, killers, history, path,
                game_keys, game_count, ply + 1, depth - 1 - reduction,
                -alpha - I32(1), -alpha, True, control,
            )
            if score > alpha and reduction > 0:
                score = -negamax(
                    bb, mailbox, state, key, undo, keys, moves, scores, tt_key, tt_move, tt_score,
                    tt_depth, tt_bound, killers, history, path,
                    game_keys, game_count, ply + 1, depth - 1,
                    -alpha - I32(1), -alpha, True, control,
                )
            if score > alpha and score < beta:
                score = -negamax(
                    bb, mailbox, state, key, undo, keys, moves, scores, tt_key, tt_move, tt_score,
                    tt_depth, tt_bound, killers, history, path,
                    game_keys, game_count, ply + 1, depth - 1,
                    -beta, -alpha, True, control,
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

    def __init__(self, tt_bits: int = TT_BITS) -> None:
        self.bb, self.mailbox, self.state, self.key = new_position()
        self.undo, self.keys = new_undo()
        self.moves = np.zeros(MAX_MOVES * MAX_PLY, dtype=np.int32)
        self.scores = np.zeros(MAX_MOVES * MAX_PLY, dtype=np.int32)
        self.tt = new_tt(tt_bits)
        self.killers, self.history, self.path, self.control = new_search_state()
        # Every position we have already been asked to move in, oldest first. Sized past the ply
        # cap so it cannot overflow in a legal game.
        self.game_keys = np.zeros(512, dtype=np.uint64)
        self.game_count = 0
        # Measured nps, used only to convert remaining time into a node budget. Starts pessimistic
        # so the first move of a game cannot overshoot; one iteration replaces it with the truth.
        self.nps = 200_000.0

    def set_position(self, fen: str) -> None:
        set_fen(self.bb, self.mailbox, self.state, self.key, fen)

    def record_position(self) -> None:
        """Append the current position to the game history the repetition check reads.

        Called once per move, before searching. A FEN with a halfmove clock of zero means the last
        move was irreversible, so nothing before it can repeat and the history is dropped -- which
        also keeps the array from filling up in a long game.
        """
        if self.state[HALFMOVE] == 0:
            self.game_count = 0
        if self.game_count < self.game_keys.shape[0]:
            self.game_keys[self.game_count] = self.key[0]
            self.game_count += 1

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
        # The last entry in game_keys is the root itself, and the search reaches the root through
        # `path`, so it is handed only the positions strictly before it.
        history_count = max(0, self.game_count - 1)
        # fmt: off
        score = negamax(
            self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, self.moves,
            self.scores, *self.tt, self.killers, self.history, self.path,
            self.game_keys, history_count, 0, depth, I32(alpha), I32(beta), True, self.control,
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
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if completed >= 1 and remaining < (deadline - started) * 0.35:
                break

            if completed >= 4:
                alpha, beta = best_score - window, best_score + window
            else:
                alpha, beta = -INFINITY, INFINITY

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
                    6 * last_nodes + 50_000,
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

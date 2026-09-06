"""The submission entrypoint. The platform imports this file and calls get_move.

Import runs once per game inside a 90 second budget and pays for every numba compilation, since
`/tmp` is wiped between games and a compiled cache could not be shipped anyway. So the warm-up
below is not optional politeness -- it is the point of module scope. Compiling lazily inside
`get_move` would spend the first move's clock on it and flag.

Everything here is a wrapper. The engine is in `engine/`; this file's job is the clock, the
python-chess legality guard, and making sure no exception can ever reach the protocol.

**There is deliberately no pondering here, and it must not be added back on the strength of a
local measurement.** `AGENTS.md:33` says the process keeps its core while the opponent thinks; that
line is a stale copy of a rule that has since changed, and the file itself warns that the two URLs
are canonical. Both canonical documents now say the opposite -- "your process is suspended on
opponent's turn, background threads won't run" and "work you leave running between own moves not
run". Nothing of ours executes between our moves.

The trap is that our harness does not suspend anything, so a ponder thread runs perfectly well
locally and SPRT scored it at +63.2 +- 23.8 Elo. That Elo cannot occur in a rated game. Any future
run will report the same phantom gain, so this is the one decision in the project that measurement
must not be allowed to overturn -- the instrument is wrong here, not the reasoning.
"""

import time

import chess
import numpy as np

from engine.position import move_to_uci
from engine.search import Searcher

# Move overhead. `harness/play.py` starts the clock *before* the request is written to the pipe and
# stops it after the reply is read, so JSON encoding and two pipe round-trips come out of our
# budget, and the 500 ms watchdog grace does not rescue us -- the referee flags on `clock < 0`
# independently of it. This is the deduction taken off every budget before we even start thinking.
MOVE_OVERHEAD_MS = 60.0

# Fraction of what is theoretically affordable that we actually spend. The asymmetry used to be
# stated as brutal -- overspend once and lose outright, underspend and lose a few centipawns -- and
# that framing was wrong in both halves.
#
# Underspending is not cheap. Halving our thinking time costs 52.5 +- 20.1 Elo (run 34033781905),
# right on the classical 50-80 per doubling. And we were not spending 0.85 of the allowance, we were
# spending 0.85 x 0.88 = 0.75, because the search consumes only 88% of the budget handed to it.
# Recovering that quarter is 0.42 doublings, about 22 Elo.
#
# Overspending is also not the cliff it sounds like, because the budget is capped at a quarter of
# the clock. Worst case the search takes 126% of its budget, so worst spend is 0.25 x SAFETY x 1.26
# of the clock, and that equals the 500 ms increment at a 1.50 s clock: below there the clock rises
# again. The engine parks at a low clock, it does not flag. Raising SAFETY moves that parking point
# from 1.93 s to 1.50 s and nothing else.
#
# 1.10 rather than 1.00 because 1.10 x 0.88 = 0.97 -- the point is to actually spend the allowance,
# not to spend 88% of it.
SAFETY = 1.10

# Assumed moves remaining. The referee adjudicates at ply 300, so a game is at most 150 moves each,
# but spreading the base clock over 150 would leave the engine playing far too fast in the opening
# where the position is still decidable. Most games end well before the cap.
ASSUMED_MOVES_LEFT = 30

_searcher = Searcher()
_last_fullmove = 10**9


def _warm_up() -> None:
    """Force every jitted function to compile now, on a position that exercises all of them.

    Kiwipete has castling available for both sides, en passant reachable, promotions one push
    away and enough material for the null-move and late-move-reduction paths to be taken. A
    shallow search over it therefore compiles the whole engine; a search from the start position
    would leave several branches uncompiled and pay for them on move one.
    """
    _searcher.set_position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    _searcher.record_position()
    _searcher.search(600.0, max_depth=6)
    _searcher.new_game()


_warm_up()


def _budget_ms(time_left_ms: int) -> float:
    """How long this move may take, in milliseconds."""
    usable = max(0.0, float(time_left_ms) - MOVE_OVERHEAD_MS)
    # The increment is banked every move, so it is spendable in full over and above the share of
    # the base clock -- but only most of it, or the clock ratchets down over a long game.
    allowance = usable / ASSUMED_MOVES_LEFT + 0.75 * 500.0
    # Never stake more than a quarter of what is left on one move, however good the position looks.
    return max(10.0, min(allowance, usable * 0.25) * SAFETY)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    global _last_fullmove
    started = time.perf_counter()
    board = chess.Board(fen)

    try:
        # A fullmove number that has gone backwards means a new game in the same process. The
        # transposition table and history would otherwise be answering with the last game's
        # positions, which is worse than starting cold.
        fullmove = board.fullmove_number
        if fullmove < _last_fullmove:
            _searcher.new_game()
        _last_fullmove = fullmove

        _searcher.set_position(fen)
        _searcher.record_position()
        move, _score, _depth = _searcher.search(_budget_ms(time_left_ms))
        uci = move_to_uci(np.int32(move))

        # The legality guard. The engine's own generator is perft-verified, but an illegal move is
        # an instant loss and python-chess is an independent implementation, so the cost of asking
        # it is worth paying on every single move.
        if chess.Move.from_uci(uci) in board.legal_moves:
            return uci
        print(f"engine proposed an illegal move {uci} in {fen}; falling back")
    # Deliberately bare: anything escaping this function forfeits the game, so there is no class
    # of exception worth letting through in exchange for a cleaner traceback.
    except Exception as error:
        elapsed = (time.perf_counter() - started) * 1000.0
        print(f"search failed after {elapsed:.0f}ms: {type(error).__name__}: {error}")

    # Last resort. Any legal move beats a crash or a forfeit, and this path costs microseconds.
    return next(iter(board.legal_moves)).uci()

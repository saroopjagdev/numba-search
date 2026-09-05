"""The submission entrypoint. The platform imports this file and calls get_move.

Import runs once per game inside a 90 second budget and pays for every numba compilation, since
`/tmp` is wiped between games and a compiled cache could not be shipped anyway. So the warm-up
below is not optional politeness -- it is the point of module scope. Compiling lazily inside
`get_move` would spend the first move's clock on it and flag.

Everything here is a wrapper. The engine is in `engine/`; this file's job is the clock, the
python-chess legality guard, and making sure no exception can ever reach the protocol.
"""

import time

import chess
import numpy as np

from engine.ponder import Ponderer
from engine.position import move_to_uci
from engine.search import Searcher

# Move overhead. `harness/play.py` starts the clock *before* the request is written to the pipe and
# stops it after the reply is read, so JSON encoding and two pipe round-trips come out of our
# budget, and the 500 ms watchdog grace does not rescue us -- the referee flags on `clock < 0`
# independently of it. This is the deduction taken off every budget before we even start thinking.
MOVE_OVERHEAD_MS = 60.0

# Fraction of what is theoretically affordable that we actually spend. The asymmetry is brutal:
# overspending once loses the game outright, underspending costs a few centipawns of depth.
SAFETY = 0.85

# Time we refuse to allocate, and the divisor applied to everything above it.
#
# The previous scheme divided the whole remaining clock by a fixed 30. That spends a constant
# *fraction* of what is left each move, which decays geometrically and so can never spend the clock
# down: across the first seven rated games it finished with between 32 and 88 seconds unused, never
# once thought for longer than 4.1 seconds, and in the one loss was mated holding 74.9 of 134.5
# seconds. Making the divisor count down with the move number was tried first and is not worth it --
# long games force its floor up, the floor then governs everything, and it buys about 10%.
#
# Protecting the reserve explicitly is the better shape, because safety stops being an emergent
# property of the divisor and becomes a structural one: the budget goes to zero as the clock
# approaches RESERVE_MS whatever the divisor is, so the divisor can then be far smaller. Simulated
# against the seven real games at the measured 75% budget consumption this is 1.30x the thinking
# time, and with every move overrunning by the worst margin the logs actually show, the clock still
# holds 17.3 seconds in the longest game and 13.4 seconds at the referee's ply-300 bound.
RESERVE_MS = 15_000.0
CLOCK_DIVISOR = 12

_searcher = Searcher()
_ponderer = Ponderer(_searcher)
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
    # Only what sits above the reserve is available to divide up. As the clock falls towards the
    # reserve this term vanishes and the increment below is all that is left, so the engine plays
    # quickly rather than flagging -- which is the right way round.
    spendable = max(0.0, usable - RESERVE_MS)
    # The increment is banked every move, so it is spendable in full over and above the share of
    # the base clock -- but only most of it, or the clock ratchets down over a long game.
    allowance = spendable / CLOCK_DIVISOR + 0.75 * 500.0
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
    # Before anything else, and outside the guard below, because everything after this point either
    # uses the engine or is the fallback that still has to be fast. We have one core: a ponder
    # search left running would halve the speed of the search that has to produce this move, which
    # costs far more than pondering ever wins. `stop()` cannot raise, for the same reason.
    _ponderer.stop()
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
            # Think on their clock. Started before returning rather than after, because there is no
            # "after" -- the platform calls us again and nothing of ours runs in between. The cost
            # charged to our clock is one thread start, tens of microseconds.
            board.push(chess.Move.from_uci(uci))
            _ponderer.start(board.fen())
            return uci
        print(f"engine proposed an illegal move {uci} in {fen}; falling back")
    # Deliberately bare: anything escaping this function forfeits the game, so there is no class
    # of exception worth letting through in exchange for a cleaner traceback.
    except Exception as error:
        elapsed = (time.perf_counter() - started) * 1000.0
        print(f"search failed after {elapsed:.0f}ms: {type(error).__name__}: {error}")

    # Last resort. Any legal move beats a crash or a forfeit, and this path costs microseconds.
    return next(iter(board.legal_moves)).uci()

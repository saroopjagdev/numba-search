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

# The move we plan the clock out to, and the shortest horizon we will ever plan against.
#
# Dividing the remaining clock by a constant spends a fixed *fraction* of it each move, which is
# geometric and therefore front-loaded: 3.7 s at the start against 0.8 s by move 80. Planning to a
# move number instead makes the profile flat, because the divisor shrinks as the clock does.
#
# Flat is what the evidence asks for. Two attempts to spend more per move were rejected at -65.9 and
# -55.2 Elo, and lining them up showed both were the same experiment -- they opened at 7.8 s a move
# and bought twenty good moves and sixty bad ones. Rated rounds 31 and 32 were both lost in the
# phase where the front-loaded profile has already decayed to well under a second. Depth in a quiet
# opening is close to worthless; depth on move 80, with a passed pawn running, is the game.
#
# EXPECTED_FINAL_FULLMOVE is measured, not guessed: rated games start from a curated opening around
# fullmove 8 and have been ending between fullmove 90 and 110. MIN_MOVES_TO_GO keeps the horizon
# from collapsing in a game that outlives the estimate -- past that point the policy degrades to
# dividing by a constant 20, which is geometric again and so cannot run the clock out.
EXPECTED_FINAL_FULLMOVE = 100
MIN_MOVES_TO_GO = 20

# The increment the rules give us. Hard-coded because the platform never tells us what it is.
INCREMENT_MS = 500.0

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


def _budget_ms(time_left_ms: int, fullmove: int) -> float:
    """How long this move may take, in milliseconds."""
    usable = max(0.0, float(time_left_ms) - MOVE_OVERHEAD_MS)
    moves_to_go = max(MIN_MOVES_TO_GO, EXPECTED_FINAL_FULLMOVE - fullmove)
    # Everything we will ever have for the moves that remain: the clock now, plus the increments
    # those moves will earn. Sharing that evenly is what makes the profile flat rather than
    # geometric -- the horizon shrinks alongside the clock, so the quotient barely moves.
    horizon = usable + (moves_to_go - 1) * INCREMENT_MS
    # Never stake more than a quarter of what is left on one move, however good the position looks.
    # This is also what makes flagging impossible: a quarter of the clock, after SAFETY, is below
    # the increment whenever the clock is short, so a short clock always recovers.
    return max(10.0, min(horizon / moves_to_go, usable * 0.25) * SAFETY)


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
        move, _score, _depth = _searcher.search(_budget_ms(time_left_ms, fullmove))
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

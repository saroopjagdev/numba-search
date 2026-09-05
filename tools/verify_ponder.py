"""Check that pondering does what it claims: runs, releases the GIL, and stops on demand.

Three properties, each of which fails silently if broken -- which is why they get an instrument
rather than a code review.

1. **The GIL is actually released.** If `negamax` were compiled without `nogil=True`, the ponder
   thread would hold the interpreter for the whole search and the main thread could not run at all.
   The agent would stop responding and lose on time, and nothing in a normal game would point at
   pondering as the cause. Measured by spinning a counter on the main thread and checking it moves.

2. **Stopping is prompt.** We have one core. A ponder search still running when the real search
   starts does not produce a wrong move, it produces a slow one -- the two halve each other -- and
   that is invisible except as mysteriously bad play under time pressure.

3. **The shared table is actually being filled**, and by the ponder searcher rather than nobody.
   Sharing is the entire point; a ponderer writing into its own private table would look perfectly
   healthy and buy exactly zero.

    uv run python tools/verify_ponder.py
"""

import sys
import time

from engine.ponder import Ponderer
from engine.search import Searcher

# A quiet middlegame with plenty of legal moves, so the ponder search has real work to do.
POSITION = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
PONDER_MS = 400.0


def main() -> None:
    main_searcher = Searcher()
    ponderer = Ponderer(main_searcher)

    if ponderer.searcher.tt[0] is not main_searcher.tt[0]:
        sys.exit("FAIL: the ponderer allocated its own table; sharing is the point")
    print("shared table    : yes (same arrays as the main searcher)")

    filled_before = int((main_searcher.tt[0] != 0).sum())

    ponderer.start(POSITION)

    # If the GIL were held by the ponder thread this loop would barely advance. It is a spin rather
    # than a sleep deliberately: sleeping releases the GIL and would pass even if nogil were off.
    spins = 0
    deadline = time.perf_counter() + PONDER_MS / 1000.0
    while time.perf_counter() < deadline:
        spins += 1

    stop_started = time.perf_counter()
    ponderer.stop()
    stop_ms = (time.perf_counter() - stop_started) * 1000.0

    filled_after = int((main_searcher.tt[0] != 0).sum())
    added = filled_after - filled_before

    print(f"main-thread spins: {spins:,} during {PONDER_MS:.0f}ms of pondering")
    print(f"stop latency     : {stop_ms:.1f}ms")
    print(f"table entries    : {filled_before:,} -> {filled_after:,}  (+{added:,})")

    failures = []
    # A blocked main thread manages a few hundred iterations at most; a free one manages millions.
    if spins < 100_000:
        failures.append(f"main thread only spun {spins} times -- the GIL is not being released")
    if stop_ms > 50.0:
        failures.append(f"stop took {stop_ms:.1f}ms -- too slow, the core would be contended")
    if added <= 0:
        failures.append("the ponder search added nothing to the shared table")
    if ponderer._thread is not None:
        failures.append("the ponder thread was not cleared after stop()")

    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        sys.exit(1)
    print("\nOK - pondering runs, releases the GIL, fills the shared table and stops on demand")


if __name__ == "__main__":
    main()

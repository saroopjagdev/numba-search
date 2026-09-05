"""Measure what the search does with the time it is given: how much it spends, and how far it
can overrun.

Time management is one of the three paths in this project where nothing mechanical catches a
mistake. It only misbehaves under clock pressure, and the failure is not gradual -- overrunning
once loses the game outright, whatever the position looked like.

Two numbers come out, and they are in tension:

- **Consumption**, elapsed over budget. Time allocated and not used is depth thrown away. Fitting
  the old policy to seven rated games put this at 75%, so a quarter of every move's allowance was
  being handed back.
- **Overrun**, the worst elapsed over budget across every position and every budget. This is the
  one that loses games. The search reads the clock only between jitted calls, so a call already
  running cannot be interrupted; what bounds it is the node budget, which is sized from the time
  actually remaining rather than from the whole allowance.

Reports the distribution rather than a mean, because a policy that averages well and has a long
tail is exactly the policy that flags once and loses.

Deliberately importable-agnostic: it uses whatever `engine` is on `sys.path`, so the same file can
be run against two checkouts and the results compared directly.

    uv run python tools/clock_fuzz.py --fens positions.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np

from engine.search import Searcher

# Short budgets are where the margin is proportionally tightest -- a fixed overhead that vanishes
# against 4 seconds is the whole allowance at 50 ms -- so the sweep is weighted towards them.
BUDGETS_MS = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)

FALLBACK_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fens", type=Path, default=None)
    parser.add_argument("--label", default="engine")
    arguments = parser.parse_args()

    if arguments.fens is not None:
        fens = [line for line in arguments.fens.read_text().splitlines() if line.strip()]
    else:
        fens = list(FALLBACK_FENS)

    searcher = Searcher()
    # One throwaway search so no measurement pays for jit compilation.
    searcher.set_position(fens[0])
    searcher.record_position()
    searcher.search(400.0)

    ratios: dict[float, list[float]] = {budget: [] for budget in BUDGETS_MS}
    depths: dict[float, list[int]] = {budget: [] for budget in BUDGETS_MS}

    for fen in fens:
        for budget in BUDGETS_MS:
            searcher.new_game()
            searcher.set_position(fen)
            searcher.record_position()
            started = time.perf_counter()
            _move, _score, depth = searcher.search(budget)
            elapsed = (time.perf_counter() - started) * 1000.0
            ratios[budget].append(elapsed / budget)
            depths[budget].append(depth)

    print(f"\n{arguments.label}: {len(fens)} positions x {len(BUDGETS_MS)} budgets\n")
    print(f"{'budget':>8} {'mean':>7} {'p50':>7} {'p95':>7} {'worst':>7}   {'mean depth':>10}")
    everything: list[float] = []
    for budget in BUDGETS_MS:
        values = np.array(ratios[budget])
        everything.extend(values.tolist())
        print(
            f"{budget:>7.0f}ms {values.mean():>6.0%} {np.percentile(values, 50):>6.0%} "
            f"{np.percentile(values, 95):>6.0%} {values.max():>6.0%}   "
            f"{np.mean(depths[budget]):>10.1f}"
        )

    combined = np.array(everything)
    print(f"\n  consumption, all budgets   {combined.mean():.0%}")
    print(f"  worst overrun              {combined.max():.0%}")
    print(f"  fraction over 100%         {(combined > 1.0).mean():.1%}")
    print(f"  fraction over 115%         {(combined > 1.15).mean():.1%}")


if __name__ == "__main__":
    main()

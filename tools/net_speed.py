"""What each network width costs in search, which is the other half of choosing one.

`tools/eval_quality.py` says wider is better and says nothing about price. Width is pure arithmetic
per evaluated node: the accumulator update is O(hidden) per moved piece and the output layer is
O(2*hidden), so 1024 does four times the work of 256 at every leaf. A net that is better per
position and slower per second can easily be worse per game, and depth is what pays for it.

Two columns, because on this machine only one of them can be trusted.

**Nodes** to reach a fixed depth is deterministic. The search is driven by a node budget rather
than a clock here, so the count does not depend on machine load at all, and it is the honest
measure of the one thing width buys inside the search: better evaluation orders moves better, and
better ordering cuts more. Identical across repeats or something is wrong.

**Seconds** is the minimum over several interleaved repeats, not the mean. Two earlier attempts at
timing this were thrown away -- one timed `Network.evaluate` in a Python loop and measured the
Python wrapper rather than the net; the next reported 1024 as *faster* than 128, which cannot be
true when it does eight times the arithmetic. Both were dominated by contention on a machine
running at 94% memory load. A minimum over repeats is the standard defence: load can only ever make
a run slower, so the fastest observation is the closest to the true cost. Interleaving the widths
within each repeat stops a slow patch of machine time from landing entirely on one net.

If the seconds column still disagrees with the nodes column times the expected O(hidden) cost, do
not believe the seconds. Believe neither, and measure on a quiet machine.

    uv run python tools/net_speed.py --nets net-files/net256.npz --fens positions.txt
"""

import argparse
import time
from pathlib import Path

from engine.nnue import Network, new_stack
from engine.position import MAX_PLY
from engine.search import INFINITY, Searcher

POSITIONS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
)

FIXED_DEPTH = 8
REPEATS = 3

# Large enough that the node budget never terminates an iteration early, so every width is made to
# search the identical tree to completion.
UNLIMITED_NODES = 1 << 60


def measure(searcher: Searcher, positions: list[str], depth: int) -> tuple[int, float]:
    """One full pass: returns (total nodes, elapsed seconds)."""
    nodes = 0
    started = time.perf_counter()
    for fen in positions:
        searcher.new_game()
        searcher.set_position(fen)
        searcher.record_position()
        _score, _move, count = searcher._iterate(depth, -INFINITY, INFINITY, UNLIMITED_NODES)
        nodes += count
    return nodes, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nets", type=Path, nargs="+", required=True)
    parser.add_argument("--depth", type=int, default=FIXED_DEPTH)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--fens", type=Path, default=None, help="one position per line")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N positions")
    arguments = parser.parse_args()

    positions = list(POSITIONS)
    if arguments.fens is not None:
        positions = [line for line in arguments.fens.read_text().splitlines() if line.strip()]
    if arguments.limit:
        positions = positions[: arguments.limit]

    searcher = Searcher()
    searcher.set_position(POSITIONS[0])
    searcher.record_position()
    searcher.search(400.0)

    networks = {}
    for path in arguments.nets:
        network = Network(path)
        if not network.available:
            raise SystemExit(f"{path}: no weights loaded")
        networks[path.stem] = network

    nodes: dict[str, set[int]] = {name: set() for name in networks}
    best: dict[str, float] = {name: float("inf") for name in networks}

    print(
        f"{len(positions)} positions to depth {arguments.depth},"
        f" {arguments.repeats} interleaved repeats"
    )
    for repeat in range(arguments.repeats):
        for name, network in networks.items():
            # Swap the loaded net into the live searcher rather than rebuilding it, so the
            # transposition table, killers and history are identical across widths and only the
            # weights differ. The accumulator stack is sized by hidden, so it has to be
            # reallocated with the net; leaving the old one would have the jitted code write a
            # 1024-wide row into a 256-wide buffer.
            searcher.network = network
            searcher.use_nnue = True
            searcher.acc = new_stack(network.hidden, MAX_PLY + 1)

            count, elapsed = measure(searcher, positions, arguments.depth)
            nodes[name].add(count)
            best[name] = min(best[name], elapsed)
        print(f"  repeat {repeat + 1} done", flush=True)

    baseline = 0.0
    print(f"\n  {'net':<10s} {'hidden':>7s} {'nodes':>12s} {'best s':>9s} {'relative':>9s}")
    for name, network in networks.items():
        counts = sorted(nodes[name])
        baseline = baseline or best[name]
        print(
            f"  {name:<10s} {network.hidden:>7d} {counts[0]:>12,} {best[name]:>9.2f}"
            f" {best[name] / baseline:>8.2f}x"
        )
        if len(counts) > 1:
            # A fixed-depth search with an unreachable node budget visits the same tree every time.
            # If it does not, the node column is not the load-immune measurement this tool claims
            # it is, and the whole comparison needs rethinking rather than reporting.
            print(f"      node counts varied across repeats: {counts} -- do not trust this row")


if __name__ == "__main__":
    main()

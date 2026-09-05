"""How often does the move played under a clock match the move a deeper search prefers?

Depth is not quality. Recovering the depths that used to be discarded when the clock ran out makes
the engine search further, which `tools/clock_fuzz.py` measures directly -- but a move taken from
an *unfinished* iteration could in principle be systematically worse than the finished shallower
move it replaces, and more depth reached that way would be worth nothing. Nothing in a node count
or a ply count can answer that.

An SPRT can, but on this machine it costs about twelve hours for roughly +-50 Elo, which is barely
finer than the effect. This is the cheaper instrument for the same question: search each position
under a real clock, search it again to a fixed depth with no clock at all, and count how often the
two agree.

The reference is deliberately **fixed depth with an effectively unlimited budget**, not a long
timed search. Under those conditions no clock logic runs in either build -- the partial-depth path
never fires and the floor that decides whether to start another iteration is never reached -- so
both builds compute the *same* reference move. A timed reference would have been contaminated: the
baseline's long search shares a code path with the baseline's short one, so it would agree with
itself more often and hand the comparison to whichever build produced the reference.

Agreement is not correctness. The reference is only this engine thinking longer, so a position
where the engine is simply wrong counts as agreement. That is fine for the purpose, which is a
*comparison* between two builds against a common yardstick, not an absolute score.

    uv run python tools/move_agreement.py --fens positions.txt --label candidate
"""

import argparse
import time
from pathlib import Path

import numpy as np

from engine.position import move_to_uci
from engine.search import Searcher, new_tt

# Budgets short enough that the clock genuinely bites, so the partial-depth path is exercised.
BUDGETS_MS = (50.0, 100.0, 200.0, 400.0, 800.0)

# Deeper than any of the budgets above reach, but not so deep the reference costs more than the
# rest of the run put together.
GOLD_DEPTH = 16

# Large enough that the clock never terminates the reference search.
UNLIMITED_MS = 1e9


def uci(move: int) -> str:
    return move_to_uci(np.int32(move))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fens", type=Path, required=True)
    parser.add_argument("--label", default="engine")
    parser.add_argument("--gold", type=Path, default=None, help="reuse or write reference moves")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N positions")
    parser.add_argument(
        "--tt-bits",
        type=int,
        default=0,
        help="shrink the transposition table; this laptop has under a gigabyte spare",
    )
    arguments = parser.parse_args()

    fens = [line for line in arguments.fens.read_text().splitlines() if line.strip()]
    if arguments.limit:
        fens = fens[: arguments.limit]
    searcher = Searcher()
    if arguments.tt_bits:
        # Both builds get the same table, so the comparison stays fair; a smaller table makes the
        # engine slightly weaker in absolute terms and that cancels.
        searcher.tt = new_tt(arguments.tt_bits)
    searcher.set_position(fens[0])
    searcher.record_position()
    searcher.search(400.0)

    gold: dict[str, str] = {}
    if arguments.gold is not None and arguments.gold.exists():
        for line in arguments.gold.read_text().splitlines():
            fen, _, move = line.rpartition("\t")
            gold[fen] = move
        print(f"loaded {len(gold)} reference moves from {arguments.gold}")

    missing = [fen for fen in fens if fen not in gold]
    if missing:
        started = time.perf_counter()
        for fen in missing:
            searcher.new_game()
            searcher.set_position(fen)
            searcher.record_position()
            move, _score, _depth = searcher.search(UNLIMITED_MS, max_depth=GOLD_DEPTH)
            gold[fen] = uci(move)
        print(
            f"computed {len(missing)} reference moves at fixed depth {GOLD_DEPTH}"
            f" in {(time.perf_counter() - started) / 60:.1f} min"
        )
        if arguments.gold is not None:
            arguments.gold.write_text("\n".join(f"{fen}\t{gold[fen]}" for fen in gold))

    # Recomputing what was loaded would defeat the point of caching, but a build that disagreed
    # with the cached reference would invalidate the comparison silently, so spot-check a few.
    for fen in fens[:3]:
        searcher.new_game()
        searcher.set_position(fen)
        searcher.record_position()
        move, _score, _depth = searcher.search(UNLIMITED_MS, max_depth=GOLD_DEPTH)
        if uci(move) != gold[fen]:
            raise SystemExit(
                f"reference mismatch on {fen}: this build says {uci(move)}, cache says {gold[fen]}."
                " The reference must be identical across builds or the comparison means nothing."
            )

    print(f"\n{arguments.label}: {len(fens)} positions against fixed depth {GOLD_DEPTH}\n")
    print(f"{'budget':>8} {'agreement':>10} {'mean depth':>11}")
    totals = []
    for budget in BUDGETS_MS:
        matches, depths = 0, []
        for fen in fens:
            searcher.new_game()
            searcher.set_position(fen)
            searcher.record_position()
            move, _score, depth = searcher.search(budget)
            depths.append(depth)
            matches += uci(move) == gold[fen]
        rate = matches / len(fens)
        totals.append(rate)
        print(f"{budget:>7.0f}ms {rate:>9.1%} {np.mean(depths):>11.1f}")

    print(f"\n  agreement, all budgets   {np.mean(totals):.1%}")


if __name__ == "__main__":
    main()

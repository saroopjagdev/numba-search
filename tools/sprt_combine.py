"""Pool sharded `tools/sprt.py` results and apply the test once, to the total.

A local SPRT is one process that stops the moment the evidence is decisive. Sharded across CI
runners it cannot work that way: no shard can see the pooled tally, so each plays its whole slice
and the test is applied here, afterwards, exactly once. Early stopping survives at batch
granularity instead -- run a batch, read the verdict, launch another only if it says `INCONCLUSIVE`.

The statistics are imported from `sprt.py` rather than reimplemented, so a fix to the LLR cannot
land in one path and not the other.

Two things this checks that a bare sum would not. Shards must be distinct, because a rerun of a
failed job can easily leave two artifacts for the same slice and silently double-count it. And
every shard must agree on the clock and the seed, because pooling games played under different
conditions produces a number that looks like a result and is not one.

    uv run python tools/sprt_combine.py results/*.json --elo0 0 --elo1 15
"""

import argparse
import json
import math
import sys
from pathlib import Path

from tools.sprt import elo_with_error_bars, log_likelihood_ratio

# Fields every shard must agree on. Pooling across different clocks or a different opening order is
# not a bigger sample, it is two samples added together.
MUST_MATCH = ("base_ms", "increment_ms", "seed", "candidate", "baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="shard JSON files")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    arguments = parser.parse_args()

    shards = [json.loads(path.read_text()) for path in arguments.results]
    if not shards:
        sys.exit("no shard results given")

    seen: dict[int, Path] = {}
    for path, shard in zip(arguments.results, shards, strict=True):
        index = int(shard["shard"])
        if index in seen:
            sys.exit(f"shard {index} appears twice: {seen[index]} and {path}")
        seen[index] = path
        for field in MUST_MATCH:
            if shard[field] != shards[0][field]:
                sys.exit(
                    f"{path}: {field} is {shard[field]!r} but {arguments.results[0]} has"
                    f" {shards[0][field]!r}. These games were not played under the same conditions"
                    " and must not be pooled."
                )

    expected = int(shards[0]["shards"])
    wins = sum(int(shard["wins"]) for shard in shards)
    draws = sum(int(shard["draws"]) for shard in shards)
    losses = sum(int(shard["losses"]) for shard in shards)
    failures: dict[str, int] = {}
    for shard in shards:
        for reason, count in shard["failures"].items():
            failures[reason] = failures.get(reason, 0) + int(count)

    lower = math.log(arguments.beta / (1.0 - arguments.alpha))
    upper = math.log((1.0 - arguments.beta) / arguments.alpha)
    llr = log_likelihood_ratio(wins, draws, losses, arguments.elo0, arguments.elo1)
    elo, margin = elo_with_error_bars(wins, draws, losses)

    verdict = "inconclusive"
    if llr >= upper:
        verdict = "accepted"
    elif llr <= lower:
        verdict = "rejected"

    total = wins + draws + losses
    clock = f"{shards[0]['base_ms'] / 1000:.0f}s + {shards[0]['increment_ms'] / 1000:.2f}s"
    print(f"SPRT  H0: {arguments.elo0:+.0f} Elo   H1: {arguments.elo1:+.0f} Elo   {clock}")
    print(f"      {shards[0]['candidate']} vs {shards[0]['baseline']}")
    print(f"      {len(shards)} of {expected} shards reported, {total} games pooled\n")
    if len(shards) < expected:
        # Not fatal. Shards are colour-balanced individually, so a missing one costs games rather
        # than skewing the result -- but a verdict reached on a partial field should be visible.
        print(f"  warning: {expected - len(shards)} shard(s) missing; this is a partial sample\n")

    print(
        f"  {verdict.upper()}  +{wins} ={draws} -{losses}   LLR {llr:+.2f}  in [{lower:.2f},"
        f" {upper:.2f}]"
    )
    print(f"  Elo {elo:+.1f} +- {margin:.1f}")
    if failures:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(failures.items()))
        print(f"  FAILURES: {detail}  <- investigate before trusting this result")
    print("\n  record this in notes/measurements.md")

    sys.exit({"accepted": 0, "rejected": 1, "inconclusive": 2}[verdict])


if __name__ == "__main__":
    main()

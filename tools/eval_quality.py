"""How well does each evaluation agree with Stockfish, on positions nothing was trained on?

The question this answers is "should the net replace the hand-crafted evaluation", and the honest
instrument for that is an SPRT. This machine cannot currently run one -- two agent processes need
about 680 MB and there is under half a gigabyte free -- so this measures the evaluations directly
instead, against the labels, on the two shards `--holdout 2` reserved and no run ever saw.

What it is not: a strength measurement. An evaluation is used inside a search, and search interacts
with it in ways a static comparison cannot see -- speed above all, since a net that is 20 cp better
per position and half the speed can easily be worse over a game. Treat a win here as necessary
rather than sufficient, and keep the SPRT as the thing that actually decides.

Reported in two spaces because they disagree about what matters. Centipawn error weights a 900 vs
1200 cp difference as heavily as 0 vs 300, when the first is two ways of saying "winning". WDL
error passes both through the same sigmoid the trainer optimised, which is much closer to what
costs games.

    uv run python tools/eval_quality.py --nets net-files/net256.npz net-files/net512.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np

from engine.eval import evaluate as hce_evaluate
from engine.nnue import Network
from engine.position import STM
from engine.search import Searcher
from tools.verify_nnue import fen_of
from training.dataset import ShardStream

# The trainer's WDL sigmoid. Same constant, so the comparison is in the space that was optimised.
SCALE = 400.0

DEFAULT_SHARDS = Path("C:/Users/ssjag/chessdata/shards_quiet")


def wdl(centipawns: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-centipawns / SCALE))


def report(name: str, predicted: np.ndarray, labels: np.ndarray) -> None:
    error = predicted - labels
    wdl_error = wdl(predicted) - wdl(labels)
    # Sign agreement ignores magnitude entirely: does it know who stands better? Positions inside
    # +-25 cp are dropped from it, because there the label's own sign is close to arbitrary.
    decisive = np.abs(labels) > 25.0
    agree = np.mean(np.sign(predicted[decisive]) == np.sign(labels[decisive]))
    print(
        f"  {name:<12s} {np.abs(error).mean():>8.1f} {np.sqrt((error**2).mean()):>8.1f}"
        f" {np.abs(wdl_error).mean():>9.4f} {np.corrcoef(predicted, labels)[0, 1]:>7.3f}"
        f" {agree:>8.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nets", type=Path, nargs="*", default=[])
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--positions", type=int, default=8192)
    parser.add_argument("--holdout", type=int, default=2)
    arguments = parser.parse_args()

    networks = {path.stem: Network(path) for path in arguments.nets}
    for name, network in networks.items():
        if not network.available:
            raise SystemExit(f"{name}: no weights loaded")

    searcher = Searcher()
    # `reserved=True` with the same holdout the training runs used. Scoring a net on shards it was
    # trained on would flatter it by exactly the amount it memorised.
    stream = ShardStream(
        arguments.shards, batch_size=1024, seed=11, holdout=arguments.holdout, reserved=True
    )
    print(f"holdout shards: {[path.name for path in stream.shards]}")

    labels: list[float] = []
    predictions: dict[str, list[float]] = {name: [] for name in ["hce", *networks]}
    started = time.perf_counter()

    for batch in stream.batches():
        offsets = np.append(batch["offsets"], len(batch["white"]))
        for position in range(len(batch["stm"])):
            if len(labels) >= arguments.positions:
                break
            low, high = int(offsets[position]), int(offsets[position + 1])
            features = batch["white"][low:high]
            fen = fen_of(features // 64, features % 64, bool(batch["stm"][position]))
            searcher.set_position(fen)
            stm = int(searcher.state[STM])
            labels.append(float(batch["score"][position]))
            predictions["hce"].append(
                float(hce_evaluate(searcher.bb, searcher.mailbox, searcher.state))
            )
            for name, network in networks.items():
                predictions[name].append(float(network.evaluate(searcher.mailbox, stm)))
        if len(labels) >= arguments.positions:
            break

    truth = np.array(labels)
    print(
        f"\n{len(truth):,} held-out positions in {time.perf_counter() - started:.0f}s,"
        f" label |mean| {np.abs(truth).mean():.0f} cp\n"
    )
    header = f"  {'eval':<12s} {'MAE cp':>8s} {'RMSE cp':>8s} {'WDL MAE':>9s} {'corr':>7s}"
    print(f"{header} {'sign':>8s}")
    for name in predictions:
        report(name, np.array(predictions[name]), truth)


if __name__ == "__main__":
    main()

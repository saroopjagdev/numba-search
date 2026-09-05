"""Cross-check the engine's NNUE inference against the trainer's reference implementation.

Training already compares its float model against `training/train.py:integer_eval`, so the
quantisation *scales* are covered. Nothing compares `integer_eval` against `engine/nnue.py`, and
those are two independently written implementations of the same arithmetic. Every way they can
disagree is silent -- the engine keeps returning plausible centipawns and simply plays worse:

- **Piece codes.** `training/preprocess.py` writes nibbles 0..11 = WP WN WB WR WQ WK BP..BK. The
  engine's mailbox uses its own constants. If those two orderings ever drift apart the net is
  queried with a permutation of the features it was trained on.
- **Perspective.** Black's view mirrors the board and swaps colours. Getting `square ^ 56` or the
  colour flip wrong produces a net that is subtly wrong for one side only.
- **Buckets.** `bucket_of` must agree with the bucket preprocessing baked into every record. A
  disagreement trains head *n* and queries head *n+1*, and all eight heads look reasonable.
- **Arithmetic.** `//` against `/`, int32 against int64 in the output dot product, the order of
  the `// QA` and the bias.

So the position goes in as a FEN and comes back out through the engine's own parser and its own
mailbox, rather than being handed feature indices directly. Handing over indices would make the
piece-code question cancel out on both sides and the test would pass while shipping a permuted net.

Runs before any net exists: with no `--net` it synthesises a random one, clamped and quantised
exactly as training would, which exercises identical arithmetic on non-degenerate weights.

    uv run python tools/verify_nnue.py
    uv run python tools/verify_nnue.py --net C:/Users/ssjag/chessdata/nets/net256.npz
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from engine.nnue import Network, bucket_of
from engine.position import STM
from engine.search import Searcher
from training.dataset import ShardStream
from training.train import integer_eval

DEFAULT_SHARDS = Path(r"C:/Users/ssjag/chessdata/shards_quiet")

# Index into this with the nibble `preprocess` wrote. Deliberately spelled out rather than derived
# from either side's constants: a test that imports the mapping it is checking proves nothing.
FEN_PIECES = "PNBRQKpnbrqk"

# The last step differs between the two by construction -- the trainer finishes in floating point,
# the engine floor-divides -- so a centipawn of disagreement is expected and anything more is not.
TOLERANCE_CP = 1.0


def fen_of(codes: np.ndarray, squares: np.ndarray, black_to_move: bool) -> str:
    """Rebuild a FEN from one record's features, so the engine parses the position itself."""
    board = [""] * 64
    for code, square in zip(codes, squares, strict=True):
        board[int(square)] = FEN_PIECES[int(code)]

    ranks = []
    for rank in range(7, -1, -1):
        line, empty = "", 0
        for file in range(8):
            piece = board[rank * 8 + file]
            if piece:
                line += (str(empty) if empty else "") + piece
                empty = 0
            else:
                empty += 1
        ranks.append(line + (str(empty) if empty else ""))
    # No castling rights and no en passant square: neither is a 768-feature input, so neither can
    # change the evaluation, and the records do not carry them.
    return f"{'/'.join(ranks)} {'b' if black_to_move else 'w'} - - 0 1"


def random_net(hidden: int, destination: Path) -> Path:
    """A quantised net with no training in it, for running this check before one exists."""
    import torch

    from training.train import Nnue, quantise, save_weights

    torch.manual_seed(0)
    model = Nnue(hidden)
    # Random init is small by design, which would leave the accumulator near zero and SCReLU almost
    # always clamped to 0 -- an arithmetic path that tests nothing. Scaling up to the clamp bounds
    # puts real magnitudes through every branch.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(40.0)
    model.clamp_weights()
    save_weights(destination, quantise(model))
    return destination.with_suffix(".npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, default=None, help="default: synthesise a random one")
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--positions", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=256, help="only used to synthesise a net")
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        net = arguments.net
        if net is None:
            net = random_net(arguments.hidden, Path(scratch) / "random")
            print(f"no --net given; synthesised a random hidden-{arguments.hidden} net")
        print(f"checking {net}\n")

        with np.load(net) as data:
            weights = {name: data[name] for name in data.files}
        network = Network(net)
        searcher = Searcher()

        stream = ShardStream(arguments.shards, batch_size=1024, seed=arguments.seed)
        batches = stream.batches()

        checked = bucket_mismatch = 0
        worst = 0.0
        worst_fen = ""
        differences: list[float] = []

        while checked < arguments.positions:
            batch = next(batches)
            reference = integer_eval(weights, batch)
            offsets = np.append(batch["offsets"], len(batch["white"]))

            for position in range(len(batch["stm"])):
                if checked >= arguments.positions:
                    break
                low, high = int(offsets[position]), int(offsets[position + 1])
                features = batch["white"][low:high]
                black_to_move = bool(batch["stm"][position])
                fen = fen_of(features // 64, features % 64, black_to_move)

                searcher.set_position(fen)
                mailbox = searcher.mailbox
                if int(bucket_of(mailbox)) != int(batch["bucket"][position]):
                    bucket_mismatch += 1

                engine_cp = float(network.evaluate(mailbox, int(searcher.state[STM])))
                difference = abs(engine_cp - float(reference[position]))
                differences.append(difference)
                if difference > worst:
                    worst, worst_fen = difference, fen
                checked += 1

    spread = np.array(differences)
    print(f"  positions checked        {checked:,}")
    print(f"  bucket disagreements     {bucket_mismatch}   (must be 0)")
    print(f"  mean |difference|        {spread.mean():.3f} cp")
    print(f"  worst |difference|       {worst:.3f} cp   (tolerance {TOLERANCE_CP})")
    print(f"  over tolerance           {int((spread > TOLERANCE_CP).sum())}")
    if worst > TOLERANCE_CP:
        print(f"  worst position           {worst_fen}")

    ok = bucket_mismatch == 0 and worst <= TOLERANCE_CP
    print("\nACCEPTED" if ok else "\nPROBLEM")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

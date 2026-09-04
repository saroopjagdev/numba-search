"""Train the NNUE and export int16 weights.

Nothing here ships -- torch never gets imported in the match container. The output is a `.npz` of
quantised integer arrays, which `engine/` loads and evaluates with numba.

Architecture: `(768 -> HIDDEN) x 2 -> 1`, SCReLU, eight output buckets by piece count.

Deliberately *not* HalfKP. Three of HalfKP's four problems are properties of the shipped net rather
than of training, so no amount of cloud GPU fixes them: a 21 MB table has poor locality on one
core shared with an opponent hammering the same L3, king moves force a full accumulator refresh
over that table, and numba cannot emit the hand-tuned int8 AVX2 that makes HalfKP pay in C++.
With plain 768 inputs there is no king refresh at all -- every move is an incremental update.

Quantisation, and why the scales are what they are
--------------------------------------------------
`QA = 255` scales the feature transformer, `QB = 64` the output layer. SCReLU squares a value
clamped to `[0, QA]`, so the square reaches `QA^2 = 65025` and must live in int32; the accumulator
itself is int16, which is what makes the incremental update cheap.

    accumulator     int16, QA * float
    screlu          clamp(acc, 0, QA)^2       -> int32, QA^2 * float
    output weights  int16, QB * float
    eval_cp         (sum(screlu * w) / QA + bias * QA * QB) * SCALE / (QA * QB)

Weights are clamped every step rather than only at export, so the network trains inside the range
it will be quantised into instead of being lopped off at the end. The output weights are held to
`+-127/QB`, which is the widest value `int16 * QB` can represent without risking overflow once
summed over the hidden layer.

None of the above is trusted. `validate_quantisation` scores held-out positions in float and in
integer arithmetic and reports the worst divergence; a few centipawns is quantisation noise, tens
of centipawns is a scale bug. That check is the gate, because a wrong shift here is otherwise
completely silent -- the net simply plays slightly worse, forever.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from training.dataset import BUCKETS, FEATURES, SCALE, ShardStream

QA = 255
QB = 64

# The widest output weight that survives quantisation to int16 at QB.
OUTPUT_CLAMP = 127.0 / QB


def screlu(x: Tensor) -> Tensor:
    """Squared clipped ReLU. The square is what gives the net a cheap non-linearity that still
    quantises exactly, since clamping to [0, 1] before squaring keeps everything in range."""
    return torch.clamp(x, 0.0, 1.0) ** 2


class Nnue(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.hidden = hidden
        # sparse sum over the active features -- exactly what the incremental accumulator does at
        # match time, so training and inference compute the same quantity by construction.
        self.transformer = nn.EmbeddingBag(FEATURES, hidden, mode="sum")
        self.transformer_bias = nn.Parameter(torch.zeros(hidden))
        self.output = nn.Parameter(torch.zeros(BUCKETS, 2 * hidden))
        self.output_bias = nn.Parameter(torch.zeros(BUCKETS))

        # Random initialisation, and nothing else. Starting from a published chess network is
        # disqualifying, so this is the only initialisation the project is permitted to use.
        bound = 1.0 / np.sqrt(hidden)
        nn.init.uniform_(self.transformer.weight, -bound, bound)
        nn.init.uniform_(self.output, -bound, bound)

    def forward(
        self,
        white: Tensor,
        black: Tensor,
        offsets: Tensor,
        stm: Tensor,
        bucket: Tensor,
    ) -> Tensor:
        white_acc = self.transformer(white, offsets) + self.transformer_bias
        black_acc = self.transformer(black, offsets) + self.transformer_bias

        side = stm.unsqueeze(1).to(white_acc.dtype)
        # stm == 0 means white to move, so `us` is the white accumulator then.
        us = white_acc * (1.0 - side) + black_acc * side
        them = black_acc * (1.0 - side) + white_acc * side

        hidden = screlu(torch.cat([us, them], dim=1))
        weights = self.output[bucket]
        return (hidden * weights).sum(dim=1) + self.output_bias[bucket]

    @torch.no_grad()
    def clamp_weights(self) -> None:
        """Hold the parameters inside the range quantisation can represent."""
        self.output.clamp_(-OUTPUT_CLAMP, OUTPUT_CLAMP)
        # The accumulator is int16 and holds QA * (sum of ~32 weights plus the bias). Bounding each
        # weight well below 127/QA leaves room for that sum without ever overflowing.
        self.transformer.weight.clamp_(-1.98, 1.98)
        self.transformer_bias.clamp_(-1.98, 1.98)


def to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    return {
        "white": torch.from_numpy(batch["white"].astype(np.int64)).to(device),
        "black": torch.from_numpy(batch["black"].astype(np.int64)).to(device),
        "offsets": torch.from_numpy(batch["offsets"]).to(device),
        "stm": torch.from_numpy(batch["stm"].astype(np.int64)).to(device),
        "bucket": torch.from_numpy(batch["bucket"].astype(np.int64)).to(device),
        "score": torch.from_numpy(batch["score"]).to(device),
    }


def loss_of(prediction: Tensor, score: Tensor) -> Tensor:
    """Mean squared error in win-probability space, not in centipawns.

    Centipawn error weights a blunder from +900 to +1200 as heavily as one from +0 to +300, when
    only the second changes the result. Passing both through a sigmoid at `SCALE` fixes that, and
    is why the loss is computed here rather than with a bare MSE on the raw evaluation.
    """
    return ((torch.sigmoid(prediction) - torch.sigmoid(score / SCALE)) ** 2).mean()


def quantise(model: Nnue) -> dict[str, np.ndarray]:
    """Integer weights in the layout the engine reads."""
    with torch.no_grad():
        transformer = model.transformer.weight.detach().cpu().numpy()
        return {
            # Left as [features, hidden], which is what torch already stores, so that the row for
            # one feature is 256 contiguous int16. That is the only layout the accumulator wants:
            # a move changes a handful of features and each one is then a single 512-byte run the
            # engine adds or subtracts. Transposing to [hidden, features] would put a feature on a
            # 1536-byte stride and cost a cache line per element.
            "transformer": np.round(transformer * QA).astype(np.int16),
            "transformer_bias": np.round(model.transformer_bias.detach().cpu().numpy() * QA).astype(
                np.int16
            ),
            "output": np.round(model.output.detach().cpu().numpy() * QB).astype(np.int16),
            "output_bias": np.round(model.output_bias.detach().cpu().numpy() * QA * QB).astype(
                np.int32
            ),
            "meta": np.array([model.hidden, QA, QB, int(SCALE)], dtype=np.int32),
        }


def save_weights(path: Path, weights: dict[str, np.ndarray]) -> None:
    """Write the quantised arrays. numpy's `savez` stub types its keyword arguments as `bool`,
    which is what the ignore is for -- the call itself is the documented usage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)  # type: ignore[arg-type]


def integer_eval(weights: dict[str, np.ndarray], batch: dict[str, np.ndarray]) -> np.ndarray:
    """Score a batch using only integer arithmetic, exactly as the engine will.

    This exists to be compared against the float model. It is the only check that the quantisation
    scales are right, and it is deliberately written from the documented formula rather than by
    reusing anything from `quantise`.
    """
    transformer = weights["transformer"].astype(np.int32)  # [features, hidden]
    transformer_bias = weights["transformer_bias"].astype(np.int32)
    output = weights["output"].astype(np.int32)
    output_bias = weights["output_bias"].astype(np.int64)
    hidden = int(weights["meta"][0])

    count = len(batch["stm"])
    offsets = np.append(batch["offsets"], len(batch["white"]))
    results = np.empty(count, dtype=np.float64)

    for position in range(count):
        start, end = int(offsets[position]), int(offsets[position + 1])
        white_acc = transformer_bias + transformer[batch["white"][start:end]].sum(axis=0)
        black_acc = transformer_bias + transformer[batch["black"][start:end]].sum(axis=0)
        if int(batch["stm"][position]) == 0:
            accumulator = np.concatenate([white_acc, black_acc])
        else:
            accumulator = np.concatenate([black_acc, white_acc])

        clipped = np.clip(accumulator, 0, QA).astype(np.int64)
        activated = clipped * clipped
        total = int((activated * output[int(batch["bucket"][position])].astype(np.int64)).sum())
        total = total // QA + int(output_bias[int(batch["bucket"][position])])
        results[position] = total * SCALE / (QA * QB)

    assert hidden * 2 == output.shape[1]
    return results


def validate_quantisation(
    model: Nnue, batch: dict[str, np.ndarray], device: torch.device
) -> tuple[float, float]:
    """Worst and mean divergence, in centipawns, between the float net and the integer net."""
    model.eval()
    tensors = to_tensors(batch, device)
    with torch.no_grad():
        reference = model(
            tensors["white"],
            tensors["black"],
            tensors["offsets"],
            tensors["stm"],
            tensors["bucket"],
        )
        reference_cp = (reference * SCALE).cpu().numpy().astype(np.float64)
    model.train()
    integer_cp = integer_eval(quantise(model), batch)
    difference = np.abs(reference_cp - integer_cp)
    return float(difference.max()), float(difference.mean())


def train(
    shards: Path,
    output: Path,
    hidden: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    device_name: str,
    holdout: int,
) -> None:
    device = torch.device(device_name)
    model = Nnue(hidden).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, steps))

    stream = ShardStream(shards, batch_size=batch_size, seed=1, holdout=holdout)
    validation = next(ShardStream(shards, batch_size=8192, seed=99, holdout=0).batches())

    print(f"{stream.total_records:,} training records across {len(stream.shards)} shards")
    print(f"hidden {hidden}, batch {batch_size}, {steps:,} steps on {device_name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    running = 0.0
    step = 0
    for step, batch in enumerate(stream.batches(), start=1):
        tensors = to_tensors(batch, device)
        prediction = model(
            tensors["white"],
            tensors["black"],
            tensors["offsets"],
            tensors["stm"],
            tensors["bucket"],
        )
        loss = loss_of(prediction, tensors["score"])

        optimiser.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        schedule.step()
        model.clamp_weights()

        running += float(loss.detach())
        if step % 200 == 0:
            elapsed = time.perf_counter() - started
            rate = step * batch_size / elapsed
            print(
                f"  step {step:>7,}/{steps:,}  loss {running / 200:.5f}"
                f"  {rate:,.0f} pos/s  {elapsed / 60:.1f}m",
                flush=True,
            )
            running = 0.0
        if step % 5000 == 0 or step == steps:
            # Checkpoint often. A free Colab session can be killed at any moment, and an epoch of
            # this size is hours -- losing one is losing an evening.
            worst, mean = validate_quantisation(model, validation, device)
            print(
                f"  quantisation divergence: worst {worst:.1f} cp, mean {mean:.2f} cp", flush=True
            )
            save_weights(output, quantise(model))
            torch.save(model.state_dict(), output.with_suffix(".pt"))
            print(f"  saved {output}", flush=True)
        if step >= steps:
            break

    worst, mean = validate_quantisation(model, validation, device)
    print(f"final quantisation divergence: worst {worst:.1f} cp, mean {mean:.2f} cp")
    save_weights(output, quantise(model))
    torch.save(model.state_dict(), output.with_suffix(".pt"))
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NNUE.")
    parser.add_argument("--shards", type=Path, default=Path(r"C:/Users/ssjag/chessdata/shards"))
    parser.add_argument(
        "--output", type=Path, default=Path(r"C:/Users/ssjag/chessdata/nets/net.npz")
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--holdout", type=int, default=4, help="shards reserved for validation")
    arguments = parser.parse_args()
    train(
        arguments.shards,
        arguments.output,
        arguments.hidden,
        arguments.batch_size,
        arguments.steps,
        arguments.learning_rate,
        arguments.device,
        arguments.holdout,
    )


if __name__ == "__main__":
    main()

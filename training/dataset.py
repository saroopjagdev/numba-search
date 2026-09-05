"""Stream the packed shards into training batches.

Nothing here ships. The decode kernel is jitted for the same reason the engine is: a batch is
16k positions of roughly 24 pieces each, so a step needs ~400k feature indices, and building those
in a Python loop would cost more than the gradient step it feeds.

The two perspectives are the whole point of the architecture. Every position produces two feature
sets over the same 768 inputs -- one as white sees it, one as black sees it, the latter with
colours swapped and squares vertically mirrored. The network is then fed
`[side-to-move accumulator, other accumulator]`, which makes it structurally colour-symmetric: a
position and its mirror image produce identical activations. That matters here specifically,
because the source data is skewed toward white (mean +210 cp, and the material distribution is
itself lopsided). A net taking white-relative inputs would learn that skew as a bias; this one
cannot represent it, provided the target is also converted to side-to-move relative, which
`load_batch` does.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numba import njit

RECORD_SIZE = 32
FEATURES = 768
BUCKETS = 8

# The centipawn scale of the sigmoid that turns an evaluation into a win probability. 400 is the
# usual figure and is not tuned here; it sets how much the loss cares about the difference between
# +300 and +600, which is very little, against +0 and +100, which is a great deal.
SCALE = 400.0


@njit(
    "int64(uint8[:], int64, int32[:], int32[:], int64[:], uint8[:], uint8[:], int16[:])", cache=True
)
def decode_batch(
    data: np.ndarray,
    count: int,
    white_indices: np.ndarray,
    black_indices: np.ndarray,
    offsets: np.ndarray,
    stm: np.ndarray,
    bucket: np.ndarray,
    score: np.ndarray,
) -> int:
    """Expand packed records into flat feature indices plus per-position offsets.

    Returns the number of feature indices written. Both perspectives always have the same piece
    count, so a single offsets array serves both.
    """
    cursor = 0
    for position in range(count):
        base = position * RECORD_SIZE
        offsets[position] = cursor

        occupancy = np.uint64(0)
        for byte in range(8):
            occupancy |= np.uint64(data[base + byte]) << np.uint64(8 * byte)

        # A straight scan of all 64 squares, not a bit-scan loop: it visits squares in ascending
        # order, which is the order the nibbles were written in, and 64 fixed iterations beat
        # shifting a mask down for each of ~24 pieces.
        slot = 0
        for square in range(64):
            if (occupancy >> np.uint64(square)) & np.uint64(1) == np.uint64(0):
                continue

            packed = data[base + 8 + slot // 2]
            code = np.int64(packed & 0xF) if slot % 2 == 0 else np.int64(packed >> 4)
            slot += 1

            white_indices[cursor] = np.int32(code * 64 + square)
            # Black's view: swap the colour of every piece and mirror the board vertically.
            flipped = code - 6 if code >= 6 else code + 6
            black_indices[cursor] = np.int32(flipped * 64 + (square ^ 56))
            cursor += 1

        score[position] = np.int16(data[base + 24]) | (np.int16(data[base + 25]) << 8)
        stm[position] = data[base + 26]
        bucket[position] = data[base + 27]
    offsets[count] = cursor
    return cursor


class ShardStream:
    """Reads shards in a random order, shuffling within a block before yielding batches.

    Preprocessing already scattered records across shards at random, which is a shuffle across the
    whole file. This adds the second half: a block of records is read, permuted in memory and cut
    into batches. Together they approximate a full shuffle of a file far larger than RAM, without
    ever needing to sort 11 GB.
    """

    def __init__(
        self,
        directory: Path,
        batch_size: int = 16384,
        block_records: int = 1 << 20,
        seed: int = 0,
        holdout: int = 0,
        reserved: bool = False,
    ) -> None:
        self.shards = sorted(directory.glob("shard*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"no shard*.bin in {directory}")
        # The last few shards are reserved for validation so that no position the net trains on is
        # ever scored against it. Shards are interchangeable, so holding out whole files is enough.
        #
        # `reserved` selects which side of that split you get: False for the training majority,
        # True for the withheld remainder. Both callers must pass the *same* holdout, or the split
        # they think they are on either side of is not the same split.
        if holdout:
            if reserved:
                self.shards = self.shards[-holdout:]
            else:
                self.shards = self.shards[:-holdout]
        elif reserved:
            raise ValueError("reserved=True is meaningless with holdout=0: nothing was withheld")
        self.batch_size = batch_size
        self.block_records = block_records
        self.rng = np.random.default_rng(seed)

    @property
    def total_records(self) -> int:
        return sum(path.stat().st_size // RECORD_SIZE for path in self.shards)

    def blocks(self) -> Iterator[np.ndarray]:
        order = self.rng.permutation(len(self.shards))
        for shard_index in order:
            path = self.shards[shard_index]
            with open(path, "rb") as handle:
                while True:
                    raw = handle.read(self.block_records * RECORD_SIZE)
                    if len(raw) < RECORD_SIZE:
                        break
                    flat = np.frombuffer(raw, dtype=np.uint8)
                    count = len(flat) // RECORD_SIZE
                    block = flat[: count * RECORD_SIZE].reshape(count, RECORD_SIZE)
                    yield block[self.rng.permutation(count)]

    def batches(self) -> Iterator[dict[str, np.ndarray]]:
        for block in self.blocks():
            for start in range(0, len(block) - self.batch_size + 1, self.batch_size):
                yield self._decode(block[start : start + self.batch_size])

    def _decode(self, records: np.ndarray) -> dict[str, np.ndarray]:
        count = len(records)
        flat = np.ascontiguousarray(records).reshape(-1)
        capacity = count * 32
        white = np.empty(capacity, dtype=np.int32)
        black = np.empty(capacity, dtype=np.int32)
        offsets = np.empty(count + 1, dtype=np.int64)
        stm = np.empty(count, dtype=np.uint8)
        bucket = np.empty(count, dtype=np.uint8)
        score = np.empty(count, dtype=np.int16)
        written = int(decode_batch(flat, count, white, black, offsets, stm, bucket, score))
        return {
            "white": white[:written],
            "black": black[:written],
            "offsets": offsets[:count],
            "stm": stm,
            "bucket": bucket,
            # White-relative on disk, side-to-move relative here. See the module docstring: the
            # architecture is colour-symmetric only if the target is too.
            "score": score.astype(np.float32) * np.where(stm == 1, -1.0, 1.0).astype(np.float32),
        }

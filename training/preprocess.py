"""Turn the Lichess evaluations database into fixed-width records the trainer can stream.

Nothing here ships. `harness/package.py` collects root `*.py` plus whatever `--include` names, so
this directory is never in the zip -- which is the point, since it imports `zstandard` and `orjson`
and neither exists in the match container.

Input is `lichess_db_eval.jsonl.zst`: 21.7 GB compressed, 394,669,566 positions, CC0. It is never
expanded to disk (~150 GB) -- the stream is decompressed, parsed and discarded record by record.

Output is 32-byte records written to `SHARDS` files chosen uniformly at random. Random sharding is
half of the shuffle; the trainer shuffles within a shard, and the two together approximate a full
shuffle over a file far larger than memory. The record stores a *position*, not a list of feature
indices: the board costs 24 bytes where 32 indices would cost 64, and expanding it is a few
nanoseconds against a training step measured in milliseconds.

    byte  0.. 7  occupancy bitboard, little-endian uint64
    byte  8..23  one nibble per occupied square in ascending square order, low nibble first;
                 the nibble is the engine's own piece code, 0..11 = WP WN WB WR WQ WK BP .. BK
    byte 24..25  int16 score in centipawns, **white-relative**, clamped to +-2000
    byte 26      side to move, 0 white 1 black
    byte 27      output bucket, 0..7, by piece count
    byte 28..31  reserved, zero

Three properties of the source were established by measurement, not by reading, because all three
are silent if wrong. `notes/measurements.md` records them:

- **`cp` is white-relative, not side-to-move.** Multi-PV lines are ordered best-first for the side
  to move, so white-relative scores run ascending when black is to move. Over 200k sampled
  records: black to move 86,599 ascending against 3 descending, and the mirror image for white.
  Getting this backwards trains a net that plays the opponent's side.
- **`depth >= 20` keeps 91.3%** of positions, so roughly 360M survive.
- **The file contains illegal positions.** Lichess evaluates boards its users set up by hand, so
  record 7,106 alone has seventeen black pieces and three black knights. They are filtered here;
  training on them would spend capacity fitting positions that cannot arise in a game.

Throughput is bounded by zstd, not by us. Decompression of this file runs at ~32 MB/s on one core
against a disk that does 379 MB/s, so the serial floor for a full pass is a bit over an hour. The
parse is therefore pushed to a worker pool and overlapped with it rather than optimised.
"""

from __future__ import annotations

import argparse
import multiprocessing
import random
import struct
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import zstandard

# 12 piece types x 64 squares. Not HalfKP: with plain piece-square inputs a king move is an
# ordinary incremental update, where HalfKP would force a full accumulator refresh.
FEATURES = 768

SHARDS = 64
RECORD_SIZE = 32

# Beyond this the position is decided and the exact number carries no information worth fitting.
# Mates map to the clamp rather than being dropped -- "winning" is the signal, not the ply count.
SCORE_CLAMP = 2000

MIN_DEPTH = 20

CHUNK_LINES = 20_000

# Piece letters in the engine's own order, so a feature index computed here and one computed by the
# engine at match time cannot drift apart.
PIECE_CODES = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}  # fmt: skip


def parse_board(field: str) -> tuple[int, list[int]]:
    """Read a FEN placement field into (occupancy, piece codes in ascending square order).

    FEN lists rank 8 first and the engine numbers a1 as square 0, so the ranks are walked in
    reverse. That ordering is not cosmetic: it means the codes come out already sorted by square,
    and a sort of 32 items avoided 394 million times is worth the one-line comment.
    """
    occupancy = 0
    codes: list[int] = []
    square = 0
    for rank in reversed(field.split("/")):
        for character in rank:
            if character.isdigit():
                square += int(character)
            else:
                occupancy |= 1 << square
                codes.append(PIECE_CODES[character])
                square += 1
    return occupancy, codes


def is_plausible(occupancy: int, codes: list[int]) -> bool:
    """Reject positions that could not occur in a game.

    Not a full legality check -- that would need attack generation for a marginal gain. This
    catches the hand-built boards the database actually contains: wrong king count, too many
    pieces or pawns of a colour, and pawns on the back ranks.
    """
    if len(codes) > 32:
        return False
    white = black = white_pawns = black_pawns = white_kings = black_kings = 0
    remaining = occupancy
    for code in codes:
        square = (remaining & -remaining).bit_length() - 1
        remaining &= remaining - 1
        if code < 6:
            white += 1
            if code == 0:
                white_pawns += 1
                if square < 8 or square >= 56:
                    return False
            elif code == 5:
                white_kings += 1
        else:
            black += 1
            if code == 6:
                black_pawns += 1
                if square < 8 or square >= 56:
                    return False
            elif code == 11:
                black_kings += 1
    return (
        white_kings == 1
        and black_kings == 1
        and white <= 16
        and black <= 16
        and white_pawns <= 8
        and black_pawns <= 8
    )


def encode(occupancy: int, codes: list[int], score: int, stm: int) -> bytes:
    """Pack one position into its 32 bytes."""
    nibbles = bytearray(16)
    for index, code in enumerate(codes):
        if index % 2 == 0:
            nibbles[index // 2] |= code
        else:
            nibbles[index // 2] |= code << 4
    # Eight buckets over 2..32 pieces. The net learns a separate output head per bucket, which is
    # most of what a phase-tapered evaluation buys, learned rather than hand-set.
    bucket = min(7, max(0, (len(codes) - 2) // 4))
    return (
        struct.pack("<Q", occupancy) + bytes(nibbles) + struct.pack("<hBBI", score, stm, bucket, 0)
    )


def best_score(evals: list[dict[str, Any]]) -> int | None:
    """The deepest evaluation's principal line, as a clamped white-relative centipawn score."""
    best_depth = -1
    chosen = None
    for entry in evals:
        if entry["depth"] > best_depth and entry["pvs"]:
            best_depth = entry["depth"]
            chosen = entry["pvs"][0]
    if chosen is None or best_depth < MIN_DEPTH:
        return None
    if "cp" in chosen:
        return max(-SCORE_CLAMP, min(SCORE_CLAMP, int(chosen["cp"])))
    mate = int(chosen["mate"])
    # A mate of 0 means the game is already over; treat it as unusable rather than guess a sign.
    if mate == 0:
        return None
    return SCORE_CLAMP if mate > 0 else -SCORE_CLAMP


def process_chunk(job: tuple[int, list[bytes]]) -> tuple[int, list[bytes]]:
    """Parse a batch of raw JSON lines into per-shard blobs. Runs in a worker process.

    The shard is drawn here rather than in the parent so the randomness costs nothing serial. It is
    seeded from the chunk index, so a re-run with the same input reproduces the same split.
    """
    index, lines = job
    rng = random.Random(index)
    buckets: list[bytearray] = [bytearray() for _ in range(SHARDS)]
    kept = 0
    for line in lines:
        record = orjson.loads(line)
        score = best_score(record["evals"])
        if score is None:
            continue
        fen = record["fen"]
        placement, _, rest = fen.partition(" ")
        occupancy, codes = parse_board(placement)
        if not is_plausible(occupancy, codes):
            continue
        buckets[rng.randrange(SHARDS)] += encode(
            occupancy, codes, score, 1 if rest[0] == "b" else 0
        )
        kept += 1
    return kept, [bytes(bucket) for bucket in buckets]


def read_chunks(source: Path, limit: int) -> Iterator[tuple[int, list[bytes]]]:
    """Decompress and split into batches of lines. This is the serial bottleneck; keep it bare."""
    index = 0
    read = 0
    tail = b""
    pending: list[bytes] = []
    with open(source, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        while True:
            block = reader.read(1 << 22)
            if not block:
                break
            lines = (tail + block).split(b"\n")
            tail = lines.pop()
            pending.extend(lines)
            read += len(lines)
            while len(pending) >= CHUNK_LINES:
                yield index, pending[:CHUNK_LINES]
                pending = pending[CHUNK_LINES:]
                index += 1
            if limit and read >= limit:
                break
    if pending:
        yield index, pending


def run(source: Path, destination: Path, limit: int, workers: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    # Sixty-four files held open for the whole pass, which is why they are not context-managed
    # individually; the `finally` below closes them. Nesting 64 `with` blocks would say the same
    # thing at far greater length.
    handles = [
        open(destination / f"shard{shard:02d}.bin", "wb", buffering=1 << 22)  # noqa: SIM115
        for shard in range(SHARDS)
    ]

    written = 0
    chunks = 0
    started = time.perf_counter()
    try:
        with multiprocessing.Pool(workers) as pool:
            for kept, blobs in pool.imap(process_chunk, read_chunks(source, limit), chunksize=1):
                written += kept
                chunks += 1
                for handle, blob in zip(handles, blobs, strict=True):
                    if blob:
                        handle.write(blob)
                if chunks % 250 == 0:
                    elapsed = time.perf_counter() - started
                    read = chunks * CHUNK_LINES
                    print(
                        f"  {read:,} read  {written:,} kept  {read / elapsed:,.0f}/s"
                        f"  {written * RECORD_SIZE / 1e9:.2f} GB  {elapsed / 60:.0f}m",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.close()

    elapsed = time.perf_counter() - started
    read = chunks * CHUNK_LINES
    print(f"read ~{read:,}, kept {written:,} in {elapsed / 60:.1f} minutes")
    print(f"{written * RECORD_SIZE / 1e9:.2f} GB across {SHARDS} shards in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the Lichess eval database.")
    parser.add_argument(
        "--source", type=Path, default=Path(r"C:/Users/ssjag/chessdata/lichess_db_eval.jsonl.zst")
    )
    parser.add_argument(
        "--destination", type=Path, default=Path(r"C:/Users/ssjag/chessdata/shards")
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after this many input records")
    parser.add_argument(
        "--workers", type=int, default=max(1, (multiprocessing.cpu_count() * 3) // 4)
    )
    arguments = parser.parse_args()
    run(arguments.source, arguments.destination, arguments.limit, arguments.workers)


if __name__ == "__main__":
    main()

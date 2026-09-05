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


def best_score(evals: list[dict[str, Any]]) -> tuple[int, str] | None:
    """The deepest evaluation's principal line, as a clamped white-relative score and its move.

    The move comes back with the score because the quiet-position filter needs it, and finding it
    a second time would mean walking the eval list twice for every one of 394 million records.
    """
    best_depth = -1
    chosen = None
    for entry in evals:
        if entry["depth"] > best_depth and entry["pvs"]:
            best_depth = entry["depth"]
            chosen = entry["pvs"][0]
    if chosen is None or best_depth < MIN_DEPTH:
        return None
    move = chosen.get("line", "").split(" ", 1)[0]
    if not move:
        return None
    if "cp" in chosen:
        return max(-SCORE_CLAMP, min(SCORE_CLAMP, int(chosen["cp"]))), move
    mate = int(chosen["mate"])
    # A mate of 0 means the game is already over; treat it as unusable rather than guess a sign.
    if mate == 0:
        return None
    return (SCORE_CLAMP if mate > 0 else -SCORE_CLAMP), move


def _knight_targets() -> list[list[int]]:
    table = []
    for square in range(64):
        file, rank = square & 7, square >> 3
        squares = []
        for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
            f, r = file + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8:
                squares.append(r * 8 + f)
        table.append(squares)
    return table


def _king_targets() -> list[list[int]]:
    table = []
    for square in range(64):
        file, rank = square & 7, square >> 3
        squares = []
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df or dr:
                    f, r = file + df, rank + dr
                    if 0 <= f < 8 and 0 <= r < 8:
                        squares.append(r * 8 + f)
        table.append(squares)
    return table


def _rays() -> list[list[list[int]]]:
    """Squares outward from each square along each of eight directions, nearest first.

    Orthogonals occupy indices 0..3 and diagonals 4..7, which is what lets the attack test pick the
    right pair of attacker types by comparing the direction index against four.
    """
    table = []
    for square in range(64):
        file, rank = square & 7, square >> 3
        per_direction = []
        for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            squares = []
            f, r = file + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                squares.append(r * 8 + f)
                f += df
                r += dr
            per_direction.append(squares)
        table.append(per_direction)
    return table


KNIGHT_TARGETS = _knight_targets()
KING_TARGETS = _king_targets()
RAYS = _rays()


def in_check(mailbox: bytearray, king_square: int, black_to_move: bool) -> bool:
    """Is the king on `king_square` attacked?

    Walks outward from the king rather than testing every enemy piece: a king has at most eight
    knight squares and eight rays, where the board has up to thirty-one other pieces. `mailbox`
    holds piece codes plus one, so that zero can mean empty.
    """
    enemy = 0 if black_to_move else 6  # code offset of the attacking side
    for square in KNIGHT_TARGETS[king_square]:
        if mailbox[square] == enemy + 2:  # knight
            return True
    for square in KING_TARGETS[king_square]:
        if mailbox[square] == enemy + 6:  # king
            return True

    # Pawns attack toward the far rank, so the attacker sits one rank back from the king.
    king_file = king_square & 7
    step = -8 if black_to_move else 8
    for side in (-1, 1):
        square = king_square + step + side
        if 0 <= square < 64 and abs((square & 7) - king_file) == 1 and mailbox[square] == enemy + 1:
            return True

    for direction, ray in enumerate(RAYS[king_square]):
        for square in ray:
            piece = mailbox[square]
            if piece:
                if piece == enemy + 5:  # queen, attacks along every direction
                    return True
                if piece == enemy + (4 if direction < 4 else 3):  # rook orthogonal, bishop diagonal
                    return True
                break  # the first piece on a ray blocks everything behind it
    return False


def is_quiet(occupancy: int, codes: list[int], black_to_move: bool, move: str) -> bool:
    """Would this position's evaluation survive a quiescence search unchanged?

    Positions whose best move is a capture or a promotion, and positions where the side to move is
    in check, are excluded. The reason is that the net is only ever asked to score the *leaves* of a
    search that has already resolved captures -- so a tactical position is both a question it will
    never be asked and a noisy label, since its true value depends on a tactic rather than on the
    structure the net can see. This is the standard rule: Stockfish's trainer calls it smart fen
    skipping, Arasan's generator applies exactly capture-plus-check, and arXiv:2412.17948 studies
    it.

    Not implemented here is that paper's stronger qsearch-delta filter, which catches quiet-looking
    positions that are nonetheless tactically resolved -- a knight forking king and rook. It needs a
    quiescence search per position, which at 394 million positions is a different program.
    """
    if len(move) < 4:
        return False
    if len(move) > 4:  # a promotion carries a fifth character
        return False

    origin = (ord(move[0]) - 97) + (ord(move[1]) - 49) * 8
    target = (ord(move[2]) - 97) + (ord(move[3]) - 49) * 8
    if not 0 <= origin < 64 or not 0 <= target < 64:
        return False

    mailbox = bytearray(64)
    remaining = occupancy
    king_square = -1
    king_code = 11 if black_to_move else 5
    for code in codes:
        square = (remaining & -remaining).bit_length() - 1
        remaining &= remaining - 1
        mailbox[square] = code + 1
        if code == king_code:
            king_square = square

    # A piece on the target square is a capture only if it is an *enemy* piece. The database writes
    # castling in UCI's king-takes-rook form -- e1h1, e8a8 -- so a friendly piece on the target is a
    # castle, which is quiet. Reading the occupancy bit alone would have thrown every castling
    # position away, and those are exactly the positions the king-safety weights need.
    occupant = mailbox[target]
    if occupant:
        enemy_low, enemy_high = (1, 6) if black_to_move else (7, 12)
        if enemy_low <= occupant <= enemy_high:
            return False

    # A pawn changing file with nothing on the target square is an en-passant capture.
    if mailbox[origin] in (1, 7) and (origin & 7) != (target & 7):
        return False
    if king_square < 0:
        return False
    return not in_check(mailbox, king_square, black_to_move)


def process_chunk(job: tuple[int, list[bytes], bool]) -> tuple[int, list[bytes]]:
    """Parse a batch of raw JSON lines into per-shard blobs. Runs in a worker process.

    The shard is drawn here rather than in the parent so the randomness costs nothing serial. It is
    seeded from the chunk index, so a re-run with the same input reproduces the same split.
    """
    index, lines, quiet_only = job
    rng = random.Random(index)
    buckets: list[bytearray] = [bytearray() for _ in range(SHARDS)]
    kept = 0
    for line in lines:
        record = orjson.loads(line)
        best = best_score(record["evals"])
        if best is None:
            continue
        score, move = best
        fen = record["fen"]
        placement, _, rest = fen.partition(" ")
        occupancy, codes = parse_board(placement)
        if not is_plausible(occupancy, codes):
            continue
        black_to_move = rest[0] == "b"
        if quiet_only and not is_quiet(occupancy, codes, black_to_move, move):
            continue
        buckets[rng.randrange(SHARDS)] += encode(occupancy, codes, score, 1 if black_to_move else 0)
        kept += 1
    return kept, [bytes(bucket) for bucket in buckets]


def read_chunks(
    source: Path, limit: int, quiet_only: bool
) -> Iterator[tuple[int, list[bytes], bool]]:
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
                yield index, pending[:CHUNK_LINES], quiet_only
                pending = pending[CHUNK_LINES:]
                index += 1
            if limit and read >= limit:
                break
    if pending:
        yield index, pending, quiet_only


def run(source: Path, destination: Path, limit: int, workers: int, quiet_only: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"quiet-position filter: {'on' if quiet_only else 'off'}", flush=True)
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
            jobs = read_chunks(source, limit, quiet_only)
            for kept, blobs in pool.imap(process_chunk, jobs, chunksize=1):
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
    parser.add_argument(
        "--no-quiet-filter",
        action="store_true",
        help="keep tactical positions, for an A/B against a filtered corpus",
    )
    arguments = parser.parse_args()
    run(
        arguments.source,
        arguments.destination,
        arguments.limit,
        arguments.workers,
        not arguments.no_quiet_filter,
    )


if __name__ == "__main__":
    main()

"""Gate the quiet-position filter against python-chess on real database records.

The filter decides what the net is allowed to learn from, and every way it can be wrong is silent.
Reject too much and the corpus quietly loses coverage; reject too little and tactical noise stays
in the labels. Neither shows up as an error -- only as a net that is worse than it should be, weeks
after the fact, with nothing pointing back here.

So the check-detection and capture-detection written in `preprocess.py` are compared against
python-chess, which is the one authority available that was not written by us, on positions taken
from the actual Lichess database rather than from anything synthetic. `board.is_check()` and
`board.is_capture()` are the ground truth; any disagreement at all is a failure.

Run:  python -m tools.verify_quiet_filter --positions 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import orjson
import zstandard

from training.preprocess import best_score, in_check, is_plausible, is_quiet, parse_board

DEFAULT_SOURCE = Path(r"C:/Users/ssjag/chessdata/lichess_db_eval.jsonl.zst")


def records(source: Path, wanted: int) -> list[dict[str, object]]:
    """The first `wanted` parseable records, straight out of the compressed stream."""
    found: list[dict[str, object]] = []
    tail = b""
    with open(source, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        while len(found) < wanted:
            block = reader.read(1 << 22)
            if not block:
                break
            lines = (tail + block).split(b"\n")
            tail = lines.pop()
            for line in lines:
                if line:
                    found.append(orjson.loads(line))
                    if len(found) >= wanted:
                        break
    return found


def run(source: Path, wanted: int) -> int:
    checked = check_mismatch = quiet_mismatch = 0
    kept = in_check_count = capture_count = promotion_count = 0
    failures: list[str] = []

    for record in records(source, wanted):
        best = best_score(record["evals"])  # type: ignore[arg-type]
        if best is None:
            continue
        _, move = best
        fen = str(record["fen"])
        placement, _, rest = fen.partition(" ")
        occupancy, codes = parse_board(placement)
        if not is_plausible(occupancy, codes):
            continue

        # python-chess needs a full FEN. The database gives one, but its move-counter fields are
        # not always present, so rebuild from the parts we rely on.
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if not board.is_valid():
            continue

        black_to_move = rest[0] == "b"
        checked += 1

        # 1. Check detection, against board.is_check().
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
        ours = in_check(mailbox, king_square, black_to_move)
        theirs = board.is_check()
        if ours != theirs:
            check_mismatch += 1
            if len(failures) < 10:
                failures.append(f"CHECK  ours={ours} python-chess={theirs}  {fen}")
        if theirs:
            in_check_count += 1

        # 2. The whole filter, against the same rule expressed through python-chess.
        try:
            parsed = chess.Move.from_uci(move)
        except ValueError:
            continue
        if parsed not in board.legal_moves:
            continue
        tactical = board.is_capture(parsed) or parsed.promotion is not None
        expected = not tactical and not theirs
        actual = is_quiet(occupancy, codes, black_to_move, move)
        if actual != expected:
            quiet_mismatch += 1
            if len(failures) < 10:
                failures.append(f"QUIET  ours={actual} expected={expected} move={move}  {fen}")
        if board.is_capture(parsed):
            capture_count += 1
        if parsed.promotion is not None:
            promotion_count += 1
        if actual:
            kept += 1

    def tally(label: str, count: int) -> None:
        print(f"  {label:<24}  {count:>8,}  {100 * count / max(checked, 1):5.1f}%")

    print(f"{checked:,} positions cross-checked against python-chess\n")
    tally("in check", in_check_count)
    tally("best move is a capture", capture_count)
    tally("best move is a promotion", promotion_count)
    tally("kept as quiet", kept)
    print()
    print(f"  check-detection mismatches {check_mismatch}")
    print(f"  whole-filter mismatches    {quiet_mismatch}")

    for line in failures:
        print(f"    {line}")

    if check_mismatch or quiet_mismatch:
        print("\nFAIL -- the filter disagrees with python-chess")
        return 1
    print("\nexact agreement")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate the quiet-position filter.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--positions", type=int, default=20_000)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.source, arguments.positions))


if __name__ == "__main__":
    main()

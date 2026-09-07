"""Read the rated-game logs and answer three questions the engine cannot answer about itself.

Every other instrument here asks "did that change help?" and answers it in Elo with error bars.
This one asks what the ladder is actually doing to us, using only facts that hold regardless of
what our evaluation thinks: who was ahead on pieces, for how long, and what the clocks did.

The piece count is the point. `blunder_scan.py` scores our moves with our own search, so it cannot
tell us that our evaluation is wrong -- an instrument may not be used to validate its own output.
Material is counted by the rules of chess, so a conversion failure shows up here even if the
evaluation is confidently mistaken about it.

Usage:  python -m tools.ladder_scan [--logs DIR] [--since ROUND]
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import Counter

import chess
import chess.pgn

# Pure material, matching `harness/referee.py`'s ply-300 adjudication: no positional term.
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

# A lead has to survive this many plies to count as real. Ten full moves is long enough that a
# recapture inside an exchange sequence cannot manufacture it, and short enough that a genuine
# extra piece in a 150-ply game always clears it.
HOLD_PLIES = 20

OUR_NAMES = ("xx", "?")


def _round_of(path: str) -> int:
    m = re.search(r"round-(\d+)", path)
    if m is None:
        raise ValueError(f"no round number in {path}")
    return int(m.group(1))


def _material(board: chess.Board, colour: chess.Color) -> int:
    return sum(v * len(board.pieces(pt, colour)) for pt, v in VALUES.items())


def _clock_seconds(comment: str) -> float | None:
    m = re.search(r"\[%clk (\d+):(\d+):([\d.]+)\]", comment or "")
    if not m:
        return None
    return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3))


class Game:
    """One rated game, reduced to the facts that do not depend on our evaluation."""

    def __init__(self, path: str) -> None:
        self.round = _round_of(path)
        with open(path, encoding="utf-8", errors="replace") as fh:
            game = chess.pgn.read_game(fh)
        if game is None:
            raise ValueError(f"no game in {path}")
        h = game.headers
        white, black = h.get("White", "?"), h.get("Black", "?")
        self.we_are_white = white in OUR_NAMES and black not in OUR_NAMES
        self.opponent = black if self.we_are_white else white
        self.termination = h.get("Termination", "?")
        table = (
            {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
            if self.we_are_white
            else {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}
        )
        self.score = table.get(h.get("Result", "?"), -1.0)

        us = chess.WHITE if self.we_are_white else chess.BLACK
        board = game.board()
        first_mover_is_white = board.turn == chess.WHITE
        self.material: list[int] = []
        clocks: list[float | None] = []
        for node in game.mainline():
            move = node.move
            if move is None:
                break
            board.push(move)
            self.material.append(_material(board, us) - _material(board, not us))
            clocks.append(_clock_seconds(node.comment))

        # A single unparsed comment makes the whole side's clock untrustworthy, so drop it rather
        # than silently averaging over a hole.
        def side(mine: bool) -> list[float]:
            got = [
                c
                for i, c in enumerate(clocks)
                if (((i % 2 == 0) == first_mover_is_white) == self.we_are_white) is mine
            ]
            return [] if any(c is None for c in got) else [c for c in got if c is not None]

        self.our_clock = side(True)
        self.their_clock = side(False)

    @property
    def plies(self) -> int:
        return len(self.material)

    @property
    def peak(self) -> int:
        """Best material we ever reached, including transient exchange spikes."""
        return max(self.material, default=0)

    @property
    def sustained(self) -> int:
        """Best material lead we actually held for HOLD_PLIES consecutive plies."""
        if self.plies < HOLD_PLIES:
            return 0
        return max(
            min(self.material[i : i + HOLD_PLIES]) for i in range(self.plies - HOLD_PLIES + 1)
        )


def _band(sustained: int) -> str:
    if sustained >= 5:
        return "ahead a piece or more"
    if sustained >= 3:
        return "ahead a minor"
    if sustained > -3:
        return "level"
    return "behind"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", default="logs", help="directory holding the rated .pgn files")
    ap.add_argument("--since", type=int, default=0, help="only print rounds at or after this")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.logs}/*.pgn"), key=_round_of)
    games = [Game(p) for p in paths]
    games = [g for g in games if g.score >= 0.0 and g.plies >= HOLD_PLIES]
    if not games:
        print(f"no rated games found under {args.logs}/")
        return

    print(
        f"{'rd':>4} {'sc':>4} {'peak':>5} {'held':>5} {'final':>6} {'plies':>5} "
        f"{'ourclk':>7} {'oppclk':>7}  termination / opponent"
    )
    for g in games:
        if g.round < args.since:
            continue
        sc = {1.0: "win", 0.5: "draw", 0.0: "loss"}[g.score]
        oc = f"{g.our_clock[-1]:.1f}" if g.our_clock else "-"
        tc = f"{g.their_clock[-1]:.1f}" if g.their_clock else "-"
        print(
            f"{g.round:>4} {sc:>4} {g.peak:>+5d} {g.sustained:>+5d} {g.material[-1]:>+6d} "
            f"{g.plies:>5} {oc:>7} {tc:>7}  {g.termination} / {g.opponent}"
        )

    print("\nresult by the material lead we actually held:")
    bands: dict[str, list[float]] = {}
    for g in games:
        bands.setdefault(_band(g.sustained), []).append(g.score)
    for name in ("ahead a piece or more", "ahead a minor", "level", "behind"):
        scores = bands.get(name, [])
        if not scores:
            continue
        c = Counter(scores)
        print(
            f"  {name:<22} n={len(scores):>3}  "
            f"{int(c[1.0])}W {int(c[0.5])}D {int(c[0.0])}L  "
            f"score={sum(scores) / len(scores):.0%}"
        )

    print("\nscore over time:")
    for i in range(0, len(games), 10):
        blk = games[i : i + 10]
        if len(blk) < 4:
            continue
        pts = sum(g.score for g in blk)
        draws = sum(1 for g in blk if g.score == 0.5)
        print(
            f"  rounds {blk[0].round:>3}-{blk[-1].round:>3}  "
            f"{pts:>4.1f}/{len(blk)} = {pts / len(blk):>4.0%}   draws {draws}/{len(blk)}"
        )

    timed = [g for g in games if g.our_clock and g.their_clock]
    if timed:
        n = len(timed)
        ours = sum(g.our_clock[-1] for g in timed) / n
        theirs = sum(g.their_clock[-1] for g in timed) / n
        slower = sum(1 for g in timed if g.our_clock[-1] < g.their_clock[-1])
        print(
            f"\nclocks over {n} games: we finish with {ours:.1f}s spare, "
            f"the field with {theirs:.1f}s"
        )
        print(f"  we are the one closer to the flag in {slower}/{n}")


if __name__ == "__main__":
    main()

"""Find where our games actually went wrong, by scoring every position we played.

A result column says we lost; it does not say whether we were outplayed slowly or threw the game
away in one move, and those two point at completely different work. A sharp drop across a single
move we made is a search failure -- the move was tactically refuted and we did not see it. A slow
slide with no single large drop is an evaluation failure, and no amount of extra depth fixes it.

Scores are from our own engine, so this measures our judgement against our own deeper judgement,
not against the truth. That is the right comparison for finding search failures (we had the
position and missed it) and the wrong one for finding evaluation failures, which by construction we
cannot see. Read the drops, treat the levels with suspicion.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.search import MATE, Searcher  # noqa: E402

NO_CLOCK = 1 << 40
LOGS = Path("C:/Users/ssjag/OneDrive/Programming/ai-chessathon/logs")
# Scores past this are mate announcements; differencing them produces meaningless cliffs.
MATE_ZONE = MATE - 1000
# Only count a drop out of a position that was still worth playing. The first version of this
# reported the largest drop anywhere, which put every loss's "worst move" deep inside an already
# decided position -- going from -16 pawns to -20 is volatility, not the move that lost the game.
STILL_COMPETITIVE = 300


def our_colour(rd: int) -> chess.Color | None:
    for f in LOGS.glob(f"aichessathon-round-{rd}-*.log"):
        m = re.search(r"Colour\s+(\w+)", f.read_text(encoding="utf-8", errors="replace"))
        if m:
            return chess.WHITE if m.group(1).strip() == "White" else chess.BLACK
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rounds", type=int, nargs="+")
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()

    searcher = Searcher(tt_bits=20)
    print(f"{'rd':>3} {'result':<9}{'worst drop':>11}{'after our move':>16}{'ply':>5}   shape")
    for rd in args.rounds:
        pgns = list(LOGS.glob(f"aichessathon-round-{rd}-*.pgn"))
        colour = our_colour(rd)
        if not pgns or colour is None:
            print(f"{rd:>3} (missing)")
            continue
        with pgns[0].open(encoding="utf-8", errors="replace") as fh:
            game = chess.pgn.read_game(fh)
        if game is None:
            print(f"{rd:>3} (unreadable)")
            continue

        board = game.board()
        searcher.new_game()
        # (ply, score-from-our-view, san, whether *we* played the move)
        scores: list[tuple[int, int, str, bool]] = []
        for ply, move in enumerate(game.mainline_moves()):
            san = board.san(move)
            # Rated games start from a curated FEN, so black moves first in some of them and ply
            # parity says nothing about who is moving. Ask the board before the push, never the
            # index -- inferring it from parity reports the opponent's moves as ours.
            mover_is_us = board.turn == colour
            board.push(move)
            if board.is_game_over():
                break
            searcher.set_position(board.fen())
            _m, sc, _d = searcher.search(NO_CLOCK, max_depth=args.depth)
            ours = int(sc) if board.turn == colour else -int(sc)
            scores.append((ply, ours, san, mover_is_us))

        # A drop measured across a move *we* made: compare the position after our move with the
        # position after the opponent's previous one.
        worst, worst_ply, worst_san = 0, -1, ""
        for i in range(1, len(scores)):
            prev, cur = scores[i - 1], scores[i]
            if abs(prev[1]) > MATE_ZONE or abs(cur[1]) > MATE_ZONE:
                continue
            if not cur[3] or prev[1] < -STILL_COMPETITIVE:
                continue
            drop = prev[1] - cur[1]
            if drop > worst:
                worst, worst_ply, worst_san = drop, cur[0], cur[2]
        levels = [s for _, s, _, _ in scores if abs(s) < MATE_ZONE]
        shape = "cliff" if worst >= 250 else ("slide" if levels and min(levels) < -200 else "flat")
        print(
            f"{rd:>3} {game.headers.get('Result','?'):<9}{worst:>11}{worst_san:>16}"
            f"{worst_ply:>5}   {shape}"
        )


if __name__ == "__main__":
    main()

"""Play two nets against each other at a *fixed depth*, so the result is eval quality alone.

`tools/sprt.py` answers the shipping question -- is this build better on the clock -- and that is
the question that matters. But it cannot separate the two things a wider net changes at once: the
evaluation gets better and every node gets slower. A width that is worth +40 Elo of judgement and
costs 2x the nodes loses on the clock while still being the better evaluator, and the two effects
have to be measured apart before the trade can be priced.

Fixed depth removes the speed term entirely: both sides search the same tree shape budget, so the
only difference left is what the evaluation says. Combined with the measured node-cost ratio and
`notes/measurements.md`'s 52.5 Elo per halving of thinking time, that gives the whole trade.

Fixed-depth play is deterministic, so one opening yields exactly one game and the 22 curated lines
would cap us at 44. Starts are therefore diversified by playing random legal moves out of each
line, keeping only positions the *baseline* net scores as near level. Filtering with the baseline
is deliberate: both engines play both colours from every start, so a start the baseline misjudges
costs each side equally.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.nnue import Network, new_stack  # noqa: E402
from engine.position import MAX_PLY, move_to_uci  # noqa: E402
from engine.search import Searcher  # noqa: E402
from tools.sprt import opening_fens  # noqa: E402

# The competition adjudicates a draw at 600 plies; matching it keeps these games comparable with
# the ones the ladder actually plays.
PLY_CAP = 600
NO_CLOCK = 1 << 40


# A fixed-depth-8 search touches a tiny fraction of the shipped 22-bit table, and every worker
# process holds two searchers with a table each -- at 22 bits that is ~80 MB apiece and enough
# parallel matches exhaust the box. Both sides get the same size, so the comparison stays fair.
TT_BITS = 18


def _searcher(net: Path) -> Searcher:
    s = Searcher(tt_bits=TT_BITS)
    s.network = Network(net)
    s.use_nnue = s.network.available
    s.acc = new_stack(s.network.hidden, MAX_PLY + 1)
    return s


def _move(s: Searcher, board: chess.Board, depth: int) -> chess.Move | None:
    """One fixed-depth move, validated against python-chess exactly as `agent.py` does."""
    s.set_position(board.fen())
    s.record_position()
    packed, _score, _depth = s.search(NO_CLOCK, max_depth=depth)
    move = chess.Move.from_uci(move_to_uci(np.int32(packed)))
    return move if move in board.legal_moves else None


def balanced_starts(baseline: Path, wanted: int, depth: int, seed: int) -> list[str]:
    """Random extensions of the curated lines that the baseline net calls roughly level."""
    rng = random.Random(seed)
    judge = _searcher(baseline)
    seen: set[str] = set()
    starts: list[str] = []
    lines = opening_fens()
    attempts = 0
    while len(starts) < wanted and attempts < wanted * 20:
        attempts += 1
        board = chess.Board(rng.choice(lines))
        for _ in range(rng.choice((2, 4))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or board.epd() in seen:
            continue
        judge.new_game()
        judge.set_position(board.fen())
        _m, score, _d = judge.search(NO_CLOCK, max_depth=depth)
        if abs(score) <= 60:
            seen.add(board.epd())
            starts.append(board.fen())
    return starts


def play(job: tuple[str, bool, Path, Path, int]) -> float:
    """Score for the candidate: 1.0 win, 0.5 draw, 0.0 loss."""
    fen, candidate_is_white, cand_net, base_net, depth = job
    cand, base = _searcher(cand_net), _searcher(base_net)
    cand.new_game()
    base.new_game()
    board = chess.Board(fen)
    while not board.is_game_over(claim_draw=True) and board.ply() < PLY_CAP:
        mover = cand if (board.turn == chess.WHITE) == candidate_is_white else base
        move = _move(mover, board, depth)
        if move is None:
            # An illegal proposal is a loss for whoever made it, the same way the referee treats it.
            return 0.0 if mover is cand else 1.0
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if candidate_is_white else 0.0
    if result == "0-1":
        return 0.0 if candidate_is_white else 1.0
    return 0.5


def elo(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Elo and one standard error, from the trinomial score. Returns (0, inf) if degenerate."""
    n = wins + draws + losses
    if n == 0:
        return 0.0, float("inf")
    score = (wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        return 0.0, float("inf")
    mean = score
    var = (wins * (1 - mean) ** 2 + draws * (0.5 - mean) ** 2 + losses * mean**2) / n
    se = math.sqrt(var / n)
    point = -400.0 * math.log10(1.0 / score - 1.0)
    lo = max(1e-9, min(1 - 1e-9, score - se))
    hi = max(1e-9, min(1 - 1e-9, score + se))
    span = (-400.0 * math.log10(1.0 / hi - 1.0)) - (-400.0 * math.log10(1.0 / lo - 1.0))
    return point, span / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="candidate .npz")
    parser.add_argument("--baseline", type=Path, required=True, help="baseline .npz")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--starts", type=int, default=150)
    parser.add_argument("--filter-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    print(f"generating up to {args.starts} balanced starts...", flush=True)
    starts = balanced_starts(args.baseline, args.starts, args.filter_depth, args.seed)
    jobs = [
        (fen, white, args.candidate, args.baseline, args.depth)
        for fen in starts
        for white in (True, False)
    ]
    print(
        f"{args.candidate.name} vs {args.baseline.name} at fixed depth {args.depth}\n"
        f"{len(starts)} starts, {len(jobs)} games, {args.workers} workers",
        flush=True,
    )

    wins = draws = losses = 0
    with Pool(args.workers) as pool:
        for done, score in enumerate(pool.imap_unordered(play, jobs), start=1):
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1
            if done % 20 == 0 or done == len(jobs):
                point, err = elo(wins, draws, losses)
                print(
                    f"  {done:>4}/{len(jobs)}  +{wins} ={draws} -{losses}"
                    f"  Elo {point:+.1f} +- {err:.1f}",
                    flush=True,
                )

    point, err = elo(wins, draws, losses)
    print(f"\n  +{wins} ={draws} -{losses}   Elo {point:+.1f} +- {err:.1f}")
    print("  record this in notes/measurements.md")


if __name__ == "__main__":
    main()

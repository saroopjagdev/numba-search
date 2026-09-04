"""Parallel arena with a sequential probability ratio test. The real measurement instrument.

`harness/arena.py` plays 20 sequential games from the standard start position. That is the wrong
positions, too few of them, and too slow: 20 games cannot distinguish a +30 Elo change from noise,
and every game from `startpos` re-tests the same opening. This tool fixes all three.

- **Curated openings.** Each line is played twice with colours reversed, so a lucky opening helps
  both sides equally. `harness/referee.py:play_match` accepts `start_fen`; the bundled arena never
  exposes it.
- **Parallel.** Concurrency defaults to half the logical cores, because every game runs *two*
  agent processes. See the warning in `_resolve_concurrency` — oversubscribing distorts results
  under a real clock, and silently.
- **SPRT.** Stops as soon as the evidence is decisive instead of at an arbitrary game count, and
  reports an Elo estimate with error bars rather than a bare score.

    uv run python tools/sprt.py --candidate . --baseline baselines/minimax
    uv run python tools/sprt.py --candidate . --baseline ../previous --elo0 0 --elo1 10

Exit status is 0 if the candidate is accepted (H1), 1 if rejected (H0), 2 if inconclusive.
"""

import argparse
import itertools
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import chess

from harness import sandbox
from harness.referee import play_match

# Balanced, well-trodden openings. Played out ~8 plies so the engines start in real positions but
# well short of anything decided. Both sides get each line once.
OPENING_LINES: tuple[str, ...] = (
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6",  # Ruy Lopez, closed
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6",  # Italian
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6",  # Scotch
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",  # Sicilian, open
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6",  # Sicilian, accelerated dragon
    "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7",  # Sicilian, closed
    "e4 e6 d4 d5 Nc3 Bb4 e5 c5",  # French, Winawer
    "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7",  # French, Tarrasch
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",  # Caro-Kann, classical
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6",  # Scandinavian
    "e4 Nf6 e5 Nd5 d4 d6 Nf3 dxe5",  # Alekhine
    "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7",  # Pirc
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7",  # QGD
    "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6",  # QGA
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4",  # Slav
    "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O",  # Nimzo-Indian
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",  # King's Indian
    "d4 Nf6 c4 e6 Nf3 b6 g3 Ba6",  # Queen's Indian
    "d4 Nf6 c4 c5 d5 b5 cxb5 a6",  # Benko
    "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7",  # Dutch, Leningrad
    "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5",  # English, reversed Sicilian
    "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7",  # Reti
)


def opening_fens() -> list[str]:
    """Turn the SAN lines into start FENs, failing loudly on a typo rather than skipping it."""
    fens = []
    for line in OPENING_LINES:
        board = chess.Board()
        for san in line.split():
            try:
                board.push_san(san)
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
                ply = len(board.move_stack)
                sys.exit(f"bad opening {line!r}: illegal move {san!r} at ply {ply}")
        fens.append(board.fen())
    return fens


# --- statistics -----------------------------------------------------------------------


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def _moments(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    """Score, per-game variance and effective game count, with a one-sided sample regularised.

    A run against a much weaker opponent produces zero losses, and the raw variance estimate then
    collapses to something degenerate: the LLR pins at zero and the test never terminates. Adding
    half a game to each bucket keeps the estimate finite and conservative — it pulls the score
    toward a draw, so it can only delay a verdict, never manufacture one.
    """
    total = wins + draws + losses
    if total == 0:
        return 0.5, 0.0, 0.0
    if wins == 0 or losses == 0:
        w, d, level = wins + 0.5, draws + 0.5, losses + 0.5
    else:
        w, d, level = float(wins), float(draws), float(losses)
    effective = w + d + level
    w, d = w / effective, d / effective
    score = w + d / 2.0
    variance = max((w + d / 4.0) - score * score, 1e-12)
    return score, variance, effective


def log_likelihood_ratio(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Generalized LLR under the trinomial model, as used by Fishtest."""
    score, variance, effective = _moments(wins, draws, losses)
    if effective == 0.0:
        return 0.0
    variance_per_game = variance / effective
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * variance_per_game)


def elo_with_error_bars(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Elo estimate and the half-width of its 95% confidence interval."""
    score, variance, effective = _moments(wins, draws, losses)
    if effective == 0.0:
        return 0.0, float("inf")
    deviation = math.sqrt(variance / effective)
    low = score_to_elo(min(max(score - 1.96 * deviation, 1e-9), 1 - 1e-9))
    high = score_to_elo(min(max(score + 1.96 * deviation, 1e-9), 1 - 1e-9))
    return score_to_elo(score), (high - low) / 2.0


# --- the arena ------------------------------------------------------------------------


@dataclass
class Tally:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    def record(self, result: str, termination: str, candidate_is_white: bool) -> None:
        with self.lock:
            if result == "draw":
                self.draws += 1
            elif (result == "white") == candidate_is_white:
                self.wins += 1
            elif result != "void":
                self.losses += 1
            if termination in {"crash", "illegal", "flag", "init", "both_failed"}:
                self.failures[termination] = self.failures.get(termination, 0) + 1


def play_one(
    candidate: Path,
    baseline: Path,
    start_fen: str,
    candidate_is_white: bool,
    base_ms: int,
    increment_ms: int,
) -> tuple[str, str]:
    white = candidate if candidate_is_white else baseline
    black = baseline if candidate_is_white else candidate
    outcome = play_match(
        sandbox.local(white), sandbox.local(black), base_ms, increment_ms, start_fen=start_fen
    )
    return outcome.result, outcome.termination


def _resolve_concurrency(requested: int | None) -> int:
    cores = os.cpu_count() or 4
    safe = max(1, cores // 2)
    if requested is None:
        return safe
    if requested > safe:
        print(
            f"warning: concurrency {requested} exceeds {safe} on {cores} logical cores. Every game "
            f"runs two agent processes, so this oversubscribes the CPU and distorts any result "
            f"measured against a real clock — silently, and in the direction of whichever engine "
            f"uses less time.",
            file=sys.stderr,
        )
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--elo0", type=float, default=0.0, help="H0: the candidate is no better")
    parser.add_argument("--elo1", type=float, default=10.0, help="H1: the candidate is better")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument(
        "--base-ms", type=int, default=10_000, help="short clocks buy games; 120000 is real"
    )
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    concurrency = _resolve_concurrency(args.concurrency)
    lower = math.log(args.beta / (1.0 - args.alpha))
    upper = math.log((1.0 - args.beta) / args.alpha)

    fens = opening_fens()
    rng = random.Random(args.seed)
    rng.shuffle(fens)
    # each opening twice, colours reversed, cycling if max-games exceeds the book
    pairings = [(fen, colour) for fen in fens for colour in (True, False)]
    schedule = list(itertools.islice(itertools.cycle(pairings), args.max_games))

    print(f"SPRT  H0: {args.elo0:+.0f} Elo   H1: {args.elo1:+.0f} Elo")
    print(f"      bounds [{lower:.2f}, {upper:.2f}]   {args.candidate} vs {args.baseline}")
    print(
        f"      {args.base_ms / 1000:.0f}s + {args.increment_ms / 1000:.1f}s, "
        f"{len(fens)} openings, concurrency {concurrency}, max {args.max_games} games\n"
    )

    tally = Tally()
    started = time.perf_counter()
    verdict = "inconclusive"

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                play_one,
                args.candidate,
                args.baseline,
                fen,
                is_white,
                args.base_ms,
                args.increment_ms,
            ): is_white
            for fen, is_white in schedule
        }
        for future in as_completed(futures):
            candidate_is_white = futures[future]
            try:
                result, termination = future.result()
            except Exception as error:
                print(f"  game failed: {error}", file=sys.stderr)
                continue
            tally.record(result, termination, candidate_is_white)

            llr = log_likelihood_ratio(tally.wins, tally.draws, tally.losses, args.elo0, args.elo1)
            elo, margin = elo_with_error_bars(tally.wins, tally.draws, tally.losses)
            rate = tally.total / max(time.perf_counter() - started, 1e-9)
            print(
                f"  {tally.total:4d}  +{tally.wins} ={tally.draws} -{tally.losses}   "
                f"LLR {llr:+6.2f}   Elo {elo:+7.1f} ± {margin:5.1f}   {rate * 60:5.1f} games/min",
                flush=True,
            )

            if llr >= upper:
                verdict = "accepted"
                break
            if llr <= lower:
                verdict = "rejected"
                break
        for future in futures:
            future.cancel()

    elo, margin = elo_with_error_bars(tally.wins, tally.draws, tally.losses)
    elapsed = time.perf_counter() - started
    print(f"\n  {verdict.upper()}  after {tally.total} games in {elapsed / 60:.1f} min")
    print(f"  +{tally.wins} ={tally.draws} -{tally.losses}   Elo {elo:+.1f} ± {margin:.1f}")
    if tally.failures:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(tally.failures.items()))
        print(f"  FAILURES: {detail}  <- investigate before trusting this result")
    print("\n  record this in notes/measurements.md")

    sys.exit({"accepted": 0, "rejected": 1, "inconclusive": 2}[verdict])


if __name__ == "__main__":
    main()

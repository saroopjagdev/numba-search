"""Assert the engine against positions whose correct evaluation is already known.

Every other instrument here answers "did that change help?" and answers it statistically, at the
+-18 Elo the SPRT can resolve. This one answers a different question -- "is the engine *wrong*
about something whose answer is fixed by the rules or by endgame theory" -- and answers it exactly.
A lone bishop cannot mate; that is not a matter of degree, and no number of games is needed to
check it.

The distinction matters because it is how the insufficient-material bug survived. It could not be
found by measuring changes, only by auditing correctness, and nothing here was auditing
correctness. Cases are cheap to add, so add one whenever a game shows the engine believing
something that is false by rule.

Only cases that are certain **by rule** are asserted. Endgame theory is deliberately excluded: a
first draft of this file asserted three "known" drawn endings and one of them was simply wrong (a
g-pawn written as a rook pawn, with a bishop that did control the queening square), which would
have reported an engine failure that was really an author failure. Positions whose answer needs a
tablebase belong in a suite that has one.

Exit code is non-zero on any failure so this can gate a build.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.search import MATE, Searcher  # noqa: E402

# No clock: these are correctness questions, so the search gets as long as the depth needs.
NO_CLOCK = 1 << 40

# kind: draw (must score ~0), win (must score clearly ahead), mate (must find the mate).
CASES: tuple[tuple[str, str, str, str], ...] = (
    # Material that cannot force mate. Drawn by rule or by theory, whatever the piece count says.
    ("K vs K", "8/8/4k3/8/8/8/4K3/8 w - - 0 1", "draw", "bare kings"),
    ("K+N vs K", "8/8/4k3/8/8/4N3/4K3/8 w - - 0 1", "draw", "no mate exists"),
    ("K+B vs K", "8/8/4k3/8/8/4B3/4K3/8 w - - 0 1", "draw", "no mate exists"),
    ("K+NN vs K", "8/8/4k3/8/8/3NN3/4K3/8 w - - 0 1", "draw", "cannot be forced"),
    ("K+B vs K+B same colour", "8/8/3bk3/8/8/4B3/4K3/8 w - - 0 1", "draw", "same-colour bishops"),
    ("K+N vs K+N", "8/8/3nk3/8/8/4N3/4K3/8 w - - 0 1", "draw", "cannot be forced"),
    # Material that can mate. The drawn-material rule must not swallow these.
    ("K+R vs K", "8/8/4k3/8/8/4R3/4K3/8 w - - 0 1", "win", ""),
    ("K+Q vs K", "8/8/4k3/8/8/4Q3/4K3/8 w - - 0 1", "win", ""),
    ("K+BB vs K", "8/8/4k3/8/8/3BB3/4K3/8 w - - 0 1", "win", ""),
    ("K+BN vs K", "8/8/4k3/8/8/3BN3/4K3/8 w - - 0 1", "win", "hard, but won"),
    # Terminal scoring.
    ("stalemate, side to move", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "draw", "stalemated"),
    ("mate in 1", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "mate", ""),
    # No case for "root is already mated": the referee ends the game, so we are never asked to
    # move there, and asserting on it would report a failure that cannot occur in play.
)

DRAW_WINDOW = 50
WIN_FLOOR = 400


def correct(kind: str, score: int) -> bool:
    if kind == "draw":
        return abs(score) <= DRAW_WINDOW
    if kind == "win":
        return score >= WIN_FLOOR
    return score >= MATE - 20


def main() -> int:
    searcher = Searcher(tt_bits=20)
    print(f"{'position':<28}{'expect':<7}{'eval':>9}  verdict")
    failures = []
    for label, fen, kind, note in CASES:
        searcher.new_game()
        searcher.set_position(fen)
        _move, score, _depth = searcher.search(NO_CLOCK, max_depth=12)
        ok = correct(kind, int(score))
        if not ok:
            failures.append((label, int(score), note))
        suffix = f"  ({note})" if note and not ok else ""
        print(f"{label:<28}{kind:<7}{int(score):>9}  {'ok' if ok else '<<< WRONG'}{suffix}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} correct, {len(failures)} wrong")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarise the platform match logs: result, init cost, and how much of the clock went unused.

The dashboard hands back one log per rated game and they accumulate faster than they can be read
one at a time. What matters across a set of them is not any single game but the trends: whether
init is creeping toward the 90 s budget, and what fraction of the clock the engine actually spends.

Both have already earned their keep. Init on the platform runs at about 0.58x the local figure,
which is the constant every local init measurement is converted through. And the clock column is
how the node-cap abort bug was confirmed fixed in production after its SPRT never returned a
verdict -- spend went from 32% to 53% of the clock between the build before and the build after.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import NamedTuple

DEFAULT_LOGS = Path(__file__).resolve().parent.parent / "logs"


class Match(NamedTuple):
    round: int
    opponent: str
    colour: str
    init: float
    moves: int
    used: float
    left: float
    slowest: float
    result: str

    @property
    def spent(self) -> float:
        """Percentage of the clock actually consumed. Zero when the agent never got to move."""
        total = self.used + self.left
        return 100.0 * self.used / total if total else 0.0


def field(text: str, label: str) -> str:
    """One `  Label   value` line from the log's key-value blocks."""
    match = re.search(rf"^\s*{label}\s+(.+?)\s*$", text, re.M)
    return match.group(1) if match else "-"


def seconds(text: str, label: str) -> float:
    value = field(text, label).split()
    return float(value[0]) if value and value[0][0].isdigit() else 0.0


def summarise(path: Path) -> Match:
    text = path.read_text(encoding="utf-8", errors="replace")
    number = re.search(r"Round\s+\w+\s+(\d+)", text)
    result = re.search(r"^RESULT\n\s*(.+?)\s*$", text, re.M)
    moves = field(text, "Moves")
    return Match(
        round=int(number.group(1)) if number else 0,
        opponent=field(text, "Opponent"),
        colour=field(text, "Colour"),
        init=seconds(text, "Ready in"),
        moves=int(moves) if moves.isdigit() else 0,
        used=seconds(text, "Time used"),
        left=seconds(text, "Left at end"),
        slowest=seconds(text, "Slowest"),
        result=result.group(1) if result else "?",
    )


def run(logs: Path) -> int:
    rows = sorted((summarise(path) for path in logs.glob("*.log")), key=lambda row: row.round)
    if not rows:
        print(f"no match logs in {logs}")
        return 1

    print(
        f"{'rd':>3} {'opponent':<18} {'col':<5} {'init':>5} {'mv':>3} "
        f"{'used':>6} {'left':>6} {'spent':>6} {'slow':>5}  result"
    )
    for row in rows:
        print(
            f"{row.round:>3} {row.opponent:<18} {row.colour:<5} {row.init:>5.1f} {row.moves:>3} "
            f"{row.used:>6.1f} {row.left:>6.1f} {row.spent:>5.0f}% {row.slowest:>5.1f}  "
            f"{row.result}"
        )

    wins = sum("Won" in row.result for row in rows)
    losses = sum("Lost" in row.result for row in rows)
    print(f"\n{wins} won, {losses} lost, {len(rows) - wins - losses} other, of {len(rows)}")

    # Init is the one number that can lose every game at once, so it gets the headline.
    worst = max(row.init for row in rows)
    print(f"init: worst {worst:.1f}s of a 90s budget ({worst / 90 * 100:.0f}%)")

    played = [row for row in rows if row.used + row.left > 0]
    if played:
        spent = [row.spent for row in played]
        print(f"clock spent: mean {sum(spent) / len(spent):.0f}%, best {max(spent):.0f}%")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise platform match logs.")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    raise SystemExit(run(parser.parse_args().logs))


if __name__ == "__main__":
    main()

"""Check the magic attack tables against a reference ray-walker, exhaustively where feasible.

`engine/bitboard.py` deliberately does not verify itself at import: the init budget is real and a
check that runs in every game forever is a check in the wrong place. It runs here instead, and
should be run after any change to the magics, the masks or the table layout.

    uv run python tools/verify_movegen.py

Sliding attacks are the one part of movegen where a wrong answer is both easy to produce and hard
to notice — a bad magic collides on some rare occupancy and the engine plays an illegal move once
every few thousand positions. So this checks *every* occupancy subset of *every* square, which is
exhaustive over the space the tables can be asked about, not a sample of it.
"""

import sys
import time
from collections.abc import Callable

import numpy as np

from engine import bitboard

U64 = np.uint64


def reference(square: int, occupancy: int, directions: tuple[tuple[int, int], ...]) -> int:
    attacks = 0
    file, rank = square & 7, square >> 3
    for file_step, rank_step in directions:
        f, r = file + file_step, rank + rank_step
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            attacks |= 1 << target
            if occupancy >> target & 1:
                break
            f += file_step
            r += rank_step
    return attacks


def subsets_of(mask: int) -> list[int]:
    out, subset = [], 0
    while True:
        out.append(subset)
        subset = (subset - mask) & mask
        if subset == 0:
            return out


def check_slider(
    name: str,
    directions: tuple[tuple[int, int], ...],
    masks: np.ndarray,
    lookup: Callable[[int, np.uint64], np.uint64],
) -> int:
    """Every occupancy subset of every square. Exhaustive, not sampled."""
    failures = 0
    tested = 0
    for square in range(64):
        mask = int(masks[square])
        for subset in subsets_of(mask):
            want = reference(square, subset, directions)
            got = int(lookup(square, U64(subset)))
            tested += 1
            if got != want:
                failures += 1
                if failures <= 5:
                    print(f"  {name} sq{square} occ 0x{subset:016X}:")
                    print(f"    got 0x{got:016X} want 0x{want:016X}")
    print(f"  {name:7} {tested:>9,} occupancies  {'FAIL' if failures else 'ok'}")
    return failures


def check_leapers() -> int:
    """Leaper tables are small enough to state the expected totals outright."""
    failures = 0
    knight_total = int(sum(int(bin(int(x)).count("1")) for x in bitboard.KNIGHT_ATTACKS))
    king_total = int(sum(int(bin(int(x)).count("1")) for x in bitboard.KING_ATTACKS))
    # Known constants: 336 knight moves and 420 king moves exist on an empty board.
    for name, actual, expected in (("knight", knight_total, 336), ("king", king_total, 420)):
        status = "ok" if actual == expected else "FAIL"
        print(f"  {name:7} {actual:>9,} attacked squares  {status}")
        failures += actual != expected

    # Pawn captures: every pawn on ranks 1-7 has 1 or 2 captures; none on the far rank.
    for colour in (0, 1):
        far_rank = range(56, 64) if colour == 0 else range(0, 8)
        for square in far_rank:
            if int(bitboard.PAWN_ATTACKS[colour, square]) != 0:
                print(f"  pawn colour {colour} sq{square}: attacks off the board edge")
                failures += 1
    print(f"  pawn    {'':>9}  {'FAIL' if failures else 'ok'}")
    return failures


def main() -> None:
    print("verifying attack tables against a reference ray-walker\n")
    started = time.perf_counter()
    failures = check_leapers()
    failures += check_slider("rook", bitboard.ROOK_DIRS, bitboard.ROOK_MASKS, bitboard.rook_attacks)
    failures += check_slider(
        "bishop", bitboard.BISHOP_DIRS, bitboard.BISHOP_MASKS, bitboard.bishop_attacks
    )

    # queen must be exactly the union, on a sample of dense occupancies
    rng = np.random.default_rng(7)
    for _ in range(2000):
        occupancy = U64(rng.integers(0, 1 << 64, dtype=np.uint64))
        square = int(rng.integers(0, 64))
        want = int(bitboard.rook_attacks(square, occupancy)) | int(
            bitboard.bishop_attacks(square, occupancy)
        )
        if int(bitboard.queen_attacks(square, occupancy)) != want:
            print(f"  queen sq{square} occ 0x{int(occupancy):016X}: not the union")
            failures += 1
    print(f"  queen   {2000:>9,} random occupancies  {'FAIL' if failures else 'ok'}")

    print(f"\n  {time.perf_counter() - started:.1f}s")
    if failures:
        sys.exit(f"\n{failures} mismatch(es) — the attack tables are wrong, fix before movegen")
    print("\nattack tables correct")


if __name__ == "__main__":
    main()

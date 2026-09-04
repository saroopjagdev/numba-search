"""Check the magic attack tables against a reference ray-walker, exhaustively where feasible.

`engine/bitboard.py` deliberately does not verify itself at import: the init budget is real and a
check that runs in every game forever is a check in the wrong place. It runs here instead, and
should be run after any change to the magics, the masks or the table layout.

    uv run python tools/verify_movegen.py

Sliding attacks are the one part of movegen where a wrong answer is both easy to produce and hard
to notice — a bad magic collides on some rare occupancy and the engine plays an illegal move once
every few thousand positions. So this checks *every* occupancy subset of *every* square, which is
exhaustive over the space the tables can be asked about, not a sample of it.

It also walks a small tree checking the incremental Zobrist key against a hash recomputed from
scratch, and checking that unmake restores the position bit for bit. **Perft cannot catch either
of these**: it never reads the key, and a make/unmake asymmetry that perft would notice is only
the subset that changes the legal move count. A drifting key is invisible until the transposition
table starts returning another position's score, which looks like a search bug, not a hash bug.
"""

import sys
import time
from collections.abc import Callable

import numpy as np

from engine import bitboard, position

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


# Positions chosen so the walk exercises every part of make_move that touches the hash: castling
# rights being lost, an en-passant file appearing and disappearing, and promotions.
ZOBRIST_POSITIONS = (
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("ep", "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1"),
    ("promotions", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1"),
)


def check_zobrist(depth: int) -> int:
    """Walk a tree verifying the incremental key and that unmake restores the position exactly."""
    failures = 0
    bb, mailbox, state, key = position.new_position()
    undo, keys = position.new_undo()
    buffer = np.zeros(position.MAX_MOVES * position.MAX_PLY, dtype=np.int32)
    nodes = 0

    def walk(remaining: int, ply: int) -> int:
        nonlocal failures, nodes
        if remaining == 0:
            return 0
        offset = ply * position.MAX_MOVES
        count = position.generate_moves(bb, mailbox, state, buffer, offset)
        side = int(state[position.STM])
        for index in range(count):
            move = buffer[offset + index]
            before = (bb.copy(), mailbox.copy(), state.copy(), U64(key[0]))
            position.make_move(bb, mailbox, state, key, undo, keys, ply, move)
            king_square = position.lsb(bb[position.WK + 6 * side])
            if not position.is_attacked(bb, king_square, 1 - side):
                nodes += 1
                expected = position.compute_key(bb, state)
                if U64(key[0]) != expected:
                    failures += 1
                    if failures <= 5:
                        uci = position.move_to_uci(move)
                        print(f"  key drift after {uci}: 0x{int(key[0]):016X} != 0x{expected:016X}")
                walk(remaining - 1, ply + 1)
            position.unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
            if (
                not np.array_equal(bb, before[0])
                or not np.array_equal(mailbox, before[1])
                or not np.array_equal(state, before[2])
                or U64(key[0]) != before[3]
            ):
                failures += 1
                if failures <= 5:
                    print(
                        f"  unmake did not restore the position after {position.move_to_uci(move)}"
                    )
        return 0

    for name, fen in ZOBRIST_POSITIONS:
        position.set_fen(bb, mailbox, state, key, fen)
        walk(depth, 0)
        print(f"  {name:11} {'FAIL' if failures else 'ok'}")
    print(f"  {'':11} {nodes:,} nodes checked")
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

    # Depth 3, not more: the walk snapshots and compares the whole position at every node, so it
    # runs at Python speed, and the point is coverage of move *kinds* -- which the four positions
    # give -- rather than node count, which perft already provides by the hundred million.
    print("\nverifying incremental Zobrist keys and make/unmake symmetry\n")
    failures += check_zobrist(3)

    print(f"\n  {time.perf_counter() - started:.1f}s")
    if failures:
        sys.exit(f"\n{failures} mismatch(es) — fix before trusting search results")
    print("\nattack tables, Zobrist keys and make/unmake all correct")


if __name__ == "__main__":
    main()

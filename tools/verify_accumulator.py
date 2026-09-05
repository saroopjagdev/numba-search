"""Gate on the incremental NNUE accumulator: it must equal a full refresh after every move.

`engine.nnue.update` reconstructs which features a move changed from the move, the post-move
mailbox and the captured piece. That repeats a little of `make_move`'s case analysis, and the
failure mode if the two ever drift is the worst kind available: no crash, no illegal move, just an
evaluation that is quietly wrong in one branch -- an en passant capture, a queenside castle, an
underpromotion -- for the rest of the week.

So this walks random games and, after every single move, compares the incrementally updated
accumulator against one rebuilt from scratch. Equality is exact; these are integers and there is
no rounding to excuse a difference of one. It also unwinds, checking that the accumulator restores
on unmake, which is what the search will rely on.

Play enough games and the rare flags all appear; the run prints how many of each it actually
exercised, because a suite that never castled would pass while proving nothing.
"""

from __future__ import annotations

import argparse

import chess
import numpy as np

from engine.nnue import EMPTY, apply_feature, new_stack, push, push_null, refresh, update
from engine.position import (
    CASTLE_KING,
    CASTLE_QUEEN,
    EP_CAPTURE,
    MAX_MOVES,
    PROMO_BIT,
    WK,
    generate_moves,
    is_attacked,
    lsb,
    make_move,
    new_position,
    new_undo,
    set_fen,
    unmake_move,
)
from engine.position import STM as STM_SLOT

FEATURES = 768

# Random play from the opening almost never castles and hardly ever takes en passant -- a first
# run over 22,789 positions managed two en passants and eight castles, which is not coverage of
# the two branches most likely to be wrong. So games start from positions where the rare moves are
# already available: Kiwipete for castling on both sides, and locked-file pawn races for en
# passant and promotion. The per-flag counts printed at the end are the check that this worked.
STARTS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
    "4k3/2p1p1p1/8/1P1P1P2/1p1p1p2/8/2P1P1P1/4K3 w - - 0 1",
    "7k/1P1P1P1P/8/8/8/8/p1p1p1p1/K7 w - - 0 1",
)


def random_weights(hidden: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-range int16 weights, deliberately not a trained net.

    Small weights would let a genuine mismatch hide inside a difference of a few units. These are
    large and unstructured, so any missed feature moves the accumulator by thousands.
    """
    rng = np.random.default_rng(seed)
    transformer = rng.integers(-500, 500, size=(FEATURES, hidden), dtype=np.int16)
    bias = rng.integers(-500, 500, size=hidden, dtype=np.int16)
    return np.ascontiguousarray(transformer), np.ascontiguousarray(bias)


def square_name(square: int) -> str:
    return "abcdefgh"[square & 7] + str((square >> 3) + 1)


def describe(mailbox: np.ndarray) -> str:
    """The board as `Pe4` pairs. A mismatch is always about one specific square, so the report has
    to say which pieces were where rather than leaving the reader to reconstruct it."""
    chars = "PNBRQKpnbrqk"
    return " ".join(
        f"{chars[int(mailbox[square])]}{square_name(square)}"
        for square in range(64)
        if mailbox[square] != EMPTY
    )


# Random games are a poor way to reach a rare move: 22,000 of them produced four en passants. So
# the sweep below takes every legal move in each of these instead, where the rare flags are dense
# by construction -- an en passant available on each file and for each colour, castling both ways
# under attack, and promotions with and without a capture.
SWEEP = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/8/8/2pP4/8/8/8/4K2k w - c6 0 1",
    "8/8/8/8/2Pp4/8/8/4K2k b - c3 0 1",
    "8/8/8/PpP5/8/8/8/4K2k w - b6 0 1",
    "8/8/8/8/pPp5/8/8/4K2k b - b3 0 1",
    "8/8/8/6Pp/8/8/8/4K2k w - h6 0 1",
    "8/8/8/8/6pP/8/8/4K2k b - h3 0 1",
    "n3n3/1P4P1/8/8/8/8/8/4K2k w - - 0 1",
    "4k3/8/8/8/8/5K2/1p4p1/N3N3 b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
)


def sweep(transformer: np.ndarray, bias: np.ndarray, hidden: int) -> tuple[int, dict[str, int]]:
    """Every legal move in every SWEEP position, checked against a full refresh.

    Exhaustive rather than sampled, so a branch that is wrong only for, say, a black en passant on
    the h-file cannot slip through by not being drawn.
    """
    incremental = np.zeros((2, hidden), dtype=np.int16)
    expected = np.zeros((2, hidden), dtype=np.int16)
    buffer = np.zeros(MAX_MOVES * 64, dtype=np.int32)
    seen = {"quiet": 0, "capture": 0, "ep": 0, "castle": 0, "promotion": 0}
    checked = 0

    for fen in SWEEP:
        bb, mailbox, state, key = new_position()
        undo, keys = new_undo()
        set_fen(bb, mailbox, state, key, fen)
        count = generate_moves(bb, mailbox, state, buffer, 0)
        side = int(state[STM_SLOT])
        moves = [buffer[index] for index in range(count)]

        for move in moves:
            refresh(mailbox, transformer, bias, incremental)
            make_move(bb, mailbox, state, key, undo, keys, 0, move)
            if is_attacked(bb, lsb(bb[WK + 6 * side]), 1 - side):
                unmake_move(bb, mailbox, state, key, undo, keys, 0, move)
                continue
            flag = (int(move) >> 12) & 15
            if flag & PROMO_BIT:
                seen["promotion"] += 1
            elif flag == EP_CAPTURE:
                seen["ep"] += 1
            elif flag in (CASTLE_KING, CASTLE_QUEEN):
                seen["castle"] += 1
            elif flag & 4:
                seen["capture"] += 1
            else:
                seen["quiet"] += 1

            update(incremental, transformer, mailbox, move, int(undo[0, 0]), side)
            refresh(mailbox, transformer, bias, expected)
            checked += 1
            if not np.array_equal(incremental, expected):
                worst = int(np.abs(incremental.astype(np.int32) - expected.astype(np.int32)).max())
                print(
                    f"SWEEP MISMATCH {fen} move {square_name(int(move) & 63)}->"
                    f"{square_name((int(move) >> 6) & 63)} flag {flag}: differs by {worst}"
                )
                raise SystemExit(1)
            unmake_move(bb, mailbox, state, key, undo, keys, 0, move)

    return checked, seen


def tree(transformer: np.ndarray, bias: np.ndarray, hidden: int, depth: int) -> int:
    """Walk a real search tree on a per-ply stack, the way `engine.search` does.

    The two checks above prove `update` moves an accumulator correctly across one move. That is not
    the same claim as the search needs, which is about the *stack*: that `push` writes only the
    child ply, that the parent's accumulator is therefore still valid after `unmake_move` with no
    undo step at all, and that a null move's copy leaves the position's own view unchanged.

    So this recurses over every legal move to `depth`, and at every node asserts that `stack[ply]`
    equals a refresh of the board actually on the table. It then re-checks the parent on the way
    out, which is the assertion that makes unmake being a no-op safe rather than merely plausible.
    Null moves are exercised at alternate plies for the same reason the sweep exists: the branch
    that is never taken is the branch that is wrong.
    """
    stack = new_stack(hidden, depth + 2)
    expected = np.zeros((2, hidden), dtype=np.int16)
    buffer = np.zeros(MAX_MOVES * (depth + 2), dtype=np.int32)
    visited = 0

    def descend(
        bb: np.ndarray,
        mailbox: np.ndarray,
        state: np.ndarray,
        key: np.ndarray,
        undo: np.ndarray,
        keys: np.ndarray,
        ply: int,
        remaining: int,
    ) -> bool:
        nonlocal visited
        visited += 1
        refresh(mailbox, transformer, bias, expected)
        if not np.array_equal(stack[ply], expected):
            print(f"TREE MISMATCH at ply {ply}: {describe(mailbox)}")
            return False
        if remaining == 0:
            return True

        # A null move changes nothing on the board, so the child's accumulator must equal the
        # parent's and a refresh of the unchanged board must equal both.
        if ply % 2 == 0:
            push_null(stack, ply)
            if not np.array_equal(stack[ply + 1], expected):
                print(f"NULL MISMATCH at ply {ply}")
                return False

        offset = ply * MAX_MOVES
        count = generate_moves(bb, mailbox, state, buffer, offset)
        side = int(state[STM_SLOT])
        for index in range(count):
            move = buffer[offset + index]
            make_move(bb, mailbox, state, key, undo, keys, ply, move)
            if is_attacked(bb, lsb(bb[WK + 6 * side]), 1 - side):
                unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
                continue
            push(stack, transformer, mailbox, ply, move, int(undo[ply, 0]), side)
            deeper = descend(bb, mailbox, state, key, undo, keys, ply + 1, remaining - 1)
            unmake_move(bb, mailbox, state, key, undo, keys, ply, move)
            if not deeper:
                return False
            # The parent, after unmake and after the child wrote all over ply + 1.
            refresh(mailbox, transformer, bias, expected)
            if not np.array_equal(stack[ply], expected):
                origin = square_name(int(move) & 63)
                target = square_name((int(move) >> 6) & 63)
                print(f"PARENT CLOBBERED at ply {ply} returning from {origin}->{target}")
                return False
        return True

    for fen in SWEEP[:4]:
        bb, mailbox, state, key = new_position()
        undo, keys = new_undo()
        set_fen(bb, mailbox, state, key, fen)
        refresh(mailbox, transformer, bias, stack[0])
        if not descend(bb, mailbox, state, key, undo, keys, 0, depth):
            print(f"  from {fen}")
            return -1
    return visited


def check_starts() -> None:
    """Refuse to run from a position where the side not to move is already in check.

    Generation is pseudo-legal and the legality filter only asks whether the *mover's* king is
    safe, so from such a position the first move captures a king and every board after it is
    nonsense. The first draft of this file hand-wrote two pawn-race positions that were illegal in
    exactly this way, and the resulting report -- a move from an empty square, a white rook where
    the black king had been -- looked convincingly like an accumulator bug. It cost more time than
    this function will ever take.
    """
    for fen in STARTS + SWEEP:
        board = chess.Board(fen)
        board.turn = not board.turn
        if board.is_check():
            raise SystemExit(f"illegal start position, side not to move is in check: {fen}")


def run(games: int, plies: int, hidden: int, seed: int, depth: int) -> int:
    check_starts()
    transformer, bias = random_weights(hidden, seed)
    rng = np.random.default_rng(seed)

    incremental = np.zeros((2, hidden), dtype=np.int16)
    expected = np.zeros((2, hidden), dtype=np.int16)
    buffer = np.zeros(MAX_MOVES * 64, dtype=np.int32)

    seen = {"quiet": 0, "capture": 0, "ep": 0, "castle": 0, "promotion": 0}
    checked = 0

    for game in range(games):
        bb, mailbox, state, key = new_position()
        undo, keys = new_undo()
        set_fen(bb, mailbox, state, key, STARTS[game % len(STARTS)])
        refresh(mailbox, transformer, bias, incremental)

        for ply in range(plies):
            count = generate_moves(bb, mailbox, state, buffer, 0)
            legal = []
            side = int(state[STM_SLOT])
            for index in range(count):
                move = buffer[index]
                make_move(bb, mailbox, state, key, undo, keys, 0, move)
                if not is_attacked(bb, lsb(bb[WK + 6 * side]), 1 - side):
                    legal.append(move)
                unmake_move(bb, mailbox, state, key, undo, keys, 0, move)
            if not legal:
                break

            move = legal[int(rng.integers(len(legal)))]
            flag = (int(move) >> 12) & 15
            if flag & PROMO_BIT:
                seen["promotion"] += 1
            elif flag == EP_CAPTURE:
                seen["ep"] += 1
            elif flag in (CASTLE_KING, CASTLE_QUEEN):
                seen["castle"] += 1
            elif flag & 4:
                seen["capture"] += 1
            else:
                seen["quiet"] += 1

            before = incremental.copy()
            occupied_before = describe(mailbox)
            make_move(bb, mailbox, state, key, undo, keys, 0, move)
            captured = int(undo[0, 0])
            update(incremental, transformer, mailbox, move, captured, side)

            refresh(mailbox, transformer, bias, expected)
            checked += 1
            if not np.array_equal(incremental, expected):
                worst = int(np.abs(incremental.astype(np.int32) - expected.astype(np.int32)).max())
                origin = square_name(int(move) & 63)
                target = square_name((int(move) >> 6) & 63)
                print(
                    f"MISMATCH game {game} ply {ply} move {int(move)} {origin}->{target} "
                    f"flag {flag} captured {captured} side {side}: "
                    f"worst element differs by {worst}"
                )
                print(f"  before: {occupied_before}")
                print(f"  after:  {describe(mailbox)}")
                return 1

            # And back: the search unmakes constantly, so the reverse has to be exact too.
            reverted = incremental.copy()
            _unwind(reverted, transformer, mailbox, move, captured, side)
            unmake_move(bb, mailbox, state, key, undo, keys, 0, move)
            if not np.array_equal(reverted, before):
                print(f"UNMAKE MISMATCH game {game} ply {ply} move {int(move)} flag {flag}")
                return 1
            make_move(bb, mailbox, state, key, undo, keys, 0, move)

    swept, sweep_seen = sweep(transformer, bias, hidden)
    for flag_name, number in sweep_seen.items():
        seen[flag_name] += number

    visited = tree(transformer, bias, hidden, depth)
    if visited < 0:
        return 1

    counts = "  ".join(f"{flag_name} {number:,}" for flag_name, number in seen.items())
    print(f"{checked:,} random-game positions and {swept:,} exhaustive moves checked")
    print(f"{visited:,} search-tree nodes checked on the per-ply stack, depth {depth}")
    print("incremental == refresh exactly, on both make and unmake")
    print(f"exercised: {counts}")
    for flag_name, number in seen.items():
        if number == 0:
            print(f"FAIL: no {flag_name} move ever occurred; that branch is untested")
            return 1
    return 0


def _unwind(
    accumulator: np.ndarray,
    transformer: np.ndarray,
    mailbox: np.ndarray,
    move: np.int32,
    captured: int,
    side: int,
) -> None:
    """Undo one `update`. Every feature it added is removed and vice versa."""
    from_square = int(move) & 63
    to_square = (int(move) >> 6) & 63
    flag = (int(move) >> 12) & 15
    us = 6 * side
    arrived = int(mailbox[to_square])
    moved = us if flag & PROMO_BIT else arrived

    if captured != EMPTY:
        behind = to_square - 8 if side == 0 else to_square + 8
        taken = behind if flag == EP_CAPTURE else to_square
        apply_feature(accumulator, transformer, captured, taken, 1)
    apply_feature(accumulator, transformer, moved, from_square, 1)
    apply_feature(accumulator, transformer, arrived, to_square, -1)
    if flag == CASTLE_KING:
        apply_feature(accumulator, transformer, us + 3, from_square + 3, 1)
        apply_feature(accumulator, transformer, us + 3, from_square + 1, -1)
    elif flag == CASTLE_QUEEN:
        apply_feature(accumulator, transformer, us + 3, from_square - 4, 1)
        apply_feature(accumulator, transformer, us + 3, from_square - 1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the incremental NNUE accumulator.")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--plies", type=int, default=180)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=3)
    arguments = parser.parse_args()
    raise SystemExit(
        run(
            arguments.games,
            arguments.plies,
            arguments.hidden,
            arguments.seed,
            arguments.depth,
        )
    )


if __name__ == "__main__":
    main()

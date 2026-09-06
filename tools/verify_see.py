"""Cross-check `see_ge` against an independent swap-off written with python-chess.

A wrong static exchange evaluation is the worst kind of bug this project can have: it never crashes,
never produces an illegal move, and never shows up in perft. It just quietly prunes good captures
and searches bad ones, and the engine plays a little worse forever. Nothing mechanical catches that,
so it gets a reference implementation.

The reference is deliberately *not* a port of the same algorithm -- two copies of one idea share its
bugs. It is the recursive definition instead: take the least valuable attacker, recapture, and let
each side decline whenever continuing loses material. It uses `board.attackers`, which recomputes
attacks from the real occupancy, so x-rays fall out for free rather than being modelled.

The two agree on model, not just on answers. Both ignore pins -- a pinned defender counts as a real
one -- because that is the standard trade for a function called at every capture, and a reference
that respected pins would report disagreements that are not bugs.

    uv run python tools/verify_see.py
"""

import random
import sys

import chess
import numpy as np

from engine.position import move_from, move_to, new_position, set_fen
from engine.search import PIECE_VALUES, see_ge

# The engine's own values, indexed the way python-chess numbers piece types (PAWN == 1).
VALUE = {
    chess.PAWN: int(PIECE_VALUES[0]),
    chess.KNIGHT: int(PIECE_VALUES[1]),
    chess.BISHOP: int(PIECE_VALUES[2]),
    chess.ROOK: int(PIECE_VALUES[3]),
    chess.QUEEN: int(PIECE_VALUES[4]),
    chess.KING: int(PIECE_VALUES[5]),
}


def _swap_off(board: chess.Board, target: int, side: chess.Color) -> int:
    """Best material `side` can win by continuing the exchange on `target`. Never negative."""
    attackers = board.attackers(side, target)
    if not attackers:
        return 0
    frm = min(attackers, key=lambda square: VALUE[board.piece_type_at(square)])  # type: ignore[index]
    moving = board.piece_at(frm)
    assert moving is not None
    victim = board.piece_at(target)
    assert victim is not None

    # A king may only take if nothing of the other side still guards the square afterwards.
    if moving.piece_type == chess.KING and board.attackers(not side, target):
        return 0

    after = board.copy(stack=False)
    after.remove_piece_at(frm)
    after.set_piece_at(target, moving)
    # Declining is always allowed, which is what makes this the *static exchange* rather than the
    # material count of playing every capture to the bitter end.
    return max(0, VALUE[victim.piece_type] - _swap_off(after, target, not side))


def reference(board: chess.Board, move: chess.Move) -> int:
    """The exchange value of `move` for the side making it."""
    victim = board.piece_at(move.to_square)
    assert victim is not None
    after = board.copy(stack=False)
    moving = after.piece_at(move.from_square)
    assert moving is not None
    after.remove_piece_at(move.from_square)
    after.set_piece_at(move.to_square, moving)
    return VALUE[victim.piece_type] - _swap_off(after, move.to_square, not board.turn)


def engine_moves(fen: str) -> dict[tuple[int, int], np.int32]:
    """The engine's encoded moves for a position, keyed by (from, to)."""
    from engine.position import generate_moves

    bb, mailbox, state, key = new_position()
    set_fen(bb, mailbox, state, key, fen)
    buffer = np.zeros(256, dtype=np.int32)
    count = generate_moves(bb, mailbox, state, buffer, 0)
    return {(int(move_from(buffer[i])), int(move_to(buffer[i]))): buffer[i] for i in range(count)}


def main() -> None:
    rng = random.Random(20260906)
    bb, mailbox, state, key = new_position()

    checked = 0
    mismatches = []
    for _game in range(300):
        board = chess.Board()
        # Random play into the middlegame, where exchanges are dense and x-rays common.
        for _ in range(rng.randint(8, 60)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if board.is_game_over():
            continue

        fen = board.fen()
        encoded = engine_moves(fen)
        set_fen(bb, mailbox, state, key, fen)

        for move in board.legal_moves:
            # Plain captures only. Promotions change the attacker's value mid-exchange and en
            # passant moves the victim off the target square; the caller never asks about either.
            if not board.is_capture(move) or move.promotion or board.is_en_passant(move):
                continue
            key_pair = (move.from_square, move.to_square)
            if key_pair not in encoded:
                continue
            truth = reference(board, move)
            for threshold in (-500, -100, 0, 1, 100, 500):
                got = bool(see_ge(bb, mailbox, state, encoded[key_pair], np.int32(threshold)))
                want = truth >= threshold
                checked += 1
                if got != want:
                    mismatches.append((fen, move.uci(), threshold, truth, got, want))

    print(f"checked {checked:,} (move, threshold) pairs across random middlegames")
    if mismatches:
        print(f"\n{len(mismatches)} MISMATCHES, first 10:")
        for fen, uci, threshold, truth, got, want in mismatches[:10]:
            print(f"  {uci} thr={threshold:+5d} reference={truth:+5d} see_ge={got} want={want}")
            print(f"    {fen}")
        sys.exit(1)
    print("OK - see_ge agrees with the independent swap-off everywhere")


if __name__ == "__main__":
    main()

"""The perft backend `tools/perft.py --backend engine` drives.

Thin: it owns the scratch arrays and hands them to the jitted counter. Everything that could be
wrong lives in `engine/position.py`, which is the point — this file exists so the exact same
node counts that verify the reference backend also verify ours.
"""

import numpy as np

from engine.position import (
    MAX_MOVES,
    MAX_PLY,
    STM,
    WK,
    generate_moves,
    is_attacked,
    lsb,
    make_move,
    move_to_uci,
    new_position,
    new_undo,
    perft,
    set_fen,
    unmake_move,
)


class EnginePerft:
    """Our move generator, behind the same interface as the python-chess reference."""

    name = "engine"

    def __init__(self) -> None:
        self.bb, self.mailbox, self.state, self.key = new_position()
        self.undo, self.keys = new_undo()
        self.buffer = np.zeros(MAX_MOVES * MAX_PLY, dtype=np.int32)

    def perft(self, fen: str, depth: int) -> int:
        set_fen(self.bb, self.mailbox, self.state, self.key, fen)
        return int(
            perft(
                self.bb,
                self.mailbox,
                self.state,
                self.key,
                self.undo,
                self.keys,
                self.buffer,
                0,
                depth,
            )
        )

    def divide(self, fen: str, depth: int) -> list[tuple[str, int]]:
        set_fen(self.bb, self.mailbox, self.state, self.key, fen)
        count = generate_moves(self.bb, self.mailbox, self.state, self.buffer, 0)
        side = int(self.state[STM])
        results = []
        for index in range(count):
            move = self.buffer[index]
            make_move(self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, 0, move)
            king_square = lsb(self.bb[WK + 6 * side])
            if not is_attacked(self.bb, king_square, 1 - side):
                nodes = int(
                    perft(
                        self.bb,
                        self.mailbox,
                        self.state,
                        self.key,
                        self.undo,
                        self.keys,
                        self.buffer,
                        1,
                        depth - 1,
                    )
                )
                results.append((move_to_uci(move), nodes))
            unmake_move(self.bb, self.mailbox, self.state, self.key, self.undo, self.keys, 0, move)
        return sorted(results)

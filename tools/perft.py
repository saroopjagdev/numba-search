"""Exact node counts against a known suite. The non-negotiable gate on movegen work.

This is where LLM-written engines silently emit illegal moves and lose on the spot. perft is an
exact integer: either the movegen is right or it is not, and there is no partial credit.

Two backends. `chess` uses python-chess and is the reference — slow, but it validates the suite
and the harness itself before we have an engine. `engine` will use our numba movegen once
`engine/` exists, and is what actually gets gated.

    uv run python tools/perft.py --depth 4
    uv run python tools/perft.py --suite kiwipete --depth 5
    uv run python tools/perft.py --divide --fen "<fen>" --depth 3

`--divide` is the debugging tool: it prints the node count under each root move, so a mismatch
against a reference gets localised to one move in a few recursions instead of being hunted
through the whole tree.
"""

import argparse
import sys
import time
from typing import Protocol

import chess

# Canonical positions. Counts from the Chess Programming Wiki; treat them as ground truth.
SUITE: dict[str, tuple[str, list[int]]] = {
    "startpos": (
        chess.STARTING_FEN,
        [1, 20, 400, 8_902, 197_281, 4_865_609, 119_060_324],
    ),
    "kiwipete": (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [1, 48, 2_039, 97_862, 4_085_603, 193_690_690],
    ),
    "position3": (
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        [1, 14, 191, 2_812, 43_238, 674_624, 11_030_083],
    ),
    "position4": (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [1, 6, 264, 9_467, 422_333, 15_833_292],
    ),
    "position5": (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [1, 44, 1_486, 62_379, 2_103_487, 89_941_194],
    ),
    "position6": (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [1, 46, 2_079, 89_890, 3_894_594, 164_075_551],
    ),
    # --- martin sedlak's TalkChess edge-case suite ---------------------------------
    # Every ladder below was generated with python-chess and its deepest entry then checked
    # against the published TalkChess count. All nine agreed exactly. Do not hand-edit these:
    # two hand-entered numbers in the first draft of this file were wrong, and a wrong expected
    # count is worse than no test — it sends you hunting a bug that does not exist.
    #
    # en passant capture that would expose the king; must not be generated
    "ep_illegal_black": (
        "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
        [1, 18, 92, 1670, 10138, 185429, 1134888],
    ),
    "ep_illegal_white": (
        "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
        [1, 13, 102, 1266, 10276, 135655, 1015133],
    ),
    # en passant capture that delivers check
    "ep_check": ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", [1, 15, 126, 1928, 13931, 206379, 1440467]),
    # castling that gives check, and castling rights lost or blocked
    "castle_short_check": (
        "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
        [1, 15, 66, 1198, 6399, 120330, 661072],
    ),
    "castle_long_check": (
        "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1",
        [1, 16, 71, 1286, 7418, 141077, 803711],
    ),
    "castle_check": ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", [1, 26, 1141, 27826, 1274206]),
    "castle_rights": ("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", [1, 44, 1494, 50509, 1720476]),
    # discovered check
    "discovered_check": ("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", [1, 29, 165, 5160, 31961, 1004658]),
    # promotion out of check, promotion giving check, under-promotion giving check
    "promo_check": (
        "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1",
        [1, 11, 133, 1442, 19174, 266199, 3821001],
    ),
    "promo_pawn": ("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", [1, 9, 40, 472, 2661, 38983, 217342]),
    "underpromo_check": ("8/P1k5/K7/8/8/8/8/8 w - - 0 1", [1, 6, 27, 273, 1329, 18135, 92683]),
    # stalemate and checkmate detection at the leaves
    "self_stalemate": ("K1k5/8/P7/8/8/8/8/8 w - - 0 1", [1, 2, 6, 13, 63, 382, 2217]),
    "stalemate_mate": (
        "8/k1P5/8/1K6/8/8/8/8 w - - 0 1",
        [1, 10, 25, 268, 926, 10857, 43261, 567584],
    ),
    "mate_stalemate": ("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", [1, 37, 183, 6559, 23527]),
}


class MoveGen(Protocol):
    """What perft needs from a backend."""

    def perft(self, fen: str, depth: int) -> int: ...

    def divide(self, fen: str, depth: int) -> list[tuple[str, int]]: ...


class PythonChess:
    """Reference backend. Correct by assumption, and far too slow to be the engine."""

    name = "python-chess"

    def perft(self, fen: str, depth: int) -> int:
        return self._count(chess.Board(fen), depth)

    def divide(self, fen: str, depth: int) -> list[tuple[str, int]]:
        board = chess.Board(fen)
        results = []
        for move in board.legal_moves:
            board.push(move)
            results.append((move.uci(), self._count(board, depth - 1)))
            board.pop()
        return sorted(results)

    def _count(self, board: chess.Board, depth: int) -> int:
        if depth == 0:
            return 1
        if depth == 1:
            return board.legal_moves.count()
        total = 0
        for move in board.legal_moves:
            board.push(move)
            total += self._count(board, depth - 1)
            board.pop()
        return total


def load_backend(name: str) -> MoveGen:
    if name == "chess":
        return PythonChess()
    if name == "engine":
        try:
            from engine.perft import EnginePerft
        except ImportError:
            sys.exit("engine backend unavailable — build engine/ first, or use --backend chess")
        return EnginePerft()
    sys.exit(f"unknown backend {name!r}")


def run_suite(backend: MoveGen, names: list[str], max_depth: int) -> int:
    failures = 0
    for name in names:
        fen, expected = SUITE[name]
        limit = min(max_depth, len(expected) - 1)
        print(f"\n{name}  (depth 1..{limit})")
        print(f"  {fen}")
        for depth in range(1, limit + 1):
            started = time.perf_counter()
            actual = backend.perft(fen, depth)
            elapsed = time.perf_counter() - started
            want = expected[depth]
            nps = actual / elapsed if elapsed > 0 else 0.0
            if actual == want:
                print(f"  depth {depth} {actual:>12,}  ok   {elapsed:6.2f}s {nps / 1e6:6.2f} Mnps")
            else:
                delta = actual - want
                print(f"  depth {depth}  {actual:>12,}  FAIL  expected {want:,} ({delta:+,})")
                failures += 1
                break  # deeper counts are meaningless once one is wrong
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="chess", choices=("chess", "engine"))
    parser.add_argument("--depth", type=int, default=4, help="maximum depth (clamped per position)")
    parser.add_argument("--suite", action="append", choices=sorted(SUITE), help="repeatable")
    parser.add_argument("--divide", action="store_true", help="per-root-move counts, for debugging")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="position for --divide")
    args = parser.parse_args()

    backend = load_backend(args.backend)

    if args.divide:
        total = 0
        for move, count in backend.divide(args.fen, args.depth):
            print(f"  {move}  {count:>12,}")
            total += count
        print(f"\n  total  {total:,}")
        return

    names = args.suite or sorted(SUITE)
    print(f"perft — backend {backend.name if hasattr(backend, 'name') else args.backend}")
    failures = run_suite(backend, names, args.depth)
    if failures:
        sys.exit(f"\n{failures} position(s) FAILED — movegen is wrong, trust nothing downstream")
    print("\nall positions correct")


if __name__ == "__main__":
    main()

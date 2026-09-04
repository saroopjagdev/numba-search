"""Measure numba compile cost for the shapes a real engine needs.

The platform gives a 90s init budget and wipes `/tmp` between games, and numba's `.nbc` cache
artifacts are native object code we cannot ship. So we pay full compilation on every single game,
forever. This script keeps that number honest: run it after any change that adds or alters a
jitted function, and append the result to `notes/measurements.md`.

Every function below carries an explicit eager signature, so compilation happens at decoration
time and the timing around it is the compile cost.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit

TIMINGS: list[tuple[str, float]] = []

BUDGET_S = 90.0
CAP_S = 75.0
TARGET_S = 60.0


def timed(name: str, signature: str) -> Callable[[Callable[..., Any]], Any]:
    """Compile a function eagerly and record how long it took."""

    def decorate(function: Callable[..., Any]) -> Any:
        started = time.perf_counter()
        compiled = njit(signature, cache=False)(function)
        TIMINGS.append((name, time.perf_counter() - started))
        return compiled

    return decorate


# --- bit primitives -------------------------------------------------------------------


@timed("popcount", "int64(uint64)")
def popcount(bb: Any) -> int:
    count = 0
    while bb:
        bb &= bb - np.uint64(1)
        count += 1
    return count


@timed("lsb_scan", "int64(uint64)")
def lsb_scan(bb: Any) -> int:
    index = 0
    while not (bb >> np.uint64(index)) & np.uint64(1):
        index += 1
    return index


# --- sliding attacks, the expensive part of movegen -----------------------------------


@timed("ray_attacks", "uint64(int64, uint64, int64[:, :])")
def ray_attacks(square: int, occupancy: Any, deltas: Any) -> Any:
    attacks = np.uint64(0)
    for direction in range(deltas.shape[0]):
        file_step = deltas[direction, 0]
        rank_step = deltas[direction, 1]
        file = (square & 7) + file_step
        rank = (square >> 3) + rank_step
        while 0 <= file < 8 and 0 <= rank < 8:
            target = rank * 8 + file
            attacks |= np.uint64(1) << np.uint64(target)
            if (occupancy >> np.uint64(target)) & np.uint64(1):
                break
            file += file_step
            rank += rank_step
    return attacks


@timed("magic_lookup", "uint64(uint64, uint64, int64, int64, uint64[:])")
def magic_lookup(occupancy: Any, magic: Any, shift: int, offset: int, table: Any) -> Any:
    return table[offset + int((occupancy * magic) >> np.uint64(shift))]


# --- move generation into a preallocated buffer ---------------------------------------


@timed("generate_moves", "int64(uint64[:], uint64[:], int64, int32[:], int64)")
def generate_moves(pieces: Any, attacks: Any, side: int, out: Any, cursor: int) -> int:
    for piece in range(6):
        bb = pieces[side * 6 + piece]
        while bb:
            source = 0
            probe = bb
            while not probe & np.uint64(1):
                probe >>= np.uint64(1)
                source += 1
            targets = attacks[source]
            while targets:
                target = 0
                scan = targets
                while not scan & np.uint64(1):
                    scan >>= np.uint64(1)
                    target += 1
                out[cursor] = np.int32(source | (target << 6) | (piece << 12))
                cursor += 1
                targets &= targets - np.uint64(1)
            bb &= bb - np.uint64(1)
    return cursor


@timed("make_unmake", "void(uint64[:], int32, int64)")
def make_unmake(pieces: Any, move: Any, side: int) -> None:
    source = np.uint64(move & 63)
    target = np.uint64((move >> 6) & 63)
    piece = (move >> 12) & 7
    pieces[side * 6 + piece] ^= (np.uint64(1) << source) | (np.uint64(1) << target)


@timed("zobrist_update", "uint64(uint64, uint64[:, :], int32, int64)")
def zobrist_update(key: Any, keys: Any, move: Any, side: int) -> Any:
    source = move & 63
    target = (move >> 6) & 63
    piece = side * 6 + ((move >> 12) & 7)
    return key ^ keys[piece, source] ^ keys[piece, target]


# --- NNUE kernels ---------------------------------------------------------------------


@timed("accumulator_update", "void(int16[:, :], int16[:, :], int64, int64, int64)")
def accumulator_update(acc: Any, weights: Any, perspective: int, feature: int, sign: int) -> None:
    width = acc.shape[1]
    if sign > 0:
        for i in range(width):
            acc[perspective, i] += weights[feature, i]
    else:
        for i in range(width):
            acc[perspective, i] -= weights[feature, i]


@timed("forward_screlu", "int32(int16[:, :], int16[:, :], int32)")
def forward_screlu(acc: Any, output_weights: Any, bias: Any) -> Any:
    total = np.int64(0)
    for perspective in range(2):
        for i in range(acc.shape[1]):
            value = acc[perspective, i]
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            clipped = np.int64(value)
            total += clipped * clipped * np.int64(output_weights[perspective, i])
    return np.int32(total // np.int64(255 * 64) + np.int64(bias))


# --- search ---------------------------------------------------------------------------


@timed("order_moves", "void(int32[:], int32[:], int64)")
def order_moves(moves: Any, scores: Any, count: int) -> None:
    for i in range(1, count):
        move = moves[i]
        score = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < score:
            moves[j + 1] = moves[j]
            scores[j + 1] = scores[j]
            j -= 1
        moves[j + 1] = move
        scores[j + 1] = score


@timed("see", "int32(uint64[:], int64, int64, int32[:])")
def see(pieces: Any, square: int, side: int, values: Any) -> Any:
    gain = np.int32(0)
    attacker = side
    for _ in range(32):
        found = -1
        for piece in range(6):
            if pieces[attacker * 6 + piece] & (np.uint64(1) << np.uint64(square)):
                found = piece
                break
        if found < 0:
            break
        gain = values[found] - gain
        pieces[attacker * 6 + found] &= ~(np.uint64(1) << np.uint64(square))
        attacker ^= 1
    return gain


@timed("tt_probe", "int64(uint64[:], int32[:], uint64, int64)")
def tt_probe(table_keys: Any, table_scores: Any, key: Any, mask: int) -> int:
    index = int(key) & mask
    if table_keys[index] == key:
        return int(table_scores[index])
    return -1 << 30


def main() -> None:
    total = sum(duration for _, duration in TIMINGS)
    width = max(len(name) for name, _ in TIMINGS)
    print(f"numba compile cost — {len(TIMINGS)} functions\n")
    for name, duration in sorted(TIMINGS, key=lambda item: -item[1]):
        print(f"  {name:<{width}}  {duration:6.2f}s")
    print(f"\n  {'TOTAL':<{width}}  {total:6.2f}s")

    # A finished engine carries several times this function count. Extrapolate honestly.
    print(f"\n  mean per function     {total / len(TIMINGS):6.2f}s")
    for multiple in (2, 3, 5):
        print(f"  extrapolated x{multiple}         {total * multiple:6.2f}s")
    print(f"\n  target {TARGET_S:.0f}s / cap {CAP_S:.0f}s / hard limit {BUDGET_S:.0f}s")


if __name__ == "__main__":
    main()

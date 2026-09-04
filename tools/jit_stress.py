"""Compile-time stress test on realistically *large* numba functions.

`tools/jit_timing.py` measures a dozen small kernels and gives a reassuring number. That number
lies by omission: numba's compile cost scales with the size and branchiness of a function body,
not with how many functions there are. The two functions that dominate a real engine's init are
the full pseudo-legal move generator and the recursive alpha-beta body, and recursion in
particular is known to be expensive to type-infer.

Neither function here is engine code we intend to keep. They are structural proxies — the right
shape and roughly the right size — used to find out what the finished engine will cost at import
before we have written it.
"""

import time
from typing import Any

import numpy as np
from numba import njit

ONE = np.uint64(1)


def main() -> None:
    timings: list[tuple[str, float]] = []

    started = time.perf_counter()

    # --- proxy 1: full pseudo-legal movegen, all piece types, castling, ep, promotions -

    @njit("int64(uint64[:], int64, int64, uint64, int32[:], int64[:, :])", cache=False)
    def generate_all(
        pieces: Any, side: int, castling: int, ep_square: Any, out: Any, deltas: Any
    ) -> int:
        cursor = 0
        occupancy = np.uint64(0)
        for i in range(12):
            occupancy |= pieces[i]
        own = np.uint64(0)
        for i in range(6):
            own |= pieces[side * 6 + i]
        enemy = occupancy & ~own

        # pawns
        pawns = pieces[side * 6]
        forward = 8 if side == 0 else -8
        start_rank = 1 if side == 0 else 6
        promo_rank = 6 if side == 0 else 1
        while pawns:
            source = 0
            probe = pawns
            while not probe & ONE:
                probe >>= ONE
                source += 1
            pawns &= pawns - ONE
            rank = source >> 3
            target = source + forward
            if 0 <= target < 64 and not (occupancy >> np.uint64(target)) & ONE:
                if rank == promo_rank:
                    for promo in range(1, 5):
                        out[cursor] = np.int32(source | (target << 6) | (promo << 12))
                        cursor += 1
                else:
                    out[cursor] = np.int32(source | (target << 6))
                    cursor += 1
                    if rank == start_rank:
                        double = target + forward
                        if not (occupancy >> np.uint64(double)) & ONE:
                            out[cursor] = np.int32(source | (double << 6))
                            cursor += 1
            for side_step in (-1, 1):
                file = source & 7
                if file + side_step < 0 or file + side_step > 7:
                    continue
                capture = source + forward + side_step
                if capture < 0 or capture > 63:
                    continue
                bit = ONE << np.uint64(capture)
                if enemy & bit or (ep_square != np.uint64(64) and capture == int(ep_square)):
                    if rank == promo_rank:
                        for promo in range(1, 5):
                            out[cursor] = np.int32(source | (capture << 6) | (promo << 12))
                            cursor += 1
                    else:
                        out[cursor] = np.int32(source | (capture << 6))
                        cursor += 1

        # knights and king, offset driven
        for piece in (1, 5):
            bb = pieces[side * 6 + piece]
            base = 0 if piece == 1 else 8
            while bb:
                source = 0
                probe = bb
                while not probe & ONE:
                    probe >>= ONE
                    source += 1
                bb &= bb - ONE
                for offset in range(base, base + 8):
                    file = (source & 7) + deltas[offset, 0]
                    rank = (source >> 3) + deltas[offset, 1]
                    if file < 0 or file > 7 or rank < 0 or rank > 7:
                        continue
                    target = rank * 8 + file
                    if own & (ONE << np.uint64(target)):
                        continue
                    out[cursor] = np.int32(source | (target << 6))
                    cursor += 1

        # sliders
        for piece in (2, 3, 4):
            bb = pieces[side * 6 + piece]
            first = 16 if piece == 2 else (20 if piece == 3 else 16)
            count = 4 if piece < 4 else 8
            while bb:
                source = 0
                probe = bb
                while not probe & ONE:
                    probe >>= ONE
                    source += 1
                bb &= bb - ONE
                for offset in range(first, first + count):
                    file_step = deltas[offset, 0]
                    rank_step = deltas[offset, 1]
                    file = (source & 7) + file_step
                    rank = (source >> 3) + rank_step
                    while 0 <= file < 8 and 0 <= rank < 8:
                        target = rank * 8 + file
                        bit = ONE << np.uint64(target)
                        if own & bit:
                            break
                        out[cursor] = np.int32(source | (target << 6))
                        cursor += 1
                        if enemy & bit:
                            break
                        file += file_step
                        rank += rank_step

        # castling
        if side == 0:
            if castling & 1 and not occupancy & np.uint64(0x60):
                out[cursor] = np.int32(4 | (6 << 6) | (1 << 15))
                cursor += 1
            if castling & 2 and not occupancy & np.uint64(0xE):
                out[cursor] = np.int32(4 | (2 << 6) | (1 << 15))
                cursor += 1
        else:
            if castling & 4 and not occupancy & np.uint64(0x6000000000000000):
                out[cursor] = np.int32(60 | (62 << 6) | (1 << 15))
                cursor += 1
            if castling & 8 and not occupancy & np.uint64(0xE00000000000000):
                out[cursor] = np.int32(60 | (58 << 6) | (1 << 15))
                cursor += 1
        return cursor

    timings.append(("generate_all (full movegen)", time.perf_counter() - started))
    started = time.perf_counter()

    # --- proxy 2: recursive alpha-beta with the full pruning stack ---------------------

    search_signature = (
        "int32(uint64[:], int64, int32, int32, int64, int32[:, :], int32[:], uint64[:], int64)"
    )

    @njit(search_signature, cache=False)
    def alphabeta(
        pieces: Any,
        side: int,
        alpha: Any,
        beta: Any,
        depth: int,
        stack: Any,
        history: Any,
        tt_keys: Any,
        ply: int,
    ) -> Any:
        if depth <= 0:
            score = np.int32(0)
            for i in range(12):
                bb = pieces[i]
                count = 0
                while bb:
                    bb &= bb - ONE
                    count += 1
                sign = 1 if i < 6 else -1
                score += np.int32(sign * count * (i % 6 + 1) * 100)
            return score if side == 0 else -score

        # transposition probe
        key = np.uint64(0)
        for i in range(12):
            key ^= pieces[i] * np.uint64(0x9E3779B97F4A7C15)
        index = int(key) & (tt_keys.shape[0] - 1)
        if tt_keys[index] == key and depth <= 2:
            return alpha

        # null move
        if depth >= 3 and beta < np.int32(30000):
            reduced = depth - 3
            score = -alphabeta(
                pieces, side ^ 1, -beta, -beta + np.int32(1), reduced,
                stack, history, tt_keys, ply + 1,
            )
            if score >= beta:
                return beta

        best = np.int32(-32000)
        moves = stack[ply]
        count = 0
        for i in range(12):
            bb = pieces[i]
            while bb and count < 8:
                source = 0
                probe = bb
                while not probe & ONE:
                    probe >>= ONE
                    source += 1
                bb &= bb - ONE
                moves[count] = np.int32(source | (i << 12))
                count += 1

        # ordering by history
        for i in range(1, count):
            move = moves[i]
            score = history[move & 4095]
            j = i - 1
            while j >= 0 and history[moves[j] & 4095] < score:
                moves[j + 1] = moves[j]
                j -= 1
            moves[j + 1] = move

        searched = 0
        for i in range(count):
            move = moves[i]
            piece = (move >> 12) & 15
            source = move & 63

            # futility
            if depth == 1 and searched > 0 and alpha > np.int32(-30000):
                continue

            pieces[piece] ^= ONE << np.uint64(source)

            # late move reduction
            reduction = 0
            if searched >= 3 and depth >= 3:
                reduction = 1
            score = -alphabeta(
                pieces, side ^ 1, -beta, -alpha, depth - 1 - reduction,
                stack, history, tt_keys, ply + 1,
            )
            if reduction > 0 and score > alpha:
                score = -alphabeta(
                    pieces, side ^ 1, -beta, -alpha, depth - 1, stack, history, tt_keys, ply + 1
                )

            pieces[piece] ^= ONE << np.uint64(source)
            searched += 1

            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    history[move & 4095] += np.int32(depth * depth)
                    tt_keys[index] = key
                    break
        return best

    timings.append(("alphabeta (recursive search)", time.perf_counter() - started))

    total = sum(duration for _, duration in timings)
    width = max(len(name) for name, _ in timings)
    print("numba compile cost — large-function proxies\n")
    for name, duration in timings:
        print(f"  {name:<{width}}  {duration:6.2f}s")
    print(f"\n  {'TOTAL':<{width}}  {total:6.2f}s")

    # exercise them once so a compile that silently deferred would show up
    pieces = np.zeros(12, dtype=np.uint64)
    pieces[0] = np.uint64(0xFF00)
    pieces[6] = np.uint64(0xFF000000000000)
    deltas = np.zeros((24, 2), dtype=np.int64)
    out = np.zeros(256, dtype=np.int32)
    generate_all(pieces, 0, 15, np.uint64(64), out, deltas)
    alphabeta(
        pieces,
        0,
        np.int32(-32000),
        np.int32(32000),
        3,
        np.zeros((64, 256), dtype=np.int32),
        np.zeros(4096, dtype=np.int32),
        np.zeros(1024, dtype=np.uint64),
        0,
    )
    print("  both functions executed cleanly")


if __name__ == "__main__":
    main()

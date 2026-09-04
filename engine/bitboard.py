"""Attack tables. Built once at import, read as compile-time constants by every jitted function.

Leaper attacks (knight, king, pawn) are direct 64-entry lookups. Sliding attacks use magic
multipliers from `engine/magics.py`: mask the occupancy down to the squares that can block,
multiply by the square's magic, shift the product's high bits into an index, and read the answer.

Table construction is jitted, so filling ~107k entries costs milliseconds rather than the couple
of seconds pure Python would take. Correctness is *not* checked here — verification against a
reference ray-walker lives in `tools/verify_movegen.py`, because the init budget is real and a
check that runs in every game forever is a check in the wrong place.
"""

import numpy as np
from numba import njit

from engine.magics import BISHOP_BITS, BISHOP_MAGICS, ROOK_BITS, ROOK_MAGICS

U64 = np.uint64

ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def _leaper_table(deltas: tuple[tuple[int, int], ...]) -> np.ndarray:
    table = np.zeros(64, dtype=U64)
    for square in range(64):
        file, rank = square & 7, square >> 3
        attacks = 0
        for file_step, rank_step in deltas:
            f, r = file + file_step, rank + rank_step
            if 0 <= f < 8 and 0 <= r < 8:
                attacks |= 1 << (r * 8 + f)
        table[square] = attacks
    return table


def _pawn_table() -> np.ndarray:
    """Capture squares only. Index [colour, square]; colour 0 is white."""
    table = np.zeros((2, 64), dtype=U64)
    for colour, forward in ((0, 1), (1, -1)):
        for square in range(64):
            file, rank = square & 7, square >> 3
            attacks = 0
            for file_step in (-1, 1):
                f, r = file + file_step, rank + forward
                if 0 <= f < 8 and 0 <= r < 8:
                    attacks |= 1 << (r * 8 + f)
            table[colour, square] = attacks
    return table


def _slider_mask(square: int, directions: tuple[tuple[int, int], ...]) -> int:
    """Squares whose occupancy can block. Edge squares are excluded: a blocker there stops
    nothing beyond itself, so it carries no information and would only double the table."""
    mask = 0
    file, rank = square & 7, square >> 3
    for file_step, rank_step in directions:
        f, r = file + file_step, rank + rank_step
        while 0 <= f < 8 and 0 <= r < 8:
            nf, nr = f + file_step, r + rank_step
            if 0 <= nf < 8 and 0 <= nr < 8:
                mask |= 1 << (r * 8 + f)
            f, r = nf, nr
    return mask


def _slider_masks(directions: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.array([_slider_mask(sq, directions) for sq in range(64)], dtype=U64)


def _offsets(bits: tuple[int, ...]) -> np.ndarray:
    offsets = np.zeros(64, dtype=np.int64)
    cursor = 0
    for square, bit_count in enumerate(bits):
        offsets[square] = cursor
        cursor += 1 << bit_count
    return offsets


KNIGHT_ATTACKS = _leaper_table(KNIGHT_DELTAS)
KING_ATTACKS = _leaper_table(KING_DELTAS)
PAWN_ATTACKS = _pawn_table()

ROOK_MASKS = _slider_masks(ROOK_DIRS)
BISHOP_MASKS = _slider_masks(BISHOP_DIRS)
ROOK_MAGIC = np.array(ROOK_MAGICS, dtype=U64)
BISHOP_MAGIC = np.array(BISHOP_MAGICS, dtype=U64)
ROOK_SHIFTS = np.array([64 - b for b in ROOK_BITS], dtype=U64)
BISHOP_SHIFTS = np.array([64 - b for b in BISHOP_BITS], dtype=U64)
ROOK_OFFSETS = _offsets(ROOK_BITS)
BISHOP_OFFSETS = _offsets(BISHOP_BITS)

ROOK_TABLE = np.zeros(sum(1 << b for b in ROOK_BITS), dtype=U64)
BISHOP_TABLE = np.zeros(sum(1 << b for b in BISHOP_BITS), dtype=U64)

_DIR_ARRAYS = {
    "rook": np.array(ROOK_DIRS, dtype=np.int64),
    "bishop": np.array(BISHOP_DIRS, dtype=np.int64),
}


@njit("uint64(int64, uint64, int64[:, :])", cache=False)
def _ray_attacks(square: int, occupancy: U64, directions: np.ndarray) -> U64:
    """Reference sliding attacks: walk each ray, stop on and include the first blocker."""
    attacks = U64(0)
    file = square & 7
    rank = square >> 3
    for d in range(directions.shape[0]):
        file_step = directions[d, 0]
        rank_step = directions[d, 1]
        f = file + file_step
        r = rank + rank_step
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            attacks |= U64(1) << U64(target)
            if (occupancy >> U64(target)) & U64(1):
                break
            f += file_step
            r += rank_step
    return attacks


@njit(
    "void(uint64[:], uint64[:], uint64[:], uint64[:], int64[:], int64[:, :])",
    cache=False,
)
def _fill_table(
    table: np.ndarray,
    masks: np.ndarray,
    magics: np.ndarray,
    shifts: np.ndarray,
    offsets: np.ndarray,
    directions: np.ndarray,
) -> None:
    """Enumerate every occupancy subset of each square's mask and store its attack set."""
    for square in range(64):
        mask = masks[square]
        subset = U64(0)
        while True:
            index = offsets[square] + np.int64((subset * magics[square]) >> shifts[square])
            table[index] = _ray_attacks(square, subset, directions)
            subset = (subset - mask) & mask
            if subset == U64(0):
                break


_fill_table(ROOK_TABLE, ROOK_MASKS, ROOK_MAGIC, ROOK_SHIFTS, ROOK_OFFSETS, _DIR_ARRAYS["rook"])
_fill_table(
    BISHOP_TABLE,
    BISHOP_MASKS,
    BISHOP_MAGIC,
    BISHOP_SHIFTS,
    BISHOP_OFFSETS,
    _DIR_ARRAYS["bishop"],
)


# The lookups below read the tables above as frozen globals, so they must be defined after the
# tables are filled. Numba captures a global array by reference at compile time.
#
# Every return is wrapped in U64(). Numba compiles that away — the value is already uint64 — but
# njit erases the signature for mypy, so without it every lookup leaks Any into whatever calls it,
# and a genuine type error in the search would then go unreported.


@njit("uint64(int64, uint64)", cache=False)
def rook_attacks(square: int, occupancy: U64) -> U64:
    blockers = occupancy & ROOK_MASKS[square]
    index = ROOK_OFFSETS[square] + np.int64((blockers * ROOK_MAGIC[square]) >> ROOK_SHIFTS[square])
    return U64(ROOK_TABLE[index])


@njit("uint64(int64, uint64)", cache=False)
def bishop_attacks(square: int, occupancy: U64) -> U64:
    blockers = occupancy & BISHOP_MASKS[square]
    index = BISHOP_OFFSETS[square] + np.int64(
        (blockers * BISHOP_MAGIC[square]) >> BISHOP_SHIFTS[square]
    )
    return U64(BISHOP_TABLE[index])


@njit("uint64(int64, uint64)", cache=False)
def queen_attacks(square: int, occupancy: U64) -> U64:
    return U64(rook_attacks(square, occupancy) | bishop_attacks(square, occupancy))


@njit("uint64(int64)", cache=False)
def knight_attacks(square: int) -> U64:
    return U64(KNIGHT_ATTACKS[square])


@njit("uint64(int64)", cache=False)
def king_attacks(square: int) -> U64:
    return U64(KING_ATTACKS[square])


@njit("uint64(int64, int64)", cache=False)
def pawn_attacks(colour: int, square: int) -> U64:
    return U64(PAWN_ATTACKS[colour, square])

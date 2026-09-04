"""Integer NNUE inference: `(768 -> H) x 2 -> 1`, SCReLU, eight output buckets.

This is the match-time half of the network. The training half is in `training/`, which never
ships; what crosses the boundary is a single `weights/nnue.npz` of integer arrays, and
`harness/package.py` already includes `weights` by default.

Arithmetic
----------
The scales are fixed by the trainer and repeated here rather than imported, because the two
implementations existing independently is the entire point of the quantisation check::

    accumulator     int16   QA * float
    screlu          clamp(acc, 0, QA)^2   -> int32, QA^2 * float
    output weights  int16   QB * float
    output bias     int32   QA * QB * float
    eval_cp         (sum(screlu * w) // QA + bias) * SCALE / (QA * QB)

Two overflow bounds, both checked rather than assumed. The accumulator holds at most 32 pieces
times a weight the trainer clamps to +-1.98, so +-16,160 against an int16 range of 32,767. The
output sum does *not* fit: `QA^2 * 127 * 512` is 4.2e9 against an int32 limit of 2.1e9, so the
dot product accumulates in int64. Getting that wrong would be silent -- a wrapped sum is still a
plausible-looking evaluation -- which is why it is written down.

Perspectives
------------
Every position is accumulated twice, once as each side sees it, the second with colours swapped
and squares mirrored vertically (`square ^ 56`). The forward pass is fed
`[side-to-move accumulator, other accumulator]`, which makes the network colour-symmetric by
construction: a position and its mirror produce identical activations. The training data is
skewed +210 cp toward white and this is what stops that skew being learnable.

Feature indices are `piece_code * 64 + square` with the piece codes of `engine.position`, which
are the codes `training/preprocess.py` writes. The two orderings agreeing is load-bearing and not
a coincidence -- it is why neither file defines its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

I64 = np.int64
Int = int | np.int64

FEATURES = 768
BUCKETS = 8

# Re-stated rather than imported from `engine.position`. Numba freezes a global into compiled code
# as a constant, so these have to be module-level names here regardless; importing them would only
# hide that. The perft suite and the accumulator test both fail loudly if they ever disagree.
EMPTY = 12
PAWN = 0
ROOK = 3
CASTLE_KING = 2
CASTLE_QUEEN = 3
EP_CAPTURE = 5
PROMO_BIT = 8

QA = 255
QB = 64
SCALE = 400

# Where the packager puts the net: `engine/` sits one level below the zip root.
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "nnue.npz"


@njit("int64(int8[:])", cache=False)
def bucket_of(mailbox: np.ndarray) -> Int:
    """Which output head this position uses. Eight buckets over 2..32 pieces.

    Must agree exactly with `training/preprocess.py`, which computes the same thing from the same
    piece count. A disagreement would train head *n* and query head *n+1* -- and since every head
    returns a superficially reasonable evaluation, nothing would ever look broken.
    """
    pieces = I64(0)
    for square in range(64):
        if mailbox[square] != EMPTY:
            pieces += 1
    return min(I64(7), max(I64(0), (pieces - 2) // 4))


@njit("void(int8[:], int16[:, :], int16[:], int16[:, :])", cache=False)
def refresh(
    mailbox: np.ndarray,
    transformer: np.ndarray,
    transformer_bias: np.ndarray,
    accumulator: np.ndarray,
) -> None:
    """Rebuild both perspectives from the board.

    `accumulator` is `int16[2, hidden]`: row 0 is white's view, row 1 is black's. The weight rows
    are contiguous in `hidden`, so each piece contributes one linear 512-byte run per perspective.
    """
    # Element loops over hoisted 1-D views. The obvious alternative, `white_accumulator += row`,
    # was measured and is worse -- 26.8 us against 16.1 -- because numba materialises a temporary
    # for the array expression rather than fusing it into the accumulate.
    #
    # 16 us is slow for 12k int16 additions and it is not the memory: the same machine under the
    # same load runs the hand-crafted evaluation in 1.7 us. It is left slow deliberately. A full
    # refresh happens once at the root and on nothing else, because `update` below moves the
    # accumulator incrementally, touching four rows per move instead of forty-eight.
    hidden = transformer.shape[1]
    white_accumulator = accumulator[0]
    black_accumulator = accumulator[1]
    for index in range(hidden):
        white_accumulator[index] = transformer_bias[index]
        black_accumulator[index] = transformer_bias[index]

    for square in range(64):
        code = I64(mailbox[square])
        if code == EMPTY:
            continue
        flipped = code - 6 if code >= 6 else code + 6
        white_row = transformer[code * 64 + square]
        black_row = transformer[flipped * 64 + (square ^ 56)]
        for index in range(hidden):
            white_accumulator[index] += white_row[index]
            black_accumulator[index] += black_row[index]


@njit("void(int16[:, :], int16[:, :], int64, int64, int64)", cache=False)
def apply_feature(
    accumulator: np.ndarray,
    transformer: np.ndarray,
    code: Int,
    square: Int,
    sign: Int,
) -> None:
    """Add (`sign` 1) or remove (`sign` -1) one piece, in both perspectives.

    This is the whole reason the weights are stored `[features, hidden]`: each of the two rows
    touched here is a single contiguous run.
    """
    hidden = transformer.shape[1]
    flipped = code - 6 if code >= 6 else code + 6
    white_row = transformer[code * 64 + square]
    black_row = transformer[flipped * 64 + (square ^ 56)]
    white_accumulator = accumulator[0]
    black_accumulator = accumulator[1]
    if sign > 0:
        for index in range(hidden):
            white_accumulator[index] += white_row[index]
            black_accumulator[index] += black_row[index]
    else:
        for index in range(hidden):
            white_accumulator[index] -= white_row[index]
            black_accumulator[index] -= black_row[index]


@njit("void(int16[:, :], int16[:, :], int8[:], int32, int64, int64)", cache=False)
def update(
    accumulator: np.ndarray,
    transformer: np.ndarray,
    mailbox: np.ndarray,
    move: np.int32,
    captured: Int,
    side: Int,
) -> None:
    """Move the accumulator across one move. Call *after* `make_move`.

    `mailbox` is the board as it now stands, `side` is the colour that just moved and `captured`
    is the piece index it took or `EMPTY` -- which is exactly what `make_move` already wrote into
    its undo slot, so nothing here re-derives state the board did not record.

    The case analysis necessarily repeats a little of `make_move`, and duplicated case analysis is
    where engines rot. It is held honest by a test that plays random games and asserts after every
    single move that this function and a full `refresh` of the resulting board agree exactly. A
    divergence in any branch -- en passant, either castle, any of the eight promotions -- fails
    immediately rather than quietly costing a few centipawns forever.
    """
    from_square = I64(move) & I64(63)
    to_square = (I64(move) >> I64(6)) & I64(63)
    flag = (I64(move) >> I64(12)) & I64(15)
    us = 6 * side

    # What now stands on the destination: the promoted piece after a promotion, else the mover.
    arrived = I64(mailbox[to_square])
    # What left the origin. Only a promotion changes identity in transit.
    moved = us + PAWN if flag & PROMO_BIT else arrived

    if captured != EMPTY:
        if flag == EP_CAPTURE:
            # The pawn taken en passant is not on the destination square but behind it.
            taken_square = to_square - 8 if side == 0 else to_square + 8
        else:
            taken_square = to_square
        apply_feature(accumulator, transformer, captured, taken_square, -1)

    apply_feature(accumulator, transformer, moved, from_square, -1)
    apply_feature(accumulator, transformer, arrived, to_square, 1)

    if flag == CASTLE_KING:
        # King steps two toward the h-file; the rook jumps from the corner to just behind it.
        apply_feature(accumulator, transformer, us + ROOK, from_square + 3, -1)
        apply_feature(accumulator, transformer, us + ROOK, from_square + 1, 1)
    elif flag == CASTLE_QUEEN:
        apply_feature(accumulator, transformer, us + ROOK, from_square - 4, -1)
        apply_feature(accumulator, transformer, us + ROOK, from_square - 1, 1)


@njit("int64(int16[:, :], int16[:, :], int32[:], int64, int64)", cache=False)
def forward(
    accumulator: np.ndarray,
    output: np.ndarray,
    output_bias: np.ndarray,
    stm: Int,
    bucket: Int,
) -> Int:
    """SCReLU both halves, dot with the bucket's output row, return centipawns.

    The result is side-to-move relative, positive meaning good for whoever is to move, which is
    what the search expects and what the trainer was given as a target.
    """
    hidden = accumulator.shape[1]
    # Side to move first. `stm` is 0 for white, and row 0 is white's view. 1-D views for the same
    # reason as in `refresh`.
    us = accumulator[stm]
    them = accumulator[1 - stm]
    weights = output[bucket]

    total = I64(0)
    for index in range(hidden):
        value = I64(us[index])
        if value < 0:
            value = I64(0)
        elif value > QA:
            value = I64(QA)
        total += value * value * I64(weights[index])

    for index in range(hidden):
        value = I64(them[index])
        if value < 0:
            value = I64(0)
        elif value > QA:
            value = I64(QA)
        total += value * value * I64(weights[hidden + index])

    total = total // QA + I64(output_bias[bucket])
    return (total * SCALE) // (QA * QB)


@njit(
    "int64(int8[:], int64, int16[:, :], int16[:], int16[:, :], int32[:], int16[:, :])", cache=False
)
def evaluate(
    mailbox: np.ndarray,
    stm: Int,
    transformer: np.ndarray,
    transformer_bias: np.ndarray,
    output: np.ndarray,
    output_bias: np.ndarray,
    scratch: np.ndarray,
) -> Int:
    """Refresh and score in one call. The reference implementation.

    Every optimisation that follows -- the incremental accumulator above all -- is correct exactly
    insofar as it agrees with this function on the same position, so it stays in the file after it
    stops being the fast path.
    """
    refresh(mailbox, transformer, transformer_bias, scratch)
    return forward(scratch, output, output_bias, stm, bucket_of(mailbox))


class Network:
    """The loaded weights, plus the scratch accumulator the jitted code writes into.

    Loading is tolerant of the file being missing: the hand-crafted evaluation has to stay
    shippable on its own, and an engine that refuses to start because a net it does not yet have
    is absent would be a worse failure than one that plays slightly weaker chess.
    """

    def __init__(self, path: Path = WEIGHTS_PATH) -> None:
        self.available = path.exists()
        if self.available:
            with np.load(path) as data:
                self.transformer = np.ascontiguousarray(data["transformer"], dtype=np.int16)
                self.transformer_bias = np.ascontiguousarray(data["transformer_bias"], np.int16)
                self.output = np.ascontiguousarray(data["output"], dtype=np.int16)
                self.output_bias = np.ascontiguousarray(data["output_bias"], dtype=np.int32)
                self.hidden = int(data["meta"][0])
        else:
            # A zero net evaluates every position as 0. It exists so the signatures below can be
            # compiled at import even when no weights are present, which keeps the JIT cost of a
            # build measurable before the net lands.
            self.hidden = 256
            self.transformer = np.zeros((FEATURES, self.hidden), dtype=np.int16)
            self.transformer_bias = np.zeros(self.hidden, dtype=np.int16)
            self.output = np.zeros((BUCKETS, 2 * self.hidden), dtype=np.int16)
            self.output_bias = np.zeros(BUCKETS, dtype=np.int32)

        if self.transformer.shape != (FEATURES, self.hidden):
            raise ValueError(
                f"transformer is {self.transformer.shape}, expected {(FEATURES, self.hidden)};"
                " the trainer stores [features, hidden] so a feature is one contiguous row"
            )
        if self.output.shape != (BUCKETS, 2 * self.hidden):
            raise ValueError(
                f"output is {self.output.shape}, expected {(BUCKETS, 2 * self.hidden)}"
            )

        self.scratch = np.zeros((2, self.hidden), dtype=np.int16)

    def evaluate(self, mailbox: np.ndarray, stm: int) -> int:
        return int(
            evaluate(
                mailbox,
                I64(stm),
                self.transformer,
                self.transformer_bias,
                self.output,
                self.output_bias,
                self.scratch,
            )
        )


def warm() -> None:
    """Force compilation of every jitted function here. `notes/invariants.md`: the JIT bill is paid
    at the start of every single game, so a function left uncompiled until the first en passant
    capture would stall the clock mid-game instead."""
    network = Network()
    mailbox = np.full(64, EMPTY, dtype=np.int8)
    mailbox[4] = 5
    mailbox[60] = 11
    network.evaluate(mailbox, 0)
    apply_feature(network.scratch, network.transformer, I64(0), I64(8), I64(1))
    apply_feature(network.scratch, network.transformer, I64(0), I64(8), I64(-1))
    update(
        network.scratch,
        network.transformer,
        mailbox,
        np.int32(4 | (12 << 6)),
        I64(EMPTY),
        I64(0),
    )
    forward(network.scratch, network.output, network.output_bias, I64(0), I64(0))
    bucket_of(mailbox)

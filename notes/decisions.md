# Decisions

Locked calls and the evidence behind them, so we don't re-litigate mid-week. If new evidence
overturns one, amend the entry with a date rather than deleting it.

---

## Engine core in numba, not python-chess — LOCKED

Magic bitboards, packed int32 moves, preallocated NumPy arrays for the move stack. python-chess is
used only at the root as a legality guard, and in offline tooling.

python-chess does ~10–40k nps; a magic-bitboard numba engine does 300k–1M nps in real search
(`black_numba` hits 7–8 Mnps in perft). That is 3–5 extra ply and is the single biggest lever we
have. The modal competitor entry is python-chess + alpha-beta + piece-square tables at depth 4–5.

## Evaluation: `(768 → 256)×2 → 1`, SCReLU, 8 output buckets, int16 — LOCKED

~394 KB. Explicitly **not** HalfKP 41024×256×2.

Four objections to HalfKP, and note that **only the first is about training**, so cloud GPU
capacity does not rescue it:
1. 41024 features are data-starved without a GPU farm.
2. A 21 MB weight table has poor cache locality on one core — and the rules put both agents on the
   same machine, so we share L3 with an opponent also hammering memory. 394 KB stays hot; 21 MB
   does not.
3. King moves force full accumulator refreshes over that table. With plain 768 inputs there is
   **no king refresh at all** — every move is a pure incremental update.
4. numba cannot emit the hand-tuned int8 AVX2 intrinsics HalfKP performance depends on.

The one sanctioned scaling axis is **width**: 512 (787 KB) and 1024 (1.6 MB) are still
cache-resident with zero king refreshes, and width is one hyperparameter on an unchanged pipeline
rather than a new feature geometry. **Burden of proof sits on the bigger net** — ship 768→256
working end to end first, then let SPRT argue us upward.

## NNUE inference in numba int16, never onnxruntime per node — LOCKED

ORT `run()` costs ~20–40 µs per call, capping us near 40k evals/sec — dominant from the first
move. A numba int16 incremental accumulator plus forward pass is ~3–8 µs, i.e. 125k–330k
evals/sec.

This **contradicts `docs/IDEAS.md`**, which recommends ONNX plus batching. That advice suits a
small non-incremental net, not a true NNUE. Batching also breaks alpha-beta outright, which cannot
know which leaves to evaluate until earlier ones cut.

## Training data: Lichess evaluations database — LOCKED

`https://database.lichess.org/lichess_db_eval.jsonl.zst`, 21,681,515,630 bytes compressed,
394,669,566 unique Stockfish-evaluated positions, **CC0** (public domain, no attribution, no
share-alike). Stream with zstandard; never expand to disk (~150 GB uncompressed).

Legal because the canonical rules say training data is unrestricted, including positions annotated
by an existing engine — what ships inside the zip is what the ban covers. We ship weights only,
from random initialisation.

Rejected: generating our own data locally with an engine. Four cores and no CUDA GPU make public
CC0 dumps strictly better.

---

## Considered and rejected

- **HalfKP 41024×256×2.** Three of its four problems are match-time properties of the shipped net
  that more training cannot fix. See above.
- **Per-node or batched onnxruntime.** See above.
- **King-bucketed / horizontally-mirrored inputs.** Only worthwhile with far more data than 394M
  positions split eight ways, and it is a fifth net to train and test in a week where search
  tuning is already the acknowledged shortfall.
- **Harvesting curated opening positions from our own ladder games into a shipped book.** Depends
  on rated-game FENs being visible on the dashboard, which is unverified; the curated set size is
  unknown so coverage is unknown; positions are "close to level" so a book buys a few plies rather
  than an advantage; it competes with search tuning, which has certain Elo; and it is the only
  item sitting in a rules grey zone. Revisit only if we are ahead on Tuesday **and** rated FENs
  turn out to be readable.
- **Shipping a precompiled numba cache.** `.nbc` files are native object code — rejected as native
  binaries — and `/tmp` is wiped per game, so it would not carry across games anyway.
- **Syzygy tablebases.** 3-4-5-man WDL is ~380 MB against a 50 MB cap. Self-generated KPK/KRK/KQK
  bitbases are tens of KB and are worth it; general tablebases are not.
- **Opening book keyed on move one.** Rated games do not start from the standard position.

---

## 2026-09-04 — JIT budget is not the binding constraint — MEASURED

Measured full-engine init projection ~9.7s, ~30s at a 3x margin, against a 60s target and 90s hard
limit. Details in [measurements.md](measurements.md).

The plan had been holding search sophistication back on the assumption that per-game compile cost
would crowd out the 90s init budget. It does not. **Search features are gated by our time to build
and test them, not by compile cost.** Null move, LMR, futility, aspiration windows, SEE and a full
quiescence search are all affordable.

This does not relax the discipline that produced the number: eager explicit signatures on every
jitted function, per-function compile timing re-measured after every change, no `parallel=True`.
The documented failure mode is a *single* pathological function — forced inlining, an accidental
second type specialisation — going from 30s to 100s on its own.

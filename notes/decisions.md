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

## 2026-09-04 — Testing strategy: relative SPRT locally, the dashboard ladder for absolute — LOCKED

**Correction to an earlier claim in the plan: there is no CCRL involvement in this competition and
there are no rated house bots.** Verified against the canonical rules page. Rated games and the
final Swiss are played against *other teams' agents*; the only rating that exists is a Glicko-2
ladder computed from those games. There is no external anchor to calibrate against.

Two layers, neither needing an external ladder:

1. **Local: relative SPRT against our own frozen previous builds** (`tools/sprt.py`). This drives
   every development decision. "Did this change help?" needs no absolute anchor, and a paired A/B
   over the same openings with colours reversed has far tighter error bars than any absolute
   rating estimate. **Keep every shipped version permanently as a gauntlet opponent** — v1 HCE,
   v2 NNUE, and so on. Real spacing, no cost, no rules ambiguity.
2. **Absolute: the dashboard ladder.** Rated rounds run hourly, 08:00–22:00, from 4 Sep 08:00.
   Actual hardware, actual 120s + 0.5s control, actual curated openings, actual field. Strictly
   more informative than any CCRL proxy.

**The bundled baselines are regression smoke tests, not rating anchors.** `greedy` is a 1-ply
material search, `minimax` is 2-ply. After Phase 2 we beat them ~100%, and a 100% score carries no
information — you cannot measure Elo above an opponent you never lose to. They remain useful only
to answer "did I break something badly enough to lose to minimax?"

**Rejected: a local Stockfish gauntlet as an absolute anchor.** It would be legal — the ban covers
what ships in the zip, not the test rig — but CCRL and `UCI_Elo` figures are calibrated at long
time controls on dedicated hardware and do not transfer to bullet on one shared core. Decisively,
it changes no decision: we ship version N+1 if SPRT says it beats N, whether we are objectively at
2200 or 2600. An afternoon of work for a number we would act on identically.

**Consequence for uploads.** The ladder is not cosmetic. Qualification is the locked-build Swiss,
but ladder rating **seeds** that Swiss, and seed order sets invite order for a 50-seat room. Every
hour without a validated submission is free measurement against the real field, discarded.

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

---

## 2026-09-04 — Training hardware: CPU only, so a run is an overnight job — MEASURED

`torch.cuda.is_available()` is **False** and there is no NVIDIA device on the machine; the project
venv is pinned to `torch 2.13.0+cpu`, which is the competition's own pin. Eight cores. Measured
throughput is ~33k positions/s uncontended, so the default 60,000 steps at batch 16,384 is ~983M
positions and **roughly eight hours**.

Consequences, none of them optional:

- **A training run is started at night and read in the morning.** There is no iterate-in-an-hour
  loop available. Between 4 Sep and the 11 Sep deadline that allows a handful of runs, not dozens,
  so the width A/B (128 / 256 / 512 / 1024) cannot be run exhaustively on this machine.
- **Colab remains worth asking for.** It needs the user's Google account, so it is not automatable,
  but a free T4 turns eight hours into two and is the only route to testing more than one width.
  The laptop run is the fallback that always happens regardless.
- **Training and SPRT contend.** Both saturate the CPU, and SPRT is timing-sensitive in a way
  training is not. They are not run together; SPRT waits for a quiet machine.

Contention is large enough to invalidate absolute timings taken during a run: with preprocessing on
nine cores, `import engine.search` measured 93s against its usual 30-34s. Any init or nps figure
taken while another job is running is worthless.

## 2026-09-04 — Feature transformer ships as [features, hidden] — MEASURED

The accumulator adds and subtracts one weight row per changed feature, so a feature must be a
contiguous run. `quantise` had transposed to `[hidden, features]`, putting a feature on a
`hidden * 2`-byte stride — one cache line per element — under a comment asserting the opposite.
Torch's `EmbeddingBag` already stores `[features, hidden]`, so the correct code is less code.

The general lesson is the one worth keeping: **the comment claimed a property the array did not
have, and nothing tested the claim.** Layout assumptions are invisible to correctness tests — the
transposed version computed identical evaluations, just slowly. It was caught by writing the engine
side against the documented layout and finding the shapes disagreed.

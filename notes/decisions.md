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

## 2026-09-05 — Width stays at 256; the A/B is resolved — MEASURED

The plan deferred `128 / 256 / 512 / 1024` to measurement in our own engine, with the burden of
proof on the bigger net. Both halves are now measured and the burden is not discharged.

`tools/eval_quality.py` on reserved holdout shards: WDL MAE 0.0656 / 0.0630 / 0.0602 / 0.0576.
Wider is better monotonically, with no saturation.

`tools/net_speed.py` at fixed depth: 1.00x / 1.35x / 1.75x / 3.09x time, at a measured EBF of 2.19,
so 0.38 / 0.71 / 1.44 plies given up. Evaluation gains are sublinear in width, cost is superlinear.
512 and 1024 lose clearly. 128 versus 256 is inside the noise; 256 wins on the trustworthy node
column, on being the locked call, and on the fact that further search work makes plies cheaper and
evaluation relatively dearer.

Understated cost, in our favour: the measurement is single-process, but a real game shares L3 with
an opponent hammering memory, which penalises the 1.6 MB net far more than the 393 KB one.

This is decided on a proxy plus a speed measurement, not on games, because the SPRT is still blocked
on memory. It is the strongest evidence obtainable on this machine. 128 is the named fallback if
init-budget pressure later forces a cut. See `notes/measurements.md`, 5 Sep.

## 2026-09-05 -- The NNUE build is the shipping build -- MEASURED

**Decision: the net ships. The hand-crafted eval is demoted to fallback.**

200 games at 30s + 0.125s against the byte-identical no-weights build: **+119 =17 -64, Elo
+98.1 +- 48.1**, LLR +2.61 against a +-2.94 bound. Formally inconclusive, practically settled -- a
95% interval of roughly [+50, +146] does not admit a reading in which the net is not better. Phase 3
made adoption conditional on beating HCE; that condition is met, and by a margin well beyond the
+15 Elo H1 the test was framed around.

We are not chasing a formal ACCEPT, and not because of cost. The batch cannot be extended -- see
`notes/measurements.md` for why seeds make batches independent rather than cumulative -- so no
amount of further play moves *this* test's LLR off 2.61. A second batch would be an independent
replication of a conclusion that is not in question. The budget is better spent on the search-side
changes that still have no verdict at all, and at 283 runner-minutes a run there is room for
several of them.

Consequences:
- HCE stays in the tree and stays correct. It is the fallback if the net ever fails validation on
  the platform, and it is the reference the next net is measured against.
- `weights/nnue.npz` is now load-bearing. Removing it silently downgrades the agent by ~100 Elo
  rather than failing, which is exactly the failure mode packaging must not have -- keep asserting
  `use_nnue True` from inside the extracted zip, never from the source tree.

**Corollary decision: SPRT runs in CI from now on, not locally.** The dev box plays one game at a
time and took ~12 hours for a verdict this run reached in 16 minutes. Games are clock-bound, so
parallelism is the only lever that exists. Local runs are for smoke tests; verdicts come from
`.github/workflows/sprt.yml`.

## 2026-09-06 -- Pondering ships, and it ponders the parent -- MEASURED

`+63.2 +- 23.8` Elo, ACCEPTED at LLR +5.98 over 400 games (run 33999768084). See
`notes/measurements.md` for the full result. This is the largest measured gain of the week.

Two things are locked by that number, not just one:

**Pondering is in.** `AGENTS.md` line 33 permits it explicitly -- the process keeps its core while
the opponent thinks -- and roughly half a bullet game's wall time was previously spent with that
core idle.

**We ponder the position after our own move, not a predicted reply.** This was the open question and
the measurement settles it in favour of the parent. Searching a guessed reply concentrates the work
on one line and wastes all of it when the guess is wrong; searching the parent means every legal
reply is a child of what was searched. Do not "improve" this later by adding a PV-based ponder-move
guess -- that is the version this beat.

Rejected along the way: joining the ponder thread with a timeout. We have one core, so continuing
while a second search still holds it is the one outcome worse than never pondering. The join is
unbounded on purpose.

**Amended 2026-09-06, later the same day -- REVERSED. Pondering is out.** The +63.2 Elo above is a
phantom gain: it is real in our own harness and impossible on the real platform. Canonical platform
docs (which `AGENTS.md:33` predates and contradicts) say the opposite of what is written above --
"your process is suspended on the opponent's turn, background threads won't run" and "work you leave
running between own moves not run". Our harness does not suspend anything, so a ponder thread ran
locally and scored exactly the Elo a free half-a-game of thinking time would predict. It cannot
occur in a rated game. `agent.py` ships with no pondering and an explicit docstring warning against
re-adding it on the strength of any future local measurement -- see `notes/measurements.md`, "6 Sep
-- thinking time worth 52.5 Elo per halving, pondering hiding it", and run 34033781905 (half time,
pondering removed: -52.5 +- 20.1, REJECTED, LLR -8.91), which is the evidence that the baseline this
entry measured against was itself confounded by pondering. Left in place above rather than deleted,
per this file's own convention of amending over erasing.

## 2026-09-09 -- Transposition table stays at 22 bits -- LOCKED

`TT_BITS` had never been tuned since it was first set. Quadrupling it to 24 bits (~319 MB, still
comfortably inside the 2 GB budget) was tested in isolation, on its own branch, against the
theory that a bullet game at the real control turns over many multiples of a 4M-entry table and a
bigger one would cut collisions.

**REJECTED by SPRT: -2.1 +- 13.5 Elo over 1000 games, LLR -3.02** (run 34303293131, seed 19, 9 Sep,
candidate `73d92fe` vs baseline `7895372`). Reverted to 22 bits. See `notes/measurements.md`, "9 Sep
-- transposition table 22 -> 24 bits: REJECTED", for the full number and reasoning.

Do not re-try this without new evidence. The node counts a 4M-entry table actually sees at 120s +
0.5s were evidently not the bottleneck the headroom-based reasoning assumed -- a useful calibration
on how much to trust "more of a resource is usually better" without a measurement behind it.

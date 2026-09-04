# Measurements

Append-only. Every JIT timing, nps figure and SPRT result, dated, with the machine it ran on.
"Did that change help?" must be answerable from this file, never from memory.

Machine unless stated otherwise: Windows 11, local laptop, `.venv` Python 3.12.14,
numba 0.67.0, numpy 2.5.2, python-chess 1.11.2. **Match containers are Linux on one dedicated
core** — these numbers are indicative, not authoritative. The platform validation log is ground
truth.

---

## 2026-09-04 — numba JIT compile cost (Phase 0, item 3)

The question this had to answer: does numba compile time cap how sophisticated the search can be?
`/tmp` is wiped per game and `.nbc` caches are native object code we cannot ship, so **every game
pays full compilation**. Budget is 90s hard, 75s working cap, 60s target.

### Small kernels — `tools/jit_timing.py`, 12 functions

| function | compile |
|---|---|
| popcount | 1.46s |
| forward_screlu | 0.40s |
| generate_moves | 0.35s |
| accumulator_update | 0.24s |
| order_moves | 0.23s |
| magic_lookup | 0.21s |
| zobrist_update | 0.18s |
| tt_probe | 0.17s |
| see | 0.16s |
| ray_attacks | 0.14s |
| make_unmake | 0.13s |
| lsb_scan | 0.07s |
| **total** | **3.76s** |

`popcount` at 1.46s is not an expensive function; it is first, and absorbs numba's one-time backend
warm-up. Marginal cost of a small kernel is **~0.2s**.

### Large functions — `tools/jit_stress.py`

Function count is the wrong unit: compile cost tracks body size and branchiness. These two proxies
are the shapes that actually dominate a real engine's init.

| function | compile |
|---|---|
| `generate_all` — full pseudo-legal movegen (6 piece types, castling, ep, promotions, ~120 lines) | 1.78–1.80s |
| `alphabeta` — recursive search with TT probe, null move, LMR, futility, history ordering | 2.94–4.48s |
| **total** | **4.74–6.26s** |

Two runs, hence the ranges. `generate_all` is stable; `alphabeta` varied by 1.5s between runs, so
treat single measurements of it as ±50% and use the upper figure when budgeting.

Recursion was the specific worry (type inference on a self-referential signature). Worst observed
4.48s. Not a problem.

### Fixed import cost

| module | import |
|---|---|
| numpy | 0.29s |
| numba | 0.88s |
| chess | 0.04s |

### Verdict

Realistic full-engine init projection: **~11s** (1.2s imports + 3.8s kernels + 6.3s large
functions, using the worst observed run). Even at **3x** for a finished engine with more and
larger functions, ~33s — against a 60s target and a 90s hard limit.

**The JIT budget is not the binding constraint on search sophistication.** This removes the main
reason the plan held back on search features. Consistent with `black_numba` at 10–30s and
`Antares` at 20–45s, both of which are complete engines.

Caveats that keep this honest:
- Windows local, not the Linux match container. Re-measure against the platform validation log.
- The proxies are structurally right but not the real engine. Re-run `tools/jit_timing.py` after
  every change that adds or alters a jitted function; compile time creeps and blowing init is a
  loss in *every* game, not a gradual weakening.
- `parallel=True` stays banned — pure compile cost, no payoff on one core.
- Do not import torch or onnxruntime at runtime; both cost seconds for nothing.

---

## 2026-09-04 — perft suite established (Phase 0, item 4)

`tools/perft.py`, 20 positions: the 6 canonical Chess Programming Wiki positions plus martin
sedlak's TalkChess edge-case suite (illegal en passant for both colours, en passant delivering
check, castling that gives check, castling rights blocked, discovered check, promotion out of
check, under-promotion giving check, self-stalemate, stalemate-vs-mate).

All 20 pass exactly at depth 4 against the python-chess reference backend. The `engine` backend is
a stub until `engine/` exists.

**The suite caught two bugs in its own first draft.** Two hand-entered expected counts (`ep_pin`,
`castle_check`) were wrong, and a third position was paired with a different case's published
number. Every ladder now in the file was generated with python-chess and its deepest entry
cross-checked against the published TalkChess count — all nine agreed exactly. A wrong expected
count is worse than no test: it sends you hunting a bug that does not exist. Do not hand-edit
those numbers.

python-chess reference throughput on perft: **0.15–0.9 Mnps**. That is the number the numba
movegen has to beat by an order of magnitude.

## 2026-09-04 — SPRT arena built and smoke-tested (Phase 0, item 4)

`tools/sprt.py`. 22 curated openings played out ~8 plies (Ruy Lopez through Reti), each played
twice with colours reversed so a lucky line helps both sides equally. Trinomial LLR as used by
Fishtest, early stopping on the bounds, Elo with error bars.

Smoke test — greedy vs random, 3s + 0.05s, concurrency 4:
**accepted after 18 games in 0.1 min, +17 =1 -0, ~320 games/min.**

Two things this pinned down:

1. **Throughput.** ~320 games/min at 3s+0.05s with 4 concurrent games. At the real 120s+0.5s
   control that falls by roughly 40x, so short clocks are the unit for iterating during the day
   and an overnight run is the unit for confirming at the real time control.
2. **Concurrency ceiling.** 8 logical cores, and every game runs *two* agent processes, so the
   safe default is 4 concurrent games. Oversubscribing silently distorts any clock-based result,
   in favour of whichever engine uses less time. `tools/sprt.py` defaults to `cores // 2` and
   warns when overridden above it.

A bug the smoke test found: with zero losses the raw variance estimate is degenerate, the LLR
pinned at 0.00, and a run against a much weaker opponent would never terminate. Fixed by adding
half a game to each bucket when the sample is one-sided — conservative, so it can only delay a
verdict, never manufacture one.

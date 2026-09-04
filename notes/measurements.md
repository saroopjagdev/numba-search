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

---

## 2026-09-04 — Phase 1 movegen: init cost and perft throughput

### Import + JIT, measured end to end

| stage | cold | warm |
|---|---|---|
| numpy + numba | 3.62s | 0.98s |
| `engine.bitboard` (JIT + fill of 107,648 entries) | 4.83s | 1.32s |
| `engine.position` (JIT) | 10.33s | 9.64s |
| **total** | **18.77s** | **11.94s** |

**Correction to a figure quoted earlier in the session: `engine.position` does not cost 46.5s.**
That was a single cold first-ever compile with a OneDrive/Defender scan in the way, and it did not
reproduce. Steady state is ~19s cold and ~12s warm against a 90s hard limit and a 60s target, so
init is not the binding constraint on Phase 1 and there is room left for search and NNUE.

Per-function compile cost within `engine.position` (10.20s total):

| function | compile |
|---|---|
| generate_moves | 3.52s |
| make_move | 1.87s |
| perft | 1.49s |
| unmake_move | 1.09s |
| is_attacked | 0.53s |
| in_check | 0.48s |
| refresh_occupancy_jit | 0.24s |
| eight small helpers combined | 0.96s |

`perft` is 1.49s of that and will not ship, so the shipped figure is ~8.7s.

### perft throughput

1.4–2.1 Mnps through the real make/unmake path. There is deliberately no bulk counting at depth 1:
skipping make/unmake on the last ply would stop testing the code most likely to be wrong. Against
python-chess at 0.15–0.9 Mnps on the same suite, this is roughly 3–10x on the reference's own best
case and far more on its typical one.

### Bug found by the very first perft run

Startpos depth 5 returned 4,865,644 against the true 4,865,609 — 35 nodes too many. `divide`
against python-chess narrowed it to six pawn moves, all of which open a line toward e1. Cause:
`is_attacked` indexed the pawn-attack table by the *attacking* colour. The reverse-colour trick
inverts it — a square is attacked by a black pawn exactly when a white pawn standing on that
square would attack the black pawn — so the index must be the defending colour, `1 - by_black`.

This is precisely the failure the plan named in advance: an illegal move generated rarely enough
that ordinary play would never surface it. One perft run found it; one line fixed it.

### Gate results — Phase 1 movegen is verified

`tools/perft.py --backend engine --depth 5` over the full 20-position suite: **97 of 97 depth
checks exact, all positions correct.** Kiwipete and positions 3-6 included, along with every
en-passant, castling-through-check, promotion and stalemate edge case in the suite.

`--suite startpos --depth 6`: **119,060,324 in 95.85s (1.24 Mnps)** — the exact figure the plan
named as the Phase 1 gate.

### Zobrist and make/unmake — the checks perft cannot make

Added to `tools/verify_movegen.py`. Perft never reads the hash key, and the only make/unmake
asymmetries it notices are the ones that change the legal move count, so both were unverified.
The walk compares the incremental key against a from-scratch recomputation at every node and
snapshots the whole position to confirm unmake restores it bit for bit, over four positions
chosen to cover lost castling rights, an en-passant file appearing and disappearing, and
promotions: **121,343 nodes, all correct, 20.2s.**

A drifting key would not have shown up as a hash bug. It would have shown up weeks later as the
transposition table returning another position's score, which reads as a search bug and is far
more expensive to chase.

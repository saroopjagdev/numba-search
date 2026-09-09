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

---

## 2026-09-04 — Phase 2: hand-crafted eval, search, and the first real agent

`engine/eval.py` (PSQT + mobility + king safety + pawn structure, tapered over 24 phase units),
`engine/search.py` (iterative deepening, Zobrist TT, quiescence, null move, LMR, futility,
aspiration windows, killers, history) and `agent.py` wired to both.

### Strength

`tools/sprt.py --candidate . --baseline baselines/minimax --base-ms 8000 --increment-ms 100`:
**ACCEPTED after 15 games, +15 =0 -0.** Elo bar is meaningless with zero losses; what the run
establishes is that the baseline never once drew or won.

Three games at the *real* 120s + 0.5s control through `harness/play.py`: won as white against
minimax, as black against minimax, and as white against greedy — **all three by checkmate**, none
by flag, adjudication or opponent failure.

Win At Chess subset, 20 positions at a 1000 ms budget: **13–15 solved**, depth 10–20 in the
non-mate positions, mate found instantly in five. The 13-to-15 spread is run-to-run variance at a
fixed clock, not a change in the engine — do not read a two-position difference as a regression.

### Init cost

**~30–34s for `import agent`, which includes the deliberate warm-up search.** Against a 90s hard
cap and the 60s working target this is comfortable but no longer trivial: Phase 1 alone was ~12s
warm, so eval and search together cost ~20s of compilation. The NNUE will add more. Re-measure
after every jitted function added, and if the cap is approached, cut features rather than margin.

The warm-up runs on kiwipete, not the start position, and that is a deliberate choice: kiwipete has
castling for both sides, en passant reachable, promotions one push away and enough material for the
null-move and LMR branches to be taken, so a shallow search over it compiles the whole engine.
Warming up from `startpos` would leave several branches to compile on move one, out of the clock.

### Clock discipline

The measurement that matters, because overrunning loses a game outright while under-spending costs
a few centipawns. Ten positions × five budgets from 30 ms to 2000 ms:

| budget | mean | max |
|---|---|---|
| 30ms | 0.75x | 1.03x |
| 100ms | 0.84x | 1.28x |
| 300ms | 0.74x | 1.16x |
| 1000ms | 0.63x | 1.02x |
| 2000ms | 0.58x | 1.14x |

**Worst observed overrun 1.28x of the search's own budget.** `agent.py` then applies a 0.85 safety
factor and caps any single move at 25% of the remaining clock, so the worst case is ~27% of
remaining on one move. It cannot flag.

Getting there took two fixes, both found by measuring rather than reasoning:

1. **The aspiration re-search escaped the clock.** A depth was treated as one indivisible unit, but
   a fail-low or fail-high starts a *new* jitted call, and the clock was read only before the
   first. WAC.008 overran a 1000 ms budget by 69%. Every call now gets its own clock read.
2. **The nps estimate divided one iteration's nodes by the whole search's elapsed time**, which
   understates it by roughly the branching factor and starved every later budget. Now measured
   per call.

The node budget has two independent caps. The time-based one is `remaining x nps x 1.2`; the other
is `6 x last_nodes`, which does not depend on nps at all. The second exists because the nps
estimate can be wrong in the dangerous direction — a shallow TT-saturated call reports millions of
nodes per second, and the next real iteration would then be handed a budget taking seconds to
spend.

### Two bugs worth remembering

**A segfault, not an exception.** `TT_MASK` was a module constant baked in at compile time, so
`new_tt(bits=16)` produced a table indexed to 22 bits. Inside nopython code that is not an
IndexError, it is a silent out-of-bounds write and a hard crash. The mask now comes from
`tt_key.shape[0]`. Anything indexing a numba array from a constant is the same trap.

**Unbounded ply.** The check extension can lengthen a line indefinitely when checks keep coming,
and every buffer is indexed by ply. Same failure mode: no bounds check, straight to a segfault.
`MAX_SEARCH_PLY = 120` caps it, with headroom below `MAX_PLY` for the quiescence tail.

### Repetition

The search scans its own path *and* the game history, because `harness/play.py` calls
`board.outcome(claim_draw=True)` — the referee claims threefold for us, so an engine blind to game
history can shuffle a won game into a draw. The agent only ever observes positions where it is to
move, which are exactly the ones at even offsets from the root, so the walk reaches into game
history only at even plies.

### Lichess evaluations database — download complete

`C:\Users\ssjag\chessdata\lichess_db_eval.jsonl.zst`, **21,681,515,630 bytes**, matching the
published size. 394,669,566 positions, CC0. The NNUE track is unblocked.

## Rated rounds 12 and 13 (4 Sep) — first games played by the real engine

Both won by checkmate. Round 12 as black vs "Minimax Three" from a Ruy Lopez Closed at move 10;
round 13 as white vs "Chess" from a Caro-Kann Classical at move 6. Rounds 10 and 11 were losses,
but those were played by the 781-byte random mover that was still the live build at the time.

### Init on the platform is half what it is locally — the most useful number so far

| | Ready in | Of the 90 s budget |
|---|---|---|
| Round 12, machine `w-75c226` | **16.2 s** | 18 % |
| Round 13, machine `w-2b4afc` | **19.7 s** | 22 % |

Local Windows measures 30–34 s for the same import. The container JITs roughly twice as fast, and
the spread between the two machines is about 20 %. Against a 75 s working cap that leaves **~55 s
of headroom**, so the NNUE's compile cost is not the constraint Phase 3 was budgeted around. Local
timings stay what we gate on — they are the pessimistic number — but the cap is no longer close.

### Clock

Slowest moves were 3.9 s and 4.0 s against a predicted budget of 3.7 s at a 120 s clock: 1.08x
including the harness round-trip, well inside the margin. Neither game came near a flag.

### Start FENs are visible in the match log

`Start FEN` appears in the header of both logs, at ply 19 and ply 11 of named opening lines. That
resolves the precondition the shelved opening-book item was gated on. Still shelved — it competes
with search work that has certain Elo and sits in a rules grey zone — but it is now known rather
than assumed.

### The bug the logs exposed: eight moves a game thrown away

Round 12 used 24.7 s of 120 s over 22 moves, with eight moves returning in 0.0–0.1 s. Replaying the
game locally reproduced it exactly, and the cause was not that those positions were easy.

The node budget's second cap is `6 * last_nodes + 50_000`, sized from the last completed call so a
wrong nps estimate cannot blow the clock. After the opponent plays the move we predicted, the
transposition table answers the early depths in a few hundred nodes, so `last_nodes` stays near
zero and the cap sits at its 50k floor while the position itself needs millions. The first depth
that has to do real work trips the cap and sets the abort flag — and the driver treated any abort
as "time is up" and ended the whole search. The result was a depth-12 move played with 3.5 s still
on the clock.

The fix distinguishes the two aborts: if the call died having used less than a quarter of the time
remaining, it was the cap and not the clock, so the cap is multiplied by four and the same depth
searched again. The clock read at the top of the loop is what actually terminates the search.

Replay of round 12 with the fix, over the eight affected moves:

| | before | after |
|---|---|---|
| time spent on those moves | 0.07–0.15 s | 1.95–4.14 s |
| depth reached | 10–16 | 13–23 |
| mate first seen | move 19 | **move 18**, at depth 23 |

Clock discipline over the ten-position, five-budget sweep **improved**: worst overrun 1.28x →
1.20x. The retry can only fire when a call finished inside a quarter of the time left, so it is
never the call that overruns.

## Phase 3, part one: the training data (4 Sep)

### Three facts about the source, all established by measurement

Each of these is silent if wrong, which is why none was taken on trust.

**`cp` is white-relative, not side-to-move.** Multi-PV lines are ordered best-first for the side to
move, so white-relative scores must run ascending when black is to move. Over 200,000 sampled
records: black to move **86,599 ascending against 3 descending**, white to move 92,350 descending
against 3. Confirmed independently against material balance — the sign of the score agrees with
the sign of material in **74.6%** of positions where both are non-zero, where the wrong perspective
would have put that near 25%.

**`depth >= 20` keeps 91.3%** of positions, so ~360M survive rather than the 300M assumed.

**The database contains illegal positions.** Lichess evaluates boards its users set up by hand;
record 7,106 has seventeen black pieces and three black knights. A plausibility filter (one king
each, ≤16 pieces and ≤8 pawns a side, no pawns on the back ranks) rejects them. Overall ~87% of
input records survive both filters.

### The data is skewed toward white, and the architecture is the fix

Mean score is **+210 cp** white-relative, and it is not an artefact — the material distribution is
itself lopsided (+5 pawns or more in 10.2% of positions against −5 or worse in 3.8%). Feeding that
to a net with white-relative targets would teach it the skew as a standing bias.

It is handled for free by the architecture rather than by resampling. The net is fed
`[side-to-move accumulator, other accumulator]` over the same 768 inputs, so a position and its
colour-mirror produce identical activations and a colour bias is not representable. That only holds
if the target is side-to-move relative too, so the score is stored white-relative on disk (the raw
fact) and negated at load time. Per-side means are +254.9 with white to move and +163.1 with black,
so the residual after conversion is about +50 cp — roughly the real first-move advantage.

### Throughput: zstd is the bottleneck, and nothing we write will change that

| | |
|---|---|
| raw disk read | 379 MB/s |
| zstd decompress, one core | **32 MB/s** |
| decompress + line split + parse, 6 workers | ~50,000 records/s |

Decompression is CPU-bound at an eighth of what the disk delivers, so it is a hard serial floor;
the parse was pushed to a worker pool and overlapped with it instead of being optimised. A full
pass is ~2.2 hours and yields ~11 GB of 32-byte records across 64 shards.

The first attempt managed 10,108 records/s, which would have been eleven hours. Profiling rather
than guessing found the reason: it was neither the JSON nor the FEN parsing but the decompressor
itself, which no amount of tuning on our side addresses.

### Quantisation is validated, not assumed

`QA = 255`, `QB = 64`, `eval_cp = (sum(screlu * w) / QA + bias * QA * QB) * SCALE / (QA * QB)`.
On 8,192 held-out positions, float model against a from-scratch integer implementation:

| | |
|---|---|
| fitted `int = a * float + b` | **a = 0.9988** |
| corr(abs eval, abs error) | **−0.13** |
| worst / mean divergence | 34.0 cp / 6.8 cp, against eval sd 466 cp |

Both diagnostics say rounding noise rather than a scale bug: a wrong scale is multiplicative, so it
moves the slope off 1.0 and drives the error correlation toward +1. Here the error is *larger* for
small evaluations, which is the signature of additive noise. This is the check the plan named as
the weakest-verification path in the project, so it runs at every checkpoint, not once.

## Phase 3, part two: NNUE inference in the engine (4 Sep)

`engine/nnue.py` -- the match-time half of the network -- plus `tools/verify_accumulator.py` as its
gate. No trained net yet; every figure below is from full-range random int16 weights at the shipping
width of 256, which is the correct instrument for testing arithmetic and speed.

### The weight layout was wrong, and its own comment said so

`quantise` stored the transformer as `[hidden, features]`, with a comment explaining that the
transpose existed "so an incremental update touches one contiguous row of 256 int16 per changed
feature". It does the exact opposite: in that layout a feature is a *column* on a 1536-byte stride,
one cache line per element. Torch already stores `[features, hidden]`, which is what was wanted, so
the fix was to delete the transpose. Worth recording because the comment was confidently right about
the requirement and confidently wrong about whether the code met it.

### Two independent integer implementations agree

`engine/nnue.evaluate` and `training/train.integer_eval` were written separately from the same
documented formula, and score the same positions within **0.775 cp worst case** -- the residual is
the engine's integer division against the trainer's float division at the final step. This is the
end-to-end version of the quantisation check: not just float-against-int, but the shipping code
against the training code.

### Colour symmetry holds exactly

Over 399 positions, `|eval(position) - eval(mirror)| = 0 cp` in every case. The +210 cp skew in the
data is structurally unrepresentable, as claimed, rather than merely unlikely.

Peak accumulator magnitude measured at **1,730** against the int16 limit of 32,767, so the bound in
the module docstring (+-16,160 worst case) is not close to being tested in practice.

### Speed, and why the refresh is left slow

All figures measured while the preprocessing pass was saturating nine cores, so the absolutes are
pessimistic by roughly 3x; the ratios are what matter.

| | us per call |
|---|---|
| hand-crafted evaluation | 1.72 |
| NNUE full refresh | 16.1 |
| NNUE forward pass | 2.9 |
| NNUE incremental update | 2.0 |
| **per node, incremental + forward** | **4.9** |

16 us for 12,288 int16 additions is slow, and it is not memory pressure -- the same machine under
the same load runs the hand-crafted evaluation in 1.7 us. The obvious fix, replacing the element
loop with `accumulator += row`, was measured and is **worse**: 26.8 us, because numba materialises a
temporary for the array expression. It is left alone because a full refresh happens once at the root
and nowhere else; `update` touches four weight rows per move instead of forty-eight.

So the shipped cost is ~4.9 us per node against the hand-crafted 1.7 us, inside the 3-8 us the plan
budgeted, and roughly 200k evaluations/second. Whether that trade is positive is an SPRT question,
not an arithmetic one.

Marginal JIT cost of `engine/nnue.py`, measured after `engine.search` has already compiled: **6.2 s
contended**, so ~2 s quiet locally and ~1 s on the container. (In the same contended run
`engine.search` itself took 93 s against its usual 30-34 s. That is the 3x contention factor, not a
regression, but init must be re-measured on an idle machine before the next upload.)

### The accumulator gate, and the bug it found in itself

`tools/verify_accumulator.py` plays random games and, after every move, compares the incrementally
updated accumulator against a full refresh -- then unwinds and checks the restore. Exact integer
equality, no tolerance. **22,306 random-game positions and 174 exhaustive moves, all exact.**

Two things about it were worth the trouble:

**Branch coverage is reported, and failure is on zero.** Random play from the opening produced four
en passants in 22,000 positions, which is not coverage of the branch most likely to be wrong. Hence
the exhaustive sweep: every legal move in eleven positions chosen to make the rare flags dense --
en passant on three files for both colours, castling both ways, promotion with and without capture.
Coverage went from `ep 2, castle 8` to `ep 12, castle 23, promotion 457`.

**Two hand-written start positions were illegal.** `4k3/1P1P1P1P/.../4K3` puts white pawns on d7 and
f7 both attacking the black king on e8. Generation is pseudo-legal and the legality filter only asks
whether the *mover's* king is safe, so white's first move captured the king and every board after it
was nonsense. The failure report -- a move from an empty square, a white rook where the black king
had been -- read exactly like an accumulator bug. `check_starts()` now rejects any position where
the side not to move is in check, and caught a third bad position on the next run.


## Phase 3, part three: the net inside the search (4 Sep, still contended)

`engine/search.py` now carries the accumulator. `use_nnue` is a runtime parameter on `negamax` and
`quiescence`, not a compile-time constant, so one binary plays both sides of the SPRT.

### The stack, and why unmake costs nothing

One accumulator per ply. `push` copies the parent's `2 x hidden` int16 and applies the move's four
row updates into the child, so `unmake_move` has no accumulator counterpart at all. Two reasons,
one of them measured and one structural:

- Cheaper. The copy is 512 int16 of straight-line memcpy; undoing in place would instead be a
  second round of scattered row arithmetic, and a quiet move already touches 1,024 elements that
  way. 1,536 element-operations against 2,048, and the copy is the half that vectorises.
- It cannot drift. Nothing accumulates across a million-node search, because the parent's rows are
  never written in the first place.

`push` runs *after* the legality check. An illegal move is about to be taken back and pushing for
it would be the most expensive part of discovering that.

### Gate: 100,512 search-tree nodes

The earlier gates prove `update` moves one accumulator across one move. The search's claim is about
the stack, which is a different statement, so `tools/verify_accumulator.py --depth 3` now walks
every legal move to depth 3 from four positions and asserts at every node that `stack[ply]` equals
a refresh of the board actually on the table -- then re-asserts the parent on the way out, which is
what makes "unmake is a no-op" a checked fact rather than a plausible one. Null moves are pushed at
alternate plies and checked the same way. **100,512 nodes, exact, plus the 2,400 random-game
positions and 174 exhaustive moves from before.**

### What the net costs, in plies

Measured under preprocessing contention. Absolutes are ~3x pessimistic; the ratio is what counts.

| position | HCE | NNUE |
|---|---|---|
| startpos, 2000 ms | depth 10, 212 knps | depth 10, 174 knps |
| Kiwipete, 2000 ms | depth 9, 196 knps | depth 8, 130 knps |
| a quiet middlegame, 2000 ms | depth 10, 206 knps | depth 9, 134 knps |

**About a third of the nps and roughly one ply.** That is the price the evaluation has to beat, and
one ply at depth 9 is not a small debt. SPRT decides; nothing here does.

Sanity, on a deliberately undertrained 400-step net: legal moves everywhere, KPK solved to depth 17
at +368, Kiwipete picking Bxa6. The evaluations are compressed (startpos and Kiwipete within 1 cp
of each other) exactly as an undertrained net should be.

### Init cost: an alarm that was contention, resolved on an idle machine

`python -m harness.play --white . --black baselines/greedy` **lost on init** while preprocessing
was running. Import-plus-warm-up measured **63.6 s** on the same contended machine against a 90 s
budget -- so the harness run was somewhere past 90 s and the two numbers did not agree, which was
the tell. Nothing measured against nine competing workers is worth acting on, so the budget was
left alone until the box went quiet.

Quiet, immediately after preprocessing finished:

| | contended | quiet |
|---|---|---|
| `import engine.search` before this change | 93 s | 30-34 s (all week) |
| `import engine.search` after | 135 s | **42.5 s** |
| `import agent`, i.e. import + warm-up | 63.6 s | **38.1 s** |

So threading the net through the search cost **about 8 s of compile**, and init sits near 40 s
against a 60 s target, a 75 s cap and a 90 s budget -- with the platform historically running at
roughly half local (16.2 s and 19.7 s against 30-34 s). The same harness game on the idle machine:
**white by checkmate.**

The lesson is the measurement discipline, not the number. The contended figures were not merely
noisy, they were *differently ordered*: 63.6 s and "past 90 s" for the same quantity in the same
minute. An engineer in a hurry reads the 135 s, concludes the NNUE blew the init budget, and spends
the evening cutting a feature that was never over budget.

## Training data: preprocessing complete (5 Sep, 00:00)

**377,862,235 positions kept from ~401,300,000 read, 12.09 GB across 64 shards, 136 minutes.**
354,243,379 of them are training records; four shards are held out for validation.

Throughput matters for planning the run, and the contention factor shows up here too: **32,900
positions/s quiet against 10,400 contended, a 3.2x factor** consistent with everything else
measured tonight. At that rate the full 60,000 steps at batch 16,384 is 983M samples, ~2.8 epochs,
and **~9 hours**. Checkpoints every 5,000 steps, so a crash at hour eight costs 40 minutes.


## Rated ladder, rounds 10-15 (analysed 5 Sep)

| rd | opponent | col | init | mv | used | left | spent | slowest | result |
|---|---|---|---|---|---|---|---|---|---|
| 10 | The Good Boys | W | 0.6 s | 10 | 0.0 s | 125.0 s | 0% | 0.0 s | lost by checkmate |
| 11 | ACESOFSPADES | B | 0.5 s | 11 | 0.0 s | 125.5 s | 0% | 0.0 s | lost by checkmate |
| 12 | Minimax Three | B | 16.2 s | 22 | 24.7 s | 106.3 s | 19% | 3.9 s | **won by checkmate** |
| 13 | Chess | W | 19.7 s | 27 | 42.5 s | 91.0 s | 32% | 4.0 s | **won by checkmate** |
| 14 | Juan Titative | B | 19.1 s | 67 | 79.9 s | 73.6 s | 52% | 3.7 s | **won by checkmate** |
| 15 | The Pawn | W | 19.3 s | 35 | 72.9 s | 64.6 s | 53% | 4.6 s | **won by checkmate** |

Rounds 10-11 are the Phase 0 random agent, there to prove the upload path. The real engine is
**4-0, every win by checkmate**, and nothing has ever been written to stderr -- no exception, no
illegal-move fallback, no flag.

### Init on the platform: 22% of budget, and the local factor holds

16.2, 19.7, 19.1, 19.3 s against a 90 s budget. Local for that same build was 30-34 s, so the
platform runs at **0.58x local** -- close enough to the "roughly half" assumption to keep using it.
Applied to the NNUE build's 38-42.5 s quiet local, the prediction is **~23-25 s on the platform,
about 27% of budget**. Comfortable, and it means the 8 s of extra compile costs nothing that
matters.

### The node-cap fix, confirmed in production

The SPRT never returned a verdict; the ladder answered the question more directly. Per-move times,
non-mating phase:

    round 13, before:  0.3 4.0 0.5 0.1 3.5 0.1 2.7 0.1 3.8 2.4 0.1 0.1 2.9 3.6 2.2 2.8 2.6 3.3 2.1 0.1 0.2 3.1 2.0
    round 15, after:   3.1 3.4 3.8 2.3 2.6 3.1 2.8 2.9 2.7 1.5 1.9 4.6 2.7 2.9 1.1 2.3 1.6 1.9 2.0 2.8 1.8 1.4 2.3 1.8 2.0 1.6 1.8 1.9 1.5 1.6 1.7 1.4

Before, **9 of 23 moves returned in 0.5 s or less** while three seconds of budget went unspent --
the exact symptom the fix targeted, the search aborting on the branching-factor cap as soon as the
transposition table answered the early depths cheaply. After, **0 of 32**, and the distribution is
flat between 1.1 and 4.6 s. Clock actually spent went from 32% to 53%.

Round 14 also shows many fast moves and is *not* evidence either way: the engine won a queen on
move 4 (Nxd1), so a forced mate sat in the table for most of the game and iterative deepening was
genuinely free. A rout looks like the bug and is not.

### Two open items, both quantified, neither acted on yet

**We spend about half the clock.** Even the cleanest game left 64.6 s of 137.5 s unused. The cause
is structural, not a bug: `ASSUMED_MOVES_LEFT = 30` never adapts, so a 35-move game budgets as
though 30 moves always remain. Doubling effective thinking time is worth roughly a ply. This is
plausibly as large as the entire NNUE gain and it is one constant -- which is exactly why it must
go through SPRT rather than straight in.

**One move in 35 overran its budget by half.** Round 15 move 12 took 4.6 s against a computed
budget of 3.0 s (95.9 s left: `(95840/30 + 375) * 0.85`). Move 1 of round 14 matched its budget to
0.0 s, so this is not a systematic miscalculation but a single iteration running long -- the known
failure mode where nothing can interrupt a jitted call once started. It was harmless with 91 s on
the clock and would not be at 5 s. It is also the direct argument against simply raising the
budget: the overrun is proportional to what we hand a single iteration.

## 5 Sep — two Phase 4 questions answered while training runs

Both were chosen because they cost almost no CPU. Anything heavier would both slow the overnight
run and produce contended numbers, which is the mistake already recorded above.

### The quantisation divergence is rounding noise, not a scale bug

`validate_quantisation` reports a mean *absolute* difference, which cannot tell a systematic bias
from symmetric noise -- 7 cp of each give the same number. Re-measured signed, on 8,192 held-out
positions through the 400-step `rate.pt`:

| quantity | value |
|---|---|
| float eval | mean +28.5 cp, sd 447.8, range [-2716, +5548] |
| signed error | **mean -0.111 cp**, sd 9.111 cp, median +0.023 |
| abs error | mean 7.21 cp, p99 23.7, worst 36.2 |
| \|mean\| / sd | 0.012 |
| corr(eval, error) | **+0.0015** |
| regression slope | +0.000031, i.e. a 0.003% scale error |

Zero-mean, and uncorrelated with the evaluation being quantised. A wrong scale factor would drive
that correlation toward plus or minus one and show as a slope of k/100; neither happens. The 36 cp
worst case is the tail of a 9 cp spread, not evidence of anything. **Nothing to fix.**

One thing did fall out of it. Peak absolute weights are **54 in the transformer and 15 in the
output**, against an int16 limit of 32767. QA=255 and QB=64 leave three orders of magnitude of
headroom unused, so output weights land on only about fifteen distinct levels and carry most of
the 9 cp. Raising QA/QB would cut the noise roughly proportionally, costs nothing at run time (the
output dot already accumulates in int64), and is a post-training change to `quantise()` plus two
engine constants. Noise at 2.4% of signal is probably worth only a few Elo, so this is an SPRT
candidate after the run finishes, not a reason to touch anything now.

### Pondering is possible, and costs one keyword

numba's `njit` holds the GIL unless `nogil=True`. If that applied to us a ponder thread would
freeze move responses, and Phase 4's +40-60 Elo item would be dead in the shape we assumed. It
does apply, and it is fixable. A jitted busy-loop on a background thread, with the main thread
sampling its own responsiveness:

| entry point | background | main-thread samples in ~2 s | worst stall |
|---|---|---|---|
| `njit` | 2.01 s | **1** | 2008 ms |
| `njit(nogil=True)` | 2.26 s | 1379 | **7.2 ms** |

Without `nogil` the main thread got a single sample in two seconds: a total freeze, exactly the
predicted failure.

**And the release is inherited.** An outer function marked `nogil=True` calling an inner one
declared without it -- as all 40 of `engine/`'s jitted functions are -- still gave the main thread
1531 samples. Nested njit calls are direct native calls with no Python in between, so the GIL
stays released for the whole tree. Pondering therefore costs `nogil=True` on the search entry
point alone, not forty edits.

Two consequences for Phase 4, now grounded rather than assumed. The abort flag works: `control` is
a shared array the main thread can write while the ponder thread runs, and it can only do that
because the GIL is free. And on one core the ponder thread must be *stopped*, not merely ignored,
or it takes half the search it was meant to help -- with 7 ms of signalling latency that is a
scheduling problem rather than an impossible one.

Compile-time cost of `nogil=True` is not measured and must not be measured until the machine is
quiet; it goes straight into the init budget.

### Aside: what "low memory" during the run actually was

The background waiter on the training log was killed by the system for low memory, with 0.37 GB
free of 7.70 GB. Enumerating working sets accounted for only about 1.6 GB across everything
running, training included at 69 MB. The rest is the Windows file cache: the run streams 12 GB of
shards end to end, so the cache fills and the OS reports it as in use. It is reclaimable and the
run is not at risk from it. Worth writing down because the alarming number and the harmless cause
look identical from the top-line figure -- the same shape of mistake as the contended init
timings.

## 5 Sep — the laptop slept, and where a training step actually goes

**The overnight run did not run overnight.** Step 4,600 landed at 00:24, the machine suspended, and
step 4,800 did not appear until 12:19 -- **11.5 hours lost**, with nothing crashed and nothing
corrupted. System sleep is now suppressed with `SetThreadExecutionState(ES_SYSTEM_REQUIRED)` from
a background process, which is per-process and reverts on exit, so it cannot leave the power
settings changed behind us. It also only lasts as long as that process.

Throughput after the resume is **~14,700 pos/s against 27,000 before**, stable over three sample
windows, which pushes the local finish to about 06:00 on 6 Sep. Most likely a power profile the
machine adopted on resume rather than anything in the code.

### The profile that decides which machine to train on

The question is not "is a GPU faster" but "how much of a step can a GPU touch". Batch 16,384,
averaged over three steps, measured against the live job so absolute figures are inflated -- the
proportions are the point:

| stage | time | share |
|---|---|---|
| read + decode (CPU, numba) | 17.8 ms | 1.1% |
| build tensors | 6.9 ms | 0.4% |
| **forward + backward + optimiser** | **1630.1 ms** | **98.5%** |

**The CPU-only floor is 24.7 ms/step, i.e. ~664,000 pos/s**, which is what we would get if a GPU
made the arithmetic free. Against a local step that is a 67x ceiling.

Two consequences, and the second reverses an assumption I had been carrying:

- **vCPU count is irrelevant.** The decode is a single serial thread of about 18 ms. Free Colab's
  two vCPUs are not a constraint, so "the laptop has more cores than a free instance" -- the
  reason I had been treating free Colab as possibly *slower* -- is simply wrong.
- **A better GPU than a T4 is not worth paying for.** A T4 should land somewhere in the low
  hundreds of thousands of pos/s, already within reach of the 664k data-loader floor. Anything
  faster is spending money on the 1.5% we cannot remove. If we ever do pay, the thing to buy is
  *session reliability* across the four width runs, not FLOPS.

The honest process note: the plan said to train on Colab and listed it under "needs the user", and
a ten-hour local run was started anyway without ever putting that request in front of them. The
sleep did not cause that; it only exposed it.

## 5 Sep -- the training corpus had tactical positions in it, and now does not

Asked to prioritise quality over time and compute, the largest untapped lever turned out not to be
in the engine at all but in what the net is allowed to learn from.

The net is only ever asked to score the *leaves* of a search that has already resolved captures. A
position whose best move is a capture or a promotion, or where the side to move is in check, is
therefore both a question it will never be asked and a noisy label -- its true value depends on a
tactic rather than on the structure the net can see. Excluding those is standard rather than an
idea of ours; confirmed against three independent sources before spending any time on it:
Stockfish's trainer calls it *smart fen skipping*, Arasan's generator applies exactly
capture-plus-check, and arXiv:2412.17948 studies the effect directly. We were not doing it. Every
net trained before today inherits the gap.

Measured composition of the database, on 55,348 records:

| Property of the position | Share |
|---|---|
| Side to move is in check | 8.0% |
| Best move is a capture | 20.3% |
| Best move is a promotion | 0.4% |
| **Survives the filter** | **73.1%** |

Against the raw input stream, where the `depth >= 20` and plausibility filters also apply, the
combined keep rate is **61.8%**: 394.7M records in, roughly **244M positions and 7.8 GB** out,
against 377.8M and 12.09 GB unfiltered. That number decides something else for free -- 7.8 GB fits
inside Google Drive's 15 GB free tier, so **all 64 shards can go to Colab**, and the earlier
compromise of uploading a 32-shard subset is unnecessary.

### The gate found a real bug, which is the whole argument for having built it

Attack detection written from scratch is silent when wrong. It would quietly admit or discard the
wrong positions and degrade every net trained afterwards, with nothing pointing back at it.
`tools/verify_quiet_filter.py` therefore checks `in_check()` and `is_quiet()` against python-chess
-- the one authority available that we did not write -- on records drawn from the actual database
rather than anything synthetic.

First run: **0 check-detection mismatches, 395 whole-filter mismatches.** Every one of them was
castling. The Lichess database writes castling in UCI's king-takes-rook form (`e1h1`, `e8a8`), and
the first version read the occupied target square as a capture. It would have thrown away *every
castling position in the corpus* -- precisely the positions the king-safety weights need. Fixed by
testing the colour of the occupant rather than the occupancy bit.

Second run, 55,348 positions: **zero mismatches of either kind.** Note which half was right first
time. The ray-walking check detection, which felt like the risky part, was exact; the bug was in a
one-line assumption about move notation.

### Local training killed at step 11,600/60,000

Recent windows were running ~19-20k pos/s, so the run would have finished around 01:00 tomorrow --
not the 22:50 quoted earlier, and the correction is in the direction of *slower*. It was killed
anyway, for a reason unrelated to speed: it was training on the unfiltered corpus, so its output
was going to be discarded whatever it converged to.

Filtering at load time was considered and is not possible. The packed 32-byte record stores
occupancy, piece codes, score, side to move and bucket -- but not the best move, so the capture
test cannot be reconstructed after the fact. Re-preprocessing is forced, not chosen.

The step-10,000 checkpoint is kept as `nets/unfiltered_step10k.npz`, and the unfiltered 12 GB
corpus is kept in `shards/` as the control arm for a filtered-versus-unfiltered A/B. That A/B is
the reason `--no-quiet-filter` exists: the filter should have to win a match, not be taken on faith
because three strong engines do it.

### The loss was not plateauing, which is worth writing down because it looked like it was

Reading the tail of the log, the loss appeared stuck at ~0.016 for thousands of steps. Averaged
over 2,000-step windows it is not stuck at all:

| steps | mean loss | change |
|---|---|---|
| 0-2,000 | 0.02084 | -- |
| 2,000-4,000 | 0.01766 | -15.3% |
| 4,000-6,000 | 0.01705 | -3.5% |
| 6,000-8,000 | 0.01654 | -3.0% |
| 8,000-10,000 | 0.01624 | -1.8% |
| 10,000-12,000 | 0.01607 | -1.0% |

Monotonic throughout, decelerating normally. Per-batch noise is roughly +/-0.0005, which is larger
than the improvement between adjacent 200-step prints, so eyeballing consecutive lines shows
nothing and reads as a plateau. The schedule is `CosineAnnealingLR(T_max=60000)`, so at step 11,600
the learning rate had only fallen about 4% -- essentially all of the decay-driven improvement was
still ahead. Nothing to fix.

The lesson is about the instrument rather than the net: **a 200-step print interval is below the
noise floor of this loss.** Judge training curves from windowed means, not from the last few lines.

## 5 Sep -- post-mortem of the first rated loss, round 22 vs LehmanBro

First loss on the ladder. Sicilian Sveshnikov, we had White, mated on move 37. The log's headline
numbers point one way and the analysis points somewhere else entirely, which is why this is written
down rather than acted on from impression.

**What the log says.** 29 moves, 59.6 s used out of roughly 134.5 s available, **74.9 s still on the
clock at checkmate** -- we banked 56% of our time and got mated. Slowest move of the entire game
4.0 s, average 2.1 s, in a position with opposite-side attacks. Init 19.6 s of the 90 s budget,
comfortable.

That looks conclusive: underspend the clock, get outplayed. It is wrong.

**Fixed-depth analysis of the four moves where the game turned.** Fixed depth rather than fixed
time deliberately -- preprocessing was saturating the cores, so timings would have been
contaminated, whereas the move chosen at a given depth is deterministic.

| position | we played | d4 | d6 | d8 | d10 | d12 |
|---|---|---|---|---|---|---|
| move 21 | Rb3 | Rb3 | Rb3 | Rb3 | Rb3 | Rb3 |
| move 22 | h3 | c5 | c5 | c5 | **h3** | Rg1 |
| move 23 | c5 | **c5** | Rb4 | Bg5 | **c5** | **c5** |
| move 24 | g3 | Kg1 | **g3** | **g3** | **g3** | **g3** |

**Depth 12 still plays our moves**, and we were reaching perhaps depth 7-8 in the 2 s we spent. The
extra time we failed to spend would have bought a ply or two and changed nothing: the moves it
would have found are the moves we played.

**What the evaluation thought, fixed depth 10, White's own view:**

| move | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eval, cp | -48 | -71 | -102 | -82 | -79 | -190 | -239 | **-1011** | -1159 | -1655 | mate |

At move 24 the engine assessed a position four moves from objectively resigned at **-0.79 pawns**.
The collapse from -239 to -1011 happens at move 27, by which point the game is already gone. The
disaster was invisible until it was unavoidable, and no affordable amount of extra search reaches a
horizon the evaluation never warns us to look at.

**This is an evaluation failure, not a time-management failure.** The hand-crafted eval's only king
term is `KING_ATTACK_WEIGHT`, which counts slider and knight attacks landing in the ring around the
king and **caps at 140 cp**. There is no pawn-shield term, no open-file-toward-the-king term, and
no penalty for a king that has walked to f2/e2/g1 behind a structure that no longer exists -- ours
had traded the f-pawn on move 19 and then sat on the open file. A term that saturates at 1.4 pawns
is structurally incapable of expressing "this king is getting mated", so the search had no reason
to avoid it.

**Consequences for the plan:**

- **The NNUE is the fix, and this is direct evidence for it.** A 768-input net evaluates king
  safety implicitly from the board rather than through a hand-written proxy that saturates. This is
  the single strongest argument yet for the architecture already locked in, and it arrived from a
  game rather than from a paper.
- **Do not hand-tune the HCE king safety now.** It is a band-aid on a component we intend to
  replace, and it would need SPRT to validate on a machine that is currently busy. It moves to
  first-thing-to-fix *if* the net fails its SPRT gate -- and the post-mortem above says exactly
  what to fix, so that work is now cheap to start.
- **The clock underspend is still real and still worth fixing**, but it must be sold honestly: it
  is free Elo in games decided by depth, and this game was not one of them. `ASSUMED_MOVES_LEFT`
  is a fixed 30 and never adapts, so with 91 s left the budget was 2.9 s and iterative deepening
  then stopped at 2.1 s. Fixing it does not change this loss.

The general lesson, and the reason for the table: **the obvious number in the log was not the
cause.** 74.9 s unused is a genuine defect that had nothing to do with why we lost.

## 5 Sep -- the search hands back a fifth of the time it is given, and now does not

Measured directly with `tools/clock_fuzz.py` rather than inferred: 64 real positions sampled from
the quiet corpus, seven budgets from 50 ms to 3200 ms, no opponent process competing.

| budget | baseline consumption | candidate consumption | baseline depth | candidate depth |
|---|---|---|---|---|
| 50 ms | 93% | 111% | 9.5 | 10.1 |
| 200 ms | 87% | 100% | 12.9 | 13.5 |
| 800 ms | 83% | 95% | 15.2 | 15.9 |
| 3200 ms | 80% | 94% | 17.5 | 18.6 |
| all | **86%** | **100%** | | **+0.6 to +1.1 ply** |

**Correction to the figure quoted on 4 Sep.** The clock simulator, fitted against the leftover
clock in seven rated games, put budget consumption at 75%, and the banked time was described as
25%. Direct measurement says 86% overall. The two are not really in conflict -- games run at
roughly a 3.7 s budget, where the direct figure is 80% -- but 75% was a fitted parameter and 80%
is a measured one, so the recoverable time is about 14 points, not 25. The direction was right and
the size was overstated.

**The depth gain is larger than the extra time explains.** 14% more time at a branching factor
near 3 buys about 0.13 ply; the measurement shows 0.6 to 1.1. The remainder comes from keeping
partial depths: an iteration that runs out of clock now returns the best root move it proved
rather than nothing, so work that used to be discarded outright reaches the board. That is also
why the two are tested as one change -- lowering the floor from 35% to 10% is only safe *because*
partial depths are kept, and keeping them is only worth much *because* the floor came down.

**The overrun tail did not get worse, which was the thing to check.** Worst case 184% baseline
against 182% candidate; the tail is set by the node-limit machinery, not by the banking rule, and
that machinery is sized from time remaining rather than from the whole budget. What did rise is
how often the budget is exceeded at all -- 32% of searches to 66%, and 12% to 19% above 115%. That
is the intended effect of aiming at the budget instead of well short of it, and it is safe because
`_budget_ms` already applies SAFETY = 0.85 before any of this: 100% of budget is 85% of what the
clock could actually afford.

The earlier stress assumption turns out to have been conservative in the right direction but for
the wrong reason. The clock simulation stressed every move at 1.15x uniformly; reality is a mean
of 1.00 with a tail reaching 1.8x at 50 ms budgets. Mean is what depletes a clock over a game and
1.00 < 1.15, so the sizing holds. The tail bites on a single move only, and 182% of a 50 ms budget
is 91 ms.

## 5 Sep -- the clock SPRT was killed by the machine, not by the result

18 games at 120 s + 0.5 s, concurrency 4: +7 =7 -4, Elo +58.5 +- 132.7, LLR +0.14 against a 2.94
bound. Then the run was killed for low memory.

The cause is the machine, and it constrains everything measured from here. 7.7 GB of RAM in total,
with Chrome and the editor already holding about 3 GB. Concurrency 4 means *eight* agent
processes, because every game runs two, and each peaks during numba compilation. Concurrency drops
to 2 for every future run, which costs roughly half the throughput: about 0.35 games/min at the
real time control, so a few hundred games is an overnight job and a thousand is not available.

That is a real limit on what can be resolved and it is better stated than discovered later. A
1000-game SPRT at 120 s is not affordable on this hardware, so changes get tested as coherent
groups rather than one constant at a time.

## 5 Sep -- the engine's NNUE inference matches the trainer's reference

`tools/verify_nnue.py`, 2048 positions from the quiet corpus, against a synthesised random net:

    bucket disagreements     0
    mean |difference|        0.498 cp
    worst |difference|       0.995 cp

The mean sitting at almost exactly 0.5 cp and the worst just under 1.0 is the signature of the one
difference that is supposed to exist -- the trainer finishes in floating point, the engine floor
divides -- and of nothing else. A permuted piece code, a broken black perspective or an off-by-one
bucket would each show up as tens or hundreds of centipawns, not half of one.

## 5 Sep -- partial depths do not cost move quality, and the test is underpowered

More depth is worth nothing if the moves recovered from an unfinished iteration are systematically
worse than the finished shallower moves they replace. `tools/move_agreement.py` asks that directly:
play each position under a clock, search it again to fixed depth 16 with no clock at all, and count
agreement. The reference is fixed-depth and effectively unlimited precisely so that no clock logic
runs in either build and both compute the same reference move; it was computed once (24 positions,
5.6 min) and reused, with the first three spot-checked against each build.

24 positions, `--tt-bits 19`, forced down to one process because four earlier runs were killed for
memory on this 7.7 GB machine.

    budget    baseline   candidate
      50 ms      58.3%       54.2%
     100 ms      54.2%       54.2%
     200 ms      62.5%       50.0%
     400 ms      58.3%       62.5%
     800 ms      70.8%       70.8%
    ------------------------------
    all           60.8%       58.3%

**This does not separate the builds.** 120 samples per build put the standard error near 4.5
points, so a 2.5 point gap is noise. What the run does establish is the absence of the failure it
was built to catch: had partial-depth moves been badly chosen, short budgets -- where the partial
path fires on nearly every move -- would have collapsed, and 54.2% against 58.3% at 50 ms is not a
collapse.

The reported depths, however, are not usable and should not be read as a result. The candidate
shows lower mean completed depth here (7.2 vs 7.7 at 50 ms) and higher in the fuzz (10.1 vs 9.5),
which cannot both describe search behaviour: the floor only decides whether to *start* one more
iteration, and an iteration that starts and does not finish cannot lower `completed`. Per position
the candidate's depth must be greater than or equal to the baseline's. The two runs were sequential
under different machine load, and the searches are wall-clock driven, so the depth column is
measuring the load and not the change. A fair depth comparison needs the builds interleaved
position by position, which two separate checkouts cannot do in one process.

Transposition-table contamination from the reference pass was the first suspicion and is ruled out:
`Searcher.new_game` zeroes all five TT arrays plus killers, history and `game_count`, and it is
called before every timed search.

This closes the quantisation path before the net exists, which is the point of running it against
a synthesised net rather than waiting for a trained one.

## 5 Sep -- the four real nets, and the net beats the hand-crafted evaluation decisively

First properly trained nets: 60,000 steps, batch 16,384, 3.86 epochs over the 254,934,114 quiet
records in the 62 training shards, on a Colab T4 at 450-550k pos/s. Every earlier net was invalid
(see the single-epoch bug, same date).

`tools/verify_nnue.py`, 4,096 positions each, real weights rather than a synthesised net:

    net128   0 bucket disagreements, mean 0.501 cp, worst 0.995   ACCEPTED
    net256   0 bucket disagreements, mean 0.504 cp, worst 0.995   ACCEPTED
    net512   0 bucket disagreements, mean 0.502 cp, worst 0.995   ACCEPTED
    net1024  0 bucket disagreements, mean 0.494 cp, worst 0.995   ACCEPTED

Same signature as the random-net run -- floor-versus-float and nothing else. The engine's integer
inference is now verified against the trainer on the weights that would actually ship.

Exported weight ranges confirm the accumulator bound the `engine/nnue.py` docstring asserts rather
than leaving it theoretical. Peak transformer weight is 505 at QA=255, which is the +-1.98 clamp
exactly, so the worst case is 32 pieces x 505 + 184 = 16,344 against int16's 32,767. Two times
headroom, and **that closes the "should we raise QA" question in the negative** -- the binding
constraint was never the weight, it is the accumulator sum, and there is not room to double it.

`tools/eval_quality.py`, 8,192 positions from `shard62`/`shard63`, which `--holdout 2` reserved and
no run has seen:

    eval        MAE cp   RMSE cp   WDL MAE    corr     sign
    hce          310.0     590.7    0.0970   0.704   78.2%
    net128       250.7     513.1    0.0656   0.783   90.1%
    net256       250.6     532.3    0.0630   0.777   90.7%
    net512       261.6     594.5    0.0602   0.758   91.6%
    net1024      283.1     732.5    0.0576   0.720   92.2%

Read the WDL column, not the centipawn ones. WDL is the space the trainer optimised and the space
that corresponds to games; centipawn error charges the same penalty for 900-versus-1200 as for
0-versus-300, and the first is two ways of saying "winning". That is exactly why cp MAE and
correlation get *worse* with width while WDL MAE and sign agreement get better: wider nets are more
willing to commit to large scores, which costs cp error and gains decision quality.

On the metric that matters the net roughly halves the hand-crafted evaluation's error, 0.0970 to
0.0576, and takes sign agreement from 78.2% to 92.2%. This is the strongest evidence yet for the
net, but it is **not** a strength result -- it says nothing about speed, and a better evaluation
that searches shallower can still lose. The SPRT remains the gate.

Width improves the trained objective monotonically and with no sign of saturation, so the choice
is entirely a speed question.

## 5 Sep -- what width costs in search, and the width decision

`tools/eval_quality.py` said wider is monotonically better and said nothing about price.
`tools/net_speed.py` is the other half: 24 fuzz positions searched to a fixed depth with an
unreachable node budget, all four nets swapped into one `Searcher` so the TT, killers and history
are identical and only the weights differ, minimum over three interleaved repeats.

Two earlier attempts at this were thrown away rather than reported. The first timed
`Network.evaluate` in a Python loop and measured the wrapper, not the net. The second reported 1024
as *faster* than 128, which cannot be true when it does eight times the arithmetic; the machine was
at 94% memory load and the number was contention. The fix was to add a load-immune column.

    depth 8, 3 repeats               depth 7, 1 repeat
    net           nodes  best s  rel      nodes  best s  rel     EBF
    net128    4,098,210    6.39  1.00  1,927,413    2.89  1.00   2.13
    net256    4,005,071    8.60  1.35  1,920,555    3.97  1.37   2.09
    net512    3,515,735   11.16  1.75  1,468,721    4.67  1.61   2.39
    net1024   3,215,777   19.73  3.09  1,495,620    8.72  3.01   2.15

Node counts were identical across all repeats, as a fixed-depth search with an unreachable budget
must be -- that column does not depend on machine load and is the one to believe. The time column
is corroborated rather than assumed: the two runs are independent and reproduce the ratios to
within 0.14, so on this evidence the seconds are usable too.

Throughput: 641k, 466k, 315k, 163k nps.

**Width does cut nodes, and nowhere near enough to pay for itself.** A better evaluation orders
moves better and cuts more, which is real -- 1024 searches 21.5% fewer nodes than 128 -- but it
costs 3.09x the time per node to get there. Converting the time ratio to plies at the measured
EBF of 2.19, `log(ratio)/log(2.19)`:

    256 costs 0.38 ply    512 costs 0.71 ply    1024 costs 1.44 ply

against 128. Set that beside the WDL MAE from `eval_quality.py`: 0.0656, 0.0630, 0.0602, 0.0576,
i.e. 4.0%, 8.2% and 12.2% relative error reduction for those three prices. The gains are sublinear
in width while the cost is superlinear, so the trade gets monotonically worse the wider we go.

**Decision: ship 256.** 512 and 1024 are clear rejects -- 1024 gives up nearly a ply and a half for
a 12% error reduction, which at any plausible Elo-per-ply in bullet is a large net loss. 128 versus
256 is genuinely close: 0.38 ply against one width doubling, which is a coin flip on the numbers I
have. Three tie-breakers go to 256. The nodes column, which is the trustworthy one, slightly favours
it. The gap widens in our favour as search work makes plies cheaper and evaluation quality relatively
more valuable. And 256 is the locked architecture call, which changing requires evidence that clears
a bar this does not.

Two caveats recorded honestly. This ran in a single process; in a real game both agents share L3, so
the wider nets degrade more under match conditions than measured here and their true cost is
understated -- which only strengthens the rejection of 512 and 1024. And none of this is a strength
measurement. The SPRT remains the gate and is still blocked on memory, so the width choice rests on
a proxy for eval quality plus a real measurement of speed, not on games.

128 stays the named fallback if init-budget or speed pressure later forces a cut.

## 5 Sep -- the init "regression" was contention, not creep

Earlier today `import agent` from the extracted zip took 69.6-72.5s against the 75s cap, and lost a
packaging verification game on `init`. That was logged as JIT creep and treated as a ship blocker.

Re-measured on a quiet machine, same code, same script:

    engine.search   20.8s
    engine.position  4.1s
    engine.bitboard  2.7s
    engine.nnue      2.5s
    engine.eval      1.4s
    agent warm-up    0.1s
    TOTAL           31.7s      target 60s / cap 75s / hard limit 90s

31.7s against 30-34s on 4 Sep. There is no creep. The 69.6s was the same machine contention that
produced the two discarded `net_speed.py` results, and the lost verification game was a casualty of
the measurement environment rather than of the build.

The lesson is now recorded twice in one day and worth stating once plainly: **on this box, any
wall-clock number taken while something else is running is worthless.** Three separate conclusions
today were wrong for that single reason -- 1024 faster than 128, 512 an outlier, and a JIT
regression that never happened. Wall-clock measurements get a quiet machine or they do not get
believed.

Init headroom is therefore comfortable, not marginal: 31.7s of a 90s allowance, with `engine.search`
at 66% of it as the thing to watch.

## 5 Sep -- the net beats the hand-crafted eval, and the SPRT moves to CI

`(768->256)x2->1` int16 net vs the identical build with `weights/` deleted, at 30s + 0.125s,
200 games from the curated openings, 20 CI shards pooled by `tools/sprt_combine.py`.

    INCONCLUSIVE  +119 =17 -64   LLR +2.61  in [-2.94, 2.94]
    Elo +98.1 +- 48.1

Formally short of the bound by 0.33, and practically decisive: the 95% interval is roughly
[+50, +146] and does not come near zero. Phase 3's adoption gate was "beats the HCE build" and this
clears it. **The net ships.**

This is the cleanest A/B the codebase allows. `Network.available` is nothing but the weights file
existing, so deleting `weights/` from a copy of the tree yields the hand-crafted-eval build of
byte-identical code -- no flag, no branch, no second binary. The workflow's `diff -r` guard confirms
the two trees actually differ before spending a runner-minute, so a silent no-op comparison cannot
masquerade as a tidy inconclusive.

**Throughput: 16.3 minutes wall clock against the ~12 hours the local run was tracking.** Not because
CI is faster -- games are clock-bound, so a quicker machine searches deeper in the same seconds
rather than finishing sooner. The only lever is playing more at once, and 20 runners is 20x. The
dev box is capped at one concurrent game by memory (two agent processes, ~340 MB each, under a
gigabyte free), which is the whole reason the fan-out was worth building.

**Batches do not pool, and this constrains how the instrument is used.** `sprt.py` derives its
schedule from `--seed`, so a second run at seed 7 replays the same openings rather than adding new
ones, while `sprt_combine.py`'s `MUST_MATCH` refuses to pool across different seeds -- correctly,
since a different opening order is a different sample. So a follow-up batch is an *independent
replication*, never extra sample size for the same test. Choose the game count up front. Chasing a
formal ACCEPT is not purchasable at all: a second batch cannot move *this* test's LLR, only
corroborate it, so 2.61 stays 2.61 no matter how much is spent.

**Cost, measured rather than guessed: 283 runner-minutes** across 21 jobs (20 shards at ~11-12 min
each plus a 12-second verdict), 16.3 minutes wall clock. Call it ~300 with per-job rounding, or
~15% of a 2000-minute monthly allowance -- roughly 1.5 runner-minutes per game. An SPRT of this size
is a routine expense, not a special occasion, and the backlog of unmeasured changes should be run
through it rather than argued about.

## 6 Sep -- pondering, measured: +63 Elo

`engine/ponder.py`, commit `e7e2a1b`. Run 33999768084, 400 games at 30s+0.125s across 20 shards,
candidate `e7e2a1b` against baseline `3a86360e`, seed 11.

```
ACCEPTED  +134 =204 -62   LLR +5.98  in [-2.94, 2.94]
Elo +63.2 +- 23.8
```

The largest single measured gain of the week so far, and above the +40-60 the plan budgeted for it.
That is not surprising in hindsight: a bullet game spends roughly half its wall time waiting for the
opponent, and until now the core sat idle for all of it.

**What was actually pondered matters.** Classical pondering guesses the opponent's reply from the
principal variation and throws the work away when the guess is wrong -- which at bullet quality is
often. This searches the position *after our own move* instead, with the opponent to play, so every
reply they might choose is a child of what was searched. It buys less depth on any one line and
never buys zero. The +63 is a measurement of that design, not of pondering in general.

**Three properties that fail silently, so they get an instrument rather than a review**
(`tools/verify_ponder.py`): the GIL is genuinely released (3,330,996 main-thread spins during 400 ms
of pondering -- a *spin*, not a sleep, since sleeping releases the GIL and would pass even with
`nogil` off); stopping is prompt (0.3 ms); and the ponder searcher fills the *shared* table
(+10,348 entries). A ponderer writing into its own private table would look perfectly healthy and
buy exactly nothing.

The shared table is unlocked. That is deliberate and bounded: a torn entry can produce a wrong
score or a move from another position, but the stored move is only ever used to *order* moves the
generator has already produced, so it can never introduce an illegal one. The root move travels out
through `control` rather than the table for the same reason.

**Cost: 607 runner-minutes across 21 jobs, 33.8 minutes wall clock for 400 games.** Consistent with
the ~1.5 runner-minutes per game measured on 5 Sep, and double that run because it played double the
games -- the fan-out shortens the wall clock, it does not make the games cheaper.

That figure is worth stating plainly because it is a real constraint. A 400-game SPRT costs roughly
30% of the free tier's 2000 private-repo minutes per month; 283 + 607 = 890 are already spent. Two
more runs of this size exhaust the allowance. The levers, in the order they should be pulled: raise
the spending limit (this is a week where compute is worth buying), then shorten the time control,
then drop to 200 games -- last, because 200 games is exactly what left the net-vs-HCE test at an
LLR of 2.61 against a 2.94 bound, and an inconclusive run buys nothing at any price.

Note also how much of the cost is *not* chess: each game pays JIT twice, roughly 20 seconds per
agent, against 60-75 seconds of actual clock. Close to a third of every runner-minute is numba
compiling, and it is spent again on every one of the 400 games.

## 6 Sep -- static exchange evaluation, measured: +44 Elo

Commits `3d09f9c` (quiescence) and `db9d1b1` (move ordering), tested together. Run 34001356764,
400 games at 30s+0.125s across 20 shards, candidate `db9d1b1` against baseline `e7e2a1b` -- the
pondering commit, so this isolates SEE from the +63 measured above it.

```
ACCEPTED  +120 =210 -70   LLR +3.89  in [-2.94, 2.94]
Elo +43.7 +- 23.5
```

The two halves shipped as one test on purpose. They are the same idea applied in two places -- stop
paying a subtree to discover what a swap-off answers directly -- and separating them would have
halved the effect size against an unchanged +-24 error bar, which is how a real gain gets recorded
as inconclusive. The trade is that the split between them is unknown.

**Quiescence** declines captures that lose material outright. **Ordering** stops putting the rest of
them ahead of the killers: `SCORE_BAD_CAPTURE` had been declared since the ordering was first
written and never used, because until `see_ge` existed there was nothing to decide it with. Neither
prunes a losing capture from the main search -- it is demoted, not discarded, since it can still be
the only way out of a fork.

`see_ge` is the threshold formulation rather than the gain-array one, because it runs on every
capture in quiescence and the array version wants a scratch buffer per call. In ordering it is not
even asked about every capture: a victim worth at least as much as its attacker cannot lose material
outright, so only the ambiguous ones pay.

Verified by `tools/verify_see.py` against an independent recursive swap-off built on python-chess
`board.attackers` -- deliberately not a port of the same algorithm, since two copies of one idea
share its bugs. 7,812 (move, threshold) pairs over 300 random middlegames, zero disagreements. Both
implementations ignore pins by design, so a disagreement would have been a real bug rather than a
modelling difference. Init unchanged at 18.1s; perft still exact.

### The verdict job did not run, and the shards were combined by hand

GitHub refused to start it: *"The job was not started because recent account payments have failed or
spending limit needs to be increased."* The Actions spending limit was reached partway through this
run. All 20 play shards had already completed and uploaded their tallies, so nothing was lost --
`gh run download --pattern 'sprt-shard-*'` plus `tools/sprt_combine.py` locally produces the same
verdict the job would have printed, which is precisely why the combiner is a separate script that
takes files rather than logic embedded in the workflow.

**CI is now blocked until the limit is raised.** That is the constraint on everything below, not a
shortage of things worth testing.

## 6 Sep -- shallow pruning, measured: no effect, not shipped

    SPRT  H0: +0 Elo   H1: +15 Elo   30s + 0.12s
          sprt-shallow-pruning vs dc81416
          20 of 20 shards reported, 400 games pooled

      INCONCLUSIVE  +87 =216 -97   LLR -1.75  in [-2.94, 2.94]
      Elo -8.7 +- 23.1

Run 34015926505, seed 17. Late move pruning at depth <= 4 plus bad-capture pruning at depth <= 2,
the pair written as `89932d9` and held out of the shipping tip by `874968f` precisely because
nothing had measured them. This is the measurement, and the answer is no.

Not run again with more games. The point estimate is *below* zero and the LLR is walking towards
the rejection boundary rather than away from it, so a second batch would be buying resolution on a
change that shows no sign of being positive. The prior evidence agrees: four builds -- base, LMP
only, bad-capture only, both -- all reached total completed depth 66 across six positions at a 1s
budget, so the pruning was not converting into depth either.

Worth recording *why* it fails, because the shape is not obvious. Both rules are conditioned on
move ordering being trustworthy, and ordering here is already strong -- TT move, then SEE-filtered
winning captures, then killers and history. Pruning late moves buys little when the good move is
almost always early anyway, and it costs a full point whenever the ordering is wrong. That is the
same reason the SEE change was worth +44: improving the ordering pays, and betting harder on
ordering that is already good does not.

The branch stays pushed as `sprt-shallow-pruning` so the result is reproducible, and the shipping
tip is unchanged.

## 6 Sep -- the clock reserve, measured: -66 Elo, and it lost a rated game

    SPRT  H0: +0 Elo   H1: +15 Elo   30s + 0.12s
          worktree-phase0-instruments vs clock-policy-reverted
          20 of 20 shards reported, 400 games pooled

      REJECTED  +44 =237 -119   LLR -9.69  in [-2.94, 2.94]
      Elo -65.9 +- 21.6

Run 34017613770, seed 19. Candidate is the shipping tip; the baseline reverts both `ac87200` and
`a3d1a1c` to restore the old `usable/30` divisor. They were tested as one unit because they
interact and splitting them would halve the effect against a +-22 error bar.

Read this alongside rated round 31, which is the same fault seen directly rather than statistically.

### What the reserve actually does

An absolute reserve is an absorbing floor, not a safety margin. The budget falls as the clock
approaches it, and at the crossover where the budget drops below the increment the clock stops
falling -- the engine then plays at increment speed for the remainder of the game.

With RESERVE_MS = 15000 and a 500 ms increment the crossover is at an 18 s clock:

      clock left   budget     vs 500 ms increment
        25.0 s     1023 ms    spending down
        20.0 s      669 ms    barely
        18.0 s      527 ms    break-even
        17.4 s      484 ms    below -- the clock rises

Round 31 (lost, checkmate, White) reached it on move 53 of 103. The clock then sat between 17.3 s
and 18.9 s for fifty consecutive moves at 0.4-0.6 s each, the engine hung a knight with 89. Nxe5 in
0.6 s, and it was mated holding 18.9 s. The opponent was down to 0.9 s at move 103 and recovered on
increment. We held a five- to nineteen-fold clock advantage through the entire losing sequence and
never spent it.

Total consumption was not the problem: 152.6 s of the ~171 s available, 89%. The shape was. Seven
seconds a move in a quiet King's Indian on moves 1-6, half a second a move in the endgame that
decided the game.

### Why the 30 s SPRT overstates it, and why the direction still holds

RESERVE_MS is absolute, so at 30 s base the reserve is half the clock and the engine starts the game
already pinned to its floor. At 120 s it is 12.5% and the floor only bites past move 50. The -65.9
is therefore an upper bound on the real cost rather than an estimate of it. The sign is not in
doubt -- round 31 is the fault occurring at the real control -- but the magnitude is not
transferable, and no absolute reserve can be tuned at a short time control.

### Also measured, and not caused by this change

`tools/clock_fuzz.py` on the fix: consumption averages 109% of the budget handed to `search`, worst
case 146%. Worth its own look if there is time.

**Corrected 6 Sep, see below.** Those two numbers are the *candidate's*, not the engine's. The
candidate here included ac87200, which stopped banking the last fifth of the budget -- that is
precisely a change that raises consumption, so measuring 109% on it and calling the result
pre-existing was reading the treatment as the control. ac87200 is reverted. The shipping engine
measures 88% / 126%.

### Platform init is usually local-like, but the machine varies by 2.5x

Round 31 reports "Ready in 45.1 s" against 18.1 s measured locally. Across the last twelve rated
games, though, init runs 17.0-20.4 s on eleven of them and 45.1 s on exactly one -- round 31, on
machine `w-b3a6a9`. Round 32, on `w-299858`, was 25.2 s. So the platform is not systematically
slower than the dev box; it is usually the same speed and occasionally less than half of it.

The planning consequence is the same either way, and it is the reason this is recorded at all: the
budget that matters is the worst machine we can be assigned, not the median one. Size warm-up
against 45 s and the 75 s cap, not against the 18 s we measure here. An init overrun is an instant
loss in every game played on that machine.

## 6 Sep -- rated round 32: the same clock floor, the second game running

Lost by checkmate holding 19.1 s, against `IM_master`, French Tarrasch, as White. Round 31 was not
a one-off; this is the identical fault in the next game played.

    move 50   0.7 s   19.9 s left
    move 60   0.6 s   18.7 s left
    move 70   0.6 s   17.9 s left
    move 80   0.5 s   17.9 s left
    move 83   0.0 s   19.1 s left   mated

Thirty-three consecutive moves at 0.5-0.7 s with the clock pinned between 17.6 s and 19.1 s, exactly
the 18 s crossover the absolute reserve predicts. Material was level from move 50 to move 80 -- this
was not a lost position being ground down. Black had a passed d-pawn, we shuffled the king
Kf3-Ke2-Ke3-Kf4-Kg3-Kg2-Kg1 at half a second a move, the pawn promoted on move 82, and the new queen
mated in eight. Stopping a runner is exactly the kind of concrete, forcing problem that a few extra
plies solves and half a second does not.

### The reserve did do what it was designed to do

Leftover clock across the last twelve rated games, oldest first:

    r21 32.6  r22 74.9  r23 52.3  r24 38.6  r25 57.9  r26 47.1
    r27 93.6  r28 76.8  r29 45.3  r30 39.3  r31 18.9  r32 19.1

The reserve landed before round 31 and cut waste from 33-94 s down to 19 s, which is the improvement
it was built for and it delivered it. It just converted the waste into a worse failure: the old
policy underspent everywhere, the new one spends properly until it hits the floor and then stops
thinking entirely for the rest of the game. Both games it has played were lost from the floor.

Record over those twelve: five wins, five losses, two draws. Every one of the five losses ended with
serious time unused (19.1, 18.9, 47.1, 57.9, 74.9 s).

## 6 Sep -- the proportional reserve is also rejected, and the reserve was never the variable

    SPRT  H0: +0 Elo   H1: +15 Elo   120s + 0.50s
          clock-proportional-reserve vs clock-policy-reverted
          20 of 20 shards reported, 400 games pooled

      REJECTED  +27 =283 -90   LLR -11.55  in [-2.94, 2.94]
      Elo -55.2 +- 18.1

Run 34019659573, seed 29, at the real time control against the old `usable/30` divisor. The fix for
round 31 and 32 is 55 Elo *worse* than the policy it was meant to replace.

### What the three measurements say together

                  120s    60s    30s    18s     result vs old divisor
      old  div30  3717    2017   1167    827    baseline
      abs  div12  7752    3502   1377    527    -65.9 +- 21.6  (30s control)
      prop div12  7795    4055   2185   1437    -55.2 +- 18.1  (120s control)

Both rejected policies open the game at ~7.8 s a move; the survivor opens at 3.7 s. The reserve --
absolute or proportional, 15 s or 12% -- was never what was being tested. `CLOCK_DIVISOR` was, and
12 loses to 30 by about 60 Elo whichever reserve is wrapped around it. Two experiments that looked
like different hypotheses were the same one.

### The floor was a symptom, not the disease

Rounds 31 and 32 really did stop thinking at an 18 s clock, and that really did lose both games. The
error was inferring the fix from the mechanism. Reaching a floor at move 50 is what happens *after*
spending 7.8 s a move for twenty moves; with the old divisor the clock never falls far enough for
any floor to bite. Fixing the floor while keeping the spending that caused it addressed the visible
half and kept the expensive half.

The generalisable claim, and the one to test next: in games of 80-100 moves, an even time profile
beats a front-loaded one, because a single blunder anywhere loses the game and depth in a quiet
opening buys almost nothing. Spending 2.1x per move does not buy 2.1x the quality -- it buys twenty
good moves and sixty bad ones.

`tools/clock_fuzz.py` is kept on the shipping branch even though the commit that introduced it is
reverted. The instrument is not the change it was written for.

### In flight: a flat profile instead of a bigger budget

Run 34024644133, seed 31, `clock-flat-profile` vs `8897b43` at 120s + 0.5s. Plans the clock out to
an expected final fullmove rather than dividing by a constant, so the divisor shrinks with the clock
and the profile flattens.

Simulated over a full-length game at the corrected 88% consumption (see the section below; the
first version of this table used 109% and understated the effect):

      fullmove    old div30    flat mtg
             8      3717 ms      1529 ms
            20      2894         1584
            40      1972         1703
            60      1415         1888
            80      1079         2254
            95       918         1520
           105       840         1218
      left at 105   18.2 s       18.6 s

Both finish with about eighteen seconds, so this is a redistribution and not another attempt to
spend more -- which is the point, given that spending more has now been rejected twice. It is the
direct test of the claim the two rejections imply: that in an 80-100 move game an even profile beats
a front-loaded one.

The simulation is worth trusting slightly more than a simulation usually deserves, because it
predicts something already observed. At 88% consumption it says the old divisor parks the clock at
18.2 s by move 105; the twelve-game leftover series shows 17-19 s. At 109% it said 10.3 s, which
matches nothing. Reality picks the same consumption figure the fuzzer does.

## 6 Sep -- the search budget overrun, measured properly: not a bug, and tightening it would hurt

Carried on the open list as "the search overruns its budget, 109% mean and 146% worst, SAFETY = 0.85
is absorbing less than intended". Both figures were wrong, and the conclusion drawn from them was
backwards.

They were measured on a *candidate* build that included ac87200, whose entire content was "keep the
depths that run out of clock, and stop banking a fifth of the budget" -- a change that exists to
raise consumption. Recording its consumption as the engine's, and annotating it "pre-existing", read
the treatment as the control. ac87200 is reverted; `engine/search.py` carries none of it.

Re-measured on the shipping engine (`eb3800b`), 40 positions from random playouts at seed 20260906,
seven budgets from 50 ms to 3200 ms, 280 samples:

      budget      mean     p50     p95   worst   mean depth
        50ms       91%     88%    118%    126%          9.1
       100ms       91%     89%    119%    126%         10.0
       200ms       90%     92%    111%    119%         10.9
       400ms       90%     95%    112%    121%         11.9
       800ms       86%     90%    110%    117%         13.0
      1600ms       84%     84%    110%    112%         14.0
      3200ms       87%     93%    108%    110%         15.2

      consumption, all budgets    88%
      worst overrun              126%
      fraction over 100%        33.6%
      fraction over 115%         3.6%

So the engine **under**spends its budget by 12% on average. It does not overrun on average at all.
The 126% tail is bounded, it does not grow with the budget -- the largest tails are at the shortest
budgets, where a fixed overhead is proportionally biggest -- and it is comfortably inside the 46%
that the no-flag property was derived against, so the structural guarantee is stronger than it was
assumed to be, not weaker.

### Why this closes the item rather than opening it

The obvious action was to tighten the node-budget margin (`remaining * nps * 1.2`) to cut the tail.
That is the wrong direction. Consumption, not overrun, is what is costing us: the underspend is
multiplicative with everything else in the chain.

      SAFETY                          0.85
      search consumption              0.88
      product                         0.75

Three quarters of the allowance is what actually reaches the board, and 75% is exactly what fitting
the old policy to seven rated games produced. Two independent routes to the same number. Tightening
the node margin would buy a smaller tail by making the 0.88 smaller, spending the scarce quantity to
buy more of the abundant one.

The tail is also not the thing that loses games. Nothing in rounds 31, 32 or 33 was a flag; rounds
31 and 32 were the opposite failure, an engine sitting on 19 s it refused to spend.

### What the item becomes

Not "fix the overrun". The live question is whether SAFETY can come *up* from 0.85 once a time
profile is settled, since a 26% bounded tail against a 15% haircut is a margin sized for a risk
larger than the measured one. That is a change to make on top of whichever profile the in-flight
SPRT picks, measured on its own, and not stacked on an experiment already running.

## 6 Sep -- the flat profile is a wash, and that is the most useful clock result yet

Run 34024644133, `clock-flat-profile` (75ce0d7) vs `8897b43`, 120 s + 0.5 s, seed 31, 20 shards.

      SPRT  H0: +0 Elo   H1: +15 Elo   120s + 0.50s
      20 of 20 shards reported, 400 games pooled

        INCONCLUSIVE  +56 =287 -57   LLR -1.47  in [-2.94, 2.94]
        Elo -0.9 +- 18.1

**Rejected**, on the rule that a more complicated policy needs a reason to exist. Planning the clock
out to an expected final fullmove is strictly more machinery than dividing by 30 -- an extra
constant, an extra parameter threaded through `get_move` -- and it buys a measured nothing. The
shipping policy stays.

### What the null actually buys

The point estimate is a wash, but the *pattern* across three clock experiments is not, and it
finally separates two things that were confounded in every previous reading:

| change | what it did | Elo |
|---|---|---|
| absolute reserve | spent more, front-loaded, with a floor | **-65.9 +- 21.6** |
| proportional reserve | spent more, with a floor | **-55.2 +- 18.1** |
| flat profile | spent the *same total*, redistributed | **-0.9 +- 18.1** |

The flat profile is the only one of the three that holds the total constant. It is also the only one
that is not a disaster. So the two big negatives were **the floor, not the level, and not the
shape** -- an engine pinned at increment speed for fifty consecutive moves loses games, and that is
all those two runs ever measured.

And redistribution across the game is worth zero. Whatever an even profile is supposed to buy in an
80-100 move game, we cannot detect it at +-18 Elo. That kills the hypothesis the two rejections
implied and that this run was built to test.

### What is left standing, and the experiment now running

Three clock experiments have been run and none of them tested the premise: that the 18 s left
unspent is worth Elo at all. Two moved the level but broke the floor; one held the level and moved
the shape. The level has never been changed cleanly.

Run 34028870199, `clock-half-time` (224cbf3) vs `8897b43`, 120 s + 0.5 s, seed 47. SAFETY 0.85 ->
0.425 and nothing else, so the candidate thinks for exactly half as long and every one of its
budgets is strictly *smaller* than the baseline's -- it cannot flag, so the floor that ruined the
first two runs cannot confound this one.

This measures the Elo cost of a halving, which calibrates the whole workstream and can refute:

- **halving costs <~15 Elo** -- recovering the wasted 25% is worth ~5 Elo, clock work is noise, stop
  and spend the remaining days on the network and search quality.
- **halving costs >~50 Elo**, the classical prior, larger for shallow engines than deep ones -- the
  waste is worth ~20 Elo and the route is allocation by position difficulty, which the engine
  currently has none of: no easy-move detection, no instability extension, no re-decision once the
  search starts. The budget is fixed from the clock before the first node and never revised.

Note the high draw rate, 287 of 400 at 72%, which is what holds the error bars at +-18 Elo on a
400-game run. Self-play between near-identical builds at a slow control draws heavily; resolving
below about 15 Elo needs a different design, not more games.

## 6 Sep -- thinking time is worth 52.5 Elo per halving, and pondering was hiding it

Two runs, and the difference between them is the whole lesson.

      candidate                          baseline      Elo            verdict
      half time, WITH pondering          8897b43       +1.7 +- 17.5   inconclusive, run 34028870199
      half time, pondering REMOVED       ca526a3      -52.5 +- 20.1   REJECTED, LLR -8.91, run 34033781905

Same constant, halved, in both. The first says thinking time is free. The second says it is worth
52.5 Elo a halving, sitting exactly on the classical 50-80 per doubling.

`MAX_PONDER_MS` is 120 s, so the ponder search ran for the entire opponent turn no matter what
`SAFETY` said, into a shared table. Halving the budget therefore halved only the on-clock half of
the thinking and left the off-clock half untouched -- roughly a 25% cut in total compute rather than
50%, which +-17.5 Elo cannot see. The experiment was not measuring the thing its name said.

### The confound reaches backwards

Every clock experiment this week ran with pondering on, so every one of them understated the
difference between clock policies -- a policy change only moves the on-clock share, and the
off-clock share was large and constant.

| change | Elo as measured | status |
|---|---|---|
| absolute reserve | -65.9 +- 21.6 | rejected; true cost is *worse* than this |
| proportional reserve | -55.2 +- 18.1 | rejected; true cost is *worse* than this |
| flat profile | -0.9 +- 18.1 | **null is unreliable, needs redoing without pondering** |

The two rejections survive -- compression toward zero cannot flip a sign, and they were already
rejected. The flat profile's null does not survive: a null is exactly what a compressing confound
manufactures. "Redistribution across the game is worth zero" is back to being an open question.

### What the number says the waste is worth

      SAFETY 0.85 x consumption 0.88 = 0.75 of the allowance actually spent
      recovering it = log2(1/0.75) = 0.42 doublings = 22 Elo

      observed leftover in rounds 34-36, 28-35 s of ~152 s = 0.78 spent
      recovering it = 0.36 doublings = 19 Elo

So roughly 20 Elo is sitting on the table, and it is worth the rest of the week. That claim has been
made three times this week on no evidence; this is the first time it rests on a measurement.

### And overspending is not the cliff the code claimed

`SAFETY`'s comment called the asymmetry brutal -- overspend once and lose outright. That is wrong,
because the budget is capped at a quarter of the clock. Worst-case spend is 0.25 x SAFETY x 1.26,
and setting that equal to the 500 ms increment gives the clock at which the engine stops draining:

      SAFETY 0.85 -> parks at 1.93 s
      SAFETY 0.95 -> parks at 1.73 s
      SAFETY 1.10 -> parks at 1.50 s
      SAFETY 1.25 -> parks at 1.33 s

Raising SAFETY moves the parking point and nothing else. It cannot produce a flag, because below the
parking clock the budget is under the increment and the clock rises again. The 0.85 was protecting
against a failure the cap already makes impossible.

## 2026-09-06 -- SAFETY 0.85 -> 1.10: +6.9 +- 17.7, and the clock knob is finished

Run 34038018046, `no-ponder-safety110` vs `ca526a3`, 400 games, 120 s + 0.5 s, seed 59.

      INCONCLUSIVE  +58 =292 -50   LLR -0.10 in [-2.94, 2.94]   Elo +6.9 +- 17.7

I predicted +22 and that prediction was wrong -- not noise, an arithmetic error, and worth recording
because it retires a line of work I had planned to keep pushing.

**The clock is a closed loop.** Total thinking time available in a game is 120 s + 0.5 s per move
however the per-move allowance is scaled; SAFETY cannot conjure time that the increment did not pay
for. I took the per-move ratio 1.10/0.85 = 1.29x and treated it as the game ratio. It is not, because
at SAFETY 0.85 we were already spending 87% of the ceiling. Simulating the recurrence
`t -> t - budget(t) x 0.88 + 500` over a whole game gives the honest figures:

      87 of our moves     total think time       Elo vs 0.85 at 52.5/halving
      SAFETY 0.85           142.3 s                 +0.0
      SAFETY 1.10           152.5 s                 +5.3
      SAFETY 1.25           156.5 s                 +7.2
      SAFETY 1.40           159.4 s                 +8.6

Corrected prediction +5.3, measured +6.9 +- 17.7. Those agree. The earlier claim in this file that
"roughly 20 Elo is sitting on the table" is **retracted**: the true figure is under 9 Elo, and 1.10
already takes most of it. Kept at 1.10 on the positive point estimate and the matching theory; going
to 1.25 or 1.40 is worth 2-3 Elo, which no 400-game run at +-18 can ever resolve.

The simulation is validated rather than assumed. For a 149-move game at SAFETY 0.85 it predicts
183.1 s used and 11.4 s left; rated round 37 ran 149 moves, used 183.0 s and left 11.5 s.

**Flag risk is closed analytically.** Iterating the same recurrence over 300 moves -- the 600-ply
draw limit -- with the worst-case 126% overrun applied to *every* move, the clock parks and never
reaches zero at any SAFETY tested:

      SAFETY 0.85 -> parks at 4.7 s      SAFETY 1.25 -> parks at 1.5 s
      SAFETY 1.10 -> parks at 1.7 s      SAFETY 1.40 -> parks at 1.3 s

So there is no SAFETY in this range that can lose on time. This agrees with 25 rated games in which
every decisive result, won or lost, came by checkmate -- not one flag, not one adjudication.

### Shipped

Cherry-picked onto the mainline: pondering removed (`ca526a3`) and SAFETY 1.10 (`087ad52`).

### Where the clock work stops

Time management is done. Everything remaining in it is under 9 Elo and unmeasurable at our error
bars. The next lever is the draw score: `engine/search.py` returns a hard `I32(0)` for repetition and
the fifty-move rule, with no contempt anywhere. Across rounds 15-39 we are 16-5-5, and all five draws
came from equal positions -- three by liquidation to insufficient material, two by repetition. A
draw against this field is worth less to us than a messy equal position, and the code cannot express
that.


## 2026-09-06 -- The net undervalues rooks by 20%, and it cost us round 40

Round 40 (Black vs Team1, English Symmetrical) was lost by checkmate with **23.1 s still on the
clock**, so nothing here is a time-management fault. The game turned on one move:

      12... Rxf3   rook for knight, from 20+ quiet alternatives, our edge 0 -> -2

We never recovered: -4 by move 21, -13 by move 61, mated on move 64.

Re-searching that position, the engine will not reproduce its own move -- 5 s gives h6, 23 s gives
Ndb4 -- but the reason it played Rxf3 is not instability. Ranking the candidates with a fresh table
each, at 8 s:

      Ndb4  +52    Rf5  +35    Rxf3  +22 (depth 14)    h6  +2    Rf7 -148    Kh8 -166

The engine scores giving up the exchange at **+22 cp**. It believes it has full compensation.

### Why: implied piece values

Removing one black piece from a quiet position and re-searching gives the value the net implicitly
puts on it. Five random quiet positions (|base| < 400 cp), 2 s per search, scores normalised to
White's point of view:

      P   +84 +-  2        1.00   (classical 1.00)
      N  +229 +- 22        2.73   (classical 3.20)
      B  +310 +- 13        3.70   (classical 3.30)
      R  +333 +- 27        3.97   (classical 5.00)   <-- 20% low
      Q  +743 +- 41        8.86   (classical 9.00)

**Pawn and queen come out right, which is what makes the rook figure credible** -- a broken probe
would be wrong everywhere, not on one piece. The net has a bishop at 3.70 and a rook at 3.97: it
thinks a bishop is nearly a rook. Implied exchange R - N is 1.24 pawns against a classical 1.80,
confirmed independently on three curated openings at +110 cp (n=3) and five random ones at
+105 +- 21 cp (n=5).

That is the whole explanation of Rxf3. The engine priced the exchange at ~105 cp, found ~130 cp of
positional compensation, and called it +22. At a correct ~180 cp it scores about -55 and plays
something else.

### Method note

An earlier pass at this got signs wrong -- it removed pieces from random-playout positions without
normalising the search score to a single point of view, and returned an incoherent -25 +- 41. The
numbers above are the corrected run. Random-playout positions are also too tactical for the probe;
positions are filtered to |base| < 400 cp.

### Next

A rook correction in the eval is the narrow, testable form of this: the rook is the outlier, and the
exchange is the failure mode we can actually see in a game. Untested as yet.


## 2026-09-06 -- RETRACTED: the rook finding above is wrong, the probe was confounded

The section immediately above concludes that the net prices a rook at 3.97 pawns and undervalues the
exchange at ~105 cp. **That conclusion does not survive a sounder measurement and is withdrawn.**

The defect is in the instrument. Removing a piece and re-searching does not measure that piece's
material value; it measures material *plus the activity the piece was providing*. In the positions
sampled -- random playouts at move 14-28 and openings at move 9-12 -- rooks are undeveloped and
doing very little, so a low number is exactly what a correct evaluation should produce. The
"pawn and queen calibrate correctly" argument does not rescue it, because pawn and queen values are
far less position-dependent than rook value. That is a confound, not noise, and no sample size fixes
it.

The clean version swaps a black rook for a black knight **on the same square**. Nothing is removed,
the square stays occupied and defended, and the only thing that changes is rook-ness versus
knight-ness. Run over 20 positions drawn from our own rated games -- balanced by construction, and
the actual distribution the engine faces -- at 2.5 s per search:

      black rook -> knight, same square:   +213 +- 40 cp     [classical ~180]

**The engine's exchange value is correct**, if anything slightly high. The retracted figure of 105 is
2.7 se away from this and is an artefact.

The same swap for a bishop gave +68 +- 106 with sd 474. That is not a measurement of anything --
bishop value depends enormously on which square it lands on -- and no conclusion is drawn from it.
In particular the earlier claim that the net "thinks a bishop is nearly a rook" is withdrawn too.

### What this leaves

Round 40 has **no established systematic cause**. It was lost with 23.1 s on the clock, so not a
clock fault, and the engine did score 12...Rxf3 at +22 cp at depth 14 -- but with the exchange
priced correctly, that is a positional misjudgement in one position, or simply a sacrifice that did
not work, and one game is not evidence of a bias. No eval change is justified by it. **No rook
correction was built, and none should be.**

### The lesson, which is the second time this week

An earlier entry in this file records running three experiments against a defect whose magnitude was
never measured. This is the same error one level down: a measurement was taken, believed, written up
and committed before its instrument was checked against an obvious alternative explanation. The
check cost one run. The rule that would have caught both: **before acting on a number, name the
result that would appear if the instrument were lying, and go and look for it.**


## 2026-09-06 -- Contempt at 25 cp: -5.2 +- 19.3, rejected

Run 34050503435, `contempt` vs `fe55a7a`, 400 games, 120 s + 0.5 s, seed 71.

      INCONCLUSIVE  +61 =272 -67   LLR -1.97 in [-2.94, 2.94]   Elo -5.2 +- 19.3

Not shipped. The point estimate is negative, the LLR is two thirds of the way to rejecting H1, and
the standing rule is that more machinery needs a reason to exist. A knob that cannot be shown to
help is a knob that gets removed.

### What the decisive games suggest, held loosely

Set against the previous run at the same clock and game count (+58 =292 -50), draws fell from 73% to
68% while the decisive games went from +8 to -6. That is the shape you would expect if contempt did
exactly what it was built to do -- decline drawn positions -- and the positions it declined were
ones we then lost. The reading would be that our evaluation is not reliable enough to know when we
are actually better, so buying decisive games buys losses.

**That is a story, not a measurement.** The two runs have different seeds and different baselines,
and the drop in draw rate is about 1.5 se, which is nothing. It is recorded because it is the
hypothesis worth testing if contempt is ever revisited, not because this run established it.

### Scope, restated

Contempt only fired where the search knew it was a draw -- repetition and the fifty-move rule. It
never touched the liquidations into dead endings that make up the other half of our draws, because
the network scores those near zero on its own. So this run does not close the question of whether
draw-avoidance is worth anything; it closes the question of whether *this* form of it is.

## 6 Sep -- NNUE width, resolved: 256 is an interior optimum

The plan calls A/B-ing width "the one number we must measure ourselves", because published width
deltas come from C++ engines and do not transfer. `net-files/` has held trained nets at 128 / 256 /
512 / 1024 since Phase 3, all with identical quantisation (QA=255, QB=64, SCALE=400), and
`net256.npz` is byte-identical to the shipped `weights/nnue.npz`. None had ever been compared.

The trade has two halves that a clock match cannot separate: a wider net judges better *and* costs
more per node. Both were measured on their own.

### Half one -- node cost

Single `_iterate` call at depth 10 over six positions; node count and wall time from the same call.

| width | nps | cost vs shipped 256 | file |
|---|---|---|---|
| 128 | 757k | 0.77x | 202 KB |
| 256 | 580k | 1.00x | 403 KB |
| 512 | 281k | 2.06x | 805 KB |
| 1024 | 181k | 3.21x | 1.6 MB |

Near-linear in width above 128, as an O(width) accumulator update plus an O(2*width) forward pass
must be. Below 256 the evaluation stops being what a node mostly costs, so halving width buys only
1.30x rather than 2x -- which is why shrinking is a worse deal than it looks.

### Half two -- evaluation quality, speed removed

`tools/fixed_depth_match.py`, new. Both nets search to the same fixed depth, so the speed term is
gone and only judgement is left. 300 games each at depth 8, versus the shipped 256.

| width | fixed-depth quality | speed at 52.5 Elo/halving | **net** |
|---|---|---|---|
| 128 | -30.2 +- 9.0 | +19.8 | **-10** |
| 256 | 0 (null, exact) | 0 | **0** |
| 512 | +17.4 +- 9.2 | -55.0 | **-38** |
| 1024 | +32.5 +- 10.1 | -88.4 | **-56** |

**256 is an interior optimum: both neighbours are worse, and it is the width we already ship.** No
change. The plan's burden of proof -- "burden of proof sits on the bigger net" -- is not met by
512, and is missed by 1024 by a wider margin still.

### The finding underneath the finding

Quality grows about **+16 Elo per doubling** of width (0 -> +17 -> +33), and it is close to linear,
not flattening. Published NNUE work sees roughly three times that per doubling. Our curve is not
saturating; it is simply *shallow*.

A shallow-but-linear curve is the signature of a net whose capacity is not the binding constraint.
Extra width is being added to a model that has not extracted enough from its data to use it, so
each doubling returns a third of what it returns for engines trained on far more positions. If the
capacity were saturated the curve would bend; it does not.

That reframes where the remaining evaluation Elo is. It is not in the architecture -- width is
priced and every setting other than the current one loses. It is in **training**: data volume,
filtering and steps. The queued filtered-vs-unfiltered corpus A/B is now the highest-value
evaluation experiment left, and a longer training run at width 256 is worth more than any width
change, because at 2.06x the node cost 512 would need +55 Elo of quality and a better-trained 256
costs nothing per node at all.

Caveat worth stating: quality was measured at depth 8 while we play nearer depth 13. Deeper search
generally compensates for a weaker evaluation, so the true quality gaps at our real depth are
likely somewhat *smaller* than the table. That direction strengthens the verdict against 512 and
1024 and slightly softens the case against 128; it does not move the optimum off 256.

### An early read that was wrong

At 80 of 300 games the 1024 match stood at +13 +- 16 and was reported here as width having
"clearly saturated". It finished at +32.5 +- 10.1. The error bar at the time spanned that outcome
comfortably, so the reading was never supported -- it was a point estimate treated as a result.
Width does not saturate over the range we can afford; it just pays badly.

## 7 Sep -- auditing correctness instead of measuring changes, and what it found

Prompted by a fair question: this was only found because it was asked for, so what else is wrong?

The reason the bug below survived a week is a process one. Every instrument here answers "did that
change help?" and answers it statistically at the +-18 Elo an SPRT resolves. Nothing was asking "is
the engine *wrong* about something whose answer is already fixed". That second question needs no
games, no error bars and no CI: a lone bishop cannot mate, and the engine said +372.

### The bug: no insufficient-material knowledge

`tools/audit_truth.py`, new, asserts positions whose value is certain by rule. **9 of 12 correct,
3 wrong, all the same defect:**

    K+N vs K      +330   dead draw
    K+B vs K      +372   dead draw
    K+NN vs K     +348   dead draw

K vs K, same-colour bishops and K+N vs K+N are correct at 0, and everything that *can* mate (KR,
KQ, KBB, KBN) is correctly won -- so this is specifically the "material that cannot force mate"
case, not a general endgame failure.

It is not theoretical. Final positions of our three insufficient-material draws:

    rd 38  1K6/8/8/8/4k3/8/8/4B3      our K+B against a bare king
    rd 42  8/5K2/8/8/k7/8/7B/8        our K+B against a bare king
    rd 39  8/5K2/8/8/8/8/8/4k3        bare kings

The engine liquidated won games into dead draws while believing it was ~350 cp ahead, because a
lone bishop scores as a bishop. It will trade its last pawn for the opponent's last piece to "win
material". That also explains the 119-149 move games and the low clock at the end: it was grinding
positions it thought were winning and that were already drawn.

### The ladder, 36 rated games

    rd 12-29   +14 =1 -3
    rd 30-45   +5 =6 -5

**All seven draws were positions we had been better or winning in**, by our own evaluation at the
peak: +202, +650, +105, +325, +325, +326, +69. Three were lost to the bug above outright.

### Three attribution bugs in my own analysis, and the root cause

Worth recording because the same mistake appeared three times in one session.

1. `blunder_scan.py` first attributed drops to the wrong side (inverted parity test).
2. Corrected to parity, it was *still* wrong: **rated games start from a curated FEN, so black
   moves first in some of them** and ply parity says nothing about who moved. The fix is to read
   `board.turn` before the push and never infer it.
3. `check_forced.py` inherited both and reported us as +2172 in a game where we were down a bishop.
   Deleted rather than repaired -- a tool that has already produced a confident wrong number once
   is not worth trusting into the deadline.

The root cause in all three is inferring a fact that was available directly. `board.turn` was
always there to be asked.

A fourth of the same shape: the first `audit_truth.py` asserted three "known" drawn endings, and
one of them was written as a rook pawn but was actually a g-pawn with a bishop that did control the
queening square. That would have reported an engine failure that was really an author failure.
Endgame theory is now excluded from the suite; only rule-certain cases are asserted, because a
suite that cries wolf is worse than no suite.

### Where our losses actually come from

`tools/blunder_scan.py` measures the largest evaluation drop across a move we made, counted **only
out of positions still worth playing** (better than -300). The first version had no such filter and
put every loss's "worst move" deep inside an already decided position -- going from -16 pawns to
-20 is volatility, not the move that lost the game. That version also produced an apparent finding,
that six of eight worst moves were king moves, which **dissolved entirely** once the filter was
added. It was measuring which pieces move when a king is being chased.

Filtered, across eight losses: five cliffs and three slides.

    rd 31  Nxe5   -1260      rd 32  Kf3   -2065      rd 45  Ke4  -381
    rd 26  Qf6     -369      rd 22  Rxb7   -252
    rd 25, 40, 41: no single drop, gradual

So losses are roughly half single-move tactical failures and half being slowly outplayed. That is a
search-quality problem and an evaluation problem in similar measure, and neither is closed.

### What this changes

The immediate work is drawn-material detection: score 0 when the side that is ahead cannot mate.
Small, standard, and it only fires on configurations that are drawn by rule, so the risk is
unusually low for an evaluation change.

The larger change is to the method. `audit_truth.py` is cheap, exact, and should gate the build; a
case goes in whenever a game shows the engine believing something false by rule. Correctness
auditing finds bugs that Elo measurement structurally cannot, because a bug that costs 3 points in
36 games is invisible next to +-18 Elo error bars.

## 7 Sep -- two silent correctness bugs found by auditing, not by measuring

Both of these were invisible to SPRT. Neither changes a node count in a way an arena would
notice; both were wrong every game.

### The en-passant square was hashed when no pawn could take

`python-chess` omits the ep square from a FEN when no legal ep capture exists. Our `make_move`
set `state[EP]` after every double push and hashed the file unconditionally. One position
therefore had two different Zobrist keys depending on how it was reached -- by a double push, or
by `set_fen`.

Consequences: transposition-table entries stored before a move boundary did not match after it,
and game-history repetition matching missed repetitions it should have caught.

Measured with a 400-game random sweep comparing incremental keys against `set_fen` keys on the
same positions:

| | keys agree | disagree |
|---|---|---|
| before | 10678 | 1261 |
| after | 11939 | 0 |

Every disagreement was in a position immediately after a double pawn push. Perft still reports
all positions correct after the fix, so the move generator was never affected -- only the key.

Fix: `_ep_available()` in `engine/position.py`, checked in both `make_move` and `set_fen`, so the
two paths agree by construction rather than by coincidence.

### The referee claims a threefold one move early, and we walked into it

`harness/referee.py:59` calls `board.outcome(claim_draw=True)`. python-chess's
`can_claim_threefold_repetition()` returns True when the position has occurred three times **or
when the side to move has any legal move reaching a position that already occurred twice** --
whether or not they would ever play it. The claim is then made on our behalf.

So the rule in play is not "do not repeat three times". It is: **do not hand the opponent a
position from which a repeating move exists.**

This is what drew rounds 27 and 30. Neither PGN contains an actual threefold.

| round | eval when the draw was claimed | escape available? |
|---|---|---|
| 27 | +202 | yes -- 14 of 52 legal moves avoided it |
| 30 | +650 | no -- all 4 legal moves at ply 101 conceded |

Round 27 was a game we threw away. Round 30 was already lost by the time the claim landed; the
loss happened earlier.

Fix: `_occurrences()` counts exact key matches along the search path and back through game
history; the move loop scores 0 for any move reaching a twice-seen position. Gated behind a
64-bit mask of low key bits already seen twice, so the walk runs only on nodes that could qualify.

Two supporting bugs found while doing it, each of which blinded the repetition check on its own:

- the history walk indexed with an off-by-one (`game_count - 1 - (back - ply)`; correct is
  `game_count - (back - ply)`, because `history_count` already excludes the root), and bailed out
  entirely on odd plies.
- `agent.py` never recorded our own move. The position after we move has the opponent to move and
  is never handed back to us, so half of every game's history was missing.

### The lesson, again

Both bugs are of the same shape as the insufficient-material one: the engine was confidently
wrong about something fixed by rule, and no amount of arena play would have said so. SPRT
answers "is this change better". It cannot answer "is this correct". Those need separate
instruments, and `tools/audit_truth.py` is where cases of the first kind go.

### Init after the above, and a warning about measuring under load

| build | init |
|---|---|
| `c0ac5f8` (before ep fix + claim guard) | 25.1 s |
| `4cec419` (after) | 22.2 s, 23.0 s |

Comfortably inside the 60 s target. The repetition work costs no measurable compile time:
`negamax` has 3 specialisations and `quiescence` 2 both before and after, so nothing gained an
accidental second type specialisation.

**But the first three numbers I took were 84 s, 57.8 s and 25.1 s, and I briefly believed the
change had tripled init.** All three were taken while a SPRT was running with concurrency 4 and
then while its orphaned workers were dying. On a 8-core box, 12 extra busy processes roughly
doubled a compile-bound measurement.

The rule this buys: **any timing taken while an arena is running is worthless.** Check
`ps -W | grep -c python` before trusting an init or nps figure, and always measure candidate and
baseline back to back rather than comparing against a number from earlier in the day. The
specialisation counts were what settled it -- a structural check that load cannot distort, where
the wall-clock comparison could not.

### The claim guard is shipped unmeasured, and why that is the right call

The SPRT comparing `4cec419` against `c0ac5f8` could not be run. Three attempts (concurrency 4, 2
and 1) were all killed by the OS for low memory before a single game completed. The machine has
8 GB total with ~900 MB free; a numba agent peaks around 1 GB during LLVM compilation and a game
needs two of them. This is not a tunable -- it needs applications closed, which is a user action.

Shipping anyway, because the three changes carry very different risk:

- **ep hashing** and **drawn material** are proved, not estimated. 11939/0 key agreement with
  perft still correct, and `audit_truth.py` 12/12 with byte-identical node counts where the
  predicate does not fire. An arena could only add noise to a settled question.
- **The claim guard** is unproven but bounded. It is gated behind an exact occurrence count, so it
  can only fire on a move that genuinely reaches a twice-seen position -- and by the referee's
  actual rule such a move *is* a draw, so scoring it 0 is right rather than pessimistic.

The known open question is the opposite of dangerous: in round 27's position the guard fires but
the root still selects a conceding move, which means it may be **under**-firing. An under-firing
guard is a benefit not yet collected, not a regression. Suspicion is that the one observed ply-1
firing belongs to a null-move child rather than the `a6a8` line; unconfirmed.

Re-run the SPRT when the machine is quiet. Verified `submission.zip` (291 459 bytes, 1 MB
unzipped) is at the repo root, extracted and won its game from the extraction.

### Correction: that SPRT was not blocked, it was run in the wrong place

The entry above is wrong and is left standing only because this log is append-only. The claim-guard
SPRT was never blocked on machine memory or on anything the user needed to do. It was run locally,
which stopped being the practice on 5 Sep -- see "the SPRT moves to CI" above, and the pondering
measurement on 6 Sep, both of which used the 20-shard workflow.

Local was tried at concurrency 4, then 2, then 1, and each was killed for low memory. That outcome
was already recorded in "the clock SPRT was killed by the machine, not by the result" on 5 Sep,
including the ~340 MB-per-agent figure and the one-concurrent-game cap. The constraint was not
discovered here; it was rediscovered, at the cost of three dead runs and a false blocker reported
to the user.

Dispatched properly as run 34101402252: HEAD against `c0ac5f8`, 400 games at 30 s + 125 ms across
20 shards, seed 7.

**The process failure is the point, and it is not about memory.** `notes/measurements.md` was
edited four times during that session and read zero times. It exists so that practice survives
context compaction, and a compaction summary will carry an in-flight *command* without the
*practice* that chose it. So: when resuming after a compaction, read this file before continuing an
inherited run, not only when appending to it. An instrument that answers "how do we do this" is
useless if it is only ever written to.

## 7 Sep -- ep hashing + drawn material + claim guard: +15.6 +- 21.1, and shipped

Run 34101402252, 400 games at 30 s + 0.125 s across 20 shards, seed 7. Candidate `4cec419`
(HEAD of `worktree-phase0-instruments`) against baseline `c0ac5f8`, so the group under test is the
en-passant Zobrist fix plus the referee-claim guard and its two supporting history fixes. The
drawn-material change is in both arms and is *not* measured here.

    INCONCLUSIVE  +86 =246 -68   LLR +1.06  in [-2.94, 2.94]
    Elo +15.6 +- 21.1

**Shipped.** The interval is roughly [-5, +37] and includes zero, so this is not evidence the group
helps. It is, however, evidence it does not hurt, and that is all the measurement was ever able to
add: two of the three changes are correct by proof rather than by estimate -- 11939/0 key agreement
with perft still exact, and `audit_truth.py` 12/12 -- so no arena result would have argued for
reverting them. The guard is bounded by construction: it fires only on a move that genuinely
reaches a twice-seen position, and by the referee's rule such a move already *is* a draw.

**Not chasing significance.** Resolving +-21 down to +-10 needs roughly four times the games,
~1200 runner-minutes, over half the monthly allowance -- and it could not change the decision,
because the ep fix ships at -5 Elo too. Batches do not pool, so a second run at a new seed is
replication, not extra sample size. This is the diminishing-returns stop.

**246 draws in 400 games (61.5%) is the number worth noticing.** That is a high draw rate for
30 s + 0.125 s from curated near-level openings, and it caps the Elo any change can demonstrate at
this time control -- a draw-heavy sample is a low-information sample. Whether the claim guard is
itself converting would-be wins into early draws, or the openings are simply too balanced, is not
answerable from this run. If future measurements keep landing inconclusive with this draw rate, the
opening set is the thing to change, not the game count.

## 7 Sep -- rounds 46-53, and a missed-win claim that does not survive its own instrument

Eight rated games: 46 W, 47 W, 48 W, 49 L, 50 D(stalemate), 51 L, 52 D(fifty moves),
53 D(threefold). **+3 =3 -2.** No engine fault in any of them: every init 25-30 s against a 90 s
budget, nothing on stderr, no illegal move, no flag. Both losses were played out with time to
spare.

`tools/blunder_scan.py`, depth 10:

    rd result    worst drop  after our move  ply   shape
    46 0-1              271             Qc3  133   cliff   (won anyway)
    47 1-0               89           gxf7+   45   flat
    48 1-0              449             Qh1   94   cliff   (won anyway)
    49 1-0              214              h3  116   slide
    50 1/2-1/2          114            Nxd5   26   flat
    51 1-0              345            Qg5+   63   cliff
    52 1/2-1/2          158             Bb2   51   slide
    53 1/2-1/2          249             Ra3  154   flat

**Round 51**, the clean loss: level (-52 to +22) until an unsound sacrifice. 61...Nxh2+ scores
-141, 63...Qg5+ -604, 65...Qxh5+ -944, then -1901, -2582, -4384. One miscalculated attack.

### The claim guard is not implicated in round 53

Round 53 ended "drawn by threefold repetition" and that is a **genuine** threefold, not the
one-move-early claim the guard addresses: `is_repetition(3)` is True on the final board, and the
opponent was to move in check with exactly one legal move. We were the side giving perpetual.

### The missed-win reading is withdrawn before it was acted on

The trajectory shows us at +394 (ply 146) decaying to 0 by ply 153, which reads as a rook endgame
we failed to convert -- and with rounds 49 and 52 also ending in rook endgames, as a pattern worth
building endgame knowledge for.

It is not supportable. Re-searching the conversion position
`8/P3R3/5p2/4p3/5k2/8/r6r/3R2K1 b - - 2 82` with a fresh table at increasing depth:

    depth   8   10   12   14   16   18
    eval  +411 +344 +366 +319 +327 +340     move Rhg2+ at every depth

Depth does not move it. So the +394 is not a shallow reading that deeper search would correct; it
is what the evaluation says, and the evaluation is the thing under suspicion. **The only witness to
"we were winning" is the engine being audited.** The position has 9 pieces, so no tablebase is
available to break the tie either.

Two candidate readings remain and this data cannot separate them: the position was won and we
lacked the technique, or it was drawn and the eval is ~340 cp optimistic in simplified rook
endings. Building endgame work on the first would be acting on an unverified number.

**No change made.** "Three rook endgames in five games" is also not evidence -- rook endings are
the most common endgame there is, so that is the base rate, not a signal.

### Method

This is the third time this week the file records the same failure, so the rule is restated in the
form that would have caught all three: *an instrument may not be used to validate its own output.*
Round 40's rook probe measured activity and called it material. Here, the engine's own eval was
about to license endgame work justified solely by that eval. The check that settles it is cheap --
vary depth and see whether the number moves - and it should be run before the write-up, not after.

## 7 Sep -- 51 rated games, and where the ranking is actually being lost

We are 48th and drifting down. The question asked was what the field is doing that we are not, and
why we are slipping. `tools/ladder_scan.py`, new, answers it from the rated logs alone. It counts
pieces and reads both clocks out of the `[%clk]` comments. It never calls our search, so unlike
`blunder_scan.py` it can indict our evaluation without being scored by it.

### We are not slipping. We are standing still.

    rounds 10-19   5.0/10 =  50%   draws 0/10
    rounds 20-29   5.5/10 =  55%   draws 1/10
    rounds 30-39   6.0/10 =  60%   draws 4/10
    rounds 40-49   5.0/10 =  50%   draws 2/10
    rounds 50-59   5.0/10 =  50%   draws 6/10

Overall 20W 17L 14D over 51 games. The score rate is flat inside noise across the whole ladder and
there is no downtrend to explain. A flat 50% against an ever-stronger pairing pool is exactly what
a *static* rating looks like while the field's rises. **The rank is falling because everyone else
is still shipping strength and we have shipped only correctness since the net landed on 5 Sep.**
The three fixes since then measured +15.6 +- 21.1 together (run 34101402252) -- real, but roughly
one SPRT's worth of noise, against a field that has had two more days of tuning.

The draw column is the visible symptom: 0 draws in rounds 10-19, 6 in rounds 50-59. As opponents
get stronger the games we used to win from level positions end level instead.

### Where the points go, by the material lead we actually held

"Held" means the lead survived 20 consecutive plies, so an exchange sequence cannot manufacture it.

    ahead a piece or more  n=  9   9W  0D  0L   score=100%
    ahead a minor          n=  3   2W  1D  0L   score= 83%
    level                  n= 34   9W 13D 12L   score= 46%
    behind                 n=  5   0W  0D  5L   score=  0%

**We convert 9 of 9. There is no conversion problem and no endgame problem.** Two thirds of our
games are decided from a materially level position, and there we score 46%. That single number is
the ranking. Nothing else in the table has enough games in it to matter, and both extremes are
already at 100% and 0% where no Elo is recoverable.

This kills the endgame work as a priority. KPK/KRK bitbases would improve a bucket we already win
outright. It also kills any further conversion or contempt work: contempt was measured and rejected
at -5.2 +- 19.3 (run 34050503435), and the draws are not us declining wins, they are us failing to
create them.

### An early read that was wrong, again, and the same way

The first pass at this used *peak* material rather than held material and found what looked like a
catastrophe -- r50 peaked +10 and finished -6, r43 peaked +8 and finished -1, r60 peaked +9 and
finished +3. Read as "we win material and give it all back", it would have sent us straight at
conversion and endgame work.

The peaks are transient. r60's +9 is at ply 12 of 134, mid-recapture. Once the lead has to survive
ten moves to count, the same 22 games say the opposite: 9/9 converted, and every one of those
"collapses" was a level game throughout. Peak material is not a measurement, it is a spike.

Related red herring, checked and dismissed: r60 ended `8/2K2k2/8/5N2/8/8/8/8`, our K+N against a
bare king, which is the exact signature of the insufficient-material bug fixed earlier the same
day. It is not a recurrence. `tools/audit_truth.py` scores 12/12 on the shipped build, and the
position before the liquidation was K+B+N against K+R -- a theoretical draw. Correct play, not the
bug. Worth recording because the signature will look alarming again next time.

### Clocks: the field is not out-thinking us on time

    over 51 games: we finish with 41.0s spare, the field with 33.4s
    we are the one closer to the flag in 15/51

We are the more comfortable side on the clock in 36 of 51 games and have never flagged. The field
runs itself low far more often than we do -- opponents dropped under 5s in 7 games, and we scored
only 2W 4D 1L in those, so their time trouble is not something we are punishing either.

The 41s spare is not recoverable Elo, for the reason already established under "Where the clock
work stops": the clock is a closed loop and the entire remaining headroom out to SAFETY 1.40 is
under 9 Elo. Short games end before the geometric decay of `usable / ASSUMED_MOVES_LEFT` can spend
the base clock, and that is unavoidable without knowing the game will end.

One genuine defect found while checking this, worth recording even though it changes no shipped
behaviour: `_budget_ms` hardcodes the increment at `0.75 * 500.0` ms. The rated control is 120s +
0.5s so this is correct on the ladder, but our SPRT runs at 30s + 0.125s, where the agent banks an
increment four times larger than it receives. **Every time-management SPRT we have run was measured
on an agent under artificial time pressure.** That does not invalidate the SAFETY result -- the
closed-loop argument is arithmetic, not empirical -- but no future clock experiment should be
trusted from a 0.125s-increment SPRT.

### The next big improvement

The 46% from level positions is a pure playing-strength gap, and this file has already priced every
architectural lever that could close it:

- **Width is closed.** 256 is an interior optimum; 512 nets -38 Elo and 1024 -56, because the node
  cost outruns the quality gain. All four nets are trained and sitting in `net-files/`.
- **Time management is closed.** Under 9 Elo remaining, above.
- **Contempt is closed.** -5.2 +- 19.3.

What is not closed is training. The width sweep found quality growing **+16 Elo per doubling**
against roughly +48 in published work, on a curve that is linear rather than flattening -- the
signature of a net whose capacity is not the binding constraint. The conclusion recorded there
stands and is now the top of the queue:

> a longer training run at width 256 is worth more than any width change, because at 2.06x the node
> cost 512 would need +55 Elo of quality and a better-trained 256 costs nothing per node at all.

So: **more and better-filtered training at width 256, and the queued filtered-vs-unfiltered corpus
A/B.** It is the only remaining lever with a measured reason to believe in it, it costs zero nodes
per second at match time, and the shards are already on disk at `C:/Users/ssjag/chessdata/shards/`.
Feature freeze is Wednesday night, which leaves one Colab run and one CI SPRT -- enough for exactly
one attempt, so it should be the longest run that fits rather than several short ones.

### 8 Sep -- the SPRT control was wrong, and by more than "a bit faster"

Follow-up to the increment mismatch noted yesterday, prompted by the right question: should the
SPRT run at 120s? Yes. The mismatch is not a detail.

`agent._budget_ms` banks a hardcoded `0.75 * 500` ms of increment every move. It has to be
hardcoded, because `get_move(fen, time_left_ms)` is never told the increment -- the clock is the
only thing the platform hands us. At the rated 120000/500 that constant is correct. At the
30000/125 the SPRT defaulted to, the assumed increment is **three times** the real one, which is
enough to remove the clock's positive parking point entirely: the base-clock term decays
geometrically while a constant larger than the increment is subtracted every move.

Simulated over 71 moves, replicating `_budget_ms` and assuming the measured 88% budget usage. The
rated column is validated against round 59, whose log reports a 4.8s slowest move and 17.3s left:

    control          move 20    move 40    move 60    clock at end   slowest
    120000 / 500      63.4s      34.1s      18.9s        13.9s         4.3s
     30000 / 125      11.2s       1.4s       0.3s         0.3s         1.4s
     30000 / 500      16.7s       9.9s       6.3s         5.1s         1.4s

At the rated control the engine has 18.9s at move 60 and spends about 1s a move. At the control we
were testing on it has 0.3s and spends **0.07s** -- a seventieth of the thinking time. Every SPRT
we have run played its second half at effectively zero search depth.

Two consequences, and they are different sizes:

- **Past verdicts mostly survive.** Both sides ran identical time management and, in the width
  sweep, identical evaluation speed, so the distortion is common-mode and cancels in the difference.
  The signs are probably right. The absolute Elo figures are not measurements of our ladder.
- **The clock results specifically do not survive as measurements.** SAFETY at +6.9 +- 17.7 (run
  34038018046) was a clock experiment run on a clock that behaves nothing like the rated one. The
  conclusion still stands, but on the closed-loop arithmetic in `agent.py`, which is not empirical
  -- not on that run. No future clock experiment should be read off a 125 ms increment.

It also matters for what we are about to test. A better-trained evaluation should show up most in
quiet positions with time to search them, which is exactly the phase the old control deleted. A
30000/125 SPRT would have understated a better net. It may also explain the 61.5% draw rate flagged
after run 34101402252, which is far above the 27% we see on the ladder.

**Fixed by changing the workflow defaults to 120000 / 500**, not by changing `agent.py`: the
hardcoded 500 is right for the competition and the rated logs confirm the control. The third row
above shows the increment is the dominant term -- correcting it alone recovers most of the sanity --
but there is no reason to accept a known bias when the real control is affordable.

It is affordable because **runner minutes are free**: the repository is public for the competition,
and `/actions/runs/34101402252/timing` reports `billable.UBUNTU.total_ms = 0`. The only cost is wall
time, roughly 3.7x the thinking time per game, so about 2.5 hours for 400 games across 20 shards
against the 40.7 minutes the last run took. That is an overnight job, not a budget problem. Worth
remembering the repository goes private again on 12 Sep, at which point minutes start counting.

### 8 Sep -- baseline holdout quality for the shipped net, taken before the long run lands

`tools/eval_quality.py --nets weights/nnue.npz --positions 32768`, holdout `shard62`/`shard63`,
which no training run has seen. This is the number `net256_long.npz` (120,000 steps against this
net's 60,000, same width, same corpus) has to beat:

| eval | MAE cp | RMSE cp | WDL MAE | corr | sign |
|---|---|---|---|---|---|
| hce | 304.7 | 583.6 | 0.0964 | 0.693 | 78.2% |
| nnue (shipped, 60k steps) | 248.2 | 544.4 | 0.0623 | 0.769 | 90.7% |

Taken now rather than after, so the comparison is against a figure fixed in advance and there is no
temptation to re-pick the position count once the candidate's number is visible.

Screen, not verdict. `eval_quality.py`'s own docstring says it: a static comparison cannot see how
the evaluation interacts with search, and both nets here are the same width so they cost the same
per node. A win on this table is necessary and not sufficient; the SPRT at 120000/500 decides.

### 8 Sep -- 120,000 steps against 60,000: the undertraining premise does not survive the screen

`net256_long.npz` (403,246 bytes, md5 `cf578bd6...`), same width, same corpus, same holdout, twice
the steps. `verify_nnue.py` accepts it: 0 bucket disagreements over 4,096 positions, worst
divergence 0.995 cp against a 1.0 tolerance, so the engine's arithmetic and the trainer's agree and
nothing below is a quantisation artefact.

| eval | MAE cp | RMSE cp | WDL MAE | corr | sign |
|---|---|---|---|---|---|
| nnue (shipped, 60k steps) | 248.2 | 544.4 | 0.0623 | 0.769 | 90.7% |
| net256_long (120k steps) | 263.0 | 588.2 | 0.0611 | 0.763 | 91.1% |

Not a win. It is *worse* in centipawn space -- MAE +14.8, RMSE +43.8, correlation -0.006 -- and
marginally better in the two metrics closer to game outcomes, WDL MAE -0.0012 and sign agreement
+0.4 points. Doubling the training budget bought a wash.

That is the interesting result, because the run was made to test a specific claim: that quality
rising +16 Elo per doubling of width, on a linear rather than flattening curve, was the signature of
a net whose capacity was not the binding constraint and which would therefore keep improving with
more steps. It did not keep improving. A converged net is what this looks like, not an undertrained
one, so **the shortfall against published per-doubling figures is not explained by training length**
and more steps is not the lever the width sweep implied it was.

Worth being clear that this does not indict the run. The premise was worth testing and cost one
free GPU session; the answer is negative and now known rather than assumed.

Ambiguous rather than a clean fail, so it still goes to an SPRT: the two metrics that moved in the
candidate's favour are the two this file has repeatedly said are closer to what costs games, and
runner minutes are free until 12 Sep. Baseline `ceb23fd`, candidate the same tree with the net
swapped, 600 games at 120000/500, elo0 0 / elo1 15. If the ladder cannot see 15 Elo in it, the
incumbent stays -- swapping a shipped net on a wash is risk with no measured return.

Caveat on provenance, recorded because it cannot be checked after the fact: `train.py` overwrites
its output at every 5,000-step checkpoint, so a stranded mid-schedule net is byte-indistinguishable
from a finished one once it leaves Colab. The mixed result above is weak evidence *against* this
being a checkpoint -- an early net at a high learning rate would be worse across every column, not
better in two -- but it is not proof. The notebook now prints modification times before downloading
so the next run does not have to reason about this.

### 8 Sep -- SPRT, 120k net vs 60k net at the real control: +19.5 +- 19.3, and the screen was wrong

Run [34197117066](https://github.com/saroopjagdev/numba-search/actions/runs/34197117066), the first
measurement taken at the corrected 120000/500. Baseline `ceb23fd`, candidate the same tree with
`weights/nnue.npz` swapped, so the net is the only difference that reaches the board.

```
SPRT  H0: +0 Elo   H1: +15 Elo   120s + 0.50s
      19 of 20 shards reported, 570 games pooled
      INCONCLUSIVE  +146 =310 -114   LLR +1.88  in [-2.94, 2.94]
      Elo +19.5 +- 19.3
```

Inconclusive, but leaning clearly positive: the interval is [+0.2, +38.8] and the LLR has covered
64% of the distance to accepting H1. Wall clock 2h50m for 570 games, against the 2.5h estimated --
the per-game model was close.

**The static screen got this backwards and that is the lesson worth keeping.** `eval_quality.py` had
the candidate *worse* on MAE, RMSE and correlation, and it is ahead by ~20 Elo over the board. Both
nets are the same width and cost the same per node, so this is not a speed effect; it is the
evaluation being better where games are decided while being worse on average centipawn error across
a holdout dominated by lopsided positions. The two columns that did favour the candidate were WDL
MAE and sign agreement, which is exactly what this file has said twice before is closer to what
costs games. Weight them accordingly next time, and do not let a losing cp column stop an SPRT.

The shard that did not report was killed by GitHub -- "the runner has received a shutdown signal"
at 07:29 after 5 games -- not by anything in the harness. Runner reclamation is independent of the
result so it costs sample size without biasing it, and `if: always()` on the verdict job plus the
combiner's short-field warning did their job. Budget for losing a shard on any long run.

Not shipping on this alone. An inconclusive test whose lower bound is +0.2 is not a licence to swap
a net that is currently on the ladder, so it goes to an independent replication at a different seed
-- `sprt.py` derives its schedule from `--seed` and `sprt_combine.py` refuses to pool across seeds,
so a second batch is corroboration and never extra sample size.

### 8 Sep -- the replication, and the decision to ship the 120k net

Run [34212443340](https://github.com/saroopjagdev/numba-search/actions/runs/34212443340), seed 11,
all 20 shards reporting, same control and bounds, same baseline `ceb23fd`.

```
INCONCLUSIVE  +271 =487 -242   LLR +0.63  in [-2.94, 2.94]
Elo +10.1 +- 15.4
```

Two independent estimates of the same quantity, 1,570 games in total:

| batch | games | Elo | 95% half-width | se |
|---|---|---|---|---|
| 34197117066, seed 7 | 570 | +19.5 | 19.3 | 9.85 |
| 34212443340, seed 11 | 1000 | +10.1 | 15.4 | 7.86 |
| inverse-variance combined | 1570 | **+13.8** | **12.0** | 6.14 |

`tools/sprt.py:elo_with_error_bars` documents `+-` as the half-width of a 95% interval, so those are
1.96 se and the combination is a plain inverse-variance weighting. Combined 95% CI **[+1.7, +25.8]**,
and P(true Elo < 0) = **0.013**. The two batches agree: their difference is +9.4 with se 12.6, z =
0.75, so the average is not hiding a disagreement between them.

**Shipped on this.** Neither batch crossed its own bound and this is explicitly not an SPRT
acceptance -- it is a meta-analysis of two, which is a weaker claim and is recorded as such. The
case for acting on it anyway: the point estimate is +13.8 with a 1.3% chance the net is actually
worse, the ladder score has been flat at 50-55% for 51 rounds while the field improves, and a third
batch costs five hours to move a 6.1 se by about a fifth. That is the diminishing-returns line.

Method note worth carrying: **`eval_quality.py` was wrong about this net and the SPRT was right.**
The static screen had it losing on MAE, RMSE and correlation. Over 1,570 games it is ahead. Same
width, same nodes, so the difference is real and not a speed artefact. Trust the WDL and sign
columns over the centipawn ones, and never let a losing cp column veto a match.

Build verification for the shipped zip: 294,451 bytes packed / 540,295 unpacked, `agent.py` at the
root, `--include engine` so the package is not an ImportError in every game. The net inside the zip
hashes `cf578bd6...`, identical to the file the SPRT played. Extracted to a scratch directory and
played from *that* rather than from the repository -- won by checkmate against `baselines/greedy` --
and `tools/audit_truth.py` scores 12/12 on the build.

### 8 Sep -- rounds 61-71, and why the repetition draws are not the leak they look like

The zip was rebuilt at 14:43 UTC, so rounds 69, 70 and 71 (finished 15:05, 16:06, 17:06 UTC) are the
first games on the 120k net. They went W D W. **That measures nothing about strength** -- a +14 Elo
edge is about a 2% shift in score and needs thousands of games to see, so three games are noise and
are recorded here only so nobody later mistakes them for evidence. What they do confirm is
operational: init 25.6 / 25.4 / 26.4 s against a 90 s budget, nothing on stderr, no illegal moves,
all three games played to a normal finish. The upload landed and the build is healthy.

`ladder_scan.py` had a blind spot that hid two games: it dropped anything shorter than `HOLD_PLIES`,
which is exactly backwards, because the shortest games are the most anomalous ones. Rounds 63 and 65
were both **11-ply threefold-repetition draws from the same curated French Advance FEN**
(`r3kbnr/pp1b1ppp/1qn1p3/3pP3/3P4/5N1P/PP2BPP1/RNBQK2R b KQkq - 0 8`), once from each colour, us
shuffling `Nc3-Na4-Nc3-Na4-Nc3` as White and driving `Qa5+/Qb6` as Black. Fixed: only a missing
result disqualifies a game now, and short games are listed explicitly.

Across all 62 rated games, 8 ended by repetition (13%), three of them inside 10 moves.

**The obvious conclusion from that -- add contempt so we stop taking these draws -- is wrong, and
our own numbers say so.** We score **45% from level positions** (n=41, 10W 17D 14L). A draw is worth
0.50. So converting a repetition into a played-on level position trades a certain 0.50 for a
historical 0.45. Removing the two short draws does not rescue the argument either: the remaining 39
level games are 44.9%. This is consistent with the contempt SPRT already on file at -5.2 +- 19.3,
and it explains *why* that result was not the puzzle it looked like at the time.

So the picture is unchanged and now better supported: **two-thirds of our games are decided from
level, and we score 45% there.** That is the ranking. Repetition draws are a symptom of it, not a
cause, and are currently worth slightly more than the alternative. The conclusion only flips if we
become better than the field from level -- which is the thing the net is meant to do, and which
these three games cannot tell us.

## 8 Sep -- audit for the finals-boundary gap: two engine changes, SPRTs dispatched

The standing question is "we're on the boundary of finals qualification, what's wrong, and how do
we get material Elo before the freeze." Since the level-position score (45%) is the ranking and the
net is where the previous session's effort went, this pass read `engine/search.py`,
`engine/eval.py` and `engine/nnue.py` end to end looking for genuine search-side bugs and
never-tuned knobs, rather than re-opening anything already LOCKED.

**Found a real bug: quiescence had no check-evasion handling.** It applied the stand-pat cutoff and
searched only captures/promotions unconditionally, including when the side to move was in check --
where standing pat is not a legal option and the only saving move can be a quiet king step or
block. A check found at the horizon was being scored on whatever captures were lying around instead
of on whether the king actually escapes. Fixed: when checked, quiescence now searches every legal
reply (not just captures), does not decrement the capture-chain depth counter (an evasion is
forced, not optional), skips delta/SEE pruning (both assume the move can be declined), and returns
a mate score rather than a stand-pat when no legal reply exists. Verified with four targeted FENs
(forced quiet block, forced king step off an open file with no capture available, Fool's mate,
standard opening) plus a full smoke game -- all correct. SPRT dispatched: run `34282198961`,
candidate `50b2358` vs baseline `7895372` (the previously shipped commit), real control
(120000/500), 1000 games across 20 shards, seed 13.

**Found a never-tuned knob: the transposition table was 22 bits (4.2M entries, ~80 MB), against a
2 GB cap and a process RSS measured at ~605 MB after warm-up.** Nothing in `decisions.md` or here
had ever sized or measured it. At the real 120s+0.5s control a bullet game is one long search that
can turn over many multiples of 4M nodes, so a small table recycles entries under real pressure.
Quadrupled to 24 bits (~319 MB), leaving roughly 1.2 GB of headroom -- comfortable even allowing for
the Windows-vs-Linux gap the JIT budget already has to account for. The arrays are static numpy
allocations sized once at construction, so the memory delta is fixed arithmetic, not something that
needed empirical profiling. Verified with the same smoke game. SPRT queued behind the quiescence
fix (candidate `74bff0c`) so the two get independent, attributable verdicts rather than a stacked
measurement -- will dispatch once the first resolves, against baseline `50b2358`.

Also amended the stale "Pondering ships" decision (see `decisions.md`) which had never been
corrected in writing after the phantom-gain reversal, even though the code and this file already
reflected it.

Read in full and found no further candidate worth an SPRT: `engine/eval.py` (the HCE fallback,
not on the shipped net's hot path so lower leverage even if improved) and the rest of
`engine/search.py` -- null move, RFP, futility, LMR, aspiration windows, PVS, mate-distance pruning
and repetition handling are all present and none showed an obvious defect on inspection. Search-
parameter re-tuning (LMR/null-move margins) remains a candidate but is lower-confidence than a
found bug or an unmeasured resource knob, and is next if both of the above land.

## 9 Sep -- quiescence check-evasion fix: first batch INCONCLUSIVE, replication dispatched

Run `34282198961`, candidate `50b2358` vs baseline `7895372` (real control, seed 13, 1000 games,
20/20 shards reported): `+211 =574 -215`, **Elo -1.4 +- 14.1, LLR -2.59, INCONCLUSIVE** (bounds
[-2.94, 2.94] -- close to the lower one).

Essentially a wash, not the clear win the reasoning predicted. Plausible explanation: the negamax
check extension already gives an in-check node one extra full ply before it can drop into
quiescence at all, so most of the positions the fix targets were already being handled one ply
higher up: the fix's benefit is real but rare, and it costs a little -- when in check, quiescence no
longer decrements its capture-chain depth, so a check-heavy line (common in bullet, where sacrificial
attacks are frequent) can spend more nodes than before. Those two effects landing close to a wash
over 1000 games is internally consistent, not surprising in hindsight.

Per the workflow's own convention (dispatch another batch only on INCONCLUSIVE, never pool across
seeds): a second, independent batch is running now on an isolated commit pinned to exactly this
change (`qsearch-only` branch = `50b2358`), seed 17, run `34303326358`, so the two can be combined by
inverse-variance meta-analysis the way the net swap was. Decision on whether to ship pending that.

## 9 Sep -- transposition table 22 -> 24 bits: SPRT dispatched, isolated

Never tuned before this session (see the audit entry above). Tested standalone rather than stacked
on the quiescence fix, on an isolated branch (`tt-only` = `7895372` + only the TT commit
cherry-picked, `73d92fe`) so its effect is attributable on its own. Run `34303293131`, seed 19, real
control, 1000 games, vs baseline `7895372`. Result pending.

## 9 Sep -- transposition table 22 -> 24 bits: REJECTED

Run `34303293131`, candidate `73d92fe` (TT change only, isolated on the `tt-only` branch) vs
baseline `7895372`, real control, seed 19, 1000 games, 20/20 shards reported: `+194 =606 -200`,
**Elo -2.1 +- 13.5, LLR -3.02, REJECTED** (crossed the lower bound of [-2.94, 2.94]).

Quadrupling the table did not help, and if anything cost a couple of Elo -- within noise of zero
either way, but not the win the node-count reasoning predicted. Reverted to 22 bits. Read together
with the quiescence result above: the node counts a 4M-entry table actually sees at this control
were evidently not the bottleneck the reasoning assumed, which is a useful calibration on how much
headroom-based reasoning to trust without a measurement backing it. Recorded as LOCKED in
`decisions.md` so this is not re-tried without new evidence.

## 9 Sep -- quiescence check-evasion fix: replication also INCONCLUSIVE, combined verdict is a wash, REVERTED

Run `34303326358`, candidate `50b2358` (qsearch-only branch) vs baseline `7895372`, real control,
seed 17, 1000 games requested, 19/20 shards reported (one shard, `10`, was killed by the runner
after 3h13m -- "lost communication with the server", not a game or code failure -- and the workflow
combines whatever shards land, flagging a short field rather than blocking on it): `+196 =567 -187`,
950 games pooled, **Elo +3.3 +- 14.0, LLR -1.23, INCONCLUSIVE**.

Combined with the first batch by inverse-variance meta-analysis, same method as the net-swap
decision (see 8 Sep entry above):

| batch | games | Elo | 95% half-width | se |
|---|---|---|---|---|
| 34282198961, seed 13 | 1000 | -1.4 | 14.1 | 7.19 |
| 34303326358, seed 17 | 950 | +3.3 | 14.0 | 7.14 |
| inverse-variance combined | 1950 | **+1.0** | **9.9** | 5.07 |

Combined 95% CI **[-9.0, +10.9]**, P(true Elo < 0) = **0.42**. The two batches do not disagree with
each other (difference +4.7, se 10.1, z = 0.46) -- this is not one good run and one bad run, it is
two independent measurements landing on the same conclusion: **no detectable effect either way**,
at nearly a coin-flip on the sign. 1,950 games is a real sample, not a small one; this is a genuine
null result, not an underpowered one.

**REVERTED.** `engine/search.py`'s `quiescence()` is back to the pre-`50b2358` check-evasion-blind
version. The reasoning behind the fix was sound chess-engine theory -- every mature engine handles
check evasion in qsearch, and stand-pat is illegal when in check -- but sound reasoning is exactly
the thing this project has twice now watched fail to predict a measured result (the TT-size entry
immediately above, and the pondering reversal in `decisions.md`). Per that same discipline, a change
does not ship on the strength of its own justification when 1,950 games of direct measurement come
back at a coin-flip. The most likely explanation is still the one recorded after the first batch:
negamax's check extension already gives most in-check nodes one extra full ply before they can drop
into quiescence at all, so the fix's benefit is real but rare, and it costs a little in check-heavy
lines via the removed depth-decrement -- close enough to cancel out that 1,950 games cannot resolve
the sign.

Not spending a third batch on this: at 5 se combined the marginal batch buys a fraction of a se, the
feature freeze is tonight, and the runner-hours are better spent verifying the final build than
chasing a result already centred on zero. Locked in `decisions.md` alongside the TT-size entry.

---

## 9 Sep — ladder audit: 50.0% score, all 19 losses by checkmate, and where they actually break

Two narrow SPRT reversals above are not the same thing as "no improvement exists" -- they only say
those two specific search-parameter tweaks did not help. The actual ladder record settles whether
more looking is warranted: parsed all 72 rated-game logs in `logs/` (rounds 10-81; rounds 10-21 have
anonymised `[White "?"] [Black "?"]` tags and cannot have our colour inferred, so they are excluded
rather than guessed). **Result: 19 wins, 22 draws, 19 losses over 60 identified games = 50.0%.**
Every one of the 19 losses ended in checkmate -- no forfeits, no illegal moves, no clock losses, so
the shortfall is pure chess strength, not robustness.

Ran `tools/blunder_scan.py` (depth 8) against all 19 losses to separate "outplayed slowly" from
"threw a good position away in one move". Three representative samples, one per shape:

- **Round 68 vs JSP (cliff).** Level game into the middlegame; black blundered on move 19 (Bg5),
  handing us +342. Our very next move (20, Rh5) collapsed that back to roughly even, and the game
  slid from there to a loss. This is exactly what singular extensions target: the position hinged on
  one narrow winning line at a critical node and the search did not find or hold it.
- **Round 77 vs adashima (cliff, later).** We won a piece cleanly (Nxa1) around move 17, then a
  sequence of king moves (Kd2/Ke2/Kd1) 5-10 plies later threw the whole point away. Static eval
  already has a king-attack-ring term (`engine/eval.py`, `KING_ATTACK_WEIGHT`); this reads as a
  search-depth failure to see the danger far enough ahead of grabbing the piece, not a missing eval
  term.
- **Round 41 vs Something (slide, no cliff).** Down ~100-250cp from around move 10 onward with no
  single large drop anywhere in 121 moves -- a slow, structurally worse game from a middlegame
  decision, ground down over a long ending. Not a search-depth problem; out of scope for a search
  change.

Two of three sampled losses are the cliff shape a deeper, more selective search targets; the third
is a genuinely different failure mode (early positional judgement / endgame technique) that no
amount of search depth fixes on its own.

**Research**: checked current engine-development consensus (chessprogramming.org, TalkChess,
Stockfish PR history) for what a search stack at our level of maturity (NMP, RFP, LMR, check
extension, aspiration, PVS, SEE ordering, killers/history, mate-distance pruning, NNUE) is still
missing. Continuation/counter-move history is the other commonly-cited addition, but current
Stockfish PR discussion puts individual increments there at roughly ~1 Elo apiece at their level --
not worth the risk for us. TT-move singular extensions (search the position without its best move at
reduced depth; if nothing else gets close, the position hinges on that one line and it earns a ply
of extra depth; the same probe hands over multicut pruning for free if the reduced search already
clears beta by itself) are the standard next addition, reported anywhere from null to +60 Elo
depending on how sharp the rest of the search already is -- exactly the "measure it, don't reason
about it" situation this project already has a process for.

Implemented in commit `0c90a26` (`engine/search.py`): `excluded_move` threaded through every
`negamax` call site (default 0), the TT-hit early return disabled during a verification search, and
extension/multicut applied when the TT move is deep and trustworthy (`tt_depth >= depth - 3`,
`bound != UPPER`, `depth >= 8`, non-mate score). Verified before dispatch: ruff/mypy clean, perft
suite exact, `audit_truth.py` 12/12, a real game played to completion against `baselines/greedy`.

SPRT dispatched: run `34365433247`, candidate `0c90a26` vs baseline `723f699` (the last known-good
tip, i.e. after both reversions above), seed 19, real control (120000+500), elo0=0/elo1=15.

**First batch: INCONCLUSIVE.** `+53 =109 -38` over 200 games (20/20 shards), **Elo +26.1 +- 32.6,
LLR +1.03** (bounds [-2.94, 2.94]). Clearly trending positive and well above the +15 H1, but the
error bar is wide enough that it does not clear the LLR bound either way. Same shape as the
quiescence-fix first batch on 9 Sep -- replicate before concluding anything. Second batch dispatched:
run `34381293249`, same candidate/baseline, seed 23, otherwise identical settings.

**Second batch: also INCONCLUSIVE**, but the same direction. `+54 =114 -32` over 200 games (20/20
shards), **Elo +38.4 +- 31.6, LLR +1.83**.

**Combined by inverse-variance meta-analysis (same method as the 9 Sep quiescence-fix combination),
400 games total:**

| batch | games | Elo | 95% half-width | se |
|---|---|---|---|---|
| 34365433247, seed 19 | 200 | +26.1 | 32.6 | 16.63 |
| 34381293249, seed 23 | 200 | +38.4 | 31.6 | 16.12 |
| inverse-variance combined | 400 | **+32.4** | **22.7** | 11.58 |

Combined 95% CI **[9.8, 55.1]**, entirely above zero. P(true Elo < 0) = 0.0025. The two batches agree
with each other (z = 0.53, not a case of one good run and one bad one). Unlike the quiescence fix,
where 1,950 games centred tightly on zero, this is a real, clearly-positive effect that a smaller
sample can resolve because the effect size is large relative to the noise -- 400 games is enough to
separate +32 Elo from zero even though it would not be enough to separate +1 Elo from zero.

**ACCEPTED.** Singular extensions + multicut ship. Locked in `decisions.md`. `submission.zip`
rebuilt and re-verified at the main repo root per the standing process.

**Full-sample follow-up**: re-ran `blunder_scan.py` across all 19 losses, not just the three above.

```
 rd result    worst drop  after our move  ply   shape
 77 0-1              328             Ke2   21   cliff
 22 0-1              251            Rxb7   22   cliff
 68 0-1              481             Ke2   39   cliff
 76 1-0              302             Be6   36   cliff
 25 1-0              154              g5   25   slide
 51 1-0              245            Rxb8   59   slide
 56 1-0              229             Ke7   54   slide
 26 1-0              258             Qf6   68   cliff
 61 1-0              243             Bb8   65   slide
 66 0-1              419             Re5   33   cliff
 45 1-0              403             Ke4   61   cliff
 40 1-0              105            Ndc2   17   slide
 81 0-1              138             Rc1   43   slide
 32 0-1             2126             Kf3  144   cliff
 49 1-0              236              h5  110   slide
 73 0-1               74             Rd3   45   slide
 78 1-0              334             Rf1  122   cliff
 31 0-1             1050            Nxe5  176   cliff
 41 0-1              106              a4   81   slide
```

10 of 19 losses (53%) are `cliff` -- a single move of ours threw away a position that was fine or
winning the move before, per our own deeper search. 9 of 19 are `slide` -- no single bad move,
already the worse side well before the loss became inevitable. This confirms the three-sample read
above generalises: roughly half the ladder's losses are exactly the horizon-effect failure singular
extensions target, and the other half are a different problem (positional judgement / endgame
technique) that this change cannot touch. If the SPRT below comes back positive, this ~53% is the
ceiling on what it can fix -- the slide half needs eval or endgame work, not search depth, and stays
open regardless of this result.

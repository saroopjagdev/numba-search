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

# Invariants

The never-break list. If a change would violate one of these, the change is wrong.

## Rules compliance — violating any of these is disqualification

- **No third-party engine ships in the zip.** Not Stockfish, Lc0 or Maia; not a wrapper, not a
  port, not a re-implementation of one. Training data annotated by an existing engine *is*
  permitted — the ban covers what ships.
- **No engine move or evaluation database for runtime lookup.** That is an engine.
- **Any shipped network is trained by us from random initialisation.** Starting from a published
  chess net is disqualifying, fine-tuning included.
- **No native binaries.** No Cython, no compiled extensions, no shipped numba `.nbc` caches.
- **No obfuscation.** Automatic disqualification, and finalists walk a judge through the build.
  `agent.py` stays readable.
- **No network calls at runtime. No subprocess calls to external binaries.** Write nowhere but
  `/tmp`.
- **Keep the provenance trail**: training scripts, logs, checkpoints, data manifest.
- **Daily Five is the user's alone.** Rules require it be "solved alone, on your own account,
  without an engine, another person or a second account." An assistant is both. Stay out.

## Platform contract

- `agent.py` at the zip root, exposing `get_move(fen: str, time_left_ms: int) -> str`.
- ≤50 MB unzipped. 1 core, 2 GB RAM, read-only FS plus a 256 MB `/tmp` wiped between games.
- Init budget 90s hard. **Working cap 75s, target 60s.** An init overrun is an instant loss in
  every game, not a gradual weakening, and local Windows timings are not authoritative against the
  Linux container.
- **Never edit `harness/`.** It mirrors the platform protocol; changing it makes every local
  result meaningless.
- **Never name a file after a module we import.** The zip is first on `sys.path`, so `chess.py`,
  `types.py` or `random.py` would shadow the real module and fail unrecognisably.
- **Package with `--include engine`.** `harness/package.py` collects `*.py` at the root and
  nothing else, so a plain `python -m harness.package` produces a zip containing `agent.py` alone,
  which passes every local check run from the working tree and then dies on `ImportError` in every
  game. `make zip` has the flag; do not call the module directly without it.
- **Test the extracted zip, not the working tree.** Unpack it somewhere clean and play a game from
  there. It is the only check that catches a missing file, because imports resolve fine from the
  repo no matter what the archive contains.
- **A current, verified `submission.zip` always sits at the main repo root.** Uploads are manual,
  on a human account, against a 10-per-day cap and a hard close — so the file that gets grabbed
  must always be the current build. Work done in a worktree must copy its zip out to the root;
  `submission.zip` is gitignored, so pushing does not cover this. This has already gone wrong
  once: a 781-byte random-mover zip sat at the root looking current while the engine sat in a
  worktree.

## Engineering

- **The root always validates the chosen move against python-chess before returning it.** An
  illegal move loses the game on the spot.
- **Every path is wrapped so any exception still returns a legal move instantly.**
- **One thread. No `parallel=True`** — pure compile-time cost with no payoff on one core.
- **Every jitted function carries an explicit eager signature** and is warmed at import.
- **Do not import torch or onnxruntime at runtime.** Offline training only.
- **Time management is checked between jitted calls, and every call carries a node budget.**
  Numba's nopython mode has no `time` module, so the original "check the clock inside the search"
  is not available at an acceptable cost; the equivalent guarantee comes from making each depth
  (and each aspiration re-search) its own call, reading the clock before every one of them, and
  bounding each call by nodes. Anything that makes a single jitted call longer-running or
  uninterruptible breaks this.
- **The node budget never depends on the nps estimate alone.** The second cap is a multiple of the
  last completed call's node count, which cannot be poisoned by an anomalous timing sample.
- **Spend at most ~85% of the theoretical budget, and at most 25% of the remaining clock on one
  move.** `harness/referee.py:67` starts the clock *before* the request is sent, so JSON encoding
  and the pipe round-trip come out of our time, and the 500 ms watchdog grace does not save us —
  `referee.py:72` flags on `clock < 0` independently. Measured worst-case overrun of the search
  against its own budget is 1.28x; the margins above are what absorb it.
- **Bound every array index that comes from search depth or ply.** Nopython code has no bounds
  checking, so an out-of-range index is a silent out-of-bounds write and a segfault, not an
  `IndexError`. Two of these shipped in one afternoon: a transposition-table mask taken from a
  module constant instead of the array's shape, and a check extension with no ply ceiling.
- **perft is exact before any search work is trusted.** startpos perft(6) = 119,060,324.
- **Re-run `tools/jit_timing.py` after any change to a jitted function.** Compile time creeps
  invisibly and then loses every game at once.

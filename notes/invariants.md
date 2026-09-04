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

## Engineering

- **The root always validates the chosen move against python-chess before returning it.** An
  illegal move loses the game on the spot.
- **Every path is wrapped so any exception still returns a legal move instantly.**
- **One thread. No `parallel=True`** — pure compile-time cost with no payoff on one core.
- **Every jitted function carries an explicit eager signature** and is warmed at import.
- **Do not import torch or onnxruntime at runtime.** Offline training only.
- **Time management is checked inside the search**, capped near 85% of the theoretical budget:
  `harness/referee.py:67` starts the clock *before* the request is sent, so JSON encoding and the
  pipe round-trip come out of our time, and the 500 ms watchdog grace does not save us —
  `referee.py:72` flags on `clock < 0` independently.
- **perft is exact before any search work is trusted.** startpos perft(6) = 119,060,324.
- **Re-run `tools/jit_timing.py` after any change to a jitted function.** Compile time creeps
  invisibly and then loses every game at once.

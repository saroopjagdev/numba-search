AGENTS.md

## Our notes

`AGENTS.md` is upstream content and stays untouched so `git merge upstream/main` remains clean.
Everything of ours lives in `notes/`:

- [notes/invariants.md](notes/invariants.md) — the never-break list. Read this before changing
  anything that ships.
- [notes/decisions.md](notes/decisions.md) — locked architecture calls and the evidence for them,
  plus what we considered and rejected. Check here before re-opening a question.
- [notes/measurements.md](notes/measurements.md) — append-only log of JIT timings, nps figures and
  SPRT results. "Did that change help?" is answered from this file, never from memory.

Instruments live in `tools/` (never in `harness/` — editing that is forbidden):
`jit_timing.py`, `jit_stress.py`, `perft.py`, `sprt.py`.

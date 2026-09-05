"""Thinking on the opponent's clock.

`AGENTS.md` line 33: "The process keeps its core while the opponent thinks, so pondering on their
time is allowed." Between our reply and their move the core is idle, and a bullet game spends
roughly half its wall time in that state. Filling the transposition table during it is free depth.

What is pondered here is the position *after our own move*, with the opponent to play -- not a guess
at their reply. Classical pondering picks the expected reply from the principal variation and is
wasted whenever the guess is wrong, which at bullet quality is often. Searching the parent instead
means every reply they might choose is a child of what we searched, so the work is useful whichever
one arrives. It buys less depth on the specific line and never buys zero.

The whole arrangement rests on three properties, all of them checked rather than assumed:

- `negamax` is compiled `nogil=True`, so the main thread can still run Python -- accept the next
  `get_move`, and stop us -- while a search is in flight.
- The ponder searcher is a *separate* `Searcher` sharing only the transposition table. Sharing
  mutable scratch would corrupt both searches; sharing the table is the entire point.
- One core. The ponder thread is always stopped and joined before the real search starts, so the
  two never compete. That join is not best-effort: splitting a single core between two searches
  would cost far more than pondering wins.
"""

import threading

from engine.search import Searcher

# Pondering is open-ended -- it runs until the opponent moves -- so the budget handed to `search`
# exists only to bound a thread that is somehow never stopped, which would otherwise spin a core
# for the rest of the process's life. A game cannot legitimately leave us pondering this long.
MAX_PONDER_MS = 120_000.0


class Ponderer:
    """Runs a background search on the opponent's clock, into the main searcher's table."""

    def __init__(self, main: Searcher) -> None:
        # Shares the table it is filling, and the read-only network. Everything else is its own.
        self.searcher = Searcher(share=main)
        self._main = main
        self._thread: threading.Thread | None = None

    def start(self, fen: str) -> None:
        """Begin pondering `fen`. Never raises; pondering is an optimisation, not a requirement."""
        self.stop()
        try:
            self.searcher.resume()
            self.searcher.set_position(fen)
            # Carry the real game's repetition history across, so the ponder search scores
            # threefold lines the way the real one would. Without it the table would be seeded with
            # draw scores for lines that are not actually draws, and those entries are read back by
            # the search that plays the game.
            count = self._main.game_count
            self.searcher.game_keys[:count] = self._main.game_keys[:count]
            self.searcher.game_count = count
            self.searcher.record_position()
            self._thread = threading.Thread(target=self._run, name="ponder", daemon=True)
            self._thread.start()
        except Exception as error:
            print(f"ponder failed to start: {type(error).__name__}: {error}")
            self._thread = None

    def _run(self) -> None:
        try:
            self.searcher.search(MAX_PONDER_MS)
        # Deliberately bare, and on a thread whose exception would otherwise print a traceback to
        # the protocol stream and be lost anyway. A failed ponder must cost nothing but its depth.
        except Exception as error:
            print(f"ponder search failed: {type(error).__name__}: {error}")

    def stop(self) -> None:
        """Stop and join. Must complete before the real search starts -- we have one core.

        Cannot raise. `get_move` calls this before it has entered its own guard, because stopping
        has to happen before anything else touches the engine, so an exception escaping here would
        forfeit the game.
        """
        thread = self._thread
        if thread is None:
            return
        try:
            self.searcher.stop()
            # No timeout. A timeout would mean carrying on while a second search still holds the
            # core, which is the one outcome worse than never pondering. The abort is polled every
            # node, so this returns in microseconds; if it ever did not, hanging is the honest
            # failure -- a wrong-but-fast move loses the game just as surely as a slow one.
            thread.join()
        except Exception as error:
            print(f"ponder failed to stop: {type(error).__name__}: {error}")
        finally:
            self._thread = None

"""Offline NNUE training. None of this ships -- it imports torch, zstandard and orjson, none of
which exist in the match container, and `harness/package.py` never collects this directory."""

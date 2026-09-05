"""Instruments. Not shipped, not imported by the agent -- see notes/invariants.md.

This file exists so `tools` is a package. Without it mypy resolves `tools/sprt.py` as both `sprt`
and `tools.sprt` the moment anything does `from tools.x import y`, and fails the gate outright
rather than type-checking the tree.
"""

SHELL := /bin/bash

.PHONY: setup play arena zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

# --include engine is not optional. harness/package.py picks up *.py at the root and nothing else,
# so without it the zip contains agent.py alone and every game dies on ImportError at init.
zip:
	uv run python -m harness.package --include engine

gate:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

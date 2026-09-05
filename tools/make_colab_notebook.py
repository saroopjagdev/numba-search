"""Generate the Colab training notebook from the real `training/` modules.

Written as a generator rather than a hand-maintained `.ipynb` for one reason: a notebook with the
training code pasted into it starts drifting from the repository the moment either is edited, and
the drift is silent -- the notebook still runs, it just trains a different net from the one the
engine expects. Regenerating means the notebook can only ever contain what is committed.

The alternative to inlining is cloning the repository inside Colab, which needs a personal access
token for a private repo. That is friction on the one step that requires the user, so the code
travels in the notebook and the only thing that has to reach Colab separately is the data.

Run:  python -m tools.make_colab_notebook
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ("__init__.py", "dataset.py", "preprocess.py", "train.py")
OUTPUT = ROOT / "notebooks" / "colab_train.ipynb"

# Where the shards are expected once Drive is mounted, and where checkpoints go. Checkpoints must
# land on Drive rather than Colab's local disk: a free session can be reclaimed at any moment and
# local disk goes with it, which would turn a killed session into a lost run instead of a resumed
# one.
DRIVE_SHARDS = "/content/drive/MyDrive/chessathon/shards"
DRIVE_NETS = "/content/drive/MyDrive/chessathon/nets"


def markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def module_cell(name: str) -> dict[str, object]:
    """A `%%writefile` cell carrying one training module verbatim."""
    body = (ROOT / "training" / name).read_text(encoding="utf-8")
    return code(f"%%writefile training/{name}\n{body}")


def build() -> dict[str, object]:
    cells: list[dict[str, object]] = [
        markdown(
            """
# Chessathon NNUE training

Trains `(768 -> 256) x 2 -> 1` SCReLU with 8 output buckets, then quantises to int16 and writes
`net256.npz` -- the file the engine loads. Nothing here ships; only the `.npz` does.

**Before running:** set *Runtime -> Change runtime type -> T4 GPU*, and put the shard files in
Google Drive under `MyDrive/chessathon/shards/`. A dozen shards (about 2.3 GB) is plenty: they were
written by scattering records uniformly at random, so any subset of them is already a uniform
random sample of the whole database and needs no reshuffling.

The code below is generated from the repository by `tools/make_colab_notebook.py`. Do not edit it
here -- edit the repository and regenerate, or the net you train stops matching the net the engine
expects.
"""
        ),
        markdown("## 1. Check the GPU actually attached"),
        code(
            """
import subprocess
import torch

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
      or "NO GPU -- set Runtime > Change runtime type > T4 GPU")
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
"""
        ),
        markdown(
            """
## 2. Write the training code

Verbatim copies of the repository modules. `numba` and `torch` are preinstalled on Colab; the
`zstandard`/`orjson` install is only needed if you use the optional preprocessing cell.
"""
        ),
        code("import pathlib\npathlib.Path('training').mkdir(exist_ok=True)"),
    ]
    cells.extend(module_cell(name) for name in MODULES)

    cells.extend(
        [
            markdown("## 3. Mount Drive and confirm the data is there"),
            code(
                f"""
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

shards = Path({DRIVE_SHARDS!r})
nets = Path({DRIVE_NETS!r})
nets.mkdir(parents=True, exist_ok=True)

files = sorted(shards.glob('shard*.bin'))
total = sum(path.stat().st_size for path in files)
print(f"{{len(files)}} shards, {{total / 1e9:.2f}} GB, about {{total // 28:,}} positions")
assert files, f"no shard*.bin in {{shards}} -- upload some first"
"""
            ),
            markdown(
                """
## 4. Throughput probe -- run this before committing to a long run

400 steps, to check the GPU is doing what the profile says it should. Measured on the laptop, a
step at batch 16,384 is **98.5% forward/backward** and only 24.7 ms of CPU work -- read, decode and
tensor construction. So the CPU floor is about **664,000 pos/s** and everything below that is the
GPU's to win. Two vCPUs are not a constraint: the decode is a single serial thread of about 18 ms,
so core *count* is irrelevant here.

Expect comfortably over 100,000 pos/s. Under about 50,000 means something is wrong -- no GPU
actually attached, or Drive I/O throttling the shard reads -- and is worth fixing before spending
an hour on the real run rather than after.
"""
            ),
            code(
                f"""
!python -m training.train \\
    --shards {DRIVE_SHARDS} \\
    --output /content/probe.npz \\
    --steps 400 --holdout 2
"""
            ),
            markdown(
                """
## 5. The real run

Checkpoints every 5,000 steps straight to Drive, so a reclaimed session costs the steps since the
last checkpoint rather than the whole run. `--steps 60000` is about 2.8 passes over 354M records at
the full dataset; scale it down in proportion if you uploaded fewer shards.
"""
            ),
            code(
                f"""
!python -m training.train \\
    --shards {DRIVE_SHARDS} \\
    --output {DRIVE_NETS}/net256.npz \\
    --hidden 256 --steps 60000 --holdout 2
"""
            ),
            markdown(
                """
## 6. Width A/B

The one number the plan says we must measure ourselves, because published width deltas come from
C++ engines at different time controls and do not transfer. All four write separate files; SPRT
picks the winner in the engine, at our own time control. This is the actual reason to be on a GPU
at all -- it is four runs, not one.
"""
            ),
            code(
                f"""
# One line per run on purpose: IPython's `!` takes a single line, and backslash continuation
# inside a loop body is not reliably joined before the shell sees it.
for width in (128, 512, 1024):
    print(f"=== hidden {{width}} ===", flush=True)
    !python -m training.train --shards {DRIVE_SHARDS}"""
                f""" --output {DRIVE_NETS}/net{{width}}.npz"""
                """ --hidden {width} --steps 60000 --holdout 2
"""
            ),
            markdown(
                """
## 7. Bring the weights home

`net256.npz` is about 400 KB. It goes to `weights/nnue.npz` in the repository, where
`harness/package.py` already picks up a root-level `weights` directory by default.
"""
            ),
            code(
                f"""
from google.colab import files

files.download('{DRIVE_NETS}/net256.npz')
"""
            ),
            markdown(
                """
## Optional: rebuild the shards here instead of uploading them

Only worth it if uploading is impossible. Colab downloads the 21.7 GB database at datacenter speed,
but preprocessing is CPU-bound and a free instance has two vCPUs against the laptop's nine workers,
so this is slower than uploading a subset over most home connections. `--limit` takes a *prefix* of
the database rather than a random sample, which is a real caveat: the ordering is not documented,
so a prefix is not guaranteed to be representative the way a subset of shards is.
"""
            ),
            code(
                f"""
!pip install -q zstandard orjson
!wget -c https://database.lichess.org/lichess_db_eval.jsonl.zst -O /content/eval.jsonl.zst
!python -m training.preprocess \\
    --input /content/eval.jsonl.zst \\
    --output {DRIVE_SHARDS} \\
    --limit 100000000
"""
            ),
        ]
    )

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

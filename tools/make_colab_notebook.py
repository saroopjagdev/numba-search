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

# Twice the 60,000 steps the shipped net was trained for. The width sweep found evaluation quality
# rising +16 Elo per doubling of width against roughly +48 in published work, on a curve that is
# linear rather than flattening -- the signature of a net whose *capacity* is not the binding
# constraint. Training longer at the width we already ship is the direct test of that, and unlike a
# width change it costs nothing per node at match time.
#
# Doubling rather than quadrupling because there is no resume. `train.py` refuses to hand back a net
# stranded mid-cosine-schedule, quite rightly, so a reclaimed session is a total loss rather than a
# shortened run. At the measured 450-550k pos/s on a T4, 120,000 steps is about 65 minutes; 240,000
# would be nearer two and a half hours and is a much worse bet against a free session with one day
# left before the freeze.
STEPS = 120_000
BATCH = 16_384


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
Google Drive under `MyDrive/chessathon/shards/`.

**Upload all 64: `shard00.bin` to `shard63.bin`, 8.42 GB, 263,154,771 positions.** They are written
locally to `C:\\Users\\ssjag\\chessdata\\shards_quiet\\` -- note *shards_quiet*, not `shards`, which
is the older unfiltered corpus kept only as a control.

These are the *filtered* shards: positions in check, and positions whose best move is a capture or
promotion, have been removed, because the net only ever scores the leaves of a search that has
already resolved captures. That filter is what makes the whole corpus fit: 8.42 GB sits inside
Drive's 15 GB free tier with room to spare, where the unfiltered 12.09 GB did not, and uploading a
subset is no longer a trade worth making.

Filtering shifts the labels, and that is expected rather than a defect. Mean score for the side to
move goes from +42.6 cp unfiltered to +75.3 cp filtered, standard deviation essentially unchanged
at 789 cp: discarding positions whose best move is a capture discards the ones where the mover is
about to win material back, and what is left is the quiet distribution the net is actually asked to
score at search leaves.

`train.py --holdout 2` reserves `shard62`-`shard63` for validation, so 62 shards, ~255M positions,
actually train:

| | positions | epochs at 120k steps | positions/parameter, 256 |
|---|---|---|---|
| 62 training shards | 255M | 7.7 | 1268 |

The 256-wide net has ~201k parameters, so 255M positions is a large corpus relative to capacity and
7.7 passes over it is not an overfitting risk -- which is the whole reason this run is worth making.

Records were scattered across the 64 shards uniformly at random when they were written, so any
subset is already a uniform random sample of the database and needs no reshuffling. That still
holds -- it is simply no longer needed.

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

from training.dataset import RECORD_SIZE

drive.mount('/content/drive')

shards = Path({DRIVE_SHARDS!r})
nets = Path({DRIVE_NETS!r})
nets.mkdir(parents=True, exist_ok=True)

EXPECTED_SHARDS = 64

files = sorted(shards.glob('shard*.bin'))
assert files, f"no shard*.bin in {{shards}} -- upload some first"

# Drive keeps syncing after the folder looks present. Runs have already started on 44, 55 and 60
# of the 64 shards and finished without complaining, which is the worst kind of failure here: the
# width A/B would have compared nets trained on different amounts of data and reported it as an
# effect of width. Refuse to start rather than measure sync progress.
assert len(files) == EXPECTED_SHARDS, (
    f"{{len(files)}} of {{EXPECTED_SHARDS}} shards visible -- Drive is still syncing."
    " Wait and re-run this cell; do not train on a partial corpus."
)
truncated = [path.name for path in files if path.stat().st_size % RECORD_SIZE]
assert not truncated, f"partially synced shards: {{truncated}}"

total = sum(path.stat().st_size for path in files)
positions = total // RECORD_SIZE
print(f"{{len(files)}} shards, {{total / 1e9:.2f}} GB, {{positions:,}} positions")
print(f"{STEPS:,} steps x {BATCH:,} = {{{STEPS} * {BATCH} / positions:.1f}} epochs over this data")
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
                f"""
## 5. The real run

**Do not change `--steps {STEPS}` or `--hidden 256`, and do not stop the cell early.** The cosine
learning-rate decay is built around the step count it is given, so a run cut short is stranded at a
high learning rate. `train.py` refuses to hand back such a net rather than let it reach a
measurement, which means an interrupted run is a lost run, not a shorter one. About 65 minutes on a
T4 at the 450-550k pos/s this has measured before.

Checkpoints land on Drive every 5,000 steps. They are worth having as evidence of progress, but for
the reason above they are *not* shippable nets -- only the final write is.

Why {STEPS:,} and not the 60,000 the shipped net used: the width sweep priced 256 as an interior
optimum, with 512 at -38 Elo and 1024 at -56, so no architectural change is left. But it also found
quality rising only +16 Elo per doubling of width where published work sees about +48, on a linear
rather than flattening curve. That is what an undertrained net looks like, not a saturated one. More
steps at the width we already ship is the direct test, and it costs nothing per node in the match.
"""
            ),
            code(
                f"""
!python -m training.train \\
    --shards {DRIVE_SHARDS} \\
    --output {DRIVE_NETS}/net256_long.npz \\
    --hidden 256 --steps {STEPS} --holdout 2
"""
            ),
            markdown(
                """
## 6. Width is already settled -- do not re-run it

An earlier version of this notebook trained 128 / 512 / 1024 here. That question is closed:
256 is an interior optimum, 512 loses 38 Elo and 1024 loses 56, because the extra node cost outruns
the quality gain at our time control. All four nets are in `net-files/`. Re-running them would spend
a GPU session re-deriving a number we already have.
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
from pathlib import Path

# Both files, and only the ones that exist. The `.npz` is the quantised net the engine loads; the
# `.pt` is the float model, and without it a run can only be repeated from scratch rather than
# continued or re-quantised. Drive keeps them too, but a reclaimed session is a bad moment to
# discover the local copy was never taken.
nets = Path('{DRIVE_NETS}')
for net in sorted(list(nets.glob('net*.npz')) + list(nets.glob('net*.pt'))):
    print(f"downloading {{net.name}} ({{net.stat().st_size / 1024:.0f}} KB)")
    files.download(str(net))
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

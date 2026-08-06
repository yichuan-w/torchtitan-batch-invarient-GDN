# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a Terminal-Bench 2.0 EVAL JSONL in the tmax task schema.

TB-2.0 is the held-out benchmark the TMax paper reports on; AI2 does not ship it
in the tmax/open-instruct format (their tmax datasets are all training). The
canonical TB-2.0 task set is published on the Hub as a Harbor task tree (89 tasks,
each with a PREBUILT public docker image), e.g. ``harborframework/terminal-bench-2.0``::

    <task>/task.toml             # [environment].docker_image = "<public image>"
    <task>/instruction.md        # the agent instruction (our problem_statement)
    <task>/environment/Dockerfile# WORKDIR + COPY protected.tar.gz.enc (baked image)
    <task>/tests/test.sh         # verifier: writes /logs/verifier/reward.txt (0/1)
    <task>/tests/test_outputs.py # + any other test helpers (grade-time fixtures)
    <task>/solution/solve.sh     # oracle solution (unused for eval)

Each output row is exactly what ``TMaxDataset`` (data.py) consumes -- the same
R2E-compatible schema, with a ``tmax`` metadata blob::

    {
      "prompt": <instruction.md>,
      "label":  <task_id>,
      "metadata": {
        "instance_id", "image" (docker.io/...), "workdir",
        "problem_statement": <instruction.md>,
        "tmax": {"test_sh", "fixtures": {relpath: content}, "reward_path"}
      }
    }

Why this maps onto the tmax path unchanged:
  - The published ``docker_image`` boots directly as the Daytona sandbox (protected
    test payload + env are baked in, exactly like the tmax corpus setup.sh images).
  - ``tests/test.sh`` already writes ``/logs/verifier/reward.txt`` with 0/1 -- the
    tmax reward contract (grading.py). Reward = that value, binary/sparse.
  - Fixtures are the ``tests/`` tree minus ``test.sh`` (uploaded to /tests/ at grade
    time). Unlike the tmax corpus we do NOT upload ``environment/`` -- those files
    are baked into the published image (the Dockerfile copies them in at build).

Run with a python that has ``huggingface_hub`` (HF_TOKEN set)::

    python -m torchtitan.experiments.rl.examples.tmax.prepare_tb2_data \
        --out <launcher>/swe_assets/tb2_eval.jsonl

``--limit N`` emits only the first N tasks (smoke).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from torchtitan.experiments.rl.examples.tmax.prepare_tmax_data import (
    _DEFAULT_IMAGE_PREFIX,
    _REWARD_PATH,
)

_HF_REPO = "harborframework/terminal-bench-2.0"

# TB-2.0 grade-time inputs live under ``tests/`` (test.sh is uploaded separately).
# The task environment is baked into the published image, so -- unlike the tmax
# corpus -- there is no ``environment/seeds`` tree to seed into the workdir.
_FIXTURE_ROOT = "tests"

# Fallback workdir when the task Dockerfile declares no WORKDIR. TB-2.0 images
# overwhelmingly use /app.
_DEFAULT_WORKDIR = "/app"


def _download() -> str:
    """Download the full TB-2.0 task tree; return the local snapshot dir."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=_HF_REPO, repo_type="dataset")


def _docker_image(task_dir: str) -> str | None:
    """Read ``[environment].docker_image`` from the task's task.toml."""
    toml_path = os.path.join(task_dir, "task.toml")
    if not os.path.exists(toml_path):
        return None
    with open(toml_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r'docker_image\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


def _workdir_from_dockerfile(task_dir: str) -> str:
    """Read the last ``WORKDIR`` from the task's environment Dockerfile.

    The published image lands the agent in its final WORKDIR; our harness cd's
    there per bash command, so it must match. Fall back to /app when absent.
    """
    dockerfile = os.path.join(task_dir, "environment", "Dockerfile")
    if not os.path.exists(dockerfile):
        return _DEFAULT_WORKDIR
    workdir = _DEFAULT_WORKDIR
    with open(dockerfile, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*WORKDIR\s+(\S+)", line)
            if m:
                workdir = m.group(1)
    return workdir


def _collect_fixtures(task_dir: str) -> dict[str, str]:
    """Gather ``{relpath: content}`` for every text file under ``tests/`` EXCEPT
    ``tests/test.sh`` (uploaded separately). Relpaths are relative to the task dir
    (e.g. ``tests/test_outputs.py``); grading.py maps ``tests/*`` -> ``/tests/*``."""
    fixtures: dict[str, str] = {}
    base = os.path.join(task_dir, _FIXTURE_ROOT)
    if not os.path.isdir(base):
        return fixtures
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            abspath = os.path.join(dirpath, fn)
            rel = os.path.relpath(abspath, task_dir)
            if rel == os.path.join("tests", "test.sh"):
                continue
            try:
                with open(abspath, encoding="utf-8") as f:
                    fixtures[rel] = f.read()
            except (UnicodeDecodeError, OSError):
                # Skip binary/unreadable fixtures (e.g. protected blobs); the
                # graded test payload is text and lives under tests/.
                continue
    return fixtures


def _to_row(task_id: str, task_dir: str, *, image_prefix: str) -> dict | None:
    """Build one output row from an extracted TB-2.0 task dir."""
    instr_path = os.path.join(task_dir, "instruction.md")
    test_path = os.path.join(task_dir, "tests", "test.sh")
    image = _docker_image(task_dir)
    if not (os.path.exists(instr_path) and os.path.exists(test_path) and image):
        return None
    with open(instr_path, encoding="utf-8") as f:
        instruction = f.read()
    with open(test_path, encoding="utf-8") as f:
        test_sh = f.read()
    if not instruction.strip() or not test_sh.strip():
        return None

    if image_prefix and "/" in image and not image.startswith(image_prefix):
        image = image_prefix + image

    return {
        "prompt": instruction,
        "label": task_id,
        "metadata": {
            "instance_id": task_id,
            "image": image,
            "workdir": _workdir_from_dockerfile(task_dir),
            "problem_statement": instruction,
            "tmax": {
                "test_sh": test_sh,
                "fixtures": _collect_fixtures(task_dir),
                "reward_path": _REWARD_PATH,
            },
        },
    }


def build_rows(
    *, limit: int | None = None, image_prefix: str = _DEFAULT_IMAGE_PREFIX
) -> list[dict]:
    """Download the TB-2.0 task tree and convert each task dir to an output row."""
    snap = _download()
    out: list[dict] = []
    for entry in sorted(os.listdir(snap)):
        task_dir = os.path.join(snap, entry)
        if not os.path.isdir(task_dir) or not os.path.exists(
            os.path.join(task_dir, "task.toml")
        ):
            continue
        row = _to_row(entry, task_dir, image_prefix=image_prefix)
        if row is not None:
            out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def _write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output tb2_eval.jsonl path")
    ap.add_argument(
        "--limit", type=int, default=None, help="emit only the first N tasks (smoke)"
    )
    ap.add_argument("--image-prefix", default=_DEFAULT_IMAGE_PREFIX)
    args = ap.parse_args()

    rows = build_rows(limit=args.limit, image_prefix=args.image_prefix)
    if not rows:
        print("ERROR: produced 0 rows", file=sys.stderr)
        sys.exit(1)
    _write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} TB-2.0 tasks -> {args.out}")


if __name__ == "__main__":
    main()

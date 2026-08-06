# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone Daytona smoke test for the tmax grading path (NO training stack).

Loads ``tmax_smoke.jsonl``, boots a real Daytona sandbox from each task's image,
runs a TRIVIAL scripted "agent" (actually solves the openthoughts join task;
otherwise just cd's into the workdir), then calls ``grade_tmax_daytona`` and prints
the reward. This proves the sandbox boot -> fixture upload -> ``bash /tests/test.sh``
-> reward.txt path end to end without vLLM / torchtitan.

Run with the daytona-only venv (import daytona), e.g.::

    DAYTONA_API_KEY=dtn_... https_proxy=http://<your-egress-proxy>:8080 \
        <your-venv>/bin/python \
        torchtitan/experiments/rl/examples/tmax/local_smoke.py \
        --data torchtitan/experiments/rl/examples/tmax/tmax_smoke.jsonl --limit 2

The high-level Daytona SDK ignores http_proxy, so we monkeypatch the API client
config to set the proxy BEFORE creating the client (verified working pattern).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

# --- Daytona proxy monkeypatch: the high-level SDK ignores http_proxy, so set the
# proxy on the API client config BEFORE creating any client. ---------------------
import daytona_api_client

_ORIG_CFG_INIT = daytona_api_client.Configuration.__init__


def _patched_cfg_init(self, *args, **kwargs):
    _ORIG_CFG_INIT(self, *args, **kwargs)
    self.proxy = os.environ.get("https_proxy")


daytona_api_client.Configuration.__init__ = _patched_cfg_init

from daytona import (  # noqa: E402 -- must import after the proxy patch
    CreateSandboxFromImageParams,
    Daytona,
    Resources,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_grading():
    """Import grading.py as a standalone module (its torchtitan import is
    TYPE_CHECKING-only, so it loads cleanly in the daytona-only venv) -- avoids
    triggering the package __init__ which pulls in the full training stack."""
    spec = importlib.util.spec_from_file_location(
        "tmax_grading", os.path.join(_HERE, "grading.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scripted_agent(sb, sample: dict) -> None:
    """A trivial scripted 'agent' standing in for the RL policy.

    For the openthoughts join task, actually solve it (produce
    /output/command_capture.txt). For any other task, just cd into the workdir --
    enough to exercise the boot + grade path (the verifier then reports whatever
    the untouched image yields).
    """
    workdir = sample["metadata"].get("workdir", "/workspace")
    instruction = sample["metadata"].get("problem_statement", "")
    sb.process.exec(f"mkdir -p {workdir}", timeout=60)

    is_join = "command_capture.txt" in instruction and "/output" in instruction
    if is_join:
        # The aa/bb seed inputs are already placed under /workspace by
        # seed_workspace_daytona (the same path the rollouter uses), so a real agent
        # can read them here. Do a full outer join by first word emitting
        # "<key> <aa field 2>" (empty when the key is bb-only), captured
        # (stdout+stderr) to /output/command_capture.txt. Matches
        # tests/expected_output.txt.
        # Sort into temp files then join (avoids process-substitution quirks under
        # the daytona session shell); -a1 -a2 = full outer join, -e '' = empty for
        # missing, -o '0,1.2' = join key + second field of file 1 (aa).
        solve = (
            "mkdir -p /output; "
            f"cd {workdir} && "
            "sort aa > /tmp/_saa && sort bb > /tmp/_sbb && "
            "join -a1 -a2 -e '' -o '0,1.2' /tmp/_saa /tmp/_sbb "
            "> /output/command_capture.txt 2>&1; echo EXIT=$?"
        )
        r = sb.process.exec(f"bash -c {json.dumps(solve)}", timeout=120)
        print(f"    [agent] join solve exit={r.exit_code} {r.result.strip()}")
    else:
        r = sb.process.exec(f"cd {workdir} && pwd && ls -la", timeout=60)
        print(f"    [agent] cd {workdir} exit={r.exit_code}")


def _run_one(client: Daytona, grading, sample: dict) -> float:
    md = sample["metadata"]
    image = md["image"]
    workdir = md.get("workdir", "/workspace")
    tmax = md["tmax"]
    instance_id = md.get("instance_id", "?")
    print(f"  task={instance_id} image={image} workdir={workdir}")

    sb = client.create(
        CreateSandboxFromImageParams(
            image=image,
            os_user="root",
            ephemeral=True,
            resources=Resources(cpu=2, memory=6, disk=10),
        )
    )
    try:
        # Seed agent inputs (environment/seeds/* -> /workspace) BEFORE the agent,
        # the SAME way the rollouter does, so this smoke exercises the real path.
        grading.seed_workspace_daytona(sb, tmax)
        _scripted_agent(sb, sample)
        reward = grading.grade_tmax_daytona(sb, tmax, workdir=workdir)
        print(f"  -> reward={reward}")
        return reward
    finally:
        try:
            sb.delete()
        except Exception as e:  # noqa: BLE001 -- best-effort cleanup
            print(f"    [warn] sandbox delete failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        default=os.path.join(_HERE, "tmax_smoke.jsonl"),
        help="path to tmax_smoke.jsonl",
    )
    ap.add_argument("--limit", type=int, default=2, help="number of tasks to run")
    args = ap.parse_args()

    if not os.environ.get("DAYTONA_API_KEY"):
        print("ERROR: DAYTONA_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    with open(args.data) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    samples = samples[: args.limit]
    if not samples:
        print(f"ERROR: no samples in {args.data}", file=sys.stderr)
        sys.exit(1)

    grading = _load_grading()
    client = Daytona()

    print(f"Running {len(samples)} tmax smoke task(s) from {args.data}")
    results: list[tuple[str, float]] = []
    for s in samples:
        rid = s["metadata"].get("instance_id", "?")
        try:
            reward = _run_one(client, grading, s)
        except Exception as e:  # noqa: BLE001 -- report per-task and continue
            print(f"  task={rid} FAILED: {type(e).__name__}: {e}")
            reward = -1.0
        results.append((rid, reward))

    print("\n=== SMOKE RESULTS ===")
    for rid, reward in results:
        print(f"  {rid}: reward={reward}")


if __name__ == "__main__":
    main()

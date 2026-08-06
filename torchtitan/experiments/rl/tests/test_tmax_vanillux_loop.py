# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from torchtitan.experiments.rl.examples.tmax import vanillux_loop
from torchtitan.experiments.rl.examples.tmax.data import TMaxSample
from torchtitan.experiments.rl.examples.tmax.rollouter import (
    _finish_reason_metrics,
    _sandbox_issue_metrics,
    _SandboxRolloutDiagnostics,
    TMaxRollouter,
)
from torchtitan.experiments.rl.harness import SandboxIssue
from torchtitan.experiments.rl.observability.metrics import Mean
from torchtitan.experiments.rl.rollout.types import Rollout, RolloutStatus


_RUN_BASH = vanillux_loop._run_bash


def _patch_self_healing_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    wrapper_path = tmp_path / "wrapper.sh"
    cwd_path = tmp_path / "cwd"
    env_path = tmp_path / "env"
    wrapper = (
        vanillux_loop._BASH_WRAPPER.replace(vanillux_loop._BASH_CWD_PATH, str(cwd_path))
        .replace(vanillux_loop._BASH_ENV_PATH, str(env_path))
        .replace(vanillux_loop._BASH_DEFAULT_CWD, str(tmp_path))
    )
    monkeypatch.setattr(vanillux_loop, "_BASH_WRAPPER_PATH", str(wrapper_path))
    monkeypatch.setattr(vanillux_loop, "_BASH_CWD_PATH", str(cwd_path))
    monkeypatch.setattr(vanillux_loop, "_BASH_ENV_PATH", str(env_path))
    monkeypatch.setattr(
        vanillux_loop, "_BASH_COMMAND_PATH_PREFIX", str(tmp_path / "command.")
    )
    monkeypatch.setattr(vanillux_loop, "_BASH_DEFAULT_CWD", str(tmp_path))
    monkeypatch.setattr(vanillux_loop, "_BASH_WRAPPER", wrapper)
    return wrapper_path, cwd_path, env_path


class _FakeAdapter:
    def __init__(self, responses: list[dict | None]) -> None:
        self._responses = iter(responses)

    async def complete(self, session_id: str, payload: dict) -> dict | None:
        return next(self._responses, None)


@pytest.fixture(autouse=True)
def _patch_sandbox_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    async def prepare_runtime(sb: Any) -> None:
        pass

    async def run_bash(sb: Any, command: str, timeout: int) -> tuple[str, int]:
        return "command output", 0

    monkeypatch.setattr(vanillux_loop, "_prepare_runtime", prepare_runtime)
    monkeypatch.setattr(vanillux_loop, "_run_bash", run_bash)


def _tool_response() -> dict:
    return {
        "content": [
            {
                "type": "tool_use",
                "name": "bash",
                "id": "call-1",
                "input": {"command": "echo ok"},
            }
        ],
        "stop_reason": "tool_use",
    }


def _run_loop(
    responses: list[dict | None],
    *,
    time_budget_sec: int = 60,
    max_turns: int = 1,
) -> tuple[int, bool, int, str]:
    sandbox: Any = object()
    adapter: Any = _FakeAdapter(responses)
    return asyncio.run(
        vanillux_loop.run_vanillux_loop(
            sandbox,
            task="test task",
            session_id="group=0/rollout=0",
            adapter=adapter,
            time_budget_sec=time_budget_sec,
            max_turns=max_turns,
        )
    )


def test_persistent_bash_wrapper_avoids_pkill_pattern_self_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    wrapper_path = tmp_path / "wrapper.sh"
    cwd_path = tmp_path / "cwd"
    env_path = tmp_path / "env"
    wrapper = vanillux_loop._BASH_WRAPPER.replace(
        vanillux_loop._BASH_CWD_PATH, str(cwd_path)
    ).replace(vanillux_loop._BASH_ENV_PATH, str(env_path))
    wrapper_path.write_text(wrapper)
    wrapper_path.chmod(0o755)
    cwd_path.write_text(str(tmp_path))
    env_path.touch()
    monkeypatch.setattr(vanillux_loop, "_BASH_WRAPPER_PATH", str(wrapper_path))

    class LocalSandbox:
        command = ""

        async def exec(self, command: str, **kwargs) -> tuple[int, str, str]:
            self.command = command
            completed = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                check=False,
                text=True,
                timeout=kwargs["timeout"] + 2,
            )
            return completed.returncode, completed.stdout, completed.stderr

    sandbox = LocalSandbox()
    pattern = f"tt_vanillux_wrapper_{os.getpid()}_{time.time_ns()}"
    command = f"pkill -f {shlex.quote(pattern)} || :; printf survived"

    output, exit_code = asyncio.run(_RUN_BASH(sandbox, command, 2))

    assert command not in sandbox.command
    assert (output, exit_code) == ("survived", 0)


def test_persistent_bash_wrapper_preserves_command_argument_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    wrapper_path = tmp_path / "wrapper.sh"
    cwd_path = tmp_path / "cwd"
    env_path = tmp_path / "env"
    wrapper = vanillux_loop._BASH_WRAPPER.replace(
        vanillux_loop._BASH_CWD_PATH, str(cwd_path)
    ).replace(vanillux_loop._BASH_ENV_PATH, str(env_path))
    wrapper_path.write_text(wrapper)
    wrapper_path.chmod(0o755)
    cwd_path.write_text(str(tmp_path))
    env_path.touch()
    monkeypatch.setattr(vanillux_loop, "_BASH_WRAPPER_PATH", str(wrapper_path))

    class LocalSandbox:
        async def exec(self, command: str, **kwargs) -> tuple[int, str, str]:
            completed = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                check=False,
                text=True,
                timeout=kwargs["timeout"] + 2,
            )
            return completed.returncode, completed.stdout, completed.stderr

    command = "printf '%s' \"$1\""

    output, exit_code = asyncio.run(_RUN_BASH(LocalSandbox(), command, 2))

    assert (output, exit_code) == (command, 0)

    sandbox = LocalSandbox()
    assert asyncio.run(_RUN_BASH(sandbox, "export _command=preserved", 2)) == (
        "",
        0,
    )
    assert asyncio.run(_RUN_BASH(sandbox, "printf '%s' \"$_command\"", 2)) == (
        "preserved",
        0,
    )
    assert asyncio.run(_RUN_BASH(sandbox, "export PATH=/tmp/missing", 2)) == ("", 0)
    assert asyncio.run(_RUN_BASH(sandbox, "printf survived", 2)) == ("survived", 0)


def test_persistent_bash_wrapper_recovers_after_tmp_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    wrapper_path, cwd_path, env_path = _patch_self_healing_runtime(
        monkeypatch, tmp_path
    )

    class LocalSandbox:
        async def exec(self, command: str, **kwargs) -> tuple[int, str, str]:
            completed = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                check=False,
                text=True,
                timeout=kwargs["timeout"] + 2,
            )
            return completed.returncode, completed.stdout, completed.stderr

    cleanup = shlex.join(["rm", "-f", str(wrapper_path), str(cwd_path), str(env_path)])
    sandbox = LocalSandbox()

    assert asyncio.run(_RUN_BASH(sandbox, f"{cleanup}; printf first", 2)) == (
        "first",
        0,
    )
    assert not wrapper_path.exists()
    cwd_path.unlink(missing_ok=True)
    env_path.unlink(missing_ok=True)
    assert asyncio.run(_RUN_BASH(sandbox, "printf second", 2)) == ("second", 0)
    assert wrapper_path.is_file() and cwd_path.is_file() and env_path.is_file()


def test_persistent_bash_wrapper_handles_large_command_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_self_healing_runtime(monkeypatch, tmp_path)

    class LocalSandbox:
        async def exec(self, command: str, **kwargs) -> tuple[int, str, str]:
            transport_path = tmp_path / "transport.sh"
            transport_path.write_text(command)
            completed = subprocess.run(
                ["bash", str(transport_path)],
                capture_output=True,
                check=False,
                text=True,
                timeout=kwargs["timeout"] + 2,
            )
            return completed.returncode, completed.stdout, completed.stderr

    command = "printf large-command-ok; # " + "x" * 256_000

    assert asyncio.run(_RUN_BASH(LocalSandbox(), command, 2)) == (
        "large-command-ok",
        0,
    )


def test_finish_reason_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_bash(sb: Any, command: str, timeout: int) -> tuple[str, int]:
        return vanillux_loop.SUBMIT_MARKER, 0

    monkeypatch.setattr(vanillux_loop, "_run_bash", run_bash)

    assert _run_loop([_tool_response()]) == (1, True, 0, "submit")


def test_finish_reason_hit_max_turns() -> None:
    assert _run_loop([_tool_response()]) == (1, False, 0, "hit_max_turns")


def test_finish_reason_hit_time_budget() -> None:
    assert _run_loop([], time_budget_sec=0) == (
        0,
        False,
        0,
        "hit_time_budget",
    )


def test_finish_reason_stopped_early() -> None:
    assert _run_loop([]) == (0, False, 0, "stopped_early")


def test_format_error_stops_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vanillux_loop, "_FORMAT_ERROR_FEEDBACK", False)
    response = {"content": [{"type": "text", "text": "not a tool call"}]}

    assert _run_loop([response]) == (1, False, 1, "stopped_early")


def test_finish_reason_metrics_are_exhaustive_fractions() -> None:
    metrics = _finish_reason_metrics(
        ["submit", "submit", "hit_time_budget", "stopped_early", "error"]
    )
    fractions = {}
    for metric in metrics:
        assert isinstance(metric.value, Mean)
        fractions[metric.key] = metric.value.value / metric.value.count

    assert fractions == {
        "rollout/finish_submit_frac": 0.4,
        "rollout/finish_hit_max_turns_frac": 0.0,
        "rollout/finish_hit_time_budget_frac": 0.2,
        "rollout/finish_stopped_early_frac": 0.2,
        "rollout/finish_error_frac": 0.2,
    }
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_sandbox_issue_metrics_count_events_and_affected_rollouts() -> None:
    metrics = _sandbox_issue_metrics(
        [
            {},
            {"command_disk_exhausted": 1},
            {
                "execute_response_recovered": 1,
                "command_output_retry": 1,
            },
            {},
        ]
    )
    values = {}
    for metric in metrics:
        assert isinstance(metric.value, Mean)
        values[metric.key] = metric.value.value / metric.value.count

    assert values == {
        "rollout/sandbox_issue_frac": 0.5,
        "rollout/sandbox_issue_events_mean": 0.75,
        "rollout/sandbox_disk_full_frac": 0.25,
        "rollout/sandbox_disk_full_events_mean": 0.25,
        "rollout/sandbox_transport_issue_frac": 0.25,
        "rollout/sandbox_transport_issue_events_mean": 0.5,
        "rollout/sandbox_provision_issue_frac": 0.0,
        "rollout/sandbox_timeout_frac": 0.0,
    }


def test_infra_failed_sibling_is_scored_as_zero_reward() -> None:
    rollouter = object.__new__(TMaxRollouter)
    rollouter._ensure_adapter = AsyncMock(return_value=object())

    async def run_sibling(**kwargs):
        rollout_idx = kwargs["rollout_idx"]
        infra_failed = rollout_idx == 1
        return (
            Rollout(
                group_id=7,
                rollout_id=rollout_idx,
                status=(
                    RolloutStatus.ERROR if infra_failed else RolloutStatus.COMPLETED
                ),
            ),
            not infra_failed,
            0,
            "error" if infra_failed else "submit",
            _SandboxRolloutDiagnostics(
                sandbox_id=f"sandbox-{rollout_idx}",
                disk_gb=6,
                issue_counts={},
                issues=(),
                num_dropped_details=0,
                infra_failed=infra_failed,
            ),
        )

    rollouter._run_agent_rollout = AsyncMock(side_effect=run_sibling)

    async def score_group(rollouts, sample):
        del sample
        return [
            Mock(reward=1.0, reward_breakdown={}),
            Mock(reward=0.0, reward_breakdown={}),
        ]

    rollouter.score_group = AsyncMock(side_effect=score_group)
    rollouter.advantage_estimator = Mock(return_value=[0.5, -0.5])
    rollouter._maybe_annotate_zero_std = Mock()

    group = asyncio.run(
        rollouter.run_group_rollouts(
            generate_fn=AsyncMock(),
            sample=object(),
            group_id=7,
            group_size=2,
            sampling=object(),
            renderer=object(),
        )
    )

    assert [rollout.reward for rollout in group.rollouts] == [1.0, 0.0]
    assert [rollout.advantage for rollout in group.rollouts] == [0.5, -0.5]
    rollouter.score_group.assert_awaited_once()
    rollouter.advantage_estimator.assert_called_once()
    rollouter._maybe_annotate_zero_std.assert_called_once()
    metric_values = {
        metric.key: metric.value.value
        for metric in group.metrics
        if metric.key.startswith("rollout/infra")
    }
    assert metric_values == {
        "rollout/infra_failed_frac": 0.5,
        "rollout/infra_failed_group_frac": 1.0,
    }


def test_rollout_dump_writes_machine_readable_sandbox_issues(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("SWE_ROLLOUT_DUMP_DIR", str(tmp_path))
    sample = TMaxSample(
        instance_id="task-123",
        image="example/image",
        workdir="/workspace",
        problem_statement="test",
    )
    issue = SandboxIssue(
        provider="daytona",
        kind="session_disk_exhausted",
        phase="session_create",
        recovered=False,
        error_type="RuntimeError",
        message="no space left on device",
        sandbox_id="sandbox-abc",
        session_id="session-def",
    )
    diagnostics = _SandboxRolloutDiagnostics(
        sandbox_id="sandbox-abc",
        disk_gb=6,
        issue_counts={"session_disk_exhausted": 1},
        issues=(issue,),
        num_dropped_details=0,
    )
    rollouter = object.__new__(TMaxRollouter)

    rollouter._maybe_dump_trace(
        rollout_id="group=1/rollout=2",
        group_id=1,
        sample=sample,
        captured=[],
        renderer=object(),
        status="completed",
        reward=0.0,
        submitted=False,
        fmt_errors=0,
        error_msg="",
        finish_reason="hit_max_turns",
        sandbox_diagnostics=diagnostics,
    )

    trace = (tmp_path / "group=1_rollout=2.txt").read_text()
    assert "finish_reason  : hit_max_turns" in trace
    assert "sandbox_id     : sandbox-abc" in trace
    payload = json.loads((tmp_path / "group=1_rollout=2.sandbox.json").read_text())
    assert payload["instance_id"] == "task-123"
    assert payload["disk_gb"] == 6
    assert payload["issue_counts"] == {"session_disk_exhausted": 1}
    assert payload["issues"] == [
        {
            "attempt": None,
            "command_id": "",
            "error_type": "RuntimeError",
            "exit_code": None,
            "kind": "session_disk_exhausted",
            "max_attempts": None,
            "message": "no space left on device",
            "phase": "session_create",
            "provider": "daytona",
            "recovered": False,
            "sandbox_id": "sandbox-abc",
            "session_id": "session-def",
        }
    ]


def test_validation_groups_skip_the_training_rollout_dump(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Validation rollouts get the controller's per-pass report, not this dump."""
    monkeypatch.setenv("SWE_ROLLOUT_DUMP_DIR", str(tmp_path))
    rollouter = object.__new__(TMaxRollouter)

    rollouter._maybe_dump_trace(
        rollout_id="group=-1/rollout=0",
        group_id=-1,
        sample=TMaxSample(
            instance_id="task-123",
            image="example/image",
            workdir="/workspace",
            problem_statement="test",
        ),
        captured=[],
        renderer=object(),
        status="completed",
        reward=0.0,
        finish_reason="submit",
        sandbox_diagnostics=_SandboxRolloutDiagnostics(
            sandbox_id="sandbox-abc",
            disk_gb=6,
            issue_counts={},
            issues=(),
            num_dropped_details=0,
        ),
    )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "group_id, expect_annotation",
    [(7, True), (-1, False)],
)
def test_zero_std_annotation_skips_validation_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path, group_id: int, expect_annotation: bool
) -> None:
    """Held-out / benchmark prompts must not land in the training skip list."""
    monkeypatch.setenv("SWE_ZERO_STD_DIR", str(tmp_path))
    rollouter = object.__new__(TMaxRollouter)
    sample = TMaxSample(
        instance_id="task-123",
        image="example/image",
        workdir="/workspace",
        problem_statement="test",
    )
    rollouts = [
        Rollout(
            group_id=group_id,
            rollout_id=idx,
            status=RolloutStatus.COMPLETED,
            reward=0.0,
        )
        for idx in range(2)
    ]

    rollouter._maybe_annotate_zero_std(sample, rollouts)

    assert (tmp_path / "task-123.json").exists() is expect_annotation


def test_rollout_carries_its_finish_reason_and_format_errors() -> None:
    """The per-rollout loop outcome must ride on the Rollout: group metrics average it
    away, so the eval trace report has no other source for 'why did THIS trial stop'."""
    fields = {f.name for f in dataclasses.fields(Rollout)}
    assert "diagnostics" in fields

    rollout = Rollout(
        group_id=-1,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        diagnostics={
            "finish_reason": "stopped_early",
            "format_errors": 1,
            "submitted": False,
            "infra_failed": False,
        },
    )

    assert rollout.diagnostics["finish_reason"] == "stopped_early"
    assert rollout.diagnostics["format_errors"] == 1
    # The keys TMaxRollouter._run_agent_rollout populates.
    source = inspect.getsource(TMaxRollouter._run_agent_rollout)
    for key in ("finish_reason", "format_errors", "submitted", "infra_failed"):
        assert f'"{key}"' in source, f"{key} no longer recorded on the Rollout"


def test_worker_info_logging_reaches_a_handler_without_duplicating(
    capsys: pytest.CaptureFixture,
) -> None:
    """Worker INFO must be emitted exactly once, whether or not the process already
    configured logging. Records fall through to logging.lastResort (WARNING-only)
    otherwise, which silently hid every agent-loop stop reason."""
    import logging

    from torchtitan.experiments.rl.actors.rollout_worker import (
        _enable_worker_info_logging,
    )

    titan = logging.getLogger("torchtitan")
    root = logging.getLogger()
    saved = (list(titan.handlers), titan.level, list(root.handlers), root.level)
    try:
        for handlers, root_level in (([], logging.WARNING), ([], logging.INFO)):
            titan.handlers.clear()
            titan.setLevel(logging.NOTSET)
            root.handlers.clear()
            root.setLevel(root_level)
            if root_level == logging.INFO:
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                root.addHandler(handler)

            _enable_worker_info_logging()
            capsys.readouterr()
            logging.getLogger("torchtitan.experiments.rl.demo").info("marker-line")
            captured = capsys.readouterr()
            emitted = (captured.out + captured.err).count("marker-line")
            assert emitted == 1, f"root_level={root_level}: emitted {emitted} times"
    finally:
        titan.handlers[:] = saved[0]
        titan.setLevel(saved[1])
        root.handlers[:] = saved[2]
        root.setLevel(saved[3])

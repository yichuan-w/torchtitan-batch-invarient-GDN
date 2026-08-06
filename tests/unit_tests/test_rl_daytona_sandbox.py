# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from torchtitan.experiments.rl.examples.tmax.vanillux_loop import _run_bash
from torchtitan.experiments.rl.harness.agents import claude_code as agent_backend
from torchtitan.experiments.rl.harness.sandbox import (
    daytona as daytona_backend,
    SandboxIssue,
    SandboxIssueTracker,
    SandboxLogContext,
)
from torchtitan.experiments.rl.harness.sandbox.daytona import (
    _build_exec_command,
    _build_observable_exec,
    _build_observable_exec_command,
    DaytonaSandbox,
)


class _SessionExecuteRequest:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _StatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def fake_daytona(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("daytona")
    module.__dict__["SessionExecuteRequest"] = _SessionExecuteRequest
    monkeypatch.setitem(sys.modules, "daytona", module)


def _process() -> SimpleNamespace:
    return SimpleNamespace(
        create_session=AsyncMock(return_value=None),
        delete_session=AsyncMock(return_value=None),
        execute_session_command=AsyncMock(
            return_value=SimpleNamespace(cmd_id="command-id")
        ),
        get_session=AsyncMock(return_value=SimpleNamespace(commands=[])),
        get_session_command=AsyncMock(return_value=SimpleNamespace(exit_code=0)),
        get_session_command_logs=AsyncMock(
            return_value=SimpleNamespace(stdout="ok", stderr="")
        ),
    )


def _filesystem() -> SimpleNamespace:
    async def download_file(path: str, _timeout: int) -> bytes:
        if path.endswith(".status"):
            return b"0\n"
        if path.endswith(".output"):
            return b"ok"
        raise FileNotFoundError(path)

    return SimpleNamespace(
        download_file=AsyncMock(side_effect=download_file),
        get_file_info=AsyncMock(return_value=SimpleNamespace(size=2)),
        upload_file=AsyncMock(return_value=None),
    )


def _sandbox_with_process(process: SimpleNamespace) -> DaytonaSandbox:
    sandbox = DaytonaSandbox("example/image")
    sandbox._sb = SimpleNamespace(process=process, fs=_filesystem())
    return sandbox


def test_disk_override_is_validated() -> None:
    assert DaytonaSandbox("example/image", disk_gb=20).disk_gb == 20
    for invalid_disk_gb in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="disk_gb must be a positive integer"):
            DaytonaSandbox("example/image", disk_gb=cast(int, invalid_disk_gb))


def test_build_exec_command_preserves_complex_command() -> None:
    command = "printf '%s\\n' \"$HOME\"; printf 'line 1\\nline 2\\n'"

    full = _build_exec_command(command, user="root", env=None, timeout=17)

    completed = subprocess.run(
        ["bash", "-c", full],
        capture_output=True,
        check=False,
        env={**os.environ, "HOME": "/example-home"},
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout.decode() == "/example-home\nline 1\nline 2\n"
    assert base64.b64encode(command.encode()).decode() in full
    assert "timeout --signal=TERM --kill-after=10s 17s bash" in full
    assert command not in full


def test_build_exec_command_preserves_nonroot_env() -> None:
    env = {"GREETING": "hello world", "LINES": "first\nsecond"}

    full = _build_exec_command("env", user="agent", env=env, timeout=9)

    runner = shlex.join(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=10s",
            "9s",
            "env",
            "--",
            "GREETING=hello world",
            "LINES=first\nsecond",
            "runuser",
            "-u",
            "agent",
            "--whitelist-environment=GREETING,LINES",
            "--",
            "bash",
        ]
    )
    assert base64.b64encode(b"env").decode() in full
    assert f'{runner} "$_tt_script_path"' in full


def test_build_exec_command_has_no_reserved_transport_env() -> None:
    full = _build_exec_command(
        "printf '%s' \"$__TORCHTITAN_EXEC_SCRIPT_B64\"",
        user="root",
        env={"__TORCHTITAN_EXEC_SCRIPT_B64": "preserved"},
        timeout=9,
    )

    completed = subprocess.run(
        ["bash", "-c", full], capture_output=True, check=False, timeout=5
    )
    assert completed.returncode == 0
    assert completed.stdout == b"preserved"


def test_exec_separates_command_and_request_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_EXEC_TIMEOUT_MIN", "240")
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(return_value=(0, "ok"))

    result = asyncio.run(sandbox.exec("echo ok", timeout=5))

    assert result == (0, "ok", "")
    call = sandbox._session_exec.await_args
    assert call is not None
    assert "timeout --signal=TERM --kill-after=10s 5s bash" in call.args[0]
    assert call.kwargs == {"command_timeout": 5, "request_timeout": 240}


def test_long_command_does_not_extend_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TT_DAYTONA_EXEC_TIMEOUT_MIN", raising=False)
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(return_value=(0, "ok"))

    asyncio.run(sandbox.exec("echo ok", timeout=3600))

    call = sandbox._session_exec.await_args
    assert call is not None
    assert "timeout --signal=TERM --kill-after=10s 3600s bash" in call.args[0]
    assert call.kwargs == {"command_timeout": 3600, "request_timeout": 120}


def test_exec_reports_timeout_exit() -> None:
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(return_value=(124, "partial output"))

    result = asyncio.run(sandbox.exec("sleep 30", timeout=3))

    assert result == (124, "partial output", "Command timed out after 3s.")

    with pytest.raises(RuntimeError, match="Command timed out after 3s"):
        asyncio.run(sandbox.exec("sleep 30", timeout=3, check=True))

    sandbox._session_exec.return_value = (137, "Killed")
    assert asyncio.run(sandbox.exec("sleep 30", timeout=3)) == (137, "Killed", "")


@pytest.mark.skipif(shutil.which("timeout") is None, reason="GNU timeout is required")
def test_wrapped_command_is_hard_timed_out() -> None:
    full = _build_exec_command("sleep 30", user="root", env=None, timeout=1)
    started = time.monotonic()

    completed = subprocess.run(
        ["bash", "-c", full], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 124
    assert time.monotonic() - started < 5


@pytest.mark.skipif(
    shutil.which("timeout") is None or shutil.which("setsid") is None,
    reason="GNU timeout and setsid are required",
)
def test_wrapped_command_preserves_successful_detached_child() -> None:
    full = _build_exec_command(
        "setsid sleep 30 >/dev/null 2>&1 & echo $!",
        user="root",
        env=None,
        timeout=2,
    )
    completed = subprocess.run(
        ["bash", "-c", full], capture_output=True, check=False, text=True, timeout=5
    )
    assert completed.returncode == 0
    pid = int(completed.stdout.strip())
    try:
        os.kill(pid, 0)
    finally:
        os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(shutil.which("timeout") is None, reason="GNU timeout is required")
def test_observable_command_finishes_while_background_child_survives() -> None:
    command_key = f"pytest_{os.getpid()}_{time.time_ns()}"
    full = _build_exec_command(
        "sleep 30 & echo $!",
        user="root",
        env=None,
        timeout=2,
    )
    observed, status_path, output_path = _build_observable_exec_command(
        full, command_key
    )

    completed = subprocess.run(
        ["bash", "-c", observed], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 0
    with open(status_path) as status_file:
        assert status_file.read().strip() == "0"
    with open(output_path) as output_file:
        pid = int(output_file.read().strip())
    try:
        os.kill(pid, 0)
    finally:
        os.kill(pid, signal.SIGKILL)
        os.unlink(status_path)
        os.unlink(output_path)


@pytest.mark.skipif(shutil.which("timeout") is None, reason="GNU timeout is required")
def test_observable_command_restores_deleted_result_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    result_dir = tmp_path / "result"
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(result_dir))
    full = _build_exec_command(
        f"rm -rf {shlex.quote(str(output_dir))} {shlex.quote(str(result_dir))}; "
        "printf restored; exit 7",
        user="root",
        env=None,
        timeout=2,
    )
    observed, status_path, output_path = _build_observable_exec_command(
        full, "deleted_dirs"
    )

    completed = subprocess.run(
        ["bash", "-c", observed], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 7
    with open(status_path) as status_file:
        assert status_file.read().strip() == "7"
    with open(output_path) as output_file:
        assert output_file.read() == "restored"


@pytest.mark.skipif(
    shutil.which("timeout") is None or shutil.which("setsid") is None,
    reason="GNU timeout and setsid are required",
)
def test_observable_supervisor_survives_inner_bash_sigkill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(tmp_path / "result"))
    full = _build_exec_command(
        "kill -9 $$",
        user="root",
        env=None,
        timeout=2,
    )
    observed, status_path, _ = _build_observable_exec_command(
        full, "inner_bash_sigkill"
    )

    completed = subprocess.run(
        ["bash", "-c", observed], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 137
    assert Path(status_path).read_text().strip() == "137"


@pytest.mark.skipif(
    shutil.which("base64") is None
    or shutil.which("pkill") is None
    or shutil.which("timeout") is None,
    reason="base64, pkill, and GNU timeout are required",
)
def test_observable_command_avoids_pkill_pattern_self_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(tmp_path / "result"))
    pattern = f"tt_daytona_wrapper_{os.getpid()}_{time.time_ns()}"
    command = f"pkill -f {shlex.quote(pattern)} || :; printf survived"
    full = _build_exec_command(command, user="root", env=None, timeout=2)
    observed, status_path, output_path = _build_observable_exec_command(
        full, "pkill_self_match"
    )

    assert pattern not in observed
    completed = subprocess.run(
        ["bash", "-c", observed], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 0
    assert Path(status_path).read_text().strip() == "0"
    assert Path(output_path).read_text() == "survived"


@pytest.mark.skipif(
    shutil.which("base64") is None
    or shutil.which("pkill") is None
    or shutil.which("timeout") is None,
    reason="base64, pkill, and GNU timeout are required",
)
def test_observable_supervisor_hides_static_wrapper_from_pkill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(tmp_path / "result"))
    pattern = "torchtitan: command output truncated"
    full = _build_exec_command(
        f"pkill -9 -f {shlex.quote(pattern)} || :; printf survived",
        user="root",
        env=None,
        timeout=2,
    )
    observed, status_path, output_path = _build_observable_exec_command(
        full, "static_wrapper_pkill"
    )

    assert pattern not in observed
    completed = subprocess.run(
        ["bash", "-c", observed], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 0
    assert Path(status_path).read_text().strip() == "0"
    assert Path(output_path).read_text() == "survived"


def test_observable_supervisor_keeps_payload_out_of_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(tmp_path / "result"))
    full = _build_exec_command(
        "printf survived; # " + "x" * 30_000,
        user="root",
        env=None,
        timeout=2,
    )
    observed, _, _ = _build_observable_exec_command(full, "payload_not_in_argv")

    transport_env, exec_name, shell, flag, materializer = shlex.split(observed)
    env_name, encoded_wrapper = transport_env.split("=", 1)
    assert env_name == "__TORCHTITAN_OBSERVABLE_WRAPPER_B64"
    assert [exec_name, shell, flag] == ["exec", "sh", "-c"]
    assert encoded_wrapper not in materializer
    assert len(materializer) < 1_000


@pytest.mark.skipif(
    shutil.which("base64") is None or shutil.which("timeout") is None,
    reason="base64 and GNU timeout are required",
)
def test_observable_command_does_not_duplicate_large_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(tmp_path / "result"))
    command = "printf survived; # " + "x" * 30_000
    full = _build_exec_command(command, user="root", env=None, timeout=2)
    observed, status_path, output_path = _build_observable_exec_command(
        full, "large_payload"
    )

    assert command not in observed
    assert len(observed) < 100_000
    completed = subprocess.run(
        ["bash", "-c", observed], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 0
    assert Path(status_path).read_text().strip() == "0"
    assert Path(output_path).read_text() == "survived"


@pytest.mark.skipif(
    shutil.which("base64") is None or shutil.which("timeout") is None,
    reason="base64 and GNU timeout are required",
)
def test_uploaded_observable_command_handles_oversized_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(tmp_path / "result"))
    monkeypatch.setattr(daytona_backend, "_EXEC_STAGING_DIR", str(tmp_path))
    command = "printf survived; # " + "x" * 256_000
    full = _build_exec_command(command, user="root", env=None, timeout=2)
    command_key = f"uploaded_{os.getpid()}_{time.time_ns()}"
    observable = _build_observable_exec(full, command_key)

    assert len(base64.b64encode(command.encode())) > 131_072
    assert len(observable.inline_command.encode()) + 1 > 131_072
    assert len(observable.uploaded_command.encode()) < 1_000
    assert command not in observable.uploaded_command
    Path(observable.wrapper_path).write_bytes(observable.wrapper)
    completed = subprocess.run(
        ["bash", "-c", observable.uploaded_command],
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert Path(observable.status_path).read_text().strip() == "0"
    assert Path(observable.output_path).read_text() == "survived"
    assert not Path(observable.wrapper_path).exists()


def test_session_exec_uploads_oversized_observable(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daytona_backend, "_EXEC_INLINE_COMMAND_LIMIT_BYTES", 1)
    process = _process()
    sandbox = _sandbox_with_process(process)

    result = asyncio.run(
        sandbox._session_exec(
            "printf once",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    upload = sandbox._sb.fs.upload_file.await_args
    assert upload is not None
    wrapper, wrapper_path, upload_timeout = upload.args
    assert b"printf once" in wrapper
    assert wrapper_path.startswith("/dev/shm/.torchtitan_exec_")
    assert upload_timeout == daytona_backend._SESSION_RPC_TIMEOUT_SEC
    request = process.execute_session_command.await_args.args[1]
    assert request.command.endswith(shlex.quote(wrapper_path))
    assert "printf once" not in request.command
    assert len(request.command.encode()) < 1_000


def test_session_exec_keeps_normal_observable_inline(fake_daytona: None) -> None:
    process = _process()
    sandbox = _sandbox_with_process(process)

    result = asyncio.run(
        sandbox._session_exec(
            "printf once",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    sandbox._sb.fs.upload_file.assert_not_awaited()
    request = process.execute_session_command.await_args.args[1]
    assert request.command.startswith("__TORCHTITAN_OBSERVABLE_WRAPPER_B64=")
    assert len(request.command.encode()) <= (
        daytona_backend._EXEC_INLINE_COMMAND_LIMIT_BYTES
    )


@pytest.mark.skipif(
    shutil.which("base64") is None or shutil.which("timeout") is None,
    reason="base64 and GNU timeout are required",
)
def test_wrapped_command_preserves_stdin() -> None:
    full = _build_exec_command(
        "IFS= read -r value; printf '%s' \"$value\"",
        user="root",
        env=None,
        timeout=2,
    )

    completed = subprocess.run(
        ["bash", "-c", full],
        input=b"caller input\n",
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b"caller input"


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("timeout") is None,
    reason="Bash and GNU timeout are required",
)
def test_wrapped_command_reports_missing_decoder(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    os.symlink(shutil.which("bash"), bin_dir / "bash")
    os.symlink(shutil.which("timeout"), bin_dir / "timeout")
    marker = tmp_path / "executed"
    full = _build_exec_command(
        f"printf executed > {shlex.quote(str(marker))}",
        user="root",
        env=None,
        timeout=2,
    )

    completed = subprocess.run(
        [str(bin_dir / "bash"), "-c", full],
        capture_output=True,
        check=False,
        env={"PATH": str(bin_dir)},
        timeout=5,
    )

    assert completed.returncode == 127
    assert not marker.exists()


@pytest.mark.skipif(shutil.which("timeout") is None, reason="GNU timeout is required")
def test_observable_command_materializes_bounded_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    result_dir = tmp_path / "result"
    pid_path = tmp_path / "child.pid"
    monkeypatch.setattr(daytona_backend, "_EXEC_OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(result_dir))
    monkeypatch.setattr(daytona_backend, "_EXEC_RAW_OUTPUT_LIMIT_BYTES", 64_000)
    full = _build_exec_command(
        "head -c 200000 /dev/zero | tr '\\0' x; "
        f"sleep 30 & printf '%s' $! > {shlex.quote(str(pid_path))}",
        user="root",
        env=None,
        timeout=2,
    )
    observed, status_path, output_path = _build_observable_exec_command(
        full, "bounded_output"
    )

    completed = subprocess.run(
        ["bash", "-c", observed], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 0
    with open(status_path) as status_file:
        assert status_file.read().strip() == "0"
    output = open(output_path, "rb").read()
    assert b"[torchtitan: command output truncated]" in output
    assert len(output) <= (
        daytona_backend._EXEC_OUTPUT_HEAD_BYTES
        + daytona_backend._EXEC_OUTPUT_TAIL_BYTES
        + 100
    )
    assert not list(output_dir.iterdir())
    pid = int(pid_path.read_text())
    try:
        os.kill(pid, 0)
    finally:
        os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(shutil.which("timeout") is None, reason="GNU timeout is required")
def test_observable_command_runs_when_capture_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "result"
    monkeypatch.setattr(
        daytona_backend,
        "_EXEC_OUTPUT_DIR",
        "/proc/torchtitan-unwritable",
    )
    monkeypatch.setattr(daytona_backend, "_EXEC_RESULT_DIR", str(result_dir))
    full = _build_exec_command(
        "printf fallback; exit 3",
        user="root",
        env=None,
        timeout=2,
    )
    observed, status_path, output_path = _build_observable_exec_command(
        full, "capture_failure"
    )

    completed = subprocess.run(
        ["bash", "-c", observed],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 3
    assert completed.stdout == "fallback"
    assert Path(status_path).read_text().strip() == "3"
    assert not Path(output_path).exists()


def test_lost_execute_response_recovers_without_replay(
    fake_daytona: None,
) -> None:
    full = "echo once"
    process = _process()
    process.execute_session_command.side_effect = ConnectionError("response lost")

    async def get_session(_session_id: str) -> SimpleNamespace:
        request = process.execute_session_command.await_args.args[1]
        return SimpleNamespace(
            commands=[SimpleNamespace(id="recovered-id", command=request.command)]
        )

    process.get_session.side_effect = get_session
    sandbox = _sandbox_with_process(process)

    result = asyncio.run(
        sandbox._session_exec(
            full,
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    process.execute_session_command.assert_awaited_once()
    execute_call = process.execute_session_command.await_args
    assert execute_call.kwargs["timeout"] == 30
    request = execute_call.args[1]
    assert request.run_async is True
    assert request.var_async is None
    assert request.additional_properties == {"async": True}
    process.get_session_command.assert_not_awaited()
    process.get_session_command_logs.assert_not_awaited()
    process.delete_session.assert_not_awaited()
    assert sandbox.issue_tracker.counts == {"execute_response_recovered": 1}
    issue = sandbox.issue_tracker.issues[0]
    assert issue.recovered
    assert issue.session_id
    assert issue.command_id == "recovered-id"


def test_result_sentinel_preserves_nonzero_exit_and_output(
    fake_daytona: None,
) -> None:
    process = _process()
    sandbox = _sandbox_with_process(process)

    async def download_file(path: str, _timeout: int) -> bytes:
        return b"7\n" if path.endswith(".status") else b"command failed\n"

    sandbox._sb.fs.download_file.side_effect = download_file

    result = asyncio.run(
        sandbox._session_exec(
            "exit 7",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (7, "command failed\n")
    process.execute_session_command.assert_awaited_once()
    process.get_session_command.assert_not_awaited()
    process.get_session_command_logs.assert_not_awaited()


def test_provider_status_recovers_when_wrapper_is_killed(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daytona_backend, "_PROVIDER_STATUS_FIRST_POLL_SEC", 0)
    process = _process()
    process.get_session_command.return_value = SimpleNamespace(exit_code=137)
    sandbox = _sandbox_with_process(process)
    sandbox._sb.fs.download_file.side_effect = FileNotFoundError("not found")

    result = asyncio.run(
        sandbox._session_exec(
            "pkill -9 bash",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (137, "ok")
    process.execute_session_command.assert_awaited_once()
    process.get_session_command.assert_awaited_once()
    process.get_session_command_logs.assert_awaited_once()
    process.delete_session.assert_not_awaited()
    assert sandbox.issue_tracker.counts == {
        "command_status_fallback": 1,
        "command_output_missing": 1,
    }


def test_missing_captured_output_uses_provider_logs(fake_daytona: None) -> None:
    process = _process()
    process.get_session_command_logs.return_value = SimpleNamespace(
        output="",
        stdout="ok",
        stderr="warning",
    )
    sandbox = _sandbox_with_process(process)

    async def download_file(path: str, _timeout: int) -> bytes:
        if path.endswith(".status"):
            return b"0\n"
        raise FileNotFoundError(path)

    sandbox._sb.fs.download_file.side_effect = download_file

    result = asyncio.run(
        sandbox._session_exec(
            "echo ok",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok\nwarning")
    process.execute_session_command.assert_awaited_once()
    process.get_session_command_logs.assert_awaited_once()
    assert sandbox.issue_tracker.counts == {"command_output_missing": 1}


def test_missing_captured_output_and_logs_returns_diagnostic(
    fake_daytona: None,
) -> None:
    process = _process()
    process.get_session_command_logs.side_effect = FileNotFoundError("not found")
    sandbox = _sandbox_with_process(process)

    async def download_file(path: str, _timeout: int) -> bytes:
        if path.endswith(".status"):
            return b"0\n"
        raise FileNotFoundError(path)

    sandbox._sb.fs.download_file.side_effect = download_file

    result = asyncio.run(
        sandbox._session_exec(
            "echo ok",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, daytona_backend._MISSING_OUTPUT_MESSAGE)
    process.execute_session_command.assert_awaited_once()
    process.get_session_command_logs.assert_awaited_once()
    assert sandbox.issue_tracker.counts == {
        "command_output_missing": 1,
        "command_logs_missing": 1,
    }


def test_session_create_enospc_is_not_retried(
    caplog: pytest.LogCaptureFixture,
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_SESSION_CREATE_RETRIES", "5")
    process = _process()
    process.create_session.side_effect = RuntimeError("no space left on device")
    tracker = SandboxIssueTracker(
        SandboxLogContext(
            instance_id="task-123",
            group_id=7,
            rollout_id=11,
        )
    )
    sandbox = DaytonaSandbox("example/image", disk_gb=20, issue_tracker=tracker)
    sandbox._sb = SimpleNamespace(process=process)
    sandbox.sandbox_id = "sandbox-id"

    with pytest.raises(RuntimeError, match="no space left on device"):
        asyncio.run(
            sandbox._session_exec(
                "echo once",
                command_timeout=5,
                request_timeout=30,
            )
        )

    process.create_session.assert_awaited_once()
    process.delete_session.assert_awaited_once()
    assert tracker.counts == {"session_disk_exhausted": 1}
    issue = tracker.issues[0]
    assert issue.phase == "session_create"
    assert issue.sandbox_id == "sandbox-id"
    assert issue.session_id
    assert issue.attempt == 1
    assert issue.max_attempts == 6
    payload = json.loads(caplog.records[-1].getMessage().split("] ", 1)[1])
    assert payload == {
        "attempt": 1,
        "command_id": "",
        "disk_gb": 20,
        "error_type": "RuntimeError",
        "event": "sandbox_issue",
        "exit_code": None,
        "group_id": 7,
        "image": "example/image",
        "instance_id": "task-123",
        "kind": "session_disk_exhausted",
        "max_attempts": 6,
        "message": "no space left on device",
        "phase": "session_create",
        "provider": "daytona",
        "recovered": False,
        "rollout_id": 11,
        "sandbox_id": "sandbox-id",
        "session_id": issue.session_id,
    }


def test_session_create_stops_when_sandbox_is_gone(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_SESSION_CREATE_RETRIES", "5")
    process = _process()
    process.create_session.side_effect = RuntimeError("sandbox not found")
    sandbox = _sandbox_with_process(process)
    sandbox.sandbox_id = "sandbox-id"

    with pytest.raises(RuntimeError, match="sandbox not found"):
        asyncio.run(
            sandbox._session_exec(
                "echo once",
                command_timeout=5,
                request_timeout=30,
            )
        )

    process.create_session.assert_awaited_once()
    process.delete_session.assert_not_awaited()
    assert sandbox.issue_tracker.counts == {"sandbox_lost": 1}
    with pytest.raises(RuntimeError, match="is no longer available"):
        asyncio.run(sandbox.exec("echo again"))


def test_nonzero_command_enospc_is_recorded() -> None:
    sandbox = DaytonaSandbox("example/image", disk_gb=20)
    sandbox._session_exec = AsyncMock(
        return_value=(1, "OSError: [Errno 28] No space left on device")
    )

    result = asyncio.run(sandbox.exec("pip install package", timeout=5))

    assert result == (1, "OSError: [Errno 28] No space left on device", "")
    assert sandbox.issue_tracker.counts == {"command_disk_exhausted": 1}
    issue = sandbox.issue_tracker.issues[0]
    assert issue.phase == "command"
    assert issue.exit_code == 1


def test_successful_command_output_does_not_misclassify_enospc_text() -> None:
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(
        return_value=(0, "binary contains no space left on device")
    )

    assert asyncio.run(sandbox.exec("strings binary")) == (
        0,
        "binary contains no space left on device",
        "",
    )
    assert sandbox.issue_tracker.counts == {}


def test_agent_loop_propagates_exec_transport_error() -> None:
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(
        side_effect=RuntimeError("no space left on device")
    )

    with pytest.raises(RuntimeError, match="no space left on device"):
        asyncio.run(_run_bash(sandbox, "echo ok", timeout=5))

    assert sandbox.issue_tracker.counts == {"exec_failed": 1}


def test_agent_loop_keeps_nonzero_command_as_observation() -> None:
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(return_value=(1, "test failed"))

    output, exit_code = asyncio.run(_run_bash(sandbox, "false", timeout=5))

    assert (output, exit_code) == ("test failed", 1)
    assert sandbox.issue_tracker.counts == {}


def test_command_output_retry_does_not_replay_command(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_RPC_RETRIES", "2")
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    process = _process()
    sandbox = _sandbox_with_process(process)
    output_calls = 0

    async def download_file(path: str, _timeout: int) -> bytes:
        nonlocal output_calls
        if path.endswith(".status"):
            return b"0\n"
        output_calls += 1
        if output_calls == 1:
            raise ConnectionError("server disconnected")
        return b"ok"

    sandbox._sb.fs.download_file.side_effect = download_file

    result = asyncio.run(
        sandbox._session_exec(
            "echo once",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    process.execute_session_command.assert_awaited_once()
    assert output_calls == 2
    assert sandbox.issue_tracker.counts == {"command_output_retry": 1}


def test_file_upload_retry_reuses_identical_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_RPC_RETRIES", "2")
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    upload_file = AsyncMock(side_effect=[ConnectionError("server disconnected"), None])
    sandbox = DaytonaSandbox("example/image")
    sandbox._sb = SimpleNamespace(fs=SimpleNamespace(upload_file=upload_file))
    sandbox.exec = AsyncMock(return_value=(0, "", ""))

    asyncio.run(sandbox.write_file("/tmp/payload", b"same bytes"))

    assert upload_file.await_count == 2
    assert [call.args for call in upload_file.await_args_list] == [
        (b"same bytes", "/tmp/payload"),
        (b"same bytes", "/tmp/payload"),
    ]
    assert sandbox.issue_tracker.counts == {"file_upload_retry": 1}


def test_delete_retries_transient_failure_and_accepts_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_RPC_RETRIES", "2")
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    sandbox = DaytonaSandbox("example/image")
    sandbox._sb = SimpleNamespace(id="sandbox-id")
    sandbox._client = SimpleNamespace(
        delete=AsyncMock(
            side_effect=[
                ConnectionError("server disconnected"),
                RuntimeError("sandbox not found"),
            ]
        )
    )

    asyncio.run(sandbox.__aexit__(None, None, None))

    assert sandbox._client.delete.await_count == 2
    assert sandbox.issue_tracker.counts == {"delete_retry": 1}


def test_heartbeat_refreshes_until_teardown() -> None:
    async def run() -> tuple[int, list[str]]:
        order: list[str] = []

        async def refresh_activity() -> None:
            order.append("refresh")

        async def delete(_sandbox) -> None:
            order.append("delete")

        sandbox = DaytonaSandbox("example/image")
        sandbox._sb = SimpleNamespace(
            id="sandbox-id",
            refresh_activity=AsyncMock(side_effect=refresh_activity),
        )
        sandbox._client = SimpleNamespace(delete=AsyncMock(side_effect=delete))
        sandbox._heartbeat_task = asyncio.create_task(sandbox._heartbeat_loop(0.001))
        while sandbox._sb.refresh_activity.await_count == 0:
            await asyncio.sleep(0.001)
        await sandbox.__aexit__(None, None, None)
        return sandbox._sb.refresh_activity.await_count, order

    refresh_count, order = asyncio.run(run())

    assert refresh_count >= 1
    assert order[-1] == "delete"


def test_heartbeat_detected_loss_propagates_after_delete() -> None:
    async def run() -> int:
        sandbox = DaytonaSandbox("example/image")
        sandbox.sandbox_id = "sandbox-id"
        sandbox._sb = SimpleNamespace(
            id="sandbox-id",
            refresh_activity=AsyncMock(side_effect=RuntimeError("sandbox not found")),
        )
        sandbox._client = SimpleNamespace(delete=AsyncMock(return_value=None))
        sandbox._heartbeat_task = asyncio.create_task(sandbox._heartbeat_loop(0.001))
        await sandbox._heartbeat_task
        with pytest.raises(RuntimeError, match="is no longer available"):
            await sandbox.__aexit__(None, None, None)
        return sandbox._client.delete.await_count

    assert asyncio.run(run()) == 1


def test_boot_retries_share_rollout_issue_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = SandboxIssueTracker(
        SandboxLogContext(instance_id="task-123", group_id=7, rollout_id=11)
    )
    received_trackers: list[SandboxIssueTracker] = []
    num_candidates = 0

    class _Candidate:
        allocated_disk_gb = 20

        def __init__(
            self, candidate_tracker: SandboxIssueTracker, *, fail: bool
        ) -> None:
            self.issue_tracker = candidate_tracker
            self.fail = fail
            self.sandbox_id = "failed-sandbox" if fail else "active-sandbox"

        async def __aenter__(self):
            if self.fail:
                self.issue_tracker.record(
                    SandboxIssue(
                        provider="daytona",
                        kind="create_retry",
                        phase="create",
                        recovered=True,
                        error_type="RuntimeError",
                        message="transient create failure",
                        attempt=1,
                        max_attempts=2,
                    )
                )
                raise RuntimeError("transient create failure")
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    def make_sandbox(image: str, **kwargs):
        nonlocal num_candidates
        candidate_tracker = kwargs["issue_tracker"]
        received_trackers.append(candidate_tracker)
        candidate = _Candidate(candidate_tracker, fail=num_candidates == 0)
        num_candidates += 1
        return candidate

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(agent_backend, "SWE_BOOT_RETRIES", 2)
    monkeypatch.setattr(agent_backend, "_BOOT_SEM", None)
    monkeypatch.setattr(agent_backend, "make_sandbox", make_sandbox)
    monkeypatch.setattr(agent_backend.asyncio, "sleep", no_sleep)

    async def run() -> None:
        async with agent_backend.boot_agent_sandbox(
            "example/image",
            install_claude=False,
            disk_gb=20,
            issue_tracker=tracker,
        ) as sandbox:
            assert sandbox.sandbox_id == "active-sandbox"

    asyncio.run(run())

    assert received_trackers == [tracker, tracker]
    assert tracker.counts == {"create_retry": 1}


def test_result_poll_disconnect_does_not_replay_command(fake_daytona: None) -> None:
    process = _process()
    sandbox = _sandbox_with_process(process)
    status_calls = 0

    async def download_file(path: str, _timeout: int) -> bytes:
        nonlocal status_calls
        if path.endswith(".output"):
            return b"ok"
        status_calls += 1
        if status_calls == 1:
            raise ConnectionError("server disconnected")
        return b"0\n"

    sandbox._sb.fs.download_file.side_effect = download_file

    result = asyncio.run(
        sandbox._session_exec(
            "echo once",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    process.execute_session_command.assert_awaited_once()
    assert status_calls == 2
    process.get_session_command.assert_not_awaited()
    process.delete_session.assert_not_awaited()
    assert sandbox.issue_tracker.counts == {"poll_transient": 1}
    assert sandbox.issue_tracker.issues[0].recovered


def test_result_poll_treats_file_404_as_pending(fake_daytona: None) -> None:
    process = _process()
    sandbox = _sandbox_with_process(process)
    status_calls = 0

    async def download_file(path: str, _timeout: int) -> bytes:
        nonlocal status_calls
        if path.endswith(".output"):
            return b"ok"
        status_calls += 1
        if status_calls == 1:
            raise _StatusError("file not found", 404)
        return b"0\n"

    sandbox._sb.fs.download_file.side_effect = download_file

    result = asyncio.run(
        sandbox._session_exec(
            "echo ok",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    assert status_calls == 2
    assert sandbox.issue_tracker.counts == {}


def test_result_poll_preserves_explicit_sandbox_404(fake_daytona: None) -> None:
    process = _process()
    sandbox = _sandbox_with_process(process)
    sandbox._sb.fs.download_file.side_effect = _StatusError("sandbox not found", 404)

    with pytest.raises(_StatusError, match="sandbox not found"):
        asyncio.run(
            sandbox._session_exec(
                "echo ok",
                command_timeout=5,
                request_timeout=30,
            )
        )

    assert sandbox.issue_tracker.counts == {"sandbox_lost": 1}


def test_poll_deadline_does_not_delete_a_possibly_successful_session(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daytona_backend, "_COMMAND_KILL_GRACE_SEC", 0)
    monkeypatch.setattr(daytona_backend, "_SESSION_POLL_GRACE_SEC", 0)
    process = _process()
    process.get_session_command.return_value = SimpleNamespace(exit_code=None)
    sandbox = _sandbox_with_process(process)
    sandbox._sb.fs.download_file.side_effect = FileNotFoundError("not found")

    with pytest.raises(TimeoutError, match="without replaying"):
        asyncio.run(
            sandbox._session_exec(
                "setsid sleep 30 &",
                command_timeout=0,
                request_timeout=30,
            )
        )

    process.execute_session_command.assert_awaited_once()
    process.get_session_command.assert_awaited_once()
    process.delete_session.assert_not_awaited()


def test_unconfirmed_execute_is_cleaned_up_without_replay(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daytona_backend, "_COMMAND_RECOVERY_DELAYS_SEC", (0.0,))
    process = _process()
    process.execute_session_command.side_effect = ConnectionError("not accepted")
    sandbox = _sandbox_with_process(process)

    with pytest.raises(ConnectionError, match="not accepted"):
        asyncio.run(
            sandbox._session_exec(
                "echo once",
                command_timeout=5,
                request_timeout=30,
            )
        )

    process.execute_session_command.assert_awaited_once()
    process.delete_session.assert_awaited_once()

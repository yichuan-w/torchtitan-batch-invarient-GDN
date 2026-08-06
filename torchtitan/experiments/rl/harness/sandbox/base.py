# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Provider-agnostic sandbox contract + backend factory.

The contract is intentionally small: async context management, command execution,
and file read/write. A coding-agent example builds its task-specific setup, agent
runner, and grader on top of this without depending on one sandbox provider.
The backend lives in a sibling module (``daytona.py``).
"""

from __future__ import annotations

import os

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


ExecResult = tuple[int, str, str]  # (exit_code, stdout, stderr)
FileContent = str | bytes | Path


@dataclass(frozen=True, slots=True)
class SandboxLogContext:
    """Stable rollout identity attached to sandbox diagnostics."""

    instance_id: str = ""
    group_id: int | None = None
    rollout_id: int | None = None


@dataclass(frozen=True, slots=True)
class SandboxIssue:
    """One provider-side problem observed while operating a sandbox."""

    provider: str
    kind: str
    phase: str
    recovered: bool
    error_type: str
    message: str
    sandbox_id: str = ""
    session_id: str = ""
    command_id: str = ""
    attempt: int | None = None
    max_attempts: int | None = None
    exit_code: int | None = None


class SandboxIssueTracker:
    """Per-rollout issue counts plus bounded event details.

    The tracker is owned by the rollout and shared across sandbox boot retries, so
    diagnostics from failed candidates are not lost before a sandbox is yielded.
    """

    def __init__(
        self,
        context: SandboxLogContext | None = None,
        *,
        max_details: int = 256,
    ) -> None:
        if max_details <= 0:
            raise ValueError(f"max_details must be positive, got {max_details}")
        self.context = context or SandboxLogContext()
        self._max_details = max_details
        self._issues: list[SandboxIssue] = []
        self._counts: Counter[str] = Counter()
        self._num_dropped_details = 0

    def record(self, issue: SandboxIssue) -> None:
        self._counts[issue.kind] += 1
        if len(self._issues) < self._max_details:
            self._issues.append(issue)
        else:
            self._num_dropped_details += 1

    @property
    def issues(self) -> tuple[SandboxIssue, ...]:
        return tuple(self._issues)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def num_events(self) -> int:
        return sum(self._counts.values())

    @property
    def num_dropped_details(self) -> int:
        return self._num_dropped_details


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.
    """

    sandbox_id: str
    allocated_disk_gb: int | None
    issue_tracker: SandboxIssueTracker

    async def __aenter__(self) -> Sandbox:
        ...

    async def __aexit__(self, exc_type, exc, tb) -> None:
        ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        ...

    async def write_file(
        self, sandbox_path: str, content: FileContent, *, user: str = "root"
    ) -> None:
        ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        ...


def _getenv(*names: str, default: str = "") -> str:
    """Return the first non-empty value among ``names`` (alias support)."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


def make_sandbox(image: str, **kwargs) -> Sandbox:
    """Factory: build the sandbox backend selected by ``TT_SANDBOX_BACKEND``.

    Only ``daytona`` -> ``DaytonaSandbox`` is bundled; the factory is the seam for
    adding another provider as a new ``sandbox`` backend. The backend is imported
    lazily so a missing optional SDK (``daytona``) only errors when it is selected.
    """
    backend = _getenv("TT_SANDBOX_BACKEND", default="daytona").lower()
    if backend != "daytona":
        raise ValueError(
            f"unknown sandbox backend {backend!r}; only 'daytona' is bundled"
        )
    from torchtitan.experiments.rl.harness.sandbox.daytona import DaytonaSandbox

    return DaytonaSandbox(image, **kwargs)

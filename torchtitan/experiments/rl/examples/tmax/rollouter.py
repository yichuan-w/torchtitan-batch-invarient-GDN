# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-agent rollouter for AI2 tmax tasks.

Forked from ``swe_r2e/rollouter.py`` (same shape: open an adapter session, boot a
fresh sandbox, run a host-side ReAct loop against the on-box adapter, then grade
and stamp the reward) but drives the FAITHFUL Vanillux agent loop
(``run_vanillux_loop``) rather than the swe_r2e host_loop. The tmax Qwen3.5-9B is
SFT'd under ``SWERLVanilluxSandboxEnv`` (single ``bash`` tool, vanillux prompts,
persistent shell, ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` to submit); the
swe_r2e host_loop's Bash/Read/Write/Edit tool set + SWE system prompt would put the
policy off-distribution, so we reproduce the tmax scaffold exactly while keeping the
host_loop transport (agent brain on the controller, each bash action dispatched to
the Daytona sandbox via ``sb.exec``). tmax specifics:

  1. Run the agent as ROOT. tmax tasks write to system paths (``/logs``,
     ``/output``, ``/home/user``, ``/app``). ``_RootSandbox`` forces every sandbox
     call to ``user="root"``.
  2. Honor the per-task ``workdir`` (best-guessed at data-prep time; /home/user,
     /app, or /workspace).
  3. Grade in place on the SUBMIT MARKER only: the tmax verifier inspects the
     container's live filesystem, so ``grade_tmax`` uploads the verifier into the
     agent's own sandbox and runs ``bash /tests/test.sh``. A rollout that never
     submits scores 0 (matches the env: tests run only on submit).

The standard scoring + advantage path is unchanged: the grade is stamped on the
last turn's ``env_rewards`` (key ``tmax_reward``) and read back by ``RewardTMax``.

Knobs read from env (the launcher sets these; see ``submit_swe_tmax_9b.sh``):
  ``SHIM_BIND_HOST`` / ``SHIM_PORT``  adapter bind address (default 127.0.0.1:18001)
  ``SWE_TIME_BUDGET_SEC``             per-agent wallclock (default 2400)
  ``TMAX_EVAL_TIMEOUT_SEC``           verifier run timeout (default 900)
  ``SWE_MAX_CONTEXT_LEN``             model context budget for the adapter session
  ``SWE_ROLLOUT_CONCURRENCY``         concurrently-active rollouts (default 16)
  ``TMAX_CALL_LIMIT`` / ``TMAX_TURN_MAX_TOKENS``  Vanillux step + per-turn caps
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import os
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from renderers import Renderer

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset, TMaxSample
from torchtitan.experiments.rl.examples.tmax.env import TMaxEnv
from torchtitan.experiments.rl.examples.tmax.grading import grade_tmax, seed_workspace
from torchtitan.experiments.rl.examples.tmax.rubric import RewardTMax, TMAX_REWARD_KEY
from torchtitan.experiments.rl.examples.tmax.vanillux_loop import run_vanillux_loop
from torchtitan.experiments.rl.harness import (
    AnthropicAdapter,
    boot_agent_sandbox,
    Sandbox,
    SandboxIssue,
    SandboxIssueTracker,
    SandboxLogContext,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import (
    GenerateFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.experiments.rl.types import RolloutTurnID

if TYPE_CHECKING:
    # Type-only: importing the generator module pulls in vLLM at import time.
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

logger = logging.getLogger(__name__)


_FINISH_REASONS = (
    "submit",
    "hit_max_turns",
    "hit_time_budget",
    "stopped_early",
    "error",
)

_DISK_ISSUE_KINDS = {
    "command_disk_exhausted",
    "session_disk_exhausted",
}
_TRANSPORT_ISSUE_KINDS = {
    "command_logs_failed",
    "command_logs_missing",
    "command_logs_retry",
    "command_output_failed",
    "command_output_missing",
    "command_output_retry",
    "command_recovery_query_failed",
    "command_status_fallback",
    "command_status_invalid",
    "command_status_query_failed",
    "command_status_timeout",
    "delete_failed",
    "delete_retry",
    "exec_failed",
    "execute_missing_command_id",
    "execute_response_recovered",
    "execute_response_unconfirmed",
    "file_upload_failed",
    "file_upload_retry",
    "heartbeat_retry",
    "poll_transient",
    "sandbox_lost",
    "session_cleanup_failed",
}
_PROVISION_ISSUE_KINDS = {
    "create_failed",
    "create_retry",
    "provision_failed",
    "provision_retry",
    "session_create_failed",
    "session_create_retry",
}
_TIMEOUT_ISSUE_KINDS = {
    "command_status_timeout",
    "command_timeout",
}


@dataclass(slots=True)
class _SandboxRolloutDiagnostics:
    sandbox_id: str
    disk_gb: int | None
    issue_counts: dict[str, int]
    issues: tuple[SandboxIssue, ...]
    num_dropped_details: int
    infra_failed: bool = False


def _sandbox_issue_metrics(
    issue_counts_by_rollout: list[Mapping[str, int]],
) -> list[m.Metric]:
    """Summarize bounded sandbox issue categories for one sibling group."""
    assert issue_counts_by_rollout, "a TMax rollout group must contain siblings"
    num_rollouts = len(issue_counts_by_rollout)

    def _num_events(counts: Mapping[str, int], kinds: set[str] | None = None) -> int:
        if kinds is None:
            return sum(counts.values())
        return sum(counts.get(kind, 0) for kind in kinds)

    def _frac(kinds: set[str] | None = None) -> float:
        return (
            sum(
                1.0
                for counts in issue_counts_by_rollout
                if _num_events(counts, kinds) > 0
            )
            / num_rollouts
        )

    def _events_mean(kinds: set[str] | None = None) -> float:
        return (
            sum(_num_events(counts, kinds) for counts in issue_counts_by_rollout)
            / num_rollouts
        )

    return [
        m.Metric("rollout/sandbox_issue_frac", m.Mean(_frac())),
        m.Metric("rollout/sandbox_issue_events_mean", m.Mean(_events_mean())),
        m.Metric("rollout/sandbox_disk_full_frac", m.Mean(_frac(_DISK_ISSUE_KINDS))),
        m.Metric(
            "rollout/sandbox_disk_full_events_mean",
            m.Mean(_events_mean(_DISK_ISSUE_KINDS)),
        ),
        m.Metric(
            "rollout/sandbox_transport_issue_frac",
            m.Mean(_frac(_TRANSPORT_ISSUE_KINDS)),
        ),
        m.Metric(
            "rollout/sandbox_transport_issue_events_mean",
            m.Mean(_events_mean(_TRANSPORT_ISSUE_KINDS)),
        ),
        m.Metric(
            "rollout/sandbox_provision_issue_frac",
            m.Mean(_frac(_PROVISION_ISSUE_KINDS)),
        ),
        m.Metric("rollout/sandbox_timeout_frac", m.Mean(_frac(_TIMEOUT_ISSUE_KINDS))),
    ]


def _finish_reason_metrics(finish_reasons: list[str]) -> list[m.Metric]:
    """Return exhaustive per-reason fractions for one completed rollout group."""
    assert finish_reasons, "a TMax rollout group must contain at least one sibling"
    unexpected = set(finish_reasons).difference(_FINISH_REASONS)
    assert not unexpected, f"unexpected TMax finish reasons: {sorted(unexpected)}"

    num_rollouts = len(finish_reasons)
    return [
        m.Metric(
            f"rollout/finish_{reason}_frac",
            m.Mean(
                sum(1.0 for finish_reason in finish_reasons if finish_reason == reason)
                / num_rollouts
            ),
        )
        for reason in _FINISH_REASONS
    ]


class _RootSandbox:
    """Sandbox wrapper that forces every operation to run as ``root``.

    tmax tasks need root to touch system paths (``/logs``, ``/output``, ``/app``).
    This delegates to the underlying sandbox with the requested ``user`` overridden
    to ``root``, so ``run_vanillux_loop`` (and ``grade_tmax``) run entirely as root.
    """

    def __init__(self, inner: Sandbox) -> None:
        self._inner = inner

    @property
    def sandbox_id(self) -> str:
        return self._inner.sandbox_id

    @property
    def allocated_disk_gb(self) -> int | None:
        return self._inner.allocated_disk_gb

    @property
    def issue_tracker(self) -> SandboxIssueTracker:
        return self._inner.issue_tracker

    async def exec(self, cmd: str, *, user: str = "root", **kwargs):
        return await self._inner.exec(cmd, user="root", **kwargs)

    async def write_file(self, sandbox_path: str, content, *, user: str = "root"):
        return await self._inner.write_file(sandbox_path, content, user="root")

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        return await self._inner.read_file(sandbox_path, user="root")


class _RolloutIssueGate:
    """Work-conserving async gate for sibling rollouts in one worker process.

    Every sibling reserves one slot. Waiters are admitted by ascending
    ``(group_id, rollout_idx)`` priority, so lower-id prompt groups get capacity first,
    but any completed sibling immediately hands its slot to the next waiter. A slow
    sibling therefore occupies only its own slot and cannot leave the worker's other
    slots idle while the gate waits to reassemble a full group.

    This matches open-instruct's environment pool at the relevant boundary: an
    environment is acquired and released per rollout, while prompt-group aggregation
    separately waits for all siblings. The gate is per worker process, not global
    across the RolloutWorker pool.

    Single-event-loop only (no threads): asyncio runs coroutine steps serially, so the
    counter/heap need no lock.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._available = capacity
        # Min-heap of (priority, tiebreak, future). tiebreak keeps the heap
        # totally ordered on equal priority (and never compares futures).
        self._waiters: list[tuple[tuple[int, int], int, asyncio.Future]] = []
        self._tiebreak = itertools.count()

    async def acquire_sibling(self, priority: tuple[int, int]) -> None:
        """Reserve one rollout slot, waiting behind lower-priority siblings."""
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._tiebreak), fut))
        self._try_admit()  # may grant synchronously if this is now the head and fits
        try:
            await fut
        except asyncio.CancelledError:
            # Granted-then-cancelled: return the slot we were handed.
            # Pending-then-cancelled: the dead future is dropped by _try_admit's
            # done() check, so there is nothing to return.
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                self._try_admit()
            raise

    def release(self) -> None:
        """Return one sibling slot and immediately admit the next waiter."""
        self._available += 1
        assert self._available <= self._capacity, "rollout gate released too many slots"
        self._try_admit()

    def _try_admit(self) -> None:
        while self._waiters and self._available:
            _, _, fut = heapq.heappop(self._waiters)
            if fut.done():  # cancelled/settled: drop and continue
                continue
            self._available -= 1
            fut.set_result(None)


class TMaxRollouter(Rollouter):
    """Drives a host-side ReAct agent (as root) in a sandbox per sibling, then runs
    the tmax verifier in that same sandbox."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: TMaxDataset.Config = field(
            default_factory=lambda: TMaxDataset.Config(seed=42)
        )
        validation_dataset: TMaxDataset.Config = field(
            default_factory=lambda: TMaxDataset.Config(seed=99, shuffle=False)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardTMax.Config(weight=1.0)],
                # RewardTMax returns zero when no verifier reward is present. Run it
                # for every status so failed rollouts remain in RewardTMax metrics.
            )
        )
        # Placeholder env (the agent loop runs in-sandbox; see env.py).
        message_env: TMaxEnv.Config = field(default_factory=TMaxEnv.Config)
        token_env: TokenEnv.Config = field(default_factory=TokenEnv.Config)
        # Centered (mean-baseline only), NOT std-normalized: matches the tmax
        # recipe's ``--advantage_normalization_type centered`` (qwen35_9b.sh).
        # Dividing by the group std amplifies rare-outcome advantages for
        # imbalanced binary-reward groups (e.g. a 30/32-pass group's 2 failures
        # get advantage ~ -3.9), which distorts the gradient and suppresses reward
        # growth; the recipe centers only to keep the advantage in [-1, 1].
        advantage: AdvantageEstimator.Config = field(
            default_factory=lambda: AdvantageEstimator.Config(
                should_std_normalize=False
            )
        )

        # Run knobs as CONFIG (not env), so they are serialized into the W&B run
        # config and each run's differences are visible. config_registry resolves
        # them from the launcher env once; the RolloutWorker pool overrides
        # rollout_concurrency to its per-worker share.
        rollout_concurrency: int = 16
        """Max concurrently-ACTIVE rollouts (per worker process). The pool total is
        num_rollout_workers x this. Each completed sibling immediately releases its
        slot to the next waiting rollout."""

        time_budget_sec: int = 2400
        """Per-rollout agent wall-clock budget (the vanillux loop stops after this)."""

        eval_timeout_sec: int = 600
        """Verifier (test.sh) run timeout."""

        max_context_tokens: int = 32768
        """Model context budget for the adapter session."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._time_budget_sec = config.time_budget_sec
        self._eval_timeout_sec = config.eval_timeout_sec
        self._max_context_tokens = config.max_context_tokens
        # Whole-rollout wall-clock guard: agent budget + eval + boot buffer.
        self._guard_sec = self._time_budget_sec + self._eval_timeout_sec + 300
        # Per-worker rollout-issue gate (one rollouter per worker proc). Each sibling
        # holds one slot, matching open-instruct's per-rollout environment acquire and
        # release. Ordering is per worker. The controller pins group-loop lanes to
        # workers, and each gate prioritizes the groups claimed by its lanes.
        self._rollout_gate = _RolloutIssueGate(config.rollout_concurrency)
        self._adapter: AnthropicAdapter | None = None
        self._adapter_lock = asyncio.Lock()

    async def _ensure_adapter(self, renderer: Renderer) -> AnthropicAdapter:
        if self._adapter is None:
            async with self._adapter_lock:
                if self._adapter is None:
                    # Direct in-process use: build the adapter for its Anthropic
                    # translation + TITO turn capture, but do NOT start() an HTTP
                    # server -- the vanillux loop calls adapter.complete() directly
                    # (no loopback HTTP, no per-worker port).
                    self._adapter = AnthropicAdapter(renderer=renderer)
        return self._adapter

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: TMaxSample,
        group_id: int,
        group_size: int,
        sampling: "SamplingConfig",
        renderer: Renderer,
    ) -> RolloutGroup:
        """Run + grade one prompt group of terminal-agent rollouts."""
        adapter = await self._ensure_adapter(renderer)

        rollout_tasks = [
            asyncio.create_task(
                self._run_agent_rollout(
                    adapter=adapter,
                    generate_fn=generate_fn,
                    sample=sample,
                    group_id=group_id,
                    rollout_idx=i,
                    sampling=sampling,
                    renderer=renderer,
                ),
                name=f"tmax_rollout_{group_id}_{i}",
            )
            for i in range(group_size)
        ]
        try:
            results = await asyncio.gather(*rollout_tasks)
        except BaseException:
            # Parent cancellation already propagates through gather. A child exception
            # does not, so let the other rollouts finish. In both cases drain without a
            # second cancel, which could interrupt an in-progress sandbox teardown.
            await asyncio.gather(*rollout_tasks, return_exceptions=True)
            raise
        rollouts = [rollout for rollout, _, _, _, _ in results]
        submitted_flags = [submitted for _, submitted, _, _, _ in results]
        fmt_errors_list = [fmt for _, _, fmt, _, _ in results]
        finish_reasons = [fr for _, _, _, fr, _ in results]
        sandbox_diagnostics = [diagnostics for _, _, _, _, diagnostics in results]

        # Fraction of siblings that never emitted the submit marker (errored or ran to
        # the budget without submitting) -> the verifier never runs -> auto reward 0.
        # Mirrors open-instruct's val/non_submitting_completion_fraction; a high value
        # means the scaffold (not the task difficulty) is capping reward.
        nonsubmit_frac = 1.0 - sum(submitted_flags) / len(submitted_flags)
        # Format errors (malformed tool-calls) per rollout + fraction of rollouts that
        # hit any -- surfaces the vanillux tool-call parse-failure rate on wandb.
        fmt_errors_mean = sum(fmt_errors_list) / len(fmt_errors_list)
        fmt_error_frac = sum(1.0 for f in fmt_errors_list if f > 0) / len(
            fmt_errors_list
        )
        # How each sibling's loop ended (run_vanillux_loop finish_reason), as a
        # per-reason fraction of the group. Surfaces the stop-reason split on wandb:
        # submit vs turn-cap vs 40min time-budget wall vs early-stop vs rollout error
        # (the "error" bucket covers timeout/exception paths where the loop never
        # returned). The five fracs sum to 1.0 per group; cudagraph should push
        # time_budget down and submit up.
        finish_metrics = _finish_reason_metrics(finish_reasons)
        sandbox_metrics = _sandbox_issue_metrics(
            [diagnostics.issue_counts for diagnostics in sandbox_diagnostics]
        )
        infra_failed_flags = [
            diagnostics.infra_failed for diagnostics in sandbox_diagnostics
        ]
        infra_failed_frac = sum(infra_failed_flags) / len(infra_failed_flags)
        group_metrics = [
            m.Metric("rollout/nonsubmit_frac", m.Mean(nonsubmit_frac)),
            m.Metric("rollout/format_errors_mean", m.Mean(fmt_errors_mean)),
            m.Metric("rollout/format_error_frac", m.Mean(fmt_error_frac)),
            m.Metric("rollout/infra_failed_frac", m.Mean(infra_failed_frac)),
            m.Metric(
                "rollout/infra_failed_group_frac",
                m.Mean(1.0 if any(infra_failed_flags) else 0.0),
            ),
            *finish_metrics,
            *sandbox_metrics,
        ]

        # Match Open-Instruct: after sandbox/reset retries are exhausted, keep the
        # failed sibling in the group with reward zero. It participates in centered
        # advantage estimation; an empty completion contributes no training tokens.
        if any(infra_failed_flags):
            logger.warning(
                "[tmax] group=%d retaining %d/%d infrastructure failures as "
                "zero-reward siblings",
                group_id,
                sum(infra_failed_flags),
                len(infra_failed_flags),
            )

        # Standard scoring + advantage path (mirrors Rollouter.run_group_rollouts).
        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        group = RolloutGroup(
            group_id=group_id,
            rollouts=rollouts,
            metrics=group_metrics,
        )
        advantages = self.advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        self._maybe_annotate_zero_std(sample, rollouts)
        return group

    def _maybe_annotate_zero_std(
        self, sample: TMaxSample, rollouts: list[Rollout]
    ) -> None:
        """Record this prompt's ``instance_id`` under ``SWE_ZERO_STD_DIR`` when its
        group has zero reward variance (all-pass or all-fail = no learning signal, so
        it is dropped by ``drop_zero_std_reward_groups``). A later run points
        ``TMaxDataset.skip_ids_path`` at the same dir to stop sampling these prompts.

        Writes ONE small file per prompt (``<instance_id>.json``), mirroring the
        rollout-trace dump: a write-once-per-file pattern is the one that persists on
        a network FUSE mount and is safe across the pooled RolloutWorker
        processes -- unlike appending to a single shared file, which object-store FUSE
        does not handle reliably. Re-encountering a prompt just overwrites its file
        (natural dedup). Best-effort; never raises into the rollout.
        """
        dump_dir = os.environ.get("SWE_ZERO_STD_DIR", "")
        if not dump_dir:
            return
        # Validation groups carry negative ids (see Controller). Their prompts are a
        # held-out or benchmark set, never sampled for training, so annotating them
        # would only pollute the skip list a later training run reads.
        if rollouts and rollouts[0].group_id < 0:
            return
        rewards = [r.reward for r in rollouts if r.reward is not None]
        if len(rewards) < 2 or statistics.pstdev(rewards) != 0.0:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            safe = sample.instance_id.replace("/", "_")
            path = os.path.join(dump_dir, f"{safe}.json")
            with open(path, "w") as f:
                json.dump({"instance_id": sample.instance_id, "reward": rewards[0]}, f)
        except OSError as e:
            logger.warning(f"[tmax] zero-std annotate failed for {dump_dir}: {e}")

    async def _run_agent_rollout(
        self,
        *,
        adapter: AnthropicAdapter,
        generate_fn: GenerateFn,
        sample: TMaxSample,
        group_id: int,
        rollout_idx: int,
        sampling: "SamplingConfig",
        renderer: Renderer,
    ) -> tuple[Rollout, bool, int, str, _SandboxRolloutDiagnostics]:
        """Boot a sandbox, run the agent as root, grade the task in place.

        Always returns ``(Rollout, submitted, fmt_errors, finish_reason,
        sandbox_diagnostics)`` (errors caught + marked terminal) so one bad sibling
        never fails the whole group.
        ``submitted`` is whether the agent emitted the submit marker (False on any
        error / no-submit); ``fmt_errors`` is the tool-call parse-failure count;
        ``finish_reason`` is how the loop ended (submit / hit_max_turns /
        hit_time_budget / stopped_early, or "error" on the timeout/exception paths
        where the loop never returned). The caller aggregates these into the group's
        ``rollout/nonsubmit_frac``, ``rollout/format_*`` and ``rollout/finish_*``
        metrics.
        """
        rollout_id = RolloutTurnID(
            group_id=group_id, rollout_id=rollout_idx, turn_id=0
        ).to_string(include_turn=False)

        status = RolloutStatus.ERROR
        reward = 0.0
        error_msg = ""
        submitted = False
        fmt_errors = 0  # total format errors this rollout (from run_vanillux_loop)
        # Default for the timeout/exception paths where run_vanillux_loop never
        # returned (it sets its own reason on the normal path).
        finish_reason = "error"
        infra_failed = False
        issue_tracker = SandboxIssueTracker(
            SandboxLogContext(
                instance_id=sample.instance_id,
                group_id=group_id,
                rollout_id=rollout_idx,
            )
        )
        sandbox: Sandbox | None = None
        rollout_timeout = None
        await self._rollout_gate.acquire_sibling((group_id, rollout_idx))
        try:
            # open_session is inside the try so a failure still releases the slot.
            adapter.open_session(
                rollout_id,
                generate_fn=generate_fn,
                sampling=sampling,
                routing_session_id=rollout_id,
                max_context_tokens=self._max_context_tokens,
            )
            async with asyncio.timeout(self._guard_sec) as rollout_timeout:
                # host_loop drives the sandbox with bash directly; it never runs the
                # Claude Code CLI, so skip the curl-based install (the tmax task
                # images have no curl, which would otherwise fail every boot).
                async with boot_agent_sandbox(
                    sample.image,
                    install_claude=False,
                    disk_gb=sample.daytona_disk_gb,
                    issue_tracker=issue_tracker,
                ) as sandbox:
                    # Force every tool command to run as root (tmax tasks touch
                    # system paths); the faithful Vanillux loop dispatches bash here.
                    root_sb = _RootSandbox(sandbox)
                    # Seed the agent-facing inputs (environment/seeds/* -> /workspace)
                    # BEFORE the agent runs -- upstream seeds at reset. Without this,
                    # seed-bearing tasks are unsolvable (inputs absent during rollout).
                    # Grading fixtures (tests/*) are uploaded later by grade_tmax.
                    await seed_workspace(root_sb, sample.tmax)
                    (
                        _turns,
                        submitted,
                        fmt_errors,
                        finish_reason,
                    ) = await run_vanillux_loop(
                        root_sb,
                        task=sample.problem_statement,
                        session_id=rollout_id,
                        adapter=adapter,
                        time_budget_sec=self._time_budget_sec,
                    )
                    # tmax runs the verifier only on the submit marker; a rollout that
                    # never submits scores 0 (matches SWERLVanilluxSandboxEnv). No
                    # git_diff: grade the agent's OWN sandbox in place.
                    if submitted:
                        reward = await grade_tmax(
                            sandbox,
                            sample.tmax,
                            workdir=sample.workdir,
                            timeout_sec=self._eval_timeout_sec,
                        )
                    else:
                        reward = 0.0
                status = RolloutStatus.COMPLETED
        except (TimeoutError, asyncio.TimeoutError):
            infra_failed = True
            reward = 0.0
            status = RolloutStatus.ERROR_TIMEOUT
            if rollout_timeout is not None and rollout_timeout.expired():
                logger.warning("[tmax] %s: wall-clock guard fired", rollout_id)
                error_msg = "wall_clock_timeout"
            else:
                logger.exception("[tmax] %s: sandbox timeout", rollout_id)
                error_msg = "sandbox_timeout"
        except Exception as e:
            infra_failed = True
            reward = 0.0
            logger.exception("[tmax] %s: rollout failed", rollout_id)
            status = RolloutStatus.ERROR
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            self._rollout_gate.release()
            captured = await adapter.finish_session(rollout_id)

        disk_gb = (
            sandbox.allocated_disk_gb
            if sandbox is not None
            else sample.daytona_disk_gb
            or int(os.environ.get("TT_DAYTONA_DISK_GB", "6"))
        )
        diagnostics = _SandboxRolloutDiagnostics(
            sandbox_id=sandbox.sandbox_id if sandbox is not None else "",
            disk_gb=disk_gb,
            issue_counts=issue_tracker.counts,
            issues=issue_tracker.issues,
            num_dropped_details=issue_tracker.num_dropped_details,
            infra_failed=infra_failed,
        )

        # Drop empty-completion turns so rollout_to_training_samples only sees
        # trainable turns (a non-final empty completion would otherwise raise).
        turns: list[RolloutTurn] = [
            RolloutTurn(
                rollout_id=RolloutTurnID(
                    group_id=group_id, rollout_id=rollout_idx, turn_id=turn_idx
                ),
                prompt_token_ids=ct.prompt_token_ids,
                completion_token_ids=ct.completion_token_ids,
                completion_logprobs=ct.completion_logprobs,
                min_policy_version=ct.min_policy_version,
                max_policy_version=ct.max_policy_version,
            )
            for turn_idx, ct in enumerate(
                ct for ct in captured if ct.completion_token_ids
            )
        ]

        # Load-bearing invariant: the off-policy age filter (controller) takes
        # min()/max() over each sample's turn policy versions, which TypeErrors
        # on None. Over-budget/empty turns carry version None and are dropped
        # above; assert none leaked so a violation fails here (clear) instead of
        # deep in the batcher.
        assert all(
            t.min_policy_version is not None and t.max_policy_version is not None
            for t in turns
        ), (
            f"{rollout_id}: a trainable RolloutTurn has a None policy version "
            "(empty-turn drop invariant broke)"
        )

        if not turns:
            status = RolloutStatus.ERROR
            if not error_msg:
                n_cap = len(captured)
                n_empty = sum(1 for ct in captured if not ct.completion_token_ids)
                error_msg = (
                    f"no_trainable_turns: captured={n_cap} empty_completions={n_empty}"
                )
        else:
            turns[-1].env_rewards = {TMAX_REWARD_KEY: float(reward)}

        if diagnostics.issue_counts:
            logger.warning(
                "[tmax_sandbox_summary] %s",
                json.dumps(
                    {
                        "event": "tmax_sandbox_summary",
                        "instance_id": sample.instance_id,
                        "group_id": group_id,
                        "rollout_id": rollout_idx,
                        "sandbox_id": diagnostics.sandbox_id,
                        "image": sample.image,
                        "disk_gb": diagnostics.disk_gb,
                        "status": str(status),
                        "submitted": submitted,
                        "reward": reward,
                        "finish_reason": finish_reason,
                        "infra_failed": diagnostics.infra_failed,
                        "issue_counts": diagnostics.issue_counts,
                        "num_dropped_details": diagnostics.num_dropped_details,
                    },
                    sort_keys=True,
                ),
            )

        logger.info(
            "[tmax] %s: status=%s reward=%.2f turns=%d",
            rollout_id,
            status,
            reward,
            len(turns),
        )
        self._maybe_dump_trace(
            rollout_id=rollout_id,
            group_id=group_id,
            sample=sample,
            captured=captured,
            renderer=renderer,
            status=str(status),
            reward=reward,
            submitted=submitted,
            fmt_errors=fmt_errors,
            error_msg=error_msg,
            finish_reason=finish_reason,
            sandbox_diagnostics=diagnostics,
        )
        return (
            Rollout(
                group_id=group_id,
                rollout_id=rollout_idx,
                status=status,
                turns=turns,
                # Keep the per-rollout loop outcome: the group metrics average these
                # away, but a trace report needs to say why THIS rollout stopped
                # (e.g. a turn truncated inside <think> that never emitted a
                # tool_call is a format_errors=1 / stopped_early rollout).
                diagnostics={
                    "finish_reason": finish_reason,
                    "format_errors": fmt_errors,
                    "submitted": submitted,
                    "infra_failed": diagnostics.infra_failed,
                },
            ),
            submitted,
            fmt_errors,
            finish_reason,
            diagnostics,
        )

    def _maybe_dump_trace(
        self,
        *,
        rollout_id: str,
        group_id: int,
        sample: TMaxSample,
        captured: list,
        renderer: Renderer,
        status: str,
        reward: float,
        submitted: bool = False,
        fmt_errors: int = 0,
        error_msg: str = "",
        finish_reason: str = "",
        sandbox_diagnostics: _SandboxRolloutDiagnostics,
    ) -> None:
        """Write a human-readable per-rollout training trace when
        ``SWE_ROLLOUT_DUMP_DIR`` is set. Format mirrors the open-instruct diagnostic
        trace: a summary header (reward / finish / submitted / num tool calls /
        response length) followed by the FULL decoded trajectory with the model's
        actions (``<tool_call>``) and the sandbox outputs (``<tool_response>``)
        interleaved -- reconstructed from the TITO-bridged prompts (turn N+1's prompt
        is turn N's prompt+completion + the new tool_response, so the growing prefix
        recovers each turn's sandbox output). Best-effort; never raises."""
        dump_dir = os.environ.get("SWE_ROLLOUT_DUMP_DIR", "")
        if not dump_dir:
            return
        # Validation groups (negative ids) get the controller's per-pass trace
        # report instead; dumping them here too would duplicate every transcript
        # into the training dump dir.
        if group_id < 0:
            return
        try:
            tokenizer = getattr(renderer, "tokenizer", None) or getattr(
                renderer, "_tokenizer", None
            )

            def _decode(ids: list[int]) -> str:
                if tokenizer is None or not ids:
                    return ""
                return tokenizer.decode(ids, skip_special_tokens=False)

            # Reconstruct the full interleaved token stream. TITO invariant: each turn's
            # prompt extends the previous prompt+completion, so appending each
            # completion and then the next prompt's delta recovers the whole
            # system+task+<tool_call>+<tool_response>+... conversation. A branch
            # (extends_previous False, e.g. compaction) resets to that turn's fresh
            # prompt (rare for tmax vanillux; noted inline).
            full_ids: list[int] = list(captured[0].prompt_token_ids) if captured else []
            branch_turns: list[int] = []
            for i, ct in enumerate(captured):
                full_ids += list(ct.completion_token_ids)
                if i + 1 < len(captured):
                    nxt = list(captured[i + 1].prompt_token_ids)
                    if nxt[: len(full_ids)] == full_ids:
                        full_ids += nxt[len(full_ids) :]  # tool_response delta
                    else:
                        branch_turns.append(i + 1)
                        full_ids = nxt  # history rewrite -> fresh render
            full_text = _decode(full_ids)

            response_len = sum(len(ct.completion_token_ids) for ct in captured)
            num_tool_calls = full_text.count("<tool_call>")
            last_model_finish = captured[-1].finish_reason if captured else None
            any_length_finish = any(ct.finish_reason == "length" for ct in captured)
            outcome = "SUCCESS" if reward and reward > 0 else "FAIL"

            header = (
                "=" * 90
                + f"\nTMAX-9B rollout trace  ({outcome}, reward={reward})\n"
                + "=" * 90
                + f"\ninstance_id    : {sample.instance_id}"
                + f"\nimage          : {sample.image}"
                + f"\nrollout_id     : {rollout_id}"
                + f"\nstatus         : {status}   submitted: {submitted}"
                + f"\nreward         : {reward}"
                + f"\nfinish_reason  : {finish_reason}"
                + f"\nmodel finish   : {last_model_finish}   any length-cap turn: {any_length_finish}"
                + f"\nsandbox_id     : {sandbox_diagnostics.sandbox_id}"
                + f"\nsandbox disk   : {sandbox_diagnostics.disk_gb} GiB"
                + f"\ninfra failed   : {sandbox_diagnostics.infra_failed}"
                + "\nsandbox issues : "
                + json.dumps(sandbox_diagnostics.issue_counts, sort_keys=True)
                + f" (details dropped={sandbox_diagnostics.num_dropped_details})"
                + f"\nnum turns      : {len(captured)}   num tool calls: {num_tool_calls}"
                + f"\nresponse length: {response_len} tokens (model-generated, all turns)"
                + f"\nformat_errors  : {fmt_errors}"
                + (f"\nerror          : {error_msg}" if error_msg else "")
                + (
                    f"\nNOTE branch/re-render at turns {branch_turns} (TITO not continued)"
                    if branch_turns
                    else ""
                )
                + "\n"
                + "=" * 90
                + "\nAGENT TRAJECTORY (decoded; <tool_call>=model action, "
                + "<tool_response>=sandbox output)\n"
                + "=" * 90
                + "\n"
            )

            os.makedirs(dump_dir, exist_ok=True)
            safe = rollout_id.replace("/", "_")
            path = os.path.join(dump_dir, f"{safe}.txt")
            with open(path, "w") as f:
                f.write(header)
                f.write(full_text)
                f.write("\n")
            logger.info("[tmax] rollout trace dumped: %s", path)
            if sandbox_diagnostics.issue_counts:
                issue_path = os.path.join(dump_dir, f"{safe}.sandbox.json")
                with open(issue_path, "w") as f:
                    json.dump(
                        {
                            "instance_id": sample.instance_id,
                            "image": sample.image,
                            "rollout_id": rollout_id,
                            "sandbox_id": sandbox_diagnostics.sandbox_id,
                            "disk_gb": sandbox_diagnostics.disk_gb,
                            "status": status,
                            "submitted": submitted,
                            "reward": reward,
                            "finish_reason": finish_reason,
                            "infra_failed": sandbox_diagnostics.infra_failed,
                            "error": error_msg,
                            "issue_counts": sandbox_diagnostics.issue_counts,
                            "num_dropped_details": (
                                sandbox_diagnostics.num_dropped_details
                            ),
                            "issues": [
                                asdict(issue) for issue in sandbox_diagnostics.issues
                            ],
                        },
                        f,
                        sort_keys=True,
                    )
                logger.info("[tmax] sandbox issue trace dumped: %s", issue_path)
        except Exception as e:
            logger.warning("[tmax] rollout trace dump failed: %s", e)

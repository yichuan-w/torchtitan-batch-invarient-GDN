# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.experiments.rl.examples.tmax.rollouter import (
    _RolloutIssueGate,
    TMaxRollouter,
)


@pytest.mark.parametrize("capacity", [0, -1])
def test_rollout_gate_rejects_nonpositive_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        _RolloutIssueGate(capacity=capacity)


def test_rollout_gate_rejects_release_without_acquire() -> None:
    gate = _RolloutIssueGate(capacity=1)
    with pytest.raises(AssertionError, match="released too many slots"):
        gate.release()


def test_rollout_gate_reuses_each_released_sibling_slot() -> None:
    async def run() -> None:
        group_size = 32
        gate = _RolloutIssueGate(capacity=group_size)

        for rollout_idx in range(group_size):
            await gate.acquire_sibling((0, rollout_idx))

        next_group = [
            asyncio.create_task(gate.acquire_sibling((1, rollout_idx)))
            for rollout_idx in range(group_size)
        ]
        await asyncio.sleep(0)
        assert not any(task.done() for task in next_group)

        gate.release()
        await asyncio.sleep(0)
        assert [idx for idx, task in enumerate(next_group) if task.done()] == [0]

        for task in next_group[1:]:
            task.cancel()
        await asyncio.gather(*next_group[1:], return_exceptions=True)

        # Return the 31 original slots and the one granted to the next group.
        for _ in range(group_size):
            gate.release()

    asyncio.run(run())


def test_rollout_gate_returns_slot_when_granted_waiter_is_cancelled() -> None:
    async def run() -> None:
        gate = _RolloutIssueGate(capacity=1)
        await gate.acquire_sibling((0, 0))

        granted_then_cancelled = asyncio.create_task(gate.acquire_sibling((1, 0)))
        await asyncio.sleep(0)

        # Grant the waiter and cancel it before the event loop resumes its task.
        gate.release()
        granted_then_cancelled.cancel()
        await asyncio.gather(granted_then_cancelled, return_exceptions=True)

        # The cancellation handler returned the granted slot.
        await asyncio.wait_for(gate.acquire_sibling((2, 0)), timeout=1.0)
        gate.release()

    asyncio.run(run())


def test_rollout_gate_admits_waiters_by_priority() -> None:
    async def run() -> None:
        gate = _RolloutIssueGate(capacity=1)
        await gate.acquire_sibling((0, 0))

        higher_id = asyncio.create_task(gate.acquire_sibling((2, 0)))
        lower_id = asyncio.create_task(gate.acquire_sibling((1, 0)))
        await asyncio.sleep(0)

        gate.release()
        await asyncio.sleep(0)
        assert lower_id.done()
        assert not higher_id.done()

        gate.release()
        await asyncio.wait_for(higher_id, timeout=1.0)
        gate.release()

    asyncio.run(run())


def test_rollout_gate_skips_cancelled_waiter_without_leaking_capacity() -> None:
    async def run() -> None:
        gate = _RolloutIssueGate(capacity=1)
        await gate.acquire_sibling((0, 0))

        cancelled = asyncio.create_task(gate.acquire_sibling((1, 0)))
        next_waiter = asyncio.create_task(gate.acquire_sibling((2, 0)))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)

        gate.release()
        await asyncio.wait_for(next_waiter, timeout=1.0)
        gate.release()

    asyncio.run(run())


def test_rollout_gate_recovers_capacity_when_active_group_is_cancelled() -> None:
    async def run() -> None:
        gate = _RolloutIssueGate(capacity=2)
        block = asyncio.Event()
        capacity_reached = asyncio.Event()
        entered: list[tuple[int, int]] = []

        async def hold_slot(priority: tuple[int, int]) -> None:
            await gate.acquire_sibling(priority)
            entered.append(priority)
            if len(entered) == 2:
                capacity_reached.set()
            try:
                await block.wait()
            finally:
                gate.release()

        async def run_group() -> None:
            await asyncio.gather(
                *(hold_slot((0, rollout_idx)) for rollout_idx in range(4))
            )

        group = asyncio.create_task(run_group())
        await asyncio.wait_for(capacity_reached.wait(), timeout=1.0)
        assert entered == [(0, 0), (0, 1)]

        # Cancel a gather with two active siblings and two waiting in the gate.
        group.cancel()
        await asyncio.gather(group, return_exceptions=True)

        await asyncio.wait_for(gate.acquire_sibling((1, 0)), timeout=1.0)
        await asyncio.wait_for(gate.acquire_sibling((1, 1)), timeout=1.0)
        gate.release()
        gate.release()

    asyncio.run(run())


def test_run_group_drains_remaining_siblings_when_one_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        rollouter = object.__new__(TMaxRollouter)
        all_started = asyncio.Event()
        num_started = 0
        failure_ready = asyncio.Event()
        allow_finish = asyncio.Event()
        completed: list[int] = []
        cancelled: list[int] = []

        async def ensure_adapter(self, renderer):
            return object()

        async def run_rollout(self, *, rollout_idx: int, **kwargs):
            nonlocal num_started
            num_started += 1
            if num_started == 3:
                all_started.set()
            await all_started.wait()
            if rollout_idx == 0:
                failure_ready.set()
                await asyncio.sleep(0)
                raise RuntimeError("rollout failed")
            try:
                await allow_finish.wait()
                completed.append(rollout_idx)
            except asyncio.CancelledError:
                cancelled.append(rollout_idx)
                raise

        monkeypatch.setattr(TMaxRollouter, "_ensure_adapter", ensure_adapter)
        monkeypatch.setattr(TMaxRollouter, "_run_agent_rollout", run_rollout)

        group = asyncio.create_task(
            rollouter.run_group_rollouts(
                generate_fn=None,
                sample=None,
                group_id=7,
                group_size=3,
                sampling=None,
                renderer=None,
            )
        )
        await asyncio.wait_for(failure_ready.wait(), timeout=1.0)
        await asyncio.sleep(0.01)

        # A child failure must not cancel siblings that may be in async teardown.
        assert not group.done()
        assert cancelled == []
        allow_finish.set()
        with pytest.raises(RuntimeError, match="rollout failed"):
            await asyncio.wait_for(group, timeout=1.0)
        assert sorted(completed) == [1, 2]
        assert cancelled == []

    asyncio.run(run())


def test_run_group_parent_cancellation_does_not_recancel_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        rollouter = object.__new__(TMaxRollouter)
        all_started = asyncio.Event()
        block = asyncio.Event()
        num_started = 0
        cleaned: list[int] = []

        async def ensure_adapter(self, renderer):
            return object()

        async def run_rollout(self, *, rollout_idx: int, **kwargs):
            nonlocal num_started
            num_started += 1
            if num_started == 3:
                all_started.set()
            try:
                await block.wait()
            except asyncio.CancelledError:
                # Model an async sandbox __aexit__ after cancellation reaches the body.
                await asyncio.sleep(0)
                cleaned.append(rollout_idx)
                raise

        monkeypatch.setattr(TMaxRollouter, "_ensure_adapter", ensure_adapter)
        monkeypatch.setattr(TMaxRollouter, "_run_agent_rollout", run_rollout)

        group = asyncio.create_task(
            rollouter.run_group_rollouts(
                generate_fn=None,
                sample=None,
                group_id=7,
                group_size=3,
                sampling=None,
                renderer=None,
            )
        )
        await asyncio.wait_for(all_started.wait(), timeout=1.0)
        group.cancel()
        result = await asyncio.gather(group, return_exceptions=True)

        assert isinstance(result[0], asyncio.CancelledError)
        assert sorted(cleaned) == [0, 1, 2]

    asyncio.run(run())


def test_rollout_gate_interleaves_groups_while_old_sibling_is_active() -> None:
    async def run() -> None:
        gate = _RolloutIssueGate(capacity=2)
        await gate.acquire_sibling((0, 0))
        await gate.acquire_sibling((0, 1))

        same_group = asyncio.create_task(gate.acquire_sibling((0, 2)))
        next_group = asyncio.create_task(gate.acquire_sibling((1, 0)))
        await asyncio.sleep(0)

        gate.release()
        await asyncio.wait_for(same_group, timeout=1.0)
        assert not next_group.done()

        # The second group starts while rollout (0, 1) is still active.
        gate.release()
        await asyncio.wait_for(next_group, timeout=1.0)

        gate.release()
        gate.release()

    asyncio.run(run())

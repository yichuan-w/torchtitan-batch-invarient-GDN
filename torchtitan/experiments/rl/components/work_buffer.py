# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run-ahead buffer of RolloutGroupWork shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finalized RolloutGroups necessarily.

"""

import asyncio
import collections
import enum
import logging
import time
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.observability import structured_logger as sl


logger = logging.getLogger(__name__)


class _RolloutGroupWorkState(enum.Enum):
    """Where a RolloutGroupWork is in the WAITING -> INFLIGHT -> FINALIZED lifecycle."""

    WAITING = "waiting"
    INFLIGHT = "inflight"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RolloutGroupWork:
    """One prompt group's work, tracked through _RolloutGroupWorkState.

    The input loop sets `group_id` + `sample`; the buffer owns `state` and `rollout_group`
    (`init=False`, so the input loop can't set them).
    """

    group_id: int
    sample: object
    """Data input produced by `rollouter.get_training_sample()`;
    passed unchanged to the env in `rollouter.run_group_rollouts`."""
    state: _RolloutGroupWorkState = field(
        default=_RolloutGroupWorkState.WAITING, init=False
    )
    rollout_group: RolloutGroup | None = field(
        default=None, init=False
    )  # set once FINALIZED
    # Monotonic wall-clock stamps for slot-occupancy observability: admitted_ts when the
    # group enters the buffer (charges a slot), claimed_ts on WAITING -> INFLIGHT. A slot's
    # occupancy (now - admitted_ts) surfaces stuck rollouts that hold an off-policy slot
    # without finalizing (e.g. a hung Daytona sandbox), which starve the trainer.
    admitted_ts: float | None = field(default=None, init=False)
    claimed_ts: float | None = field(default=None, init=False)
    # Number of later-admitted groups selected while this group was INFLIGHT.
    bypass_count: int = field(default=0, init=False)
    # TODO(async-rl): emit JSON lifecycle logging per RolloutGroupWork keyed by group_id:
    # admitted/claimed/finalized/batched/trained/dropped timestamps + policy version at admission and
    # at trainer consumption, for faithful end-to-end visibility.


class RolloutGroupWorkBuffer(Configurable):
    """Run-ahead buffer of RolloutGroupWork shared by the data-input, rollout, and batcher loops.

    Each entry is a RolloutGroupWork moving WAITING -> INFLIGHT -> FINALIZED. An active-slot budget caps
    run-ahead at `max_active_rollout_groups` active slots. By default, the batcher
    takes the oldest-admitted FINALIZED group while skipping INFLIGHT stragglers.
    A configured sliding selection window limits each selection to the first
    few active admissions while still allowing the prefix to refill after every
    selected group.

    For details on the buffer's callers, check the diagram in the controller.py file.

    NOTE: a work slot is **NOT** released when marked as FINALIZED or taken by the batcher.
    Instead, it is only released on `release_active_groups` calls by the trainer
    or data filtering. This is done this way so that we can guarantee we never have more
    than `max_active_rollout_groups` in the entire pipeline (buffer+queue+training).
    Otherwise, we would produce born-stale examples.

    Entry lifecycle vs active slot:
        entry:        WAITING -> INFLIGHT -> FINALIZED -> removed by take_finalized()
        active slot:  charged by add_work() ............ freed by release_active_groups()

    Example:
        # max_offpolicy_steps=1, num_groups_per_train_step=2 -> capacity=4
        await buffer.add_work(g0); await buffer.add_work(g1)   # 2/4 active
        await buffer.add_work(g2); await buffer.add_work(g3)   # 4/4 active (cap)
        g0 = await buffer.take_finalized()                     # g0 leaves the dict; still 4/4 active
        g1 = await buffer.take_finalized()                     # g1 leaves the dict; still 4/4 active
        slot_task = asyncio.create_task(buffer.wait_for_slot())  # waits: take_finalized did not free a slot
        assert not slot_task.done()
        await buffer.release_active_groups(2, reason="trained")  # trainer pulled -> a slot frees
        assert await slot_task                                   # wait_for_slot now returns
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Group-selection ordering; capacity is supplied by the controller."""

        num_groups_in_selection_window: int | None = None
        """Sliding admission-order selection window. None scans all active
        groups, 1 is strict FIFO, and larger values scan the first W entries in
        the current active map. Removing any selected group shifts later entries
        into the prefix, matching a FIFO element buffer."""

        max_bypass_groups: int | None = None
        """Stop selecting more groups while any INFLIGHT group has been bypassed
        by this many later selections. None disables the brake. The caller must
        guarantee that a stalled group eventually finalizes; this setting never
        evicts or times out inflight work."""

        strict_fifo: bool = False
        """Compatibility alias for ``num_groups_in_selection_window=1``."""

        def __post_init__(self) -> None:
            window = self.num_groups_in_selection_window
            if window is not None and window < 1:
                raise ValueError(
                    "num_groups_in_selection_window must be positive, got " f"{window}"
                )
            if self.strict_fifo and window not in (None, 1):
                raise ValueError(
                    "strict_fifo=True conflicts with "
                    f"num_groups_in_selection_window={window}; use 1 or unset it"
                )
            max_bypass = self.max_bypass_groups
            if max_bypass is not None and max_bypass < 1:
                raise ValueError(
                    f"max_bypass_groups must be positive, got {max_bypass}"
                )

        def resolved_num_groups_in_selection_window(self) -> int | None:
            """Resolve the legacy strict-FIFO alias to its one-group window."""
            if self.strict_fifo:
                return 1
            return self.num_groups_in_selection_window

    def __init__(
        self,
        config: Config,
        *,
        max_active_rollout_groups: int,
        initial_active_rollout_groups: int | None = None,
    ) -> None:
        self._num_groups_in_selection_window = (
            config.resolved_num_groups_in_selection_window()
        )
        self._max_bypass_groups = config.max_bypass_groups
        if max_active_rollout_groups < 1:
            raise ValueError(
                "max_active_rollout_groups must be positive, got "
                f"{max_active_rollout_groups}"
            )
        if (
            self._num_groups_in_selection_window is not None
            and self._num_groups_in_selection_window > max_active_rollout_groups
        ):
            raise ValueError(
                "num_groups_in_selection_window must not exceed "
                f"max_active_rollout_groups ({max_active_rollout_groups}), got "
                f"{self._num_groups_in_selection_window}"
            )
        if initial_active_rollout_groups is None:
            initial_active_rollout_groups = max_active_rollout_groups
        if not 1 <= initial_active_rollout_groups <= max_active_rollout_groups:
            raise ValueError(
                "initial_active_rollout_groups must be between 1 and "
                f"max_active_rollout_groups ({max_active_rollout_groups}), got "
                f"{initial_active_rollout_groups}"
            )
        self._max_active_rollout_groups = max_active_rollout_groups
        self._effective_active_rollout_groups = initial_active_rollout_groups
        self._active_rollout_groups = 0
        # metric: Per-flush peak active slots; reset on `.metrics()` call.
        self._active_rollout_groups_peak_since_flush = 0
        self._work_by_group_id: collections.OrderedDict[
            int, RolloutGroupWork
        ] = collections.OrderedDict()
        self._max_bypass_count = 0
        self._window_stall_started_ts: float | None = None
        self._window_stall_sec_since_flush = 0.0
        self._max_bypass_stall_started_ts: float | None = None
        self._max_bypass_stall_sec_since_flush = 0.0
        self._max_bypass_stall_count = 0
        # TODO(async-rl): Current we use a condition that alerts ALL rollout workers. There is no need to
        # alert all of them. Consider changing it to an async queue + event.

        # One Condition guards all three waits (slot-free / claimable-WAITING / takeable-FINALIZED):
        # every mutation notify_all()s and waiters re-check their predicate.
        self._condition = asyncio.Condition()
        self._closed = False

    def _has_active_slot_available(self) -> bool:
        return self._active_rollout_groups < self._effective_active_rollout_groups

    def _group_ids_in_selection_window(self) -> list[int]:
        window = self._num_groups_in_selection_window
        if window is None:
            return list(self._work_by_group_id)
        group_ids: list[int] = []
        for index, group_id in enumerate(self._work_by_group_id):
            if index == window:
                break
            group_ids.append(group_id)
        return group_ids

    def _num_blocked_finalized_groups(self, window_group_ids: set[int]) -> int:
        if self._num_groups_in_selection_window is None:
            return 0
        return sum(
            work.state is _RolloutGroupWorkState.FINALIZED
            and group_id not in window_group_ids
            for group_id, work in self._work_by_group_id.items()
        )

    def _finish_window_stall(self) -> None:
        if self._window_stall_started_ts is None:
            return
        self._window_stall_sec_since_flush += (
            time.monotonic() - self._window_stall_started_ts
        )
        self._window_stall_started_ts = None

    def _inflight_at_max_bypass(self) -> list[RolloutGroupWork]:
        max_bypass = self._max_bypass_groups
        if max_bypass is None:
            return []
        return [
            work
            for work in self._work_by_group_id.values()
            if work.state is _RolloutGroupWorkState.INFLIGHT
            and work.bypass_count >= max_bypass
        ]

    def _finish_max_bypass_stall(self) -> None:
        if self._max_bypass_stall_started_ts is None:
            return
        self._max_bypass_stall_sec_since_flush += (
            time.monotonic() - self._max_bypass_stall_started_ts
        )
        self._max_bypass_stall_started_ts = None

    async def wait_for_slot(self) -> bool:
        """Wait until one more rollout group may enter the active off-policy window.

        Example:
            # False means the buffer was closed, so the data input loop exits.
            group_index = 0
            while await buffer.wait_for_slot():
                await buffer.add_work(RolloutGroupWork(group_id=group_index, sample=sample))
                group_index += 1
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or self._has_active_slot_available()
            )
            return not self._closed

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Admit one rollout group as WAITING and charge one active slot.

        If close wins the race after ``wait_for_slot()`` returns, discard the
        prepared work without charging it.
        """
        async with self._condition:
            if self._closed:
                return
            if not self._has_active_slot_available():
                raise RuntimeError(
                    "RolloutGroupWorkBuffer.add_work called without an active slot"
                )
            self._active_rollout_groups += 1
            self._active_rollout_groups_peak_since_flush = max(
                self._active_rollout_groups_peak_since_flush,
                self._active_rollout_groups,
            )
            work.admitted_ts = time.monotonic()
            self._work_by_group_id[work.group_id] = work
            self._condition.notify_all()

    async def claim_next(self) -> RolloutGroupWork | None:
        """Rollout loop: claim the oldest WAITING group (WAITING -> INFLIGHT). None once closed."""
        async with self._condition:
            while True:
                if self._closed:
                    return None
                for work in self._work_by_group_id.values():
                    if work.state is _RolloutGroupWorkState.WAITING:
                        work.state = _RolloutGroupWorkState.INFLIGHT
                        work.claimed_ts = time.monotonic()
                        return work
                await self._condition.wait()

    async def finalize_work(self, rollout_group: RolloutGroup) -> None:
        """Rollout loop: store the produced RolloutGroup on its work entry (INFLIGHT -> FINALIZED) and wake the batcher."""
        async with self._condition:
            work = self._work_by_group_id.get(rollout_group.group_id)
            if work is None:
                # run()'s shutdown called close(); drop the result.
                return
            work.rollout_group = rollout_group
            work.state = _RolloutGroupWorkState.FINALIZED
            if not self._inflight_at_max_bypass():
                self._finish_max_bypass_stall()
            self._condition.notify_all()

    @sl.log_trace_span("take_finalized")
    async def take_finalized(self) -> RolloutGroup | None:
        """Return the oldest finalized group in the sliding active prefix.

        ``num_groups_in_selection_window=None`` preserves unbounded take-any.
        Otherwise, each selection scans the current active map's first W entries.
        Removing any selected group shifts later entries into that prefix, so the
        window limits instantaneous look-ahead but not lifetime bypass count.
        If an INFLIGHT group reaches ``max_bypass_groups``, selection stalls until
        that group finalizes.
        Active-slot accounting is unchanged: taking a group does not release its credit.

        Example:
            # W=2, g0 INFLIGHT, g1 FINALIZED, g2 FINALIZED -> returns g1, then
            # g2 because removing g1 shifts g2 into the current prefix.
            await buffer.take_finalized()
        """
        async with self._condition:
            try:
                while True:
                    if self._closed:
                        return None
                    inflight_at_max_bypass = self._inflight_at_max_bypass()
                    if inflight_at_max_bypass:
                        self._finish_window_stall()
                        if self._max_bypass_stall_started_ts is None:
                            self._max_bypass_stall_started_ts = time.monotonic()
                            self._max_bypass_stall_count += 1
                            logger.warning(
                                "Group selection stalled: %d INFLIGHT group(s) "
                                "reached max_bypass_groups=%d; group_ids=%s",
                                len(inflight_at_max_bypass),
                                self._max_bypass_groups,
                                [work.group_id for work in inflight_at_max_bypass],
                            )
                        await self._condition.wait()
                        continue
                    self._finish_max_bypass_stall()
                    window_group_ids = self._group_ids_in_selection_window()
                    bypassed_inflight_work: list[RolloutGroupWork] = []
                    for group_id in window_group_ids:
                        work = self._work_by_group_id.get(group_id)
                        if work is None:
                            continue
                        if work.state is _RolloutGroupWorkState.FINALIZED:
                            for older_work in bypassed_inflight_work:
                                older_work.bypass_count += 1
                                self._max_bypass_count = max(
                                    self._max_bypass_count, older_work.bypass_count
                                )
                            rollout_group = work.rollout_group
                            del self._work_by_group_id[group_id]
                            self._finish_window_stall()
                            self._condition.notify_all()
                            return rollout_group
                        if work.state is _RolloutGroupWorkState.INFLIGHT:
                            bypassed_inflight_work.append(work)

                    blocked_finalized = self._num_blocked_finalized_groups(
                        set(window_group_ids)
                    )
                    if blocked_finalized and self._window_stall_started_ts is None:
                        self._window_stall_started_ts = time.monotonic()
                    elif not blocked_finalized:
                        self._finish_window_stall()
                    await self._condition.wait()  # nothing takeable yet -> wait
            finally:
                self._finish_window_stall()
                self._finish_max_bypass_stall()

    async def release_active_groups(self, count: int, *, reason: str) -> None:
        """Free active slots: the trainer releases trained slots after its weight pull; the batcher
        releases untrainable/filtered slots immediately.

        Args:
            count:  Number of rollout groups leaving the active window.
            reason: Metric suffix such as `"trained"` or `"untrainable_group"`.

        Example:
            # trainer pulled weights after a step over 8 groups -> free their 8 slots
            await buffer.release_active_groups(8, reason="trained")
            # batcher dropped one zero-std group -> free its single slot
            await buffer.release_active_groups(1, reason="untrainable_group")
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return
        async with self._condition:
            if count > self._active_rollout_groups:
                raise RuntimeError(
                    f"release_active_groups({count}) exceeds active count {self._active_rollout_groups}"
                )
            self._active_rollout_groups -= count
            sl.log_trace_scalar({f"rollout_buffer/released/{reason}": float(count)})
            self._condition.notify_all()

    async def grow_effective_capacity(self) -> bool:
        """Grow cold-start admission headroom for retained downstream groups.

        A trainable group remains charged after the batcher takes it, so growing
        the effective capacity by one admits exactly one replacement generation
        group. Filtered and stale groups instead call ``release_active_groups`` and
        must not grow the capacity. Returns whether the capacity grew.
        """
        async with self._condition:
            if self._effective_active_rollout_groups == self._max_active_rollout_groups:
                return False
            self._effective_active_rollout_groups += 1
            sl.log_trace_scalar({"rollout_buffer/effective_capacity_growth": 1.0})
            self._condition.notify_all()
            return True

    async def close(self) -> None:
        """run() shutdown calls this once. Sets `_closed`, drops buffered work, and wakes every waiter.

        After this, all four waiters unblock and exit their loops: wait_for_slot() returns False, and
        claim_next()/take_finalized() return None.
        """
        async with self._condition:
            self._finish_window_stall()
            self._finish_max_bypass_stall()
            self._closed = True
            self._work_by_group_id.clear()
            self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        """Trainer loop: point-in-time buffer gauges for this step; resets the per-flush peak."""
        states = [work.state for work in self._work_by_group_id.values()]
        state_enum = _RolloutGroupWorkState
        # Slot occupancy (seconds since admission) of the groups still holding an active slot
        # but not yet finalized (WAITING + INFLIGHT). max surfaces a stuck straggler occupying
        # a slot; mean is the typical in-flight wait. Both are 0 when the buffer is empty.
        now = time.monotonic()
        window_group_ids = set(self._group_ids_in_selection_window())
        eligible_finalized_groups = sum(
            work.state is state_enum.FINALIZED and group_id in window_group_ids
            for group_id, work in self._work_by_group_id.items()
        )
        blocked_finalized_groups = self._num_blocked_finalized_groups(window_group_ids)
        num_inflight_at_max_bypass = len(self._inflight_at_max_bypass())
        head_work = next(iter(self._work_by_group_id.values()), None)
        head_wall_age_sec = (
            now - head_work.admitted_ts
            if head_work is not None and head_work.admitted_ts is not None
            else 0.0
        )
        window_stall_sec = self._window_stall_sec_since_flush
        if self._window_stall_started_ts is not None:
            window_stall_sec += now - self._window_stall_started_ts
            self._window_stall_started_ts = now
        self._window_stall_sec_since_flush = 0.0
        max_bypass_stall_sec = self._max_bypass_stall_sec_since_flush
        if self._max_bypass_stall_started_ts is not None:
            max_bypass_stall_sec += now - self._max_bypass_stall_started_ts
            self._max_bypass_stall_started_ts = now
        self._max_bypass_stall_sec_since_flush = 0.0
        occupancy_secs = [
            now - work.admitted_ts
            for work in self._work_by_group_id.values()
            if work.state is not state_enum.FINALIZED and work.admitted_ts is not None
        ]
        out = [
            # A zero window gauge means the optional bound is disabled.
            m.Metric(
                "rollout_buffer/selection_window_groups",
                m.NoReduce(float(self._num_groups_in_selection_window or 0)),
            ),
            m.Metric(
                "rollout_buffer/eligible_finalized_groups",
                m.NoReduce(float(eligible_finalized_groups)),
            ),
            m.Metric(
                "rollout_buffer/blocked_finalized_groups",
                m.NoReduce(float(blocked_finalized_groups)),
            ),
            m.Metric(
                "rollout_buffer/head_wall_age_sec",
                m.NoReduce(head_wall_age_sec),
            ),
            m.Metric(
                "rollout_buffer/head_bypass_count",
                m.NoReduce(float(head_work.bypass_count if head_work else 0)),
            ),
            m.Metric(
                "rollout_buffer/max_bypass_count",
                m.NoReduce(float(self._max_bypass_count)),
            ),
            m.Metric(
                "rollout_buffer/max_bypass_groups",
                m.NoReduce(float(self._max_bypass_groups or 0)),
            ),
            m.Metric(
                "rollout_buffer/num_inflight_at_max_bypass",
                m.NoReduce(float(num_inflight_at_max_bypass)),
            ),
            m.Metric(
                "rollout_buffer/max_bypass_stall_sec",
                m.NoReduce(max_bypass_stall_sec),
            ),
            m.Metric(
                "rollout_buffer/max_bypass_stall_count",
                m.NoReduce(float(self._max_bypass_stall_count)),
            ),
            m.Metric(
                "rollout_buffer/window_stall_sec",
                m.NoReduce(window_stall_sec),
            ),
            m.Metric(
                "rollout_buffer/slot_occupancy_max_sec",
                m.NoReduce(max(occupancy_secs, default=0.0)),
            ),
            m.Metric(
                "rollout_buffer/slot_occupancy_mean_sec",
                m.NoReduce(
                    sum(occupancy_secs) / len(occupancy_secs) if occupancy_secs else 0.0
                ),
            ),
            m.Metric(
                "rollout_buffer/num_groups_waiting",
                m.NoReduce(float(states.count(state_enum.WAITING))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_inflight",
                m.NoReduce(float(states.count(state_enum.INFLIGHT))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_finalized",
                m.NoReduce(float(states.count(state_enum.FINALIZED))),
            ),
            m.Metric(
                "rollout_buffer/active_slots_in_use_peak",
                m.NoReduce(float(self._active_rollout_groups_peak_since_flush)),
            ),
            m.Metric(
                "rollout_buffer/available_active_slots",
                m.NoReduce(
                    float(
                        self._effective_active_rollout_groups
                        - self._active_rollout_groups
                    )
                ),
            ),
            m.Metric(
                "rollout_buffer/effective_active_group_capacity",
                m.NoReduce(float(self._effective_active_rollout_groups)),
            ),
            m.Metric(
                "rollout_buffer/max_active_group_capacity",
                m.NoReduce(float(self._max_active_rollout_groups)),
            ),
        ]
        # Next interval starts from the current gauge, not 0: slots stay occupied across a flush.
        self._active_rollout_groups_peak_since_flush = self._active_rollout_groups
        return out

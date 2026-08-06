# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.components.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.experiments.rl.controller import (
    _should_drop_group_at_batcher,
    _split_rollout_concurrency,
    _will_validate_after_step,
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.controller_metrics import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    MetricsTimer,
)

from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
    rl_grpo_qwen3_0_6b_varlen,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.types import (
    RolloutTurnID,
    TrainingSample,
    TrainingSampleGroup,
)


def _tp2_batch_invariant_config(
    *,
    mixed_precision_param: Literal["bfloat16", "float32"],
    training_dtype: Literal["bfloat16", "float32"] = "float32",
) -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen()
    trainer_debug = dataclasses.replace(
        config.trainer.debug, batch_invariant=True, deterministic=True
    )
    generator_debug = dataclasses.replace(
        config.generator.debug, batch_invariant=True, deterministic=True
    )
    trainer = dataclasses.replace(
        config.trainer,
        debug=trainer_debug,
        training=dataclasses.replace(
            config.trainer.training,
            dtype=training_dtype,
            mixed_precision_param=mixed_precision_param,
        ),
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=2,
            enable_sequence_parallel=False,
        ),
    )
    generator = dataclasses.replace(config.generator, debug=generator_debug)
    return dataclasses.replace(config, trainer=trainer, generator=generator)


def test_batch_invariant_accepts_singleton_fsdp_mixed_precision() -> None:
    config = _tp2_batch_invariant_config(mixed_precision_param="bfloat16")

    assert config.trainer.training.dtype == "float32"
    assert config.trainer.parallelism.data_parallel_shard_degree == 1


def test_batch_invariant_keeps_full_bfloat16_training_supported() -> None:
    config = _tp2_batch_invariant_config(
        training_dtype="bfloat16", mixed_precision_param="bfloat16"
    )

    assert config.trainer.training.dtype == "bfloat16"


@pytest.mark.parametrize("training_dtype", ["bfloat16", "float32"])
def test_batch_invariant_rejects_float32_forward_parameters(
    training_dtype: Literal["bfloat16", "float32"],
) -> None:
    with pytest.raises(
        ValueError, match="batch_invariant requires bfloat16 forward parameters"
    ):
        _tp2_batch_invariant_config(
            training_dtype=training_dtype, mixed_precision_param="float32"
        )


def _training_sample(*, group_id: int, rollout_id: int) -> TrainingSample:
    return TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=group_id, rollout_id=rollout_id, turn_id=0),
        token_ids=[1, 2, 3],
        loss_mask=[False, True, True],
        logprobs=[0.0, 0.1, 0.2],
        advantage=[0.0, 1.0, 1.0],
    )


def _trainable_group(group_id: int, *, num_samples: int) -> TrainingSampleGroup:
    return TrainingSampleGroup(
        group_id=group_id,
        training_samples=[
            _training_sample(group_id=group_id, rollout_id=i)
            for i in range(num_samples)
        ],
        metrics=[],
    )


def _build_batcher(
    *, num_groups_per_train_step: int, initial_policy_version: int = 0
) -> Batcher:
    return Batcher.Config().build(
        num_groups_per_train_step=num_groups_per_train_step,
        dp_degree=1,
        pad_id=0,
        initial_policy_version=initial_policy_version,
    )


def _rollout_group(
    group_id: int,
    *,
    reward: float,
    min_policy_version: int,
) -> RolloutGroup:
    rollout_id = RolloutTurnID(group_id=group_id, rollout_id=0, turn_id=0)
    turn = RolloutTurn(
        rollout_id=rollout_id,
        prompt_token_ids=[1],
        completion_token_ids=[2],
        completion_logprobs=[0.0],
        min_policy_version=min_policy_version,
        max_policy_version=min_policy_version,
    )
    rollout = Rollout(
        group_id=group_id,
        rollout_id=0,
        turns=[turn],
        status=RolloutStatus.COMPLETED,
        reward=reward,
    )
    return RolloutGroup(group_id=group_id, rollouts=[rollout])


@pytest.mark.parametrize(
    ("global_concurrency", "num_workers", "max_num_workers", "expected"),
    [
        (512, 8, None, [64] * 8),
        (10, 3, None, [4, 3, 3]),
        (3, 8, None, [1, 1, 1]),
        (512, 64, 40, [13] * 32 + [12] * 8),
    ],
)
def test_split_rollout_concurrency_preserves_global_limit(
    global_concurrency: int,
    num_workers: int,
    max_num_workers: int | None,
    expected: list[int],
) -> None:
    capacities = _split_rollout_concurrency(
        global_concurrency, num_workers, max_num_workers=max_num_workers
    )

    assert capacities == expected
    assert sum(capacities) == global_concurrency
    assert min(capacities) >= 1
    assert max(capacities) - min(capacities) <= 1


@pytest.mark.parametrize(
    ("global_concurrency", "num_workers", "max_num_workers"),
    [(0, 1, None), (1, 0, None), (-1, 1, None), (1, -1, None), (1, 1, 0)],
)
def test_split_rollout_concurrency_rejects_nonpositive_values(
    global_concurrency: int, num_workers: int, max_num_workers: int | None
) -> None:
    with pytest.raises(ValueError):
        _split_rollout_concurrency(
            global_concurrency, num_workers, max_num_workers=max_num_workers
        )


@pytest.mark.parametrize(
    ("group_age", "max_offpolicy_steps", "expected"),
    [
        (0, 0, False),
        (1, 0, True),
        (3, 4, False),
        (4, 4, False),
        (5, 4, True),
    ],
)
def test_batcher_staleness_keeps_fully_on_policy_groups(
    group_age: int, max_offpolicy_steps: int, expected: bool
) -> None:
    assert (
        _should_drop_group_at_batcher(
            group_age=group_age, max_offpolicy_steps=max_offpolicy_steps
        )
        is expected
    )


def test_validation_group_ids_are_unique_across_passes() -> None:
    controller = object.__new__(Controller)
    controller._next_validation_group_id = -1

    assert controller._allocate_validation_group_ids(3) == [-1, -2, -3]
    assert controller._allocate_validation_group_ids(2) == [-4, -5]


@pytest.mark.parametrize("run_async", [False, True])
def test_zero_step_run_only_validates_once(run_async: bool) -> None:
    """With no training steps there is nothing to overlap, so the eval-only path
    stays blocking even when the pass is configured async."""

    async def run() -> None:
        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(
            async_loop=SimpleNamespace(
                num_training_steps=0,
                validation=ValidationConfig(run_async=run_async),
            )
        )
        controller.start_step = 0
        controller._validate_and_log = AsyncMock(return_value={})

        await controller.run()

        controller._validate_and_log.assert_awaited_once_with(step=0)
        assert not hasattr(controller, "_group_buffer")

    asyncio.run(run())


@pytest.mark.parametrize(
    ("num_samples", "interval", "step", "num_training_steps", "expected"),
    [
        (0, 10, 10, 200, False),
        (30, 0, 10, 200, False),
        (30, 10, 10, 200, True),
        (30, 10, 11, 200, False),
        (30, 10, 200, 200, True),
        (30, 0, 200, 200, True),
    ],
)
def test_will_validate_after_step(
    num_samples: int,
    interval: int,
    step: int,
    num_training_steps: int,
    expected: bool,
) -> None:
    validation = ValidationConfig(num_samples=num_samples, interval=interval)

    assert (
        _will_validate_after_step(
            validation=validation,
            step=step,
            num_training_steps=num_training_steps,
        )
        is expected
    )


@pytest.mark.parametrize(
    "step, trainer_policy_version, expected_log_step",
    [
        # Blocking pass: the trainer sits on the step being validated.
        (10, 10, 10),
        # Background pass: the trainer has moved on, so the metrics land on the live
        # step -- W&B drops anything logged behind the last committed one. The
        # evaluated version travels in validation/policy_version instead.
        (10, 17, 17),
    ],
)
def test_validate_and_log_commits_validation_metrics(
    step: int, trainer_policy_version: int, expected_log_step: int
) -> None:
    async def run() -> None:
        controller = object.__new__(Controller)
        controller._trainer_policy_version = trainer_policy_version
        controller._validation_results = {}
        metrics = [m.Metric("validation/reward", m.NoReduce(0.5))]
        controller.validate = AsyncMock(return_value=metrics)
        controller.metrics_processor = MagicMock()

        aggregated = await controller._validate_and_log(step=step)

        assert aggregated == {"validation/reward": 0.5}
        controller.metrics_processor.log.assert_called_once_with(
            step=expected_log_step,
            metrics=metrics,
            is_validation=True,
            commit=True,
        )

    asyncio.run(run())


def test_trainer_requires_batch_reserved_for_live_policy() -> None:
    async def run() -> None:
        controller = object.__new__(Controller)
        controller._trainer_policy_version = 5
        controller.config = SimpleNamespace(
            async_loop=AsyncLoopConfig(max_offpolicy_steps=3)
        )
        queue = asyncio.Queue()
        await queue.put(
            SimpleNamespace(target_policy_version=4, min_policy_versions=[4])
        )

        with pytest.raises(RuntimeError, match="wrong policy version"):
            await controller._take_reserved_training_batch(queue)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("min_policy_version", "raises"),
    [(1, False), (0, True)],
)
def test_trainer_verifies_reserved_batch_freshness(
    min_policy_version: int, raises: bool
) -> None:
    async def run() -> None:
        controller = object.__new__(Controller)
        controller._trainer_policy_version = 5
        controller.config = SimpleNamespace(
            async_loop=AsyncLoopConfig(max_offpolicy_steps=4)
        )
        packed = SimpleNamespace(
            target_policy_version=5,
            min_policy_versions=[min_policy_version],
        )
        queue = asyncio.Queue()
        await queue.put(packed)

        if raises:
            with pytest.raises(RuntimeError, match="admitted stale training data"):
                await controller._take_reserved_training_batch(queue)
        else:
            assert await controller._take_reserved_training_batch(queue) is packed

    asyncio.run(run())


def test_batcher_counts_trainable_groups_not_rollouts() -> None:
    # Target is 2 GROUPS. A single group with many rollouts is not a full batch; two groups are,
    # regardless of how many rollouts each contributes.
    batcher = _build_batcher(num_groups_per_train_step=2)
    assert (
        batcher.add_training_samples(
            training_sample_group=_trainable_group(0, num_samples=8)
        )
        is None
    )
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=1)
    )
    assert batch is not None
    assert batch.target_policy_version == 0
    assert batcher.next_batch_policy_version == 1


def test_batcher_reserves_policy_versions_in_order() -> None:
    batcher = _build_batcher(num_groups_per_train_step=1, initial_policy_version=7)

    first = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=1)
    )
    second = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=1)
    )

    assert first is not None and first.target_policy_version == 7
    assert second is not None and second.target_policy_version == 8
    assert batcher.next_batch_policy_version == 9


def test_batcher_carries_metric_only_groups_until_trainable_batch() -> None:
    # Metric-only (empty) groups do not count toward the target and cannot form a zero-token batch;
    # they ride along until a trainable group completes the batch.
    batcher = _build_batcher(num_groups_per_train_step=1)
    metric_only = TrainingSampleGroup(group_id=0, training_samples=[], metrics=[])
    assert batcher.add_training_samples(training_sample_group=metric_only) is None
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )
    assert batch is not None
    assert batch.num_global_valid_tokens > 0


def test_microbatch_grid_spreads_pad_rows_across_cells() -> None:
    # 5 real rows, local_batch_size=2, dp_degree=2 -> 4 cells x 2 = 8 rows (3 pad).
    # Round-robin dealing spreads the pad rows so no (microbatch, rank) cell is all-pad.
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=2)).build(
        num_groups_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        initial_policy_version=0,
    )
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )
    assert batch is not None
    cells = [microbatch for ranks in batch.microbatches for microbatch in ranks]
    assert len(cells) == 4  # 2 microbatches x 2 ranks
    for cell in cells:
        assert cell.loss_mask.any(dim=1).any()  # at least one real (non-pad) row


def test_compute_perf_ratio_metrics_reads_flushed_means() -> None:
    time_metrics = [
        m.Metric("timing/step/total", m.Mean.from_list([2.0])),
        m.Metric("timing/step/forward_backward", m.Mean.from_list([0.5])),
        m.Metric("timing/step/optim", m.Mean.from_list([0.5])),
    ]
    ratios = {
        metric.key: metric.value.value
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100, time_metrics=time_metrics
        )
    }
    assert ratios["perf/trainer/tokens_per_second_full_step"] == 50.0
    assert ratios["perf/trainer/step_time_ratio/fwd_bwd"] == 0.5
    assert ratios["perf/trainer/tokens_per_second_fwd_bwd"] == 100.0


def test_compute_perf_ratio_metrics_skips_missing_spans() -> None:
    # Only `total` recorded -> emit the full-step throughput, skip every ratio whose span is absent.
    time_metrics = [m.Metric("timing/step/total", m.Mean.from_list([2.0]))]
    keys = {
        metric.key
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100, time_metrics=time_metrics
        )
    }
    assert keys == {
        "batch/num_global_valid_tokens",
        "perf/trainer/tokens_per_second_full_step",
    }


def test_compute_perf_ratio_metrics_emits_token_count_without_total() -> None:
    metrics = compute_perf_ratio_metrics(num_global_valid_tokens=100, time_metrics=[])
    assert len(metrics) == 1
    assert metrics[0].key == "batch/num_global_valid_tokens"


def test_metrics_timer_flush_drains() -> None:
    timer = MetricsTimer()
    with timer.record("timing/x"):
        pass
    assert timer.flush()  # non-empty on first read
    assert timer.flush() == []  # drained on the second read


def test_rollout_id_to_string_is_callable_and_uses_int_group_id() -> None:
    rollout_id = RolloutTurnID(group_id=5, rollout_id=2, turn_id=0)
    assert rollout_id.to_string() == "group=5/rollout=2/turn=0"
    assert rollout_id.to_string(include_turn=False) == "group=5/rollout=2"


def test_take_finalized_does_not_release_active_slot() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=1)
        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        await buffer.take_finalized()

        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()

        await buffer.release_active_groups(1, reason="trained")
        assert await waiter

    asyncio.run(run())


def test_cold_start_capacity_grows_to_full_limit() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(
            max_active_rollout_groups=3,
            initial_active_rollout_groups=1,
        )

        assert await buffer.wait_for_slot()
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))

        second_slot = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not second_slot.done()
        assert await buffer.grow_effective_capacity()
        assert await second_slot
        await buffer.add_work(RolloutGroupWork(group_id=1, sample=object()))

        third_slot = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not third_slot.done()
        assert await buffer.grow_effective_capacity()
        assert await third_slot
        await buffer.add_work(RolloutGroupWork(group_id=2, sample=object()))
        assert not await buffer.grow_effective_capacity()

        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/effective_active_group_capacity"] == 3
        assert values["rollout_buffer/max_active_group_capacity"] == 3
        assert values["rollout_buffer/available_active_slots"] == 0

        blocked = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not blocked.done()
        await buffer.close()
        assert not await blocked

    asyncio.run(run())


def test_dropped_group_releases_without_growing_cold_start_capacity() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(
            max_active_rollout_groups=2,
            initial_active_rollout_groups=1,
        )
        assert await buffer.wait_for_slot()
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))
        await buffer.release_active_groups(1, reason="untrainable_group")

        assert await buffer.wait_for_slot()
        await buffer.add_work(RolloutGroupWork(group_id=1, sample=object()))
        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/effective_active_group_capacity"] == 1
        assert values["rollout_buffer/available_active_slots"] == 0

    asyncio.run(run())


def test_batcher_loop_replenishes_each_group_exactly_once() -> None:
    class FakeGroupBuffer:
        def __init__(self) -> None:
            self.groups = iter(
                [
                    _rollout_group(0, reward=0.0, min_policy_version=5),
                    # Fresh against the live trainer (5 - 1 == 4), but stale for
                    # the batch reserved for policy version 6 (6 - 1 == 5).
                    _rollout_group(1, reward=1.0, min_policy_version=1),
                    _rollout_group(2, reward=0.5, min_policy_version=5),
                ]
            )
            self.releases: list[tuple[int, str]] = []
            self.num_capacity_growths = 0

        async def take_finalized(self) -> RolloutGroup | None:
            return next(self.groups, None)

        async def release_active_groups(self, count: int, *, reason: str) -> None:
            self.releases.append((count, reason))

        async def grow_effective_capacity(self) -> bool:
            self.num_capacity_growths += 1
            return True

    class FakeTrainingSampleBuilder:
        def __init__(self) -> None:
            self.group_ids: list[int] = []

        def build_from_group(
            self, *, rollout_group: RolloutGroup
        ) -> TrainingSampleGroup:
            self.group_ids.append(rollout_group.group_id)
            if rollout_group.group_id == 0:
                return TrainingSampleGroup(
                    group_id=rollout_group.group_id,
                    training_samples=[],
                    metrics=[],
                )
            return _trainable_group(rollout_group.group_id, num_samples=1)

    async def run() -> None:
        controller = object.__new__(Controller)
        controller._trainer_policy_version = 5
        controller.config = SimpleNamespace(
            async_loop=AsyncLoopConfig(max_offpolicy_steps=4)
        )
        group_buffer = FakeGroupBuffer()
        training_batch_queue = asyncio.Queue()
        training_sample_builder = FakeTrainingSampleBuilder()

        await controller._batcher_loop(
            group_buffer=group_buffer,
            training_sample_builder=training_sample_builder,
            batcher=_build_batcher(
                num_groups_per_train_step=1, initial_policy_version=6
            ),
            training_batch_queue=training_batch_queue,
        )

        assert group_buffer.releases == [
            (1, "untrainable_group"),
            (1, "stale_dropped"),
        ]
        assert training_sample_builder.group_ids == [0, 2]
        assert group_buffer.num_capacity_growths == 1
        packed = await training_batch_queue.get()
        assert packed is not None
        assert packed.target_policy_version == 6
        assert await training_batch_queue.get() is None

    asyncio.run(run())


def test_initial_active_groups_must_not_exceed_full_capacity() -> None:
    with pytest.raises(ValueError, match="initial_active_rollout_groups"):
        AsyncLoopConfig(
            max_active_rollout_groups=8,
            initial_active_rollout_groups=9,
        )


def test_full_active_capacity_must_fit_one_train_batch() -> None:
    with pytest.raises(ValueError, match="num_groups_per_train_step"):
        AsyncLoopConfig(
            num_groups_per_train_step=8,
            max_active_rollout_groups=7,
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"num_groups_per_train_step": 0}, "num_groups_per_train_step"),
        ({"group_size": 0}, "group_size"),
        (
            {"max_offpolicy_steps": -1, "max_active_rollout_groups": 8},
            "max_offpolicy_steps",
        ),
    ],
)
def test_async_loop_counts_must_be_valid(overrides: dict[str, int], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        AsyncLoopConfig(**overrides)


def test_take_finalized_skips_inflight_straggler() -> None:
    """take-any: a still-INFLIGHT head must NOT block taking a later FINALIZED group."""

    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=2)
        for gid in (0, 1):
            if not await buffer.wait_for_slot():
                raise RuntimeError("buffer closed unexpectedly")
            await buffer.add_work(RolloutGroupWork(group_id=gid, sample=object()))
        # Claim both (WAITING -> INFLIGHT); finalize only the SECOND. The head g0
        # stays INFLIGHT -- strict FIFO would stall here; take-any returns g1.
        assert (await buffer.claim_next()).group_id == 0
        assert (await buffer.claim_next()).group_id == 1
        await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
        taken = await buffer.take_finalized()
        assert taken is not None and taken.group_id == 1

    asyncio.run(run())


def test_take_finalized_uses_admission_order_when_multiple_are_ready() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=2)
        for group_id in (0, 1):
            assert await buffer.wait_for_slot()
            await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))
            assert (await buffer.claim_next()).group_id == group_id

        # Finalize in reverse order before the batcher scans. The buffer chooses
        # the older admission, not the earlier completion timestamp.
        await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        taken = await buffer.take_finalized()
        assert taken is not None and taken.group_id == 0

    asyncio.run(run())


def test_sliding_selection_window_refills_after_each_take() -> None:
    async def run() -> None:
        window = 2
        buffer = RolloutGroupWorkBuffer.Config(
            num_groups_in_selection_window=window
        ).build(max_active_rollout_groups=4)
        for group_id in range(4):
            assert await buffer.wait_for_slot()
            await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))
            assert (await buffer.claim_next()).group_id == group_id

        # Group 2 is initially outside [0, 1], so it cannot be selected yet.
        await buffer.finalize_work(RolloutGroup(group_id=2, rollouts=[]))
        blocked_take = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0.001)
        assert not blocked_take.done()

        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/selection_window_groups"] == window
        assert values["rollout_buffer/eligible_finalized_groups"] == 0
        assert values["rollout_buffer/blocked_finalized_groups"] == 1
        assert values["rollout_buffer/window_stall_sec"] > 0
        assert values["rollout_buffer/available_active_slots"] == 0

        # Completing group 1 makes it selectable. Removing it shifts group 2
        # into [0, 2], without waiting for the inflight head group 0.
        await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
        assert (await asyncio.wait_for(blocked_take, timeout=1)).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2

        # Repeating the slide lets the same head be bypassed more than W - 1 times.
        await buffer.finalize_work(RolloutGroup(group_id=3, rollouts=[]))
        assert (await buffer.take_finalized()).group_id == 3
        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/head_bypass_count"] == 3
        assert values["rollout_buffer/max_bypass_count"] == 3
        assert values["rollout_buffer/head_bypass_count"] > window - 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("num_groups_in_selection_window", "strict_fifo"),
    [(1, False), (None, True)],
)
def test_one_group_selection_window_matches_strict_fifo(
    num_groups_in_selection_window: int | None, strict_fifo: bool
) -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(
            num_groups_in_selection_window=num_groups_in_selection_window,
            strict_fifo=strict_fifo,
        ).build(max_active_rollout_groups=2)
        for group_id in (0, 1):
            assert await buffer.wait_for_slot()
            await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))
            assert (await buffer.claim_next()).group_id == group_id

        await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
        blocked_take = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not blocked_take.done()

        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        assert (await asyncio.wait_for(blocked_take, timeout=1)).group_id == 0
        assert (await buffer.take_finalized()).group_id == 1

    asyncio.run(run())


def test_close_wakes_take_blocked_by_selection_window() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(num_groups_in_selection_window=2).build(
            max_active_rollout_groups=3
        )
        for group_id in range(3):
            assert await buffer.wait_for_slot()
            await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))
            assert (await buffer.claim_next()).group_id == group_id

        await buffer.finalize_work(RolloutGroup(group_id=2, rollouts=[]))
        blocked_take = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not blocked_take.done()
        await buffer.close()
        assert await asyncio.wait_for(blocked_take, timeout=1) is None

    asyncio.run(run())


def test_add_work_does_not_charge_after_close() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(num_groups_in_selection_window=1).build(
            max_active_rollout_groups=1
        )
        assert await buffer.wait_for_slot()
        await buffer.close()

        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))

        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/num_groups_waiting"] == 0
        assert values["rollout_buffer/active_slots_in_use_peak"] == 0
        assert await buffer.claim_next() is None
        assert await buffer.take_finalized() is None

    asyncio.run(run())


@pytest.mark.parametrize("window", [0, -1])
def test_selection_window_must_be_positive(window: int) -> None:
    with pytest.raises(ValueError, match="num_groups_in_selection_window"):
        RolloutGroupWorkBuffer.Config(num_groups_in_selection_window=window)


def test_selection_window_rejects_conflicting_strict_fifo() -> None:
    with pytest.raises(ValueError, match="strict_fifo=True conflicts"):
        RolloutGroupWorkBuffer.Config(
            num_groups_in_selection_window=2,
            strict_fifo=True,
        )


@pytest.mark.parametrize("max_bypass", [0, -1])
def test_max_bypass_must_be_positive(max_bypass: int) -> None:
    with pytest.raises(ValueError, match="max_bypass_groups"):
        RolloutGroupWorkBuffer.Config(max_bypass_groups=max_bypass)


def test_selection_window_must_fit_active_capacity() -> None:
    with pytest.raises(ValueError, match="must not exceed max_active_rollout_groups"):
        RolloutGroupWorkBuffer.Config(num_groups_in_selection_window=3).build(
            max_active_rollout_groups=2
        )


def test_async_loop_rejects_selection_window_larger_than_capacity() -> None:
    with pytest.raises(ValueError, match="must not exceed max_active_rollout_groups"):
        AsyncLoopConfig(
            num_groups_per_train_step=1,
            max_active_rollout_groups=2,
            group_buffer=RolloutGroupWorkBuffer.Config(
                num_groups_in_selection_window=3
            ),
        )


def test_cancelled_window_take_stops_stall_timer() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(num_groups_in_selection_window=2).build(
            max_active_rollout_groups=3
        )
        for group_id in range(3):
            assert await buffer.wait_for_slot()
            await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))
            assert (await buffer.claim_next()).group_id == group_id

        await buffer.finalize_work(RolloutGroup(group_id=2, rollouts=[]))
        blocked_take = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0.001)
        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/window_stall_sec"] > 0

        blocked_take.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked_take
        buffer.metrics()  # Drain the final partial interval recorded on cancellation.
        await asyncio.sleep(0.001)
        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/window_stall_sec"] == 0

    asyncio.run(run())


def test_max_bypass_stalls_until_inflight_group_finishes() -> None:
    async def run() -> None:
        max_bypass = 2
        buffer = RolloutGroupWorkBuffer.Config(
            num_groups_in_selection_window=2,
            max_bypass_groups=max_bypass,
        ).build(max_active_rollout_groups=4)
        for group_id in range(4):
            assert await buffer.wait_for_slot()
            await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))
            assert (await buffer.claim_next()).group_id == group_id

        for group_id in (1, 2):
            await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))
            assert (await buffer.take_finalized()).group_id == group_id

        # Group 3 is now inside the sliding prefix, but group 0 has reached the
        # bypass limit. Max-age protection stalls further selection.
        await buffer.finalize_work(RolloutGroup(group_id=3, rollouts=[]))
        blocked_take = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0.001)
        assert not blocked_take.done()
        values = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert values["rollout_buffer/max_bypass_groups"] == max_bypass
        assert values["rollout_buffer/num_inflight_at_max_bypass"] == 1
        assert values["rollout_buffer/max_bypass_stall_sec"] > 0
        assert values["rollout_buffer/max_bypass_stall_count"] == 1
        assert values["rollout_buffer/eligible_finalized_groups"] == 1
        assert values["rollout_buffer/available_active_slots"] == 0

        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        assert (await asyncio.wait_for(blocked_take, timeout=1)).group_id == 0
        assert (await buffer.take_finalized()).group_id == 3

    asyncio.run(run())


def test_replenishment_enters_sliding_selection_window() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(num_groups_in_selection_window=2).build(
            max_active_rollout_groups=3
        )
        for group_id in (0, 1, 2):
            assert await buffer.wait_for_slot()
            await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))
            assert (await buffer.claim_next()).group_id == group_id

        await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
        assert (await buffer.take_finalized()).group_id == 1
        await buffer.release_active_groups(1, reason="untrainable_group")

        assert await buffer.wait_for_slot()
        await buffer.add_work(RolloutGroupWork(group_id=3, sample=object()))
        assert (await buffer.claim_next()).group_id == 3
        await buffer.finalize_work(RolloutGroup(group_id=3, rollouts=[]))

        # Group 3 was appended outside [0, 2], so it remains blocked until 2 is
        # selected and shifts the current prefix to [0, 3].
        blocked_take = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not blocked_take.done()
        await buffer.finalize_work(RolloutGroup(group_id=2, rollouts=[]))
        assert (await asyncio.wait_for(blocked_take, timeout=1)).group_id == 2
        assert (await buffer.take_finalized()).group_id == 3

    asyncio.run(run())


def test_untrainable_group_releases_before_training() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=1)
        batcher = Batcher.Config().build(
            num_groups_per_train_step=1,
            dp_degree=1,
            pad_id=0,
            initial_policy_version=0,
        )

        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))

        training_sample_group = TrainingSampleGroup(
            group_id=0, training_samples=[], metrics=[]
        )
        await buffer.release_active_groups(1, reason="untrainable_group")
        assert (
            batcher.add_training_samples(training_sample_group=training_sample_group)
            is None
        )

    asyncio.run(run())


def test_compute_policy_age_metrics_raises_on_consume_time_staleness() -> None:
    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=4,
            min_policy_versions=[0],
            max_offpolicy_steps=3,
        )


def test_async_validation_skips_while_a_pass_is_in_flight() -> None:
    """A second pass must not start under the first: it would also mean pulling new
    weights into the eval generators mid-pass, mislabelling the version scored."""

    async def run() -> None:
        controller = object.__new__(Controller)
        controller._validation_task = None
        controller.metrics_processor = MagicMock()
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_validate(*, step: int):
            started.set()
            await release.wait()
            return []

        controller._validate_and_log = _slow_validate

        assert not controller._validation_pass_in_flight()
        controller._start_async_validation(step=20)
        await started.wait()
        assert controller._validation_pass_in_flight()

        first_task = controller._validation_task
        controller._start_async_validation(step=40)
        assert controller._validation_task is first_task
        controller.metrics_processor.log.assert_called_once()
        skipped = controller.metrics_processor.log.call_args.kwargs["metrics"]
        assert [metric.key for metric in skipped] == ["validation/skipped"]

        release.set()
        await controller._await_pending_validation()
        assert not controller._validation_pass_in_flight()

    asyncio.run(run())


def test_eval_generator_failure_does_not_end_training() -> None:
    """Validation is observability: an eval-generator failure must disable the
    evaluator, not propagate into the trainer loop."""

    async def run() -> None:
        controller = object.__new__(Controller)
        controller.eval_generator_router = MagicMock()
        controller._eval_rollout_workers = [MagicMock()]

        async def _boom():
            raise RuntimeError("connection closed by peer")

        ok = await controller._guard_eval_generators(_boom(), what="eval pull")

        assert ok is False
        assert controller.eval_generator_router is None
        assert controller._eval_rollout_workers == []

    asyncio.run(run())


def test_training_pull_failure_still_propagates() -> None:
    """The training generators are load-bearing; their failure must not be swallowed."""

    async def run() -> None:
        controller = object.__new__(Controller)
        controller.eval_generator_router = None

        router = MagicMock()

        async def _boom(**kwargs):
            raise RuntimeError("trainer weight sync failed")

        router.pull_model_state_dict = _boom
        controller.generator_router = router

        with pytest.raises(RuntimeError, match="trainer weight sync failed"):
            await controller._pull_generator_weights(policy_version=7)

    asyncio.run(run())


def test_validate_skips_when_the_evaluator_is_disabled() -> None:
    """After a disabling failure, a pass reports a skip instead of raising."""

    async def run() -> None:
        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(
            num_eval_generators=1,
            async_loop=SimpleNamespace(validation=ValidationConfig(num_samples=89)),
        )
        controller.eval_generator_router = None

        metrics = await controller.validate(step=20)

        assert [metric.key for metric in metrics] == ["validation/skipped"]

    asyncio.run(run())

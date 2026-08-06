# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.experiments.rl.examples.tmax import (
    config_registry as tmax_config_registry,
)
from torchtitan.experiments.rl.examples.tmax.config_registry import (
    rl_grpo_qwen3_4b_tmax,
    rl_grpo_qwen3_5_9b_tmax,
)
from torchtitan.experiments.rl.rollout.types import Rollout, RolloutStatus


@pytest.mark.parametrize(
    ("lr_override", "expected_lr"),
    [(None, 1e-6), ("2e-7", 2e-7)],
)
def test_tmax_9b_uses_open_instruct_optimizer_and_mixed_precision(
    monkeypatch: pytest.MonkeyPatch,
    lr_override: str | None,
    expected_lr: float,
) -> None:
    monkeypatch.delenv("SWE_GDN_BI", raising=False)
    if lr_override is None:
        monkeypatch.delenv("SWE_LR", raising=False)
    else:
        monkeypatch.setenv("SWE_LR", lr_override)

    config = rl_grpo_qwen3_5_9b_tmax()

    training = config.trainer.training
    assert training.dtype == "float32"
    assert training.mixed_precision_param == "bfloat16"
    assert training.mixed_precision_reduce == "float32"

    optimizer = config.trainer.optimizer
    assert optimizer.implementation == "fused"
    assert len(optimizer.param_groups) == 1
    param_group = optimizer.param_groups[0]
    assert param_group.optimizer_name == "AdamW"
    assert param_group.optimizer_kwargs == {
        "lr": expected_lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }


def test_tmax_batch_invariant_uses_fp32_master_and_bf16_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWE_LR", raising=False)
    monkeypatch.setenv("SWE_GDN_BI", "1")

    config = rl_grpo_qwen3_5_9b_tmax()

    assert config.trainer.debug.batch_invariant
    training = config.trainer.training
    assert training.dtype == "float32"
    assert training.mixed_precision_param == "bfloat16"
    assert training.mixed_precision_reduce == "float32"


def test_tmax_4b_batch_invariant_uses_fp32_master_and_bf16_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWE_GDN_BI", raising=False)
    monkeypatch.delenv("SWE_LR", raising=False)

    config = rl_grpo_qwen3_4b_tmax()

    assert config.trainer.debug.batch_invariant
    training = config.trainer.training
    assert training.dtype == "float32"
    assert training.mixed_precision_param == "bfloat16"
    assert training.mixed_precision_reduce == "float32"


@pytest.mark.parametrize(
    ("time_budget_override", "expected_time_budget"),
    [(None, 2400), ("1800", 1800)],
)
def test_tmax_time_budget(
    monkeypatch: pytest.MonkeyPatch,
    time_budget_override: str | None,
    expected_time_budget: int,
) -> None:
    if time_budget_override is None:
        monkeypatch.delenv("SWE_TIME_BUDGET_SEC", raising=False)
    else:
        monkeypatch.setenv("SWE_TIME_BUDGET_SEC", time_budget_override)

    config = rl_grpo_qwen3_5_9b_tmax()

    assert config.rollouter.time_budget_sec == expected_time_budget


def test_tmax_include_ids_only_filters_training_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include_path = "/mnt/shared/tmax-medium-difficulty-ids.txt"
    monkeypatch.setattr(tmax_config_registry, "_INCLUDE_IDS", include_path)

    config = rl_grpo_qwen3_5_9b_tmax()

    assert config.rollouter.train_dataset.include_ids_path == include_path
    assert config.rollouter.validation_dataset.include_ids_path == ""


def test_tmax_9b_configures_bs32_spp8_no_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWE_NUM_GROUPS_PER_TRAIN_STEP", "32")
    monkeypatch.setenv("SWE_GROUP_SIZE", "8")
    monkeypatch.setenv("SWE_DROP_ZERO_STD", "0")
    monkeypatch.setenv("SWE_ROLLOUT_CONCURRENCY", "512")
    monkeypatch.setenv("SWE_NUM_ROLLOUT_WORKERS", "8")
    monkeypatch.setenv("SWE_MAX_ACTIVE_GROUPS", "112")
    monkeypatch.setenv("SWE_INITIAL_ACTIVE_GROUPS", "80")

    async_loop = rl_grpo_qwen3_5_9b_tmax().async_loop

    assert async_loop.num_groups_per_train_step == 32
    assert async_loop.group_size == 8
    assert not async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert async_loop.max_active_rollout_groups == 112
    assert async_loop.initial_active_rollout_groups == 80


def test_tmax_keeps_zero_reward_infra_siblings_for_oi_parity() -> None:
    config = rl_grpo_qwen3_5_9b_tmax()
    builder = config.async_loop.training_sample_builder

    assert not builder.drop_groups_with_untrainable_rollouts

    rubric_config = config.rollouter.rubric
    assert rubric_config.error_reward is None
    assert rubric_config.truncation_reward is None

    rollout = Rollout(
        group_id=0,
        rollout_id=0,
        status=RolloutStatus.ERROR,
    )
    output = asyncio.run(rubric_config.build().score_group([rollout], object()))[0]

    assert output.reward == 0.0
    assert output.reward_breakdown == {"RewardTMax": 0.0}


@pytest.mark.parametrize(
    (
        "max_active_override",
        "initial_active_override",
        "rollout_concurrency",
        "expected_max_active",
        "expected_initial_active",
    ),
    [
        (None, None, 512, 40, 32),
        (None, None, 1000, 40, 32),
        (None, None, 1024, 40, 40),
        ("32", None, 512, 32, 32),
        ("40", "24", 1024, 40, 24),
    ],
)
def test_tmax_9b_cold_start_capacity(
    monkeypatch: pytest.MonkeyPatch,
    max_active_override: str | None,
    initial_active_override: str | None,
    rollout_concurrency: int,
    expected_max_active: int,
    expected_initial_active: int,
) -> None:
    monkeypatch.delenv("SWE_GDN_BI", raising=False)
    monkeypatch.delenv("SWE_OFFPOLICY_STEPS", raising=False)
    monkeypatch.delenv("SWE_SELECTION_WINDOW_GROUPS", raising=False)
    monkeypatch.delenv("SWE_MAX_BYPASS_GROUPS", raising=False)
    monkeypatch.delenv("SWE_STRICT_FIFO", raising=False)
    monkeypatch.setenv("SWE_NUM_ROLLOUT_WORKERS", "8")
    monkeypatch.setenv("SWE_ROLLOUT_CONCURRENCY", str(rollout_concurrency))
    for name, value in (
        ("SWE_MAX_ACTIVE_GROUPS", max_active_override),
        ("SWE_INITIAL_ACTIVE_GROUPS", initial_active_override),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    async_loop = rl_grpo_qwen3_5_9b_tmax().async_loop

    assert async_loop.max_offpolicy_steps == 4
    assert async_loop.max_active_rollout_groups == expected_max_active
    assert async_loop.initial_active_rollout_groups == expected_initial_active
    assert async_loop.resolved_max_active_rollout_groups() == expected_max_active
    assert (
        async_loop.resolved_initial_active_rollout_groups() == expected_initial_active
    )


def test_tmax_9b_defaults_to_unbounded_take_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWE_SELECTION_WINDOW_GROUPS", raising=False)
    monkeypatch.delenv("SWE_MAX_BYPASS_GROUPS", raising=False)
    monkeypatch.delenv("SWE_STRICT_FIFO", raising=False)

    group_buffer = rl_grpo_qwen3_5_9b_tmax().async_loop.group_buffer

    assert group_buffer.num_groups_in_selection_window is None
    assert group_buffer.max_bypass_groups is None
    assert not group_buffer.strict_fifo


def test_tmax_9b_rejects_worker_split_without_group_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWE_INITIAL_ACTIVE_GROUPS", raising=False)
    monkeypatch.setenv("SWE_MAX_ACTIVE_GROUPS", "40")
    monkeypatch.setenv("SWE_NUM_ROLLOUT_WORKERS", "15")
    monkeypatch.setenv("SWE_ROLLOUT_CONCURRENCY", "1024")

    with pytest.raises(ValueError, match="cannot keep every trajectory gate supplied"):
        rl_grpo_qwen3_5_9b_tmax()


def test_tmax_9b_configures_sliding_selection_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWE_SELECTION_WINDOW_GROUPS", "12")
    monkeypatch.delenv("SWE_MAX_BYPASS_GROUPS", raising=False)
    monkeypatch.setenv("SWE_STRICT_FIFO", "0")

    group_buffer = rl_grpo_qwen3_5_9b_tmax().async_loop.group_buffer

    assert group_buffer.num_groups_in_selection_window == 12
    assert group_buffer.max_bypass_groups is None
    assert not group_buffer.strict_fifo


@pytest.mark.parametrize(
    ("max_bypass_override", "expected"),
    [("32", 32), ("off", None), ("", None)],
)
def test_tmax_9b_configures_max_bypass_override(
    monkeypatch: pytest.MonkeyPatch,
    max_bypass_override: str,
    expected: int | None,
) -> None:
    monkeypatch.setenv("SWE_SELECTION_WINDOW_GROUPS", "12")
    monkeypatch.setenv("SWE_MAX_BYPASS_GROUPS", max_bypass_override)

    group_buffer = rl_grpo_qwen3_5_9b_tmax().async_loop.group_buffer

    assert group_buffer.max_bypass_groups == expected

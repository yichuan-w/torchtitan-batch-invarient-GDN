# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the Qwen3.5 train/inference-mismatch arms (MATH and Search-R1).

The arms must differ ONLY in how the trainer and generator are aligned
numerically; anything else drifting apart would invalidate the comparison the
blog post draws between them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fla", reason="Qwen3.5 GDN needs flash-linear-attention")

from torchtitan.experiments.rl.examples.dapo_math.config_registry import (  # noqa: E402
    rl_dapo_qwen3_5_35b_a3b_math_8k_bi,
    rl_dapo_qwen3_5_9b_math_8k_bi,
    rl_dapo_qwen3_5_9b_math_8k_unified,
    rl_dapo_qwen3_5_9b_math_8k_vllm_salt,
)
from torchtitan.experiments.rl.examples.search_r1.config_registry import (  # noqa: E402
    rl_grpo_qwen3_5_9b_search_r1_unified,
    rl_grpo_qwen3_5_9b_search_r1_unified_bi,
)


def test_arms_differ_only_in_numerics_alignment():
    native = rl_dapo_qwen3_5_9b_math_8k_vllm_salt()
    unified = rl_dapo_qwen3_5_9b_math_8k_unified()
    bi = rl_dapo_qwen3_5_9b_math_8k_bi()

    # vLLM native: its own model implementation, no batch invariance.
    assert native.generator.backend == "vllm_native"
    assert not native.trainer.debug.batch_invariant

    # Unified: one shared model definition, stock kernels.
    assert unified.generator.backend == "torchtitan_wrapper"
    assert not unified.trainer.debug.batch_invariant
    assert not unified.generator.gdn_trainer_parity

    # BI: the same definition, aligned kernels on both sides.
    assert bi.generator.backend == "torchtitan_wrapper"
    assert bi.trainer.debug.batch_invariant
    assert bi.trainer.debug.deterministic
    assert bi.generator.debug.batch_invariant
    assert bi.generator.gdn_trainer_parity

    # Everything that would otherwise confound the comparison is shared.
    for arm in (unified, bi):
        assert arm.async_loop.batcher.batch.seq_len == (
            native.async_loop.batcher.batch.seq_len
        )
        assert arm.async_loop.max_offpolicy_steps == (
            native.async_loop.max_offpolicy_steps
        )
        assert arm.async_loop.group_size == native.async_loop.group_size
        assert arm.generator.sampling.max_tokens == native.generator.sampling.max_tokens
        assert arm.trainer.parallelism == native.trainer.parallelism
        assert arm.trainer.optimizer == native.trainer.optimizer
        # FP32 master weights + BF16 forward in every arm, so precision of the
        # optimizer state is not what separates them.
        assert arm.trainer.training.dtype == "float32"
        assert arm.trainer.training.mixed_precision_param == "bfloat16"

    # Batch invariance requires sequence parallel off and TP=1 on both sides.
    assert not bi.trainer.parallelism.enable_sequence_parallel
    assert bi.trainer.parallelism.tensor_parallel_degree == 1
    assert bi.generator.parallelism.tensor_parallel_degree == 1


def test_moe_arm_keeps_expert_parallelism_off():
    """EP=1 on both sides isolates the kernels from MoE all-to-all dispatch."""
    moe = rl_dapo_qwen3_5_35b_a3b_math_8k_bi()
    assert moe.trainer.parallelism.expert_parallel_degree == 1
    assert moe.generator.parallelism.expert_parallel_degree == 1
    assert moe.trainer.parallelism.data_parallel_shard_degree == 16
    assert moe.generator.parallelism.data_parallel_degree == 8


def test_search_r1_bi_arm_differs_only_in_numerics_alignment():
    unified = rl_grpo_qwen3_5_9b_search_r1_unified()
    bi = rl_grpo_qwen3_5_9b_search_r1_unified_bi()

    for side in ("trainer", "generator"):
        assert not getattr(unified, side).debug.batch_invariant
        assert getattr(bi, side).debug.batch_invariant
        assert getattr(bi, side).debug.deterministic
    assert bi.generator.gdn_trainer_parity
    assert not unified.generator.gdn_trainer_parity
    # FP32 GDN recurrent state, and sequence parallel off, which BI requires.
    assert bi.generator.mamba_ssm_cache_dtype == "float32"
    assert not bi.trainer.parallelism.enable_sequence_parallel

    # Same model, data, batch, optimizer and sampling in both arms.
    assert bi.generator.backend == unified.generator.backend == "torchtitan_wrapper"
    assert bi.async_loop.group_size == unified.async_loop.group_size
    assert bi.async_loop.max_offpolicy_steps == unified.async_loop.max_offpolicy_steps
    assert bi.async_loop.num_training_steps == unified.async_loop.num_training_steps
    assert bi.generator.sampling == unified.generator.sampling
    assert bi.trainer.optimizer == unified.trainer.optimizer
    assert bi.trainer.loss == unified.trainer.loss
    assert bi.num_rollout_workers == unified.num_rollout_workers


def test_search_r1_bi_arm_resets_the_prefix_cache_on_weight_sync():
    """A reset re-prefills in-flight samples, so none straddles a weight version."""
    bi = rl_grpo_qwen3_5_9b_search_r1_unified_bi()
    assert bi.generator.enable_prefix_caching
    assert not bi.generator.salt_prefix_cache_on_weight_sync
    assert bi.generator.reset_prefix_cache_on_weight_sync
    assert bi.generator.reset_running_requests_on_weight_sync

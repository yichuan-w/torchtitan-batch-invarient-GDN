# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-node Qwen3-4B-Base DAPO-Math recipes."""

from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.dapo_math.data import AIME2025Dataset
from torchtitan.experiments.rl.examples.dapo_math.rollouter import DapoMathRollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import LeastLoadedRoutingStrategy
from torchtitan.models.qwen3 import model_registry


def _qwen3_4b_dapo_math_config(
    *,
    max_response_tokens: int,
    max_total_tokens: int,
    dump_folder: str,
) -> Controller.Config:
    """Build the shared Qwen3-4B DAPO-Math configuration."""
    num_validation_samples = 30
    validation_dataset = AIME2025Dataset.Config(
        num_samples=num_validation_samples,
    )
    return Controller.Config(
        model_spec=model_registry(
            "4B",
            attn_backend="varlen",
            # Compute vocabulary logits in fp32; the rest of the forward uses bf16.
            converters=[LMHeadCastConverter.Config()],
        ),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Base",
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_groups_per_train_step=8,
            group_size=16,
            max_offpolicy_steps=4,
            validation=ValidationConfig(
                num_samples=num_validation_samples,
            ),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=max_total_tokens),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=DapoMathRollouter.Config(
            validation_dataset=validation_dataset,
            token_env=TokenEnv.Config(
                max_rollout_tokens=max_total_tokens,
                max_num_turns=1,
            ),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
        num_generators=6,
        generator_router=InterGeneratorRouter.Config(
            strategy=LeastLoadedRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/response_length/mean",
                "timing/validate",
            ],
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(
                lr=1e-6,
                betas=(0.9, 0.98),
                weight_decay=0.1,
            ),
            # A minimum factor of 1 keeps the learning rate constant.
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=100,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )


def rl_dapo_qwen3_4b_math_8k() -> Controller.Config:
    """Run 8K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=8192,
        max_total_tokens=10240,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_8k",
    )


def rl_dapo_qwen3_4b_math_32k() -> Controller.Config:
    """Run 32K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=32768,
        max_total_tokens=34816,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_32k",
    )


# ---------------------------------------------------------------------------
# Qwen3.5 (Gated DeltaNet) train/inference-mismatch arms
#
# The MATH workload of
# https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/
# The three arms differ ONLY in how hard they make the trainer and the generator
# agree numerically, so their logprob-gap and reward curves are comparable:
#
#   _vllm_salt  vLLM's own model implementation, trainer weights loaded into it
#               (two model definitions). The blog's "vLLM native" arm.
#   _unified    one shared TorchTitan model definition on both sides, stock
#               kernels. The blog's "Titan unified, w/o BI" arm.
#   _bi         the same definition plus batch-invariant kernels: recurrent GDN
#               forward on the trainer, recurrent prefill AND decode on the
#               generator, num_splits=1 attention, batch-invariant GEMM.
#               The blog's "Titan unified, w/ BI" arm.
#
# The off-policy window is a CLI override on top of any arm, e.g.
# ``--async_loop.max_offpolicy_steps 12`` for the sweep of 4 / 12 / 32.
#
# Checkpoints are expected under ``example_checkpoint`` (see
# ``scripts/download_hf_assets.py``); override with ``--hf_assets_path``.
# ---------------------------------------------------------------------------

_EXAMPLE_CKPT_DIR = "torchtitan/experiments/rl/example_checkpoint"


def _qwen3_5_9b_dapo_math(
    *,
    batch_invariant: bool,
    max_response_tokens: int,
    unified_generator: bool = False,
    vllm_native_salt: bool = False,
) -> Controller.Config:
    """Adapt the Qwen3-4B DAPO-Math workload to Qwen3.5-9B (GDN hybrid).

    Topology: 8-way FSDP trainer at TP=1 (FullAC) and one generator replica at
    DP=8, TP=1, so no cross-GPU reduction order enters the comparison.
    """
    import dataclasses

    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.experiments.rl.examples.search_r1.config_registry import (
        _set_max_seq_len,
    )
    from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry

    config_fns = {
        8192: rl_dapo_qwen3_4b_math_8k,
        32768: rl_dapo_qwen3_4b_math_32k,
    }
    try:
        config = config_fns[max_response_tokens]()
    except KeyError:
        raise ValueError(
            f"unsupported Qwen3.5 DAPO max_response_tokens={max_response_tokens}; "
            f"expected one of {sorted(config_fns)}"
        ) from None
    config.model_spec = qwen3_5_model_registry(
        "9B",
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    _set_max_seq_len(config.model_spec, config.async_loop.batcher.batch.seq_len)
    config.hf_assets_path = f"{_EXAMPLE_CKPT_DIR}/Qwen3.5-9B"
    config.dump_folder = (
        f"outputs/rl/qwen3_5_9b_dapo_math_{max_response_tokens // 1024}k"
    )
    # GDN is not torch.compile-clean; run eager.
    config.compile = CompileConfig(enable=False, backend="aot_eager")
    config.renderer = RendererConfig(name="qwen3.5", enable_thinking=True)
    config.num_generators = 1
    # The 8K jobs run the async rollouter in the controller process so the
    # trainer can share that host. Spawning the 32-worker pool alongside eight
    # FSDP ranks can exceed Monarch's 30-second proc-mesh startup deadline.
    # The 32K jobs retain the separate controller host and worker pool.
    config.num_rollout_workers = 0 if max_response_tokens == 8192 else 32
    config.async_loop = dataclasses.replace(
        config.async_loop,
        validation=dataclasses.replace(config.async_loop.validation, interval=10),
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        ac_config=FullAC.Config(),
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        loss=dataclasses.replace(config.trainer.loss, num_chunks=16),
    )
    config.generator = dataclasses.replace(
        config.generator,
        backend="vllm_native",
        gpu_memory_limit=0.6,
        vllm_additional_config={"gdn_prefill_backend": "triton"},
        cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
        enable_prefix_caching=True,
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            data_parallel_degree=8,
            tensor_parallel_degree=1,
        ),
    )

    if batch_invariant or unified_generator:
        trainer_debug = dataclasses.replace(
            config.trainer.debug,
            seed=42,
            batch_invariant=batch_invariant,
            deterministic=batch_invariant,
        )
        generator_debug = dataclasses.replace(
            config.generator.debug,
            seed=42,
            batch_invariant=batch_invariant,
            deterministic=batch_invariant,
        )
        config.trainer = dataclasses.replace(
            config.trainer,
            debug=trainer_debug,
            # Keep FP32 master parameters and optimizer states so the 1e-6 AdamW
            # updates are not rounded away. FSDP casts forward parameters to BF16,
            # preserving the trainer/generator compute contract in both controls.
            training=dataclasses.replace(
                config.trainer.training,
                dtype="float32",
                mixed_precision_param="bfloat16",
                mixed_precision_reduce="float32",
            ),
        )
        config.generator = dataclasses.replace(
            config.generator,
            backend="torchtitan_wrapper",
            vllm_additional_config={},
            gdn_trainer_parity=batch_invariant,
            mamba_ssm_cache_dtype="float32",
            debug=generator_debug,
            enable_prefix_caching=True,
            cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
            salt_prefix_cache_on_weight_sync=True,
            reset_prefix_cache_on_weight_sync=False,
            reset_running_requests_on_weight_sync=False,
        )
    elif vllm_native_salt:
        config.trainer = dataclasses.replace(
            config.trainer,
            debug=dataclasses.replace(
                config.trainer.debug,
                seed=42,
                batch_invariant=False,
                deterministic=False,
            ),
            training=dataclasses.replace(
                config.trainer.training,
                dtype="float32",
                mixed_precision_param="bfloat16",
                mixed_precision_reduce="float32",
            ),
        )
        config.generator = dataclasses.replace(
            config.generator,
            debug=dataclasses.replace(
                config.generator.debug,
                seed=42,
                batch_invariant=False,
                deterministic=False,
            ),
            mamba_ssm_cache_dtype="float32",
            enable_prefix_caching=True,
            salt_prefix_cache_on_weight_sync=True,
            reset_prefix_cache_on_weight_sync=False,
            reset_running_requests_on_weight_sync=False,
        )
    return config


def _qwen3_5_35b_a3b_dapo_math(
    *,
    batch_invariant: bool,
    unified_generator: bool = False,
    vllm_native_salt: bool = False,
    smoke: bool = False,
) -> Controller.Config:
    """The same three arms on the MoE Qwen3.5-35B-A3B.

    ``EP = 1`` on both sides is deliberate: with expert parallelism off there is
    no MoE all-to-all, so the comparison isolates the kernels from cross-GPU
    dispatch the same way ``TP = 1`` does for the dense runs. The extra kernel
    MoE needs is a batch-invariant ``bmm`` for the router gate (the generator
    hands it a 3-D batch, the trainer a 2-D one), patched in by
    ``experiments/rl/batch_invariance.py``.

    ``smoke=True`` is a one-step, 1024-token version for checking the path on a
    small allocation.
    """
    import dataclasses

    from torchtitan.experiments.rl.examples.search_r1.config_registry import (
        _set_max_seq_len,
    )
    from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry

    config = _qwen3_5_9b_dapo_math(
        batch_invariant=batch_invariant,
        max_response_tokens=8192,
        unified_generator=unified_generator,
        vllm_native_salt=vllm_native_salt,
    )
    config.model_spec = qwen3_5_model_registry(
        "35B-A3B",
        attn_backend="varlen",
        moe_comm_backend="standard",
        converters=[LMHeadCastConverter.Config()],
    )
    _set_max_seq_len(config.model_spec, config.async_loop.batcher.batch.seq_len)
    config.hf_assets_path = f"{_EXAMPLE_CKPT_DIR}/Qwen3.5-35B-A3B-Base"
    config.dump_folder = "outputs/rl/qwen3_5_35b_a3b_dapo_math_8k"
    config.num_generators = 4
    config.num_rollout_workers = 0
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=16,
            tensor_parallel_degree=1,
            expert_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        # A complete BF16 rank is about 65.4 GiB before runtime state. Keep the
        # comparison free of EP while retaining headroom for the 640 MiB FP32
        # MoE combine temporary used by the 10,240-token vLLM warmup.
        gpu_memory_limit=0.95,
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            data_parallel_degree=8,
            tensor_parallel_degree=1,
            expert_parallel_degree=1,
        ),
    )

    if smoke:
        smoke_max_total_tokens = 1024
        config.async_loop = dataclasses.replace(
            config.async_loop,
            num_training_steps=1,
            num_groups_per_train_step=1,
            group_size=16,
            max_offpolicy_steps=0,
            training_sample_builder=dataclasses.replace(
                config.async_loop.training_sample_builder,
                drop_zero_std_reward_groups=False,
            ),
            batcher=dataclasses.replace(
                config.async_loop.batcher,
                batch=dataclasses.replace(
                    config.async_loop.batcher.batch,
                    seq_len=smoke_max_total_tokens,
                ),
            ),
            validation=dataclasses.replace(
                config.async_loop.validation,
                num_samples=1,
                interval=1,
            ),
        )
        config.rollouter = dataclasses.replace(
            config.rollouter,
            validation_dataset=dataclasses.replace(
                config.rollouter.validation_dataset,
                num_samples=1,
            ),
            token_env=dataclasses.replace(
                config.rollouter.token_env,
                max_rollout_tokens=smoke_max_total_tokens,
            ),
        )
        config.trainer = dataclasses.replace(
            config.trainer,
            checkpoint=dataclasses.replace(
                config.trainer.checkpoint,
                load_only=True,
            ),
        )
        config.generator = dataclasses.replace(
            config.generator,
            sampling=dataclasses.replace(
                config.generator.sampling,
                max_tokens=256,
            ),
            cudagraph=VLLMCudagraphConfig(enable=False),
        )
        _set_max_seq_len(config.model_spec, smoke_max_total_tokens)

    return config


def rl_dapo_qwen3_5_9b_math_8k_vllm_salt() -> Controller.Config:
    """Qwen3.5-9B 8K, vLLM-native generator (two model definitions)."""
    return _qwen3_5_9b_dapo_math(
        batch_invariant=False,
        max_response_tokens=8192,
        vllm_native_salt=True,
    )


def rl_dapo_qwen3_5_9b_math_8k_unified() -> Controller.Config:
    """Qwen3.5-9B 8K, one shared model definition, stock kernels (no BI)."""
    return _qwen3_5_9b_dapo_math(
        batch_invariant=False,
        max_response_tokens=8192,
        unified_generator=True,
    )


def rl_dapo_qwen3_5_9b_math_8k_bi() -> Controller.Config:
    """Qwen3.5-9B 8K, shared definition + batch-invariant kernels (parity arm)."""
    return _qwen3_5_9b_dapo_math(batch_invariant=True, max_response_tokens=8192)


def rl_dapo_qwen3_5_9b_math_8k() -> Controller.Config:
    """Qwen3.5-9B 8K baseline: vLLM-native generator, BF16 master weights."""
    return _qwen3_5_9b_dapo_math(batch_invariant=False, max_response_tokens=8192)


def rl_dapo_qwen3_5_9b_math_32k() -> Controller.Config:
    """Qwen3.5-9B 32K baseline: vLLM-native generator, BF16 master weights."""
    return _qwen3_5_9b_dapo_math(batch_invariant=False, max_response_tokens=32768)


def rl_dapo_qwen3_5_9b_math_32k_bi() -> Controller.Config:
    """Qwen3.5-9B 32K, shared definition + batch-invariant kernels."""
    return _qwen3_5_9b_dapo_math(batch_invariant=True, max_response_tokens=32768)


def rl_dapo_qwen3_5_35b_a3b_math_8k_vllm_salt() -> Controller.Config:
    """Qwen3.5-35B-A3B (MoE) 8K, vLLM-native generator."""
    return _qwen3_5_35b_a3b_dapo_math(batch_invariant=False, vllm_native_salt=True)


def rl_dapo_qwen3_5_35b_a3b_math_8k_unified() -> Controller.Config:
    """Qwen3.5-35B-A3B (MoE) 8K, one shared model definition, no BI."""
    return _qwen3_5_35b_a3b_dapo_math(batch_invariant=False, unified_generator=True)


def rl_dapo_qwen3_5_35b_a3b_math_8k_bi() -> Controller.Config:
    """Qwen3.5-35B-A3B (MoE) 8K, shared definition + batch-invariant kernels."""
    return _qwen3_5_35b_a3b_dapo_math(batch_invariant=True)


def rl_dapo_qwen3_5_35b_a3b_math_smoke_unified() -> Controller.Config:
    """One-step 35B-A3B smoke on the unified generator."""
    return _qwen3_5_35b_a3b_dapo_math(
        batch_invariant=False, unified_generator=True, smoke=True
    )


def rl_dapo_qwen3_5_35b_a3b_math_smoke_bi() -> Controller.Config:
    """One-step 35B-A3B smoke on the batch-invariant unified generator."""
    return _qwen3_5_35b_a3b_dapo_math(batch_invariant=True, smoke=True)


def rl_dapo_qwen3_5_35b_a3b_math_smoke_vllm_salt() -> Controller.Config:
    """One-step 35B-A3B smoke on the vLLM-native generator."""
    return _qwen3_5_35b_a3b_dapo_math(
        batch_invariant=False, vllm_native_salt=True, smoke=True
    )

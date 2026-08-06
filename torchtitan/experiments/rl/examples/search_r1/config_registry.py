# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the Search-R1 example.

These set the full Search-R1 recipe entirely from the example's config — the core
defaults are unchanged, so every other config keeps vanilla GRPO. ``ConfigManager``
discovers these directly from the example module::

    --module search_r1 \\
        --config rl_grpo_qwen3_1_7b_search_r1
"""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
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
from torchtitan.experiments.rl.examples.search_r1.rollouter import SearchR1Rollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry
from torchtitan.protocols.model_spec import ModelSpec


def rl_grpo_qwen3_1_7b_search_r1() -> Controller.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-1.7B.

    Runs on 8 GPUs: 4 generator (TP=4) + 1 trainer (TP=1), with a dense retrieval
    server on the spare GPUs. Requires a running retrieval server and the QA parquet
    data; see ``README.md``.
    """
    return Controller.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_groups_per_train_step=8,
            group_size=8,
            validation=ValidationConfig(num_samples=500),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=4096),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SearchR1Rollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,  # first run loads HF; restarts resume from DCP
                # Mid-run checkpoints so a preempted run resumes; full last save
                # (not model-only) keeps it resumable; keep_latest_k caps disk.
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=DAPOLoss.Config(
                ratio_clip_low=0.2,
                ratio_clip_high=0.28,
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            # cudagraph on: decode-only graphs (FULL_DECODE_ONLY) are safe at this
            # config's large batch; plain full graphs corrupted here before. See #3668.
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
            ),
        ),
    )


def rl_grpo_qwen3_8b_search_r1() -> Controller.Config:
    """GRPO Search-R1 for Qwen3-8B — same recipe as the 1.7B config.

    Only the model and GPU split differ. 8 GPUs: 2 generator (TP=2) + 4 trainer
    (TP=4) + retriever on the spare GPUs. The fp32 trainer needs TP=4 to avoid OOM.
    """
    # TODO: use mixed precision (fp32 master + bf16 compute) via FSDP + activation
    # checkpointing, which is more memory-efficient and could keep the split generator-heavy.
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=4
        ),
    )
    # 0.6 (vs the 0.9 default) reserves room for the weight-sync memory spike, which
    # OOMs the 8B generator otherwise.
    # TODO(@meetv18): the spike is likely GPU-Direct weight transfer being on by default;
    # make the transfer device configurable (CPU default) so this cap can be raised.
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=2
        ),
    )
    return config


def _set_max_seq_len(model_spec: ModelSpec, max_seq_len: int) -> None:
    """Raise/lower the model's context length by setting every attention layer's
    RoPE ``max_seq_len`` (``ModelSpec.model.max_seq_len`` is a read-only property
    derived from it). Hybrid-safe: linear-attention (GDN) layers have no ``rope``
    and are skipped."""
    for layer in model_spec.model.layers:
        rope = getattr(getattr(layer, "attention", None), "rope", None)
        if rope is not None:
            rope.max_seq_len = max_seq_len


def rl_grpo_qwen3_5_9b_search_r1() -> Controller.Config:
    """GRPO Search-R1 for Qwen3.5-9B (Gated DeltaNet hybrid) -- linear-attn check.

    The dense 1.7B/8B Search-R1 recipe verbatim (clean held-out NQ exact-match eval),
    swapping in the Qwen3.5-9B GDN model to validate the linear-attention
    (GatedDeltaNet) implementation end to end. The trainer runs the FLA chunked GDN
    forward/backward; the generator runs vLLM-native GDN (recurrent decode + chunked
    prefill, the correct GDN decode path the torchtitan wrapper lacks), synced
    torchtitan -> HF by the model's state_dict_adapter. A rising EM curve (matching
    the dense curve) confirms the GDN forward + backward + weight sync are correct;
    a flat/falling curve points at the linear-attention implementation.

    Topology (multi-host): 1 controller host (dense-retrieval server) + 1 trainer host
    (FSDP-8) + N generator hosts (vLLM-native GDN, DP-8 x TP-1). Only the model and
    the GDN toolchain differ from the dense config; the RL recipe (8 groups x 8, temp
    1.0, std-normalized advantage, DAPO clip-higher, 500 steps, NQ-500 eval) is
    unchanged so the two EM curves are directly comparable. Needs the CUDA 13 GDN
    toolchain (``SWE_GDN=1`` in run.sh switches CUDA_HOME to the pip cu13 ptxas).
    """
    config = rl_grpo_qwen3_1_7b_search_r1()
    # Qwen3.5-9B GDN hybrid + fp32 lm_head cast (RL logprob math needs fp32 logits),
    # mirroring the swe_r2e/tmax GDN recipes.
    config.model_spec = qwen3_5_model_registry(
        "9B", attn_backend="varlen", converters=[LMHeadCastConverter.Config()]
    )
    # GDN's default context is 262144; cap RoPE (== the generator's vLLM
    # max_model_len) to the Search-R1 sequence length so vLLM does not allocate a
    # 256k-token KV cache. 4096 matches the batcher/dense-qwen3 context.
    _set_max_seq_len(config.model_spec, 4096)
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3.5-9B"
    # Re-run the held-out NQ-500 eval every 25 steps (the base default interval=0 runs
    # it only once pre-training). This is the whole point of the run: watch EM grow to
    # confirm the GDN gradients are correct. ~34s/pass over 500 steps = ~11min total.
    config.async_loop = dataclasses.replace(
        config.async_loop,
        validation=dataclasses.replace(config.async_loop.validation, interval=25),
    )
    # GDN is not torch.compile-clean; run eager (matches the qwen3_5 recipes).
    config.compile = CompileConfig(enable=False, backend="aot_eager")
    # qwen3.5 chat template; thinking off gave the best Search-R1 EM in the dense
    # ablation (the renderer drops this knob if the qwen3.5 template ignores it).
    config.renderer = RendererConfig(name="qwen3.5", enable_thinking=False)
    # FSDP-8 trainer (9B on one 8-GPU host). FullAC recomputes activations to fit;
    # the chunked loss bounds the fp32 lm_head logits over the 248320-token vocab.
    config.trainer = dataclasses.replace(
        config.trainer,
        ac_config=FullAC.Config(),
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
        loss=ChunkedLossWrapper.Config(
            num_chunks=8,
            loss_fn=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
        ),
    )
    # vLLM-native GDN generator. DP-8 x TP-1 = 8 single-GPU engines per host (the 9B
    # ~18GB fits one GPU), avoiding the TP>1 custom-all-reduce path. gdn_prefill
    # triton kernel + GDN-safe FULL_DECODE_ONLY decode graphs (#3668) + forced prefix
    # caching (GDN defaults it OFF, so multi-turn rollouts re-prefill every turn).
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
    return config


def rl_grpo_qwen3_5_9b_search_r1_unified() -> Controller.Config:
    """UNIFIED GDN generation variant of :func:`rl_grpo_qwen3_5_9b_search_r1`.

    Instead of ``vllm_native`` (vLLM's own GDN model), generation runs
    TorchTitan's OWN model for every layer (``backend="torchtitan_wrapper"``);
    the GDN layers become vLLM-cache aware via the injected
    ``VLLMGatedDeltaNetCore`` -- params + fla math stay TorchTitan's, only the
    paged conv/ssm cache is vLLM's. Everything else (data, reward, trainer, DP-8)
    is identical, so its EM curve is directly comparable to the native run.

    cudagraph OFF (the fla chunk decode is not cudagraph-capturable) and prefix
    caching OFF (unified GDN mamba prefix caching not yet validated; add it once
    the base unified path is confirmed on the multi-host cluster).
    """
    config = rl_grpo_qwen3_5_9b_search_r1()
    config.generator = dataclasses.replace(
        config.generator,
        backend="torchtitan_wrapper",
        cudagraph=VLLMCudagraphConfig(enable=False),
        enable_prefix_caching=False,
    )
    return config


def rl_grpo_qwen3_5_9b_search_r1_unified_prefix() -> Controller.Config:
    """Unified GDN + prefix caching ON (diagnostic).

    Same as :func:`rl_grpo_qwen3_5_9b_search_r1_unified` but
    ``enable_prefix_caching=True`` (align-mode mamba prefix caching via the
    unified core's paged conv/ssm state). Multi-turn Search-R1 grows the prompt
    each turn, so cache hits exercise the GDN recurrent-state restore path. This
    is the discriminating run: unified+prefix-OFF is clean and native+prefix-ON
    is clean, so if THIS regresses (EM drop / malformed generations) the custom
    core's prefix-cache state handling is the culprit.
    """
    config = rl_grpo_qwen3_5_9b_search_r1_unified()
    config.generator = dataclasses.replace(
        config.generator,
        enable_prefix_caching=True,
    )
    return config


def rl_grpo_qwen3_5_9b_search_r1_unified_bi() -> Controller.Config:
    """Batch-invariant arm of the Search-R1 train/inference-mismatch comparison.

    The multi-turn workload of
    https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/

    :func:`rl_grpo_qwen3_5_9b_search_r1_unified` with the trainer and generator
    aligned op-for-op, and nothing else changed, so the two EM curves are
    comparable:

    - ``batch_invariant`` + ``deterministic`` on BOTH sides: batch-invariant
      GEMM / log_softmax / mean, ``num_splits=1`` attention, deterministic
      collectives.
    - ``gdn_trainer_parity``: the generator runs the trainer's fla recurrence
      for prefill AND decode, so the GDN layers agree bitwise rather than
      chunked-vs-recurrent.
    - FP32 GDN state cache, and a fixed seed on both sides.
    - Sequence parallel off, which batch invariance requires.

    Prefix caching stays on under the RESET policy (not the salt policy the tmax
    recipe uses): a reset re-prefills in-flight samples under the new weights, so
    no sample straddles a weight version.

    The runs behind the post also raised ``num_rollout_workers`` to 32 to keep
    the multi-turn rollouts off the controller GIL. That is a throughput knob,
    not a numerics one, so it is left at the shared default here.
    """
    config = rl_grpo_qwen3_5_9b_search_r1_unified()
    bi_debug = dict(batch_invariant=True, deterministic=True, seed=42)
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=dataclasses.replace(config.trainer.debug, **bi_debug),
        parallelism=dataclasses.replace(
            config.trainer.parallelism, enable_sequence_parallel=False
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        debug=dataclasses.replace(config.generator.debug, **bi_debug),
        gdn_trainer_parity=True,
        mamba_ssm_cache_dtype="float32",
        cache_cast_linear_weights=True,
        enable_prefix_caching=True,
        cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
        salt_prefix_cache_on_weight_sync=False,
        reset_prefix_cache_on_weight_sync=True,
        reset_running_requests_on_weight_sync=True,
    )
    return config

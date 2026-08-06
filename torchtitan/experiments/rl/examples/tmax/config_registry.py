# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the tmax terminal-agent (host_loop) example.

``ConfigManager`` discovers these from the fully-qualified example module path::

    python -m torchtitan.experiments.rl.train \
        --module torchtitan.experiments.rl.examples.tmax \
        --config rl_grpo_qwen3_5_27b_tmax \
        --hf_assets_path <path/to/Qwen3.6-27B>

(The short ``--module tmax`` form additionally requires ``tmax`` in
``torchtitan/experiments/__init__.py::_supported_experiments``, a core file this
example deliberately does not modify. A multi-host launcher supplies its own ``--module``.)

The 27B tmax config clones the Qwen3.6-27B SWE-R2E recipe
(``rl_grpo_qwen3_5_27b_swe_r2e``) verbatim. The 9B recipe restores standard
mixed precision (fp32 master parameters, bf16 compute, fp32 AdamW states) and
the optimizer settings from open-instruct's TMax recipe. Both swap the rollouter
to ``TMaxRollouter`` + ``TMaxDataset``. The tmax JSONL path comes from
``SWE_PROMPT_DATA`` (set by the launcher's ``PROMPT_DATA``), matching the
swe_r2e convention.
"""

from __future__ import annotations

import dataclasses
import os

from torchtitan.components.optimizer import default_adamw, OptimizersContainer

from torchtitan.config import DebugConfig

from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.rl.actors.generator import VLLMCudagraphConfig
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.controller import (
    _split_rollout_concurrency,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.eval_trace_recorder import ValidationTraceRecorder
from torchtitan.experiments.rl.examples.swe_r2e.config_registry import (
    _CKPT_DIR,
    _qwen3_rl_model_registry,
    _set_max_seq_len,
    rl_grpo_qwen3_5_27b_swe_r2e as _swe_27b,
    rl_grpo_qwen3_5_9b_swe_r2e as _swe_9b,
)
from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset
from torchtitan.experiments.rl.examples.tmax.rollouter import TMaxRollouter
from torchtitan.experiments.rl.losses import DPPOLoss, GRPOLoss

# tmax JSONL path, supplied by the launcher (PROMPT_DATA -> SWE_PROMPT_DATA).
# Empty by default; TMaxDataset raises a clear error if it is not set.
_DEFAULT_DATA = os.environ.get("SWE_PROMPT_DATA", "")

# Optional train-only instance-ID whitelist. Validation deliberately remains on
# the original holdout split so curriculum selection cannot contaminate eval.
_INCLUDE_IDS = os.environ.get("SWE_INCLUDE_PROMPTS", "")

# Optional zero-std skip source (SWE_ZERO_STD_DIR output from a prior run): every
# instance_id in it is dropped at dataset load so all-pass / all-fail prompts (no
# learning signal) are not sampled again. Empty = keep all rows.
_SKIP_IDS = os.environ.get("SWE_SKIP_PROMPTS", "")

# Terminal-Bench 2.0 eval (rl_grpo_qwen3_5_9b_tmax_tb2_eval): the TB-2.0 JSONL
# (prepare_tb2_data.py output, tmax schema) and the trained DCP checkpoint dir to
# score. Empty by default; the eval config falls back to _DEFAULT_DATA / base HF
# weights if unset. TB-2.0 ships exactly 89 tasks.
_TB2_DATA = os.environ.get("SWE_TB2_DATA", "")
_TB2_CKPT = os.environ.get("SWE_TB2_CKPT", "")
_TB2_NUM_TASKS = 89

# Inline TB-2.0 validation (rl_grpo_qwen3_5_9b_tmax): point the TRAINING recipe's
# validation_dataset at the TB-2.0 JSONL so the periodic pass reports the real
# benchmark instead of a tmax holdout slice. Empty = keep the holdout split.
_TB2_VAL_DATA = os.environ.get("SWE_TB2_VAL_DATA", "")
# Sampling for the TB-2.0 pass. Matches the Harbor Vanillux2Agent defaults the
# published avg@k numbers were produced with (temperature 0.7, top_p 0.95, k=5),
# so the inline curve is comparable to the standalone TB-2.0 eval job.
_TB2_VAL_TEMPERATURE = float(os.environ.get("SWE_TB2_VAL_TEMPERATURE", "0.7"))
_TB2_VAL_TOP_P = float(os.environ.get("SWE_TB2_VAL_TOP_P", "0.95"))
_TB2_VAL_K = int(os.environ.get("SWE_TB2_VAL_K", "5"))

# Full TMax-9B recipe context (open-instruct qwen35_9b.sh: response_length 65536)
# and per-turn generation cap (per_turn_max_tokens 16384). The context is the
# generator's vLLM max_model_len AND the trainer batcher's packing width: both are
# raised together (the controller mirrors the batcher width into the trainer
# seq_len), or a full episode is truncated by vLLM / dropped during packing.
_TMAX_9B_CONTEXT = 65536
_TMAX_9B_PER_TURN_TOKENS = 16384
# Held-out prompts per periodic validation pass (greedy, n=1). Runs concurrently, so its
# wall time is ~one rollout regardless of count; 32 gives a stable enough solve-rate.
_TMAX_9B_VAL_SAMPLES = 32
# Reserve the last N rows of the JSONL as a held-out validation slice, disjoint from
# training, so periodic validation measures generalization (not training-set recall).
# Must be >= _TMAX_9B_VAL_SAMPLES so a validation pass can draw distinct held-out tasks.
_TMAX_9B_HOLDOUT_N = 64


def _tmax_rollouter() -> TMaxRollouter.Config:
    """Train/validation datasets for the tmax rollouter (rubric + env defaults live
    on the rollouter Config). Train and validation read the same JSONL but disjoint
    slices via holdout_n (last N rows = validation)."""
    return TMaxRollouter.Config(
        train_dataset=TMaxDataset.Config(
            data_path=_DEFAULT_DATA,
            seed=42,
            # SWE_DISABLE_SHUFFLE=1 -> take training rows in file order (0,1,2,...)
            # for deterministic per-rollout inspection / open-instruct cross-check.
            shuffle=(os.environ.get("SWE_DISABLE_SHUFFLE", "0") != "1"),
            holdout_n=_TMAX_9B_HOLDOUT_N,
            split="train",
            include_ids_path=_INCLUDE_IDS,
            skip_ids_path=_SKIP_IDS,
        ),
        # SWE_TB2_VAL_DATA swaps the held-out tmax slice for the whole TB-2.0 task
        # set (holdout_n=0 -> the file IS the validation split), so the periodic
        # pass reports the benchmark. Unset keeps the tmax holdout.
        validation_dataset=(
            TMaxDataset.Config(
                data_path=_TB2_VAL_DATA, seed=99, shuffle=False, holdout_n=0
            )
            if _TB2_VAL_DATA
            else TMaxDataset.Config(
                data_path=_DEFAULT_DATA,
                seed=99,
                shuffle=False,
                holdout_n=_TMAX_9B_HOLDOUT_N,
                split="validation",
                skip_ids_path=_SKIP_IDS,
            )
        ),
        # Run knobs resolved from the launcher env ONCE, into config fields so they
        # land in the W&B run config (per-run differences are visible). Same env
        # names + defaults as before; the RolloutWorker pool splits
        # rollout_concurrency across workers.
        rollout_concurrency=int(os.environ.get("SWE_ROLLOUT_CONCURRENCY", "16")),
        time_budget_sec=int(os.environ.get("SWE_TIME_BUDGET_SEC", "2400")),
        eval_timeout_sec=int(os.environ.get("TMAX_EVAL_TIMEOUT_SEC", "600")),
        max_context_tokens=int(os.environ.get("SWE_MAX_CONTEXT_LEN", "32768")),
    )


def _tmax_9b_validation() -> ValidationConfig:
    """Periodic held-out eval for the 9B recipe.

    Two modes, selected by ``SWE_TB2_VAL_DATA``:

    - unset: greedy pass@1 over 32 held-out tmax prompts (the historical behavior).
      The trained-batch reward is pinned near ~0.5 by drop_zero_std and is NOT a
      learning signal, so this is the real solve-rate curve.
    - set: the full Terminal-Bench 2.0 task set at the Harbor Vanillux2Agent
      sampling defaults (temperature 0.7, top_p 0.95, k=5), which makes the inline
      ``validation/reward/mean`` directly comparable to the published avg@5 and
      ``validation/pass_at_k`` to pass@5.

    With ``SWE_NUM_EVAL_GENERATORS`` the pass runs on its own generator hosts and
    ``run_async`` lets training continue through it. The eval generators pull only
    on the steps they score, so a background pass measures a frozen policy version.
    """
    if _TB2_VAL_DATA:
        num_samples = int(os.environ.get("SWE_VAL_SAMPLES", _TB2_NUM_TASKS))
        group_size, temperature, top_p = (
            _TB2_VAL_K,
            _TB2_VAL_TEMPERATURE,
            _TB2_VAL_TOP_P,
        )
    else:
        # SWE_VAL_SAMPLES=0 skips the pre/periodic held-out validation entirely
        # (e.g. a pure step-time / speedup run); defaults to the paper's 32.
        num_samples = int(os.environ.get("SWE_VAL_SAMPLES", _TMAX_9B_VAL_SAMPLES))
        group_size, temperature, top_p = 1, 0.0, 1.0
    return ValidationConfig(
        num_samples=num_samples,
        interval=int(os.environ.get("SWE_VAL_INTERVAL", "20")),
        group_size=group_size,
        temperature=temperature,
        top_p=top_p,
        # Async needs somewhere else to run; without eval generators it would just
        # contend with rollout collection for the training ones.
        run_async=int(os.environ.get("SWE_NUM_EVAL_GENERATORS", "0")) > 0,
        trace=ValidationTraceRecorder.Config(
            enable=os.environ.get("SWE_VAL_TRACES", "1") == "1"
        ),
    )


def _tmax_recipe_loss(loss):
    """Apply the tmax recipe's DEFAULT loss to a base loss config.

    The recipe (open-instruct qwen35_9b.sh loss_fn dppo) is DPPO: UNCLIPPED -A*ratio
    + a TV divergence trust-region mask (delta=0.1) that drops the loss on tokens
    pushed further off-policy past the divergence ball (the mask replaces the PPO
    clip -- faithful to open-instruct, no ratio clip). SWE_LOSS=dapo reverts to the
    swe base's DAPO clip-higher for a clean A/B. Only loss_fn is swapped; other loss
    fields (e.g. num_chunks) are preserved.
    """
    _which = os.environ.get("SWE_LOSS", "dppo").lower()
    if _which == "dapo":
        return loss
    if _which == "grpo":
        # Standard GRPO clipped surrogate (swaps the DPPO trust-region for the PPO
        # clip). Only loss_fn swapped; num_chunks etc. preserved.
        return dataclasses.replace(loss, loss_fn=GRPOLoss.Config())
    return dataclasses.replace(
        loss,
        loss_fn=DPPOLoss.Config(
            divergence_threshold=float(
                os.environ.get("SWE_DPPO_DIVERGENCE_THRESHOLD", "0.1")
            ),
            divergence_type="tv",
            # Truncated-IS ratio cap (0 = disabled/recipe-faithful). SWE_DPPO_RATIO_CAP=2
            # clamps the surrogate ratio so a residual GDN gen/train logprob-mismatch
            # tail cannot spike the gradient (our logdiff max ~2 vs open-instruct ~0.5).
            ratio_cap=float(os.environ.get("SWE_DPPO_RATIO_CAP", "0")),
        ),
    )


def _tmax_9b_adamw(lr: float = 1e-6) -> OptimizersContainer.Config:
    """Build the AdamW config used by open-instruct's TMax-9B recipe.

    With the 9B trainer's fp32 master parameters, ordinary fused AdamW keeps its
    optimizer states in fp32. ``fused_opt_states_bf16`` must not be used here:
    it would intentionally quantize the moment states and break recipe parity.
    Full checkpoints from the former bf16-state recipe are not compatible; start
    this recipe from a fresh dump folder or a model-only checkpoint.
    """
    optimizer = default_adamw(
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    optimizer.implementation = "fused"
    return optimizer


def rl_grpo_qwen3_5_27b_tmax() -> Controller.Config:
    """Qwen3.6-27B (Gated DeltaNet hybrid) tmax terminal-agent on a single 8-GPU node.

    Same recipe as ``rl_grpo_qwen3_5_27b_swe_r2e`` (trainer FSDP-8 + vLLM-native GDN
    generator TP-4, bf16 master/Adam + FullAC + chunked DAPO loss, max_tokens=8192,
    off-policy 2, drop_zero_std False, 8 groups x group_size 8, host_loop agent)
    with the rollouter swapped to ``TMaxRollouter`` (runs the agent as root, grades
    with ``bash /tests/test.sh`` -> reward.txt in the agent's own sandbox).
    """
    config = _swe_27b()
    config.rollouter = _tmax_rollouter()
    config.trainer = dataclasses.replace(
        config.trainer, loss=_tmax_recipe_loss(config.trainer.loss)
    )
    return config


def rl_grpo_qwen3_5_9b_tmax() -> Controller.Config:
    """Qwen3.5-9B (Gated DeltaNet hybrid, text-only) AI2 tmax terminal-agent recipe.

    Base = ``rl_grpo_qwen3_5_9b_swe_r2e`` (9B GDN, generator DP-8 x TP-1),
    rollouter swapped to ``TMaxRollouter``. Matches the paper's open-instruct run
    (``scripts/tmax/RL/qwen35_9b.sh``): ``group_size=32``
    (num_samples_per_prompt_rollout), off-policy 4 (async_steps), per-turn 16384,
    full 65536 context (response_length), and ``drop_zero_std_reward_groups=True``
    (``filter_zero_std_samples``) -- terminal tasks are sparse binary, so keeping
    all-fail groups would zero out the gradient. Temperature 1.0, constant LR,
    and beta 0 are inherited from the swe base. When 32 groups leave queued
    siblings behind every worker gate (including concurrency 512 and 1000), it
    uses the paper implementation's ``async_steps * num_groups = 32`` prompt-group
    start. At concurrency 1024 it starts all 40 groups so every worker has queued
    siblings behind its 128 active slots instead of draining on slow tails.
    The trainer restores standard
    mixed precision (fp32 master parameters, bf16 FSDP compute/reduce in fp32)
    and open-instruct's fused AdamW settings: lr 1e-6, betas (0.9, 0.999),
    eps 1e-8, and no weight decay.
    ``num_groups_per_train_step=8`` matches the target run's
    ``num_unique_prompts_rollout``.

    Two knobs must move together with the context: the batcher packing width
    (``seq_len``) and the model RoPE / vLLM max_model_len, both to 65536. The loss
    is re-chunked to 32 chunks (from 16) so the per-chunk fp32 logits stay in the
    validated ~1 GiB envelope at the 4x longer sequence. The full end-to-end active
    ceiling is (off+1) x num_groups x group_size = 5 x 8 x 32 = 1280 sibling slots,
    while the OI-aligned cold-start admission is 4 x 8 x 32 = 1024 logical
    siblings. ``SWE_ROLLOUT_CONCURRENCY`` throttles the sandbox count and may raise
    startup admission to preserve queued work at higher concurrency.
    """
    config = _swe_9b()
    config.rollouter = _tmax_rollouter()
    assert config.model_spec is not None
    _set_max_seq_len(config.model_spec, _TMAX_9B_CONTEXT)
    # Interleaved thinking: keep each turn's <think> in later prompts (the tmax
    # recipe's preserve_thinking, shown to help agentic RL). The qwen3.5 renderer
    # defaults to preserve_all_thinking=False, which strips prior-turn reasoning;
    # tmax's single-user + tool-loop structure makes preserve_all_thinking the
    # clean match (every past turn stays in the current cycle). Trade-off: prompts
    # grow with retained thinking, so the 65536 context fills sooner.
    config.renderer = dataclasses.replace(config.renderer, preserve_all_thinking=True)
    num_groups_per_train_step = int(
        os.environ.get("SWE_NUM_GROUPS_PER_TRAIN_STEP", "8")
    )
    group_size = int(os.environ.get("SWE_GROUP_SIZE", "32"))
    max_offpolicy_steps = int(os.environ.get("SWE_OFFPOLICY_STEPS", "4"))
    max_active_rollout_groups = int(os.environ.get("SWE_MAX_ACTIVE_GROUPS", "40"))
    num_groups_in_selection_window_env = os.environ.get("SWE_SELECTION_WINDOW_GROUPS")
    num_groups_in_selection_window = (
        int(num_groups_in_selection_window_env)
        if num_groups_in_selection_window_env
        else None
    )
    max_bypass_groups_env = os.environ.get("SWE_MAX_BYPASS_GROUPS")
    if not max_bypass_groups_env or max_bypass_groups_env.lower() == "off":
        max_bypass_groups = None
    else:
        max_bypass_groups = int(max_bypass_groups_env)
    num_rollout_workers = int(os.environ.get("SWE_NUM_ROLLOUT_WORKERS", "8"))
    rollout_concurrency = config.rollouter.rollout_concurrency
    # Open-Instruct cold-starts with async_steps * global_batch_size prompt
    # groups. Also keep at least one queued group per sibling-gate shard: without
    # that headroom, a concurrency=1024 launch would admit exactly 1024 siblings
    # and every early completion would leave its slot idle until a full group tail
    # finished. Keep the OI-aligned 32-group start whenever it supplies that
    # per-worker headroom, and use all 40 groups at concurrency=1024.
    oi_initial_active_groups = max_offpolicy_steps * num_groups_per_train_step
    worker_concurrencies = (
        _split_rollout_concurrency(
            rollout_concurrency,
            num_rollout_workers,
            max_num_workers=max_active_rollout_groups,
        )
        if num_rollout_workers > 0
        else [rollout_concurrency]
    )
    min_work_conserving_groups = sum(
        worker_concurrency // group_size + 1
        for worker_concurrency in worker_concurrencies
    )
    if min_work_conserving_groups > max_active_rollout_groups:
        raise ValueError(
            "The rollout-worker split cannot keep every trajectory gate supplied: "
            f"it needs at least {min_work_conserving_groups} active groups, but "
            f"SWE_MAX_ACTIVE_GROUPS={max_active_rollout_groups}. Reduce "
            "SWE_ROLLOUT_CONCURRENCY or SWE_NUM_ROLLOUT_WORKERS, or increase "
            "SWE_MAX_ACTIVE_GROUPS."
        )
    default_initial_active_groups = min(
        max_active_rollout_groups,
        max(oi_initial_active_groups, min_work_conserving_groups),
    )
    initial_active_rollout_groups = int(
        os.environ.get("SWE_INITIAL_ACTIVE_GROUPS", str(default_initial_active_groups))
    )
    config.async_loop = dataclasses.replace(
        config.async_loop,
        # Total optimizer steps. Swe base = 100; SWE_TRAIN_STEPS raises it (e.g. 500
        # for a long "wash" run that streams zero-std prompt annotations to
        # SWE_ZERO_STD_LOG for a later SWE_SKIP_PROMPTS pass).
        num_training_steps=int(os.environ.get("SWE_TRAIN_STEPS", "100")),
        num_groups_per_train_step=num_groups_per_train_step,
        group_size=group_size,
        # Policy-age cap. Open-Instruct uses async_steps=4 and initially admits
        # async_steps * num_groups = 32 prompt groups.
        max_offpolicy_steps=max_offpolicy_steps,
        # Buffer size (run-ahead groups), DECOUPLED from the staleness cap above.
        # The explicit full capacity is (off+1)*num_groups = 40 because Titan
        # charges the eight trainable groups through trainer weight pull. The
        # separate initial limit below prevents that downstream headroom from being
        # filled with policy-version-0 generation groups.
        max_active_rollout_groups=max_active_rollout_groups,
        # Start with the OI 32-group population when it leaves gate headroom; raise
        # it as needed at high rollout concurrency. Then admit one replacement
        # whenever a trainable group moves downstream until the full end-to-end cap
        # is reached. Zero-std and stale groups release their existing slot instead.
        initial_active_rollout_groups=initial_active_rollout_groups,
        # Override the generator's vLLM max_num_seqs (decode batch cap per engine).
        # Unset = derived from the rollout pool (capped 512). SWE_MAX_NUM_SEQS=512
        # removes the cap so vLLM batches as many concurrent rollouts as KV allows.
        generator_max_num_seqs=(
            int(os.environ["SWE_MAX_NUM_SEQS"])
            if os.environ.get("SWE_MAX_NUM_SEQS")
            else None
        ),
        # Batcher take order. Default take-any preserves historical throughput.
        # SWE_SELECTION_WINDOW_GROUPS=W enables sliding-prefix selection;
        # SWE_MAX_BYPASS_GROUPS optionally applies direct-bypass stall protection;
        # SWE_STRICT_FIFO=1 remains a compatibility alias for W=1.
        group_buffer=dataclasses.replace(
            config.async_loop.group_buffer,
            num_groups_in_selection_window=num_groups_in_selection_window,
            max_bypass_groups=max_bypass_groups,
            strict_fifo=os.environ.get("SWE_STRICT_FIFO", "0") == "1",
        ),
        training_sample_builder=TrainingSampleBuilder.Config(
            drop_zero_std_reward_groups=(
                os.environ.get("SWE_DROP_ZERO_STD", "1") == "1"
            ),
            # Open-Instruct keeps an exhausted sandbox/reset failure as a
            # zero-reward sibling. It participates in centered advantage while
            # its empty completion contributes no training tokens.
            drop_groups_with_untrainable_rollouts=False,
        ),
        batcher=dataclasses.replace(
            config.async_loop.batcher,
            batch=dataclasses.replace(
                config.async_loop.batcher.batch, seq_len=_TMAX_9B_CONTEXT
            ),
        ),
        # Periodic held-out eval every 20 steps (+ start/end): the trained-batch reward is
        # locked near ~0.5 by drop_zero_std, so it is NOT a learning signal; a greedy
        # (temp=0, n=1) pass over held-out tmax tasks via the same Daytona rollout+grade path
        # gives the real solve-rate curve inline, with no separate eval job or ckpt download.
        # The swe_r2e base sets num_samples=0 (off); we turn it on here. To eval the real
        # terminal-bench@2.0 benchmark instead, point the rollouter's validation_dataset at
        # TB-2.0 tasks in the tmax task format (see examples/tmax/data.py schema).
        validation=_tmax_9b_validation(),
    )
    # RolloutWorker pool: run group rollouts across N CPU processes on the
    # controller host, off the controller GIL (the per-turn agent orchestration --
    # adapter, Daytona HTTP, grading -- otherwise serializes on one GIL and caps
    # throughput). SWE_NUM_ROLLOUT_WORKERS=0 keeps the in-process path; default 8.
    # The global SWE_ROLLOUT_CONCURRENCY is split across the pool.
    config.num_rollout_workers = num_rollout_workers
    # Dedicated eval capacity: SWE_NUM_EVAL_GENERATORS=1 adds one generator host that
    # only serves the periodic validation pass, so a full Terminal-Bench 2.0 sweep
    # runs alongside training instead of stalling the step. Its rollouts get their own
    # CPU worker processes for the same reason the training pool exists (the per-turn
    # agent orchestration serializes on one GIL).
    config.num_eval_generators = int(os.environ.get("SWE_NUM_EVAL_GENERATORS", "0"))
    config.num_eval_rollout_workers = (
        int(os.environ.get("SWE_NUM_EVAL_ROLLOUT_WORKERS", "4"))
        if config.num_eval_generators
        else 0
    )
    # Cap the eval sandbox burst. A k=5 sweep over 89 TB-2.0 tasks is 445 rollouts;
    # admitting them all at once puts 445 sandbox creates on top of the training
    # pool's, and Daytona rate-limits creation well below that.
    config.eval_rollout_concurrency = int(
        os.environ.get("SWE_EVAL_ROLLOUT_CONCURRENCY", "128")
    )
    # Weight-sync KV policy. Default (SWE_SALT_KV=1): keep in-flight KV AND the prefix
    # cache (no preempt, no full re-prefill) and salt the prefix cache per GROUP (its n
    # samples share one namespace), so a NEW group recomputes its prefix under the new
    # weights while an in-flight group keeps reusing its own KV. Mirrors open-instruct's
    # per-prompt inflight update (cache_salt=base_request_id +
    # inflight_updates_recompute_kv_cache=False), and drops the per-step full-batch
    # re-prefill storm. SWE_SALT_KV=0 reverts to the reset-and-re-prefill path.
    _salt_kv = os.environ.get("SWE_SALT_KV", "1") == "1"
    # cudagraph FULL_DECODE_ONLY: ~3x GDN decode throughput (local bench 27->85 tok/s on
    # the 4B unified), which directly cuts the time-budget nonsubmit rate (see the
    # finish-reason analysis: ~30% of rollouts die on the wall). tmax DEFAULTS it ON
    # (SWE_GEN_CUDAGRAPH default "1" here, vs "0" in the swe base). Stays
    # FULL_DECODE_ONLY -- a mixed prefill-decode FULL graph corrupts (#3668).
    # SWE_GEN_CUDAGRAPH=0 reverts to eager decode (smaller gen/train logprob mismatch,
    # ~3x slower). The SWE_GDN_BI block below also forces it on (bitwise-safe there).
    _cudagraph_on = os.environ.get("SWE_GEN_CUDAGRAPH", "1") == "1"
    config.generator = dataclasses.replace(
        config.generator,
        sampling=dataclasses.replace(
            config.generator.sampling, max_tokens=_TMAX_9B_PER_TURN_TOKENS
        ),
        cudagraph=VLLMCudagraphConfig(enable=_cudagraph_on, mode="FULL_DECODE_ONLY"),
        salt_prefix_cache_on_weight_sync=_salt_kv,
        reset_prefix_cache_on_weight_sync=not _salt_kv,
        reset_running_requests_on_weight_sync=not _salt_kv,
    )
    # 32 chunks keeps per-chunk fp32 lm_head logits ~1.2 GiB at seq_len 65536
    # (16 chunks -> ~2.3 GiB, an OOM risk); 65536 % 32 == 0 for the chunk split.
    # Save a full training-state checkpoint every 20 steps (matches the paper's
    # save_freq 20) so the run is resumable after a crash and each snapshot is
    # eval-able; keep last_save_model_only so the final step-100 save is a clean
    # model-only export for serving. The swe base uses interval=10000 (final save
    # only), which risks losing the whole ~20h run to any mid-run crash.
    # Loss: the tmax recipe's DPPO is the DEFAULT (SWE_LOSS=dapo reverts). See
    # _tmax_recipe_loss. num_chunks=32 (chunked lm-head loss) is preserved.
    _loss = _tmax_recipe_loss(dataclasses.replace(config.trainer.loss, num_chunks=32))
    config.trainer = dataclasses.replace(
        config.trainer,
        loss=_loss,
        optimizer=_tmax_9b_adamw(),
        training=dataclasses.replace(
            config.trainer.training,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        checkpoint=dataclasses.replace(config.trainer.checkpoint, interval=20),
    )
    # Optional trainer FSDP width override for a fwd/bwd speed experiment. The multi-host launcher
    # launcher derives the trainer host count from data_parallel_shard_degree, so
    # SWE_DP_SHARD=16 -> 2 trainer hosts (FSDP-16), spreading the packed rows across
    # 2x DP ranks -> ~half the microbatches per rank -> ~2x faster fwd/bwd. Default
    # unset (0) keeps the base FSDP-8.
    _dp_shard = int(os.environ.get("SWE_DP_SHARD", "0"))
    if _dp_shard:
        config.trainer = dataclasses.replace(
            config.trainer,
            parallelism=dataclasses.replace(
                config.trainer.parallelism, data_parallel_shard_degree=_dp_shard
            ),
        )
    # Optional AC-policy override for a fwd/bwd speed experiment. The base is FullAC
    # (recompute the whole forward -- needed to fit seq 65536). SWE_AC=selective swaps
    # in per-op SAC, which saves the expensive aten op outputs (projections, flash-attn
    # in the 25% softmax layers) and recomputes the rest -> less recompute, more memory.
    # Caveat: the fla GDN kernel is a dynamo-disabled custom autograd op invisible to the
    # SAC aten policy, so the 75% GDN layers' kernel is recomputed regardless; the win is
    # capped and only materializes if the extra saved activations fit at seq 65536.
    _ac = os.environ.get("SWE_AC", "").lower()
    if _ac == "selective":
        config.trainer = dataclasses.replace(
            config.trainer, ac_config=SelectiveAC.Config()
        )
    elif _ac in ("none", "off"):
        # No activation checkpointing: keep the full forward activations instead of
        # recomputing them in backward -> less compute, much more memory. A memory
        # probe at seq 65536 (likely OOMs; FullAC exists because the activations are
        # large), paired with SWE_LOCAL_BSZ.
        config.trainer = dataclasses.replace(config.trainer, ac_config=None)
    # Optional per-rank microbatch width override (rows per forward pass). Default 1
    # (one 65536-token row per forward). SWE_LOCAL_BSZ=4 packs 4 rows/forward ->
    # fewer, larger microbatches but 4x the activation memory per forward.
    _lbsz = int(os.environ.get("SWE_LOCAL_BSZ", "0"))
    if _lbsz:
        config.async_loop = dataclasses.replace(
            config.async_loop,
            batcher=dataclasses.replace(
                config.async_loop.batcher,
                batch=dataclasses.replace(
                    config.async_loop.batcher.batch, local_batch_size=_lbsz
                ),
            ),
        )
    # Optional learning-rate override (SWE_LR). Rebuild through the TMax helper so
    # changing lr preserves the open-instruct betas, eps, weight decay, fp32 states,
    # and fused implementation.
    _lr = float(os.environ.get("SWE_LR", "0") or "0")
    if _lr > 0:
        config.trainer = dataclasses.replace(
            config.trainer, optimizer=_tmax_9b_adamw(lr=_lr)
        )
    # Batch-invariant GDN mode (SWE_GDN_BI=1): make the generator's decode == prefill
    # == trainer logprobs BITWISE by routing BOTH the trainer and the unified vLLM
    # wrapper GDN core through the SAME fla recurrent kernel (recurrent-everywhere).
    # One switch flips the four coupled settings the path requires:
    #   - trainer.debug batch_invariant  -> _RecurrentFwdChunkBwd (recurrent fwd, chunk bwd)
    #   - generator.debug batch_invariant -> the _forward_recurrent_bi decode path
    #   - generator.backend torchtitan_wrapper -> the unified GDN core that holds it
    #   - generator.gdn_trainer_parity   -> _TRAINER_PARITY_FLA (selects the fla recurrence)
    # Under BI the recurrent-everywhere decode is FULL_DECODE_ONLY cudagraph-capturable
    # and its PAGED conv/ssm state makes prefix caching bitwise (846c51b0), so both stay
    # ON. The weight-sync KV policy is INHERITED from the tmax base's salt-KV setting
    # (SWE_SALT_KV, default salt-on): recurrent-BI prefix reuse is bitwise WITHIN a
    # policy version, so salt is compatible -- only sync-straddling in-flight samples
    # carry the usual off-policy drift (generator __post_init__ warns, no longer errors).
    # SP must be off for BI (already is at trainer TP=1; set explicitly).
    if os.environ.get("SWE_GDN_BI", "0") == "1":
        _bi = DebugConfig(batch_invariant=True, deterministic=True)
        config.trainer = dataclasses.replace(
            config.trainer,
            debug=_bi,
            parallelism=dataclasses.replace(
                config.trainer.parallelism, enable_sequence_parallel=False
            ),
        )
        config.generator = dataclasses.replace(
            config.generator,
            backend="torchtitan_wrapper",
            gdn_trainer_parity=True,
            debug=_bi,
            enable_prefix_caching=True,
            cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
        )
    return config


def rl_grpo_qwen3_4b_tmax() -> Controller.Config:
    """Qwen3-4B (DENSE softmax attention) tmax recipe in BATCH-INVARIANT mode.

    A numerics control for the Qwen3.5-9B GDN recipe. It reuses every tmax delta
    from ``rl_grpo_qwen3_5_9b_tmax`` (TMaxRollouter, group_size 32, off-policy,
    chunked DPPO loss, 65536 context, preserve_all_thinking) but swaps the model to
    the DENSE Qwen3-4B (no Gated DeltaNet) and turns on ``batch_invariant`` +
    ``deterministic`` on BOTH the trainer and the generator.

    The generator runs the SAME torchtitan model inside vLLM
    (``backend="torchtitan_wrapper"``, not the 9B base's ``vllm_native`` GDN) with
    matched varlen attention (``num_splits=1`` / FA3) and the fp32 lm_head, so batch
    invariance makes the trainer and generator per-token logprobs bitwise-identical
    (see ``tests/test_bitwise_parity.py``). Expectation:
    ``bit_wise/logprob_diff/{mean,abs_mean,max}`` collapse to ~0 -- isolating how
    much of the 9B GDN logprob drift is GDN-specific (chunk-parallel train vs
    recurrent decode) rather than generic batch/kernel nondeterminism.

    The trainer keeps the 9B recipe's fp32 master parameters and uses bf16 forward
    parameters through FSDP mixed precision. It also requires no sequence parallel
    and the reset (not salt) prefix-cache policy; the 9B tmax base enables salt-KV,
    so it is turned off here.
    """
    _bi = DebugConfig(batch_invariant=True, deterministic=True)
    config = rl_grpo_qwen3_5_9b_tmax()
    # DENSE Qwen3-4B (softmax attention, varlen batch-invariant path); fp32 lm_head.
    config.model_spec = _qwen3_rl_model_registry("4B", attn_backend="varlen")
    _set_max_seq_len(config.model_spec, _TMAX_9B_CONTEXT)
    config.hf_assets_path = f"{_CKPT_DIR}/Qwen3-4B"
    # Dense Qwen3 chat template (the 9B base uses the qwen3.5 renderer).
    config.renderer = dataclasses.replace(config.renderer, name="qwen3")
    # Trainer: batch-invariant + deterministic; SP must be off (already is at TP=1).
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_bi,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, enable_sequence_parallel=False
        ),
    )
    # Generator: run the SAME torchtitan model in vLLM (not vllm_native GDN), drop the
    # GDN-only engine config + mamba cache dtype, and use the reset (not salt)
    # prefix-cache policy required under batch invariance.
    config.generator = dataclasses.replace(
        config.generator,
        backend="torchtitan_wrapper",
        vllm_additional_config={},
        mamba_ssm_cache_dtype="auto",
        debug=_bi,
        salt_prefix_cache_on_weight_sync=False,
        reset_prefix_cache_on_weight_sync=True,
        reset_running_requests_on_weight_sync=True,
    )
    return config


def rl_grpo_qwen3_5_9b_tmax_tb2_eval() -> Controller.Config:
    """Eval-only: score the Qwen3.5-9B tmax policy on the full Terminal-Bench 2.0
    benchmark (89 tasks), greedy pass@1, via the same Daytona rollout + grade path.

    Base = ``rl_grpo_qwen3_5_9b_tmax`` (same model / generator / renderer so the
    trainer->generator weight sync works unchanged). Three changes make it eval-only:

      1. Datasets point at the TB-2.0 JSONL (``SWE_TB2_DATA``, prepare_tb2_data.py
         output). ``holdout_n=0`` makes both splits read the WHOLE file, so a
         validation pass scores all 89 tasks; the train stream only feeds the
         transient background collection that ``run()`` cancels once the 0-step
         trainer returns.
      2. ``num_training_steps=0`` -> ``run()`` does only the pre-training validation
         pass (= the TB-2.0 solve-rate), no optimizer steps. ``interval=0`` disables
         mid-training validation.
      3. The trained DCP checkpoint (``SWE_TB2_CKPT``, e.g. the run's
         ``checkpoint/step-100``) loads as the INITIAL model weights (not a resume):
         a fresh dump dir has no ``checkpoint/`` to resume, so CheckpointManager
         falls to ``initial_load_path``. ``initial_load_in_hf=False`` -> native titan
         DCP (the run saved it that way); model-only -> just the policy weights.

    Set ``SWE_ROLLOUT_CONCURRENCY`` >= 89 so all tasks run at once (validation shares
    the global rollout semaphore). Greedy (temp=0, n=1) is applied by ``validate()``.
    """
    config = rl_grpo_qwen3_5_9b_tmax()
    tb2_data = _TB2_DATA or _DEFAULT_DATA
    config.rollouter = dataclasses.replace(
        config.rollouter,
        train_dataset=TMaxDataset.Config(
            data_path=tb2_data, seed=42, holdout_n=0, split="train", shuffle=False
        ),
        validation_dataset=TMaxDataset.Config(
            data_path=tb2_data, seed=99, holdout_n=0, split="validation", shuffle=False
        ),
    )
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=0,
        validation=dataclasses.replace(
            _tmax_9b_validation(),
            num_samples=int(os.environ.get("SWE_VAL_SAMPLES", _TB2_NUM_TASKS)),
            interval=0,
            # Nothing to overlap with at 0 training steps.
            run_async=False,
            group_size=_TB2_VAL_K,
            temperature=_TB2_VAL_TEMPERATURE,
            top_p=_TB2_VAL_TOP_P,
        ),
    )
    if _TB2_CKPT:
        config.trainer = dataclasses.replace(
            config.trainer,
            checkpoint=dataclasses.replace(
                config.trainer.checkpoint,
                enable=True,
                initial_load_path=_TB2_CKPT,
                initial_load_in_hf=False,
                initial_load_model_only=True,
            ),
        )
    return config

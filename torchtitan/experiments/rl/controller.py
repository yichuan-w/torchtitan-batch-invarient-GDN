# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TLDR:
_data_input_loop -> _rollout_loop -> _batcher_loop -> training_batch_queue -> _trainer_loop
           |               ^                    ^
           v               |                    |
           +-------RolloutGroupWorkBuffer-------+

Detailed diagram:

_data_input_loop                                      _rollout_loop[N] (group workers)
+--------------------------------------------------+  +--------------------------------------------------+
| group_buffer.wait_for_slot()                     |  | work = group_buffer.claim_next()                  |
| sample = rollouter.get_training_sample()         |  | group = rollouter.run_group_rollouts(work.sample) |
| work = RolloutGroupWork(group_id, sample)        |  | group_buffer.finalize_work(group)                 |
| group_buffer.add_work(work)                      |  +-----------------------+--------------------------+
+-----------------------+--------------------------+                          ^ |
                        |                                                     | |
                        | adds work entry                                     | | updates same entry
                        v                                                     | v
RolloutGroupWorkBuffer
+---------------------------------------------------------------------------------------------------------------------+
| active slots = (max_offpolicy_steps + 1) * num_groups_per_train_step                                                |
|                                                                                                                     |
| caller            group_buffer call                                            state / active slot                  |
| _data_input_loop  add_work(RolloutGroupWork)                                   WAITING; slot acquired               |
| _rollout_loop[N]  claim_next()                                                 WAITING -> INFLIGHT                  |
| _rollout_loop[N]  finalize_work(RolloutGroup)                                  INFLIGHT -> FINALIZED                |
| _batcher_loop     RolloutGroup = take_finalized()                              FINALIZED -> taken (slot still held) |
| _batcher_loop     release_active_groups(1, "untrainable_group")                slot released                        |
| _trainer_loop     release_active_groups(num_groups_per_train_step, "trained")  slots released after weight pull     |
+---------------------------------------------------------------------------------------------------------------------+
                                                  |
                                                  | group = group_buffer.take_finalized()
                                                  v
_batcher_loop
+----------------------------------------------------------------------------------------+
| training_sample_group = training_sample_builder.build_from_group(rollout_group=group)  |
| if no trainable samples: group_buffer.release_active_groups(1, "untrainable_group")    |
| maybe_training_batch = batcher.add_training_samples(training_sample_group)             |
| training_batch_queue.put(TrainingBatch)                                                |
+-----------------------+------------------------------+---------------------------------+
                        |  ^                           |
          add group     |  | maybe_training_batch      | put batch
                        v  |                           v
Batcher                                              training_batch_queue
+-------------------------------------------------+   +----------------------------------------------+
| accumulated TrainingSampleGroups                |   | size 1; holds TrainingBatch | None           |
| pack at num_groups_per_train_step               |   +---------------------+------------------------+
+-------------------------------------------------+                         |
                                                                         | packed = training_batch_queue.get()
                                                                         v
_trainer_loop
+----------------------------------------------------------------------------------------------------------+
| train batch -> optim -> push/pull weights -> buffer.release_active_groups(num_groups_per_train_step)     |
+----------------------------------------------------------------------------------------------------------+

Backpressure (each loop: what it consumes/produces, and what gates each side):
_data_input_loop
  produces: RolloutGroupWork into group_buffer
    waits for:    a free active slot (group_buffer.wait_for_slot)
    unblocked by: _trainer_loop release_active_groups(num_groups_per_train_step, "trained") after the pull
                  (and _batcher_loop release_active_groups(1,"untrainable_group"))
_rollout_loop[N]
  consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
    waits for:    a claimable WAITING entry
    unblocked by: _data_input_loop group_buffer.add_work()
  produces: RolloutGroup (group_buffer.finalize_work)
    waits for:    nothing (admits its own claimed slot)
    unblocked by: n/a

_batcher_loop
  consumes: the oldest-admitted group that is FINALIZED (group_buffer.take_finalized)
    waits for:    any active group becoming FINALIZED
    unblocked by: _rollout_loop[N] group_buffer.finalize_work()
  produces: TrainingBatch (training_batch_queue.put)
    waits for:    a free training_batch_queue slot (maxsize=1)
    unblocked by: _trainer_loop training_batch_queue.get()
_trainer_loop
  consumes: a TrainingBatch (training_batch_queue.get)
    waits for:    a TrainingBatch in the queue
    unblocked by: _batcher_loop training_batch_queue.put()
"""

import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# PYTORCH_CUDA_ALLOC_CONF is set in torchtitan/experiments/rl/__init__.py (before torch is imported)
# and in train.py; see the note there.
import torch  # noqa: F401
import torchstore as ts
from monarch.actor import ProcMesh, this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import CompileConfig, Configurable
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.rollout_worker import RolloutWorker
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import Batcher
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.components.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.experiments.rl.controller_metrics import (
    combine_microbatch_metrics,
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    compute_rollout_metrics,
    MetricsTimer,
)
from torchtitan.experiments.rl.eval_trace_recorder import (
    EvalSummary,
    ValidationTraceRecorder,
)
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.types import RoutingContext
from torchtitan.experiments.rl.types import Completion, TrainingBatch
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


def _split_rollout_concurrency(
    global_concurrency: int,
    num_workers: int,
    *,
    max_num_workers: int | None = None,
) -> list[int]:
    """Split a global rollout limit across worker shards that can receive work."""
    if global_concurrency < 1:
        raise ValueError(
            f"rollout_concurrency must be positive, got {global_concurrency}"
        )
    if num_workers < 1:
        raise ValueError(f"num_rollout_workers must be positive, got {num_workers}")
    if max_num_workers is not None and max_num_workers < 1:
        raise ValueError(f"max_num_workers must be positive, got {max_num_workers}")

    num_active_workers = min(
        global_concurrency,
        num_workers,
        max_num_workers if max_num_workers is not None else num_workers,
    )
    base, remainder = divmod(global_concurrency, num_active_workers)
    return [base + (worker_id < remainder) for worker_id in range(num_active_workers)]


def _task_id(sample: object, index: int) -> str:
    """Name a validation prompt for the trace report.

    Datasets that carry a stable task identity expose ``instance_id`` (the
    convention across the coding-agent examples); anything else is named by its
    position in the pass.
    """
    instance_id = getattr(sample, "instance_id", None)
    return instance_id if isinstance(instance_id, str) else f"prompt_{index:04d}"


def _should_drop_group_at_batcher(*, group_age: int, max_offpolicy_steps: int) -> bool:
    """Drop groups that are already past the configured policy-age limit."""
    return group_age > max_offpolicy_steps


@dataclass(kw_only=True, slots=True)
class ValidationConfig:
    """Held-out validation that runs at the start and end of training, and
    optionally every ``interval`` steps in between."""

    num_samples: int = 20
    """Held-out prompts scored per validation pass. 0 skips validation."""

    interval: int = 0
    """Run a mid-training validation pass every ``interval`` optimizer steps (in addition to
    the start/end passes). 0 = only start/end. The pass reuses idle generator capacity and
    disjoint (negative) group ids, so it overlaps ongoing training-rollout collection."""

    group_size: int = 1
    """Trials per held-out prompt. 1 = pass@1. k > 1 reports avg@k and pass@k, which is
    far less noisy on sparse binary rewards -- at the cost of k times the environments."""

    temperature: float = 0.0
    """Sampling temperature. The default 0.0 is greedy; a benchmark whose published
    number uses sampling (e.g. terminal-bench avg@k at 0.7) must match it here."""

    top_p: float = 1.0
    """Nucleus sampling cutoff, paired with ``temperature``."""

    run_async: bool = False
    """Run the pass as a background task instead of blocking the training step.

    Off (default): the trainer awaits the pass, so its metrics land on the step that
    produced them. On: training continues while the pass runs, and its metrics are
    logged when it finishes, tagged with ``validation/policy_version``. A pass still
    in flight when the next interval arrives is skipped, not queued. Meaningful with
    ``num_eval_generators > 0``: otherwise the pass competes with rollout collection
    for the training generators."""

    trace: ValidationTraceRecorder.Config = field(
        default_factory=ValidationTraceRecorder.Config
    )
    """Per-pass browsable trace report (off by default)."""

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError(
                f"validation.group_size must be positive, got {self.group_size}"
            )
        if self.num_samples < 0:
            raise ValueError(
                f"validation.num_samples must be non-negative, got {self.num_samples}"
            )


def _will_validate_after_step(
    *, validation: ValidationConfig, step: int, num_training_steps: int
) -> bool:
    """Return whether a non-empty validation pass follows this training step."""
    return validation.num_samples > 0 and (
        step == num_training_steps
        or (validation.interval > 0 and step % validation.interval == 0)
    )


@dataclass(kw_only=True, slots=True)
class AsyncLoopConfig(Configurable.Config):
    num_training_steps: int = 10
    """Optimizer steps to run."""

    num_groups_per_train_step: int = 8
    """Global number of prompt groups, across all DPs, whose surviving rollouts compose
    one train step (the global_batch_size, in groups)."""

    group_size: int = 8
    """Sibling rollouts sampled per prompt (the GRPO group)."""

    max_offpolicy_steps: int = 3
    """Max train-steps a rollout may lag its reserved consuming policy version. A
    group that exceeds this cap is stale-dropped before entering the batch cohort,
    and the trainer never trains on older data. 0 = fully on-policy (sync): generator
    and trainer alternate in lockstep."""

    max_active_rollout_groups: int | None = None
    """Rollout buffer size: peak concurrent active rollout groups (the run-ahead
    depth). None couples it to the staleness cap: (max_offpolicy_steps + 1) *
    num_groups_per_train_step. Set LARGER to DECOUPLE run-ahead from staleness -- more
    rollouts generate concurrently (higher generator utilization, less trainer
    starvation) WITHOUT loosening max_offpolicy_steps: groups still stale-drop at the
    batcher once they age past it. This is a mean_age (queue size) vs max_age
    (drop) decoupling. Generator max_num_seqs scales with this (the max rollouts on the
    fly); it is a ceiling, not a KV reservation, so vLLM pages KV on demand and admits
    fewer / preempts when tight rather than OOM-ing."""

    initial_active_rollout_groups: int | None = None
    """Cold-start admission limit in prompt groups. None starts at the full
    ``max_active_rollout_groups`` capacity (the historical behavior). When smaller
    than the full capacity, each trainable group retained downstream grows the
    admission limit by one until the full capacity is reached. Dropped groups free
    their existing slot instead. This preserves one-result-in/one-prompt-out
    generation inventory without filling the downstream headroom at policy version
    zero."""

    generator_max_num_seqs: int | None = None
    """Explicit override for the generator's vLLM max_num_seqs (the scheduler's max
    concurrent decode batch per engine). None derives it from the rollout pool:
    ceil(resolved_max_active_rollout_groups * group_size / num_generator_shards),
    capped at 512. Set (e.g. 512) to remove the per-engine batch cap so vLLM batches
    as many concurrent rollouts as KV allows; harmless as a ceiling (vLLM admits
    fewer / preempts when KV is tight) but grows cudagraph capture sizes."""

    group_buffer: RolloutGroupWorkBuffer.Config = field(
        default_factory=RolloutGroupWorkBuffer.Config
    )
    training_sample_builder: TrainingSampleBuilder.Config = field(
        default_factory=TrainingSampleBuilder.Config
    )
    batcher: Batcher.Config = field(default_factory=Batcher.Config)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def resolved_max_active_rollout_groups(self) -> int:
        """Rollout buffer size in groups: the explicit override if set, else the
        staleness-coupled default ``(max_offpolicy_steps + 1) * num_groups_per_train_step``."""
        if self.max_active_rollout_groups is not None:
            return self.max_active_rollout_groups
        return (self.max_offpolicy_steps + 1) * self.num_groups_per_train_step

    def resolved_initial_active_rollout_groups(self) -> int:
        """Resolve and validate the cold-start active-group admission limit."""
        if self.num_groups_per_train_step < 1:
            raise ValueError(
                "num_groups_per_train_step must be positive, got "
                f"{self.num_groups_per_train_step}"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.max_offpolicy_steps < 0:
            raise ValueError(
                "max_offpolicy_steps must be non-negative, got "
                f"{self.max_offpolicy_steps}"
            )
        max_active_rollout_groups = self.resolved_max_active_rollout_groups()
        if max_active_rollout_groups < 1:
            raise ValueError(
                "max_active_rollout_groups must be positive, got "
                f"{max_active_rollout_groups}"
            )
        if max_active_rollout_groups < self.num_groups_per_train_step:
            raise ValueError(
                "max_active_rollout_groups must be at least "
                f"num_groups_per_train_step ({self.num_groups_per_train_step}), "
                f"got {max_active_rollout_groups}"
            )
        num_groups_in_selection_window = (
            self.group_buffer.resolved_num_groups_in_selection_window()
        )
        if (
            num_groups_in_selection_window is not None
            and num_groups_in_selection_window > max_active_rollout_groups
        ):
            raise ValueError(
                "num_groups_in_selection_window must not exceed "
                f"max_active_rollout_groups ({max_active_rollout_groups}), got "
                f"{num_groups_in_selection_window}"
            )
        if self.initial_active_rollout_groups is None:
            return max_active_rollout_groups
        if not 1 <= self.initial_active_rollout_groups <= max_active_rollout_groups:
            raise ValueError(
                "initial_active_rollout_groups must be between 1 and "
                f"max_active_rollout_groups ({max_active_rollout_groups}), got "
                f"{self.initial_active_rollout_groups}"
            )
        return self.initial_active_rollout_groups

    def __post_init__(self) -> None:
        self.resolved_initial_active_rollout_groups()


class Controller(Configurable):
    """Top-level RL async training orchestrator.

    Owns a `PolicyTrainer` actor (gradient updates), a `VLLMGenerator` actor
    (sampling), and a `Rollouter` (datasets + rubric + env construction).

    Check the docstring at the top of the file for more details.

    Example:

        config = config_registry.rl_grpo_qwen3_0_6b_varlen()
        controller = config.build()
        trainer_mesh = ...        # provisioned by the caller (see train.py)
        generator_meshes = ...
        await controller.setup_async(
            trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
        )
        await controller.run()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        async_loop: AsyncLoopConfig = field(default_factory=AsyncLoopConfig)
        """How the data->rollout->batch->train loop is sized and coordinated."""

        rollouter: Rollouter.Config
        """The rollouter: its datasets, envs, and rubric."""
        # TODO: support multiple rollouters for data mixing.

        renderer: RendererConfig
        """Message-to-token renderer config."""

        rollout_recorder: RolloutSampleRecorder.Config = field(
            default_factory=RolloutSampleRecorder.Config
        )
        """JSONL recorder to save sampled rollouts to disk for further inspection and debugging."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        # TODO: put generator, num generators and generator router in a separate config
        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        num_generators: int = 1
        """Number of generator replicas to spawn as separate proc meshes.

        This is distinct from intra-generator parallelism controlled by
        ``generator.parallelism``. Total generator GPU/process usage is
        ``num_generators * generator_world_size``.
        """

        num_eval_generators: int = 0
        """Generator replicas reserved for validation, on their own proc meshes.

        0 (default) runs validation through the training generators. N > 0 spawns N
        more generators, each sized like a training one, that are kept out of the
        training router and out of the RolloutWorker pool -- so a validation pass
        never competes with rollout collection. They pull weights only on the steps
        they evaluate, which freezes the policy under eval at that version while
        training moves on (see ``ValidationConfig.run_async``).
        """

        num_eval_rollout_workers: int = 0
        """CPU RolloutWorker processes dedicated to validation group rollouts.

        0 runs validation in-process on the controller. Requires
        ``num_eval_generators > 0``: these workers route only to the eval generators.
        """

        eval_rollout_concurrency: int = 0
        """Concurrently-ACTIVE validation rollouts across the eval worker pool.

        0 (default) admits the whole pass at once: fastest, but it bursts
        ``num_samples * group_size`` environments. Set it lower when the environment
        provider rate-limits creation, or when that burst would contend with the
        training pool for the same quota. Only the admission rate changes -- the pass
        still covers every prompt.
        """

        num_rollout_workers: int = 0
        """CPU RolloutWorker processes to run group rollouts off the controller GIL.

        0 (default) runs rollouts in-process on the controller (the original
        path). N > 0 spawns N single-proc CPU actors on the controller host; the
        controller pins rollout-loop lanes to workers round-robin, so the agent
        orchestration (adapter, Daytona HTTP, grading) runs across N GILs. A lane
        claims the next group from the shared buffer; group id is not a permanent
        worker assignment. The global rollout concurrency is split across workers.
        """

        torchstore_reset_interval: int = 0
        """Compatibility field. Runtime TorchStore recycling is unsupported."""

        torchstore_volume_placement: str = "trainer"
        """Where TorchStore hosts StorageVolumes: trainer, controller, or controller_sharded."""

        generator_router: InterGeneratorRouter.Config = field(
            default_factory=InterGeneratorRouter.Config
        )
        """Generator routing strategy configuration."""

        # TODO: rename it to metrics_processor
        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        def __post_init__(self):
            if self.num_generators < 1:
                raise ValueError(
                    f"num_generators must be at least 1, got {self.num_generators}"
                )
            if self.num_eval_generators < 0:
                raise ValueError(
                    "num_eval_generators must be non-negative, got "
                    f"{self.num_eval_generators}"
                )
            if self.num_eval_rollout_workers < 0:
                raise ValueError(
                    "num_eval_rollout_workers must be non-negative, got "
                    f"{self.num_eval_rollout_workers}"
                )
            if self.num_eval_rollout_workers > 0 and self.num_eval_generators == 0:
                raise ValueError(
                    "num_eval_rollout_workers requires num_eval_generators > 0: the "
                    "eval workers route only to the dedicated eval generators."
                )
            if self.eval_rollout_concurrency < 0:
                raise ValueError(
                    "eval_rollout_concurrency must be non-negative, got "
                    f"{self.eval_rollout_concurrency}"
                )
            if self.torchstore_reset_interval != 0:
                raise ValueError(
                    "torchstore_reset_interval must be 0. Recycling TorchStore's "
                    "Gloo transport can asynchronously destroy a custom ProcessGroup "
                    "and abort the trainer process."
                )
            if self.torchstore_volume_placement not in {
                "trainer",
                "controller",
                "controller_sharded",
            }:
                raise ValueError(
                    "torchstore_volume_placement must be 'trainer', 'controller', "
                    "or 'controller_sharded', "
                    f"got {self.torchstore_volume_placement!r}"
                )
            if self.generator.checkpoint.enable:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpoint.enable=False."
                )
            # RL policy inputs are shaped by BatchConfig, not TrainingConfig.
            if self.trainer.parallelism.enable_sequence_parallel:
                sp_degree = self.trainer.parallelism.tensor_parallel_degree
                seq_len = self.async_loop.batcher.batch.seq_len
                if sp_degree > 1 and seq_len % sp_degree != 0:
                    raise ValueError(
                        f"RL batcher sequence length ({seq_len}) must be divisible "
                        f"by sequence parallel degree ({sp_degree})."
                    )

            # Mirror the batcher width into trainer.training.seq_len for the model build.
            self.trainer.training.seq_len = self.async_loop.batcher.batch.seq_len

            # TODO: add a check so that all seq_len related variables make sense
            # e.g. rollout max length cannot be larger than the model max_seq_len
            # or the packing len, etc.

            if self.trainer.debug.batch_invariant:
                if torch.version.hip is not None:
                    raise ValueError(
                        "batch_invariant mode is not supported on ROCm: the varlen "
                        "attention path cannot force num_splits=1 (rejected by ROCm), "
                        "so split-k reductions are non-deterministic."
                    )
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                training = self.trainer.training
                # FSDP2 applies MixedPrecisionPolicy even on a singleton FSDP
                # axis, so this field controls the forward parameter dtype
                # independently of the master parameter dtype.
                if training.mixed_precision_param != "bfloat16":
                    raise ValueError(
                        "batch_invariant requires bfloat16 forward parameters via "
                        "training.mixed_precision_param='bfloat16'; got "
                        f"training.dtype={training.dtype!r}, "
                        "training.mixed_precision_param="
                        f"{training.mixed_precision_param!r}"
                    )
                if self.generator.model_dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 generator dtype, "
                        f"got {self.generator.model_dtype!r}"
                    )
                if self.trainer.parallelism.enable_sequence_parallel:
                    raise ValueError(
                        "batch_invariant mode doesn't support SP now. "
                        "SP uses reduce-scatter which only supports Ring in NCCL "
                        "and has not been validated for determinism."
                    )

            if (
                not self.generator_router.hot_swap
                and not self.generator.reset_prefix_cache_on_weight_sync
            ):
                raise ValueError(
                    "generator_router.hot_swap=False requires "
                    "generator.reset_prefix_cache_on_weight_sync=True, else requests admitted after a "
                    "pull reuse KV cached under the old weights."
                )

            # FULL cudagraph is only correct with the flex attention backend
            cudagraph = self.generator.cudagraph
            if (
                cudagraph.enable
                and cudagraph.mode == "FULL"
                and self.model_spec is not None
            ):
                from torchtitan.models.common.attention import FlexAttention

                inner_attn = self.model_spec.model.layers[0].attention.inner_attention
                if not isinstance(inner_attn, FlexAttention.Config):
                    raise ValueError(
                        "cudagraph mode 'FULL' is only supported with the flex "
                        "attention backend; the varlen backend corrupts FULL capture "
                        "of mixed prefill+decode batches (#3709). Use FULL_DECODE_ONLY "
                        "or FULL_AND_PIECEWISE."
                    )

    def __init__(self, config: Config):
        self.config = config
        self.trainer: PolicyTrainer | None = None
        self.generator_router: InterGeneratorRouter | None = None
        # Validation-only generators + workers (num_eval_generators > 0). Held out of
        # the training router so a pass never steals rollout-collection capacity.
        self.eval_generator_router: InterGeneratorRouter | None = None
        self._eval_rollout_workers: list = []
        # In-flight background validation pass (validation.run_async); None = idle.
        self._validation_task: asyncio.Task | None = None
        # Aggregated metrics of each completed pass, keyed by the policy version it
        # scored. A background pass has no return value to the caller, so the
        # pre/post reward summary reads the start-step entry from here.
        self._validation_results: dict[int, dict[str, float]] = {}
        # CPU RolloutWorker actors (num_rollout_workers > 0); empty = in-process.
        self._rollout_workers: list = []
        # Validation shares the generator prefix cache with training. Never reuse a
        # group id within one process, so salt-prefix mode cannot hit blocks created
        # by an earlier validation pass under older weights.
        self._next_validation_group_id = -1
        # Resume step (0 = fresh); set in setup_async from the loaded checkpoint.
        self.start_step = 0
        # Live policy versions; re-seeded from start_step in run() and advanced by
        # the trainer loop. Initialized here because a validation pass reads the
        # trainer version to place its metrics, including the pre-training pass.
        self._trainer_policy_version = 0
        self._generator_policy_version = 0
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.renderer = config.renderer.build(tokenizer_path=config.hf_assets_path)

        # Carry the base seed and renderer stop tokens on the sampling config so
        # the generator reads them off each request; the rollouter offsets the
        # seed per sample. Avoids the generator depending on request_id format.
        self._sampling = replace(
            config.generator.sampling,
            seed=config.generator.debug.seed,
            stop_token_ids=list(self.renderer.get_stop_token_ids()),
        )
        # TODO: pass our own tokenizer to the renderer and read pad/eos off it
        # once `renderers` supports bring-your-own-tokenizer
        # (https://github.com/PrimeIntellect-ai/renderers/pull/70).
        # Until then, reach into the renderer's tokenizer for the pad id (eos doubles as pad).
        self._rollouter: Rollouter = config.rollouter.build()
        self.rollout_recorder = config.rollout_recorder.build(
            dump_dir=config.dump_folder
        )
        self.validation_trace_recorder: ValidationTraceRecorder = (
            config.async_loop.validation.trace.build(dump_dir=config.dump_folder)
        )

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")

        # A crash can leave a background validation pass running against actors we
        # are about to stop; drop it before touching them.
        if self._validation_task is not None and not self._validation_task.done():
            self._validation_task.cancel()
            await asyncio.gather(self._validation_task, return_exceptions=True)

        if self.trainer is not None:
            try:
                await self.trainer.close.call()
            except Exception:
                logger.exception("trainer.close failed")

        for role, router in (
            ("generator", self.generator_router),
            ("eval_generator", self.eval_generator_router),
        ):
            if router is None:
                continue
            close_results = await router.fanout("close", return_exceptions=True)
            for idx, result in enumerate(close_results):
                if isinstance(result, BaseException):
                    actor_name = role if len(close_results) == 1 else f"{role}[{idx}]"
                    logger.error(
                        "%s.close failed",
                        actor_name,
                        exc_info=(type(result), result, result.__traceback__),
                    )

        try:
            self.metrics_processor.close()
        except Exception:
            logger.exception("metrics_processor close failed")

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _get_rank_0_value(self, result):
        """Extract rank 0 result from a Monarch ValueMesh.

        Monarch actor endpoints return results from all ranks in the mesh.
        This method picks out rank 0's result. This should be used in cases
        where all ranks return the same result.
        """
        return result.get(0)

    def _allocate_validation_group_ids(self, num_groups: int) -> list[int]:
        """Return negative group ids that are unique for this controller process."""
        if num_groups < 0:
            raise ValueError(f"num_groups must be non-negative, got {num_groups}")
        group_ids = list(
            range(
                self._next_validation_group_id,
                self._next_validation_group_id - num_groups,
                -1,
            )
        )
        self._next_validation_group_id -= num_groups
        return group_ids

    def _make_generate_fn(
        self, metrics_prefix: str, *, use_eval_generators: bool = False
    ) -> GenerateFn:
        """Build the rollouter's `GenerateFn`: route a completion via the generator router, namespacing
        generation metrics with `metrics_prefix` and pinning sticky routing on `routing_session_id` (a sample's
        turns reuse one generator's prefix KV).

        ``use_eval_generators`` routes to the validation-only generators when the run
        has them; without them it falls back to the training router (today's behavior).
        """
        # TODO: make this a pluggable config (a GenerateFn factory) so non-router generate backends can be swapped in.
        router = (
            self.eval_generator_router
            if use_eval_generators and self.eval_generator_router is not None
            else self.generator_router
        )

        @sl.log_trace_span("generate")
        async def generate(
            prompt_token_ids: list[int],
            *,
            request_id: str,
            routing_session_id: str | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> Completion | None:
            result = await router.route(
                "generate",
                prompt_token_ids,
                request_id=request_id,
                # VLLMGenerator.generate also requires this field for its
                # intra-mesh DP routing.
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
                metrics_prefix=metrics_prefix,
                # Load is measured as in-flight request count (one unit per call).
                routing_ctx=RoutingContext(
                    estimated_cost=1,
                    session_id=routing_session_id,
                ),
            )
            return self._get_rank_0_value(result)

        return generate

    @sl.log_trace_span("setup_async")
    async def setup_async(
        self,
        *,
        trainer_mesh: ProcMesh,
        generator_meshes: list[ProcMesh],
        eval_generator_meshes: list[ProcMesh] | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Kept separate from ``__init__`` because actor spawning, torch
        elastic env setup, TorchStore initialization, and the initial
        weight push/pull are all ``await``-based runtime side effects
        that cannot run in a synchronous constructor.

        The trainer and generator meshes are provisioned by the caller
        (see ``create_proc_mesh``) on disjoint GPUs; this method only
        spawns the actors on them and synchronizes initial weights from
        trainer to generator. Must be called before :meth:`run`.

        Args:
            trainer_mesh: ProcMesh the trainer actor is spawned on.
            generator_meshes: ProcMesh objects the generator actors are spawned on.
            eval_generator_meshes: ProcMesh objects for validation-only generators
                (``num_eval_generators``). They get their own router, so training
                rollouts are never routed to them.
        """
        eval_generator_meshes = eval_generator_meshes or []
        if len(eval_generator_meshes) != self.config.num_eval_generators:
            raise ValueError(
                f"expected {self.config.num_eval_generators} eval generator mesh(es), "
                f"got {len(eval_generator_meshes)}"
            )
        # Peak concurrent rollout sequences (buffer groups * group_size, or the
        # validation pass); sizes max_num_seqs below. Uses the resolved buffer size
        # (the max rollouts on the fly): max_num_seqs is a CEILING, not a KV
        # reservation -- vLLM pages KV on demand and admits fewer / preempts when KV
        # is tight, so a larger ceiling matches the run-ahead pool without OOM.
        async_loop = self.config.async_loop
        validation = async_loop.validation
        num_validation_rollouts = validation.num_samples * validation.group_size
        rollout_concurrency = max(
            async_loop.resolved_max_active_rollout_groups() * async_loop.group_size,
            # A dedicated eval generator sizes itself off the validation pass alone.
            0 if eval_generator_meshes else num_validation_rollouts,
        )
        # Renderer thread pool: render work is CPU-bound, so size to CPU count (decoupled from rollout concurrency).
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=os.cpu_count())
        )

        config = self.config
        if not generator_meshes:
            raise ValueError("setup_async requires at least one generator mesh")
        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        generator_dp_degree = max(config.generator.parallelism.data_parallel_degree, 1)
        num_generator_dp_shards = len(generator_meshes) * generator_dp_degree

        # Ceiling (not target) for the generator's max_num_seqs: the per-generator
        # upper bound on concurrently scheduled sequences. vLLM may admit fewer if KV
        # is tight; this also sets CUDA-graph capture sizes. An explicit override
        # (generator_max_num_seqs) removes the derived per-engine cap.
        max_num_seqs = async_loop.generator_max_num_seqs or min(
            math.ceil(rollout_concurrency / num_generator_dp_shards), 512
        )

        logger.info(
            "max_num_seqs=%d per generator (rollout_concurrency=%d / generator_dp_shards=%d)",
            max_num_seqs,
            rollout_concurrency,
            num_generator_dp_shards,
        )

        # TODO(observability): the mesh_spawn span wraps ~80 LoC of branching
        # provisioner logic. Pull a PerHostProvisioner.spawn_meshes(...) helper and
        # shrink this span to a single call.
        with sl.log_trace_span("mesh_spawn"):
            # Store proc meshes for cleanup
            self._proc_meshes = [
                trainer_mesh,
                *generator_meshes,
                *eval_generator_meshes,
            ]

            await setup_torch_elastic_env_async(trainer_mesh)
            for generator_mesh in (*generator_meshes, *eval_generator_meshes):
                await setup_torch_elastic_env_async(generator_mesh)

            # Spawn actors on their respective meshes
            self.trainer = trainer_mesh.spawn(
                "trainer",
                PolicyTrainer,
                config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
                output_dir=config.dump_folder,
            )

            # TODO: torch.compile with aot_eager backend (inductor crashes the vLLM engine on the shared model path).
            generators = []
            for idx, generator_mesh in enumerate(generator_meshes):
                actor_name = (
                    "generator" if len(generator_meshes) == 1 else f"generator_{idx}"
                )
                generator = generator_mesh.spawn(
                    actor_name,
                    VLLMGenerator,
                    config.generator,
                    model_spec=config.model_spec,
                    model_path=config.hf_assets_path,
                    compile_config=config.compile,
                    max_num_seqs=max_num_seqs,
                    output_dir=config.dump_folder,
                )
                generators.append(generator)
            self.generator_router = config.generator_router.build(generators=generators)

            # Validation-only generators. Same engine config, sized for the
            # validation pass instead of the rollout pool, on their own router.
            eval_generators = []
            for idx, eval_mesh in enumerate(eval_generator_meshes):
                actor_name = (
                    "eval_generator"
                    if len(eval_generator_meshes) == 1
                    else f"eval_generator_{idx}"
                )
                eval_generators.append(
                    eval_mesh.spawn(
                        actor_name,
                        VLLMGenerator,
                        config.generator,
                        model_spec=config.model_spec,
                        model_path=config.hf_assets_path,
                        compile_config=config.compile,
                        max_num_seqs=async_loop.generator_max_num_seqs
                        or min(
                            math.ceil(
                                num_validation_rollouts
                                / (len(eval_generator_meshes) * generator_dp_degree)
                            ),
                            512,
                        ),
                        output_dir=config.dump_folder,
                    )
                )
            if eval_generators:
                self.eval_generator_router = config.generator_router.build(
                    generators=eval_generators
                )

        # Spawn the CPU RolloutWorker pool on the controller host: each worker runs
        # group rollouts (agent orchestration + adapter + Daytona HTTP + grading) in
        # its own process, off the controller GIL. One single-proc CPU mesh per
        # worker (like the generators, so each is individually addressable). The
        # global rollout concurrency is split exactly across reachable worker lanes;
        # each worker sets its own share before its rollouter builds the sibling gate.
        requested_num_workers = config.num_rollout_workers
        if requested_num_workers < 0:
            raise ValueError(
                "num_rollout_workers must be non-negative, got "
                f"{requested_num_workers}"
            )
        if requested_num_workers > 0:
            global_conc = getattr(config.rollouter, "rollout_concurrency", None)
            if global_conc is None:
                global_conc = int(os.environ.get("SWE_ROLLOUT_CONCURRENCY", "16"))
            worker_concurrencies = _split_rollout_concurrency(
                global_conc,
                requested_num_workers,
                max_num_workers=config.async_loop.resolved_max_active_rollout_groups(),
            )
            if len(worker_concurrencies) < requested_num_workers:
                logger.warning(
                    "Capping rollout workers from %d to %d: each worker needs a "
                    "trajectory slot and a rollout-group lane (global concurrency=%d, "
                    "max active groups=%d)",
                    requested_num_workers,
                    len(worker_concurrencies),
                    global_conc,
                    config.async_loop.resolved_max_active_rollout_groups(),
                )
            with sl.log_trace_span("rollout_worker_spawn"):
                host = this_host()
                for worker_id, worker_conc in enumerate(worker_concurrencies):
                    worker_mesh = host.spawn_procs(per_host={"rollout_workers": 1})
                    self._proc_meshes.append(worker_mesh)
                    self._rollout_workers.append(
                        worker_mesh.spawn(
                            f"rollout_worker_{worker_id}",
                            RolloutWorker,
                            config,
                            rollout_concurrency=worker_conc,
                        )
                    )
                # Each worker builds its own generate-only router over the shared generators.
                await asyncio.gather(
                    *(w.setup.call(generators) for w in self._rollout_workers)
                )
            logger.info(
                "Spawned %d RolloutWorker(s) (requested %d), concurrency by "
                "worker=%s (global target %d)",
                len(worker_concurrencies),
                requested_num_workers,
                worker_concurrencies,
                global_conc,
            )

        # Validation RolloutWorker pool: same shape as the training pool but routed
        # only to the eval generators, so a pass never shares a GIL, a generator, or
        # a rollout-concurrency budget with training. Each worker's cap is its share
        # of the whole validation pass, since the pass runs its prompts at once.
        if config.num_eval_rollout_workers > 0:
            eval_concurrencies = _split_rollout_concurrency(
                max(
                    config.eval_rollout_concurrency or num_validation_rollouts,
                    config.num_eval_rollout_workers,
                ),
                config.num_eval_rollout_workers,
            )
            with sl.log_trace_span("eval_rollout_worker_spawn"):
                host = this_host()
                for worker_id, worker_conc in enumerate(eval_concurrencies):
                    worker_mesh = host.spawn_procs(per_host={"rollout_workers": 1})
                    self._proc_meshes.append(worker_mesh)
                    self._eval_rollout_workers.append(
                        worker_mesh.spawn(
                            f"eval_rollout_worker_{worker_id}",
                            RolloutWorker,
                            config,
                            rollout_concurrency=worker_conc,
                        )
                    )
                await asyncio.gather(
                    *(w.setup.call(eval_generators) for w in self._eval_rollout_workers)
                )
            logger.info(
                "Spawned %d eval RolloutWorker(s), concurrency by worker=%s "
                "(validation pass = %d rollouts)",
                len(eval_concurrencies),
                eval_concurrencies,
                num_validation_rollouts,
            )

        # Trainer placement is the fast path: one colocated volume per trainer
        # rank. Controller placement keeps the volume outside detached worker
        # proc meshes when long-lived child actors are unreliable. The sharded
        # controller mode retains one volume per trainer rank so transfers do
        # not serialize through the singleton controller volume.
        with sl.log_trace_span("torchstore_init"):
            if config.torchstore_volume_placement == "controller":
                await ts.initialize()
            elif config.torchstore_volume_placement == "controller_sharded":
                num_storage_volumes = (
                    trainer_parallelism.data_parallel_replicate_degree
                    * dp_shard
                    * trainer_parallelism.tensor_parallel_degree
                    * trainer_parallelism.pipeline_parallel_degree
                    * trainer_parallelism.context_parallel_degree
                )
                volume_mesh = this_host().spawn_procs(
                    per_host={"torchstore_volumes": num_storage_volumes}
                )
                self._proc_meshes.append(volume_mesh)
                await ts.initialize(
                    num_storage_volumes=num_storage_volumes,
                    mesh=volume_mesh,
                    strategy=ts.LocalRankStrategy(),
                )
            else:
                await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Resume: __init__ ran CheckpointManager.load(); read back the restored policy_version
        # (0 if fresh) so the loop resumes at the right step and generators pull at that version.
        # TODO(resume): only model/optimizer/policy_version are restored. The active-slot rollout
        #   buffer (in-flight rollouts) and the dataset stream position are NOT restored -- a resumed
        #   run refills the buffer and re-reads data from the start. Need to recycle prompts.
        self.start_step = self._get_rank_0_value(
            await self.trainer.get_policy_version.call()
        )
        if self.start_step > 0:
            logger.info(f"Resuming RL training from step {self.start_step}")

        # Initial weight sync: only the trainer loads weights; generators pull at start_step.
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self._pull_generator_weights(policy_version=self.start_step)

    async def _guard_eval_generators(self, awaitable, *, what: str) -> bool:
        """Await an eval-generator call, converting a failure into a disabled evaluator.

        Validation is observability: it must not be able to end a training run. On
        failure this drops the eval router, so later steps skip validation entirely
        and training continues with its own generators untouched.

        Returns:
            Whether the call succeeded.
        """
        try:
            await awaitable
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "%s failed; disabling further validation for this run "
                "(training continues)",
                what,
            )
            self.eval_generator_router = None
            self._eval_rollout_workers = []
            return False

    async def _pull_generator_weights(
        self, *, policy_version: int, include_eval: bool = True
    ) -> None:
        """Pull the just-pushed weights into the training generators, and into the
        eval generators unless a validation pass is using their current weights.

        The eval pull has to happen inside the trainer's weight-sync window, while
        ``policy_version`` is still the version in TorchStore. Pulling on every idle
        step (not only evaluated ones) keeps the eval generators' lifecycle identical
        to a training generator's; once a pass starts, ``include_eval`` goes False so
        it keeps scoring one frozen version while training moves on.
        """
        training_pull = self.generator_router.pull_model_state_dict(
            policy_version=policy_version
        )
        if not (include_eval and self.eval_generator_router is not None):
            await training_pull
            return
        # Concurrent: the eval pull must not add its latency to the training step.
        await asyncio.gather(
            training_pull,
            self._guard_eval_generators(
                self.eval_generator_router.pull_model_state_dict(
                    policy_version=policy_version
                ),
                what=f"eval generator weight pull at version {policy_version}",
            ),
        )

    # TODO: fold validation into a Validator(Configurable) the controller attaches, instead of 4 methods.
    async def _run_validation_group(
        self,
        *,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
        worker,
    ) -> RolloutGroup:
        """Run one validation prompt group, on an eval worker process when there is one."""
        if worker is not None:
            # metrics_prefix=None: the controller aggregates validation metrics from
            # the returned rollouts, so the worker must not also emit "rollout/*".
            return self._get_rank_0_value(
                await worker.run_group.call(
                    sample=sample,
                    group_id=group_id,
                    group_size=group_size,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    metrics_prefix=None,
                )
            )
        return await self._rollouter.run_group_rollouts(
            generate_fn=self._make_generate_fn(
                metrics_prefix="validation_generator", use_eval_generators=True
            ),
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
            renderer=self.renderer,
        )

    @sl.log_trace_span("_collect_validation_rollouts")
    async def _collect_validation_rollouts(
        self, *, num_groups: int, group_size: int, sampling: SamplingConfig, step: int
    ) -> tuple[list[object], list[RolloutGroup], list[m.Metric]]:
        """Run ``num_groups`` held-out prompts x ``group_size`` trials concurrently.

        Returns the surviving prompts, their scored groups (index-aligned), and the
        validation metrics.
        """
        # TODO(naming): reserve "sample" for TrainingSample; rename the rollouter's raw-prompt "sample" -> "prompt"/"data_input".
        samples = [self._rollouter.get_validation_sample() for _ in range(num_groups)]
        # Negative, process-unique ids keep validation disjoint from training and
        # from prior validation prefix-cache salts.
        group_ids = self._allocate_validation_group_ids(num_groups)
        num_workers = len(self._eval_rollout_workers)
        group_results = await asyncio.gather(
            *(
                self._run_validation_group(
                    sample=sample,
                    group_id=group_id,
                    group_size=group_size,
                    sampling=sampling,
                    worker=(
                        self._eval_rollout_workers[index % num_workers]
                        if num_workers
                        else None
                    ),
                )
                for index, (sample, group_id) in enumerate(
                    zip(samples, group_ids, strict=True)
                )
            ),
            return_exceptions=True,
        )

        # Keep the groups that succeeded; log + count the ones that raised.
        kept_samples: list[object] = []
        rollout_groups: list[RolloutGroup] = []
        num_failed_groups = 0
        for sample, group_id, result in zip(
            samples, group_ids, group_results, strict=True
        ):
            if isinstance(result, BaseException):
                logger.error(
                    f"validation group {group_id} (step={step}) failed; dropping",
                    exc_info=(type(result), result, result.__traceback__),
                )
                num_failed_groups += 1
                continue
            kept_samples.append(sample)
            rollout_groups.append(result)

        rollouts = [rollout for group in rollout_groups for rollout in group.rollouts]
        metrics = compute_rollout_metrics(prefix="validation", rollouts=rollouts)
        # Re-key the rollouter's own group metrics (e.g. tmax nonsubmit_frac,
        # finish-reason split) into the validation namespace, so the eval curve
        # carries the same diagnostics as the training curve without colliding.
        metrics.extend(
            replace(metric, key=metric.key.replace("rollout/", "validation/", 1))
            for group in rollout_groups
            for metric in group.metrics
            if metric.key.startswith("rollout/")
        )
        # pass@k: fraction of prompts solved by at least one trial. With
        # group_size=1 this equals the mean reward; with k > 1 it is the headline
        # benchmark number alongside validation/reward/mean (= avg@k).
        if group_size > 1:
            metrics.extend(
                m.Metric(
                    "validation/pass_at_k",
                    m.Mean(
                        1.0
                        if any(
                            rollout.reward is not None and rollout.reward > 0
                            for rollout in group.rollouts
                        )
                        else 0.0
                    ),
                )
                for group in rollout_groups
            )
        metrics.append(
            m.Metric("validation/group_failures", m.Sum(float(num_failed_groups)))
        )
        return kept_samples, rollout_groups, metrics

    # TODO: we currently determine validation.num_samples
    # but what if i want to run the entire dataset?
    @sl.log_trace_span("validate")
    async def validate(self, *, step: int) -> list[m.Metric]:
        """Run one validation pass over held-out prompts.

        Args:
            step: Policy version this pass scores (0 for the pre-training pass);
                tagged into logged rollout samples and the trace report directory.

        Returns:
            Validation rollout metrics, generation metrics, and validation
            timing.
        """
        t_validate_start = time.perf_counter()
        validation = self.config.async_loop.validation
        if validation.num_samples == 0:  # skip validation (e.g. loss guard CI)
            return []
        if self.config.num_eval_generators > 0 and self.eval_generator_router is None:
            # An earlier eval-generator failure disabled the evaluator; training
            # keeps running, but there is nothing left to score on.
            logger.warning(f"step {step}: eval generators unavailable; skipping")
            return [m.Metric("validation/skipped", m.Sum(1.0))]
        sampling = replace(
            self._sampling,
            temperature=validation.temperature,
            top_p=validation.top_p,
        )

        samples, rollout_groups, metrics = await self._collect_validation_rollouts(
            num_groups=validation.num_samples,
            group_size=validation.group_size,
            sampling=sampling,
            step=step,
        )

        self.rollout_recorder.record(is_validation=True, rollout_groups=rollout_groups)
        summary = self._record_validation_traces(
            step=step, samples=samples, rollout_groups=rollout_groups
        )
        if summary is not None:
            metrics.append(
                m.Metric("validation/trace_pass_at_k", m.NoReduce(summary.pass_at_k))
            )

        metrics.append(m.Metric("validation/policy_version", m.NoReduce(float(step))))
        t_validate_s = time.perf_counter() - t_validate_start
        metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return metrics

    def _record_validation_traces(
        self, *, step: int, samples: list[object], rollout_groups: list[RolloutGroup]
    ) -> EvalSummary | None:
        """Write the browsable per-pass trace report; never fail the pass over it."""
        if not self.validation_trace_recorder.enabled:
            return None
        try:
            return self.validation_trace_recorder.record(
                policy_version=step,
                groups=rollout_groups,
                task_ids=[
                    _task_id(sample, index) for index, sample in enumerate(samples)
                ],
                decode=self.renderer._tokenizer.decode,
            )
        except Exception:
            logger.exception("validation trace report failed at step %d", step)
            return None

    async def run(self) -> None:
        """Start every async loop and run until training completes or a stage crashes.

        Producers (_data_input_loop, _rollout_loop[N], _batcher_loop) loop forever; _trainer_loop is the
        only finite loop -- it runs num_training_steps, then returns, which drives shutdown.

        Shutdown (healthy):  _trainer_loop finishes N steps -> run() finally ->
          group_buffer.close()  (wakes _data_input_loop / _rollout_loop / _batcher_loop blocked on the buffer)
          -> task.cancel()      (wakes anything blocked on training_batch_queue.put/get; close does NOT wake these)
          -> gather(..., return_exceptions=True)
        Shutdown (crash):    any loop raises -> appears in `done` -> run() re-raises -> same finally.
        """
        async_loop = self.config.async_loop
        num_training_steps = async_loop.num_training_steps
        validation = async_loop.validation
        # A blocking pre-training pass leaves the trainer and every training
        # generator idle for its whole duration. That is fine for a short held-out
        # pass, but a full benchmark sweep can take over an hour, and a generator
        # mesh that receives no traffic that long gets reaped -- the next generate
        # then fails with a gloo "connection closed by peer". So when the pass is
        # async, start training first and run the pre-training pass alongside it,
        # on the eval generators that already hold the start-step weights.
        run_pre_validation_async = validation.run_async and num_training_steps > 0
        logger.info(
            f"Running pre-training validation ({'async' if run_pre_validation_async else 'blocking'}); "
            f"then {num_training_steps} steps of async RL training"
        )

        if not run_pre_validation_async:
            sl.log_trace_instant("validation_start")
            await self._validate_and_log(step=self.start_step)
            if num_training_steps == 0:
                return
        sl.log_trace_instant("training_start")

        # Two policy version pointers, seeded from the resumed step: the trainer advances at the
        # optimizer step; the generator version advances when a weight pull completes.
        self._trainer_policy_version = self.start_step
        self._generator_policy_version = self.start_step

        # Buffer capacity = run-ahead depth (may exceed the staleness window when
        # max_active_rollout_groups is set; staleness is still bounded by the batcher's
        # stale-drop at max_offpolicy_steps).
        max_active_rollout_groups = async_loop.resolved_max_active_rollout_groups()
        initial_active_rollout_groups = (
            async_loop.resolved_initial_active_rollout_groups()
        )
        logger.info(
            "Active rollout-group capacity: initial=%d, max=%d",
            initial_active_rollout_groups,
            max_active_rollout_groups,
        )

        self._group_buffer = async_loop.group_buffer.build(
            max_active_rollout_groups=max_active_rollout_groups,
            initial_active_rollout_groups=initial_active_rollout_groups,
        )

        # training_sample_builder
        training_sample_builder = async_loop.training_sample_builder.build()

        # batcher
        batcher = async_loop.batcher.build(
            num_groups_per_train_step=async_loop.num_groups_per_train_step,
            dp_degree=self.trainer_dp_degree,
            pad_id=self.renderer._tokenizer.eos_token_id,
            initial_policy_version=self.start_step,
        )

        # training_batch_queue
        training_batch_queue: asyncio.Queue[TrainingBatch | None] = asyncio.Queue(
            maxsize=1
        )

        # rollout_loop
        generate_fn = self._make_generate_fn(metrics_prefix="generator")

        # One rollout loop task per full-capacity buffer slot. The data-input loop
        # initially fills only initial_active_rollout_groups when cold-start
        # admission is configured, then uses the remaining tasks as retained
        # trainable groups open the downstream headroom.
        # With a RolloutWorker pool, each loop task is pinned to a worker round-robin.
        # The tasks all claim from one shared buffer, so a group is owned by whichever
        # lane claims it, not by group_id modulo num_workers. Without a pool
        # (num_rollout_workers=0), worker=None keeps the in-process path.
        num_workers = len(self._rollout_workers)
        rollout_tasks = [
            asyncio.create_task(
                self._rollout_loop(
                    group_buffer=self._group_buffer,
                    generate_fn=generate_fn,
                    worker=(
                        self._rollout_workers[group_worker_id % num_workers]
                        if num_workers
                        else None
                    ),
                ),
                name=f"rollout_worker_{group_worker_id}",
            )
            for group_worker_id in range(max_active_rollout_groups)
        ]

        # data_input_loop
        data_input_task = asyncio.create_task(
            self._data_input_loop(self._group_buffer), name="data_input"
        )

        # training_sample_batcher_loop
        batcher_task = asyncio.create_task(
            self._batcher_loop(
                group_buffer=self._group_buffer,
                training_sample_builder=training_sample_builder,
                batcher=batcher,
                training_batch_queue=training_batch_queue,
            ),
            name="batcher",
        )

        # trainer_loop
        trainer_task = asyncio.create_task(
            self._trainer_loop(
                training_batch_queue, num_training_steps=num_training_steps
            ),
            name="trainer",
        )

        # Pre-training validation, now that the training loops own the training
        # generators. The eval generators still hold the start-step weights from
        # setup_async, so this pass scores the starting policy.
        if run_pre_validation_async:
            sl.log_trace_instant("validation_start")
            self._start_async_validation(step=self.start_step)

        # run everything until trainer finishes its number of steps
        # or some other loop breaks
        background_tasks = [
            data_input_task,
            *rollout_tasks,
            batcher_task,
        ]
        try:
            done, _ = await asyncio.wait(
                [trainer_task, *background_tasks], return_when=asyncio.FIRST_COMPLETED
            )
            # The trainer is the finite clock: it runs num_training_steps then returns -> training is done.
            # Producers loop forever, so a producer in `done` means it crashed (await re-raises) or wrongly
            # returned cleanly (the RuntimeError). Check producers even when the trainer also finished this
            # wakeup, so a simultaneous producer crash isn't hidden behind the finished trainer.
            for task in done:
                if task is trainer_task:
                    continue
                await task  # raises if task crashed; returns if task ended cleanly
                raise RuntimeError(f"{task.get_name()} exited unexpectedly")
            if trainer_task in done:
                await trainer_task
        finally:
            # Graceful first: buffer.close() (awaited) wakes loops blocked on the buffer so they return.
            # Then cancel covers anything blocked on the queue (which close does not wake); gather awaits all.
            await self._group_buffer.close()
            for task in (*background_tasks, trainer_task):
                task.cancel()
            await asyncio.gather(
                *background_tasks, trainer_task, return_exceptions=True
            )

        # Drain any background pass first: it holds the eval generators, and its
        # curve point would otherwise be lost when the controller exits.
        await self._await_pending_validation()
        # Post-training validation (held-out eval after the final step). Pull the
        # eval generators here rather than relying on the last step's sync: that
        # sync is skipped while a background pass is in flight, so the eval
        # generators can still be holding an older version. TorchStore holds the
        # final push, so this pull is the final policy either way.
        if self.eval_generator_router is not None:
            with sl.log_trace_span("eval_generator_pull_model_state_dict"):
                await self.eval_generator_router.pull_model_state_dict(
                    policy_version=self._trainer_policy_version
                )
        # Label the pass with the version actually reached, not the requested step
        # count: the loop also exits early when the batcher drains.
        post_validation = await self._validate_and_log(
            step=self._trainer_policy_version
        )
        self._log_reward_delta(
            self._validation_results.get(self.start_step, {}), post_validation
        )

    async def _validate_and_log(self, *, step: int) -> dict[str, float]:
        """Run one validation pass, log it, and return its aggregated values for the pre/post delta.

        Metrics are logged at ``self._trainer_policy_version`` (the live training
        step), not at ``step``. For a blocking pass those are the same; for a
        background pass the trainer has already moved on, and W&B drops anything
        logged at an earlier step than the last committed one. The evaluated policy
        version is carried in ``validation/policy_version`` and in the trace report's
        directory name instead.
        """
        metrics = await self.validate(step=step)
        if metrics:
            self.metrics_processor.log(
                step=max(step, self._trainer_policy_version),
                metrics=metrics,
                is_validation=True,
                commit=True,
            )
        aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
        self._validation_results[step] = aggregated
        return aggregated

    def _validation_pass_in_flight(self) -> bool:
        """True while a background validation pass is still running.

        Gates BOTH the eval-generator weight pull and the launch of the next pass:
        pulling mid-pass would swap the weights underneath it, so the result would
        no longer belong to the policy version it is labelled with.
        """
        return self._validation_task is not None and not self._validation_task.done()

    def _start_async_validation(self, *, step: int) -> None:
        """Launch a background validation pass over the weights just pulled at ``step``.

        A pass still running when the next interval comes around is skipped rather
        than queued: queueing would stack passes behind a slow benchmark forever, and
        the newer policy version is the one worth measuring. The skip is counted so a
        thin eval curve is visible rather than silent.
        """
        if self._validation_pass_in_flight():
            logger.warning(
                "step %d: previous validation pass still running; skipping this one. "
                "Raise validation.interval or add eval capacity.",
                step,
            )
            self.metrics_processor.log(
                step=step,
                metrics=[m.Metric("validation/skipped", m.Sum(1.0))],
                is_validation=True,
                commit=True,
            )
            return

        async def _run() -> None:
            try:
                await self._validate_and_log(step=step)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("background validation at step %d failed", step)

        self._validation_task = asyncio.create_task(
            _run(), name=f"validation_step_{step}"
        )

    async def _await_pending_validation(self) -> None:
        """Wait for an in-flight background validation pass before shutting down."""
        task = self._validation_task
        if task is None or task.done():
            return
        logger.info("Waiting for the in-flight background validation pass to finish")
        await task

    def _log_reward_delta(self, pre: dict[str, float], post: dict[str, float]) -> None:
        """Console pre/post reward summary, visible without scrolling back through the loop."""
        reward_keys = sorted(key for key in set(pre) | set(post) if "reward" in key)
        logger.info("=" * 60)
        logger.info("Validation reward (pre / post):")
        for key in reward_keys:
            logger.info(
                f"  {key}:  {pre.get(key, float('nan')):+.3f}  /  {post.get(key, float('nan')):+.3f}"
            )
        logger.info("=" * 60)

    async def _data_input_loop(self, group_buffer: RolloutGroupWorkBuffer) -> None:
        """produces a RolloutGroupWork into group_buffer.
        waits for:    a free active slot (group_buffer.wait_for_slot)
        unblocked by: _trainer_loop release_active_groups(num_groups_per_train_step, "trained")
            after the pull (and _batcher_loop release_active_groups(1,"untrainable_group"))

        Separate from `_rollout_loop`, so slow data prep (e.g. on-the-fly question generation) overlaps
        generation instead of serializing in front of it.
        """
        # TODO(resume): persist dataset position so a restarted job continues the data stream, not from scratch.
        group_index = 0

        # TODO(perf): Slots are current released in batches, while this loop is a single producer.
        # we could a) increase the number of threads; b) revisit how we release slots and see if
        # we can release them on the batcher while still preserving max off-policy steps.
        # finally, c) we need to check how will this data input loop truly overlaps with the rollout loop.
        while await group_buffer.wait_for_slot():
            with sl.log_trace_span("get_training_sample"):
                # to_thread: Dont block on dataset reads
                sample = await asyncio.to_thread(self._rollouter.get_training_sample)
            await group_buffer.add_work(
                RolloutGroupWork(
                    group_id=group_index,
                    sample=sample,
                )
            )
            group_index += 1
        logger.info("Buffer closed; data input loop stopping")

    async def _rollout_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        generate_fn: GenerateFn,
        worker=None,
    ) -> None:
        """Generate + score one group at a time; a failed group becomes an empty group + a failure metric.

        When ``worker`` is a RolloutWorker actor, the group is run in that worker
        process (off the controller GIL) and returned; otherwise it runs in-process
        via ``self._rollouter``. Either way the controller records the raw group and
        finalizes it into the buffer.

        Staleness is bounded by the buffer's active-slot budget. Raw rollouts are recorded before any drop,
        so dropped groups stay inspectable on disk.

        consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
            waits for:    a claimable WAITING entry
            unblocked by: _data_input_loop group_buffer.add_work()
        produces: RolloutGroup (group_buffer.finalize_work)
            waits for:    nothing (admits its own claimed slot)
            unblocked by: n/a
        """
        while True:
            work = await group_buffer.claim_next()
            if work is None:  # group_buffer closed/shutdown signal
                logger.info("Buffer closed; rollout worker stopping")
                return
            try:
                with sl.log_trace_span("rollout_group"):
                    if worker is not None:
                        # Dispatch to a RolloutWorker process; it runs run_group_rollouts
                        # + compute_rollout_metrics and returns the finalized group.
                        group = self._get_rank_0_value(
                            await worker.run_group.call(
                                sample=work.sample, group_id=work.group_id
                            )
                        )
                    else:
                        group = await self._rollouter.run_group_rollouts(
                            generate_fn=generate_fn,
                            sample=work.sample,
                            group_id=work.group_id,
                            group_size=self.config.async_loop.group_size,
                            sampling=self._sampling,
                            renderer=self.renderer,
                        )
                        # Preserve rollouter-set group metrics (e.g. tmax
                        # nonsubmit_frac / format_errors); append the standard
                        # computed ones instead of overwriting them.
                        group.metrics = compute_rollout_metrics(
                            prefix="rollout", rollouts=group.rollouts
                        ) + list(group.metrics)

                # save rollout for inspection
                self.rollout_recorder.record(
                    is_validation=False,
                    rollout_groups=[group],
                )
            except Exception:
                logger.exception(f"rollout group {work.group_id} failed; dropping")
                group = RolloutGroup(
                    group_id=work.group_id,
                    rollouts=[],
                    metrics=[m.Metric("rollout/group_failures", m.Sum(1.0))],
                )
            await group_buffer.finalize_work(group)

    async def _batcher_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        training_sample_builder: TrainingSampleBuilder,
        batcher: Batcher,
        training_batch_queue: "asyncio.Queue[TrainingBatch | None]",
    ) -> None:
        """Take finalized groups, build training_samples, accumulate them, and queue each ready training batch.

        On a clean close/shutdown the group_buffer drains and returns None; we forward a `None` sentinel
        so the trainer stops.

        consumes: the oldest-admitted group that is FINALIZED (group_buffer.take_finalized)
            waits for:    any active group becoming FINALIZED
            unblocked by: _rollout_loop[N] group_buffer.finalize_work()
        produces: TrainingBatch (training_batch_queue.put)
            waits for:    a free training_batch_queue slot (maxsize=1)
            unblocked by: _trainer_loop training_batch_queue.get()
        """
        while True:
            rollout_group = await group_buffer.take_finalized()
            if rollout_group is None:  # closed and drained
                logger.info("Buffer drained; batcher loop stopping")
                break

            # [buffer trace] classify this completed group's solve status for per-step
            # visibility. full-solve (all rollouts pass) and not-solve (all fail) groups
            # are zero-std -> dropped by drop_zero_std; only partial-solve groups train.
            _rw = [r.reward for r in rollout_group.rollouts if r.reward is not None]
            _n = len(_rw)
            _ns = sum(1 for x in _rw if x and x > 0.0)
            _cls = (
                "full_solve"
                if _n and _ns == _n
                else "not_solve"
                if _ns == 0
                else "partial_solve"
            )

            # Bind freshness to the policy version that will consume the batch, not
            # the trainer's version at arrival. This is the analogue of Open-Instruct's
            # max_result_age_steps check against the batch's reserved training step.
            #
            # Check the raw group before TrainingSampleBuilder computes zero-std and
            # reward metrics. Otherwise a metric-only group can be fresh on arrival,
            # age while waiting for siblings, and leak stale reward into a later step.
            group_min_policy_version = min(
                (
                    turn.min_policy_version
                    for rollout in rollout_group.rollouts
                    for turn in rollout.turns
                    if turn.min_policy_version is not None
                ),
                default=None,
            )
            target_policy_version = batcher.next_batch_policy_version
            if group_min_policy_version is not None:
                group_age = target_policy_version - group_min_policy_version
                if _should_drop_group_at_batcher(
                    group_age=group_age,
                    max_offpolicy_steps=self.config.async_loop.max_offpolicy_steps,
                ):
                    logger.info(
                        "[buffer] complete group_id=%d solved=%d/%d class=%s "
                        "-> RELEASE(stale_dropped) target_ver=%d cur_ver=%d",
                        rollout_group.group_id,
                        _ns,
                        _n,
                        _cls,
                        target_policy_version,
                        self._trainer_policy_version,
                    )
                    await group_buffer.release_active_groups(1, reason="stale_dropped")
                    sl.log_trace_scalar({"rollout_buffer/dropped/stale": 1.0})
                    continue

            with sl.log_trace_span("training_sample_builder"):
                training_sample_group = training_sample_builder.build_from_group(
                    rollout_group=rollout_group
                )

            if not training_sample_group.training_samples:
                logger.info(
                    "[buffer] complete group_id=%d solved=%d/%d class=%s "
                    "-> RELEASE(zero_std, dropped) target_ver=%d cur_ver=%d",
                    rollout_group.group_id,
                    _ns,
                    _n,
                    _cls,
                    target_policy_version,
                    self._trainer_policy_version,
                )
                await group_buffer.release_active_groups(1, reason="untrainable_group")
            else:
                # Open one cold-start headroom slot for the retained group. The
                # group remains charged through trainer consumption and weight
                # pull, while its replacement can enter generation immediately.
                # Once the configured full capacity is reached this is a no-op.
                await group_buffer.grow_effective_capacity()
                logger.info(
                    "[buffer] complete group_id=%d solved=%d/%d class=%s "
                    "-> TRAINABLE target_ver=%d cur_ver=%d",
                    rollout_group.group_id,
                    _ns,
                    _n,
                    _cls,
                    target_policy_version,
                    self._trainer_policy_version,
                )

            # We put a group in. We may get a batch back
            # if there are enough accumulated trainable groups to return one.
            with sl.log_trace_span("batcher_pack"):
                maybe_training_batch = await asyncio.to_thread(
                    batcher.add_training_samples,
                    training_sample_group=training_sample_group,
                )
            if maybe_training_batch is not None:
                logger.info(
                    "[batcher_loop] packed a training batch "
                    f"for policy version {maybe_training_batch.target_policy_version} "
                    f"({len(maybe_training_batch.microbatches)} microbatch(es)); "
                    "putting on queue"
                )
                await training_batch_queue.put(maybe_training_batch)
                logger.info("[batcher_loop] batch enqueued")
        await training_batch_queue.put(None)
        # TODO(async-rl): if finite datasets are supported, drain a final partial batch here.

    async def _trainer_loop(
        self,
        training_batch_queue: "asyncio.Queue[TrainingBatch | None]",
        *,
        num_training_steps: int,
    ) -> None:
        """Run num_training_steps optimizer steps: train one packed batch, publish trainer weights,
        then pull them into generators, log metrics.

        consumes: a TrainingBatch (training_batch_queue.get)
            waits for:    a TrainingBatch in the queue
            unblocked by: _batcher_loop training_batch_queue.put()
        """
        validation = self.config.async_loop.validation
        for step in range(self.start_step + 1, num_training_steps + 1):
            sl.set_step(step)  # propagate the step counter to the actors
            will_validate = _will_validate_after_step(
                validation=validation, step=step, num_training_steps=num_training_steps
            )
            # Phase logging: the training step is otherwise a black box until its
            # end-of-step metrics flush, so a hang inside it is invisible. Log each
            # actor-call boundary so a stall is localized in the stdout timeline.
            logger.info(f"[trainer_loop] step {step}: begin; trainer.sync_log_step")
            await self.trainer.sync_log_step.call(step)
            logger.info(f"[trainer_loop] step {step}: generator fanout sync_log_step")
            await self.generator_router.fanout("sync_log_step", step)
            # The eval generators get the same per-step ping. They are otherwise
            # idle between passes (tens of minutes at a realistic interval), and an
            # idle generator mesh has been seen to lose its worker processes -- the
            # next call then fails with a gloo "connection closed by peer". The
            # training generators, which receive this every step, do not.
            if self.eval_generator_router is not None:
                await self._guard_eval_generators(
                    self.eval_generator_router.fanout("sync_log_step", step),
                    what=f"eval generator sync_log_step at step {step}",
                )
            logger.info(f"[trainer_loop] step {step}: awaiting training batch")
            step_timer = MetricsTimer()

            with sl.log_trace_span("train_step"), step_timer.record(
                "timing/step/total"
            ):
                # Waits for a TrainingBatch to be ready (or None on shutdown).
                with sl.log_trace_span("wait_for_training_batch"), step_timer.record(
                    "timing/step/wait_for_training_batch"
                ):
                    packed = await self._take_reserved_training_batch(
                        training_batch_queue
                    )

                if packed is None:
                    logger.info("Batcher closed and drained; stopping training")
                    break
                logger.info(
                    f"[trainer_loop] step {step}: got batch, "
                    f"{len(packed.microbatches)} microbatch(es), "
                    f"{packed.num_global_valid_tokens} valid tokens"
                )

                # Policy age is computed HERE, at consumption time, against the live trainer version, so it is
                # faithful to what this step trains on -- not the version when the batch was packed.
                policy_age_panel = compute_policy_age_metrics(
                    trainer_policy_version=self._trainer_policy_version,
                    min_policy_versions=packed.min_policy_versions,
                    max_offpolicy_steps=self.config.async_loop.max_offpolicy_steps,
                )

                # TODO(async): can't stream microbatches (interleave pack->train) — the loss is normalized by
                #   packed.num_global_valid_tokens (sum over ALL microbatches), needed before any fwd/bwd. To
                #   support streaming, accumulate raw loss/token counts across microbatches and scale before optim.
                with sl.log_trace_span("forward_backward"), step_timer.record(
                    "timing/step/forward_backward"
                ):
                    # fwd_bwd on all microbatches (logged per microbatch so a stall
                    # inside the first cross-host collective is visible in the timeline).
                    microbatch_metrics = []
                    num_microbatches = len(packed.microbatches)
                    for mb_idx, microbatch in enumerate(packed.microbatches):
                        logger.info(
                            f"[trainer_loop] step {step}: forward_backward "
                            f"microbatch {mb_idx + 1}/{num_microbatches}"
                        )
                        microbatch_metrics.append(
                            self._get_rank_0_value(
                                await self.trainer.forward_backward.call(
                                    microbatch, packed.num_global_valid_tokens
                                )
                            )
                        )

                    fwd_bwd_metrics = combine_microbatch_metrics(microbatch_metrics)
                    logger.info(
                        f"[trainer_loop] step {step}: forward_backward done, "
                        f"loss={fwd_bwd_metrics['loss/mean']:.4f}"
                    )

                    if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                        logger.error("Loss is NaN/Inf; training diverged")
                        break

                with sl.log_trace_span("optim_step"), step_timer.record(
                    "timing/step/optim"
                ):
                    optim_result = self._get_rank_0_value(
                        await self.trainer.optim_step.call()
                    )
                self._trainer_policy_version = optim_result.policy_version
                logger.info(f"[trainer_loop] step {step}: optim done")

                # Weight sync: publish new weights before the next train step (push then pull, both awaited).
                # TODO(perf): overlap weight sync (today both are awaited synchronously).
                with sl.log_trace_span(
                    "trainer_push_model_state_dict"
                ), step_timer.record("timing/step/push_model_state_dict"):
                    await self.trainer.push_model_state_dict.call()
                logger.info(f"[trainer_loop] step {step}: weights pushed")
                # The eval generators pull whenever they are idle, inside this
                # weight-sync window (the only point where TorchStore still holds
                # optim_result.policy_version). Pulling every idle step -- not only
                # evaluated ones -- keeps them exercising the same path as a training
                # generator instead of sitting untouched between passes. A pass in
                # flight keeps its weights: pulling now would swap them mid-pass, and
                # _start_async_validation skips this step anyway.
                with sl.log_trace_span(
                    "generator_pull_model_state_dict"
                ), step_timer.record("timing/step/pull_model_state_dict"):
                    await self._pull_generator_weights(
                        policy_version=optim_result.policy_version,
                        include_eval=not self._validation_pass_in_flight(),
                    )
                self._generator_policy_version = optim_result.policy_version
                logger.info(f"[trainer_loop] step {step}: weights pulled (step done)")

                # Release one train step's group slots after the pull; the batcher packs exactly
                # num_groups_per_train_step trainable groups per batch.
                logger.info(
                    "[buffer] step %d: RELEASE(trained) %d trainable group slots "
                    "(this step trained on %d partial-solve groups)",
                    step,
                    self.config.async_loop.num_groups_per_train_step,
                    self.config.async_loop.num_groups_per_train_step,
                )
                await self._group_buffer.release_active_groups(
                    self.config.async_loop.num_groups_per_train_step,
                    reason="trained",
                )

            # TODO(metrics): See if metrics are being computed at the right place. E.g. should we put all
            # rollout related metrics here, or move all of them to the rollouter.
            time_metrics = step_timer.flush()
            self.metrics_processor.log(
                step=step,
                is_validation=False,
                # An async pass logs at whatever step it finishes on, so it can never
                # close this step's W&B transaction -- commit here instead.
                commit=not (will_validate and not validation.run_async),
                metrics=[
                    *packed.metrics,
                    *[
                        m.Metric(key, m.NoReduce(value))
                        for key, value in fwd_bwd_metrics.items()
                    ],
                    *[
                        m.Metric(key, m.NoReduce(value))
                        for key, value in optim_result.metrics.items()
                    ],
                    *self._group_buffer.metrics(),
                    *time_metrics,
                    *policy_age_panel,
                    *compute_perf_ratio_metrics(
                        num_global_valid_tokens=packed.num_global_valid_tokens,
                        time_metrics=time_metrics,
                    ),
                ],
            )

            # Save full training state for resume; CheckpointManager writes only on its interval
            # and the final step. After the divergence guard so a NaN step isn't checkpointed.
            with sl.log_trace_span("trainer_save_checkpoint"):
                await self.trainer.save_checkpoint.call(
                    step, last_step=(step == num_training_steps)
                )

            # Periodic held-out validation. Skip the final step -- run() does a post-training
            # pass right after the loop. Runs on this step's just-pulled weights, overlapping
            # the background rollout collection (disjoint negative group ids), so the curve
            # tracks the live policy without a separate eval job or checkpoint download.
            if will_validate and step != num_training_steps:
                if validation.run_async:
                    self._start_async_validation(step=step)
                else:
                    await self._validate_and_log(step=step)

    async def _take_reserved_training_batch(
        self, training_batch_queue: "asyncio.Queue[TrainingBatch | None]"
    ) -> TrainingBatch | None:
        """Take the batch reserved for the live trainer policy and verify invariants."""
        max_offpolicy_steps = self.config.async_loop.max_offpolicy_steps
        packed = await training_batch_queue.get()
        if packed is None:
            return None

        if packed.target_policy_version != self._trainer_policy_version:
            raise RuntimeError(
                "training batch consumed by the wrong policy version: "
                f"target_policy_version={packed.target_policy_version}, "
                f"trainer_policy_version={self._trainer_policy_version}"
            )

        max_policy_age = max(
            (
                packed.target_policy_version - min_policy_version
                for min_policy_version in packed.min_policy_versions
            ),
            default=0,
        )
        if max_policy_age > max_offpolicy_steps:
            raise RuntimeError(
                "batcher admitted stale training data for its reserved policy version: "
                f"max_policy_age={max_policy_age}, "
                f"max_offpolicy_steps={max_offpolicy_steps}, "
                f"target_policy_version={packed.target_policy_version}"
            )
        return packed

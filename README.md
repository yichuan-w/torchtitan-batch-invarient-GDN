# Batch-invariant GDN: code for the train/inference mismatch blog post

This is the code snapshot behind the blog post

**[Defending Against the Training-Inference Numeric Mismatch in RL (Especially Linear
Attention) -- and Whether It Helps Async
RL](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)**

It is a fork of [pytorch/torchtitan](https://github.com/pytorch/torchtitan) plus the
work described in the post: one model definition shared by the trainer and the vLLM
generator, and batch-invariant (BI) kernels for Qwen3.5's Gated DeltaNet (GDN) linear
attention, so the trainer's recomputed logprob `pi` matches the generator's rollout
logprob `mu` **bit for bit** at step 0.

## Please read this first

**This repo is messy, and deliberately so.** It is a branch of a larger ongoing
agentic-RL research project, not a curated release. You will find half-finished
experiments, one-off diagnostic recipes, env-var switches with long comments,
debug docs, and TODOs that have nothing to do with this blog post. Nothing has been
tidied up for publication beyond removing internal paths. Please be understanding --
and read the "Where the blog's pieces live" table below rather than the file tree, or
you will spend a long time in code that is not part of the story.

Upstream torchtitan's own README is kept at
[`README_torchtitan_upstream.md`](./README_torchtitan_upstream.md).

## The result in one paragraph

Three arms, same data and recipe: (1) **vLLM native** -- trainer weights loaded into
vLLM's own model implementation, the standard two-engine setup; (2) **unified, no BI**
-- trainer and generator share one model definition but run stock kernels;
(3) **unified + BI** -- the same definition with batch-invariant kernels everywhere in
the forward. Arm 3 reaches `bit_wise/logprob_diff/abs_mean == 0` at step 0 on all
three workloads, so every nonzero value later is off-policy staleness rather than
precision noise. It costs 2-3x trainer throughput on math and search, ~5x on the
terminal agent, and it does **not** buy a clear accuracy win. The post argues for it
as a debugging tool rather than a production default.

## Install

Verified on H100 / CUDA 13.0, Python 3.12. See
[`torchtitan/experiments/rl/README.md`](./torchtitan/experiments/rl/README.md) for the
long-form version of steps 1-5.

```bash
# 0. env
pip install uv
uv venv --python 3.12 titan-rl
source titan-rl/bin/activate

# 1. RL loop deps: Monarch (controller), TorchStore (weight sync), renderers
uv pip install torchmonarch
uv pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install pygtrie portpicker
uv pip install "git+https://github.com/PrimeIntellect-ai/renderers.git@main"

# 2. PyTorch nightly + a matching vLLM wheel + torchcomms
uv pip install torch vllm torchcomms --pre \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --index-strategy unsafe-best-match

# 3. FlashAttention 3 (H100/H200 and newer; FA2 ships with PyTorch for A100)
uv pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130

# 4. batch-invariant kernels (REQUIRED for the BI arm): mm/addmm/log_softmax/mean
uv pip install --no-deps "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"

# 5. Qwen3.5 GDN kernels (fla) -- the chunked and recurrent gated-delta-rule kernels
uv pip install flash-linear-attention av torchvision

# 6. torchtitan itself
uv pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # Monarch-spawned workers inherit this
```

Then download a checkpoint and smoke-test the loop end to end (this is also how you
confirm the install; it needs one GPU pair, not the full topology):

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B \
  --local_dir torchtitan/experiments/rl/example_checkpoint --all

# plain async RL loop
python -m torchtitan.experiments.rl.train --module alphabet_sort \
  --config rl_grpo_qwen3_0_6b_varlen

# the same loop in batch-invariant mode (bit_wise/logprob_diff/max should be 0)
python -m torchtitan.experiments.rl.train --module alphabet_sort \
  --config rl_grpo_qwen3_0_6b_varlen_batch_invariant
```

Notes that cost real time if you miss them:

- The **vLLM wheel must match the PyTorch nightly build date** (both from the cu130
  nightly index). A mismatched pair fails at import or, worse, at the first
  custom-op call.
- `flash-linear-attention` is only needed for Qwen3.5 (GDN); dense Qwen3 does not use
  it. On Hopper, `fla`'s tilelang-backed backward may need `apache-tvm-ffi` +
  `tilelang` and a `CUDA_HOME` pointing at a CUDA 13 toolchain.
- W&B is on by default (`wandb login`, or pass `--metrics.no-enable-wandb`).

## Where the blog's pieces live

| Blog | Knob / code |
|---|---|
| §3.1 one model definition (unified) | `generator.backend="torchtitan_wrapper"`. The two-engine baseline is `backend="vllm_native"`. Wrapper: `torchtitan/experiments/rl/models/vllm_wrapper.py`; the GDN layer that borrows only vLLM's paged cache: `models/gdn_vllm_unified.py` (see [`GDN_UNIFIED_PREFILL.md`](./torchtitan/experiments/rl/models/GDN_UNIFIED_PREFILL.md)) |
| §3.2 recurrent forward / chunked backward (trainer) | `_RecurrentFwdChunkBwd` in `torchtitan/models/qwen3_5/model.py`, selected when `debug.batch_invariant` is on |
| §3.2 recurrent prefill **and** decode (generator) | `_forward_recurrent_bi` / `set_trainer_parity_fla` in `torchtitan/experiments/rl/models/gdn_vllm_unified.py`, enabled by `generator.gdn_trainer_parity=True` |
| §2 BI GEMM / RMSNorm / log_softmax, deterministic NCCL | `DebugConfig(batch_invariant=True, deterministic=True)` -> `set_batch_invariance()` in `torchtitan/distributed/utils.py`, backed by `batch_invariant_ops` |
| §2 attention `num_splits=1` (no split-KV) | `torchtitan/models/common/attention.py` (trainer) and `torchtitan/experiments/rl/models/attention.py` (generator) |
| MoE: batch-invariant `bmm` for the router gate | `patch_bmm_for_batch_invariance()` in `torchtitan/experiments/rl/batch_invariance.py` (also the flex-attention tile pinning and the small-M matmul tuning) |
| §4.1 off-policy window (`offpolicy = 4 / 12 / 32`) | `AsyncLoopConfig.max_offpolicy_steps` in `torchtitan/experiments/rl/controller.py` |
| §4.1 keep the KV cache across a weight swap | `generator.salt_prefix_cache_on_weight_sync` (default) vs `reset_prefix_cache_on_weight_sync` / `reset_running_requests_on_weight_sync` |
| The full bitwise-parity derivation | [`torchtitan/experiments/rl/models/GDN_BITWISE_PARITY.md`](./torchtitan/experiments/rl/models/GDN_BITWISE_PARITY.md) -- component by component, with the ablation table |

Metrics the post plots, as logged:

- `bit_wise/logprob_diff/{mean,abs_mean,max}` and `bit_wise/ratio_tokens_different/mean`
  (computed in `torchtitan/experiments/rl/losses/{dapo,dppo}.py`)
- `rollout_reward/avg_train_reward` -- the de-biased training-reward curve
- `validation_reward/_mean` -- held-out eval (AIME2025 for math, EM for search)
- `perf/trainer/tokens_per_second_full_step` -- the throughput number quoted for all
  three workloads

## Running the three workloads

Every recipe is a function in `torchtitan/experiments/rl/examples/<name>/config_registry.py`
and is selected by name:

```bash
python -m torchtitan.experiments.rl.train --module <example> --config <recipe_fn>
```

### MATH (single-turn, long generation) -- §4.2.1

```bash
python -m torchtitan.experiments.rl.train --module dapo_math --config rl_dapo_qwen3_4b_math_8k
```

See [`examples/dapo_math/README.md`](./torchtitan/experiments/rl/examples/dapo_math/README.md).
**Caveat:** what is checked in here is the *single-node Qwen3-4B* DAPO-Math recipe. The
post's math runs are Qwen3.5-9B-Base on 40 GPUs (8-way FSDP trainer + 4 vLLM generators
at DP=8) driven by a multi-host launcher that is not part of this repo, so the exact 9B
recipe file is not here. The knobs that define the arms are the ones in the table above,
and the off-policy sweep is `AsyncLoopConfig.max_offpolicy_steps = 4 / 12 / 32`.

### Search-R1 (multi-turn, short generation) -- §4.2.2

```bash
# needs the e5/wiki-18 dense retriever on http://127.0.0.1:8000/retrieve first
python -m torchtitan.experiments.rl.train --module search_r1 --config rl_grpo_qwen3_5_9b_search_r1          # vLLM native
python -m torchtitan.experiments.rl.train --module search_r1 --config rl_grpo_qwen3_5_9b_search_r1_unified  # unified
```

See [`examples/search_r1/README.md`](./torchtitan/experiments/rl/examples/search_r1/README.md)
for the retriever setup. `..._unified_prefix` adds align-mode mamba prefix caching. The
BI arm is the unified recipe plus the four coupled settings listed for tmax below.

### TMax terminal agent (multi-turn, long generation) -- §4.2.3

```bash
# one-time: materialize allenai/tmax-15k-open-instruct into the task JSONL
# (writes a 5-task tmax_smoke.jsonl next to --out as well)
HF_TOKEN=... python -m torchtitan.experiments.rl.examples.tmax.prepare_tmax_data \
  --out ./tmax_train.jsonl

# unified, no BI
SWE_PROMPT_DATA=./tmax_train.jsonl DAYTONA_API_KEY=... \
  python -m torchtitan.experiments.rl.train --module tmax --config rl_grpo_qwen3_5_9b_tmax

# unified + BI (the arm the post calls "with BI")
SWE_GDN_BI=1 SWE_PROMPT_DATA=./tmax_train.jsonl DAYTONA_API_KEY=... \
  python -m torchtitan.experiments.rl.train --module tmax --config rl_grpo_qwen3_5_9b_tmax
```

`SWE_GDN_BI=1` is the single switch for the BI arm; it flips the four coupled settings
that path requires (see the comment at that branch in
`examples/tmax/config_registry.py`):

1. trainer `debug.batch_invariant` -> recurrent forward, chunked backward
2. generator `debug.batch_invariant` -> the recurrent-everywhere decode path
3. generator `backend="torchtitan_wrapper"` -> the unified GDN core that holds it
4. generator `gdn_trainer_parity=True` -> select the `fla` recurrence

Each rollout needs a [Daytona](https://www.daytona.io/) sandbox, so a real TMax run
needs an API key and enough sandbox quota; `examples/tmax/local_smoke.py` is the
smallest thing that exercises the path. `rl_grpo_qwen3_4b_tmax` is the **dense**
(softmax-attention) control from the post's methodology: same recipe, no GDN, BI on.

**One operational trap, repeated from the post:** BI makes generation several times
slower, so the agent's wall-clock budgets have to be widened by roughly the same
factor. If you skip that, rollouts that would have finished get killed for running
long, and an unsubmitted rollout scores 0 -- the slowdown then shows up as a *reward*
regression instead of a throughput one.

## Verifying parity yourself

```bash
pytest torchtitan/experiments/rl/tests/test_qwen3_5_gdn_bi.py     # GDN batch invariance
pytest torchtitan/experiments/rl/tests/test_bitwise_parity.py     # trainer/generator parity (GPU)
pytest torchtitan/experiments/rl/tests/test_qwen3_5_gdn_packing.py
```

Parity holds only when the trainer and generator run the **same parallelism** (the
post's runs are `TP = 1` on both sides; DP and FSDP do not reduce along the batch
axis, so they add no mismatch).

## What is not here

- The multi-host launcher and job configs used for the 40-56 GPU runs.
- The W&B projects and the plot data behind the figures.
- The blog's MoE arm as a dedicated recipe (Qwen3.5-35B-A3B; the model itself is in
  `torchtitan/models/qwen3_5/`, and the extra piece it needed is the batch-invariant
  `bmm` above).

## Credit

Built on [torchtitan](https://github.com/pytorch/torchtitan) and its RL stack, with the
TorchTitan team. The kernels come from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (Gated
DeltaNet) and Thinking Machines'
[batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops),
whose [post](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
is the framing for all of this. Terminal-agent rollouts run on
[Daytona](https://www.daytona.io/) sandboxes.

Questions or corrections: yichuan_wang@berkeley.edu

```bibtex
@misc{wang2026traininginferencemismatch,
  title  = {Defending Against the Training-Inference Numeric Mismatch in RL (Especially Linear Attention) -- and Whether It Helps Async RL},
  author = {Wang, Yichuan and the TorchTitan team},
  year   = {2026},
  month  = {August},
  url    = {https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/}
}
```

# Pass-rate data curation for SWE-R2E RL

This is the recipe + the tools for the "data washing" step the user asked for:
filter the raw R2E task pool down to a **learnability band** (e.g. 20-70% pass
rate) before training, so binary-reward GRPO actually gets a gradient.

It follows a rejection-sampling pass@k -> per-instance pass-rate -> exclusive-band
filter recipe. Below: (1) why curate at all; (2) the two-stage technique and how
our two scripts implement it; (3) the step-2 training plan on the curated data.

---

## 1. Why curate at all (the problem the band solves)

GRPO/DAPO learns from *within-group reward variance*. With a binary reward (solved
/ not), a prompt-group of K rollouts gives a gradient only if it is **mixed**
(some solve, some don't). A group that is all-fail (0/K) or all-solve (K/K) has
zero std -> zero advantage -> zero gradient, and the soft filter drops it.

On the **full** R2E pool the Qwen3-32B base policy solves only **~3%** of tasks
(measured live: 17 solved / 575 graded rollouts). Those solves cluster on a few
easy tasks; the rest are ~never solved. So the per-task pass-rate is bimodal:
mostly 0% (too hard), a few ~100% (too easy), very little in between. Training on
the full pool starves the trainer -- almost every group is all-fail, dropped,
and the trainer waits forever for a mixed group (we saw 0 train steps land).

The fix is a **curriculum**: pre-measure each task's pass-rate and keep only the
middle band (not-too-hard, not-too-easy). Every kept task tends to produce mixed
groups -> non-zero advantage -> a real gradient every step.

---

## 2. The two-stage technique (reference recipe)

The end-to-end flow is **two stages**: an offline rejection-sampling data wash,
then RL training on the resulting banded set.

### Stage 1 - data washing / learnability filtering (rejection sampling, pass@k)
- Run the **same actor + grader you will train with** over the raw task pool, N
  samples per task (~8-10 is typical), recording per-(task, sample) `resolved`
  (0/1). This is an embarrassingly-parallel job: shard the
  `dataset_size * num_samples` (instance, sample) attempts across hosts, run each
  attempt independently, write a per-instance result, resume by result existence,
  and aggregate pass@k with the standard unbiased estimator.
- Aggregate to a per-instance summary grouped by instance_id ->
  `[instance_id, resolved, num_samples]`. `pass_rate = resolved / num_samples`.
- **Band filter** (THE primitive): keep instances whose pass rate falls in an
  EXCLUSIVE band `pass_min < pass_rate < pass_max` (e.g. **0.2 .. 0.7**),
  dropping 0% (too hard / broken env) and 100% (too easy), and stamp the measured
  pass rate onto the output JSONL as metadata.
- Datasets are effectively **named by band** (e.g. `40-75`, `10-50`, `10-70`). A
  realistic funnel drops most of the raw pool: tens of thousands of raw tasks
  narrow through a correlation filter, then a pass-rate band, to a few thousand
  final tasks.
- **Secondary filters** (optional, after banding): tool-call<->reward correlation
  >= 0, has_repro_or_test, no gibberish / banned commands, drop broken envs.

### Stage 2 - train on the banded set
- Async GRPO-style RL with a **normal group size** (e.g. 64; smaller, ~16, for
  debug).
- Keep **online** zero-variance filtering even on banded data: drop any group
  whose reward std collapses (below ~1e-3) as the model improves. Plus drop-NaN
  and off-policy age management (e.g. `max_offpolicy_steps=16`).
- Optional cheaper online curriculum: run a small pilot batch and drop a prompt
  *before* spending the full rollout budget if the pilot shows no reward variance.
- **Curriculum schedule**: train the easier band first (e.g. 40-75), then
  *continue from that checkpoint* on a harder band (e.g. 10-50). The band is a
  moving target -- re-wash with the improved policy between stages.

Key principle: **measure the band with the exact reward the trainer uses.** We
curate on binary `solved` (default reward), not the dense fraction, so the band
matches what GRPO will see.

---

## 3. Our implementation (two scripts, mirrors the two stages)

### `curate_passrate.py` -- stage 1 worker (per host)
A standalone, embarrassingly-parallel worker (no controller / trainer / mesh /
weight sync), modeled on `local_smoke_harness.py` + the distributed
rejection-sampling launcher pattern:
- one in-process vLLM `AsyncLLMEngine` serving the policy (continuous batching for
  the K-way-per-task fanout), one `AnthropicAdapter`;
- shards the pool with `--shard-id i --num-shards N` (strided, so each host gets a
  difficulty-balanced slice of the repo-sorted pool);
- per (task, sample): `boot_agent_sandbox -> run_claude_code -> git_diff ->
  evaluate_r2e` -> binary `solved` (the exact training path, minus turn capture);
- writes `<out-dir>/results/<instance_id>/sample_<k>.json`
  (`{instance_id, sample_idx, solved, reward, applied, status, error, num_turns}`),
  **resumable** (an existing result is skipped) so a preempted host re-runs only
  its missing attempts;
- bounded by `--concurrency` (per-host active rollouts) + a per-attempt wall-clock
  guard. Reuses the harness env knobs (`SWE_BOOT_CONCURRENCY`,
  `SWE_TIME_BUDGET_SEC`, `SWE_EVAL_TIMEOUT_SEC`, `SWE_MAX_CONTEXT_LEN`,
  `DAYTONA_API_KEY`).

Launch N hosts pointed at one shared `--out-dir`. K should match the training
group size (8-16) and use the SAME base checkpoint the trainer starts from
(pass-rate is policy-dependent).

### `aggregate_passrate.py` -- stage 2 filter (pure CPU)
Reads `<out-dir>/results`, rolls up per-instance `(resolved, graded)` ->
`pass_rate`, prints the full histogram, then keeps the **exclusive** band
(`--pass-min 0.2 --pass-max 0.7`, `--min-samples 8`) and joins back to the source
JSONL, stamping `metadata.pass_rate / resolved / num_samples`. Output is a drop-in
for `SWER2EDataset`. Also writes `instance_summary.csv`. ERROR/timeout attempts
are excluded from the denominator by default (infra failures, not difficulty).

```
# stage 1: on each of N hosts (run.sh SWE_CURATE branch sets shard id)
python -m ...swe_r2e.curate_passrate --model .../Qwen3-32B \
    --out-dir <bucket>/curate_out --shard-id $i --num-shards $N --k 8 \
    --tensor-parallel 8 --concurrency 24
# stage 2: once, on a CPU-only box
python torchtitan/experiments/rl/examples/swe_r2e/aggregate_passrate.py \
    --results-dir <bucket>/curate_out/results \
    --source-jsonl <bucket>/r2e_subset_4p5k.jsonl \
    --out <bucket>/r2e_band_20_70.jsonl --pass-min 0.2 --pass-max 0.7 --min-samples 8
```

---

## 4. Step-2 training on the curated band

Point the existing async config at the curated JSONL via `SWE_PROMPT_DATA` and
restore **normal** group settings (the current `num_groups_per_train_step=1 /
group_size=16` is only the sparse-full-R2E starvation workaround):
- `group_size` ~16-32, `num_groups_per_train_step` > 1 (the banded data makes ~every
  group mixed, so multi-group steps fill quickly);
- keep the soft filter on (drop zero-std groups) -- as the policy improves, banded
  tasks drift toward all-solve; the online zero-variance filter catches that;
- keep binary/sparse reward (see `[[feedback_swe_sparse_reward]]`);
- when reward plateaus, re-wash with the improved checkpoint and re-band downward
  (curriculum), as in the 40-75 -> 10-50 continuation.

## 5. Cost / sizing

Full R2E = 4578 tasks. At K=8 that is ~36.6k agent rollouts (each a multi-turn
Claude Code episode, ~3-4 min, up to ~70 turns) + a grading sandbox each. With ~90
hosts x ~16-24 concurrent rollouts/host the wash is a few hours wall-clock. The
20-70% band is expected to be a *small* fraction of 4578 (such funnels typically
drop ~60-90%); the histogram from stage 2 tells us the real yield and lets us widen
the band if it is too thin at the 32B base's ~3% solve rate.

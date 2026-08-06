# Unified-model GDN generation: prefill acceleration + trainer alignment

How TorchTitan's own `GatedDeltaNet` (GDN) runs *inside* vLLM under the
`torchtitan_wrapper` (unified) path, and how the **prefill** path is aligned to the
trainer in three targeted steps so the generator's prefill logprobs approach the
trainer's bitwise -- **without** paying the full recurrent-everywhere cost.

Code: `torchtitan/experiments/rl/models/gdn_vllm_unified.py`.
Related: `GDN_BITWISE_PARITY.md` (the bitwise/parity story), `gdn_vllm_titan.py`
(the `vllm_native` variant that keeps vLLM's whole native GDN model and swaps only
the recurrence kernel).

---

## 1. What "unified model" means here

Two ways to run Qwen3.5 GDN generation in vLLM:

- **`vllm_native`** (`gdn_vllm_titan.py`): keep vLLM's native GDN model, swap only
  the recurrence kernel.
- **`torchtitan_wrapper` / unified** (this file): every non-GDN layer already runs
  TorchTitan code via the wrapper; this module makes the **GDN layer unified too**.
  Its **parameters stay TorchTitan's** (they live in `qwen3_5.model.GatedDeltaNet`),
  while the stateful **conv + recurrence borrow ONLY vLLM's paged cache management**
  and helper kernels so continuous-batch generation works.

**Mechanism** (`gdn_vllm_unified.py:19-25`): `VLLMGatedDeltaNetCore` is a
parameter-less `MambaBase` layer. vLLM's hybrid KV-cache discovery
(`get_kv_cache_spec` -> `MambaSpec`) sees it via `static_forward_context` and
allocates the paged `conv_state` + `ssm_state`. `GatedDeltaNet` computes its
projections and gates, then delegates the conv + recurrence to this core (a drop-in
for `_causal_conv` + `GatedDeltaKernel`). The 3 depthwise convs (conv_q/k/v) fuse
channel-wise into the single fused conv vLLM's `causal_conv1d` kernels expect
(depthwise -> identical math).

## 2. Cache management (borrowed from vLLM, unchanged)

We reuse vLLM's cache strategy as-is:

- **Full-attention layers**: normal paged KV cache.
- **GDN layers**: paged **`conv_state`** + **`ssm_state`**, both **constant length
  per request** (conv keeps `conv_kernel_size - 1` history columns; ssm is the
  recurrent state matrix). One state slot per sequence
  (`non_spec_state_indices_tensor`).
- **Prefix-cache hit**: **align mode** *(TODO)* -- match to the last chunk-aligned
  block for the conv and SSM state. It works, but is **tricky for async RL**: the
  in-flight conv/SSM state is not trivially reusable across a weight swap the way a
  token KV page is, so prefix reuse across policy versions needs care.

## 3. Goal: drive prefill logprob diff -> 0

For the big run we want the generator's **prefill** logprobs to match the trainer's
so DPPO/importance-sampling is clean (a fat prefill logprob tail poisons the
unclipped surrogate). Decode is inherently recurrent and has no trainer equivalent
(the trainer never decodes), so the target is specifically the **prefill** path.

The core patch is on the vLLM decode/prefill core:
`class VLLMGatedDeltaNetCore(Module, MambaBase)` (`gdn_vllm_unified.py:94`).

## 4. Three code paths (by flag)

Set once at engine build from `VLLMGenerator.Config.gdn_trainer_parity`
(`set_trainer_parity_fla`, `gdn_vllm_unified.py:84-91`):

1. **Recurrent-everywhere (bitwise)** -- `_TRAINER_PARITY_FLA and
   is_in_batch_invariant_mode()` (`:497`): fla conv + fla **recurrent** kernel for
   **both** prefill and decode, so `decode == prefill == trainer` bitwise (decode is
   inherently recurrent, so the only way prefill/trainer can match it is the same
   recurrent kernel). Trainer forward is likewise swapped to fla recurrent under BI
   (`model.py` `_RecurrentFwdChunkBwd`). Most faithful, slowest.

2. **Trainer-parity chunk** -- `_TRAINER_PARITY_FLA=True`, not BI (`:542-704`):
   fresh prefill = external fla conv + external fla **chunk** with
   `initial_state=None` -> prefill bitwise vs trainer; decode uses vLLM single-step
   conv update + fla chunk on length-1 segments and CONTINUES from the seeded paged
   state.

3. **Vendored (default, fastest)** -- flag off (`:706-882`): vLLM's paged conv
   kernels for mixed/continuation, fast vendored decode, and a **3-step aligned
   prefill chunk** (below). This is the "accelerate prefill 3 step" path: keep the
   fast vendored decode, but align just the prefill chunk to the trainer so its
   logprobs approach bitwise at a fraction of path #1's cost.

## 5. The "3-step" prefill acceleration (vendored path)

Location: `_run_prefill_chunk` and the pure-fresh conv branch in the vendored path
(`gdn_vllm_unified.py:706-855`). Each step migrates one op from vLLM's vendored
kernel to the trainer's, so the fast default path's prefill converges to the
trainer bitwise.

### STEP 1 -- our fla chunk, not vLLM's vendored chunk (`:840-853`)
Run the prefill chunk through the trainer's kernel
`_external_fla_chunk_gated_delta_rule` (the SAME `fla` package the trainer uses),
not vLLM's vendored `chunk_gated_delta_rule` (a different copy -> a parity gap). RAW
q/k with `use_qk_l2norm_in_kernel=True`; fla derives `chunk_indices` from
`cu_seqlens` internally.

```python
out, final_state = _external_fla_chunk_gated_delta_rule(
    q, k, v, g, beta,
    initial_state=initial_state,
    output_final_state=True,
    cu_seqlens=cu_seqlens,
    use_qk_l2norm_in_kernel=True,
)
```

### STEP 2 -- eager fp32 gate + RAW q/k + in-kernel l2norm + GVA head-expand (`:825-839`)
Replace vLLM's `fused_post_conv_prep` (which rounds the l2norm in a *different*
kernel) with the trainer's exact pre-chunk math:

- **eager fp32 gate**: `g = -exp(A_log.float()) * softplus(a.float() + dt_bias.float())`
  and `beta = sigmoid(b.float())`, computed in fp32 in eager (state-independent --
  depends only on a/b -- so it applies to BOTH fresh and continuation).
- **RAW q/k with in-kernel l2norm**: pass raw q/k and let the fla chunk do the l2norm
  (`use_qk_l2norm_in_kernel=True`), matching the trainer instead of vLLM's separately
  rounded l2norm.
- **GVA head-expand**: when `num_k_heads < num_v_heads`, `repeat_interleave` q/k up to
  the value-head count (grouped-value attention), matching the trainer's expansion.

This is the "patch `fused_post_conv_prep`" item: we do NOT call it; we recompute the
gate + l2norm the trainer's way.

### STEP 3 -- fresh-prefill conv on the trainer kernel + `initial_state=None` + seed the paged state (`:709-736`, `:815-824`)
For a **pure-fresh prefill batch** (num_prefills > 0, no decodes, no continuation):

- **trainer's fla conv**: `_external_fla_causal_conv1d(...)` (bitwise-aligned),
  instead of vLLM's paged conv kernel. fla conv is stateless, so we then
  **`_seed_conv_state`** the paged `conv_state` so a later decode step continues
  correctly.
- **`initial_state=None`, NOT a zero tensor** (`:815-824`): fla's
  `USE_INITIAL_STATE` is a triton `constexpr` (and autotune key), so `None` and an
  all-zero tensor dispatch **different compiled kernels** with divergent fp
  reductions -- a zero tensor is NOT bitwise with the trainer's `None`. Continuation
  (chunked prefill / prefix hit) instead carries the paged state, transposed from
  vLLM's value-first `[.., HV, V, K]` layout to fla's key-first `[.., HV, K, V]`.

Mixed batches and chunked-prefill continuation keep vLLM's paged conv kernels
(no clean trainer equivalent), so only the clean fresh-prefill case gets the full
bitwise treatment; the chunk (STEP 1) + gate/l2norm (STEP 2) alignment still applies
to continuation.

## 6. Decode path (kept fast/vendored)

Decode stays on vLLM's fused kernels (`:764-804`):

- **pure-decode batch** -> `fused_recurrent_gated_delta_rule_packed_decode`.
- **mixed batch decode tokens** -> `fused_sigmoid_gating_delta_rule_update` (avoids
  routing length-1 decodes through the prefill kernel; keeps the unified path close
  to native GDN).

Both use `use_qk_l2norm_in_kernel=True`. Decode has no trainer reference, so it only
has to stay correct and continue from the paged state.

## 7. Full batch-invariant decode (recurrent-everywhere)

`_forward_recurrent_bi` (`gdn_vllm_unified.py:273-410`), used when
`gdn_trainer_parity` is on AND batch-invariant (BI) mode is active (`:497`). Goal:
`decode == prefill == trainer` **bitwise**.

Idea: decode is inherently recurrent, so the ONLY way prefill can match decode is to
also run recurrent (not chunk). So this path uses the fla **recurrent** kernel for
**both** prefill and decode:

- **conv**: fla stateful pair -- full `causal_conv1d` (with `output_final_state`) for
  prefill, `causal_conv1d_update` for decode -- against a side conv-state buffer keyed
  by vLLM slot ids.
- **recurrence**: fla `fused_recurrent_gated_delta_rule` against the paged `ssm_state`
  via gather/scatter (upstream fla recurrent has no slot indexing).

Two tricks make decode bitwise-equal to prefill:

1. **Run the recurrence in fp32.** The recurrent kernel is not step-consistent in
   bf16 (full-seq prefill vs 1-token decode differ ~1e-5 from bf16 rounding); in fp32
   they match to ~1e-8, which vanishes on write-back to the bf16 activation. The
   trainer upcasts identically.
2. **Fresh prefill passes a materialized ZERO state, NOT `None`.** fla's recurrent
   kernel compiles `USE_INITIAL_STATE` from `(h0 is not None)`, so a `None` prefill
   vs a tensor decode select different binaries with divergent fp reductions ->
   `decode != prefill`. A zero init makes both the SAME binary -> resume-exact.

> Subtle contrast worth remembering: the vendored 3-step path (STEP 3) uses
> `initial_state=None` for fresh -- to match the trainer's **chunk** kernel, whose
> `USE_INITIAL_STATE` constexpr wants None. The recurrent-BI path uses a **zero**
> state for fresh -- to keep prefill and decode on the SAME **recurrent** binary.
> Opposite choices, because they target different kernels' constexpr dispatch.

Fresh prefill only (`initial_state` None/zero, matching the RL rollout); prefix
caching is not supported on this path.

## 8. Summary

| Path | prefill kernel | decode kernel | prefill vs trainer | speed |
|---|---|---|---|---|
| Recurrent-everywhere (BI) | fla recurrent | fla recurrent | bitwise | slowest |
| Trainer-parity chunk | fla chunk (fresh, init=None) | vLLM update + fla chunk | bitwise (fresh) | mid |
| **Vendored + 3-step (default)** | **fla chunk + eager fp32 gate/l2norm + fla conv (fresh)** | **vLLM packed/update** | **near-bitwise (fresh)** | **fastest** |

The "accelerate prefill 3 step" = the vendored (fast, default) path's three targeted
migrations -- (1) fla chunk, (2) eager fp32 gate + RAW q/k + in-kernel l2norm + GVA
head-expand instead of `fused_post_conv_prep`, (3) fresh-prefill fla conv +
`initial_state=None` + seed the paged conv_state -- which push the fast path's
prefill logprobs toward the trainer's bitwise while keeping vLLM's fast vendored
decode and paged cache management.

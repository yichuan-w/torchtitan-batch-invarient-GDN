# GDN generator/trainer bitwise parity

How to make the vLLM **generator** produce **bit-identical** per-token logprobs to
the TorchTitan **trainer** for the Qwen3.5 Gated-DeltaNet (GDN) hybrid model, so the
DPPO importance ratio `pi/mu` has no numeric drift.

Verified `num_diff = 0`, `max |logprob diff| = 0.0` (was ~9 nats before) for BOTH:
- a single-sequence prefill, and
- the PACKED path (= real training): trainer packs 2 samples in one row
  (`cu_seqlens=[0, la, n]`), generator serves them as 2 requests -- bitwise per segment.

Opt-in via `VLLMGenerator.Config.gdn_trainer_parity = True` on the `torchtitan_wrapper`
generator (calls `set_trainer_parity_fla`). Prefill is bitwise; decode continues from the
seeded paged state (correct, not bitwise -- see caveats).

---

## 1. Why they diverge (and what "bitwise" requires)

The generator and trainer are two different code paths. Even with identical weights,
bf16 matmuls are **not associative** (`(a+b)+c != a+(b+c)`), so any difference in
reduction order, tiling, kernel implementation, or algorithm changes the low bits.
Over a deep model those per-op differences accumulate and blow up on
numerically-sensitive tokens (long-range recall, code/symbol tokens).

Bitwise parity therefore needs **two** things:

1. **Every op batch-invariant** — the kernel must give the same result regardless of
   batch shape / chunking (fixed reduction order).
2. **The generator must run the trainer's kernels op-for-op** — same conv, same gate,
   same l2norm, same recurrence, same state layout, same call arguments.

We align the **generator to the trainer** (not vice-versa) because only the trainer's
`fla` path has a backward; it is the fixed reference that training uses.

The dense (softmax) Qwen3 model reaches bitwise 0 for free because every op it uses
(`mm`/`addmm`, softmax attention) has a batch-invariant version. GDN adds two ops with
no drop-in BI version — the hybrid full-attention (blocked from `flex`) and the GDN
recurrence — which is why it needs the explicit alignment below.

---

## 2. The recipe (component by component)

Qwen3.5-9B is hybrid: `full_attention_interval=4`, so **1/4 layers are full softmax
attention, 3/4 are GDN**. Both the backbone and the GDN layer must be aligned.

### 2a. Backbone: batch-invariant mm + split-K-free attention

Turn on batch-invariant mode on **both** sides (`debug.batch_invariant=True` +
`set_batch_invariance(True)` on the generator). This:

- routes `mm`/`addmm`/`_log_softmax`/`mean` through fixed-reduction-order Triton
  kernels (the `batch_invariant_ops` package) -> BI projections / MoE / norms;
- forces **`num_splits=1`** on the varlen (FlashAttention) full-attention layers.

**The split-K point.** Attention non-invariance comes from *split-K / split-KV*
(flash-decoding): the KV dimension is split across thread blocks and the split count
depends on `max_k`, so the reduction order changes with batch composition. Setting
`num_splits=1` disables split-KV -> fixed order -> batch-invariant. This is already
wired on both sides under BI mode:

```python
# trainer: torchtitan/models/common/attention.py (VarlenAttention.forward)
if fa_impl in (None, "FA2") or is_in_batch_invariant_mode():
    varlen_kwargs["num_splits"] = 1   # disable split-KV -> batch invariant

# generator CUSTOM backend: torchtitan/experiments/rl/models/attention.py
# (PyTorchVarlenAttentionImpl) sets the same num_splits=1 under BI mode.
```

**Use `varlen`, not `flex`, for GDN.** `flex` is the other BI-pinnable attention, but
GDN's mamba page alignment forces the attention `block_size` to **528**, and
`flex_attention` requires a power-of-2 block (`528 = 16*33` -> `ValueError`). Varlen
FlashAttention only needs `block_size % 16 == 0`, so it accepts 528 *and* is BI via
`num_splits=1`. So build the model with `attn_backend="varlen"`.

### 2b. GDN gate: eager fp32, not the fused Triton kernel

The gate `g` (log-space decay) and `beta` (update gate) do **not** depend on the conv;
they come straight from the `a`/`b` projections. The trainer computes them eager fp32:

```python
# trainer: torchtitan/models/qwen3_5/model.py  (GatedDeltaNet.forward)
g = -torch.exp(self.A_log.float()) * F.softplus(xa.float() + self.dt_bias.float())
beta = torch.sigmoid(xb.float())
```

The vLLM generator normally fuses gate + l2norm + split into one Triton kernel
(`fused_post_conv_prep`), which rounds differently. Reproduce the eager form instead:

```python
# generator: gdn_vllm_unified.py  (VLLMGatedDeltaNetCore parity path, _fla_recurrence)
g = (
    -torch.exp(A_log.float())
    * F.softplus(a_slice_THv.float() + dt_bias.float())
).unsqueeze(0)                                   # [1, T, HV], fp32, log-space
beta = torch.sigmoid(b_slice_THv.float()).unsqueeze(0)
```

`a_THv`/`b_THv` are the raw `in_proj_a`/`in_proj_b` outputs (== trainer `xa`/`xb`).
`.unsqueeze(0)` adds the batch dim `fla` expects.

### 2c. Conv: `fla causal_conv1d` with `cu_seqlens` (trainer PACKED path)

Real training PACKS several rollout samples into one row, so `positions` restart at each
sample and `_cu_seqlens_from_positions` returns `cu_seqlens=[0, la, ...]`. The trainer
then uses the PACKED conv path -- `fla`'s `causal_conv1d` with `cu_seqlens` (resets at each
boundary), NOT eager `F.conv1d` and NOT vLLM's paged `causal_conv1d_fn`:

```python
# trainer: torchtitan/models/qwen3_5/model.py  (GatedDeltaNet._causal_conv, cu_seqlens != None)
y = _fla_causal_conv1d(
    x.reshape(1, bs * seqlen, channels),
    weight=weight.squeeze(1),   # [C, k], per projection (conv_q/k/v)
    bias=bias, activation="silu", cu_seqlens=cu_seqlens,
)
```

Match it in the generator core with the per-request `cu_seqlens`. The 3 conv weights are
fused (`[C, k]`); one depthwise fla conv over `q|k|v` == the trainer's 3 separate convs
(channels independent). Each generator request is one segment (`cu_seqlens=[0, len]`), which
equals a packed trainer segment (reset at its start):

```python
# generator: gdn_vllm_unified.py  (section 1, external-fla + pure-prefill)
conv_out_TC = _external_fla_causal_conv1d(
    mixed_qkv_TC.unsqueeze(0),          # [1, n, C] channels-last (as the trainer)
    weight=conv_weight,                 # [C, kw]
    bias=conv_bias,
    activation="silu",
    cu_seqlens=m.non_spec_query_start_loc,
)
conv_out_TC = (conv_out_TC[0] if isinstance(conv_out_TC, tuple) else conv_out_TC).squeeze(0)
```

> The trainer now unifies single-sample rows onto this SAME fla+cu path: a single sample
> yields `cu_seqlens=[0, L]` (see `_cu_seqlens_from_positions`), so a single-sequence
> prefill is ALSO fla conv + fla chunk and bitwise-matches the per-request generator.
> (Previously single-sample rows used eager `F.pad+F.conv1d+F.silu` with `cu_seqlens=None`,
> a DIFFERENT kernel -- so the generator's fla+cu did NOT match a single non-packed row,
> and a sample's logprobs depended on whether it got packed with others. Unifying the
> trainer removed that gap. Under TP/SP the single-segment path falls back to non-packed,
> since parity only matters under FSDP where the generator runs TP=1.)

### 2d. L2 norm: in-kernel (both sides)

The trainer l2-normalizes q/k **inside** the chunk kernel (`use_qk_l2norm_in_kernel=True`,
`torchtitan/models/qwen3_5/model.py` `GatedDeltaKernel.forward`). So pass **raw** q/k to
the generator's chunk call with `use_qk_l2norm_in_kernel=True` too — do NOT pre-normalize
(that is what `fused_post_conv_prep` did). The GVA head-expand must also match the trainer
(`repeat_interleave` when `HV > H`):

```python
# generator: gdn_vllm_unified.py
q, k, v = _split_qkv(conv_out_TC[start:end])   # RAW conv output, not l2-normed
if q.shape[2] != v.shape[2]:                   # GVA, matches model.py GatedDeltaKernel
    rep = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(rep, dim=2)
    k = k.repeat_interleave(rep, dim=2)
```

### 2e. Recurrence: external fla chunk, `cu_seqlens=None`, `initial_state=None`

Use the **trainer's external `fla` chunk kernel** (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`),
not vLLM's vendored copy. For a single non-packed sequence the trainer calls it
**batched** (`cu_seqlens=None`) and **stateless** (`initial_state=None`):

```python
# generator: gdn_vllm_unified.py  (PACKED / cu_seqlens path)
out, final_state = _external_fla_chunk_gated_delta_rule(
    q, k, v, g, beta,
    initial_state=None,                       # NOT a zero tensor -- see note
    output_final_state=True,
    cu_seqlens=m.non_spec_query_start_loc,    # varlen, resets at sample boundaries
    use_qk_l2norm_in_kernel=True,
)
```

Two subtleties that each break bitwise if wrong:

- **`cu_seqlens` must match the trainer's batching.** Real training is PACKED, so the
  trainer's chunk uses `cu_seqlens` (varlen) -- pass the per-request `cu_seqlens` here too.
  (For a genuine single non-packed sequence the trainer uses `cu_seqlens=None` batched; then
  match that instead. Mismatching the two rounds differently.)
- **`initial_state=None`, not a zero tensor.** `fla`'s `USE_INITIAL_STATE` is a Triton
  `constexpr`; `None` and a real (even all-zero) tensor take *different* kernel branches
  with different reductions. The trainer is stateless (`initial_state=None`), so a zero
  tensor here is NOT bitwise-equal — pass `None`.

### 2f. Paged state layout (for the `vllm_native` / `TT_GDN_UNIFIED_KERNEL` path)

vLLM's paged `ssm_state` is `[.., HV, head_v_dim, head_k_dim]` (value-first `[V, K]`);
`fla` uses key-first `[K, V]` at the default `transpose_state_layout=False`. Bridge with
an explicit transpose on the gather/scatter (do NOT flip the kernel layout flag — it is a
compute-path flag and changing it fattens the tail). This is the `state_v_first` fix in
`gdn_vllm_titan.py`:

```python
# gather: paged [V, K] slot -> fla [K, V]
initial_state = ssm_state[state_idx].transpose(-1, -2).contiguous()
...
# scatter: fla [K, V] -> paged [V, K] slot
ssm_state[state_idx] = final_state.transpose(-1, -2).to(ssm_state.dtype)
```

(The old `state_v_first=True` kwarg did not exist in `fla` and was silently dropped by
`**kwargs`; the explicit transpose is also 27B-safe when `head_k_dim != head_v_dim`.)

---

## 3. Result

Progression on the trace (short 1500-tok single-seq prefill, `wrapper + varlen + BI`):

| config | max \|Δlogp\| | ratio>2/>5/>10 | p50 |
|---|---|---|---|
| vendored vLLM fla | 4.57 | 7/4/2 | 6.1e-3 |
| + external fla recurrence | 3.52 | 6/1/1 | 5.9e-3 |
| + eager gate | 1.84 | 4/1/0 | 6.3e-3 |
| + conv / cu_seqlens=None / initial_state=None | **0.0** | **0/0/0** | **0.0** |

`bitwise_equal = True`, `num_diff = 0/1499`.

PACKED path (real training; trainer packs 2 x 2500-tok samples, generator serves 2 requests
in one prefill batch, `cu=[0,2500,5000]`): `seg A num_diff 0/2499 max 0.0`,
`seg B num_diff 0/2499 max 0.0` -- bitwise per segment, **provided prefill is not chunked**
(see caveats: `max_num_batched_tokens >= max_model_len`).

DECODE continuation (prefill 1500-tok, then decode 48 tokens): generated text is coherent,
decode logprobs finite, and decode(stream) vs the generator's own 2nd-pass prefill has
`max 0.35, ratio>2: 0` -- i.e. the seeded paged conv_state + fla decode is numerically
consistent (a broken seed would blow up the first decode token). decode is NOT bitwise vs
trainer prefill (recurrent vs chunk; inherent, see caveats).

Every component matters: swapping only the recurrence, or only the gate, only tightens
the chaotic tail; bitwise needs all of conv + gate + l2norm + recurrence + state layout
aligned AND the backbone batch-invariant (num_splits=1 + BI mm).

---

## 4. How to run

In code, enable the parity path via the generator config
`VLLMGenerator.Config.gdn_trainer_parity = True` (torchtitan_wrapper only), which calls
`gdn_vllm_unified.set_trainer_parity_fla(True)` and raises `max_num_batched_tokens` to
`max_model_len`. The repro (`repro_bi_gdn.py`) calls `set_trainer_parity_fla(True)` directly:

The driver script used for those numbers is not checked in. The equivalent
in-repo checks are the GDN batch-invariance and parity tests:

```bash
pytest torchtitan/experiments/rl/tests/test_qwen3_5_gdn_bi.py
pytest torchtitan/experiments/rl/tests/test_bitwise_parity.py   # needs a GPU
```

End to end, the same path is what ``SWE_GDN_BI=1`` turns on for the tmax recipe
(see ``examples/tmax/config_registry.py``): trainer + generator
``batch_invariant``, ``backend="torchtitan_wrapper"``, ``gdn_trainer_parity=True``.

Needs `batch_invariant_ops` installed:
`uv pip install --no-deps "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"`.

---

## 5. Caveats / scope

- **Packed AND single-seq both bitwise.** The trainer runs the fla+cu path for BOTH multi-sample
  packed rows AND single-sample rows (`_cu_seqlens_from_positions` returns `[0, L]` for a single
  sample), matching the per-request generator. Verified: PACKED 2x2500 seg A/B num_diff 0/2499;
  single-seq [prompt+gen] 2nd-pass-prefill vs trainer num_diff 0/48. (Under TP/SP the single
  segment falls back to non-packed -- parity only applies under FSDP, generator TP=1.)
- **Decode continues, but is NOT bitwise vs trainer prefill.** Prefill SEEDS the paged
  conv_state (last `kw-1` pre-conv inputs, oldest first -- exactly what vLLM's
  `causal_conv1d_fn` writes / `causal_conv1d_update` reads), then decode runs vLLM's
  single-step conv update + the external fla chunk on length-1 segments. This keeps
  generation correct and continuous, but decode builds the recurrent state incrementally
  (chunk(prompt) + per-step) whereas the trainer builds it in one contiguous chunk -- a
  different fp reduction order, so decode logprobs can't be bitwise with trainer prefill.
  This matters because the training old-logprob comes from DECODE (`generator.py` streaming
  logprobs -> `dppo.py generator_logprobs`), so prefill bitwise parity does NOT by itself
  remove the DPPO ratio tail on response tokens.
- **No chunked prefill.** Bitwise parity needs each request prefilled as ONE contiguous fla
  chunk. vLLM's default `max_num_batched_tokens=2048` splits longer prefills; continuation
  chunks (`has_initial_state=True`) fall back to the vendored kernels (non-bitwise). The
  generator sets `max_num_batched_tokens = max_model_len` when `gdn_trainer_parity` is on so
  a fresh request is not chunked (the scheduler may still chunk to co-run with decodes).
- **Slow parity path.** external fla conv + fla chunk + `num_splits=1` + no cudagraph is a
  correctness/debug mode, not the fast serving path.
- **Practical alternative:** for the fast path, tolerate the residual with DPPO
  `ratio_cap=2` (the "fat tail" max is a non-poisoning outlier anyway). Bitwise parity is
  the correctness ceiling; ratio_cap is the shipping mitigation.

## Related code
- `gdn_vllm_unified.py` — `VLLMGatedDeltaNetCore`, the `gdn_trainer_parity` path
  (`set_trainer_parity_fla`); prefill on external fla + `_seed_conv_state` for decode.
- `gdn_vllm_titan.py` — `TitanFLAGatedDeltaNet` (vllm_native + `TT_GDN_UNIFIED_KERNEL`), the
  `state_v_first` state-layout fix.
- `torchtitan/models/qwen3_5/model.py` — the trainer `GatedDeltaNet` / `GatedDeltaKernel`
  (the reference implementation). `_cu_seqlens_from_positions` now returns `[0, L]` for a
  single sample so the trainer runs fla+cu for single-sample rows too (unified with the
  per-request generator; TP/SP single-segment falls back to non-packed).
- `torchtitan/models/common/attention.py`, `experiments/rl/models/attention.py` —
  `num_splits=1` under batch-invariant mode.

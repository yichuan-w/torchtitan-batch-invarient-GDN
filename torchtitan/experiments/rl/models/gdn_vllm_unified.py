# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified-model GDN generation core: TorchTitan's own GatedDeltaNet running
inside vLLM under the ``torchtitan_wrapper`` path, borrowing ONLY vLLM's paged
conv+ssm state cache management.

Contrast with ``gdn_vllm_titan.py`` (the ``vllm_native`` path), which keeps
vLLM's whole native GDN model and swaps just the recurrence kernel. Here every
non-GDN layer already runs TorchTitan code (via the wrapper); this module makes
the GDN layer unified too: its parameters stay TorchTitan's (they live in
``qwen3_5.model.GatedDeltaNet``), while the stateful conv + recurrence use
vLLM's native GDN helper kernels against vLLM's paged cache so continuous-batch
generation works.

Mechanism: ``VLLMGatedDeltaNetCore`` is a parameter-less ``MambaBase`` layer.
vLLM's hybrid KV-cache discovery (``get_kv_cache_spec`` -> ``MambaSpec``) sees it
via ``static_forward_context`` and allocates the paged conv_state + ssm_state.
``GatedDeltaNet`` computes its projections and gates, then delegates the conv +
recurrence to this core (drop-in for ``_causal_conv`` + ``GatedDeltaKernel``).
The 3 depthwise convs (conv_q/k/v) fuse channel-wise into the one fused conv
vLLM's causal_conv1d kernels expect -- depthwise, so identical math.

Legend (tensor shape suffixes, this module):
  T  = num actual tokens in the flattened batch (all requests concatenated)
  Ck = key conv/proj channels = num_k_heads * head_k_dim
  Cv = value channels        = num_v_heads * head_v_dim
  Hk = num key heads, Hv = num value heads, Dk = head_k_dim, Dv = head_v_dim
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# External fla (the SAME package the trainer uses). Enabled via the generator
# config flag ``gdn_trainer_parity`` (-> set_trainer_parity_fla) so the wrapper's
# GDN prefill runs the trainer's exact chunk + conv kernels (the vendored vLLM fla
# is a different copy -> a parity gap). We align the GENERATOR to the TRAINER
# because only the trainer's fla path has a backward (training needs it), so the
# trainer's kernels are the reference.
from fla.modules.convolution import (
    causal_conv1d as _external_fla_causal_conv1d,
    causal_conv1d_update as _external_fla_causal_conv1d_update,
)
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _external_fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _external_fla_fused_recurrent_gated_delta_rule,
)
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.experiments.rl.models.gdn_fla_paged import gather_transposed_paged_state
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as _vllm_chunk_gated_delta_rule,
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# Set once at engine-build time from VLLMGenerator.Config.gdn_trainer_parity (see
# generator.py). When True the wrapper GDN core runs the trainer's external fla
# conv + chunk kernels on the prefill path so the generator's PREFILL logprobs
# match the trainer's bitwise; decode continues from the paged state. Default off
# -> vLLM vendored kernels (faster). See GDN_BITWISE_PARITY.md.
_TRAINER_PARITY_FLA = False


def set_trainer_parity_fla(enable: bool) -> None:
    """Route the wrapper GDN prefill through the trainer's external fla kernels."""
    global _TRAINER_PARITY_FLA
    _TRAINER_PARITY_FLA = enable
    logger.info(f"[gdn-unified] wrapper GDN trainer-parity fla kernels = {enable}")


class VLLMGatedDeltaNetCore(Module, MambaBase):
    """Paged-cache GDN core for the unified (torchtitan_wrapper) path.

    Holds NO learnable parameters -- projections, conv weights, gates, norm and
    out_proj all live in the enclosing ``GatedDeltaNet``. This layer only owns
    the vLLM cache plumbing (state discovery + paged conv/ssm state) and runs
    vLLM's native GDN helper kernels against it. Decode uses the recurrent update
    path; prefill uses the varlen chunk path, matching vLLM native's split.

    Non-speculative decoding only (asserts otherwise). The recurrent
    batch-invariant path also supports decode cudagraphs.

    Prefix caching: validated for BOTH the vendored path and the recurrent-everywhere
    trainer-parity path. With ``enable_prefix_caching=True`` vLLM auto-selects
    ``mamba_cache_mode='align'`` for this wrapper (it does not declare
    ``supports_mamba_prefix_caching``), caching the conv/ssm state at block
    boundaries. On the trainer-parity path a reused prefix arrives as a continuation
    (``prefill_has_initial_state``); ``_forward_recurrent_bi`` restores the cached
    conv+ssm state and continues the fla recurrence, which is bitwise-equal to a
    fresh full prefill (causal recurrence -> identical boundary state). A repro
    (generate a prompt twice) confirms the second run hits the cache
    (num_cached_tokens > 0) and reproduces the no-cache output token-for-token.
    """

    _logged_parity = False

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        num_k_heads: int
        num_v_heads: int
        head_k_dim: int
        head_v_dim: int
        conv_kernel_size: int = 4
        activation: str = "silu"

    def __init__(self, config: Config) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        self.num_spec = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )

        # TODO(qwen3.5-gdn-unified-tp): support TP by passing local head counts
        # and using the local fused conv/projection width. Current unified GDN
        # generation is validated only for pure DP / TP=1.
        if self.tp_size != 1:
            raise ValueError(
                "VLLMGatedDeltaNetCore currently supports tensor_parallel_size=1 "
                f"only, got tensor_parallel_size={self.tp_size}."
            )

        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.activation = config.activation

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        _, ssm_dtype = self.get_state_dtype()
        if ssm_dtype != torch.float32:
            raise ValueError(
                "VLLMGatedDeltaNetCore requires mamba_ssm_cache_dtype='float32' "
                f"for the triton/FLA recurrent state, got {ssm_dtype}."
            )

        # vLLM populates this via the KV-cache allocator: (conv_state, ssm_state).
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.prefix = f"model.layers.{config.layer_idx}.linear_attn"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate GDN layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    # ---- MambaBase contract ------------------------------------------------
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def _seed_conv_state(
        self,
        conv_state: torch.Tensor,
        mixed_qkv_TC: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_idx: torch.Tensor,
        state_len: int,
    ) -> None:
        """Seed the paged conv_state for freshly-prefilled sequences.

        The trainer-parity prefill runs the external fla conv (which does NOT touch
        vLLM's paged conv_state), so a subsequent decode step would read stale
        history. Write it here: ``conv_state[idx]`` is ``(conv_dim, state_len)``
        holding the last ``state_len`` PRE-conv inputs, oldest at index 0 -- exactly
        what vLLM's ``causal_conv1d_fn`` stores and ``causal_conv1d_update`` reads.
        Sequences shorter than ``state_len`` are left zero-padded on the left.
        Vectorized (no ``.tolist()``/python loop -> no per-layer GPU->CPU sync):
        gather each segment's last ``state_len`` PRE-conv inputs and scatter into
        ``conv_state[state_idx]``.
        """
        starts = cu_seqlens[:-1]  # [n_seq]
        ends = cu_seqlens[1:]  # [n_seq]
        # token index for window pos j (0=oldest): end - state_len + j
        offs = torch.arange(state_len, device=mixed_qkv_TC.device)
        tok_idx = ends[:, None] - state_len + offs[None, :]  # [n_seq, state_len]
        in_seg = tok_idx >= starts[:, None]  # False = left-pad (short segment)
        gathered = mixed_qkv_TC[tok_idx.clamp_min(0)]  # [n_seq, state_len, conv_dim]
        gathered = gathered * in_seg[:, :, None].to(gathered.dtype)
        # conv_state[idx] is (conv_dim, state_len) -> transpose the gathered window.
        conv_state[state_idx] = gathered.transpose(1, 2).to(conv_state.dtype)

    # ---- recurrent-everywhere BI path -------------------------------------
    def _split_qkv(
        self, mixed_qkv_slice_TC: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = mixed_qkv_slice_TC.shape[0]
        q = (
            mixed_qkv_slice_TC[:, : self.key_dim]
            .contiguous()
            .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
        )
        k = (
            mixed_qkv_slice_TC[:, self.key_dim : 2 * self.key_dim]
            .contiguous()
            .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
        )
        v = (
            mixed_qkv_slice_TC[:, 2 * self.key_dim :]
            .float()
            .view(1, num_tokens, self.num_v_heads, self.head_v_dim)
        )
        return q, k, v

    def _forward_recurrent_bi(
        self,
        m: GDNAttentionMetadata,
        n: int,
        mixed_qkv_TC: torch.Tensor,
        a_THv: torch.Tensor,
        b_THv: torch.Tensor,
        *,
        out_1THvDv: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Recurrent-everywhere GDN: fla conv + fla RECURRENT kernel for BOTH prefill
        and decode, so decode == prefill == trainer bitwise.

        Conv (fla causal_conv1d / causal_conv1d_update) and recurrence (fla
        fused_recurrent) both run against vLLM's PAGED conv_state + ssm_state via
        gather/scatter, so vLLM's prefix cache saves/restores them at block
        boundaries. Fresh sequences start from a zero initial state; prefix-cache
        continuation sequences (m.prefill_has_initial_state) restore the cached
        conv+ssm state and continue the recurrence -- bitwise-equal to a fresh full
        prefill because the recurrence is causal (the boundary state is identical).
        """
        conv_dim = mixed_qkv_TC.shape[1]
        ssm_state = self.kv_cache[1]
        nsi = m.non_spec_state_indices_tensor
        assert nsi is not None
        # Use vLLM's paged conv_state (kv_cache[0]) directly so a prefix-cache hit
        # restores the reused prefix's conv history. Stored [.., conv_dim, W-1] (the
        # last W-1 pre-conv inputs); fla's causal_conv1d state is [N, conv_dim, W]
        # whose trailing W-1 columns equal it and whose column 0 is a don't-care
        # (verified) -- pad a zero column on read, drop it on write.
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_W = self.conv_kernel_size

        def _pad_w1_to_w(state_w1: torch.Tensor) -> torch.Tensor:
            # [n, conv_dim, W-1] -> [n, conv_dim, W]; leading column is a don't-care.
            pad = state_w1.new_zeros(state_w1.shape[0], state_w1.shape[1], 1)
            return torch.cat([pad, state_w1], dim=-1)

        # Batch-independent fp32 gate scalars: hoist out of the per-segment closure
        # (identical fp32 ops in identical order -> pure CSE, bitwise-unchanged).
        A_neg_exp = -torch.exp(A_log.float())
        dt_bias_f = dt_bias.float()

        def _recurrence(
            conv_out_TC: torch.Tensor,
            seg: slice,
            slot_idx: torch.Tensor,
            cu_seqlens: torch.Tensor,
            ssm_init: str,
            ssm_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            q, k, v = self._split_qkv(conv_out_TC)
            # Keep V in fp32 so fla allocates its output in fp32. The kernel loads Q/K
            # into fp32 registers itself, so materializing fp32 Q/K tensors is redundant.
            # The fp32 output is what makes the recurrence stepped-consistent before it
            # is written back to the bf16 activation buffer; the trainer does the same.
            # fp32 eager gate, identical to the trainer.
            g = (A_neg_exp * F.softplus(a_THv[seg].float() + dt_bias_f)).unsqueeze(0)
            beta = torch.sigmoid(b_THv[seg].float()).unsqueeze(0)
            # Recurrent initial state (fp32). Always a materialized tensor so the
            # USE_INITIAL_STATE constexpr picks the SAME compiled kernel across
            # fresh/continuation/decode (-> bitwise). paged ssm_state is [.., V, K];
            # fla wants [.., K, V]. Three modes:
            #   "all"  (decode): every seq resumes -> a fused indexed copy, which is
            #          cudagraph-capturable (NO boolean/data-dependent indexing).
            #   "mask" (mixed prefill, eager -- not captured): per-seq boolean restore.
            #   "zero" (fresh prefill): zeros.
            if ssm_init == "all":
                # Fuse the paged gather and V/K transpose into one copy kernel. The
                # resulting contiguous tensor is byte-identical to PyTorch's
                # advanced-index + transpose + fp32 + contiguous sequence, so FLA
                # itself and all recurrent arithmetic remain unchanged.
                initial_state = gather_transposed_paged_state(ssm_state, slot_idx)
            else:
                n_seq = int(cu_seqlens.numel()) - 1
                initial_state = q.new_zeros(
                    n_seq, v.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
                )
                if ssm_init == "mask" and ssm_mask is not None:
                    initial_state[ssm_mask] = (
                        ssm_state[slot_idx[ssm_mask]].transpose(-1, -2).float()
                    )
            out, final_state = _external_fla_fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[slot_idx] = final_state.transpose(-1, -2).to(ssm_state.dtype)
            return out

        num_decode_tokens = m.num_decode_tokens

        # Decode segment: 1 token/seq; resume conv + ssm from the paged cache.
        if m.num_decodes > 0:
            dec_slots = nsi[: m.num_decodes]
            cache_w = _pad_w1_to_w(conv_state[dec_slots])  # [num_decodes, conv_dim, W]
            conv_out, cache_w = _external_fla_causal_conv1d_update(
                mixed_qkv_TC[:num_decode_tokens],
                cache_w,
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
            )
            conv_state[dec_slots] = cache_w[..., 1:].to(conv_state.dtype)
            out_1THvDv[:, :num_decode_tokens] = _recurrence(
                conv_out,
                slice(0, num_decode_tokens),
                dec_slots,
                m.non_spec_query_start_loc[: m.num_decodes + 1],
                "all",
            )

        # Prefill segment: fresh AND/OR prefix-cache continuation sequences.
        if m.num_prefills > 0:
            assert m.prefill_state_indices is not None
            pf_state_idx = m.prefill_state_indices
            pf_has_init = m.prefill_has_initial_state  # per-seq bool, or None (fresh)
            pf_start = num_decode_tokens if m.num_decodes > 0 else 0
            if m.num_decodes == 0:
                pf_cu = m.non_spec_query_start_loc  # 0-based (verified prefill path)
            else:
                assert m.prefill_query_start_loc is not None
                pf_cu = m.prefill_query_start_loc - m.prefill_query_start_loc[0]
            n_pf = int(pf_cu.numel()) - 1
            # Per-sequence conv initial state [n_pf, conv_dim, W]: continuation rows
            # restore the paged W-1 history (padded to W); fresh rows stay zero (==
            # no-init, verified bitwise).
            # Prefill runs eager (FULL_DECODE_ONLY graphs only pure-decode), so the
            # host sync in .any() and the boolean-mask restore below are fine here.
            _has_cont = pf_has_init is not None and bool(pf_has_init.any())
            conv_init = mixed_qkv_TC.new_zeros(n_pf, conv_dim, conv_W)
            if _has_cont:
                conv_init[pf_has_init, :, 1:] = conv_state[pf_state_idx[pf_has_init]]
            conv_out, conv_final = _external_fla_causal_conv1d(
                mixed_qkv_TC[pf_start:n].unsqueeze(0),
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
                cu_seqlens=pf_cu,
                initial_state=conv_init,
                output_final_state=True,
            )
            conv_out = conv_out.squeeze(0)
            conv_state[pf_state_idx] = conv_final[..., 1:].to(conv_state.dtype)
            out_1THvDv[:, pf_start:n] = _recurrence(
                conv_out,
                slice(pf_start, n),
                pf_state_idx,
                pf_cu,
                "mask" if _has_cont else "zero",
                pf_has_init,
            )

        return out_1THvDv

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        mixed_qkv_BTC: torch.Tensor,
        a_BTHv: torch.Tensor,
        b_BTHv: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """conv + gated-delta recurrence against vLLM's paged state.

        Args:
            mixed_qkv_BTC: (bs, seqlen, key_dim*2 + value_dim) -- concat of the
                q/k/v projections, PRE-conv (bs==1 under vLLM: one flattened row).
            a_BTHv: (bs, seqlen, num_v_heads) alpha gate input.
            b_BTHv: (bs, seqlen, num_v_heads) beta gate input.
            A_log: (num_v_heads,) decay parameter.
            dt_bias: (num_v_heads,) dt bias parameter.
            conv_weight: fused depthwise conv weight (conv_dim, kernel_size).
            conv_bias: fused conv bias (conv_dim,) or None.

        Returns:
            (bs, seqlen, num_v_heads, head_v_dim) core output (pre gated-norm).
        """
        bs, seqlen, conv_dim = mixed_qkv_BTC.shape
        if bs != 1:
            raise ValueError(
                "VLLMGatedDeltaNetCore expects vLLM's flattened batch layout "
                f"with batch size 1, got batch size {bs}."
            )
        out_BTHvDv = mixed_qkv_BTC.new_zeros(
            bs, seqlen, self.num_v_heads, self.head_v_dim
        )

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        # Dummy run (profiling / warmup): no metadata -> nothing to compute.
        if attn_metadata_raw is None:
            return out_BTHvDv
        assert isinstance(attn_metadata_raw, dict)
        m = attn_metadata_raw[self.prefix]
        assert isinstance(m, GDNAttentionMetadata)
        assert (
            m.spec_sequence_masks is None
        ), "VLLMGatedDeltaNetCore does not support speculative decoding"

        n = m.num_actual_tokens
        if n > seqlen:
            raise ValueError(
                "VLLMGatedDeltaNetCore received more actual tokens than the "
                f"flattened input length: num_actual_tokens={n}, seqlen={seqlen}."
            )
        if n == 0:
            return out_BTHvDv

        # Flatten (bs==1) to the token layout vLLM's kernels use: (T, C).
        mixed_qkv_TC = mixed_qkv_BTC.reshape(bs * seqlen, conv_dim)[:n]
        a_THv = a_BTHv.reshape(bs * seqlen, self.num_v_heads)[:n]
        b_THv = b_BTHv.reshape(bs * seqlen, self.num_v_heads)[:n]

        # conv_state stored (dim, width-1) in DS layout; SD layout needs a
        # transpose so the conv kernels see (..., dim, width-1).
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self.kv_cache[1]
        nsi = m.non_spec_state_indices_tensor  # one state slot per sequence
        assert nsi is not None
        num_seq = m.num_decodes + m.num_prefills
        assert m.non_spec_query_start_loc is not None
        assert m.non_spec_query_start_loc.numel() >= num_seq + 1
        assert nsi.numel() >= num_seq

        # Recurrent-everywhere: fla conv + fla RECURRENT kernel for prefill AND decode
        # so decode == prefill == trainer bitwise (decode is inherently recurrent, so
        # the only way prefill/trainer can match it is to use that same kernel -- not
        # chunk). Gated on the trainer-parity path (fla kernels) + batch-invariant
        # mode; the trainer forward is likewise swapped to fla recurrent under BI
        # (model.py _RecurrentFwdChunkBwd). This supersedes the chunk-prefill parity
        # branch below when both are on.
        if _TRAINER_PARITY_FLA and is_in_batch_invariant_mode():
            self._forward_recurrent_bi(
                m,
                n,
                mixed_qkv_TC,
                a_THv,
                b_THv,
                out_1THvDv=out_BTHvDv[:, :n],
                A_log=A_log,
                dt_bias=dt_bias,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
            )
            return out_BTHvDv

        if _TRAINER_PARITY_FLA and not VLLMGatedDeltaNetCore._logged_parity:
            VLLMGatedDeltaNetCore._logged_parity = True
            logger.info(
                "[gdn-unified] GDN prefill on trainer external fla kernels "
                "(bitwise vs trainer); decode continues from the seeded paged state"
            )

        # split helper (nested; used by both the trainer-parity and vendored paths)
        out_1THvDv = mixed_qkv_TC.new_empty(1, n, self.num_v_heads, self.head_v_dim)

        def _split_qkv(
            mixed_qkv_slice_TC: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = mixed_qkv_slice_TC.shape[0]
            q_1THkDk = (
                mixed_qkv_slice_TC[:, : self.key_dim]
                .contiguous()
                .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
            )
            k_1THkDk = (
                mixed_qkv_slice_TC[:, self.key_dim : 2 * self.key_dim]
                .contiguous()
                .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
            )
            v_1THvDv = (
                mixed_qkv_slice_TC[:, 2 * self.key_dim :]
                .contiguous()
                .view(1, num_tokens, self.num_v_heads, self.head_v_dim)
            )
            return q_1THkDk, k_1THkDk, v_1THvDv

        # ================= trainer-parity path (external fla) =================
        # Fresh prefill matches the trainer op-for-op (external fla conv + external
        # fla chunk with initial_state=None), so the generator's PREFILL logprobs are
        # bitwise-identical to the trainer's. Decode has no trainer equivalent (the
        # trainer never decodes): it only has to stay correct and CONTINUE from the
        # paged state, so it uses vLLM's single-step conv update + the external fla
        # chunk on length-1 segments. Fresh prefill SEEDS the paged conv_state so the
        # next decode step reads valid history. Chunked-prefill / prefix-cache
        # continuation has no clean trainer match -> vendored fallback (correct, not
        # bitwise). See GDN_BITWISE_PARITY.md.
        if _TRAINER_PARITY_FLA:
            state_len = self.conv_kernel_size - 1
            num_decode_tokens = m.num_decode_tokens

            def _fla_recurrence(
                conv_out_slice_TC: torch.Tensor,
                a_slice_THv: torch.Tensor,
                b_slice_THv: torch.Tensor,
                state_idx: torch.Tensor,
                cu_seqlens: torch.Tensor,
                initial_state: torch.Tensor | None,
            ) -> torch.Tensor:
                # Eager fp32 gate (model.py:514-517), RAW q/k with in-kernel l2norm,
                # GVA head-expand (model.py:243-247), then the trainer's EXTERNAL fla
                # chunk -- gate + l2norm + recurrence all match the trainer.
                q, k, v = _split_qkv(conv_out_slice_TC)
                if q.shape[2] != v.shape[2]:
                    rep = v.shape[2] // q.shape[2]
                    q = q.repeat_interleave(rep, dim=2)
                    k = k.repeat_interleave(rep, dim=2)
                g = (
                    -torch.exp(A_log.float())
                    * F.softplus(a_slice_THv.float() + dt_bias.float())
                ).unsqueeze(0)
                beta = torch.sigmoid(b_slice_THv.float()).unsqueeze(0)
                out, final_state = _external_fla_chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
                # Paged ssm_state is value-first [.., HV, V, K]; fla state is key-first
                # [.., HV, K, V]. transpose(-1, -2) maps between them (also correct when
                # head_k_dim != head_v_dim). transpose_state_layout stays False (trainer
                # default) so the compute path matches; transpose the storage instead.
                ssm_state[state_idx] = final_state.transpose(-1, -2).to(ssm_state.dtype)
                return out

            # ---- decode segment (front slice): keep generation correct only ----
            if num_decode_tokens > 0:
                dec_state_idx = nsi[:num_decode_tokens]
                dec_conv_TC = causal_conv1d_update(
                    mixed_qkv_TC[:num_decode_tokens],
                    conv_state,
                    conv_weight,
                    conv_bias,
                    self.activation,
                    conv_state_indices=dec_state_idx,
                    validate_data=False,
                )
                dec_init = ssm_state[dec_state_idx].transpose(-1, -2).contiguous()
                out_1THvDv[:, :num_decode_tokens] = _fla_recurrence(
                    dec_conv_TC,
                    a_THv[:num_decode_tokens],
                    b_THv[:num_decode_tokens],
                    dec_state_idx,
                    m.non_spec_query_start_loc[: m.num_decodes + 1],
                    dec_init,
                )

            # ---- prefill segment (back slice): bitwise-matches the trainer ----
            if m.num_prefills > 0:
                assert m.prefill_query_start_loc is not None
                assert m.prefill_state_indices is not None
                pf_start = num_decode_tokens
                pf_state_idx = m.prefill_state_indices
                pf_cu = m.prefill_query_start_loc
                pf_mixed_TC = mixed_qkv_TC[pf_start:n]
                has_init = m.prefill_has_initial_state
                if has_init is not None and bool(has_init.any()):
                    # Continuation (chunked prefill): no clean trainer equivalent ->
                    # vLLM paged conv + cached-state chunk (correct, not bitwise for
                    # these tokens). Scope the conv to the PREFILL slice only
                    # (pf_mixed_TC + prefill-rebased query_start_loc / state_indices /
                    # has_initial_state, which vLLM builds as the num_decodes: tail).
                    # The decode segment above already ran causal_conv1d_update on the
                    # decode tokens; re-running conv over the FULL batch here would
                    # advance the decode sequences' conv_state a SECOND time and
                    # corrupt their later decode steps. metadata=None -> the kernel
                    # builds its grid from the prefill-only query_start_loc.
                    pf_conv_TC = causal_conv1d_fn(
                        pf_mixed_TC.transpose(0, 1),
                        conv_weight,
                        conv_bias,
                        activation=self.activation,
                        conv_states=conv_state,
                        has_initial_state=has_init,
                        cache_indices=pf_state_idx,
                        query_start_loc=pf_cu,
                    ).transpose(0, 1)
                    pf_init = ssm_state[pf_state_idx].clone()
                    pf_init[~has_init] = 0
                    q, k, v, g, beta = fused_post_conv_prep(
                        conv_output=pf_conv_TC,
                        a=a_THv[pf_start:n],
                        b=b_THv[pf_start:n],
                        A_log=A_log,
                        dt_bias=dt_bias,
                        num_k_heads=self.num_k_heads,
                        head_k_dim=self.head_k_dim,
                        head_v_dim=self.head_v_dim,
                        apply_l2norm=True,
                        output_g_exp=False,
                    )
                    out, final_state = _vllm_chunk_gated_delta_rule(
                        q.unsqueeze(0),
                        k.unsqueeze(0),
                        v.unsqueeze(0),
                        g.unsqueeze(0),
                        beta.unsqueeze(0),
                        initial_state=pf_init,
                        output_final_state=True,
                        cu_seqlens=pf_cu,
                        chunk_indices=m.chunk_indices,
                        chunk_offsets=m.chunk_offsets,
                        use_qk_l2norm_in_kernel=False,
                    )
                    out_1THvDv[:, pf_start:n] = out
                    ssm_state[pf_state_idx] = final_state.to(ssm_state.dtype)
                else:
                    # Fresh full prefill = the trainer's packed path. external fla conv
                    # + external fla chunk (initial_state=None, NOT a zero tensor: fla's
                    # USE_INITIAL_STATE constexpr branches differently) match the trainer.
                    pf_conv_TC = _external_fla_causal_conv1d(
                        pf_mixed_TC.unsqueeze(0),
                        weight=conv_weight,
                        bias=conv_bias,
                        activation="silu",
                        cu_seqlens=pf_cu,
                    )
                    if isinstance(pf_conv_TC, tuple):
                        pf_conv_TC = pf_conv_TC[0]
                    pf_conv_TC = pf_conv_TC.squeeze(0)
                    out_1THvDv[:, pf_start:n] = _fla_recurrence(
                        pf_conv_TC,
                        a_THv[pf_start:n],
                        b_THv[pf_start:n],
                        pf_state_idx,
                        pf_cu,
                        None,
                    )
                    # Seed paged conv_state so a later decode step continues correctly.
                    self._seed_conv_state(
                        conv_state, pf_mixed_TC, pf_cu, pf_state_idx, state_len
                    )

            out_BTHvDv[:, :n] = out_1THvDv[:, :n]
            return out_BTHvDv

        # ===================== vendored path (flag off) =======================
        # 1) Causal conv against the paged conv_state (reuse vLLM's kernels). A
        # batch with any prefill uses the varlen fn; pure-decode uses the update.
        # STEP 3: a PURE-FRESH prefill batch (no decodes, no continuation) uses the
        # trainer's fla conv (bitwise-aligned) + seeds the paged conv_state for a
        # later decode step; mixed/continuation keep vLLM's paged conv kernels.
        _pf_has_init = m.prefill_has_initial_state
        _pure_fresh_prefill = (
            m.num_prefills > 0
            and m.num_decodes == 0
            and (_pf_has_init is None or not bool(_pf_has_init.any()))
        )
        if _pure_fresh_prefill:
            conv_out_TC = _external_fla_causal_conv1d(
                mixed_qkv_TC.unsqueeze(0),
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
                cu_seqlens=m.prefill_query_start_loc,
            )
            if isinstance(conv_out_TC, tuple):
                conv_out_TC = conv_out_TC[0]
            conv_out_TC = conv_out_TC.squeeze(0)
            # fla conv is stateless -> seed the paged conv_state so decode continues.
            self._seed_conv_state(
                conv_state,
                mixed_qkv_TC,
                m.prefill_query_start_loc,
                m.prefill_state_indices,
                self.conv_kernel_size - 1,
            )
        elif m.num_prefills > 0:
            conv_out_TC = causal_conv1d_fn(
                mixed_qkv_TC.transpose(0, 1),
                conv_weight,
                conv_bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=m.has_initial_state,
                cache_indices=nsi,
                query_start_loc=m.non_spec_query_start_loc,
                metadata=m,
            ).transpose(0, 1)
        else:
            conv_out_TC = causal_conv1d_update(
                mixed_qkv_TC,
                conv_state,
                conv_weight,
                conv_bias,
                self.activation,
                conv_state_indices=nsi[:n],
                validate_data=False,
            )

        # 2) Recurrent attention against paged state. Match vLLM native's split:
        # decode tokens use a recurrent update path, while prefill tokens use the
        # chunk kernel. This avoids routing mixed-batch length-1 decodes through
        # the prefill kernel and keeps the unified path closer to native GDN.
        def _run_decode_sigmoid_update(
            start: int,
            end: int,
            state_idx: torch.Tensor,
            cu_seqlens: torch.Tensor,
        ) -> None:
            if end <= start:
                return
            q, k, v = _split_qkv(conv_out_TC[start:end])
            out, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a_THv[start:end],
                b=b_THv[start:end],
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=state_idx,
                use_qk_l2norm_in_kernel=True,
            )
            out_1THvDv[:, start:end] = out

        def _run_packed_decode(end: int, state_idx: torch.Tensor) -> None:
            if end <= 0:
                return
            out_T1HvDv = out_1THvDv[0, :end].unsqueeze(1)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=conv_out_TC[:end].contiguous(),
                a=a_THv[:end].contiguous(),
                b=b_THv[:end].contiguous(),
                A_log=A_log,
                dt_bias=dt_bias,
                scale=self.head_k_dim**-0.5,
                initial_state=ssm_state,
                out=out_T1HvDv,
                ssm_state_indices=state_idx,
                use_qk_l2norm_in_kernel=True,
            )

        def _run_prefill_chunk(
            start: int,
            end: int,
            state_idx: torch.Tensor,
            has_initial_state: torch.Tensor | None,
            cu_seqlens: torch.Tensor,
        ) -> None:
            if end <= start:
                return
            # STEP 3: fresh -> initial_state=None. fla's USE_INITIAL_STATE is a
            # triton constexpr (+ autotune key): None and an all-zero tensor
            # dispatch DIFFERENT compiled kernels, so a zero tensor is NOT bitwise
            # with the trainer's None. Continuation carries the paged state
            # (value-first [.., HV, V, K] -> transpose to fla key-first [.., HV, K, V]).
            if has_initial_state is None or not bool(has_initial_state.any()):
                initial_state = None
            else:
                initial_state = ssm_state[state_idx].transpose(-1, -2).contiguous()
                initial_state[~has_initial_state] = 0
            # STEP 2 (own-kernel migration): eager fp32 gate + RAW q/k with in-kernel
            # l2norm + GVA head-expand, matching the trainer -- instead of vLLM's
            # fused_post_conv_prep (which rounds the l2norm in a different kernel).
            # The gate depends only on a/b (state-independent), so this applies to
            # BOTH fresh and continuation.
            q, k, v = _split_qkv(conv_out_TC[start:end])
            if q.shape[2] != v.shape[2]:
                rep = v.shape[2] // q.shape[2]
                q = q.repeat_interleave(rep, dim=2)
                k = k.repeat_interleave(rep, dim=2)
            g = (
                -torch.exp(A_log.float())
                * F.softplus(a_THv[start:end].float() + dt_bias.float())
            ).unsqueeze(0)
            beta = torch.sigmoid(b_THv[start:end].float()).unsqueeze(0)
            # STEP 1: our fla chunk (same as trainer + parity), not vLLM's vendored
            # chunk. RAW q/k -> use_qk_l2norm_in_kernel=True; fla derives
            # chunk_indices from cu_seqlens internally.
            out, final_state = _external_fla_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            out_1THvDv[:, start:end] = out
            ssm_state[state_idx] = final_state.transpose(-1, -2).to(ssm_state.dtype)

        num_decode_tokens = m.num_decode_tokens
        if m.num_decodes > 0:
            if m.num_prefills == 0:
                _run_packed_decode(num_decode_tokens, nsi[:num_decode_tokens])
            else:
                _run_decode_sigmoid_update(
                    0,
                    num_decode_tokens,
                    nsi[: m.num_decodes],
                    m.non_spec_query_start_loc[: m.num_decodes + 1],
                )

        if m.num_prefills > 0:
            assert m.prefill_query_start_loc is not None
            assert m.prefill_state_indices is not None
            prefill_start = num_decode_tokens if m.num_decodes > 0 else 0
            _run_prefill_chunk(
                prefill_start,
                n,
                m.prefill_state_indices,
                m.prefill_has_initial_state,
                m.prefill_query_start_loc,
            )

        out_BTHvDv[:, :n] = out_1THvDv[:, :n]
        return out_BTHvDv


def log_unified_gdn_active() -> None:
    logger.info(
        "[gdn-unified] GatedDeltaNet running as a TorchTitan unified layer with "
        "vLLM paged cache management and native GDN helper kernels"
    )

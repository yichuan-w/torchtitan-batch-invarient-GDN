# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the GDN (Gated DeltaNet) generation layer on TorchTitan's own FLA kernels.

Motivation: the vLLM-native Qwen3.5 generation path runs vLLM's vendored FLA GDN
kernels, while the trainer runs the standalone ``fla`` package (see
``torchtitan/models/qwen3_5/model.py``). Two different implementations of the
same recurrence is a source of train/infer logprob drift. This module swaps ONLY
the GDN linear-attention layer's recurrence to the SAME ``fla`` ops the trainer
uses, while reusing all of vLLM's hybrid machinery (paged conv/ssm state cache,
GDN attention metadata, weight loading). Everything else (full attention, MoE,
projections) stays vLLM-native.

Mechanism: vLLM's ``QwenGatedDeltaNetAttention`` is a ``PluggableLayer``, so we
register an out-of-tree subclass under its name; vLLM then instantiates ours
wherever it would build the native layer. We override only ``_forward_core`` --
the conv + recurrence numeric core -- and keep the native ``__init__`` (params,
projections, cache spec) unchanged.

Scope: non-speculative decoding only (asserts otherwise). Requires the generator
cudagraph to be disabled (the generator hook forces this when enabled).

Opt-in via env ``TT_GDN_UNIFIED_KERNEL=1`` (default off -> vLLM-native, easy
fallback). Call ``register_titan_gdn()`` before the vLLM engine is built.
"""

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
)

from torchtitan.tools.logging import logger

from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import op_registry_oot, PluggableLayer
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# The vLLM class name we shadow. vLLM's Qwen3.5 model instantiates this class
# directly (gqa_interleaved_layout=False), so registering under this name is
# enough to route generation through us.
_VLLM_GDN_CLASS_NAME = "QwenGatedDeltaNetAttention"


class TitanFLAGatedDeltaNet(QwenGatedDeltaNetAttention):
    """vLLM GDN layer whose recurrence runs the trainer's ``fla`` chunk kernel.

    Inherits ``__init__`` and everything else from the native layer; only the
    conv+recurrence core is overridden. Prefill and decode are handled uniformly
    by a single varlen ``chunk_gated_delta_rule`` call (decode = length-1
    sequences), which is exactly the kernel the trainer uses -- so both sides run
    identical FLA math.
    """

    _logged_active = False

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        # Dummy run (profiling / warmup): no metadata, nothing to compute.
        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        m = attn_metadata_raw[self.prefix]
        assert isinstance(m, GDNAttentionMetadata)
        assert (
            m.spec_sequence_masks is None
        ), "TitanFLAGatedDeltaNet does not support speculative decoding"

        if not TitanFLAGatedDeltaNet._logged_active:
            TitanFLAGatedDeltaNet._logged_active = True
            logger.info("[gdn-titan] GDN generation running on TorchTitan fla kernels")

        n = m.num_actual_tokens
        if n == 0:
            return
        mixed_qkv = mixed_qkv[:n]
        b = b[:n]
        a = a[:n]

        # conv_state is stored (dim, width-1) in DS layout; SD layout needs a
        # transpose so the conv kernels see (..., dim, width-1).
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self.kv_cache[1]  # [num_blocks, HV, head_v_dim, head_k_dim]
        nsi = m.non_spec_state_indices_tensor  # one state slot per sequence
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # 1) Causal conv (reuse vLLM's paged conv-state kernels). A batch with
        # any prefill goes through the varlen fn (it also covers peeled decode
        # tokens at the front); a pure-decode batch uses the single-step update.
        if m.num_prefills > 0:
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=m.has_initial_state,
                cache_indices=nsi,
                query_start_loc=m.non_spec_query_start_loc,
                metadata=m,
            ).transpose(0, 1)
        else:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=nsi[:n],
                validate_data=True,
            )

        # 2) Split q/k/v and compute the gates with the trainer's fp32 formula
        # (bf16 sigmoid on beta has large relative error near saturation).
        q, k, v = self.rearrange_mixed_qkv(mixed_qkv)  # [1, n, H*, D*]
        g = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        beta = torch.sigmoid(b.float())
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)

        # Grouped value attention: expand q/k key heads to value heads, matching
        # the trainer kernel (fla also handles HV>H internally; we expand for an
        # exact match to the trainer path).
        if q.shape[2] != v.shape[2]:
            repeat = v.shape[2] // q.shape[2]
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        # 3) One varlen chunk call over the whole non-spec batch. Gather the
        # recurrent state per sequence (fresh prefills start from zero), run the
        # SAME chunk kernel as the trainer, then scatter the final state back.
        n_seq = m.num_decodes + m.num_prefills
        state_idx = nsi[:n_seq]
        # vLLM's paged ssm_state is [.., HV, head_v_dim, head_k_dim] (value first),
        # but fla's chunk kernel uses key-first [.., HV, K, V] state. The trainer's
        # GatedDeltaKernel calls chunk with the DEFAULT transpose_state_layout=False,
        # so to match the trainer's forward NUMERICS (transpose_state_layout is a
        # kernel compute-path flag, not just a storage layout) we keep False here
        # and transpose the paged state around the call instead of flipping the
        # kernel layout. transpose(-1, -2) maps the [V, K] slot <-> fla's [K, V]
        # state; this is also correct when head_k_dim != head_v_dim (27B), unlike a
        # raw position-copy. (The prior `state_v_first=True` kwarg did not exist in
        # fla -- it was silently dropped by **kwargs, so this already ran at the
        # default False; the transposes just make the slot layout explicit + safe.)
        initial_state = ssm_state[state_idx].transpose(-1, -2).contiguous()
        if m.has_initial_state is not None:
            initial_state[~m.has_initial_state] = 0

        out, final_state = _fla_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=m.non_spec_query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[state_idx] = final_state.transpose(-1, -2).to(ssm_state.dtype)
        core_attn_out[:n] = out.squeeze(0)


def register_titan_gdn() -> None:
    """Route vLLM's GDN layer through TorchTitan fla kernels (idempotent)."""
    if _VLLM_GDN_CLASS_NAME in op_registry_oot:
        return
    PluggableLayer.register_oot(TitanFLAGatedDeltaNet, name=_VLLM_GDN_CLASS_NAME)
    logger.info(
        "[gdn-titan] registered TitanFLAGatedDeltaNet as OOT override for "
        f"{_VLLM_GDN_CLASS_NAME}; GDN generation will use TorchTitan fla kernels"
    )

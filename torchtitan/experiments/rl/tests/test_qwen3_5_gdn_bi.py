# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch-invariant Qwen3.5 GDN grouped-value-attention tests."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

import torchtitan.experiments.rl.models.gdn_vllm_unified as gdn_unified
import torchtitan.models.qwen3_5.model as qwen3_5_model


def test_recurrent_bi_uses_gva_forward_and_expanded_chunk_backward():
    batch_size, seq_len = 1, 3
    num_key_heads, num_value_heads, head_dim = 2, 4, 8
    q_BTHK = torch.randn(
        batch_size,
        seq_len,
        num_key_heads,
        head_dim,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k_BTHK = torch.randn_like(q_BTHK, requires_grad=True)
    v_BTHV = torch.randn(
        batch_size,
        seq_len,
        num_value_heads,
        head_dim,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    g_BTH = torch.randn(
        batch_size,
        seq_len,
        num_value_heads,
        dtype=torch.float32,
        requires_grad=True,
    )
    beta_BTH = torch.randn_like(g_BTH, requires_grad=True)

    def fake_recurrent(q, k, v, g, *, initial_state, **kwargs):
        assert q.shape[2] == num_key_heads
        assert k.shape[2] == num_key_heads
        assert v.shape[2] == num_value_heads
        assert q.dtype == k.dtype == torch.bfloat16
        assert v.dtype == torch.float32
        assert initial_state.shape == (
            batch_size,
            num_value_heads,
            head_dim,
            head_dim,
        )
        return v.float(), None

    def fake_chunk(q, k, v, g, beta, **kwargs):
        assert q.shape[2] == num_value_heads
        assert k.shape[2] == num_value_heads
        q_term_BTHV = q.float().mean(dim=-1, keepdim=True)
        k_term_BTHV = k.float().mean(dim=-1, keepdim=True)
        out_BTHV = (
            v.float() + q_term_BTHV + k_term_BTHV + g.unsqueeze(-1) + beta.unsqueeze(-1)
        )
        return out_BTHV, None

    with (
        patch.object(
            qwen3_5_model,
            "_fla_fused_recurrent_gated_delta_rule",
            side_effect=fake_recurrent,
        ),
        patch.object(
            qwen3_5_model,
            "_fla_chunk_gated_delta_rule",
            side_effect=fake_chunk,
        ),
    ):
        out_BTHV = qwen3_5_model._RecurrentFwdChunkBwd.apply(
            q_BTHK, k_BTHK, v_BTHV, g_BTH, beta_BTH, None
        )
        out_BTHV.float().sum().backward()

    for tensor in (q_BTHK, k_BTHK, v_BTHV, g_BTH, beta_BTH):
        assert tensor.grad is not None


def test_torch_native_still_expands_grouped_value_heads_in_bi():
    num_key_heads, num_value_heads, head_dim = 2, 4, 8
    kernel = object.__new__(qwen3_5_model.GatedDeltaKernel)
    torch.nn.Module.__init__(kernel)
    kernel.backend = "torch_native"
    q_BTHK = torch.randn(1, 3, num_key_heads, head_dim)
    k_BTHK = torch.randn_like(q_BTHK)
    v_BTHV = torch.randn(1, 3, num_value_heads, head_dim)
    g_BTH = torch.randn(1, 3, num_value_heads)
    beta_BTH = torch.randn_like(g_BTH)

    def fake_torch_native(q, k, v, g, beta, **kwargs):
        assert q.shape[2] == k.shape[2] == v.shape[2] == num_value_heads
        return v

    with (
        patch.object(qwen3_5_model, "is_in_batch_invariant_mode", return_value=True),
        patch.object(
            qwen3_5_model,
            "_torch_native_gated_delta",
            side_effect=fake_torch_native,
        ),
    ):
        out_BTHV = kernel(q_BTHK, k_BTHK, v_BTHV, g_BTH, beta_BTH)

    assert out_BTHV is v_BTHV


def test_generator_recurrent_bi_zero_state_uses_value_heads():
    num_tokens = 3
    num_key_heads, num_value_heads = 2, 4
    key_head_dim = value_head_dim = 8
    conv_kernel_size = 4
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    conv_dim = 2 * key_dim + value_dim

    core = object.__new__(gdn_unified.VLLMGatedDeltaNetCore)
    torch.nn.Module.__init__(core)
    core.num_k_heads = num_key_heads
    core.num_v_heads = num_value_heads
    core.head_k_dim = key_head_dim
    core.head_v_dim = value_head_dim
    core.key_dim = key_dim
    core.value_dim = value_dim
    core.conv_kernel_size = conv_kernel_size
    core.kv_cache = (
        torch.zeros(1, conv_dim, conv_kernel_size - 1, dtype=torch.bfloat16),
        torch.zeros(
            1,
            num_value_heads,
            value_head_dim,
            key_head_dim,
            dtype=torch.float32,
        ),
    )

    metadata = SimpleNamespace(
        non_spec_state_indices_tensor=torch.tensor([0]),
        num_decode_tokens=0,
        num_decodes=0,
        num_prefills=1,
        prefill_state_indices=torch.tensor([0]),
        prefill_has_initial_state=None,
        non_spec_query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        prefill_query_start_loc=None,
    )
    mixed_qkv_TC = torch.randn(num_tokens, conv_dim, dtype=torch.bfloat16)
    a_TH = torch.randn(num_tokens, num_value_heads, dtype=torch.bfloat16)
    b_TH = torch.randn_like(a_TH)

    def fake_conv(x, **kwargs):
        final_state = torch.zeros(1, conv_dim, conv_kernel_size, dtype=torch.bfloat16)
        return x, final_state

    def fake_recurrent(q, k, v, g, *, initial_state, **kwargs):
        assert q.shape[2] == num_key_heads
        assert k.shape[2] == num_key_heads
        assert v.shape[2] == num_value_heads
        assert q.dtype == k.dtype == torch.bfloat16
        assert v.dtype == torch.float32
        assert initial_state.shape == (
            1,
            num_value_heads,
            key_head_dim,
            value_head_dim,
        )
        final_state = torch.zeros_like(initial_state)
        return torch.zeros_like(v), final_state

    with (
        patch.object(gdn_unified, "is_conv_state_dim_first", return_value=True),
        patch.object(
            gdn_unified,
            "_external_fla_causal_conv1d",
            side_effect=fake_conv,
        ),
        patch.object(
            gdn_unified,
            "_external_fla_fused_recurrent_gated_delta_rule",
            side_effect=fake_recurrent,
        ),
    ):
        out_1THV = core._forward_recurrent_bi(
            metadata,
            num_tokens,
            mixed_qkv_TC,
            a_TH,
            b_TH,
            out_1THvDv=torch.empty(
                1,
                num_tokens,
                num_value_heads,
                value_head_dim,
                dtype=torch.bfloat16,
            ),
            A_log=torch.zeros(num_value_heads),
            dt_bias=torch.zeros(num_value_heads),
            conv_weight=torch.zeros(conv_dim, conv_kernel_size),
            conv_bias=None,
        )

    assert out_1THV.shape == (
        1,
        num_tokens,
        num_value_heads,
        value_head_dim,
    )

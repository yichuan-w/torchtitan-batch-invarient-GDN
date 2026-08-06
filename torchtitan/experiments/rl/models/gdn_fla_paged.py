# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor copies between vLLM's paged GDN state and FLA's state layout.

Shape suffix legend:
  N = num sequences, S = num cache slots, H = num value heads,
  K = key head dimension, V = value head dimension.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_transposed_paged_state_kernel(
    paged_state,
    initial_state,
    slot_indices,
    stride_s,
    stride_h,
    stride_v,
    stride_k,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_nh, i_block = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    offsets = i_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < K * V
    o_k, o_v = offsets // V, offsets % V

    slot = tl.load(slot_indices + i_n).to(tl.int64)
    initial_offset = i_nh * K * V
    values = tl.load(
        paged_state
        + slot * stride_s
        + i_h * stride_h
        + o_v * stride_v
        + o_k * stride_k,
        mask=mask,
        other=0,
    )
    tl.store(initial_state + initial_offset + offsets, values, mask=mask)


def gather_transposed_paged_state(
    paged_state_SHVK: torch.Tensor,
    slot_indices_N: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[H, V, K]`` slots as contiguous fp32 ``[N, H, K, V]``."""
    num_sequences = slot_indices_N.numel()
    _, num_heads, value_dim, key_dim = paged_state_SHVK.shape
    initial_state_NHKV = torch.empty(
        num_sequences,
        num_heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=paged_state_SHVK.device,
    )
    block_size = 1024
    grid = (
        num_sequences * num_heads,
        triton.cdiv(key_dim * value_dim, block_size),
    )
    _gather_transposed_paged_state_kernel[grid](
        paged_state=paged_state_SHVK,
        initial_state=initial_state_NHKV,
        slot_indices=slot_indices_N,
        stride_s=paged_state_SHVK.stride(0),
        stride_h=paged_state_SHVK.stride(1),
        stride_v=paged_state_SHVK.stride(2),
        stride_k=paged_state_SHVK.stride(3),
        H=num_heads,
        K=key_dim,
        V=value_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return initial_state_NHKV


__all__ = ["gather_transposed_paged_state"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA tests for GDN paged-state layout conversion."""

import pytest
import torch

from torchtitan.experiments.rl.models.gdn_fla_paged import gather_transposed_paged_state


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged-state conversion requires CUDA"
)


@pytest.mark.parametrize(
    (
        "num_slots",
        "num_heads",
        "value_dim",
        "key_dim",
        "slot_indices",
        "dtype",
        "noncontiguous",
    ),
    [
        (13, 4, 24, 32, [11], torch.float32, False),
        (19, 8, 40, 64, [17, 3, 12], torch.bfloat16, True),
        (
            23,
            32,
            128,
            128,
            [22, 0, 7, 15, 3, 9, 11, 5],
            torch.bfloat16,
            True,
        ),
    ],
)
def test_gather_transposed_paged_state_matches_pytorch(
    num_slots: int,
    num_heads: int,
    value_dim: int,
    key_dim: int,
    slot_indices: list[int],
    dtype: torch.dtype,
    noncontiguous: bool,
) -> None:
    physical_key_dim = key_dim + 7 if noncontiguous else key_dim
    storage = torch.randn(
        num_slots,
        num_heads,
        value_dim,
        physical_key_dim,
        device="cuda",
        dtype=dtype,
    )
    state_SHVK = storage[..., :key_dim]
    assert state_SHVK.is_contiguous() != noncontiguous
    slots_N = torch.tensor(slot_indices, device="cuda", dtype=torch.int32)

    expected_NHKV = state_SHVK[slots_N].transpose(-1, -2).float().contiguous()
    actual_NHKV = gather_transposed_paged_state(state_SHVK, slots_N)

    assert torch.equal(actual_NHKV, expected_NHKV)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for Qwen3.5 grouped MoE checkpoint conversion."""

import unittest

import torch

from torchtitan.models.qwen3_5 import qwen3_5_configs
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter


class TestQwen35MoEStateDictAdapter(unittest.TestCase):
    def setUp(self) -> None:
        config = qwen3_5_configs["debugmodel_moe"](
            attn_backend="flex", moe_comm_backend="standard"
        )
        self.adapter = Qwen35StateDictAdapter(config, hf_assets_path=None)

    def test_grouped_expert_layout_and_roundtrip(self) -> None:
        num_experts, hidden_dim, expert_dim = 3, 7, 5
        gate_up = torch.arange(
            num_experts * 2 * expert_dim * hidden_dim, dtype=torch.float32
        ).reshape(num_experts, 2 * expert_dim, hidden_dim)
        down = torch.arange(
            num_experts * hidden_dim * expert_dim, dtype=torch.float32
        ).reshape(num_experts, hidden_dim, expert_dim)
        hf_state = {
            "lm_head.weight": torch.empty(1, 1),
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }

        tt_state = self.adapter.from_hf(hf_state)
        self.assertEqual(
            tt_state["layers.0.moe.experts.w1_EFD"].shape,
            (num_experts, expert_dim, hidden_dim),
        )
        self.assertEqual(
            tt_state["layers.0.moe.experts.w2_EDF"].shape,
            (num_experts, hidden_dim, expert_dim),
        )
        self.assertTrue(
            torch.equal(
                tt_state["layers.0.moe.experts.w1_EFD"], gate_up[:, :expert_dim]
            )
        )
        self.assertTrue(
            torch.equal(
                tt_state["layers.0.moe.experts.w3_EFD"], gate_up[:, expert_dim:]
            )
        )
        self.assertTrue(torch.equal(tt_state["layers.0.moe.experts.w2_EDF"], down))

        restored_hf = self.adapter.to_hf(tt_state)
        self.assertTrue(
            torch.equal(
                restored_hf["model.language_model.layers.0.mlp.experts.gate_up_proj"],
                gate_up,
            )
        )
        self.assertTrue(
            torch.equal(
                restored_hf["model.language_model.layers.0.mlp.experts.down_proj"],
                down,
            )
        )


if __name__ == "__main__":
    unittest.main()

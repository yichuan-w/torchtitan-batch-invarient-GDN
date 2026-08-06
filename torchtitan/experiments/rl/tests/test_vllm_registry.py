# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

from torchtitan.experiments.rl.models.vllm_registry import model_spec_to_hf_config_dict


def _spec():
    rope = SimpleNamespace(max_seq_len=128, theta=10000.0)
    attn = SimpleNamespace(n_heads=4, n_kv_heads=2, head_dim=8, rope=rope)
    ffn = SimpleNamespace(w1=SimpleNamespace(out_features=32))
    layer = SimpleNamespace(attention=attn, feed_forward=ffn, moe=None, delta_net=None)
    model = SimpleNamespace(
        layers=[layer],
        first_attention=attn,
        vocab_size=256,
        dim=32,
        norm=SimpleNamespace(eps=1e-6),
        enable_weight_tying=False,
    )
    return SimpleNamespace(name="fake", model=model)


def test_model_spec_to_hf_config_omits_fake_special_token_ids_without_assets():
    hf = model_spec_to_hf_config_dict(_spec())

    assert "bos_token_id" not in hf
    assert "eos_token_id" not in hf
    assert "pad_token_id" not in hf


def test_model_spec_to_hf_config_reads_special_token_ids_from_hf_assets(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 151645, "pad_token_id": 151643})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token_id": 151643, "eos_token_id": 123})
    )

    hf = model_spec_to_hf_config_dict(_spec(), hf_assets_path=str(tmp_path))

    assert hf["bos_token_id"] == 151643
    assert hf["eos_token_id"] == 151645
    assert hf["pad_token_id"] == 151643

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from unittest.mock import Mock

from torchtitan.experiments.rl.examples.tmax import hf_upload


def test_copy_non_weight_assets(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "output"
    assets.mkdir()
    output.mkdir()
    (assets / "config.json").write_text("{}")
    (assets / "tokenizer.json").write_text("{}")
    (assets / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    (assets / "model.safetensors.index.json").write_text("{}")

    hf_upload._copy_non_weight_assets(assets, output)

    assert (output / "config.json").is_file()
    assert (output / "tokenizer.json").is_file()
    assert not (output / "model-00001-of-00002.safetensors").exists()
    assert not (output / "model.safetensors.index.json").exists()


def test_async_upload_does_not_put_token_on_command_line(
    monkeypatch, tmp_path: Path
) -> None:
    popen = Mock(return_value=Mock())
    monkeypatch.setattr(hf_upload.subprocess, "Popen", popen)
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    hf_upload.upload_hf_checkpoint_async(
        "/checkpoint/step-20",
        "/assets",
        "owner/model",
        log_path=tmp_path / "upload.log",
        private=False,
    )

    command = popen.call_args.args[0]
    assert "secret-token" not in command
    assert "--public" in command
    assert popen.call_args.kwargs["env"]["HF_TOKEN"] == "secret-token"
    assert popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"] == "0"

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert one TMax checkpoint to Hugging Face format and upload it."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _is_weight_asset(path: Path) -> bool:
    name = path.name
    return (
        name.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt"))
        or name == "model.safetensors.index.json"
    )


def _copy_non_weight_assets(hf_assets_dir: Path, output_dir: Path) -> None:
    for source in hf_assets_dir.rglob("*"):
        if not source.is_file() or _is_weight_asset(source):
            continue
        relative_path = source.relative_to(hf_assets_dir)
        if any(part in {".cache", ".git"} for part in relative_path.parts):
            continue
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _convert_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    hf_assets_dir: Path,
    export_dtype: str,
) -> None:
    from scripts.checkpoint_conversion.convert_to_hf import convert_to_hf

    convert_to_hf(
        input_dir=checkpoint_dir,
        output_dir=output_dir,
        model_name="qwen3_5",
        model_flavor="9B",
        hf_assets_path=hf_assets_dir,
        export_dtype=export_dtype,
    )


def upload_hf_checkpoint(
    checkpoint_dir: str | Path,
    hf_assets_dir: str | Path,
    repo_id: str,
    *,
    checkpoint_name: str | None = None,
    export_dtype: str = "bfloat16",
    staging_dir: str | Path | None = None,
    private: bool = True,
) -> str:
    """Convert and upload one DCP checkpoint under ``checkpoint_name/``."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi

    checkpoint_dir = Path(checkpoint_dir)
    hf_assets_dir = Path(hf_assets_dir)
    checkpoint_name = checkpoint_name or checkpoint_dir.name

    if not (checkpoint_dir / ".metadata").is_file():
        raise ValueError(f"Incomplete DCP checkpoint: {checkpoint_dir}")
    if not hf_assets_dir.is_dir():
        raise ValueError(f"HF assets directory does not exist: {hf_assets_dir}")
    if export_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported export dtype: {export_dtype}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    staging_root = Path(staging_dir) if staging_dir is not None else None
    if staging_root is not None:
        staging_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f"hf-{checkpoint_name}-", dir=staging_root
    ) as temporary_dir:
        output_dir = Path(temporary_dir) / checkpoint_name
        output_dir.mkdir()
        _convert_checkpoint(checkpoint_dir, output_dir, hf_assets_dir, export_dtype)
        _copy_non_weight_assets(hf_assets_dir, output_dir)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=output_dir,
            path_in_repo=checkpoint_name,
            commit_message=f"Upload {checkpoint_name}",
        )

    return f"https://huggingface.co/{repo_id}/tree/main/{checkpoint_name}"


def upload_hf_checkpoint_async(
    checkpoint_dir: str | Path,
    hf_assets_dir: str | Path,
    repo_id: str,
    *,
    checkpoint_name: str | None = None,
    export_dtype: str = "bfloat16",
    staging_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    private: bool = True,
) -> subprocess.Popen[bytes]:
    """Start an upload subprocess and return immediately."""
    command = [
        sys.executable,
        "-m",
        "torchtitan.experiments.rl.examples.tmax.hf_upload",
        str(checkpoint_dir),
        str(hf_assets_dir),
        repo_id,
        "--export-dtype",
        export_dtype,
    ]
    if checkpoint_name is not None:
        command.extend(("--checkpoint-name", checkpoint_name))
    if staging_dir is not None:
        command.extend(("--staging-dir", str(staging_dir)))
    if not private:
        command.append("--public")

    output = open(log_path, "ab") if log_path is not None else subprocess.DEVNULL
    try:
        return subprocess.Popen(
            command,
            env={**os.environ, "HF_HUB_OFFLINE": "0"},
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        if log_path is not None:
            output.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("hf_assets_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--checkpoint-name")
    parser.add_argument(
        "--export-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    url = upload_hf_checkpoint(
        args.checkpoint_dir,
        args.hf_assets_dir,
        args.repo_id,
        checkpoint_name=args.checkpoint_name,
        export_dtype=args.export_dtype,
        staging_dir=args.staging_dir,
        private=not args.public,
    )
    print(url, flush=True)


if __name__ == "__main__":
    main()

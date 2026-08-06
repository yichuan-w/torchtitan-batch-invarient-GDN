# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.experiments.rl.actors.trainer import (
    _cast_state_dict_parameters_for_transfer,
)


class _ModelWithPersistentState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.weight = weight
        self.tied_weight = weight
        self.register_buffer(
            "expert_bias", torch.ones(2, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "num_updates", torch.ones((), dtype=torch.int64), persistent=True
        )


def test_transfer_cast_preserves_persistent_buffer_dtypes() -> None:
    model = checkpoint_wrapper(_ModelWithPersistentState())
    state_dict = model.state_dict()

    assert any(
        "_checkpoint_wrapped_module" in name
        for name, _ in model.named_buffers(remove_duplicate=False)
    )

    transferred = _cast_state_dict_parameters_for_transfer(
        state_dict, model, torch.bfloat16
    )

    assert transferred["weight"].dtype == torch.bfloat16
    assert transferred["tied_weight"].dtype == torch.bfloat16
    assert transferred["expert_bias"].dtype == torch.float32
    assert transferred["num_updates"].dtype == torch.int64
    assert state_dict["weight"].dtype == torch.float32


def test_transfer_cast_rebuilds_zero_local_dtensor(tmp_path) -> None:
    if dist.is_initialized():
        return

    rendezvous = tmp_path / "rdzv"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        mesh = DeviceMesh("cpu", [0])
        local_tensor = torch.empty_strided((0, 2), (2, 1), dtype=torch.float32)
        tensor = DTensor.from_local(
            local_tensor,
            mesh,
            [Shard(0)],
            run_check=False,
            shape=torch.Size((1, 2)),
            stride=(2, 1),
        )

        transferred = _cast_state_dict_parameters_for_transfer(
            {"weight": tensor}, torch.nn.Linear(2, 1), torch.bfloat16
        )["weight"]

        assert isinstance(transferred, DTensor)
        assert transferred.shape == tensor.shape
        assert transferred.placements == tensor.placements
        assert transferred.to_local().shape == local_tensor.shape
        assert transferred.to_local().dtype == torch.bfloat16
        assert tensor.to_local().dtype == torch.float32
    finally:
        dist.destroy_process_group()

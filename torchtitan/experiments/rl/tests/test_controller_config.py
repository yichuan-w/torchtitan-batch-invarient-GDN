# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.controller import Controller, ValidationConfig


def _config(**overrides) -> SimpleNamespace:
    """The scalar fields Controller.Config.__post_init__ reaches before its first
    raise, so a test can exercise one check without building the whole config."""
    return SimpleNamespace(
        **{
            "num_generators": 1,
            "num_eval_generators": 0,
            "num_eval_rollout_workers": 0,
            "eval_rollout_concurrency": 0,
            "torchstore_reset_interval": 0,
            **overrides,
        }
    )


def test_torchstore_transport_recycle_is_rejected():
    with pytest.raises(ValueError, match="torchstore_reset_interval must be 0"):
        Controller.Config.__post_init__(_config(torchstore_reset_interval=32))


def test_torchstore_volume_placement_is_validated():
    with pytest.raises(ValueError, match="torchstore_volume_placement"):
        Controller.Config.__post_init__(_config(torchstore_volume_placement="worker"))


def test_eval_rollout_workers_require_an_eval_generator():
    with pytest.raises(ValueError, match="num_eval_rollout_workers requires"):
        Controller.Config.__post_init__(
            _config(num_eval_generators=0, num_eval_rollout_workers=4)
        )


def test_negative_eval_generators_is_rejected():
    with pytest.raises(ValueError, match="num_eval_generators must be non-negative"):
        Controller.Config.__post_init__(_config(num_eval_generators=-1))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"group_size": 0}, "validation.group_size must be positive"),
        ({"num_samples": -1}, "validation.num_samples must be non-negative"),
    ],
)
def test_validation_config_is_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ValidationConfig(**kwargs)

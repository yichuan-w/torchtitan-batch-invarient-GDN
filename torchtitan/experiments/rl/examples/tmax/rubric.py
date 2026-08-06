# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward for the tmax terminal-agent example.

Grading a tmax task requires running its verifier inside the sandbox, which the
rollouter already does while collecting the rollout. So the reward fn does not
re-grade: ``TMaxRollouter`` stamps the grade onto the rollout's last turn
``env_rewards`` (key ``tmax_reward``) and this fn reads it back -- keeping the
reward on the standard rubric/advantage path (and in the reward-breakdown metric)
without duplicating the expensive sandbox eval. Mirrors ``RewardR2E``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.rl.rollout.types import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn

# env_rewards key the rollouter stamps the tmax grade under.
TMAX_REWARD_KEY = "tmax_reward"


class RewardTMax(RewardFn):
    """Reads the tmax verifier reward (0.0/1.0) the rollouter attached to the rollout."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        for turn in reversed(rollout.turns):
            if TMAX_REWARD_KEY in turn.env_rewards:
                return float(turn.env_rewards[TMAX_REWARD_KEY])
        return 0.0

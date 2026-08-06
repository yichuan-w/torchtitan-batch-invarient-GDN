# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Placeholder MessageEnv for the tmax terminal-agent example.

Like the swe_r2e coding-agent harness, the whole agent loop runs inside the
sandbox / host-side ReAct loop -- the framework's env <-> generator loop is
bypassed (``TMaxRollouter`` overrides ``run_group_rollouts``). This class exists
only to satisfy ``Rollouter.Config.message_env`` (a required field) and to carry
the per-rollout task spec; its ``init`` / ``step`` are never called.
"""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.tmax.data import TMaxSample

_NOT_DRIVEN_MSG = (
    "TMaxEnv is not driven via the env loop; TMaxRollouter runs the agent in "
    "the sandbox. See examples/tmax/rollouter.py."
)


class TMaxEnv(MessageEnv):
    """Carries one tmax task spec. The agent loop runs in-sandbox (see module doc)."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: TMaxSample) -> None:
        self.sample = env_input

    async def init(self) -> MessageEnvInitOutput:
        raise NotImplementedError(_NOT_DRIVEN_MSG)

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        raise NotImplementedError(_NOT_DRIVEN_MSG)

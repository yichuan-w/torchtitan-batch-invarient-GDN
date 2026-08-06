# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converts one generated rollout group into trainable training_samples.

RolloutGroup -> data-validity filters -> rollout_to_training_samples -> TrainingSampleGroup
"""

import logging
import statistics
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import Rollout, RolloutGroup
from torchtitan.experiments.rl.types import (
    RolloutTurnID,
    TrainingSample,
    TrainingSampleGroup,
)

logger = logging.getLogger(__name__)


class TrainingSampleBuilder(Configurable):
    """Build trainable samples and metrics the trainer consumes from rollout groups.

    Example:
        builder = config.async_loop.training_sample_builder.build()
        output: TrainingSampleGroup = builder.build_from_group(rollout_group=rollout_group)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """TrainingSample builder filtering knobs."""

        drop_zero_std_reward_groups: bool = True
        """Drop zero-reward-variance groups;"""

        drop_groups_with_untrainable_rollouts: bool = True
        """Drop a group when any sibling has no completion tokens.

        When false, untrainable siblings still contribute their scored reward to
        group advantage estimation, but produce no training tokens. This matches
        systems that retain failed environment rollouts as zero-reward siblings.
        """

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_from_group(self, *, rollout_group: RolloutGroup) -> TrainingSampleGroup:
        """Apply group-level filters and convert surviving rollouts to training_samples.

        Example:
            # rollouts=[] from a failed generation
            build_from_group(rollout_group=failed_group)
            # -> TrainingSampleGroup(group_id=..., training_samples=[], metrics=[failure_metric])
        """
        metrics: list[m.Metric] = list(rollout_group.metrics)

        # TODO(async-rl): generalize these group-level filters into an ordered, user-pluggable filter
        # list (e.g. drop groups with all-0 / all-1 advantage), run in sequence on each RolloutGroup.
        # Failed generation: empty group carrying its failure metric.
        if not rollout_group.rollouts:
            return TrainingSampleGroup(
                group_id=rollout_group.group_id, training_samples=[], metrics=metrics
            )

        # A rollout with no completion tokens cannot produce a training sample.
        # Most recipes drop its whole centered-advantage group. Some environments
        # intentionally score infrastructure failures as zero, in which case the
        # empty sibling must stay in reward centering while contributing no tokens.
        num_untrainable_rollouts = sum(
            not any(turn.completion_token_ids for turn in rollout.turns)
            for rollout in rollout_group.rollouts
        )
        if num_untrainable_rollouts:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_untrainable_rollouts",
                    m.Sum(float(num_untrainable_rollouts)),
                )
            )
        if (
            num_untrainable_rollouts
            and self.config.drop_groups_with_untrainable_rollouts
        ):
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_groups_dropped_untrainable", m.Sum(1.0)
                )
            )
            return TrainingSampleGroup(
                group_id=rollout_group.group_id, training_samples=[], metrics=metrics
            )

        # Zero-std reward: no learning signal across siblings.
        # TODO(robustness): if EVERY group is zero-std (e.g. a degenerate model emitting a constant
        # reward), the batcher never reaches num_groups_per_train_step, the trainer never steps, and
        # no step metrics are logged (they flush at train step) -> a silent hang. Emit a warning /
        # heartbeat (e.g. when N consecutive groups drop with no batch packed) so this is visible.
        rewards = [rollout.reward for rollout in rollout_group.rollouts]
        is_zero_std = len(rewards) > 1 and statistics.pstdev(rewards) == 0.0
        metrics.append(
            m.Metric(
                "rollout_reward/group_zero_std_frac",
                m.Mean(1.0 if is_zero_std else 0.0),
            )
        )
        # Split the dropped (zero-std) groups by cause -- all-solved (too easy) vs
        # all-failed (too hard), the analog of open-instruct's filtered_prompts_solved
        # / filtered_prompts_zero. Both are dropped by drop_zero_std, so this shows
        # WHY the trainable batch shrinks: over training all-failed should fall (the
        # policy learns) while all-solved may rise. The complement of their sum (among
        # zero-std groups) is the trainable mixed band; together they are the scalar
        # summary of the per-group solve-rate ("solve_rate_hist") distribution.
        group_mean_reward = statistics.mean(rewards)
        metrics.append(
            m.Metric(
                "rollout_reward/group_all_solved_frac",
                m.Mean(1.0 if is_zero_std and group_mean_reward > 0.0 else 0.0),
            )
        )
        metrics.append(
            m.Metric(
                "rollout_reward/group_all_failed_frac",
                m.Mean(1.0 if is_zero_std and group_mean_reward == 0.0 else 0.0),
            )
        )
        # Avg train reward = open-instruct's val/avg_group_performance_pre_filter,
        # computed on the attempted TRAINING batch, not held-out validation. It is
        # the mean reward over all consumed groups, including the zero-std groups
        # dropped below: all-solved contributes 1, all-failed contributes 0, and a
        # mixed group contributes its solve fraction. Emitting before the drop makes
        # the per-step group mean equal to open-instruct's helper for equal group
        # sizes. Unlike kept-only reward, this can rise as more prompts become solved.
        if rewards:
            metrics.append(
                m.Metric(
                    "rollout_reward/avg_train_reward",
                    m.Mean(statistics.mean(rewards)),
                )
            )
        if self.config.drop_zero_std_reward_groups and is_zero_std:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_groups_dropped_zero_std", m.Sum(1.0)
                )
            )
            return TrainingSampleGroup(
                group_id=rollout_group.group_id, training_samples=[], metrics=metrics
            )

        # One rollout may branch into multiple trainable training_samples.
        training_samples: list[TrainingSample] = []
        branches_per_rollout: list[float] = []
        for rollout in rollout_group.rollouts:
            rollout_training_samples = self.rollout_to_training_samples(rollout)
            training_samples.extend(rollout_training_samples)
            branches_per_rollout.append(float(len(rollout_training_samples)))

        min_policy_versions = [
            training_sample.min_policy_version for training_sample in training_samples
        ]
        max_policy_versions = [
            training_sample.max_policy_version for training_sample in training_samples
        ]
        advantages = [rollout.advantage for rollout in rollout_group.rollouts]
        metrics += [
            m.Metric(
                "training_sample_builder/num_training_samples",
                m.Sum(float(len(training_samples))),
            ),
            m.Metric("advantage", m.SummaryStats.from_list(advantages)),
            m.Metric(
                "rollout/branches_per_rollout", m.Mean.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/branches_per_rollout", m.Max.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/min_policy_version", m.Min.from_list(min_policy_versions)
            ),
            m.Metric(
                "rollout/max_policy_version", m.Max.from_list(max_policy_versions)
            ),
        ]
        return TrainingSampleGroup(
            group_id=rollout_group.group_id,
            training_samples=training_samples,
            metrics=metrics,
        )

    def rollout_to_training_samples(self, rollout: Rollout) -> list[TrainingSample]:
        """Pack a scored `Rollout` into training training_samples — usually one, or several where its turns branch.

        Override this to change how a rollout becomes trainable training_samples (the main extension point of
        this class). Turns that share a growing prefix (each prompt continues the previous prompt +
        completion) are packed into ONE training_sample: prompts and env replies are masked out, completions
        are trained. A new training_sample (branch) opens wherever that prefix breaks (history was edited).

        This branching has three common causes:
        1) The rollout loop purposely edits the history: For example, when compacting a long conversation;
        2) Thinking is removed from history: This is a flag in the `renderer` used in the `Rollouter`. Choose
            `preserve_all_thinking=True` to avoid stripping thinking (default=True);
        3) Some error or change from retokenization: Those can be avoided by doing token-in-token-out (TITO),
            i.e. you give tokens to the generators, and you append (bridge) the tokens received, leaving no room for changes.
            TITO is our default;

        Each packed training_sample records the min/max policy version across its turns: `min_policy_version`
        = the oldest (opening) turn's version (the off-policy filter reads it), `max_policy_version` = the
        newest version any of its turns reached.

        Example (5 turns; the env compacts history before turn 3, so the prefix breaks -> 2 training_samples).
        P = prompt; C = completion; E = env reply
            turn0 (v5): prompt=[P1]              completion=[C1]
            turn1 (v6): prompt=[P1,C1,E1]        completion=[C2]
            turn2 (v6): prompt=[P1,C1,E1,C2,E2]  completion=[C3]
            turn3 (v6): prompt=[P2]              completion=[C4]   # history compacted -> prefix breaks
            turn4 (v7): prompt=[P2,C4,E4]        completion=[C5]
            # -> training_sample 0: token_ids=[P1,C1,E1,C2,E2,C3], loss_mask=[0,1,0,1,0,1]
            #              rollout_id.turn_id=0, min_policy_version=5, max_policy_version=6
            #    training_sample 1: token_ids=[P2,C4,E4,C5],        loss_mask=[0,1,0,1]
            #              rollout_id.turn_id=3, min_policy_version=6, max_policy_version=7
        """
        rollout_advantage = rollout.advantage
        if rollout_advantage is None:
            raise ValueError(
                f"rollout {rollout.group_id}/rollout={rollout.rollout_id} has no advantage; the Rollouter "
                "must fill it (via its advantage estimator) before training_samples are built."
            )
        training_samples: list[TrainingSample] = []

        # Used to check if [P1, C1] is prefix of [P1,C1,E1] in the docstring example
        prev_prompt_and_completion: list[int] = []

        # Skip if no completion (nothing to train on). This happens when the prompt is too
        # long in the first turn, before any generation. We keep these rollouts for debugging.
        # TODO(async-rl): confirm this signal actually reaches the rollout_recorder.
        for turn_idx, rollout_turn in enumerate(rollout.turns):
            if not rollout_turn.completion_token_ids:
                if len(rollout.turns) == 1:

                    continue

                # empty rollouts should only happen when initial prompt is too long
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: "
                    f"non-final turn {turn_idx} has no completion"
                )

            prompt = rollout_turn.prompt_token_ids
            # True when this prompt continues the previous one (prefix-preserving);
            # False when the env edited history -> open a new training_sample (branch).
            extends_prev = (
                prompt[: len(prev_prompt_and_completion)] == prev_prompt_and_completion
            )
            if not training_samples or not extends_prev:
                # Start a new training_sample; its RolloutTurnID marks the turn the segment begins at.
                training_samples.append(
                    TrainingSample(
                        min_policy_version=rollout_turn.min_policy_version,
                        max_policy_version=rollout_turn.max_policy_version,
                        rollout_id=RolloutTurnID(
                            group_id=rollout.group_id,
                            rollout_id=rollout.rollout_id,
                            turn_id=turn_idx,
                        ),
                        token_ids=[],
                        loss_mask=[],
                        logprobs=[],
                        advantage=[],
                    )
                )
                # New branch (first turn or a branch): no shared prefix.
                prefix_len = 0
            else:
                prefix_len = len(prev_prompt_and_completion)

            # Append this turn's new info to `training_sample`: prefix delta (untrained) + completion (trained).
            training_sample = training_samples[-1]
            prompt_delta = prompt[prefix_len:]
            num_delta = len(prompt_delta)
            num_completion = len(rollout_turn.completion_token_ids)
            training_sample.token_ids += prompt_delta
            training_sample.loss_mask += [False] * num_delta
            training_sample.logprobs += [0.0] * num_delta
            training_sample.advantage += [0.0] * num_delta
            # Widen the segment's version span to include this turn (the min stays the opening turn's).
            training_sample.max_policy_version = max(
                training_sample.max_policy_version, rollout_turn.max_policy_version
            )
            training_sample.token_ids += rollout_turn.completion_token_ids
            training_sample.loss_mask += [True] * num_completion
            training_sample.logprobs += rollout_turn.completion_logprobs
            training_sample.advantage += [rollout_advantage] * num_completion

            prev_prompt_and_completion = prompt + rollout_turn.completion_token_ids

        # TITO integrity. NOTE: counting loss_mask=True vs sum(completion lengths) is
        # TAUTOLOGICAL -- the loop above sets [True]*len(completion) and
        # [False]*len(prompt_delta) by construction, so that equality always holds and
        # can never catch a dedup error (the old assert here was a no-op). The real,
        # non-tautological corruption is a per-turn completion token/logprob length
        # mismatch from capture: the loss pairs trainer-vs-generator logprobs 1:1, so a
        # mismatch shifts the ratio. Warn (don't crash a live rollout) if it happens.
        for turn_idx, t in enumerate(rollout.turns):
            n_ids, n_lp = len(t.completion_token_ids), len(t.completion_logprobs)
            if t.completion_token_ids and n_ids != n_lp:
                logger.warning(
                    f"[training_sample_builder] rollout {rollout.group_id}/"
                    f"rollout={rollout.rollout_id} turn {turn_idx}: completion "
                    f"token/logprob length mismatch ({n_ids} ids vs {n_lp} logprobs) "
                    "-- trained logprobs would be shifted; check generator capture"
                )

        # Visibility: how a trajectory split into training samples. Expect 1 for an
        # append-only TITO loop (tmax vanillux); >1 means a mid-trajectory branch
        # (history rewrite or a re-render), so surface that at INFO -- it is the
        # training-side echo of the adapter's mid-trajectory re-render warning.
        num_samples = len(training_samples)
        trained_tokens = sum(sum(s.loss_mask) for s in training_samples)
        msg = (
            f"[training_sample_builder] rollout {rollout.group_id}/"
            f"rollout={rollout.rollout_id} -> {num_samples} sample(s), "
            f"{trained_tokens} trained tokens, {len(rollout.turns)} turn(s)"
        )
        if num_samples > 1:
            logger.info(f"{msg} (trajectory SPLIT into {num_samples} samples)")
        else:
            logger.debug(msg)

        return training_samples

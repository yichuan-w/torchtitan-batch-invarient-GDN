# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DPPO loss: unclipped importance-ratio surrogate with a divergence trust-region mask.

Faithful to open-instruct's ``loss_fn=dppo`` (the tmax recipe; DPPO paper
https://arxiv.org/abs/2602.04879). The surrogate is the UNCLIPPED ``-A * ratio``
(no PPO ratio clip); a per-token trust-region MASK zeros the loss for tokens that
would push the policy FURTHER from the rollout (behavior) policy AND whose
behavior<->policy divergence has already exceeded a threshold ``delta``. The mask
REPLACES the PPO clip as the trust region (that is the DPPO contribution). Tokens
that move the ratio back toward 1 are never masked, preserving PPO's asymmetry.

The divergence is the binary (Bernoulli over ``{sampled token, all others}``)
approximation from Eqs. 13/14 of the DPPO paper -- computed from only the
per-token logprobs, so it needs no extra forward pass. For TITO rollouts the
generator (vLLM) logprobs ARE the behavior/old policy, matching the recipe's
``--use_vllm_logprobs true``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, compute_logprobs
from torchtitan.config import CompileConfig

logger = logging.getLogger(__name__)

# Clamp |log(pi_theta/pi_old)| before exp() so a large generator/trainer
# logprob mismatch cannot overflow exp() to inf/NaN.
_MAX_LOG_RATIO = 10.0
# Clamp logprobs before exp() when forming the Bernoulli probabilities for the
# divergence (mirrors open-instruct's compute_binary_divergence).
_MIN_LOGPROB_FOR_PROB = -30.0

# SWE_DEBUG_MAX_LOGDIFF=1 debug: only log/dump chunks whose worst |diff| exceeds
# this, and show a +/-window of tokens around the argmax. Fixed constants (one env
# switch, no extra knobs).
_DEBUG_MAX_LOGDIFF_THRESH = 1.0
_DEBUG_MAX_LOGDIFF_WINDOW = 8


class DPPOLoss(BaseLoss):
    """Unclipped importance-ratio surrogate gated by a DPPO divergence mask.

    Faithful to open-instruct's ``loss_fn=dppo`` (tmax recipe): the per-token loss
    is the UNCLIPPED ``-advantage * ratio`` -- there is NO PPO ratio clip. The sole
    trust region is a 0/1 divergence mask that zeros the loss (value and gradient)
    of tokens outside the ball (divergence > delta) that are being pushed further
    off-policy; the mask replaces the clip (DPPO paper, Eq. 12). A token whose
    generator logprob is non-finite is dropped. The scalar loss sums per-token
    losses over loss positions divided by ``global_valid_tokens``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        divergence_threshold: float = 0.1
        """DPPO trust-region radius ``delta``: a token is eligible for masking only
        once its binary behavior<->policy divergence exceeds this."""

        divergence_type: str = "tv"
        """``"tv"`` (total variation, the recipe default) or ``"kl"`` binary divergence."""

        ratio_cap: float = 0.0
        """Truncated importance-sampling cap on the surrogate ratio. 0.0 = disabled
        (unclipped, the recipe default). When > 0 the ratio in ``-A * ratio`` is
        clamped to ``[0, ratio_cap]`` (e.g. 2.0) so a few tokens with a large
        generator<->trainer logprob mismatch (e.g. a residual GDN train/infer
        divergence tail) cannot spike the gradient. The DPPO TV mask bounds
        probability-mass movement but NOT low-probability high-ratio tokens, so this
        cap is the tool that lets DPPO tolerate a larger gen/train mismatch."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        del compile_config
        self.divergence_threshold = config.divergence_threshold
        self.divergence_type = config.divergence_type
        self.ratio_cap = config.ratio_cap

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        *,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DPPO (unclipped ratio + divergence-mask) surrogate loss.

        Args mirror :class:`DAPOLoss`. ``generator_logprobs`` are the rollout
        (behavior/old) logprobs; ``advantages`` are per-token (0 on prompt/pad).
        """
        trainer_logprobs = compute_logprobs(logits, labels)
        # Drop tokens with a non-finite generator logprob (no valid old-policy
        # reference; e.g. vLLM under cudagraph), same as DAPO.
        response_mask = loss_mask
        raw_log_ratio = trainer_logprobs - generator_logprobs
        loss_mask = loss_mask & torch.isfinite(raw_log_ratio)
        log_ratio = torch.clamp(
            torch.nan_to_num(raw_log_ratio), -_MAX_LOG_RATIO, _MAX_LOG_RATIO
        )
        ratio = torch.exp(log_ratio)

        # Optional truncated-IS cap: clamp the ratio so outlier tokens (large
        # gen<->trainer logprob mismatch) cannot spike the gradient. 0.0 = disabled
        # (recipe default, unclipped). The clamp saturates gradient above the cap.
        if self.ratio_cap > 0.0:
            uncapped_ratio = ratio
            ratio = ratio.clamp(max=self.ratio_cap)

        # Unclipped importance-weighted surrogate: -A * ratio. Faithful to
        # open-instruct DPPO (pg_losses = -adv * ratio, no PPO clip); the DPPO mask
        # below is the only trust region.
        token_loss = -(advantages * ratio)

        # DPPO trust-region mask (detached; it gates gradient, not part of it).
        # bad = pushing further off-policy (ratio>1 with A>0, or ratio<1 with A<0)
        # while already outside the divergence ball. Never masks tokens moving the
        # ratio back toward 1, so corrective updates always flow.
        with torch.no_grad():
            mu = torch.exp(
                torch.clamp(generator_logprobs, min=_MIN_LOGPROB_FOR_PROB, max=0.0)
            )
            pi = torch.exp(
                torch.clamp(trainer_logprobs, min=_MIN_LOGPROB_FOR_PROB, max=0.0)
            )
            if self.divergence_type == "kl":
                eps = 1e-9
                mu_c = mu.clamp(eps, 1.0 - eps)
                pi_c = pi.clamp(eps, 1.0 - eps)
                divergence = mu_c * (mu_c.log() - pi_c.log()) + (1.0 - mu_c) * (
                    (1.0 - mu_c).log() - (1.0 - pi_c).log()
                )
            else:  # total variation (recipe default)
                divergence = (mu - pi).abs()
            outside_region = divergence > self.divergence_threshold
            bad_high = (advantages > 0) & (ratio > 1.0) & outside_region
            bad_low = (advantages < 0) & (ratio < 1.0) & outside_region
            dppo_mask = (~(bad_high | bad_low)).to(token_loss.dtype)

        token_loss = token_loss * dppo_mask

        masked_loss = token_loss * loss_mask
        loss_denominator = (
            max(global_valid_tokens, 1) if global_valid_tokens is not None else 1
        )
        loss = masked_loss.sum() / loss_denominator

        with torch.no_grad():
            diff = trainer_logprobs - generator_logprobs
            diff_for_metrics = torch.where(loss_mask, diff, torch.zeros_like(diff))
            # Debug (SWE_DEBUG_MAX_LOGDIFF=1): locate the worst |diff| token, log its
            # token id + both logprobs + a local window, and dump the full per-token
            # arrays (labels/diff/loss_mask/logprobs) so the max can be inspected as a
            # heat map and reproduced. Off by default; threshold-gated so only large
            # maxima log. Loss is not compiled, so the Python/.item() path is safe.
            if os.environ.get("SWE_DEBUG_MAX_LOGDIFF", "0") == "1":
                self._debug_max_logdiff(
                    diff,
                    diff_for_metrics,
                    loss_mask,
                    labels,
                    trainer_logprobs,
                    positions,
                )
            masked_ratio = ratio * loss_mask
            # KL(vllm_sampling || local_trainer) via Schulman k1/k3 estimators on the
            # sampled tokens (matches open-instruct debug/vllm_local_kl_*). Sampling
            # policy p = vLLM (generator), target q = local trainer; logratio =
            # log q - log p = trainer_logprobs - generator_logprobs = diff.
            #   k1 = E_p[log p - log q] = mean(-logratio)          (unbiased, signed)
            #   k3 = E_p[(q/p) - 1 - log(q/p)] = mean(exp(r) - r - 1)  (>= 0, low var)
            # Masked-out positions have diff_for_metrics == 0 -> k3 term is exp(0)-0-1
            # == 0, so they drop out of the sum. Normalized by loss_denominator like
            # the bit_wise/* metrics so aggregation matches abs_mean.
            k3_for_metrics = torch.exp(diff_for_metrics) - diff_for_metrics - 1.0
            # Per-token reverse-KL integrand p * (log p - log q) evaluated at the
            # sampled token (p = vLLM/generator, q = local trainer): explicitly
            # probability-weighted, unlike the sample-based k1/k3. At loss_mask=True
            # positions generator_logprobs is finite (non-finite ratios are masked
            # above), so exp() is finite; masked-out positions are zeroed.
            reverse_kl_tok = torch.exp(generator_logprobs) * (
                generator_logprobs - trainer_logprobs
            )
            reverse_kl_for_metrics = torch.where(
                loss_mask, reverse_kl_tok, torch.zeros_like(diff)
            )
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": masked_ratio.sum() / loss_denominator,
                # Fraction of trained tokens the DPPO trust region KEEPS (1.0 = no
                # masking; lower = more off-policy tokens dropped).
                "loss/dppo_mask_kept_frac": (dppo_mask * loss_mask).sum()
                / loss_denominator,
                "loss/dppo_divergence_mean": (divergence * loss_mask).sum()
                / loss_denominator,
                "loss/generator_logprob_nan_frac": (
                    (~torch.isfinite(generator_logprobs)).float() * response_mask
                ).sum()
                / loss_denominator,
                "bit_wise/logprob_diff/mean": diff_for_metrics.float().sum()
                / loss_denominator,
                "bit_wise/logprob_diff/abs_mean": diff_for_metrics.abs().float().sum()
                / loss_denominator,
                "bit_wise/ratio_tokens_different/mean": (
                    (diff_for_metrics.abs() > 1e-6).float() * loss_mask
                ).sum()
                / loss_denominator,
                "bit_wise/logprob_diff/max": diff_for_metrics.abs().max(),
                "debug/vllm_local_kl_k1_mean": -diff_for_metrics.float().sum()
                / loss_denominator,
                "debug/vllm_local_kl_k3_mean": k3_for_metrics.float().sum()
                / loss_denominator,
                "debug/vllm_local_reverse_kl_mean": reverse_kl_for_metrics.float().sum()
                / loss_denominator,
            }
            if self.ratio_cap > 0.0:
                metrics["loss/ratio_capped_frac"] = (
                    (uncapped_ratio > self.ratio_cap).float() * loss_mask
                ).sum() / loss_denominator

        return loss, metrics

    @staticmethod
    def _debug_max_logdiff(
        diff: torch.Tensor,
        diff_for_metrics: torch.Tensor,
        loss_mask: torch.Tensor,
        labels: torch.Tensor,
        trainer_logprobs: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Inspect the worst |trainer_logprob - generator_logprob| token.

        Enabled by the single switch SWE_DEBUG_MAX_LOGDIFF=1. Only acts when the
        chunk's max |diff| exceeds ``_DEBUG_MAX_LOGDIFF_THRESH`` so routine small
        diffs are silent. Logs the argmax token id, both logprobs, whether it is
        trained (loss_mask), and a +/-window of token ids / diffs / mask; and
        torch.saves the full per-token arrays (labels/diff/loss_mask/trainer_logprobs)
        under SWE_DUMP_DIR/max_logdiff for an offline heat map + repro. With chunked
        loss this runs per sequence-chunk, so ``t`` is the offset within the chunk.
        """
        absd = diff_for_metrics.abs()
        mx = float(absd.max())
        if mx < _DEBUG_MAX_LOGDIFF_THRESH:
            return
        flat = int(absd.argmax())
        seq_len = absd.shape[-1]
        b, t = flat // seq_len, flat % seq_len
        w = _DEBUG_MAX_LOGDIFF_WINDOW
        lo, hi = max(0, t - w), min(seq_len, t + w + 1)
        gen_lp = float(trainer_logprobs[b, t] - diff[b, t])  # gen = trainer - diff
        # positions reset to 0 at each packed-sample boundary; report the max
        # token's own position and its distance to the nearest sample start, so a
        # boundary state-bleed artifact (pos small / dist ~0) is unambiguous.
        pos_info = ""
        if positions is not None:
            row_pos = positions[b]
            resets = (row_pos == 0).nonzero(as_tuple=True)[0]
            dist = int((resets - t).abs().min()) if resets.numel() else -1
            pos_info = f" pos={int(row_pos[t])} dist_to_sample_start={dist}"
        logger.warning(
            "[max_logdiff] |diff|=%.4f at (b=%d,t=%d) token_id=%d "
            "trainer_lp=%.4f gen_lp=%.4f trained=%s%s | window[%d:%d] "
            "token_ids=%s diffs=%s mask=%s pos=%s",
            mx,
            b,
            t,
            int(labels[b, t]),
            float(trainer_logprobs[b, t]),
            gen_lp,
            bool(loss_mask[b, t]),
            pos_info,
            lo,
            hi,
            labels[b, lo:hi].tolist(),
            [round(float(x), 3) for x in diff[b, lo:hi]],
            [int(x) for x in loss_mask[b, lo:hi]],
            positions[b, lo:hi].tolist() if positions is not None else None,
        )
        try:
            rank = os.environ.get("RANK", "0")
            base = os.path.join(os.environ.get("SWE_DUMP_DIR", "/tmp"), "max_logdiff")
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, f"maxlogdiff_r{rank}_{flat}.pt")
            torch.save(
                {
                    "labels": labels.detach().cpu(),
                    "diff": diff.detach().cpu(),
                    "loss_mask": loss_mask.detach().cpu(),
                    "trainer_logprobs": trainer_logprobs.detach().cpu(),
                    "positions": (
                        positions.detach().cpu() if positions is not None else None
                    ),
                    "argmax_bt": (b, t),
                    "max_absdiff": mx,
                },
                path,
            )
            logger.warning("[max_logdiff] dumped arrays -> %s", path)
        except Exception as e:  # noqa: BLE001 -- debug path, never fail the step
            logger.warning("[max_logdiff] dump failed: %s", e)

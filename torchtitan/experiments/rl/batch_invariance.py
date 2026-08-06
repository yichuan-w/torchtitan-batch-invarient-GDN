# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch-invariance helpers for bitwise trainer/generator numerics parity.

Groups the pieces that make the vLLM generator match the trainer op-for-op under
batch-invariant mode: a model-config converter that pins FlexAttention kernel
options, plus two runtime patches (bmm and the v2 logprob kernel).
"""

import logging
from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import FlexAttention
from torchtitan.protocols.model import ModelConfigConverter

logger = logging.getLogger(__name__)


class BatchInvariantFlexConverter(ModelConfigConverter):
    """Pin flex attention kernel options for batch-invariant mode.

    Sets fixed BLOCK_M/BLOCK_N=16 and BACKEND=TRITON on all
    FlexAttention layers.

    BACKEND=TRITON is to avoid flex_decode kernel.
    """

    # the triton BLOCK_N tile size needs to be pinned for stable numerics and
    # needs to match vLLM's for identical results, today vLLM default is 16
    # TODO: run some experiments to determine impact of small vs large tile sizes
    _BLOCK_M = 16
    _BLOCK_N = 16

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config):
        for layer_cfg in model_config.layers:
            # Hybrid models (Qwen3.5) have linear-attention layers with no
            # ``attention`` config to pin -- skip them.
            attention = getattr(layer_cfg, "attention", None)
            if attention is None:
                continue
            inner = attention.inner_attention
            if isinstance(inner, FlexAttention.Config):
                inner.kernel_options["BACKEND"] = "TRITON"
                inner.kernel_options["BLOCK_M"] = self._BLOCK_M
                inner.kernel_options["BLOCK_N"] = self._BLOCK_N
        return model_config


_batch_invariant_bmm_lib: torch.library.Library | None = None


def patch_bmm_for_batch_invariance() -> None:
    """Override ``aten::bmm`` with vLLM's batch-invariant bmm kernel.

    torchtitan's batch-invariant mode (``batch_invariant_ops``, applied by
    ``set_batch_invariance``) overrides ``mm``/``addmm``/``_log_softmax``/
    ``mean.dim`` but not ``bmm``. The MoE router gate (3-D activation @ 2-D
    weight) lowers to ``aten::bmm`` in the generator but ``aten::mm`` in the
    trainer, so without this the generator's gate scores drift from the
    trainer's and flip top-k expert routing, breaking on-policy logprob parity.

    TODO: Investigate how to drop bmm batch invariant patch in generator.
    """
    global _batch_invariant_bmm_lib
    if _batch_invariant_bmm_lib is not None:
        return
    from vllm.model_executor.layers.batch_invariant import bmm_batch_invariant

    _batch_invariant_bmm_lib = torch.library.Library("aten", "IMPL")
    _batch_invariant_bmm_lib.impl("bmm", bmm_batch_invariant, "CUDA")
    # pyrefly: ignore[bad-assignment]
    torch.bmm = bmm_batch_invariant


def force_logprobs_fn_for_batch_invariance() -> None:
    """Make vLLM's v2 logprob path dispatch.

    The v2 GPU sampler computes per-token logprobs with a fused Triton kernel
    (``compute_token_logprobs`` -> ``_topk_log_softmax_kernel``) that inlines
    ``log(softmax(logits))`` and never calls PyTorch ops.

    Swapping the kernel to routes the generator and the trainer through
    the same set of ops, so logprobs match bit-for-bit.
    """
    import vllm.v1.worker.gpu.sample.logprob as vllm_logprob

    from torchtitan.components.loss import compute_logprobs

    def generator_compute_token_logprobs(
        logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-token logprobs for vLLM's v2 sampler (replaces its fused kernel).

        Args:
            logits: ``[N, V]`` next-token logits for N sampled positions
                (V = vocab_size).
            token_ids: ``[N, K]`` the K token ids to score per position (vLLM
                passes the sampled token's logprob plus any top-k logprobs it requested).

        Returns:
            ``[N, K]`` logprob of each of the K token ids at each position.
        """
        # vLLM gives token_ids [N, K]. SamplingParams(logprobs=0) makes real
        # requests K=1, but we can't assert that: vLLM's kernel warmup probes
        # this patched fn with K>1 (e.g. K=6), so we must handle any K. Map the
        # N positions to one sequence (B=1, S=N) and reuse the trainer's
        # compute_logprobs (one token per position) once per column.
        #
        # NOTE: each element of token_ids is scored independently, after the
        # whole generated sequence is marterialized. We iterate column-by-column
        # purely because torchtitan's compute_logprobs takes one token id per
        # position; it is not a cross-column/cross-position dependency.
        logits = logits.unsqueeze(0)  # [1, N, V]  (B=1, S=N)
        token_ids = token_ids.to(torch.int64)
        per_column = [
            compute_logprobs(logits, token_ids[:, k].unsqueeze(0))  # [1, N]
            for k in range(token_ids.shape[1])
        ]
        return torch.stack(per_column, dim=-1).squeeze(0)  # [N, K]

    vllm_logprob.compute_token_logprobs = generator_compute_token_logprobs
    logger.info(
        "Patched vLLM compute_token_logprobs with trainer's implementation "
        "so generator and trainer share one logprob code path"
    )


_small_m_matmul_patched = False


def patch_matmul_for_small_m_batch_invariance() -> None:
    """Tune batch_invariant_ops persistent-matmul output tiles for small M.

    batch_invariant_ops.matmul_persistent always launches its Triton kernel with
    BLOCK_SIZE_M=128. On DECODE the matmul is [num_seqs, K] @ [K, N] with num_seqs
    small (e.g. 8), so a 128-row M-tile wastes ~94% of the work -- and this kernel is
    ~70% of the batch-invariant GDN decode CUDA time. Shrinking BLOCK_SIZE_M to fit M
    (min(128, next_pow2(M)), floor 16) removes that waste. BLOCK_SIZE_K (which sets the
    per-output-element K-reduction order, hence the numerics) is UNCHANGED, so the
    result is bitwise-identical to the upstream kernel for every M -- verified with
    torch.equal across M in {8,16,64,128,2000} -- and M>=128 (e.g. trainer/prefill)
    keeps the original BLOCK_SIZE_M=128 unchanged. For bf16 small-M outputs no wider
    than 4096, BLOCK_SIZE_N=32 exposes enough output tiles to occupy the H100 while
    leaving the per-element K reduction unchanged. Wider outputs keep the upstream
    N tile because the smaller tile is slower there. Idempotent; safe to call after
    set_batch_invariance. No-op (logs) if the package internals are unavailable.
    """
    global _small_m_matmul_patched
    if _small_m_matmul_patched:
        return
    try:
        import batch_invariant_ops.batch_invariant_ops as _bi
        import triton
    except Exception as exc:  # package layout changed / not installed
        logger.warning(f"small-M BI matmul patch skipped (import failed): {exc}")
        return

    # Upstream defaults (batch_invariant_ops.matmul_persistent), keyed by dtype name.
    _cfgs = {
        "bfloat16": dict(
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            num_stages=3,
            num_warps=8,
        ),
        "float16": dict(
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            num_stages=3,
            num_warps=8,
        ),
        "float32": dict(
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
            num_stages=3,
            num_warps=8,
        ),
    }

    def _small_m_matmul_persistent(a, b, bias=None):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        num_sms = _bi.get_compute_units()
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        cfg = dict(_cfgs[str(a.dtype).split(".")[-1]])
        # Tune only output tiles; BLOCK_SIZE_K stays fixed -> same reduction.
        cfg["BLOCK_SIZE_M"] = min(
            cfg["BLOCK_SIZE_M"], max(16, triton.next_power_of_2(M))
        )
        if a.dtype == torch.bfloat16 and cfg["BLOCK_SIZE_M"] < 128 and N <= 4096:
            cfg["BLOCK_SIZE_N"] = 32
            cfg["num_warps"] = 4
        # Small-M (decode) is HBM-bound loading the weights; num_stages=4 pipelines
        # those loads ~1.3-1.5x faster than the upstream 3 (measured). num_stages is
        # scheduling only -> same K accumulation -> bitwise-identical (verified
        # torch.equal ns 3 vs 4 vs 5). Large M (>=128, prefill) keeps upstream ns=3.
        if cfg["BLOCK_SIZE_M"] < 128:
            cfg["num_stages"] = 4

        def grid(meta):
            return (
                min(
                    num_sms,
                    triton.cdiv(M, meta["BLOCK_SIZE_M"])
                    * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
                ),
            )

        _bi.matmul_kernel_persistent[grid](
            a,
            b,
            c,
            bias,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            NUM_SMS=num_sms,
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            C_LARGE=c.numel() > 2**31,
            HAS_BIAS=bias is not None,
            **cfg,
        )
        return c

    # mm_batch_invariant / addmm_batch_invariant call the module-global
    # matmul_persistent, so replacing it here reroutes both (late binding).
    _bi.matmul_persistent = _small_m_matmul_persistent
    _small_m_matmul_patched = True
    logger.info(
        "Patched batch_invariant_ops.matmul_persistent with small-M output tiles "
        "(BLOCK_SIZE_K unchanged -> bitwise-identical; faster small-M decode)"
    )

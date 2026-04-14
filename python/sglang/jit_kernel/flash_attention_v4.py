from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api

try:
    from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception as _e:  # pragma: no cover
    _flash_attn_varlen_func = None
    _flash_attn_import_error = _e
else:
    _flash_attn_import_error = None


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _apply_softcap(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    return scores


def _mla_attention_ref_paged(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if page_table is None:
        raise RuntimeError("FA4 MLA prototype requires page_table metadata.")
    if cache_seqlens is None:
        raise RuntimeError("FA4 MLA prototype requires cache_seqlens metadata.")
    if cu_seqlens_q is None:
        raise RuntimeError("FA4 MLA prototype requires cu_seqlens_q metadata.")
    if k_cache.shape[1] != 1 or v_cache.shape[1] != 1:
        raise RuntimeError("FA4 MLA prototype requires --page-size 1.")
    if k_cache.shape[2] != 1 or v_cache.shape[2] != 1:
        raise RuntimeError(
            "FA4 MLA prototype requires a single KV head (MQA-style MLA cache)."
        )
    if (
        q_nope.dtype != torch.bfloat16
        or q_rope.dtype != torch.bfloat16
        or k_cache.dtype != torch.bfloat16
        or v_cache.dtype != torch.bfloat16
    ):
        raise RuntimeError(
            "FA4 MLA prototype requires BF16 Q/KV tensors; launch with --kv-cache-dtype bf16."
        )

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode():
        raise RuntimeError(
            "FA4 MLA prototype does not support CUDA graph capture; launch with --disable-cuda-graph."
        )

    num_batches = cu_seqlens_q.numel() - 1
    if cache_seqlens.numel() != num_batches:
        raise RuntimeError(
            "FA4 MLA prototype expected cache_seqlens to match cu_seqlens_q batch count."
        )
    if page_table.shape[0] != num_batches:
        raise RuntimeError(
            "FA4 MLA prototype expected page_table rows to match cu_seqlens_q batch count."
        )

    output = torch.zeros_like(q_nope)
    softmax_lse = torch.full(
        (q_nope.shape[1], q_nope.shape[0]),
        -torch.inf,
        dtype=torch.float32,
        device=q_nope.device,
    )

    k_cache_flat = k_cache.reshape(-1, k_cache.shape[-1]).float()
    v_cache_flat = v_cache.reshape(-1, v_cache.shape[-1]).float()
    softmax_scale = softmax_scale or (
        1.0 / math.sqrt(q_nope.shape[-1] + q_rope.shape[-1])
    )

    for batch_idx in range(num_batches):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        q_len = q_end - q_start
        kv_len = int(cache_seqlens[batch_idx].item())

        if q_len == 0:
            continue
        if kv_len == 0:
            continue
        if page_table.shape[1] < kv_len:
            raise RuntimeError(
                "FA4 MLA prototype expected page_table width to cover cache_seqlens."
            )
        if causal and kv_len < q_len:
            raise RuntimeError(
                "FA4 MLA prototype expected causal MLA calls to satisfy kv_len >= q_len."
            )

        token_indices = page_table[batch_idx, :kv_len].to(dtype=torch.long)
        k_seq = k_cache_flat.index_select(0, token_indices)
        v_seq = v_cache_flat.index_select(0, token_indices)

        q_nope_seq = q_nope[q_start:q_end].float()
        q_rope_seq = q_rope[q_start:q_end].float()
        scores = torch.einsum("qhd,kd->hqk", q_nope_seq, v_seq)
        scores = scores + torch.einsum("qhd,kd->hqk", q_rope_seq, k_seq)
        scores = _apply_softcap(scores * softmax_scale, softcap)

        if causal:
            max_key_index = kv_len - q_len + torch.arange(
                q_len, device=scores.device, dtype=torch.long
            )
            key_index = torch.arange(kv_len, device=scores.device, dtype=torch.long)
            causal_mask = key_index.unsqueeze(0) <= max_key_index.unsqueeze(1)
            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min
            )

        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        output[q_start:q_end] = torch.einsum("hqk,kd->qhd", probs, v_seq).to(
            dtype=q_nope.dtype
        )
        softmax_lse[:, q_start:q_end] = lse

    return output, softmax_lse


@debug_kernel_api
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (-1, -1),
    learnable_sink: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_softmax_lse: bool = False,
    **_: object,
):
    if _flash_attn_varlen_func is None:  # pragma: no cover
        raise ImportError(
            "Vendored FlashAttention CUTE is not available (cannot import "
            "flash_attn.cute). Please check your source tree."
        ) from _flash_attn_import_error

    q, k, v = [_maybe_contiguous(t) for t in (q, k, v)]
    cu_seqlens_q, cu_seqlens_k = [
        _maybe_contiguous(t) for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    seqused_q, seqused_k = [_maybe_contiguous(t) for t in (seqused_q, seqused_k)]
    page_table = _maybe_contiguous(page_table)

    if learnable_sink is None and sinks is not None:
        learnable_sink = sinks

    if window_size == (-1, -1):
        window_size = (None, None)

    result = _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_lse=return_softmax_lse,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result


@debug_kernel_api
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: Optional[int] = None,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata=None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    sinks: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_softmax_lse: bool = False,
    **_: object,
):
    if qv is not None:
        if k is not None or v is not None:
            raise NotImplementedError(
                "FA4 MLA prototype does not support updating KV cache in-place."
            )
        if rotary_cos is not None or rotary_sin is not None or rotary_seqlens is not None:
            raise NotImplementedError("FA4 MLA prototype does not support rotary embedding.")
        if cache_batch_idx is not None or cache_leftpad is not None:
            raise NotImplementedError(
                "FA4 MLA prototype does not support non-consecutive batch indices or left padding."
            )
        if q_descale is not None or k_descale is not None or v_descale is not None:
            raise NotImplementedError("FA4 MLA prototype does not support descale.")

        if isinstance(cache_seqlens, int):
            if cu_seqlens_q is None:
                raise RuntimeError(
                    "FA4 MLA prototype cannot expand scalar cache_seqlens without cu_seqlens_q."
                )
            cache_seqlens = torch.full(
                (cu_seqlens_q.numel() - 1,),
                cache_seqlens,
                dtype=torch.int32,
                device=q.device,
            )

        output, softmax_lse = _mla_attention_ref_paged(
            q_nope=qv,
            q_rope=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
        )
        if return_softmax_lse:
            return output, softmax_lse
        return output

    if k is not None or v is not None:
        raise NotImplementedError("FA4 does not support updating KV cache in-place.")
    if rotary_cos is not None or rotary_sin is not None or rotary_seqlens is not None:
        raise NotImplementedError("FA4 path does not support rotary embedding.")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise NotImplementedError(
            "FA4 path does not support non-consecutive batch indices or left padding."
        )
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise NotImplementedError("FA4 path does not support descale.")

    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    result = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap if softcap != 0.0 else None,
        window_size=window_size,
        num_splits=num_splits if num_splits != 0 else 1,
        pack_gqa=pack_gqa,
        learnable_sink=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_softmax_lse=True,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result

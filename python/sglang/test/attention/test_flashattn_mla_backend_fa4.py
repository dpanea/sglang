import math
import unittest

import torch

from sglang.jit_kernel.flash_attention_v4 import flash_attn_with_kvcache
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleVerifyInput


def _apply_softcap(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    return scores


def _reference_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("qhd,kd->hqk", q_nope.float(), c_kv.float())
    scores = scores + torch.einsum("qhd,kd->hqk", q_rope.float(), k_rope.float())
    scores = _apply_softcap(scores * softmax_scale, softcap)
    if causal:
        q_len = q_nope.shape[0]
        kv_len = c_kv.shape[0]
        max_key_index = kv_len - q_len + torch.arange(
            q_len, device=scores.device, dtype=torch.long
        )
        key_index = torch.arange(kv_len, device=scores.device, dtype=torch.long)
        mask = key_index.unsqueeze(0) <= max_key_index.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous()
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,kd->qhd", probs, c_kv.float()).to(q_nope.dtype)
    return output, lse


class _MockRunner:
    def __init__(
        self,
        *,
        page_size: int = 1,
        kv_cache_dtype: torch.dtype = torch.bfloat16,
        disable_cuda_graph: bool = True,
    ):
        self.device = "cuda"
        self.sliding_window_size = None
        self.kv_cache_dtype = kv_cache_dtype
        self.page_size = page_size
        self.attn_cp_size = 1
        self.token_to_kv_pool = object()
        self.req_to_token_pool = type(
            "ReqToTokenPool",
            (),
            {"req_to_token": torch.zeros((1, 1), dtype=torch.int32, device="cuda")},
        )()
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": 16,
                "attention_arch": AttentionArch.MLA,
                "is_encoder_decoder": False,
                "is_local_attention_model": False,
            },
        )()
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "bf16",
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": None,
                "enable_deterministic_inference": False,
                "disable_cuda_graph": disable_cuda_graph,
            },
        )()


class _MockMLARunner:
    def __init__(
        self,
        *,
        context_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        topk: int,
        speculative_num_draft_tokens: int,
    ):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.kv_cache_dtype = torch.bfloat16
        self.page_size = 1
        self.sliding_window_size = None
        self.attn_cp_size = 1
        self.token_to_kv_pool = MLATokenToKVPool(
            size=context_len * 4,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        self.req_to_token_pool = type(
            "ReqToTokenPool",
            (),
            {
                "size": 4,
                "req_to_token": torch.zeros(
                    (4, context_len), dtype=torch.int32, device="cuda"
                ),
            },
        )()
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": context_len,
                "attention_arch": AttentionArch.MLA,
                "is_encoder_decoder": False,
                "is_local_attention_model": False,
            },
        )()
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "bf16",
                "speculative_eagle_topk": topk,
                "speculative_num_draft_tokens": speculative_num_draft_tokens,
                "enable_deterministic_inference": False,
                "disable_cuda_graph": True,
            },
        )()


def _build_verify_custom_mask(
    prefix_lens: list[int],
    draft_masks: list[torch.Tensor],
) -> torch.Tensor:
    rows = []
    device = draft_masks[0].device
    for prefix_len, draft_mask in zip(prefix_lens, draft_masks):
        draft_mask = draft_mask.to(dtype=torch.bool, device=device)
        draft_tokens = draft_mask.shape[1]
        prefix_part = torch.ones(
            (draft_tokens, prefix_len), dtype=torch.bool, device=device
        )
        rows.append(torch.cat((prefix_part, draft_mask), dim=1).reshape(-1))
    return torch.cat(rows, dim=0)


def _build_verify_spec_info(
    prefix_lens: list[int],
    draft_masks: list[torch.Tensor],
    topk: int,
) -> EagleVerifyInput:
    draft_token_num = draft_masks[0].shape[0]
    total_queries = len(prefix_lens) * draft_token_num
    custom_mask = _build_verify_custom_mask(prefix_lens, draft_masks)
    return EagleVerifyInput(
        draft_token=torch.arange(total_queries, dtype=torch.long, device="cuda"),
        custom_mask=custom_mask,
        positions=torch.arange(total_queries, dtype=torch.int64, device="cuda"),
        retrive_index=torch.full(
            (len(prefix_lens), draft_token_num),
            -1,
            dtype=torch.long,
            device="cuda",
        ),
        retrive_next_token=torch.full(
            (len(prefix_lens), draft_token_num),
            -1,
            dtype=torch.long,
            device="cuda",
        ),
        retrive_next_sibling=torch.full(
            (len(prefix_lens), draft_token_num),
            -1,
            dtype=torch.long,
            device="cuda",
        ),
        retrive_cum_len=None,
        spec_steps=draft_token_num - 1,
        topk=topk,
        draft_token_num=draft_token_num,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=sum(prefix_lens),
        seq_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int32),
    )


def _expected_expand_metadata(
    draft_token_locs: list[list[int]],
    draft_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_page_rows = []
    expected_seqlens = []
    for token_locs, draft_mask in zip(draft_token_locs, draft_masks):
        token_locs_tensor = torch.tensor(
            token_locs, dtype=torch.int32, device=draft_mask.device
        )
        for row_mask in draft_mask.to(dtype=torch.bool):
            valid = token_locs_tensor[row_mask]
            invalid = token_locs_tensor[~row_mask]
            expected_page_rows.append(torch.cat((valid, invalid), dim=0))
            expected_seqlens.append(int(row_mask.sum().item()))
    return torch.stack(expected_page_rows, dim=0), torch.tensor(
        expected_seqlens, dtype=torch.int32, device=draft_masks[0].device
    )


def _dense_target_verify_reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    prefix_c_kv: list[torch.Tensor],
    prefix_k_rope: list[torch.Tensor],
    draft_c_kv: list[torch.Tensor],
    draft_k_rope: list[torch.Tensor],
    draft_masks: list[torch.Tensor],
    softmax_scale: float,
    softcap: float,
) -> torch.Tensor:
    outputs = []
    query_index = 0
    for req_idx, draft_mask in enumerate(draft_masks):
        for row_mask in draft_mask.to(dtype=torch.bool):
            kv_c = torch.cat(
                (prefix_c_kv[req_idx], draft_c_kv[req_idx][row_mask]), dim=0
            )
            kv_rope = torch.cat(
                (prefix_k_rope[req_idx], draft_k_rope[req_idx][row_mask]), dim=0
            )
            ref_output, _ = _reference_attention(
                q_nope[query_index : query_index + 1],
                q_rope[query_index : query_index + 1],
                kv_c,
                kv_rope,
                softmax_scale=softmax_scale,
                causal=False,
                softcap=softcap,
            )
            outputs.append(ref_output)
            query_index += 1
    return torch.cat(outputs, dim=0)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlashAttention4MLAPrototype(unittest.TestCase):
    def test_tree_mask_metadata_path_matches_reference(self):
        device = "cuda"
        dtype = torch.bfloat16
        softmax_scale = 1.0 / math.sqrt(6)
        softcap = 0.0

        q_nope = torch.tensor(
            [
                [[0.5, -0.2, 0.1, 0.3]],
                [[-0.1, 0.2, 0.6, -0.4]],
                [[0.4, 0.1, -0.3, 0.2]],
            ],
            device=device,
            dtype=dtype,
        )
        q_rope = torch.tensor(
            [
                [[0.2, -0.1]],
                [[0.0, 0.3]],
                [[-0.2, 0.4]],
            ],
            device=device,
            dtype=dtype,
        )
        c_kv_tokens = torch.tensor(
            [
                [0.1, 0.0, 0.2, 0.3],
                [0.4, -0.2, 0.1, 0.0],
                [-0.3, 0.5, 0.2, -0.1],
                [0.2, 0.1, -0.4, 0.6],
                [0.7, -0.5, 0.3, 0.1],
            ],
            device=device,
            dtype=dtype,
        )
        k_rope_tokens = torch.tensor(
            [
                [0.2, 0.1],
                [-0.1, 0.3],
                [0.5, -0.2],
                [0.0, 0.4],
                [-0.3, 0.6],
            ],
            device=device,
            dtype=dtype,
        )

        k_cache = k_rope_tokens.view(-1, 1, 1, 2)
        v_cache = c_kv_tokens.view(-1, 1, 1, 4)
        page_table = torch.tensor(
            [
                [1, 0],
                [1, 3],
                [1, 4],
            ],
            device=device,
            dtype=torch.int32,
        )
        cache_seqlens = torch.tensor([1, 2, 2], device=device, dtype=torch.int32)
        cu_seqlens_q = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

        output, softmax_lse = flash_attn_with_kvcache(
            q=q_rope,
            qv=q_nope,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            softmax_scale=softmax_scale,
            softcap=softcap,
            causal=False,
            return_softmax_lse=True,
        )

        expected_output_parts = []
        expected_lse_parts = []
        for q_idx, token_indices in enumerate(([1], [1, 3], [1, 4])):
            ref_output, ref_lse = _reference_attention(
                q_nope[q_idx : q_idx + 1],
                q_rope[q_idx : q_idx + 1],
                c_kv_tokens[list(token_indices)],
                k_rope_tokens[list(token_indices)],
                softmax_scale=softmax_scale,
                causal=False,
                softcap=softcap,
            )
            expected_output_parts.append(ref_output)
            expected_lse_parts.append(ref_lse)

        expected_output = torch.cat(expected_output_parts, dim=0)
        expected_lse = torch.cat(expected_lse_parts, dim=0)

        torch.testing.assert_close(output, expected_output, rtol=1e-3, atol=3e-3)
        torch.testing.assert_close(
            softmax_lse, expected_lse.T.contiguous(), rtol=1e-3, atol=3e-3
        )

    def test_nonverify_causal_path_matches_reference(self):
        device = "cuda"
        dtype = torch.bfloat16
        softmax_scale = 1.0 / math.sqrt(6)
        softcap = 1.5

        q_nope = torch.randn(3, 2, 4, device=device, dtype=dtype)
        q_rope = torch.randn(3, 2, 2, device=device, dtype=dtype)
        c_kv_tokens = torch.randn(5, 4, device=device, dtype=dtype)
        k_rope_tokens = torch.randn(5, 2, device=device, dtype=dtype)

        output, softmax_lse = flash_attn_with_kvcache(
            q=q_rope,
            qv=q_nope,
            k_cache=k_rope_tokens.view(-1, 1, 1, 2),
            v_cache=c_kv_tokens.view(-1, 1, 1, 4),
            page_table=torch.tensor([[0, 1, 2, 3, 4]], device=device, dtype=torch.int32),
            cache_seqlens=torch.tensor([5], device=device, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 3], device=device, dtype=torch.int32),
            softmax_scale=softmax_scale,
            softcap=softcap,
            causal=True,
            return_softmax_lse=True,
        )

        expected_output, expected_lse = _reference_attention(
            q_nope,
            q_rope,
            c_kv_tokens,
            k_rope_tokens,
            softmax_scale=softmax_scale,
            causal=True,
            softcap=softcap,
        )

        torch.testing.assert_close(output, expected_output, rtol=1e-3, atol=4e-3)
        torch.testing.assert_close(
            softmax_lse, expected_lse.T.contiguous(), rtol=1e-3, atol=4e-3
        )

    def test_backend_rejects_nonprototype_settings(self):
        with self.assertRaisesRegex(ValueError, "--page-size 1"):
            FlashAttentionBackend(_MockRunner(page_size=2), fa_impl_ver=4)

        with self.assertRaisesRegex(ValueError, "BF16 KV cache"):
            FlashAttentionBackend(
                _MockRunner(kv_cache_dtype=torch.float16),
                fa_impl_ver=4,
            )

        with self.assertRaisesRegex(ValueError, "--disable-cuda-graph"):
            FlashAttentionBackend(
                _MockRunner(disable_cuda_graph=False),
                fa_impl_ver=4,
            )

    def test_target_verify_metadata_matches_dense_custom_mask(self):
        device = "cuda"
        prefix_lens = [2, 1]
        draft_token_num = 3
        topk = 2
        draft_masks = [
            torch.tensor(
                [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
                dtype=torch.bool,
                device=device,
            ),
            torch.tensor(
                [[1, 0, 0], [0, 1, 0], [1, 1, 1]],
                dtype=torch.bool,
                device=device,
            ),
        ]
        runner = _MockMLARunner(
            context_len=8,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            topk=topk,
            speculative_num_draft_tokens=draft_token_num,
        )
        backend = FlashAttentionBackend(runner, fa_impl_ver=4)
        req_to_token = runner.req_to_token_pool.req_to_token
        req_to_token[0, :5] = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
        req_to_token[1, :4] = torch.tensor([6, 7, 8, 9], dtype=torch.int32, device=device)
        spec_info = _build_verify_spec_info(prefix_lens, draft_masks, topk=topk)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            input_ids=spec_info.draft_token,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.long, device=device),
            seq_lens=torch.tensor(prefix_lens, dtype=torch.int32, device=device),
            out_cache_loc=torch.tensor([3, 4, 5, 7, 8, 9], dtype=torch.int64, device=device),
            seq_lens_sum=sum(prefix_lens),
            seq_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int32),
            positions=spec_info.positions,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
            attn_backend=backend,
            spec_info=spec_info,
        )

        backend.init_forward_metadata(forward_batch)

        metadata = backend.forward_metadata
        metadata_expand = backend.forward_metadata_spec_decode_expand
        expected_expand_page_table, expected_expand_seqlens = _expected_expand_metadata(
            draft_token_locs=[[3, 4, 5], [7, 8, 9]],
            draft_masks=draft_masks,
        )

        torch.testing.assert_close(
            metadata.page_table,
            torch.tensor([[1, 2], [6, 7]], dtype=torch.int32, device=device),
        )
        torch.testing.assert_close(
            metadata.cache_seqlens_int32,
            torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        )
        torch.testing.assert_close(
            metadata_expand.page_table, expected_expand_page_table
        )
        torch.testing.assert_close(
            metadata_expand.cache_seqlens_int32, expected_expand_seqlens
        )
        torch.testing.assert_close(
            metadata_expand.cu_seqlens_k,
            torch.tensor([0, 1, 3, 5, 6, 7, 10], dtype=torch.int32, device=device),
        )

    def test_target_verify_forward_path_matches_dense_tree_reference(self):
        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.bfloat16
        kv_lora_rank = 8
        qk_rope_head_dim = 4
        num_heads = 2
        prefix_lens = [2, 1]
        draft_token_num = 3
        topk = 2
        softmax_scale = 1.0 / math.sqrt(kv_lora_rank + qk_rope_head_dim)
        softcap = 0.0

        draft_masks = [
            torch.tensor(
                [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
                dtype=torch.bool,
                device=device,
            ),
            torch.tensor(
                [[1, 0, 0], [0, 1, 0], [1, 1, 1]],
                dtype=torch.bool,
                device=device,
            ),
        ]
        runner = _MockMLARunner(
            context_len=8,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            topk=topk,
            speculative_num_draft_tokens=draft_token_num,
        )
        backend = FlashAttentionBackend(runner, fa_impl_ver=4)
        layer = RadixAttention(
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            scaling=softmax_scale,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=kv_lora_rank,
            prefix="attn_mqa",
        )

        req_to_token = runner.req_to_token_pool.req_to_token
        req_to_token[0, :5] = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
        req_to_token[1, :4] = torch.tensor([6, 7, 8, 9], dtype=torch.int32, device=device)
        prefix_locs = torch.tensor([1, 2, 6], dtype=torch.int64, device=device)
        draft_locs = torch.tensor([3, 4, 5, 7, 8, 9], dtype=torch.int64, device=device)

        prefix_c_kv = torch.randn(prefix_locs.numel(), 1, kv_lora_rank, dtype=dtype, device=device)
        prefix_k_rope = torch.randn(prefix_locs.numel(), 1, qk_rope_head_dim, dtype=dtype, device=device)
        runner.token_to_kv_pool.set_mla_kv_buffer(layer, prefix_locs, prefix_c_kv, prefix_k_rope)

        draft_k_nope = torch.randn(draft_locs.numel(), 1, kv_lora_rank, dtype=dtype, device=device)
        draft_k_rope = torch.randn(draft_locs.numel(), 1, qk_rope_head_dim, dtype=dtype, device=device)
        q_nope = torch.randn(
            draft_locs.numel(), num_heads, kv_lora_rank, dtype=dtype, device=device
        )
        q_rope = torch.randn(
            draft_locs.numel(), num_heads, qk_rope_head_dim, dtype=dtype, device=device
        )

        spec_info = _build_verify_spec_info(prefix_lens, draft_masks, topk=topk)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            input_ids=spec_info.draft_token,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.long, device=device),
            seq_lens=torch.tensor(prefix_lens, dtype=torch.int32, device=device),
            out_cache_loc=draft_locs,
            seq_lens_sum=sum(prefix_lens),
            seq_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int32),
            positions=spec_info.positions,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
            attn_backend=backend,
            spec_info=spec_info,
        )

        backend.init_forward_metadata(forward_batch)
        output = backend.forward(
            q_nope,
            draft_k_nope,
            torch.zeros(1, dtype=dtype, device=device),
            layer,
            forward_batch,
            q_rope=q_rope,
            k_rope=draft_k_rope,
        )

        prefix_c_kv_by_req = [
            prefix_c_kv[:2, 0, :],
            prefix_c_kv[2:, 0, :],
        ]
        prefix_k_rope_by_req = [
            prefix_k_rope[:2, 0, :],
            prefix_k_rope[2:, 0, :],
        ]
        draft_c_kv_by_req = [
            draft_k_nope[:3, 0, :],
            draft_k_nope[3:, 0, :],
        ]
        draft_k_rope_by_req = [
            draft_k_rope[:3, 0, :],
            draft_k_rope[3:, 0, :],
        ]
        expected_output = _dense_target_verify_reference(
            q_nope=q_nope,
            q_rope=q_rope,
            prefix_c_kv=prefix_c_kv_by_req,
            prefix_k_rope=prefix_k_rope_by_req,
            draft_c_kv=draft_c_kv_by_req,
            draft_k_rope=draft_k_rope_by_req,
            draft_masks=draft_masks,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )

        torch.testing.assert_close(
            output.view_as(expected_output),
            expected_output,
            rtol=5e-2,
            atol=5e-2,
        )

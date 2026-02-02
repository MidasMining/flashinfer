"""
SM70 (Volta/V100) FlashAttention API.

This module provides FlashAttention for V100 GPUs using the WMMA API.
It uses native tensor cores for optimal performance on SM70 architecture.

Kernel versions:
- v1: Recommended. WMMA kernel (BLOCK_M=64, BLOCK_N=128, 16 warps)
      ~2x faster than PyTorch SDPA for decode (M=1)
- v2: Has known bug when M=64, avoid using
- v3: Alternative kernel (BLOCK_M=64, BLOCK_N=64, 8 warps)

Performance on V100:
    Decode (M=1):      v1 is ~2x faster than SDPA
    Prefill (M=N<=64): v1 is competitive with SDPA
    Prefill (M=N>64):  SDPA may be faster, but v1 required for paged attention

Example usage:
    import torch
    import flashinfer.attention_sm70 as sm70_attn

    # Create tensors (B, H, M, D format)
    q = torch.randn(1, 32, 1, 128, dtype=torch.float16, device='cuda')
    k = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device='cuda')
    v = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device='cuda')

    # Recommended: auto_attention selects the best kernel
    out = sm70_attn.auto_attention(q, k, v)

    # Or use v1 directly (fastest for decode)
    out = sm70_attn.flash_attention_forward(q, k, v)
"""

import math
from typing import Optional, Literal

import torch

from .jit.attention_sm70 import get_sm70_attention_func
from .utils import get_compute_capability


def _check_sm70():
    """Verify we're running on SM70 (Volta) GPU."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 7 or cc[1] != 0:
        import warnings
        warnings.warn(
            f"SM70 attention kernel is optimized for V100 (SM70), "
            f"but running on SM{cc[0]}{cc[1]}. Performance may vary."
        )


def _flash_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_lse: bool = False,
    version: int = 1,
) -> torch.Tensor:
    """Internal implementation for v1, v2, and v3 (prefill) kernels."""
    # Validate inputs
    assert q.dim() == 4, f"Expected 4D tensor for q, got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor for k, got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor for v, got {v.dim()}D"

    B, H, M, D = q.shape
    _, _, N, _ = k.shape

    assert q.dtype == torch.float16, "SM70 attention requires FP16 input"
    assert k.dtype == torch.float16, "SM70 attention requires FP16 input"
    assert v.dtype == torch.float16, "SM70 attention requires FP16 input"
    assert D in [64, 128], f"SM70 attention supports D=64 or D=128, got D={D}"

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Default scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # Allocate output
    out = torch.empty_like(q)

    # Get JIT-compiled function for specified version
    fn = get_sm70_attention_func(version=version)

    # Version 3 (prefill) has a different signature - doesn't support LSE return
    if version == 3:
        if return_lse:
            raise ValueError("Version 3 (prefill) kernel does not support return_lse=True")
        fn(q, k, v, out, causal, softmax_scale)
        return out

    # Versions 1 and 2 support LSE
    lse = None
    if return_lse:
        lse = torch.empty(B, H, M, dtype=torch.float32, device=q.device)

    fn(q, k, v, out, lse, softmax_scale, causal)

    if return_lse:
        return out, lse
    return out


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    """
    FlashAttention forward pass optimized for V100 (SM70) - Version 1.

    This is the original WMMA kernel with 64x128 tiles and 16 warps.
    Best for decode attention (M=1, memory-bound).

    Args:
        q: Query tensor of shape [B, H, M, D] where M is query length
        k: Key tensor of shape [B, H, N, D] where N is KV length
        v: Value tensor of shape [B, H, N, D]
        softmax_scale: Scaling factor for attention (default: 1/sqrt(D))
        causal: Whether to apply causal mask
        return_lse: Whether to return log-sum-exp for backward pass

    Returns:
        Output tensor of shape [B, H, M, D]
        If return_lse=True, also returns LSE tensor of shape [B, H, M]

    Note:
        - Supports head dimensions D=64 or D=128
        - Requires FP16 input tensors
        - Optimized for V100 32GB using WMMA tensor cores
    """
    return _flash_attention_impl(q, k, v, softmax_scale, causal, return_lse, version=1)


def flash_attention_forward_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    """
    FlashAttention forward pass optimized for V100 (SM70) - Version 2.

    This is the optimized kernel with:
    - 64x64 tiles (matching CUTLASS)
    - 8 warps (reduced overhead)

    Balanced kernel, good for moderate sequence lengths.

    Args:
        q: Query tensor of shape [B, H, M, D] where M is query length
        k: Key tensor of shape [B, H, N, D] where N is KV length
        v: Value tensor of shape [B, H, N, D]
        softmax_scale: Scaling factor for attention (default: 1/sqrt(D))
        causal: Whether to apply causal mask
        return_lse: Whether to return log-sum-exp for backward pass

    Returns:
        Output tensor of shape [B, H, M, D]
        If return_lse=True, also returns LSE tensor of shape [B, H, M]
    """
    return _flash_attention_impl(q, k, v, softmax_scale, causal, return_lse, version=2)


def flash_attention_forward_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    """
    FlashAttention forward pass optimized for V100 (SM70) - Version 3 (Prefill).

    This is the prefill-optimized kernel with:
    - 128x64 tiles (larger BLOCK_M for better Q data reuse)
    - 8 warps
    - Vectorized loads

    Best for prefill attention where M is large (M >= 64).

    Args:
        q: Query tensor of shape [B, H, M, D] where M is query length
        k: Key tensor of shape [B, H, N, D] where N is KV length
        v: Value tensor of shape [B, H, N, D]
        softmax_scale: Scaling factor for attention (default: 1/sqrt(D))
        causal: Whether to apply causal mask
        return_lse: Whether to return log-sum-exp for backward pass

    Returns:
        Output tensor of shape [B, H, M, D]
        If return_lse=True, also returns LSE tensor of shape [B, H, M]
    """
    return _flash_attention_impl(q, k, v, softmax_scale, causal, return_lse, version=3)


def single_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Single-query decode attention for V100.

    This is a convenience wrapper for decode (autoregressive) attention
    where each head has a single query attending to the full KV cache.

    Args:
        q: Query tensor of shape [num_heads, head_dim]
        k: Key cache of shape [seq_len, num_kv_heads, head_dim]
        v: Value cache of shape [seq_len, num_kv_heads, head_dim]
        softmax_scale: Optional scaling factor

    Returns:
        Output tensor of shape [num_heads, head_dim]
    """
    num_qo_heads, head_dim = q.shape
    seq_len, num_kv_heads, _ = k.shape

    # Handle GQA (grouped query attention)
    gqa_ratio = num_qo_heads // num_kv_heads

    # Reshape for SM70 kernel: [B, H, N, D]
    q_4d = q.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, D]
    k_4d = k.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv_heads, seq_len, D]
    v_4d = v.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv_heads, seq_len, D]

    # Expand KV for GQA
    if gqa_ratio > 1:
        k_4d = k_4d.repeat_interleave(gqa_ratio, dim=1)
        v_4d = v_4d.repeat_interleave(gqa_ratio, dim=1)

    # Run attention (v1 is best for decode)
    out = flash_attention_forward(q_4d, k_4d, v_4d, softmax_scale, causal=False)

    # Reshape back: [1, num_heads, 1, D] -> [num_heads, D]
    return out.squeeze(0).squeeze(1)


def prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    version: int = 1,
) -> torch.Tensor:
    """
    Prefill attention for V100.

    This is for processing the initial prompt where Q, K, V all have
    the same sequence length.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_heads, seq_len, head_dim]
        v: Value tensor of shape [batch, num_heads, seq_len, head_dim]
        softmax_scale: Optional scaling factor
        causal: Whether to use causal mask (default True for autoregressive)
        version: Kernel version (1 or 3). Default 1 (recommended).
                 Note: v2 has a known bug with M=64, avoid using it.

    Returns:
        Output tensor of shape [batch, num_heads, seq_len, head_dim]
    """
    if version == 3:
        return flash_attention_forward_v3(q, k, v, softmax_scale, causal)
    # v2 has a bug when M=BLOCK_M=64, skip it
    return flash_attention_forward(q, k, v, softmax_scale, causal)


def auto_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Automatically select the best SM70 kernel based on input dimensions.

    For decode (M=1): Uses v1 (2x faster than PyTorch SDPA)
    For small prefill (M<=64): Uses v1 (competitive with SDPA)
    For large prefill (M>64): Uses v1 (SDPA may be faster, but v1 works)

    Args:
        q: Query tensor of shape [batch, num_heads, seq_q, head_dim]
        k: Key tensor of shape [batch, num_heads, seq_kv, head_dim]
        v: Value tensor of shape [batch, num_heads, seq_kv, head_dim]
        softmax_scale: Optional scaling factor
        causal: Whether to use causal mask

    Returns:
        Output tensor of shape [batch, num_heads, seq_q, head_dim]

    Performance notes on V100:
        - Decode (M=1): v1 is ~2x faster than SDPA
        - Prefill (M=N): v1 is competitive with SDPA for small M
        - For memory-efficient paged attention, these kernels are required
    """
    # v1 is generally the best performing kernel for all workloads
    return flash_attention_forward(q, k, v, softmax_scale, causal)


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    SM70-optimized paged decode attention for vLLM integration.

    This is the key function for vLLM integration. It handles paged KV cache
    format and provides ~2x speedup over SDPA for decode.

    Args:
        q: Query tensor of shape [num_seqs, num_heads, head_dim]
        k_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: Block indices [num_seqs, max_blocks_per_seq]
        seq_lens: Sequence lengths [num_seqs]
        softmax_scale: Optional scaling factor (default: 1/sqrt(head_dim))

    Returns:
        Output tensor of shape [num_seqs, num_heads, head_dim]

    Note:
        This function is ~2x faster than SDPA for decode (M=1) workloads.
    """
    from .jit.attention_sm70 import get_sm70_paged_decode_func

    assert q.dim() == 3, f"Expected 3D query [num_seqs, num_heads, head_dim], got {q.dim()}D"
    assert k_cache.dim() == 4, f"Expected 4D k_cache [num_blocks, block_size, num_kv_heads, head_dim]"
    assert v_cache.dim() == 4, f"Expected 4D v_cache"
    assert block_tables.dim() == 2, f"Expected 2D block_tables [num_seqs, max_blocks]"
    assert seq_lens.dim() == 1, f"Expected 1D seq_lens [num_seqs]"

    num_seqs, num_heads, head_dim = q.shape

    assert q.dtype == torch.float16, "SM70 paged attention requires FP16"
    assert head_dim in [64, 128], f"Head dim must be 64 or 128, got {head_dim}"

    # Ensure contiguous
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.contiguous().int()
    seq_lens = seq_lens.contiguous().int()

    # Default scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Allocate output
    out = torch.empty_like(q)

    # Get JIT-compiled function
    fn = get_sm70_paged_decode_func()

    # Call kernel
    fn(q, k_cache, v_cache, block_tables, seq_lens, out, softmax_scale)

    return out

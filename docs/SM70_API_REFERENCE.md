# SM70 FlashAttention API Reference

## Module: `flashinfer.attention_sm70`

This module provides FlashAttention optimized for NVIDIA Volta (V100) GPUs.

### Quick Start

```python
import torch
from flashinfer.attention_sm70 import (
    flash_attention_forward,    # Dense attention (recommended)
    paged_decode_attention,     # Paged KV cache decode
    auto_attention,             # Auto-select best kernel
    single_decode_attention,    # Convenience wrapper for decode
    prefill_attention,          # Prefill with causal mask
)
```

---

## Dense Attention Functions

### `flash_attention_forward`

The primary dense attention function. Recommended for most use cases.

```python
def flash_attention_forward(
    q: torch.Tensor,              # [B, H, M, D]
    k: torch.Tensor,              # [B, H, N, D]
    v: torch.Tensor,              # [B, H, N, D]
    softmax_scale: float = None,  # Default: 1/sqrt(D)
    causal: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:                # [B, H, M, D]
```

**Parameters:**
- `q`: Query tensor, shape `[batch, num_heads, seq_q, head_dim]`
- `k`: Key tensor, shape `[batch, num_heads, seq_kv, head_dim]`
- `v`: Value tensor, shape `[batch, num_heads, seq_kv, head_dim]`
- `softmax_scale`: Attention scaling factor (default: `1/sqrt(head_dim)`)
- `causal`: Apply causal (triangular) mask
- `return_lse`: Return log-sum-exp for backward pass

**Returns:**
- Output tensor, shape `[batch, num_heads, seq_q, head_dim]`
- If `return_lse=True`: tuple of (output, lse) where lse has shape `[B, H, M]`

**Example:**
```python
import torch
from flashinfer.attention_sm70 import flash_attention_forward

# Decode attention (M=1)
q = torch.randn(1, 32, 1, 128, dtype=torch.float16, device='cuda')
k = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device='cuda')
v = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device='cuda')

output = flash_attention_forward(q, k, v, causal=False)
print(output.shape)  # torch.Size([1, 32, 1, 128])
```

---

### `auto_attention`

Automatically selects the best kernel based on input dimensions.

```python
def auto_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = None,
    causal: bool = False,
) -> torch.Tensor:
```

**Notes:**
- Currently uses v1 kernel for all cases (best overall performance)
- May be updated to use different kernels based on profiling

---

### `single_decode_attention`

Convenience wrapper for single-query decode attention.

```python
def single_decode_attention(
    q: torch.Tensor,              # [num_heads, head_dim]
    k: torch.Tensor,              # [seq_len, num_kv_heads, head_dim]
    v: torch.Tensor,              # [seq_len, num_kv_heads, head_dim]
    softmax_scale: float = None,
) -> torch.Tensor:                # [num_heads, head_dim]
```

**Example:**
```python
from flashinfer.attention_sm70 import single_decode_attention

q = torch.randn(32, 128, dtype=torch.float16, device='cuda')
k = torch.randn(2048, 8, 128, dtype=torch.float16, device='cuda')  # GQA: 8 KV heads
v = torch.randn(2048, 8, 128, dtype=torch.float16, device='cuda')

output = single_decode_attention(q, k, v)
print(output.shape)  # torch.Size([32, 128])
```

---

### `prefill_attention`

Prefill attention for initial prompt processing.

```python
def prefill_attention(
    q: torch.Tensor,              # [B, H, M, D]
    k: torch.Tensor,              # [B, H, N, D]
    v: torch.Tensor,              # [B, H, N, D]
    softmax_scale: float = None,
    causal: bool = True,
    version: int = 1,             # 1 or 3
) -> torch.Tensor:
```

**Notes:**
- `version=1`: Original WMMA kernel (recommended)
- `version=3`: Prefill-optimized kernel
- `version=2`: Has known bug with M=64, avoid using

---

## Paged Attention Functions

### `paged_decode_attention`

Native paged decode attention for vLLM integration.

```python
def paged_decode_attention(
    q: torch.Tensor,              # [num_seqs, num_heads, head_dim]
    k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,   # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,       # [num_seqs]
    softmax_scale: float = None,
) -> torch.Tensor:                # [num_seqs, num_heads, head_dim]
```

**Parameters:**
- `q`: Query tensor for each sequence, shape `[num_seqs, num_heads, head_dim]`
- `k_cache`: Paged key cache, shape `[num_blocks, block_size, num_kv_heads, head_dim]`
- `v_cache`: Paged value cache, same shape as k_cache
- `block_tables`: Block indices for each sequence, shape `[num_seqs, max_blocks_per_seq]`
- `seq_lens`: Actual sequence length for each sequence, shape `[num_seqs]`
- `softmax_scale`: Attention scaling factor

**KV Cache Format:**
```
k_cache[block_idx, token_idx, kv_head_idx, dim_idx]
       ^^^^^^^^^   ^^^^^^^^^   ^^^^^^^^^^^   ^^^^^^^
       Physical    Token       KV head       Feature
       block ID    within      index         dimension
                   block
```

**Example:**
```python
from flashinfer.attention_sm70 import paged_decode_attention

# Setup
num_seqs = 4
num_heads = 32
num_kv_heads = 8  # GQA ratio = 4
head_dim = 128
block_size = 16
num_blocks = 100

# Create paged KV cache
k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                      dtype=torch.float16, device='cuda')
v_cache = torch.randn_like(k_cache)

# Block tables: which physical blocks each sequence uses
# Sequence 0 uses blocks [0, 1, 2], sequence 1 uses blocks [3, 4], etc.
block_tables = torch.tensor([
    [0, 1, 2, 0, 0],  # seq 0: 48 tokens (3 blocks)
    [3, 4, 0, 0, 0],  # seq 1: 32 tokens (2 blocks)
    [5, 6, 7, 8, 0],  # seq 2: 64 tokens (4 blocks)
    [9, 0, 0, 0, 0],  # seq 3: 16 tokens (1 block)
], dtype=torch.int32, device='cuda')

seq_lens = torch.tensor([48, 32, 64, 16], dtype=torch.int32, device='cuda')

# Query (one per sequence)
q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.float16, device='cuda')

# Run attention
output = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
print(output.shape)  # torch.Size([4, 32, 128])
```

---

## Data Types and Constraints

### Supported Data Types

| Tensor | Supported Types |
|--------|-----------------|
| Query | `torch.float16` |
| Key | `torch.float16` |
| Value | `torch.float16` |
| Output | `torch.float16` |
| Block Tables | `torch.int32` |
| Sequence Lengths | `torch.int32` |

### Constraints

| Parameter | Supported Values |
|-----------|------------------|
| Head dimension (D) | 64, 128 |
| Block size | Any (typically 16, 32) |
| Batch size | Any |
| Sequence length | Any |
| Number of heads | Any |
| GQA ratio | Any (num_heads / num_kv_heads must be integer) |

---

## Error Handling

The kernels validate inputs and raise descriptive errors:

```python
# Invalid head dimension
>>> flash_attention_forward(q_d32, k_d32, v_d32)
AssertionError: SM70 attention supports D=64 or D=128, got D=32

# Invalid dtype
>>> flash_attention_forward(q_bf16, k_bf16, v_bf16)
AssertionError: SM70 attention requires FP16 input

# Shape mismatch
>>> paged_decode_attention(q_wrong_shape, ...)
AssertionError: Expected 3D query [num_seqs, num_heads, head_dim], got 4D
```

---

## Performance Tips

1. **Use FP16**: The kernel only supports FP16; ensure inputs are `.half()`
2. **Contiguous tensors**: Call `.contiguous()` on non-contiguous inputs
3. **Batch decode**: For multiple sequences, use `paged_decode_attention` instead of looping
4. **Avoid small batches for prefill**: For M>64, PyTorch SDPA may be faster

---

## JIT Compilation

Kernels are JIT-compiled on first use:

```python
# First call compiles the kernel (takes a few seconds)
output = flash_attention_forward(q, k, v)

# Subsequent calls use cached compiled kernel
output = flash_attention_forward(q, k, v)  # Fast
```

To clear the JIT cache:
```bash
rm -rf ~/.cache/flashinfer/
```

---

## Thread Safety

The JIT compilation uses `@functools.cache` which is thread-safe. Multiple threads can safely call the attention functions concurrently after initial compilation.

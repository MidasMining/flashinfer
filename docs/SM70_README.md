# SM70 (Volta/V100) FlashAttention Support

Native FlashAttention implementation for NVIDIA Volta GPUs using WMMA tensor cores.

## Features

- ✅ **~2x decode speedup** over PyTorch SDPA
- ✅ **Native paged KV cache** support for vLLM
- ✅ **GQA support** (grouped-query attention)
- ✅ **JIT compiled** - no pre-compilation needed
- ✅ **Drop-in vLLM patch** - works with existing models

## Quick Start

### Standalone Usage

```python
import torch
from flashinfer.attention_sm70 import flash_attention_forward

# Decode attention (M=1) - 2x faster than SDPA
q = torch.randn(1, 32, 1, 128, dtype=torch.float16, device='cuda')
k = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device='cuda')
v = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device='cuda')

output = flash_attention_forward(q, k, v)
```

### vLLM Integration

```python
import sys
sys.path.insert(0, "/path/to/flashinfer-src/python")

from flashinfer.vllm_sm70_patch import apply_sm70_patch
apply_sm70_patch()  # Apply BEFORE importing vLLM

from vllm import LLM
llm = LLM(model="your-model", dtype="float16")
```

### Paged Attention

```python
from flashinfer.attention_sm70 import paged_decode_attention

output = paged_decode_attention(
    q,            # [num_seqs, num_heads, head_dim]
    k_cache,      # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,
    block_tables, # [num_seqs, max_blocks]
    seq_lens,     # [num_seqs]
)
```

## Performance

| Workload | vs SDPA | Notes |
|----------|---------|-------|
| Decode (M=1) | **~2x faster** | Primary LLM inference case |
| Prefill (M≤64) | ~1x | Competitive |
| Prefill (M>64) | ~0.5x | SDPA faster for large prefill |

## Requirements

- NVIDIA V100 GPU (SM70)
- CUDA 11.x or 12.x
- PyTorch 2.0+
- FP16 tensors only

## Documentation

- [SM70 FlashAttention Overview](SM70_FLASHATTENTION.md)
- [API Reference](SM70_API_REFERENCE.md)
- [vLLM Integration Guide](SM70_VLLM_INTEGRATION.md)

## Supported Configurations

| Parameter | Values |
|-----------|--------|
| Head dimension | 64, 128 |
| Data type | FP16 |
| Block sizes | Any (16, 32 typical) |
| GQA | Any ratio |
| Causal mask | Yes |
| Sliding window | No |
| ALiBi | No |

## Files

```
csrc/
├── attention_sm70_volta.cu           # Dense attention (v1)
├── attention_sm70_volta_prefill.cu   # Prefill kernel (v3)
├── attention_sm70_volta_paged.cu     # Native paged decode
└── *_binding.cu                      # TVM-FFI bindings

flashinfer/
├── attention_sm70.py                 # Python API
└── jit/attention_sm70.py             # JIT module

python/flashinfer/
└── vllm_sm70_patch.py                # vLLM integration
```

## Why SM70 Needs Special Kernels

Volta GPUs lack the `ldmatrix` PTX instruction used by standard FlashInfer kernels. This implementation:

1. Uses explicit shared memory loads
2. Implements WMMA-based matrix multiply
3. Stays within V100's 96KB shared memory limit
4. Uses warp shuffles for efficient reductions

## License

Apache 2.0 (same as FlashInfer)

## Authors

- FlashInfer Team
- Co-authored with Claude Opus 4.5

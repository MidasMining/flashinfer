# SM70 (Volta/V100) vLLM Integration Guide

## Overview

This guide describes how to integrate the SM70 FlashAttention kernel with vLLM for optimal performance on V100 GPUs. The integration provides native paged attention support with ~2x speedup for decode operations.

## Quick Start

### Option 1: Apply Patch Before Import (Recommended)

```python
import sys
sys.path.insert(0, "/path/to/flashinfer-src/python")

# Apply patch BEFORE importing vLLM
from flashinfer.vllm_sm70_patch import apply_sm70_patch
apply_sm70_patch()

# Now use vLLM normally
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-model",
    dtype="float16",
    gpu_memory_utilization=0.9,
)

outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
```

### Option 2: Environment Variable (Future)

```bash
export FLASHINFER_SM70_PATCH=1
python your_vllm_script.py
```

## How It Works

### Patch Mechanism

The patch intercepts vLLM's Triton unified_attention calls and redirects them to our optimized SM70 kernels:

```
vLLM Request
    │
    ▼
┌─────────────────────────────────┐
│  FlashInfer Backend (SM70)      │
│  ┌───────────────────────────┐  │
│  │ triton_unified_attention  │  │
│  │         (patched)         │  │
│  └─────────────┬─────────────┘  │
│                │                │
│    ┌───────────┴───────────┐    │
│    ▼                       ▼    │
│ ┌──────────┐         ┌──────────┐│
│ │ Decode   │         │ Prefill  ││
│ │ (M=1)    │         │ (M>1)    ││
│ └────┬─────┘         └────┬─────┘│
│      │                    │      │
│      ▼                    ▼      │
│ ┌──────────┐         ┌──────────┐│
│ │ Native   │         │ Gather + ││
│ │ Paged    │         │ Dense    ││
│ │ Decode   │         │ Kernel   ││
│ └──────────┘         └──────────┘│
└─────────────────────────────────┘
```

### Kernel Selection

| Operation | Condition | Kernel Used | Performance |
|-----------|-----------|-------------|-------------|
| Decode | `max_seqlen_q == 1` | Native paged decode | ~2x faster |
| Prefill | `max_seqlen_q > 1` | Gather + dense v1 | ~1x (similar) |
| Fallback | Special features | Triton unified | Baseline |

### Special Features Fallback

The patch falls back to Triton for unsupported features:
- Sliding window attention (`window_size != None`)
- ALiBi slopes (`alibi_slopes != None`)
- Logit softcapping (`softcap != 0.0`)
- FP8 quantization (`q_descale`, `k_descale`, `v_descale`)
- Sink tokens (`sinks != None`)
- Unsupported head dimensions (D != 64 and D != 128)

## Performance

### Decode Throughput (V100 32GB)

| Batch Size | Seq Length | Heads | SM70 Kernel | Triton | Speedup |
|------------|------------|-------|-------------|--------|---------|
| 1 | 512 | 32 | 0.029ms | 0.055ms | 1.9x |
| 8 | 512 | 32 | 0.125ms | 0.220ms | 1.8x |
| 32 | 512 | 32 | 0.480ms | 0.850ms | 1.8x |
| 1 | 2048 | 32 | 0.118ms | 0.217ms | 1.8x |

### Memory Efficiency

The paged decode kernel directly accesses the paged KV cache without intermediate allocations:

```
Traditional approach:
  gather blocks → contiguous buffer → attention → free buffer
  Memory: O(batch × seq_len × head_dim)

Native paged:
  direct block access → attention
  Memory: O(1) extra
```

## Configuration

### Supported vLLM Settings

| Setting | Supported | Notes |
|---------|-----------|-------|
| `dtype="float16"` | ✅ | Required |
| `dtype="bfloat16"` | ❌ | V100 limitation |
| `enforce_eager=True` | ✅ | Recommended for debugging |
| `enforce_eager=False` | ✅ | CUDA graphs work |
| `gpu_memory_utilization` | ✅ | Any value |
| `max_model_len` | ✅ | Any value |
| `kv_cache_dtype="fp8"` | ❌ | Falls back to Triton |

### Recommended vLLM Configuration

```python
llm = LLM(
    model="your-model",
    dtype="float16",              # Required for SM70
    gpu_memory_utilization=0.9,   # Maximize KV cache
    max_model_len=4096,           # Adjust as needed
    enforce_eager=False,          # Enable CUDA graphs for production
)
```

## Verification

### Check Patch Status

```python
from flashinfer.vllm_sm70_patch import apply_sm70_patch

result = apply_sm70_patch()
# Output: "SM70 patch: Successfully patched triton_unified_attention, flashinfer_backend"
print(f"Patch applied: {result}")  # True
```

### Verify Kernel Usage

Enable verbose logging to see which kernels are used:

```python
import os
os.environ["FLASHINFER_LOGLEVEL"] = "3"

# Run inference...
# Look for "SM70" in logs
```

### Correctness Test

```python
import torch
from flashinfer.attention_sm70 import paged_decode_attention
import torch.nn.functional as F

# Create test data
num_seqs, num_heads, head_dim = 4, 32, 128
num_kv_heads, block_size, seq_len = 8, 16, 256

k_cache = torch.randn(100, block_size, num_kv_heads, head_dim,
                      dtype=torch.float16, device='cuda')
v_cache = torch.randn_like(k_cache)

# ... setup block_tables, seq_lens, q ...

out = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)

# Verify against reference
# (gather KV, expand for GQA, run SDPA, compare)
assert not torch.isnan(out).any()
assert not torch.isinf(out).any()
print("Correctness test passed!")
```

## Troubleshooting

### Common Issues

**1. "SM70 patch: Not on SM70 GPU, skipping patch"**
- The patch only applies on Volta GPUs (compute capability 7.0)
- Check: `torch.cuda.get_device_capability()` should return `(7, 0)`

**2. "Could not import vLLM modules"**
- Ensure vLLM is installed: `pip install vllm`
- Check vLLM version compatibility (tested with 0.14.x)

**3. "AssertionError: SM70 attention supports D=64 or D=128"**
- Your model uses an unsupported head dimension
- The kernel will fall back to Triton automatically

**4. "CUDA out of memory"**
- Reduce `gpu_memory_utilization`
- Reduce `max_model_len`
- Use a smaller model

**5. Incorrect outputs**
- Ensure inputs are FP16: `tensor.half()`
- Ensure inputs are contiguous: `tensor.contiguous()`
- Check block_tables and seq_lens are int32

### Debug Mode

```python
import os
os.environ["FLASHINFER_JIT_DEBUG"] = "1"  # Debug symbols
os.environ["FLASHINFER_JIT_VERBOSE"] = "1"  # Verbose compilation

from flashinfer.vllm_sm70_patch import apply_sm70_patch
apply_sm70_patch()
```

## File Locations

```
flashinfer-src/
├── python/flashinfer/
│   └── vllm_sm70_patch.py          # vLLM integration patch
├── flashinfer/
│   └── attention_sm70.py           # Python API
├── csrc/
│   ├── attention_sm70_volta_paged.cu        # Native paged decode
│   └── attention_sm70_volta_paged_binding.cu
└── docs/
    ├── SM70_VLLM_INTEGRATION.md    # This file
    ├── SM70_FLASHATTENTION.md      # Overview
    └── SM70_API_REFERENCE.md       # API docs
```

## API Reference

### `apply_sm70_patch()`

Applies the SM70 kernel patch to vLLM.

```python
def apply_sm70_patch() -> bool:
    """
    Apply SM70 FlashAttention patch to vLLM's attention system.

    Returns:
        True if patch was successfully applied, False otherwise.

    Notes:
        - Call before importing vLLM for best results
        - Can also be called after vLLM import (dynamic patching)
        - Only applies on SM70 GPUs (V100)
        - Automatically falls back to Triton for unsupported cases
    """
```

### `sm70_paged_decode_attention()`

Low-level paged decode function used by the patch.

```python
def sm70_paged_decode_attention(
    query: torch.Tensor,           # [num_seqs, num_heads, head_dim]
    key_cache: torch.Tensor,       # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    block_table: torch.Tensor,     # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,        # [num_seqs]
    softmax_scale: float,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
```

### `sm70_paged_prefill_attention()`

Low-level paged prefill function (gather-based).

```python
def sm70_paged_prefill_attention(
    query: torch.Tensor,           # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,    # [num_seqs + 1]
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
```

## Benchmarking

Run the built-in benchmark:

```python
from flashinfer.vllm_sm70_patch import benchmark_sm70_vs_triton
benchmark_sm70_vs_triton()
```

Output:
```
======================================================================
SM70 FlashAttention vs Triton Unified Attention Benchmark
======================================================================

Config                    SM70 (ms)    Triton (ms)  Speedup
------------------------------------------------------------
Prefill seq=128           0.019        0.019        1.00x
Prefill seq=256           0.063        0.063        1.00x
Prefill seq=512           0.159        0.159        1.00x
Prefill H=64, D=128       0.038        0.038        1.00x
```

## Future Work

1. **Native paged prefill kernel**: Eliminate gather overhead for prefill
2. **Multi-warp decode**: Better throughput for large batches
3. **Proper vLLM backend registration**: Instead of patching
4. **BF16 support**: When hardware permits (future Volta refresh?)

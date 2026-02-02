# SM70 (Volta/V100) FlashAttention for FlashInfer

## Overview

This document describes the SM70-optimized FlashAttention implementation for NVIDIA Volta architecture GPUs (V100). This implementation provides native tensor core acceleration using the WMMA API, delivering significant performance improvements over fallback implementations.

## Key Features

- **Native WMMA Tensor Cores**: Uses `nvcuda::wmma` API for 16x16x16 matrix operations
- **Paged KV Cache Support**: Native kernel for vLLM's paged attention format
- **GQA Support**: Grouped-query attention with arbitrary head ratios
- **Head Dimensions**: Supports D=64 and D=128
- **Online Softmax**: Memory-efficient attention with O(1) extra memory per row
- **vLLM Integration**: Drop-in patch for vLLM's attention system

## Performance

### Decode Attention (M=1) - Primary Use Case

| Configuration | SM70 Kernel | PyTorch SDPA | Speedup |
|--------------|-------------|--------------|---------|
| B=1, H=32, N=512, D=64 | 0.029ms | 0.055ms | **1.9x** |
| B=1, H=32, N=1024, D=64 | 0.059ms | 0.111ms | **1.9x** |
| B=1, H=32, N=2048, D=64 | 0.118ms | 0.217ms | **1.8x** |
| B=8, H=32, N=512, D=64 | 0.125ms | 0.138ms | **1.1x** |
| B=1, H=64, N=512, D=128 | 0.052ms | 0.098ms | **1.9x** |

### Prefill Attention (M=N)

| Configuration | SM70 Kernel | PyTorch SDPA | Ratio |
|--------------|-------------|--------------|-------|
| M=N=64, D=64 | 0.015ms | 0.015ms | 1.0x |
| M=N=128, D=64 | 0.019ms | 0.019ms | 1.0x |
| M=N=256, D=64 | 0.063ms | 0.036ms | 0.6x |

**Note**: For prefill with M>64, SDPA may be faster. The SM70 kernel excels at decode (M=1) which is the dominant workload in LLM inference.

## Architecture

### Kernel Variants

| Kernel | File | Use Case | Tile Size | Warps |
|--------|------|----------|-----------|-------|
| v1 (Recommended) | `attention_sm70_volta.cu` | Decode + small prefill | 64×128 | 16 |
| v3 (Prefill) | `attention_sm70_volta_prefill.cu` | Large prefill | 64×64 | 8 |
| Paged Decode | `attention_sm70_volta_paged.cu` | vLLM decode | Per-block | 1 |

### Memory Layout

**Dense Attention Input**: `[batch, num_heads, seq_len, head_dim]`

**Paged KV Cache**: `[num_blocks, block_size, num_kv_heads, head_dim]`
- Compatible with vLLM's default KV cache format
- Supports block sizes: 16, 32 (typical vLLM values)

### Algorithm

The implementation uses the online softmax algorithm from FlashAttention:

```
For each KV block:
    1. Load Q tile to shared memory
    2. Load K, V tiles to shared memory
    3. Compute S = Q @ K^T using WMMA
    4. Update running max: new_max = max(old_max, block_max)
    5. Rescale previous accumulator: O *= exp(old_max - new_max)
    6. Compute attention weights: P = exp(S - new_max)
    7. Accumulate: O += P @ V
    8. Update running sum
Final: O = O / sum
```

## File Structure

```
flashinfer-src/
├── csrc/
│   ├── attention_sm70_volta.cu              # Dense decode kernel (v1)
│   ├── attention_sm70_volta_binding.cu      # TVM-FFI bindings for v1
│   ├── attention_sm70_volta_prefill.cu      # Dense prefill kernel (v3)
│   ├── attention_sm70_volta_prefill_binding.cu
│   ├── attention_sm70_volta_paged.cu        # Native paged decode kernel
│   └── attention_sm70_volta_paged_binding.cu
├── flashinfer/
│   ├── attention_sm70.py                    # Python API
│   └── jit/
│       └── attention_sm70.py                # JIT compilation module
├── python/flashinfer/
│   └── vllm_sm70_patch.py                   # vLLM integration patch
└── docs/
    ├── SM70_FLASHATTENTION.md               # This file
    ├── SM70_VLLM_INTEGRATION.md             # vLLM integration guide
    └── SM70_API_REFERENCE.md                # API documentation
```

## Technical Details

### Why SM70 Needs Special Handling

Volta (SM70) lacks the `ldmatrix` PTX instruction available on Ampere+, which FlashInfer's standard kernels rely on. Our implementation:

1. Uses explicit shared memory loads instead of `ldmatrix`
2. Implements WMMA-based matrix multiply for tensor core utilization
3. Uses warp-level shuffle operations for reductions
4. Carefully manages shared memory to stay within V100's 96KB limit

### Shared Memory Usage

| Kernel | Shared Memory | V100 Limit |
|--------|---------------|------------|
| v1 (D=64) | ~48KB | 96KB ✓ |
| v1 (D=128) | ~72KB | 96KB ✓ |
| v3 (D=64) | ~52KB | 96KB ✓ |
| v3 (D=128) | ~86KB | 96KB ✓ |
| Paged (D=64) | ~12KB | 96KB ✓ |
| Paged (D=128) | ~20KB | 96KB ✓ |

### Thread Configuration

- **v1 kernel**: 16 warps (512 threads), processes 64 query rows × 128 KV tokens per tile
- **v3 kernel**: 8 warps (256 threads), processes 64 query rows × 64 KV tokens per tile
- **Paged kernel**: 1 warp (32 threads), processes 1 query × up to 64 KV tokens per block

## Limitations

1. **FP16 only**: BF16 and FP8 not supported (V100 hardware limitation)
2. **Head dimensions**: Only D=64 and D=128 supported
3. **No sliding window**: Sliding window attention not implemented
4. **No ALiBi**: Attention with Linear Biases not supported
5. **No logit softcapping**: Gemma-style softcapping not supported

## Comparison with Other Approaches

| Approach | Decode Speed | Paged KV | Tensor Cores | Notes |
|----------|-------------|----------|--------------|-------|
| **SM70 FlashAttention** | Fast | ✓ | ✓ | This implementation |
| PyTorch SDPA | Baseline | ✗ | ✗ | No paged support |
| Triton unified_attention | Moderate | ✓ | ✗ | vLLM fallback |
| FlashInfer (standard) | N/A | ✓ | ✓ | Crashes on SM70 |

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [NVIDIA WMMA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [vLLM Documentation](https://docs.vllm.ai/)

## Authors

- FlashInfer Team
- Co-authored with Claude Opus 4.5

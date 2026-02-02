# V100 SM70 FlashAttention Benchmark Results

## Environment
- GPU: Tesla V100 32GB SXM2 (SM70)
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- Baseline: PyTorch SDPA with `mem_efficient` backend (CUTLASS-based)

## Decode Attention (M=1, Non-Causal) ✅ WORKING

This is the primary use case for LLM inference. All results verified correct.

| Config | SDPA (ms) | SM70 WMMA (ms) | Speedup | Status |
|--------|-----------|----------------|---------|--------|
| H=32, N=512, D=64 | 0.076 | 0.056 | **1.36x** | ✅ PASS |
| H=32, N=1024, D=64 | 0.133 | 0.094 | **1.41x** | ✅ PASS |
| H=32, N=2048, D=64 | 0.238 | 0.139 | **1.71x** | ✅ PASS |
| H=32, N=4096, D=64 | 0.447 | 0.245 | **1.83x** | ✅ PASS |
| H=32, N=512, D=128 | 0.068 | 0.068 | 0.99x | ✅ PASS |
| H=32, N=1024, D=128 | 0.118 | 0.104 | **1.13x** | ✅ PASS |
| H=32, N=2048, D=128 | 0.189 | 0.184 | 1.03x | ✅ PASS |

**Summary**: SM70 kernel is **1.4x-1.8x faster** for decode with D=64.

## Prefill Attention (M=N, Causal) ⚠️ KNOWN ISSUES

| Issue | Description | Impact |
|-------|-------------|--------|
| Multi-block bugs | Incorrect results when M > 64 | Numerical errors |
| Causal mask bugs | Fragment position decoding errors | Wrong attention pattern |
| Performance | 0.4x-0.6x vs PyTorch SDPA | Slower than baseline |

**Recommendation**: Use PyTorch SDPA for prefill.

### Prefill Performance (for reference)

| Config | SDPA (ms) | SM70 (ms) | Relative |
|--------|-----------|-----------|----------|
| M=N=256, D=64 | 0.057 | 0.087 | 0.65x |
| M=N=512, D=64 | 0.108 | 0.193 | 0.56x |
| M=N=1024, D=64 | 0.282 | 0.532 | 0.53x |

## Analysis

### Why Decode Works Well
1. **Single query (M=1)** fits in one WMMA tile - no multi-block complexity
2. **Non-causal** - no causal mask to decode
3. **Memory-bound** - benefits from vectorized loads

### Why Prefill Has Issues
1. **Fragment position decoding** for causal mask is complex on Volta
2. **Multi-block coordination** for online softmax rescaling
3. **PyTorch uses production CUTLASS kernel** (`fmha_cutlassF_f16_aligned_64x64_rf_sm70`)

## Recommended Usage

```python
import torch
import flashinfer.attention_sm70 as sm70

# DECODE attention (recommended - 1.4x-1.8x faster)
# Works for: single token generation, autoregressive inference
output = sm70.single_decode_attention(q, k_cache, v_cache)

# PREFILL attention (use PyTorch - more reliable)
# Works for: initial prompt processing
output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True
)
```

## Technical Details

### Kernel Configuration
| Parameter | V1 Kernel |
|-----------|-----------|
| Block M | 64 |
| Block N | 128 |
| Warps | 16 (512 threads) |
| WMMA Tiles | 16×16×16 |
| Shared Memory | ~79 KB |

### PyTorch Backend on V100
```
fmha_cutlassF_f16_aligned_64x64_rf_sm70
```
A CUTLASS-based memory-efficient attention kernel optimized for SM70.

## Future Work to Fix Prefill

1. **Fix causal mask fragment decoding**
   - Volta WMMA has different fragment layout than Ampere
   - Need to verify row/col position mapping

2. **Fix multi-block online softmax**
   - Rescaling logic may have race conditions
   - Consider using atomics or separate reduction pass

3. **Performance optimizations**
   - Smaller tiles (64×64) to match CUTLASS
   - Software pipelining for load/compute overlap
   - Consider CuTe DSL for more flexible layout control

## Key Takeaway

**For LLM inference decode (the dominant phase), the SM70 kernel provides 1.4x-1.8x speedup.** This is valuable since decode generates most tokens during inference.

For prefill, fall back to PyTorch SDPA until the causal mask and multi-block issues are fixed.

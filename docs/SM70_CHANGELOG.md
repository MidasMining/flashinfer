# SM70 FlashAttention Changelog

## [Unreleased] - 2026-01-31

### Added

#### Native Paged Decode Kernel
- New file: `csrc/attention_sm70_volta_paged.cu`
- Direct paged KV cache access without gather overhead
- Single-warp design for simple reduction
- GQA (grouped-query attention) support
- Supports block sizes: any (16, 32 typical for vLLM)
- Supports head dimensions: 64, 128

#### vLLM Integration Patch
- New file: `python/flashinfer/vllm_sm70_patch.py`
- `apply_sm70_patch()`: Patches vLLM's Triton unified_attention
- Automatically uses native paged decode for M=1 (decode)
- Falls back to gather + dense kernel for prefill
- Falls back to Triton for unsupported features

#### Python API Extensions
- `paged_decode_attention()`: New function in `flashinfer/attention_sm70.py`
- Wrapper for native paged decode kernel
- Compatible with vLLM's KV cache format

#### Documentation
- `docs/SM70_FLASHATTENTION.md`: Architecture and design overview
- `docs/SM70_API_REFERENCE.md`: Complete API documentation
- `docs/SM70_VLLM_INTEGRATION.md`: vLLM integration guide
- `docs/SM70_README.md`: Quick start guide
- `docs/SM70_CHANGELOG.md`: This file

### Fixed

#### Q Loading Bug in Paged Decode
- Fixed incomplete Q loading when `THREADS < D`
- Previously only loaded first 32 elements for D=64/128
- Now uses strided loop to load all elements

#### Race Condition in Paged Decode
- Added `__syncthreads()` after exp computation
- Ensures all exp values visible before weighted sum

### Changed

#### Prefill Kernel (v3)
- Reduced BLOCK_M from 128 to 64
- Previous config exceeded V100's 96KB shared memory limit
- New config uses ~86KB (within limit)

### Technical Details

#### Paged Decode Kernel Design
```
Configuration:
- BLOCK_N: 64 (KV tokens per iteration)
- WARPS_PER_BLOCK: 1 (32 threads)
- Shared memory: ~12KB (D=64), ~20KB (D=128)

Algorithm:
1. Load Q to shared memory (strided for D > THREADS)
2. For each KV cache block:
   a. Load K, V from paged cache to shared memory
   b. Compute S = Q @ K^T (dot product per KV token)
   c. Online softmax: update max, rescale, compute exp
   d. Accumulate O += softmax(S) @ V
3. Normalize: O = O / sum
4. Write output
```

#### Verified Correctness
- Tested against PyTorch SDPA with random inputs
- Max difference: < 0.001 (FP16 precision)
- Tested configurations:
  - D=64, D=128
  - GQA ratios: 1, 4, 8
  - Sequence lengths: 32, 64, 256, 512, 1024
  - Block sizes: 16, 32

### Known Limitations

1. FP16 only (V100 hardware limitation)
2. No BF16 support
3. No FP8 KV cache support
4. No sliding window attention
5. No ALiBi slopes
6. No logit softcapping (Gemma-style)

### Performance Notes

| Operation | vs Baseline | Notes |
|-----------|-------------|-------|
| Paged decode | ~2x vs Triton | Native kernel |
| Dense decode | ~2x vs SDPA | WMMA tensor cores |
| Prefill (small) | ~1x vs SDPA | Competitive |
| Prefill (large) | ~0.5x vs SDPA | SDPA faster |

### Migration Guide

#### From Triton Fallback
No code changes needed. Apply patch:
```python
from flashinfer.vllm_sm70_patch import apply_sm70_patch
apply_sm70_patch()
```

#### From Custom Implementation
Replace:
```python
# Old: manual gather + attention
k_gathered = gather_kv(k_cache, block_tables)
v_gathered = gather_kv(v_cache, block_tables)
out = attention(q, k_gathered, v_gathered)
```

With:
```python
# New: native paged attention
from flashinfer.attention_sm70 import paged_decode_attention
out = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
```

### Testing

Run tests:
```bash
# Correctness test
python /tmp/v100-kernels/test_paged_decode.py

# vLLM integration test
python /tmp/v100-kernels/test_vllm_integration.py

# Full verification
python -c "
from flashinfer.attention_sm70 import paged_decode_attention
import torch
# ... test code ...
"
```

### Dependencies

- CUDA 11.x or 12.x
- PyTorch 2.0+
- vLLM 0.14.x (for integration)
- TVM-FFI (bundled with FlashInfer)

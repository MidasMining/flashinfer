# SM70 FlashAttention Installation Guide

## Requirements

### Hardware
- NVIDIA V100 GPU (Volta architecture, SM70)
- 16GB or 32GB VRAM recommended

### Software
- CUDA 11.x or 12.x
- Python 3.8+
- PyTorch 2.0+ with CUDA support

## Installation Methods

### Method 1: Development Install (Recommended)

Clone and install FlashInfer in development mode:

```bash
# Clone repository
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer

# Install in development mode
pip install --no-build-isolation -e . -v

# Verify installation
python -c "from flashinfer.attention_sm70 import flash_attention_forward; print('OK')"
```

### Method 2: From Source Directory

If you have the source but don't want to install:

```bash
# Add to Python path
export PYTHONPATH="/path/to/flashinfer-src/python:$PYTHONPATH"
export PYTHONPATH="/path/to/flashinfer-src:$PYTHONPATH"

# Verify
python -c "from flashinfer.attention_sm70 import flash_attention_forward; print('OK')"
```

### Method 3: Standalone Module

Copy the necessary files to your project:

```bash
# Required files
flashinfer/
├── attention_sm70.py
├── utils.py
└── jit/
    ├── __init__.py
    ├── core.py
    ├── env.py
    └── attention_sm70.py

csrc/
├── attention_sm70_volta.cu
├── attention_sm70_volta_binding.cu
├── attention_sm70_volta_paged.cu
├── attention_sm70_volta_paged_binding.cu
└── tvm_ffi_utils.h
```

## First Run: JIT Compilation

On first use, kernels are JIT-compiled:

```python
from flashinfer.attention_sm70 import flash_attention_forward
import torch

# First call triggers compilation (may take 30-60 seconds)
q = torch.randn(1, 32, 1, 128, dtype=torch.float16, device='cuda')
k = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device='cuda')
v = torch.randn_like(k)

output = flash_attention_forward(q, k, v)  # Compiles here
```

### Compilation Output

You should see:
```
[FlashInfer] Compiling attention_sm70_volta...
[FlashInfer] Compilation successful
```

### Cache Location

Compiled kernels are cached at:
```
~/.cache/flashinfer/<version>/<compute_capability>/cached_ops/
```

### Clear Cache

If you modify kernel source:
```bash
rm -rf ~/.cache/flashinfer/
```

## Verification

### Basic Functionality

```python
import torch
from flashinfer.attention_sm70 import flash_attention_forward

# Create test tensors
q = torch.randn(1, 32, 1, 128, dtype=torch.float16, device='cuda')
k = torch.randn(1, 32, 512, 128, dtype=torch.float16, device='cuda')
v = torch.randn_like(k)

# Run attention
output = flash_attention_forward(q, k, v)

# Verify
assert output.shape == q.shape
assert not torch.isnan(output).any()
print("Basic test passed!")
```

### Correctness Check

```python
import torch
import torch.nn.functional as F
from flashinfer.attention_sm70 import flash_attention_forward

q = torch.randn(1, 8, 1, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 8, 128, 64, dtype=torch.float16, device='cuda')
v = torch.randn_like(k)

# Our kernel
out_sm70 = flash_attention_forward(q, k, v)

# Reference
out_ref = F.scaled_dot_product_attention(q, k, v)

# Compare
diff = (out_sm70 - out_ref).abs().max().item()
print(f"Max difference: {diff:.6f}")
assert diff < 0.001, "Correctness check failed!"
print("Correctness test passed!")
```

### Paged Attention

```python
import torch
from flashinfer.attention_sm70 import paged_decode_attention

num_seqs, num_heads, head_dim = 4, 32, 128
num_kv_heads, block_size = 8, 16

k_cache = torch.randn(100, block_size, num_kv_heads, head_dim,
                      dtype=torch.float16, device='cuda')
v_cache = torch.randn_like(k_cache)

block_tables = torch.zeros(num_seqs, 10, dtype=torch.int32, device='cuda')
for i in range(num_seqs):
    for j in range(4):
        block_tables[i, j] = i * 4 + j

seq_lens = torch.tensor([64, 48, 32, 16], dtype=torch.int32, device='cuda')
q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.float16, device='cuda')

output = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)

assert output.shape == (num_seqs, num_heads, head_dim)
assert not torch.isnan(output).any()
print("Paged attention test passed!")
```

## vLLM Integration

### Install vLLM

```bash
pip install vllm
```

### Apply Patch

```python
import sys
sys.path.insert(0, "/path/to/flashinfer-src/python")

from flashinfer.vllm_sm70_patch import apply_sm70_patch
result = apply_sm70_patch()
print(f"Patch applied: {result}")
# Expected: "SM70 patch: Successfully patched triton_unified_attention"
```

### Full Test

```python
import sys
sys.path.insert(0, "/path/to/flashinfer-src/python")

from flashinfer.vllm_sm70_patch import apply_sm70_patch
apply_sm70_patch()

from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", dtype="float16")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=20))
print(outputs[0].outputs[0].text)
```

## Troubleshooting

### CUDA Version Mismatch

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**: Ensure CUDA toolkit version matches PyTorch's CUDA version:
```bash
nvcc --version  # Check CUDA toolkit
python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA
```

### Compilation Errors

```
error: identifier "__ldmatrix" is undefined
```

**Solution**: This error should NOT occur with SM70 kernels. If it does, ensure you're using the correct kernel files (not standard FlashInfer kernels).

### Out of Memory

```
CUDA out of memory
```

**Solution**:
1. Reduce batch size
2. Reduce sequence length
3. Use smaller model
4. Lower `gpu_memory_utilization` in vLLM

### Wrong Device

```
AssertionError: SM70 attention kernel is optimized for V100 (SM70)
```

**Solution**: The kernel is designed for V100. On other GPUs, use standard FlashInfer or PyTorch SDPA.

### Import Errors

```
ModuleNotFoundError: No module named 'flashinfer'
```

**Solution**:
```bash
# Option 1: Install
pip install --no-build-isolation -e /path/to/flashinfer -v

# Option 2: Add to path
export PYTHONPATH="/path/to/flashinfer-src:$PYTHONPATH"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASHINFER_JIT_VERBOSE` | Verbose compilation output | 0 |
| `FLASHINFER_JIT_DEBUG` | Debug symbols in compiled kernels | 0 |
| `FLASHINFER_LOGLEVEL` | API logging level (0-5) | 0 |
| `FLASHINFER_WORKSPACE_BASE` | Custom cache directory | `~/.cache/flashinfer` |

## Support

- GitHub Issues: https://github.com/flashinfer-ai/flashinfer/issues
- Documentation: See `docs/SM70_*.md` files

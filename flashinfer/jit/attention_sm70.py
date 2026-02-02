"""
JIT module generator for SM70 (Volta/V100) FlashAttention.

Three kernel versions are available:
- v1: Original WMMA kernel (BLOCK_M=64, BLOCK_N=128, 16 warps) - good for decode
- v2: Optimized kernel (BLOCK_M=64, BLOCK_N=64, 8 warps) - balanced
- v3 (prefill): Prefill-optimized (BLOCK_M=128, BLOCK_N=64, 8 warps) - best for prefill
"""

import functools
import shutil
from pathlib import Path

from . import env as jit_env
from .core import gen_jit_spec


def get_sm70_attention_uri(version: int = 1):
    """Generate URI for SM70 attention module."""
    if version == 3:
        return "attention_sm70_volta_prefill"
    if version == 2:
        return "attention_sm70_volta_v2"
    return "attention_sm70_volta"


@functools.cache
def gen_sm70_attention_module(version: int = 1):
    """Generate and load the SM70 FlashAttention module.

    Args:
        version: 1 for original WMMA kernel, 2 for optimized, 3 for prefill-optimized
    """
    uri = get_sm70_attention_uri(version)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)

    # Source files based on version
    if version == 3:
        source_files = [
            "attention_sm70_volta_prefill.cu",
            "attention_sm70_volta_prefill_binding.cu"
        ]
    elif version == 2:
        source_files = [
            "attention_sm70_volta_v2.cu",
            "attention_sm70_volta_v2_binding.cu"
        ]
    else:
        source_files = [
            "attention_sm70_volta.cu",
            "attention_sm70_volta_binding.cu"
        ]

    sources = []
    for fname in source_files:
        src = jit_env.FLASHINFER_CSRC_DIR / fname
        dst = gen_directory / fname
        shutil.copy(src, dst)
        sources.append(dst)

    # Compile with SM70 target
    spec = gen_jit_spec(
        uri,
        sources,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DFLASHINFER_SM70_ATTENTION",
        ],
    )

    return spec.build_and_load()


def get_sm70_attention_func(version: int = 1):
    """Get the SM70 FlashAttention forward function.

    Args:
        version: 1 for original, 2 for optimized, 3 for prefill-optimized
    """
    module = gen_sm70_attention_module(version)
    if version == 3:
        return module["FlashAttentionSM70PrefillForward"]
    if version == 2:
        return module["flash_attention_sm70_v2_forward"]
    return module["flash_attention_sm70_forward"]


# Paged attention module
def get_sm70_paged_uri():
    """Generate URI for SM70 paged attention module."""
    return "attention_sm70_volta_paged"


@functools.cache
def gen_sm70_paged_module():
    """Generate and load the SM70 paged attention module."""
    uri = get_sm70_paged_uri()
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)

    source_files = [
        "attention_sm70_volta_paged.cu",
        "attention_sm70_volta_paged_binding.cu"
    ]

    sources = []
    for fname in source_files:
        src = jit_env.FLASHINFER_CSRC_DIR / fname
        dst = gen_directory / fname
        shutil.copy(src, dst)
        sources.append(dst)

    spec = gen_jit_spec(
        uri,
        sources,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DFLASHINFER_SM70_ATTENTION",
        ],
    )

    return spec.build_and_load()


def get_sm70_paged_decode_func():
    """Get the SM70 paged decode attention function."""
    module = gen_sm70_paged_module()
    return module["FlashAttentionSM70PagedDecode"]

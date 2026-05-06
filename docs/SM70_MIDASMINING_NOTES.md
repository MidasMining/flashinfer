# SM70 — MidasMining production deployment notes

Findings from extending this fork for the MidasMining vLLM-tq deployment (Qwen3.6-35B-A3B-AWQ on a single Tesla V100 32GB).

## Our commits in this branch

| Commit | What |
|---|---|
| `f767aac` | feat(sm70): add head_dim=256 support to paged decode attention |
| `d21693b` | fix(sm70): chunk inner loop by BLOCK_N within each paged block |

## Why D=256 was needed

Modern Qwen-family models (Qwen3.6, Qwen3-Coder, Qwen3-Next, etc.) use head_dim=256, double the 128 the original `attention_sm70_volta_paged.cu` supported. We added the template instantiation, the SMEM opt-in (`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` — D=256 needs ~68 KB which exceeds V100's default 48 KB cap but fits the 96 KB hardware ceiling), and explicit gencode coverage.

## The chunk-loop bug fix

While wiring the fork into vLLM-tq we hit `cudaErrorIllegalAddress` on real inputs that the synthetic smoke tests didn't catch. Root cause:

The decode kernel's load loop assumed each paged KV block fits in one `BLOCK_N=64` chunk. For `block_size <= 64` that's fine. **But vLLM v1 uses `block_size = 1056` for its paged KV cache.** With `seq_len > 64`, `tokens_in_block` exceeds `BLOCK_N`, and the load loop wrote past `smem.k[BLOCK_N * KV_STRIDE]` and `smem.s[BLOCK_N]` bounds.

The fix adds an inner chunk loop that processes BLOCK_N tokens at a time within each paged block. Each chunk runs a full online-softmax phase (load → score → softmax-update → V-accumulate) and updates the running `(row_max, row_sum, smem.o[])` state across chunks.

**This is a real bug independent of head_dim.** Anyone using flashinfer SM70 with vLLM v1's default paged cache (block_size=1056) hits it as soon as their context grows past 64 tokens. Worth upstreaming as the chunk fix isn't D=256-specific.

Validated with stress tests at seq_len ∈ {9, 64, 128, 256, 512, 1056, 2000, 4000} on vLLM-shaped inputs (B=1, num_kv_heads=2, head_dim=256, GQA 8:1) — all pass with finite output.

## Production benchmark — kernel didn't compete at our shape

Despite the correct, working D=256 + chunk-fix variant, **this fork's decode kernel is ~5× slower than vLLM's stock TRITON_ATTN at production shapes**, and ~5× slower than 1Cat-vLLM's `flash_attn_v100` Volta backend. Measured on Qwen3.6-35B-A3B-AWQ at 14K context (CUDA graphs, fp16 KV — no TQ to isolate the kernel):

| Backend | Decode @ 14K |
|---|---:|
| flashinfer SM70 fork (with fixes) | ~10 t/s |
| vLLM TRITON_ATTN | ~47 t/s |
| 1Cat `flash_attn_v100` (multi-warp + Flash-Decoding split-K) | ~54 t/s |

Why this kernel underperforms at our shape:

1. **Single-warp-per-CTA design** (`WARPS_PER_BLOCK=1`, 32 threads). At decode shape (1 batch × 16 Q heads → 16 CTAs across 80 SMs), effective utilization is ~5%.
2. **Scalar fp16 inner loop** — `__half2float` per element with no half2 vectorization or tensor cores. The dot product becomes the bottleneck.
3. **Many sequential chunks at long context** — `block_size=1056 / BLOCK_N=64 → 17` chunks per paged block, each running a full online-softmax with multiple `__syncthreads`.
4. **`.contiguous()` copy per call** from vLLM's strided KV view — ~1.7 ms/token of pure memcpy.

The "3.4-5× over SDPA" claim in `docs/SM70_BENCHMARK_RESULTS.md` was for D=64/128 + small/short contexts where one-warp-per-CTA scalar hides under SDPA launch overhead. At D=256 + paged decode + 14K context, it doesn't translate.

## When this fork is still the right choice

- **Small head_dim + short context** — the original 1.4-1.8× speedup for D=64 still holds for matching workloads.
- **Reference implementation for a multi-warp port** — the fork is a clean, single-warp version. Adding multi-warp parallelism + half2 vectorization + Flash-Decoding split-K (≈ 1Cat's approach) would likely close the gap.
- **The chunk-loop fix is independent of architecture choice** — it's a real bug fix that shouldn't be lost regardless of what happens with the rest of the kernel.

## Cross-references

- Full investigation log: [MidasMining/inference-research](https://192.168.1.45:3000/MidasMining/inference-research) (Gitea private), specifically `volta-sm70/INVESTIGATION_LOG.md` Round 4 for the detailed analysis of why this didn't beat vLLM Triton.
- vLLM integration source: [MidasMining/vllm-tq](https://github.com/MidasMining/vllm-tq) `test-vibha-wht` branch. Phase 1 commit `c800aa0a3` is where the fork was wired in then unwired in favor of 1Cat-vLLM's `flash_attn_v100`.
- Alternative Volta paged attention: [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) — the vLLM fork bundled `flash_attn_v100` (multi-warp + Flash-Decoding) which we eventually adopted instead.

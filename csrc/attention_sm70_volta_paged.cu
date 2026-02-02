/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * SM70 (Volta/V100) Paged Attention Kernel
 *
 * This kernel handles paged KV cache format used by vLLM.
 * Optimized for decode attention (M=1) where we achieve ~2x speedup over SDPA.
 *
 * KV Cache Format: [num_blocks, block_size, num_kv_heads, head_dim]
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

namespace flashinfer {
namespace sm70 {
namespace paged {

// ============================================================================
// CONFIGURATION
// ============================================================================
template<int D>
struct PagedConfig {
    // For decode (M=1), use single warp for simpler reduction
    static constexpr int BLOCK_N = 64;   // KV tokens per iteration
    static constexpr int WARPS_PER_BLOCK = 1;  // Single warp for easy reduction
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    // Shared memory layout
    static constexpr int PAD = 8;
    static constexpr int KV_STRIDE = D + PAD;
    static constexpr int S_STRIDE = BLOCK_N + PAD;

    struct SmemLayout {
        __half q[D + PAD];                    // Single query
        __half k[BLOCK_N * KV_STRIDE];        // Key block
        __half v[BLOCK_N * KV_STRIDE];        // Value block
        float s[BLOCK_N];                     // Attention scores
        float o[D + PAD];                     // Output accumulator
        float row_max;                        // Running max
        float row_sum;                        // Running sum
    };

    static constexpr size_t SMEM_SIZE = sizeof(SmemLayout);
};

// ============================================================================
// DECODE KERNEL (M=1, most important case)
// ============================================================================
template<int D>
__global__ void __launch_bounds__(PagedConfig<D>::THREADS_PER_BLOCK)
flash_attention_sm70_paged_decode_kernel(
    const __half* __restrict__ Q,             // [num_seqs, num_heads, D]
    const __half* __restrict__ K_cache,       // [num_blocks, block_size, num_kv_heads, D]
    const __half* __restrict__ V_cache,       // [num_blocks, block_size, num_kv_heads, D]
    const int* __restrict__ block_tables,     // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,         // [num_seqs]
          __half* __restrict__ Out,           // [num_seqs, num_heads, D]
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int block_size,
    const float softmax_scale
) {
    using Config = PagedConfig<D>;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int THREADS = Config::THREADS_PER_BLOCK;
    constexpr int KV_STRIDE = Config::KV_STRIDE;
    constexpr float NEG_INF = -1e30f;

    // Grid: [num_seqs, num_heads, 1]
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    if (seq_idx >= num_seqs) return;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    // GQA: map query head to KV head
    const int gqa_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / gqa_ratio;

    const int tid = threadIdx.x;

    // Shared memory
    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    // Load query (single row) - use strided access since THREADS may be < D
    const __half* q_ptr = Q + (size_t)seq_idx * num_heads * D + head_idx * D;
    for (int i = tid; i < D; i += THREADS) {
        smem.q[i] = q_ptr[i];
        smem.o[i] = 0.0f;
    }
    if (tid == 0) {
        smem.row_max = NEG_INF;
        smem.row_sum = 0.0f;
    }
    __syncthreads();

    // Block table for this sequence
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const int num_kv_blocks = (seq_len + block_size - 1) / block_size;

    // Process KV cache blocks
    int kv_pos = 0;
    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        const int physical_block = seq_block_table[block_idx];
        const int tokens_in_block = min(block_size, seq_len - kv_pos);

        // Pointer to K and V in this block
        // Layout: [num_blocks, block_size, num_kv_heads, D]
        // For block b, token t, head h, dim d:
        //   offset = b * (block_size * num_kv_heads * D) + t * (num_kv_heads * D) + h * D + d
        const __half* k_block = K_cache +
            (size_t)physical_block * block_size * num_kv_heads * D;
        const __half* v_block = V_cache +
            (size_t)physical_block * block_size * num_kv_heads * D;

        // Load K and V for this block
        for (int i = tid; i < tokens_in_block * D; i += THREADS) {
            int token = i / D;
            int d = i % D;
            // Access: token * num_kv_heads * D + kv_head_idx * D + d
            size_t kv_offset = (size_t)token * num_kv_heads * D + kv_head_idx * D + d;
            smem.k[token * KV_STRIDE + d] = k_block[kv_offset];
            smem.v[token * KV_STRIDE + d] = v_block[kv_offset];
        }
        __syncthreads();

        // Compute attention scores: S = Q @ K^T
        // Using simple dot product (WMMA overkill for M=1)
        for (int k_idx = tid; k_idx < tokens_in_block; k_idx += THREADS) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < D; ++d) {
                sum += __half2float(smem.q[d]) * __half2float(smem.k[k_idx * KV_STRIDE + d]);
            }
            smem.s[k_idx] = sum * softmax_scale;
        }
        __syncthreads();

        // Online softmax: find max (single warp, so warp shuffle works)
        float local_max = NEG_INF;
        for (int k_idx = tid; k_idx < tokens_in_block; k_idx += THREADS) {
            local_max = fmaxf(local_max, smem.s[k_idx]);
        }

        // Reduce max across warp (32 threads)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        float block_max = __shfl_sync(0xffffffff, local_max, 0);

        // Rescale previous accumulator
        float old_max = smem.row_max;
        float new_max = fmaxf(old_max, block_max);
        float exp_diff = __expf(old_max - new_max);

        // Compute exp and sum
        float local_sum = 0.0f;
        for (int k_idx = tid; k_idx < tokens_in_block; k_idx += THREADS) {
            float e = __expf(smem.s[k_idx] - new_max);
            smem.s[k_idx] = e;  // Store exp values for weighted sum
            local_sum += e;
        }
        __syncthreads();  // Ensure all exp values are written before reduction

        // Reduce sum across warp (32 threads)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        float block_sum = __shfl_sync(0xffffffff, local_sum, 0);

        // Update running stats
        if (tid == 0) {
            smem.row_sum = exp_diff * smem.row_sum + block_sum;
            smem.row_max = new_max;
        }
        __syncthreads();  // Ensure row_sum/row_max are visible

        // Rescale output accumulator
        for (int d = tid; d < D; d += THREADS) {
            smem.o[d] *= exp_diff;
        }
        __syncthreads();

        // Accumulate weighted values: O += S @ V
        for (int d = tid; d < D; d += THREADS) {
            float acc = 0.0f;
            for (int k_idx = 0; k_idx < tokens_in_block; ++k_idx) {
                acc += smem.s[k_idx] * __half2float(smem.v[k_idx * KV_STRIDE + d]);
            }
            smem.o[d] += acc;
        }
        __syncthreads();

        kv_pos += tokens_in_block;
    }

    // Write output: O = O / sum
    __half* out_ptr = Out + (size_t)seq_idx * num_heads * D + head_idx * D;
    float inv_sum = 1.0f / fmaxf(smem.row_sum, 1e-12f);
    for (int d = tid; d < D; d += THREADS) {
        out_ptr[d] = __float2half_rn(smem.o[d] * inv_sum);
    }
}

// ============================================================================
// LAUNCHER
// ============================================================================
template<int D>
cudaError_t launch_flash_attention_sm70_paged_decode(
    const __half* Q,
    const __half* K_cache,
    const __half* V_cache,
    const int* block_tables,
    const int* seq_lens,
    __half* Out,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int max_blocks_per_seq,
    int block_size,
    float softmax_scale,
    cudaStream_t stream
) {
    using Config = PagedConfig<D>;

    dim3 grid(num_seqs, num_heads, 1);
    dim3 block(Config::THREADS_PER_BLOCK);
    size_t smem = Config::SMEM_SIZE;

    flash_attention_sm70_paged_decode_kernel<D><<<grid, block, smem, stream>>>(
        Q, K_cache, V_cache, block_tables, seq_lens, Out,
        num_seqs, num_heads, num_kv_heads, max_blocks_per_seq,
        block_size, softmax_scale
    );

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t launch_flash_attention_sm70_paged_decode<64>(
    const __half*, const __half*, const __half*,
    const int*, const int*, __half*,
    int, int, int, int, int, float, cudaStream_t);

template cudaError_t launch_flash_attention_sm70_paged_decode<128>(
    const __half*, const __half*, const __half*,
    const int*, const int*, __half*,
    int, int, int, int, int, float, cudaStream_t);

}  // namespace paged
}  // namespace sm70
}  // namespace flashinfer

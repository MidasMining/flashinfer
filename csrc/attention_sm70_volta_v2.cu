/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * SM70 (Volta/V100) FlashAttention - Optimized Version 2
 * Key optimizations over V1:
 * 1. Smaller tiles (64x64) matching CUTLASS
 * 2. Fewer warps (8 instead of 16) - less scheduling overhead
 * 3. Simpler shared memory layout
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

namespace flashinfer {
namespace sm70 {
namespace v2 {

// ============================================================================
// KERNEL CONFIGURATION - OPTIMIZED FOR PREFILL
// ============================================================================
template<int D>
struct KernelConfigV2 {
    // Smaller tiles (64x64) matching CUTLASS kernel
    static constexpr int BLOCK_M = 64;
    static constexpr int BLOCK_N = 64;

    // Fewer warps reduces scheduling overhead
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    // WMMA dimensions
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;

    // Tile counts
    static constexpr int TILES_M = BLOCK_M / WMMA_M;  // 4
    static constexpr int TILES_N = BLOCK_N / WMMA_N;  // 4
    static constexpr int TILES_K = (D + WMMA_K - 1) / WMMA_K;

    // Strides with padding to avoid bank conflicts
    static constexpr int PAD = 8;
    static constexpr int Q_STRIDE = D + PAD;
    static constexpr int KV_STRIDE = D + PAD;
    static constexpr int S_STRIDE = BLOCK_N + PAD;
    static constexpr int O_STRIDE = D + PAD;

    // Shared memory layout
    struct SmemLayout {
        __half q[BLOCK_M * Q_STRIDE];
        __half kv[BLOCK_N * KV_STRIDE];
        float s[BLOCK_M * S_STRIDE];
        __half p[BLOCK_M * S_STRIDE];
        float o[BLOCK_M * O_STRIDE];
        float row_max[BLOCK_M];
        float row_sum[BLOCK_M];
    };

    static constexpr size_t SMEM_SIZE = sizeof(SmemLayout);
};

// ============================================================================
// FORWARD KERNEL
// ============================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(KernelConfigV2<D>::THREADS_PER_BLOCK, 2)
flash_attention_sm70_v2_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ Out,
           float* __restrict__ softmax_lse,
    const int B,
    const int H,
    const int M,
    const int N,
    const float softmax_scale
) {
    using Config = KernelConfigV2<D>;
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int THREADS = Config::THREADS_PER_BLOCK;
    constexpr int WARPS = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE = Config::Q_STRIDE;
    constexpr int KV_STRIDE = Config::KV_STRIDE;
    constexpr int S_STRIDE = Config::S_STRIDE;
    constexpr int O_STRIDE = Config::O_STRIDE;
    constexpr int TILES_M = Config::TILES_M;
    constexpr int TILES_N = Config::TILES_N;
    constexpr int TILES_K = Config::TILES_K;
    constexpr float NEG_INF = -1e30f;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_m = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;

    const int valid_q_rows = min(BLOCK_M, M - start_row);
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    if constexpr (IS_CAUSAL) {
        const int max_key_pos = start_row + valid_q_rows - 1;
        if (max_key_pos < 0) return;
        num_n_tiles = min(num_n_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
    }

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Global pointers
    const __half* q_ptr = Q + (size_t)batch_head_id * M * D + start_row * D;
    const __half* k_ptr = K + (size_t)batch_head_id * N * D;
    const __half* v_ptr = V + (size_t)batch_head_id * N * D;
          __half* out_ptr = Out + (size_t)batch_head_id * M * D + start_row * D;
    float* lse_ptr = softmax_lse ? (softmax_lse + (size_t)batch_head_id * M + start_row) : nullptr;

    // Shared memory
    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* sQ = smem.q;
    __half* sKV = smem.kv;
    float* sS = smem.s;
    __half* sP = smem.p;
    float* sO = smem.o;
    float* sRowMax = smem.row_max;
    float* sRowSum = smem.row_sum;

    // Initialize
    if (tid < BLOCK_M) {
        sRowMax[tid] = NEG_INF;
        sRowSum[tid] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * O_STRIDE; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Load Q
    for (int i = tid; i < valid_q_rows * D; i += THREADS) {
        int row = i / D;
        int col = i % D;
        sQ[row * Q_STRIDE + col] = q_ptr[row * D + col];
    }
    __syncthreads();

    // Main loop over K/V tiles
    for (int block_n = 0; block_n < num_n_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        if constexpr (IS_CAUSAL) {
            if (start_col > start_row + valid_q_rows - 1) continue;
        }

        // Load K
        for (int i = tid; i < valid_k_rows * D; i += THREADS) {
            int row = i / D;
            int col = i % D;
            sKV[row * KV_STRIDE + col] = k_ptr[(start_col + row) * D + col];
        }
        __syncthreads();

        // Compute S = Q @ K^T
        const int total_s_tiles = TILES_M * TILES_N;
        const int tiles_per_warp = (total_s_tiles + WARPS - 1) / WARPS;

        for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
            const int global_tile = warp_id * tiles_per_warp + tile_idx;
            if (global_tile >= total_s_tiles) break;

            const int tm = global_tile / TILES_N;
            const int tn = global_tile % TILES_N;

            if (tm * 16 >= valid_q_rows || tn * 16 >= valid_k_rows) continue;

            fragment<accumulator, 16, 16, 16, float> acc;
            fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int tk = 0; tk < TILES_K; ++tk) {
                fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
                fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;

                load_matrix_sync(a_frag, sQ + tm * 16 * Q_STRIDE + tk * 16, Q_STRIDE);
                load_matrix_sync(b_frag, sKV + tn * 16 * KV_STRIDE + tk * 16, KV_STRIDE);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            // Apply scale and causal mask
            #pragma unroll
            for (int i = 0; i < acc.num_elements; ++i) {
                // Decode fragment position using Volta's layout
                const int frag_row = (lane_id & 1) + ((lane_id >> 2) & 1) * 8 +
                                     ((lane_id >> 4) & 1) * 4 + ((i >> 1) & 1) * 2;
                const int frag_col = ((lane_id >> 1) & 1) * 2 + ((lane_id >> 3) & 1) * 8 +
                                     (i & 1) + ((i >> 2) & 1) * 4;

                const int global_row = start_row + tm * 16 + frag_row;
                const int global_col = start_col + tn * 16 + frag_col;

                bool valid = (tm * 16 + frag_row < valid_q_rows) &&
                             (tn * 16 + frag_col < valid_k_rows);

                if constexpr (IS_CAUSAL) {
                    valid = valid && (global_col <= global_row);
                }

                acc.x[i] = valid ? acc.x[i] * softmax_scale : NEG_INF;
            }

            store_matrix_sync(sS + tm * 16 * S_STRIDE + tn * 16, acc, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Online softmax
        constexpr int THREADS_PER_ROW = 8;
        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int t_in_row = tid % THREADS_PER_ROW;
            const unsigned mask = 0xff << ((tid / 8) * 8);  // Mask for 8 threads

            float* sS_row = sS + row * S_STRIDE;
            __half* sP_row = sP + row * S_STRIDE;

            // Find max
            float thread_max = NEG_INF;
            for (int c = t_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                thread_max = fmaxf(thread_max, sS_row[c]);
            }

            // Warp reduce max
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) {
                thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, o));
            }

            const float old_max = sRowMax[row];
            const float new_max = fmaxf(old_max, thread_max);
            const float exp_diff = __expf(old_max - new_max);

            // Compute exp and sum
            float thread_sum = 0.0f;
            for (int c = t_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                float e = __expf(sS_row[c] - new_max);
                thread_sum += e;
                sP_row[c] = __float2half_rn(e);
            }

            // Zero padding
            for (int c = valid_k_rows + t_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                sP_row[c] = __float2half(0.0f);
            }

            // Warp reduce sum
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) {
                thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, o);
            }

            if (t_in_row == 0) {
                sRowSum[row] = exp_diff * sRowSum[row] + thread_sum;
                sRowMax[row] = new_max;
            }

            // Rescale previous output
            if (block_n > 0) {
                float* sO_row = sO + row * O_STRIDE;
                for (int c = t_in_row; c < D; c += THREADS_PER_ROW) {
                    sO_row[c] *= exp_diff;
                }
            }
        }
        __syncthreads();

        // Load V (reuse KV buffer)
        for (int i = tid; i < valid_k_rows * D; i += THREADS) {
            int row = i / D;
            int col = i % D;
            sKV[row * KV_STRIDE + col] = v_ptr[(start_col + row) * D + col];
        }
        __syncthreads();

        // Compute O += P @ V
        constexpr int TILES_D = (D + 15) / 16;
        const int total_o_tiles = TILES_M * TILES_D;
        const int o_tiles_per_warp = (total_o_tiles + WARPS - 1) / WARPS;

        for (int tile_idx = 0; tile_idx < o_tiles_per_warp; ++tile_idx) {
            const int global_tile = warp_id * o_tiles_per_warp + tile_idx;
            if (global_tile >= total_o_tiles) break;

            const int tm = global_tile / TILES_D;
            const int td = global_tile % TILES_D;

            if (tm * 16 >= valid_q_rows) continue;

            fragment<accumulator, 16, 16, 16, float> o_frag;
            load_matrix_sync(o_frag, sO + tm * 16 * O_STRIDE + td * 16, O_STRIDE, mem_row_major);

            #pragma unroll
            for (int tk = 0; tk < TILES_N; ++tk) {
                if (tk * 16 >= valid_k_rows) break;

                fragment<matrix_a, 16, 16, 16, half, row_major> p_frag;
                fragment<matrix_b, 16, 16, 16, half, row_major> v_frag;

                load_matrix_sync(p_frag, sP + tm * 16 * S_STRIDE + tk * 16, S_STRIDE);
                load_matrix_sync(v_frag, sKV + tk * 16 * KV_STRIDE + td * 16, KV_STRIDE);
                mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            store_matrix_sync(sO + tm * 16 * O_STRIDE + td * 16, o_frag, O_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    // Write output
    for (int i = tid; i < valid_q_rows * D; i += THREADS) {
        const int row = i / D;
        const int col = i % D;
        const float inv_sum = 1.0f / fmaxf(sRowSum[row], 1e-24f);
        out_ptr[row * D + col] = __float2half_rn(sO[row * O_STRIDE + col] * inv_sum);
    }

    // Write LSE
    if (lse_ptr && tid < valid_q_rows) {
        const float sum = fmaxf(sRowSum[tid], 1e-24f);
        lse_ptr[tid] = sRowMax[tid] + logf(sum);
    }
}

// ============================================================================
// LAUNCHER
// ============================================================================
template<int D>
cudaError_t launch_flash_attention_sm70_v2(
    const __half* Q, const __half* K, const __half* V,
    __half* Out, float* softmax_lse,
    int B, int H, int M, int N,
    float softmax_scale, bool is_causal,
    cudaStream_t stream
) {
    using Config = KernelConfigV2<D>;

    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::SMEM_SIZE;

    cudaFuncSetAttribute(
        is_causal ? flash_attention_sm70_v2_kernel<D, true> : flash_attention_sm70_v2_kernel<D, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem
    );

    if (is_causal) {
        flash_attention_sm70_v2_kernel<D, true><<<grid, block, smem, stream>>>(
            Q, K, V, Out, softmax_lse, B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_sm70_v2_kernel<D, false><<<grid, block, smem, stream>>>(
            Q, K, V, Out, softmax_lse, B, H, M, N, softmax_scale
        );
    }

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t launch_flash_attention_sm70_v2<64>(
    const __half*, const __half*, const __half*, __half*, float*,
    int, int, int, int, float, bool, cudaStream_t);

template cudaError_t launch_flash_attention_sm70_v2<128>(
    const __half*, const __half*, const __half*, __half*, float*,
    int, int, int, int, float, bool, cudaStream_t);

}  // namespace v2
}  // namespace sm70
}  // namespace flashinfer

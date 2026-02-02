/*
 * SM70 (Volta/V100) FlashAttention - Prefill Optimized Variant
 *
 * Key optimizations vs the decode kernel:
 * 1. Larger BLOCK_M (128) for better data reuse in prefill
 * 2. More aggressive K/V tiling (BLOCK_N=64) to fit in shared memory
 * 3. Better warp scheduling for square-ish attention patterns
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

namespace flashinfer {
namespace sm70 {

// ============================================================================
// PREFILL KERNEL CONFIGURATION
// ============================================================================
// V100 shared memory limit: 96KB per SM, but we target ~48KB for better occupancy.
// With BLOCK_M=64, D=128: q=17KB, kv=17KB, s=9KB, o=35KB = 78KB (fits in 96KB)
template<int D>
struct PrefillConfig {
    // Reduced BLOCK_M to fit V100 shared memory (96KB limit)
    static constexpr int BLOCK_M = 64;   // Fits within V100 smem
    static constexpr int BLOCK_N = 64;   // Same as v2
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    static constexpr int THREADS_PER_ROW = 4;  // 256 threads / 64 rows for softmax

    static constexpr int PAD = 8;
    static constexpr int Q_STRIDE = D + PAD;
    static constexpr int KV_STRIDE = D + PAD;
    static constexpr int S_STRIDE = BLOCK_N + PAD;
    static constexpr int O_STRIDE = D + PAD;

    struct alignas(128) SmemLayout {
        alignas(16) __half q[BLOCK_M * Q_STRIDE];
        union {
            alignas(16) __half k[BLOCK_N * KV_STRIDE];
            alignas(16) __half v[BLOCK_N * KV_STRIDE];
        } reuse_kv;
        union {
            alignas(16) float  s[BLOCK_M * S_STRIDE];
            alignas(16) __half p[BLOCK_M * S_STRIDE];
        } reuse_sp;
        alignas(16) float o[BLOCK_M * O_STRIDE];
        alignas(16) float row_max[BLOCK_M];
        alignas(16) float row_sum[BLOCK_M];
    };

    static constexpr size_t TOTAL_SMEM = ((sizeof(SmemLayout) + 127) & ~size_t(127));
};

// ============================================================================
// HELPER: Vectorized load
// ============================================================================
template<int D>
__device__ __forceinline__ void load_tile_vectorized(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int rows, int src_stride, int dst_stride,
    int tid, int num_threads
) {
    constexpr int VEC_SIZE = 8;  // Load 8 halfs at once
    const int vec_per_row = D / VEC_SIZE;

    for (int idx = tid; idx < rows * vec_per_row; idx += num_threads) {
        int row = idx / vec_per_row;
        int col = (idx % vec_per_row) * VEC_SIZE;

        if (row < rows) {
            // Use float4 for 8 halfs (16 bytes)
            float4 val = *reinterpret_cast<const float4*>(src + row * src_stride + col);
            *reinterpret_cast<float4*>(dst + row * dst_stride + col) = val;
        }
    }
}

// ============================================================================
// PREFILL FORWARD KERNEL
// ============================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(PrefillConfig<D>::THREADS_PER_BLOCK, 1)
flash_attention_sm70_prefill_kernel(
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
    using Config = PrefillConfig<D>;
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int THREADS = Config::THREADS_PER_BLOCK;
    constexpr int WARPS = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE = Config::Q_STRIDE;
    constexpr int KV_STRIDE = Config::KV_STRIDE;
    constexpr int S_STRIDE = Config::S_STRIDE;
    constexpr int O_STRIDE = Config::O_STRIDE;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    const float NEG_INF = -1e30f;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_m = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;

    const int valid_q_rows = min(BLOCK_M, M - start_row);
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    if constexpr (IS_CAUSAL) {
        const int max_key_pos = start_row + valid_q_rows - 1;
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
    __half* sK = smem.reuse_kv.k;
    __half* sV = smem.reuse_kv.v;
    float*  sS = smem.reuse_sp.s;
    __half* sP = smem.reuse_sp.p;
    float*  sO = smem.o;
    float*  sRowMax = smem.row_max;
    float*  sRowSum = smem.row_sum;

    // Initialize shared memory
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        sRowMax[i] = NEG_INF;
        sRowSum[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * O_STRIDE; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Load Q once (reused across all K/V tiles)
    load_tile_vectorized<D>(q_ptr, sQ, valid_q_rows, D, Q_STRIDE, tid, THREADS);
    __syncthreads();

    // Main loop over K/V tiles
    for (int block_n = 0; block_n < num_n_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= N) break;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        // Load K tile
        load_tile_vectorized<D>(k_ptr + start_col * D, sK, valid_k_rows, D, KV_STRIDE, tid, THREADS);
        __syncthreads();

        // Compute S = Q @ K^T using WMMA
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
        constexpr int num_tiles_m = BLOCK_M / WMMA_M;  // 8
        constexpr int num_tiles_n = BLOCK_N / WMMA_N;  // 4
        constexpr int num_tiles_k = D / WMMA_K;
        constexpr int total_tiles = num_tiles_m * num_tiles_n;  // 32
        constexpr int tiles_per_warp = (total_tiles + WARPS - 1) / WARPS;  // 4

        for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp + tile_idx;
            if (global_tile_idx >= total_tiles) break;

            const int tile_m_idx = global_tile_idx / num_tiles_n;
            const int tile_n_idx = global_tile_idx % num_tiles_n;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) {
                // Store NEG_INF for invalid tiles
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> inf_frag;
                fill_fragment(inf_frag, NEG_INF);
                store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, inf_frag, S_STRIDE, mem_row_major);
                continue;
            }

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            // Apply scaling and masks
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                const int frag_row = (lane_id & 1) + ((i >> 1) & 1) * 2
                                   + ((lane_id >> 2) & 1) * 8 + ((lane_id >> 4) & 1) * 4;
                const int frag_col = (i & 1) + ((lane_id >> 1) & 1) * 2
                                   + ((i >> 2) & 1) * 4 + ((lane_id >> 3) & 1) * 8;

                const bool is_valid = (tile_m + frag_row < valid_q_rows) &&
                                      (tile_n + frag_col < valid_k_rows);

                if constexpr (IS_CAUSAL) {
                    const int global_m = start_row + tile_m + frag_row;
                    const int global_n = start_col + tile_n + frag_col;
                    acc_frag.x[i] = is_valid
                        ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * softmax_scale)
                        : NEG_INF;
                } else {
                    acc_frag.x[i] = is_valid ? acc_frag.x[i] * softmax_scale : NEG_INF;
                }
            }

            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Online softmax - buffer S reads to avoid union aliasing
        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            const unsigned mask = __activemask();
            const int row_leader = __ffs(mask) - 1;

            float* sS_row = sS + row * S_STRIDE;
            __half* sP_row = sP + row * S_STRIDE;

            // Buffer S values to avoid S/P union aliasing
            constexpr int MAX_ITER = (BLOCK_N + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            float my_s[MAX_ITER];
            int my_count = 0;

            float thread_max = NEG_INF;
            for (int c = thread_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                float s_val = sS_row[c];
                my_s[my_count++] = s_val;
                thread_max = fmaxf(thread_max, s_val);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, THREADS_PER_ROW));

            const float row_max_val = __shfl_sync(mask, thread_max, row_leader, THREADS_PER_ROW);
            const float old_max = sRowMax[row];
            const float new_max = fmaxf(old_max, row_max_val);
            const float exp_diff = __expf(old_max - new_max);

            float thread_sum = 0.0f;
            int idx = 0;
            for (int c = thread_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                float e = __expf(fmaxf(my_s[idx++] - new_max, -80.0f));
                thread_sum += e;
                sP_row[c] = __float2half_rn(e);
            }

            for (int c = valid_k_rows + thread_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                sP_row[c] = __float2half(0.0f);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_sum += __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);

            float row_sum_val = __shfl_sync(mask, thread_sum, row_leader, THREADS_PER_ROW);

            if (thread_in_row == 0) {
                sRowSum[row] = exp_diff * sRowSum[row] + row_sum_val;
                sRowMax[row] = new_max;
            }

            // Scale previous output
            if (block_n > 0) {
                float* sO_row = sO + row * O_STRIDE;
                for (int c = thread_in_row; c < D; c += THREADS_PER_ROW) {
                    sO_row[c] *= exp_diff;
                }
            }
        }
        __syncthreads();

        // Load V tile (reuses K memory)
        load_tile_vectorized<D>(v_ptr + start_col * D, sV, valid_k_rows, D, KV_STRIDE, tid, THREADS);
        __syncthreads();

        // Compute O += P @ V using WMMA
        constexpr int num_tiles_m_pv = BLOCK_M / WMMA_M;
        constexpr int num_tiles_n_pv = D / WMMA_N;
        constexpr int num_tiles_k_pv = BLOCK_N / WMMA_K;
        constexpr int total_tiles_pv = num_tiles_m_pv * num_tiles_n_pv;
        constexpr int tiles_per_warp_pv = (total_tiles_pv + WARPS - 1) / WARPS;

        for (int tile_idx = 0; tile_idx < tiles_per_warp_pv; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp_pv + tile_idx;
            if (global_tile_idx >= total_tiles_pv) break;

            const int tile_m_idx = global_tile_idx / num_tiles_n_pv;
            const int tile_d_idx = global_tile_idx % num_tiles_n_pv;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_d = tile_d_idx * WMMA_N;

            if (tile_m >= valid_q_rows) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> v_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;

            load_matrix_sync(o_frag, sO + tile_m * O_STRIDE + tile_d, O_STRIDE, mem_row_major);

            #pragma unroll
            for (int tile_k = 0; tile_k < num_tiles_k_pv; ++tile_k) {
                const int k_offset = tile_k * WMMA_K;
                if (k_offset >= valid_k_rows) break;

                load_matrix_sync(p_frag, sP + tile_m * S_STRIDE + k_offset, S_STRIDE);
                load_matrix_sync(v_frag, sV + k_offset * KV_STRIDE + tile_d, KV_STRIDE);
                mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            store_matrix_sync(sO + tile_m * O_STRIDE + tile_d, o_frag, O_STRIDE, mem_row_major);
        }
        __syncthreads();
    }

    // Write final output
    for (int i = tid; i < valid_q_rows * D; i += THREADS) {
        const int row = i / D;
        const int col = i % D;
        const float sum_clamped = fmaxf(sRowSum[row], 1e-24f);
        const float val = sO[row * O_STRIDE + col] / sum_clamped;
        out_ptr[row * D + col] = __float2half_rn(val);
    }

    if (lse_ptr && tid < valid_q_rows) {
        const float sum = fmaxf(sRowSum[tid], 1e-24f);
        lse_ptr[tid] = sRowMax[tid] + logf(sum);
    }
}

// ============================================================================
// LAUNCHER
// ============================================================================
template<int D>
cudaError_t launch_flash_attention_sm70_prefill(
    const __half* Q, const __half* K, const __half* V,
    __half* Out, float* softmax_lse,
    int B, int H, int M, int N,
    float softmax_scale, bool is_causal,
    cudaStream_t stream
) {
    using Config = PrefillConfig<D>;

    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    cudaFuncSetAttribute(
        flash_attention_sm70_prefill_kernel<D, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(
        flash_attention_sm70_prefill_kernel<D, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    if (is_causal) {
        flash_attention_sm70_prefill_kernel<D, true><<<grid, block, smem, stream>>>(
            Q, K, V, Out, softmax_lse, B, H, M, N, softmax_scale);
    } else {
        flash_attention_sm70_prefill_kernel<D, false><<<grid, block, smem, stream>>>(
            Q, K, V, Out, softmax_lse, B, H, M, N, softmax_scale);
    }

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t launch_flash_attention_sm70_prefill<64>(
    const __half*, const __half*, const __half*, __half*, float*,
    int, int, int, int, float, bool, cudaStream_t);

template cudaError_t launch_flash_attention_sm70_prefill<128>(
    const __half*, const __half*, const __half*, __half*, float*,
    int, int, int, int, float, bool, cudaStream_t);

}  // namespace sm70
}  // namespace flashinfer

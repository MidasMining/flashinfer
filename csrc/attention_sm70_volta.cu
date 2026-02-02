/*
 * Copyright (c) 2025 by FlashInfer team.
 * Based on ai-bond/flash-attention-v100 (BSD-3-Clause)
 *
 * SM70 (Volta/V100) FlashAttention implementation using WMMA API.
 * This provides native tensor core support for V100 GPUs.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

namespace flashinfer {
namespace sm70 {

// ============================================================================
// VOLTA SM70 WMMA CONSTANTS
// ============================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// KERNEL CONFIGURATIONS
// ============================================================================
template<int D>
struct KernelConfig {
    // Block sizes tuned for V100
    static constexpr int BLOCK_M = (D <= 64) ? 64 : 32;
    static constexpr int BLOCK_N = (D <= 64) ? 128 : 128;
    static constexpr int WARPS_PER_BLOCK = 16;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    static constexpr int THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M;

    // Padding to avoid bank conflicts
    static constexpr int PAD = (8 - (D % 32) + 32) % 32;
    static constexpr int Q_STRIDE = D + PAD;
    static constexpr int KV_STRIDE = D + PAD;
    static constexpr int S_STRIDE = BLOCK_N + PAD;
    static constexpr int O_STRIDE = D + PAD;

    // Shared memory layout
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
// VECTORIZED LOAD WITH SWIZZLE
// ============================================================================
__device__ __forceinline__ void store_u4_swizzle(
    const uint4* g_vec, uint4* s_vec, int rows, int vec_per_row,
    int stride_u4, int lane_id, int warp_id, int total_warps) {
    const int ROW_GROUP = 8;
    for (int base_row = warp_id * ROW_GROUP; base_row < rows; base_row += total_warps * ROW_GROUP) {
        const int r = lane_id % ROW_GROUP;
        const int c = lane_id / ROW_GROUP;
        const int row = base_row + r;
        if (row >= rows) continue;

        for (int col0 = 0; col0 < vec_per_row; col0 += 4) {
            int cg = min(4, vec_per_row - col0);
            int c_eff = (cg == 4) ? (c ^ (r & 3)) : ((c + r) % cg);
            if (c_eff >= cg) continue;

            int col = col0 + c_eff;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < rows && col < vec_per_row) {
                val = __ldg(&g_vec[row * vec_per_row + col]);
            }
            s_vec[row * stride_u4 + col] = val;
        }
    }
}

// ============================================================================
// INITIALIZE SHARED MEMORY
// ============================================================================
template<typename Config>
__device__ __forceinline__ void init_smem(char* smem_raw) {
    constexpr int N_U4 = Config::TOTAL_SMEM / 16;
    // Use ALL threads (not just lane_id) to avoid redundant work and races
    const int tid = threadIdx.x;
    constexpr int THREADS = Config::THREADS_PER_BLOCK;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));

    #pragma unroll 1
    for (int i = tid; i < N_U4; i += THREADS) {
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};"
                     :: "r"(addr + (i << 4)), "r"(0) : "memory");
    }
    __syncthreads();  // Ensure all threads finish before proceeding
}

// ============================================================================
// FORWARD KERNEL
// ============================================================================
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(KernelConfig<D>::THREADS_PER_BLOCK, 2)
flash_attention_sm70_forward_kernel(
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
    using Config = KernelConfig<D>;
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    constexpr int THREADS_PER_ROW = Config::THREADS_PER_ROW;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int Q_STRIDE = Config::Q_STRIDE;
    constexpr int KV_STRIDE = Config::KV_STRIDE;
    constexpr int S_STRIDE = Config::S_STRIDE;
    constexpr int O_STRIDE = Config::O_STRIDE;
    constexpr int PER_UINT4 = 8;
    const float NEG_INF = -1e30f;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= B * H) return;

    const int block_m = blockIdx.x;
    const int start_row = block_m * BLOCK_M;
    if (start_row >= M) return;

    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int valid_q_rows = min(BLOCK_M, M - start_row);

    // Early exit for causal
    if constexpr (IS_CAUSAL) {
        const int max_key_pos = start_row + valid_q_rows - 1;
        if (max_key_pos < 0) {
            num_n_tiles = 0;
        } else {
            num_n_tiles = min(num_n_tiles, (max_key_pos + BLOCK_N) / BLOCK_N);
        }
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
    init_smem<Config>(smem_raw);
    auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);

    __half* sQ = smem.q;
    __half* sK = smem.reuse_kv.k;
    __half* sV = smem.reuse_kv.v;
    float*  sS = smem.reuse_sp.s;
    __half* sP = smem.reuse_sp.p;
    float*  sO = smem.o;
    float*  sRowMax = smem.row_max;
    float*  sRowSum = smem.row_sum;

    // Vector strides
    const int d_stride_uint4 = (D + PER_UINT4 - 1) / PER_UINT4;
    const int q_stride_uint4 = (Q_STRIDE + PER_UINT4 - 1) / PER_UINT4;
    const int kv_stride_uint4 = (KV_STRIDE + PER_UINT4 - 1) / PER_UINT4;

    // Initialize row max
    if (tid < BLOCK_M) {
        sRowMax[tid] = NEG_INF;
        sRowSum[tid] = 0.0f;
    }

    // Initialize output accumulator
    for (int i = tid; i < BLOCK_M * O_STRIDE; i += THREADS_PER_BLOCK) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Load Q
    const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
    uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
    store_u4_swizzle(q_vec, sQ_vec, valid_q_rows, d_stride_uint4, q_stride_uint4,
                     lane_id, warp_id, WARPS_PER_BLOCK);
    __syncthreads();

    // Main loop over K/V tiles
    for (int block_n = 0; block_n < num_n_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        if (start_col >= N) break;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        if constexpr (IS_CAUSAL) {
            if (start_col >= start_row + valid_q_rows) continue;
        }

        // Load K
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_ptr + start_col * D);
        uint4* sK_vec = reinterpret_cast<uint4*>(sK);
        store_u4_swizzle(k_vec, sK_vec, valid_k_rows, d_stride_uint4, kv_stride_uint4,
                         lane_id, warp_id, WARPS_PER_BLOCK);
        __syncthreads();

        // Compute S = Q @ K^T using WMMA
        const int num_tiles_m = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n = (BLOCK_N + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles = num_tiles_m * num_tiles_n;
        const int tiles_per_warp = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        for (int tile_idx = 0; tile_idx < tiles_per_warp; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp + tile_idx;
            if (global_tile_idx >= total_tiles) break;

            const int tile_m_idx = global_tile_idx / num_tiles_n;
            const int tile_n_idx = global_tile_idx % num_tiles_n;

            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;

            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) continue;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
                const int k_offset = k_tile * WMMA_K;
                if (k_offset >= D) break;

                load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_offset, Q_STRIDE);
                load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_offset, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            // Apply scaling and causal mask
            // Use correct Volta WMMA fragment layout (verified empirically)
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                // Correct fragment position decoding for SM70 WMMA
                const int frag_row = (lane_id & 1)
                                   + ((i >> 1) & 1) * 2
                                   + ((lane_id >> 2) & 1) * 8
                                   + ((lane_id >> 4) & 1) * 4;
                const int frag_col = (i & 1)
                                   + ((lane_id >> 1) & 1) * 2
                                   + ((i >> 2) & 1) * 4
                                   + ((lane_id >> 3) & 1) * 8;

                const bool is_valid = (tile_m + frag_row < valid_q_rows) &&
                                      (tile_n + frag_col < valid_k_rows);

                if constexpr (IS_CAUSAL) {
                    const int global_m = start_row + tile_m + frag_row;
                    const int global_n = start_col + tile_n + frag_col;
                    acc_frag.x[i] = is_valid
                        ? ((global_n > global_m) ? NEG_INF : acc_frag.x[i] * softmax_scale)
                        : NEG_INF;
                } else {
                    // For non-causal, mask invalid positions with NEG_INF too
                    // to prevent reading uninitialized/wrong K values
                    acc_frag.x[i] = is_valid
                        ? acc_frag.x[i] * softmax_scale
                        : NEG_INF;
                }
            }
            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Online softmax
        if (tid < valid_q_rows * THREADS_PER_ROW) {
            const int row = tid / THREADS_PER_ROW;
            const int thread_in_row = tid % THREADS_PER_ROW;
            const unsigned mask = __activemask();
            const int row_leader = __ffs(mask) - 1;

            float* sS_row = sS + row * S_STRIDE;
            __half* sP_row = sP + row * S_STRIDE;

            // IMPORTANT: sS (float) and sP (half) share a union. Due to different element
            // sizes, writes to sP for row R can corrupt sS reads for row R-1 at higher
            // column indices. To avoid this, we buffer all S reads before any P writes.
            // Max iterations per thread: ceil(BLOCK_N / THREADS_PER_ROW) = 16
            constexpr int MAX_ITER = (BLOCK_N + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
            float my_s[MAX_ITER];
            int my_count = 0;

            // Phase 1: Read all S values into registers (no P writes yet)
            float thread_max = NEG_INF;
            for (int c = thread_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                float s_val = sS_row[c];
                my_s[my_count++] = s_val;
                thread_max = fmaxf(thread_max, s_val);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, THREADS_PER_ROW));

            const float row_max = __shfl_sync(mask, thread_max, row_leader, THREADS_PER_ROW);
            const float old_max = sRowMax[row];
            const float new_max = fmaxf(old_max, row_max);
            const float exp_diff = __expf(old_max - new_max);

            // Phase 2: Compute exp and write P (S is now fully buffered, safe to write P)
            float thread_sum = 0.0f;
            int idx = 0;
            for (int c = thread_in_row; c < valid_k_rows; c += THREADS_PER_ROW) {
                float e = __expf(fmaxf(my_s[idx++] - new_max, -80.0f));
                thread_sum += e;
                sP_row[c] = __float2half_rn(e);
            }

            // Fill padding with zeros
            for (int c = valid_k_rows + thread_in_row; c < BLOCK_N; c += THREADS_PER_ROW) {
                sP_row[c] = __float2half(0.0f);
            }

            #pragma unroll
            for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1)
                thread_sum += __shfl_down_sync(mask, thread_sum, o, THREADS_PER_ROW);

            float row_sum = __shfl_sync(mask, thread_sum, row_leader, THREADS_PER_ROW);

            if (thread_in_row == 0) {
                sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
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

        // Load V
        const uint4* v_vec = reinterpret_cast<const uint4*>(v_ptr + start_col * D);
        uint4* sV_vec = reinterpret_cast<uint4*>(sV);
        store_u4_swizzle(v_vec, sV_vec, valid_k_rows, d_stride_uint4, kv_stride_uint4,
                         lane_id, warp_id, WARPS_PER_BLOCK);
        __syncthreads();

        // Compute O += P @ V using WMMA
        const int num_tiles_m_pv = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_pv = (D + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_pv = (BLOCK_N + WMMA_K - 1) / WMMA_K;
        const int total_tiles_pv = num_tiles_m_pv * num_tiles_n_pv;
        const int tiles_per_warp_pv = (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

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
    for (int i = tid; i < valid_q_rows * D; i += THREADS_PER_BLOCK) {
        const int row = i / D;
        const int col = i % D;

        const float sum_clamped = fmaxf(sRowSum[row], 1e-24f);
        const float inv_sum = 1.0f / sum_clamped;
        const float val = sO[row * O_STRIDE + col] * inv_sum;

        out_ptr[row * D + col] = __float2half_rn(val);
    }

    // Write LSE if requested
    if (lse_ptr && tid < valid_q_rows) {
        const float sum = fmaxf(sRowSum[tid], 1e-24f);
        lse_ptr[tid] = sRowMax[tid] + logf(sum);
    }
}

// ============================================================================
// LAUNCHER TEMPLATES
// ============================================================================
template<int D>
cudaError_t launch_flash_attention_sm70_forward(
    const __half* Q, const __half* K, const __half* V,
    __half* Out, float* softmax_lse,
    int B, int H, int M, int N,
    float softmax_scale, bool is_causal,
    cudaStream_t stream
) {
    using Config = KernelConfig<D>;

    const int grid_x = (M + Config::BLOCK_M - 1) / Config::BLOCK_M;
    const dim3 grid(grid_x, 1, B * H);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;

    auto kernel = is_causal ?
        flash_attention_sm70_forward_kernel<D, true> :
        flash_attention_sm70_forward_kernel<D, false>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    if (is_causal) {
        flash_attention_sm70_forward_kernel<D, true><<<grid, block, smem, stream>>>(
            Q, K, V, Out, softmax_lse, B, H, M, N, softmax_scale
        );
    } else {
        flash_attention_sm70_forward_kernel<D, false><<<grid, block, smem, stream>>>(
            Q, K, V, Out, softmax_lse, B, H, M, N, softmax_scale
        );
    }

    return cudaGetLastError();
}

// Explicit instantiations for common head dimensions
template cudaError_t launch_flash_attention_sm70_forward<64>(
    const __half*, const __half*, const __half*, __half*, float*,
    int, int, int, int, float, bool, cudaStream_t);

template cudaError_t launch_flash_attention_sm70_forward<128>(
    const __half*, const __half*, const __half*, __half*, float*,
    int, int, int, int, float, bool, cudaStream_t);

}  // namespace sm70
}  // namespace flashinfer

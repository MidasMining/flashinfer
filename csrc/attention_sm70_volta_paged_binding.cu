/*
 * Copyright (c) 2025 by FlashInfer team.
 * TVM-FFI bindings for SM70 paged attention kernel.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm70 {
namespace paged {

// Forward declaration
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
);

}  // namespace paged
}  // namespace sm70
}  // namespace flashinfer

using namespace flashinfer;

void FlashAttentionSM70PagedDecode(
    TensorView q,             // [num_seqs, num_heads, head_dim]
    TensorView k_cache,       // [num_blocks, block_size, num_kv_heads, head_dim]
    TensorView v_cache,       // [num_blocks, block_size, num_kv_heads, head_dim]
    TensorView block_tables,  // [num_seqs, max_blocks_per_seq]
    TensorView seq_lens,      // [num_seqs]
    TensorView out,           // [num_seqs, num_heads, head_dim]
    double softmax_scale
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_tables);
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(out);

    // Get dimensions
    int num_seqs = q.size(0);
    int num_heads = q.size(1);
    int D = q.size(2);

    int block_size = k_cache.size(1);
    int num_kv_heads = k_cache.size(2);
    int max_blocks_per_seq = block_tables.size(1);

    // Validate dimensions
    TVM_FFI_ICHECK(D == 64 || D == 128 || D == 256)
        << "Head dim must be 64, 128, or 256, got " << D;
    TVM_FFI_ICHECK_EQ(k_cache.size(3), D);
    TVM_FFI_ICHECK_EQ(v_cache.size(1), block_size);
    TVM_FFI_ICHECK_EQ(v_cache.size(2), num_kv_heads);
    TVM_FFI_ICHECK_EQ(v_cache.size(3), D);
    TVM_FFI_ICHECK_EQ(out.size(0), num_seqs);
    TVM_FFI_ICHECK_EQ(out.size(1), num_heads);
    TVM_FFI_ICHECK_EQ(out.size(2), D);

    ffi::CUDADeviceGuard device_guard(q.device().device_id);
    cudaStream_t stream = get_stream(q.device());

    const __half* q_ptr = static_cast<const __half*>(q.data_ptr());
    const __half* k_ptr = static_cast<const __half*>(k_cache.data_ptr());
    const __half* v_ptr = static_cast<const __half*>(v_cache.data_ptr());
    const int* block_tables_ptr = static_cast<const int*>(block_tables.data_ptr());
    const int* seq_lens_ptr = static_cast<const int*>(seq_lens.data_ptr());
    __half* out_ptr = static_cast<__half*>(out.data_ptr());

    cudaError_t status;

    if (D == 64) {
        status = sm70::paged::launch_flash_attention_sm70_paged_decode<64>(
            q_ptr, k_ptr, v_ptr, block_tables_ptr, seq_lens_ptr, out_ptr,
            num_seqs, num_heads, num_kv_heads, max_blocks_per_seq,
            block_size, static_cast<float>(softmax_scale), stream
        );
    } else if (D == 128) {
        status = sm70::paged::launch_flash_attention_sm70_paged_decode<128>(
            q_ptr, k_ptr, v_ptr, block_tables_ptr, seq_lens_ptr, out_ptr,
            num_seqs, num_heads, num_kv_heads, max_blocks_per_seq,
            block_size, static_cast<float>(softmax_scale), stream
        );
    } else {
        // D == 256 (asserted above)
        status = sm70::paged::launch_flash_attention_sm70_paged_decode<256>(
            q_ptr, k_ptr, v_ptr, block_tables_ptr, seq_lens_ptr, out_ptr,
            num_seqs, num_heads, num_kv_heads, max_blocks_per_seq,
            block_size, static_cast<float>(softmax_scale), stream
        );
    }

    TVM_FFI_ICHECK(status == cudaSuccess)
        << "SM70 paged attention failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(FlashAttentionSM70PagedDecode, FlashAttentionSM70PagedDecode);

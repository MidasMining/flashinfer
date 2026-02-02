/*
 * Copyright (c) 2025 by FlashInfer team.
 * TVM-FFI bindings for SM70 prefill-optimized attention kernel.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm70 {

// Forward declaration
template<int D>
cudaError_t launch_flash_attention_sm70_prefill(
    const __half* Q, const __half* K, const __half* V,
    __half* Out, float* softmax_lse,
    int B, int H, int M, int N,
    float softmax_scale, bool is_causal,
    cudaStream_t stream);

}  // namespace sm70
}  // namespace flashinfer

using namespace flashinfer;

void FlashAttentionSM70PrefillForward(
    TensorView q,       // [B, H, M, D]
    TensorView k,       // [B, H, N, D]
    TensorView v,       // [B, H, N, D]
    TensorView out,     // [B, H, M, D]
    bool is_causal,
    double softmax_scale
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(out);

    // Get dimensions
    int B = q.size(0);
    int H = q.size(1);
    int M = q.size(2);
    int D = q.size(3);
    int N = k.size(2);

    // Validate
    TVM_FFI_ICHECK_EQ(k.size(0), B);
    TVM_FFI_ICHECK_EQ(k.size(1), H);
    TVM_FFI_ICHECK_EQ(k.size(3), D);
    TVM_FFI_ICHECK_EQ(v.size(0), B);
    TVM_FFI_ICHECK_EQ(v.size(1), H);
    TVM_FFI_ICHECK_EQ(v.size(2), N);
    TVM_FFI_ICHECK_EQ(v.size(3), D);

    ffi::CUDADeviceGuard device_guard(q.device().device_id);
    cudaStream_t stream = get_stream(q.device());

    const __half* q_ptr = static_cast<const __half*>(q.data_ptr());
    const __half* k_ptr = static_cast<const __half*>(k.data_ptr());
    const __half* v_ptr = static_cast<const __half*>(v.data_ptr());
    __half* out_ptr = static_cast<__half*>(out.data_ptr());

    cudaError_t status;

    // Dispatch based on head dimension
    if (D == 64) {
        status = sm70::launch_flash_attention_sm70_prefill<64>(
            q_ptr, k_ptr, v_ptr, out_ptr, nullptr,
            B, H, M, N, static_cast<float>(softmax_scale), is_causal, stream
        );
    } else if (D == 128) {
        status = sm70::launch_flash_attention_sm70_prefill<128>(
            q_ptr, k_ptr, v_ptr, out_ptr, nullptr,
            B, H, M, N, static_cast<float>(softmax_scale), is_causal, stream
        );
    } else {
        TVM_FFI_ICHECK(false) << "Unsupported head dimension: " << D
                              << ". SM70 prefill kernel supports D=64 or D=128.";
    }

    TVM_FFI_ICHECK(status == cudaSuccess)
        << "SM70 prefill attention failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(FlashAttentionSM70PrefillForward, FlashAttentionSM70PrefillForward);

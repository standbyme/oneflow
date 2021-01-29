/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include <cusparse.h>

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
    }                                                                          \
  }

namespace oneflow {
namespace {
class SpmmCSRGpuFloatKernel final : public user_op::OpKernel {
 public:
  SpmmCSRGpuFloatKernel() = default;
  ~SpmmCSRGpuFloatKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const int64_t A_num_rows = ctx->Attr<int64_t>("a_rows");
    const int64_t A_num_cols = ctx->Attr<int64_t>("a_cols");

    const user_op::Tensor *a_csrRowOffsets = ctx->Tensor4ArgNameAndIndex("a_csrRowOffsets", 0);
    const user_op::Tensor *a_csrColInd = ctx->Tensor4ArgNameAndIndex("a_csrColInd", 0);
    const user_op::Tensor *a_csrValues = ctx->Tensor4ArgNameAndIndex("a_csrValues", 0);
    const user_op::Tensor *b = ctx->Tensor4ArgNameAndIndex("b", 0);

    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t *a_csrRowOffsets_ptr = a_csrRowOffsets->dptr<int32_t>();
    const int32_t *a_csrColInd_ptr = a_csrColInd->dptr<int32_t>();
    const float *a_csrValues_ptr = a_csrValues->dptr<float>();
    const float *b_ptr = b->dptr<float>();

    float *out_ptr = out_tensor->mut_dptr<float>();

    int A_nnz = a_csrColInd->shape().At(0);
    int B_num_rows = b->shape().At(0);
    int B_num_cols = b->shape().At(1);

    int ldb = B_num_cols;
    int ldc = B_num_cols;
    int B_size = B_num_rows * B_num_cols;
    int C_size = A_num_rows * B_num_cols;

    const int32_t *hA_csrOffsets = a_csrRowOffsets_ptr;
    const int32_t *hA_columns = a_csrColInd_ptr;
    const float *hA_values = a_csrValues_ptr;
    const float *hB = b_ptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int32_t *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    OF_CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int32_t)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int32_t)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dB, B_size * sizeof(float)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dC, C_size * sizeof(float)));

    OF_CUDA_CHECK(cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int32_t),
                             cudaMemcpyHostToDevice));
    OF_CUDA_CHECK(
        cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    OF_CUDA_CHECK(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    OF_CUDA_CHECK(cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz, dA_csrOffsets,
                                     dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, out_ptr, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
        matB, &beta, matC, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &bufferSize))
    OF_CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, dBuffer))

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device memory deallocation
    OF_CUDA_CHECK(cudaFree(dBuffer));
    OF_CUDA_CHECK(cudaFree(dA_csrOffsets));
    OF_CUDA_CHECK(cudaFree(dA_columns));
    OF_CUDA_CHECK(cudaFree(dA_values));
    OF_CUDA_CHECK(cudaFree(dB));
    OF_CUDA_CHECK(cudaFree(dC));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPMM_CSR_KERNEL(device, dtype)            \
  REGISTER_USER_KERNEL("spmm_csr")                         \
      .SetCreateFn<SpmmCSRGpuFloatKernel>()                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_SPMM_CSR_KERNEL(DeviceType::kGPU, float)
}  // namespace
}  // namespace oneflow

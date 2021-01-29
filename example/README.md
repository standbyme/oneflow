# README

# cuSPARSE: Only vaild in CUDA 11
cmake/third_party.cmake: 51 libcusparse_static.a 

# Kernel
oneflow/user/kernels/spmm_coo_kernel.cu
oneflow/python/test/ops/test_spmm_coo.py

oneflow/user/kernels/spmm_csr_kernel.cu
oneflow/python/test/ops/test_spmm_csr.py

oneflow/python/ops/linalg.py

# Layer
oneflow/python/ops/layers.py: 33 GraphConvolution

# Example
example/gcn.py
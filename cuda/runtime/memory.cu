#include <cuda_runtime.h>

#include "exception.cuh"
#include <cstdint>

extern "C" rustCudaError_t cuda_setup_mem_pool()
{
    cudaMemPool_t mempool;
    CUDA_OK(cudaDeviceGetDefaultMemPool(&mempool, 0));
    uint64_t threshold = UINT64_MAX;
    CUDA_OK(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    return CUDA_SUCCESS_MOON;
}
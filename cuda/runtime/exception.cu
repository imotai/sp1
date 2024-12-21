#include <cuda_runtime.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <sstream>

#include "exception.cuh"

extern "C" const rustCudaError_t CUDA_SUCCESS_MOON =
rustCudaError_t{message : cudaGetErrorString(cudaSuccess)};

extern "C" const rustCudaError_t CUDA_OUT_OF_MEMORY =
rustCudaError_t{message : cudaGetErrorString(cudaErrorMemoryAllocation)};

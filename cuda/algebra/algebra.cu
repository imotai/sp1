#include <cstdint>
#include "algebra.cuh"

template <typename T>
__global__ void addKernel(T *a, T *b, T *c, size_t n)
{
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

extern "C" void *addKernelu32Ptr()
{
    return (void *)addKernel<uint32_t>;
}

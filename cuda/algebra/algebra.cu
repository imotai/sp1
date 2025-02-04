#include <cstdint>
#include "algebra.cuh"

#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"

template <typename U, typename T>
__global__ void addKernel(U *a, T *b, U *c, size_t n)
{
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

template <typename U, typename T>
__global__ void addAssignKernel(U *a, T *b, size_t n)
{
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        a[i] += b[i];
    }
}

extern "C" void *addKernelu32Ptr()
{
    return (void *)addKernel<uint32_t, uint32_t>;
}

extern "C" void *add_baby_bear_kernel()
{
    return (void *)addKernel<bb31_t, bb31_t>;
}

extern "C" void *add_baby_bear_ext_ext_kernel()
{
    return (void *)addKernel<bb31_extension_t, bb31_extension_t>;
}

extern "C" void *add_baby_bear_base_ext_kernel()
{
    return (void *)addKernel<bb31_extension_t, bb31_t>;
}

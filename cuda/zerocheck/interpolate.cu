#include "interpolate.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"

template<typename K, size_t MEMORY_SIZE>
__global__ void interpolate_row(
    const K *input,
    bb31_extension_t *__restrict__ output,
    size_t outputHeight,
    size_t width)
{
    size_t inputHeight = outputHeight << 1;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < outputHeight; i += blockDim.x * gridDim.x)
    {
        for (size_t j = blockDim.y * blockIdx.y + threadIdx.y; j < width; j += blockDim.y * gridDim.y)
        {
            K zeroValue = K::load(input, j * inputHeight + (i << 1));
            K oneValue = K::load(input, j * inputHeight + (i << 1) + 1);

            // Calculate slope.
            K slope = oneValue - zeroValue;
            K slope_times_two = slope + slope;
            K slope_times_four = slope_times_two + slope_times_two;

            K y_0 = zeroValue;
            K y_2 = slope_times_two + zeroValue;
            K y_4 = slope_times_four + zeroValue;

            bb31_extension_t::store(output, j * outputHeight + i, y_0);
            bb31_extension_t::store(output, 1 * (outputHeight * width) + (j * outputHeight + i), y_2);
            bb31_extension_t::store(output, 2 * (outputHeight * width) + (j * outputHeight + i), y_4);
        }
    }
}

extern "C" void *interpolate_row_baby_bear_kernel()
{
    return (void *)interpolate_row<bb31_t, 32>;
}

extern "C" void *interpolate_row_baby_bear_extension_kernel()
{
    return (void *)interpolate_row<bb31_extension_t, 32>;
}
#include "interpolate.cuh"
#include "../fields/kb31_extension_t.cuh"
#include "../fields/kb31_t.cuh"

template <typename K>
__global__ void interpolate_row(
    const K *input,
    K *__restrict__ output,
    size_t inputHeight,
    size_t width,
    size_t outputHeight,
    size_t offset,
    size_t globalHeight)
{
    bool oddNumRows = (inputHeight & 1) == 1;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < outputHeight; i += blockDim.x * gridDim.x)
    {
        for (size_t j = blockDim.y * blockIdx.y + threadIdx.y; j < width; j += blockDim.y * gridDim.y)
        {
            size_t rowIdx = i + offset;
            K zeroValue = K::load(input, j * inputHeight + (rowIdx << 1));
            K oneValue;
            
            if (oddNumRows && (rowIdx == (globalHeight - 1)))
            {
                oneValue = K::zero();
            }
            else
            {
                // Load the next row value
                oneValue = K::load(input, j * inputHeight + (rowIdx << 1) + 1);
            }

            // Calculate slope.
            K slope = oneValue - zeroValue;
            K slope_times_two = slope + slope;
            K slope_times_four = slope_times_two + slope_times_two;

            K y_0 = zeroValue;
            K y_2 = slope_times_two + zeroValue;
            K y_4 = slope_times_four + zeroValue;

            K::store(output, j * outputHeight + i, y_0);
            K::store(output, 1 * (outputHeight * width) + (j * outputHeight + i), y_2);
            K::store(output, 2 * (outputHeight * width) + (j * outputHeight + i), y_4);
        }
    }
}

extern "C" void *interpolate_row_koala_bear_kernel()
{
    return (void *)interpolate_row<kb31_t>;
}

extern "C" void *interpolate_row_koala_bear_extension_kernel()
{
    return (void *)interpolate_row<kb31_extension_t>;
}
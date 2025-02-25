#include "jagged.cuh"

#include <stdio.h>
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"

template <typename EF>
__global__ void jaggedPopulate(
    EF *f,
    const EF *eq_z_col,
    const EF *eq_z_row,
    size_t offset,
    size_t colOffset,
    size_t height,
    size_t width)
{

    for (size_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x; rowIdx < height; rowIdx += gridDim.x * blockDim.x)
    {
        EF eq_z_row_val = EF::load(eq_z_row, rowIdx);
        for (size_t colIdx = blockIdx.y * blockDim.y + threadIdx.y; colIdx < width; colIdx += gridDim.y * blockDim.y)
        {
            EF eq_z_col_val = EF::load(eq_z_col, colIdx + colOffset);
            size_t outIdx = colIdx * height + rowIdx;
            EF value = eq_z_col_val * eq_z_row_val;
            EF::store(f, outIdx + offset, value);
        }
    }
}

extern "C" void *jagged_baby_bear_extension_populate()
{
    return (void *)jaggedPopulate<bb31_extension_t>;
}

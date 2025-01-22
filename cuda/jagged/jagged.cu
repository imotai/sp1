#include "jagged.cuh"

#include <stdio.h>
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"

template <typename EF, bool row_major>
__global__ void jaggedTablePopulate(
    EF *f,
    const EF *eq_ztab,
    const EF *eq_z_col,
    const EF *eq_z_row,
    size_t offset,
    size_t tableIdx,
    size_t height,
    size_t width)
{
    for (size_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x; rowIdx < height; rowIdx += gridDim.x * blockDim.x)
    {
        for (size_t colIdx = blockIdx.y * blockDim.y + threadIdx.y; colIdx < width; colIdx += gridDim.y * blockDim.y)
        {

            EF eq_ztab_val = EF::load(eq_ztab, tableIdx);
            EF eq_z_col_val = EF::load(eq_z_col, colIdx);
            EF eq_z_row_val = EF::load(eq_z_row, rowIdx);

            size_t outIdx = row_major ? rowIdx * width + colIdx : colIdx * height + rowIdx;

            EF value = eq_ztab_val * eq_z_col_val * eq_z_row_val;
            EF::store(f, outIdx + offset, value);
        }
    }
}

extern "C" void *jagged_table_baby_bear_extension_populate_row_major()
{
    return (void *)jaggedTablePopulate<bb31_extension_t, true>;
}

extern "C" void *jagged_table_baby_bear_extension_populate_col_major()
{
    return (void *)jaggedTablePopulate<bb31_extension_t, false>;
}

template <typename EF, bool row_major>
__global__ void jaggedColumnPopulate(
    EF *f,
    const EF *eq_z_col,
    const EF *eq_z_row,
    size_t offset,
    size_t height,
    size_t width)
{
    for (size_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x; rowIdx < height; rowIdx += gridDim.x * blockDim.x)
    {
        for (size_t colIdx = blockIdx.y * blockDim.y + threadIdx.y; colIdx < width; colIdx += gridDim.y * blockDim.y)
        {

            EF eq_z_col_val = EF::load(eq_z_col, colIdx);
            EF eq_z_row_val = EF::load(eq_z_row, rowIdx);

            size_t outIdx = row_major ? rowIdx * width + colIdx : colIdx * height + rowIdx;

            EF value = eq_z_col_val * eq_z_row_val;
            EF::store(f, outIdx + offset, value);
        }
    }
}

extern "C" void *jagged_column_baby_bear_extension_populate_row_major()
{
    return (void *)jaggedColumnPopulate<bb31_extension_t, true>;
}

extern "C" void *jagged_column_baby_bear_extension_populate_col_major()
{
    return (void *)jaggedColumnPopulate<bb31_extension_t, false>;
}
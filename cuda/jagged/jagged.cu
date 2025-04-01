#include "jagged.cuh"

#include <stdio.h>
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"

#include "../reduce/reduction.cuh"

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

template <typename F, typename EF>
__global__ void jaggedSumAsPoly(
    EF *result,
    const F **base,
    const EF *eqzCol,
    const EF *eqzRow,
    size_t offset,
    size_t colOffset,
    size_t halfHeight,
    size_t width,
    size_t stackingWidth)
{
    EF evalZero = EF::zero();
    EF evalHalf = EF::zero();

    for (size_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x; rowIdx < halfHeight; rowIdx += gridDim.x * blockDim.x)
    {
        size_t evenRowIdx = rowIdx << 1;
        size_t oddRowIdx = (rowIdx << 1) + 1;

        EF eqzRowZero = EF::load(eqzRow, evenRowIdx);
        EF eqzRowOne = EF::load(eqzRow, oddRowIdx);
        for (size_t colIdx = blockIdx.y * blockDim.y + threadIdx.y; colIdx < width; colIdx += gridDim.y * blockDim.y)
        {
            EF eqzColVal = EF::load(eqzCol, colIdx + colOffset);

            EF jaggedValZero = eqzColVal * eqzRowZero;
            EF jaggedValOne = eqzColVal * eqzRowOne;

            size_t halfOffset = offset / 2;
            size_t baseIdx = colIdx * halfHeight + rowIdx + halfOffset;

            size_t evenIdx = baseIdx << 1;
            size_t oddIdx = (baseIdx << 1) + 1;

            size_t evenBatchIdx = evenIdx / stackingWidth;
            size_t oddBatchIdx = oddIdx / stackingWidth;

            F baseZero = F::load(base[evenBatchIdx], evenIdx % stackingWidth);
            F baseOne = F::load(base[oddBatchIdx], oddIdx % stackingWidth);

            evalZero += jaggedValZero * baseZero;
            evalHalf += (jaggedValZero + jaggedValOne) * (baseZero + baseOne);
        }
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    EF *shared = reinterpret_cast<EF *>(memory);

    AddOp<EF> op;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    EF evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared, op);
    EF evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared, op);

    if (threadIdx.x == 0)
    {
        EF::store(result, gridDim.x * blockIdx.y + blockIdx.x, evalZeroblockSum);
        EF::store(result, gridDim.x * gridDim.y + gridDim.x * blockIdx.y + blockIdx.x, evalHalfblockSum);
    }
}

extern "C" void *jagged_baby_bear_base_ext_sum_as_poly()
{
    return (void *)jaggedSumAsPoly<bb31_t, bb31_extension_t>;
}

template <typename EF>
__global__ void jaggedVirtualFixLastVariable(
    EF **result,
    const EF *eqzCol,
    const EF *eqzRow,
    EF alpha,
    size_t offset,
    size_t colOffset,
    size_t halfHeight,
    size_t width,
    size_t stackingWidth)
{
    for (size_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x; rowIdx < halfHeight; rowIdx += gridDim.x * blockDim.x)
    {
        size_t evenRowIdx = rowIdx << 1;
        size_t oddRowIdx = (rowIdx << 1) + 1;

        EF eqzRowZero = EF::load(eqzRow, evenRowIdx);
        EF eqzRowOne = EF::load(eqzRow, oddRowIdx);
        for (size_t colIdx = blockIdx.y * blockDim.y + threadIdx.y; colIdx < width; colIdx += gridDim.y * blockDim.y)
        {
            EF eqzColVal = EF::load(eqzCol, colIdx + colOffset);

            EF jaggedValZero = eqzColVal * eqzRowZero;
            EF jaggedValOne = eqzColVal * eqzRowOne;

            // Compute value = zeroValue * (1 - alpha) + oneValue * alpha
            EF value = alpha * (jaggedValOne - jaggedValZero) + jaggedValZero;

            size_t halfOffset = offset / 2;
            size_t baseIdx = colIdx * halfHeight + rowIdx + halfOffset;

            size_t batchIdx = baseIdx / stackingWidth;
            size_t baseIdxInBatch = baseIdx % stackingWidth;
            EF::store(result[batchIdx], baseIdxInBatch, value);
        }
    }
}

extern "C" void *jagged_baby_bear_extension_virtual_fix_last_variable()
{
    return (void *)jaggedVirtualFixLastVariable<bb31_extension_t>;
}


#include "hadamard.cuh"

#include "../fields/kb31_t.cuh"
#include "../fields/kb31_extension_t.cuh"

#include "../reduce/reduction.cuh"

template <typename F, typename EF>
__global__ void hadamardUnivariatePolyEval(
    EF *__restrict__ result,
    const F *__restrict__ base_mle,
    const EF *__restrict__ ext_mle,
    size_t numVariablesMinusOne,
    size_t numPolys)
{
    size_t height = 1 << numVariablesMinusOne;
    size_t inputHeight = height << 1;
    EF evalZero = EF::zero();
    EF evalHalf = EF::zero();
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x)
    {
        for (size_t j = blockDim.y * blockIdx.y + threadIdx.y; j < numPolys; j += blockDim.y * gridDim.y)
        {
            size_t evenIdx = j * inputHeight + (i << 1);
            size_t oddIdx = evenIdx + 1;
            F zeroValBase = F::load(base_mle, evenIdx);
            F oneValBase = F::load(base_mle, oddIdx);
            EF zeroValExt = EF::load(ext_mle, evenIdx);
            EF oneValExt = EF::load(ext_mle, oddIdx);

            evalZero += zeroValExt * zeroValBase;
            evalHalf += (zeroValExt + oneValExt) * (zeroValBase + oneValBase);
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

extern "C" void *hadamard_univariate_poly_eval_koala_bear_base_ext_kernel()
{
    return (void *)hadamardUnivariatePolyEval<kb31_t, kb31_extension_t>;
}

extern "C" void *hadamard_univariate_poly_eval_koala_bear_base_kernel()
{
    return (void *)hadamardUnivariatePolyEval<kb31_t, kb31_t>;
}

extern "C" void *hadamard_univariate_poly_eval_koala_bear_ext_kernel()
{
    return (void *)hadamardUnivariatePolyEval<kb31_extension_t, kb31_extension_t>;
}

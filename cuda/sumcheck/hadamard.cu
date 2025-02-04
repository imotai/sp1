
#include "hadamard.cuh"

#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"

#include "../reduce/reduction.cuh"

template <typename F, typename EF>
__global__ void hadamardUnivariatePolyEval(
    EF *__restrict__ result,
    const F *__restrict__ base_mle,
    const EF *__restrict__ ext_mle,
    size_t numVariablesMinusOne)
{
    size_t height = 1 << numVariablesMinusOne;
    EF evalZero = EF::zero();
    EF evalHalf = EF::zero();
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x)
    {
        F zeroValBase = F::load(base_mle, i << 1);
        F oneValBase = F::load(base_mle, (i << 1) + 1);
        EF zeroValExt = EF::load(ext_mle, i << 1);
        EF oneValExt = EF::load(ext_mle, (i << 1) + 1);

        evalZero += zeroValExt * zeroValBase;
        evalHalf += (zeroValExt + oneValExt) * (zeroValBase + oneValBase);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    EF *shared = reinterpret_cast<EF *>(memory);

    AddOp<EF> op;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    EF evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared, op);
    EF evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared, op);

    EF::store(result, blockIdx.x, evalZeroblockSum);
    EF::store(result, gridDim.x + blockIdx.x, evalHalfblockSum);
}

extern "C" void *hadamard_univariate_poly_eval_baby_bear_base_ext_kernel()
{
    return (void *)hadamardUnivariatePolyEval<bb31_t, bb31_extension_t>;
}

extern "C" void *hadamard_univariate_poly_eval_baby_bear_base_kernel()
{
    return (void *)hadamardUnivariatePolyEval<bb31_t, bb31_t>;
}

extern "C" void *hadamard_univariate_poly_eval_baby_bear_ext_kernel()
{
    return (void *)hadamardUnivariatePolyEval<bb31_extension_t, bb31_extension_t>;
}

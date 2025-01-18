#include "dot.cuh"
#include "reduction.cuh"

template <typename F, typename BaseF>
__global__ void partialBlockInnerProductKernel(
    F *__restrict__ partial,
    F *__restrict__ A,
    BaseF *__restrict__ B,
    size_t width,
    size_t height)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    AddOp<F> op;

    F thread_val = op.initial();

    size_t batchIdx = blockDim.y * blockIdx.y + threadIdx.y;
    if (batchIdx >= width)
    {
        return;
    }

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height;
         i += blockDim.x * gridDim.x)
    {
        op.evalAssign(thread_val, A[batchIdx * height + i] * B[batchIdx * height + i]);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    F *shared = reinterpret_cast<F *>(memory);

    // // Warp-level reduction within tiles
    thread_val = partialBlockReduce(block, tile, thread_val, shared, op);

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0)
    {
        partial[batchIdx * gridDim.x + blockIdx.x] = shared[0];
    }
}

extern "C" void *partial_inner_product_baby_bear_kernel()
{
    return (void *)partialBlockInnerProductKernel<bb31_t, bb31_t>;
}

extern "C" void *partial_inner_product_baby_bear_extension_kernel()
{
    return (void *)partialBlockInnerProductKernel<bb31_extension_t, bb31_extension_t>;
}

extern "C" void *partial_inner_product_baby_bear_base_extension_kernel()
{
    return (void *)partialBlockInnerProductKernel<bb31_extension_t, bb31_t>;
}

template <typename EF, typename F>
__global__ void partialBlockDotKernel(
    EF *__restrict__ partial,
    F *__restrict__ A,
    EF *B,
    size_t width,
    size_t height)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    AddOp<EF> op;

    EF thread_val = op.initial();

    size_t batchIdx = blockDim.y * blockIdx.y + threadIdx.y;
    if (batchIdx >= width)
    {
        return;
    }

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height;
         i += blockDim.x * gridDim.x)
    {
        op.evalAssign(thread_val, B[i] * A[batchIdx * height + i]);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    EF *shared = reinterpret_cast<EF *>(memory);

    // // Warp-level reduction within tiles
    thread_val = partialBlockReduce(block, tile, thread_val, shared, op);

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0)
    {
        partial[batchIdx * gridDim.x + blockIdx.x] = shared[0];
    }
}

extern "C" void *partial_dot_baby_bear_kernel()
{
    return (void *)partialBlockDotKernel<bb31_t, bb31_t>;
}

extern "C" void *partial_dot_baby_bear_extension_kernel()
{
    return (void *)partialBlockDotKernel<bb31_extension_t, bb31_extension_t>;
}

extern "C" void *partial_dot_baby_bear_base_extension_kernel()
{
    return (void *)partialBlockDotKernel<bb31_extension_t, bb31_t>;
}

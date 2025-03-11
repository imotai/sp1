#include "logup_gkr_sum.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"

template<typename K, size_t MEMORY_SIZE>
__global__ void logup_gkr_poly(
    const K *p_0_in,
    const K *p_1_in,
    const bb31_extension_t *q_0_in,
    const bb31_extension_t *q_1_in,
    bb31_extension_t *__restrict__ out,
    bb31_extension_t * eqLagrange,
    bb31_extension_t lambda,
    size_t inputHeight,
    size_t width)
{
    size_t outputHeight = (inputHeight-1) / 2 + 1;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < outputHeight; i += blockDim.x * gridDim.x)
    {
        bb31_extension_t eq = bb31_extension_t::load(eqLagrange, i);
        bool padding = (i == (outputHeight - 1)) && (inputHeight & 1);
        for (size_t j = blockDim.y * blockIdx.y + threadIdx.y; j < width; j += blockDim.y * gridDim.y)
        {
            K p_0_0 = K::load(p_0_in, j * inputHeight + (i << 1));
            K p_0_1 = padding ? K::zero() : K::load(p_0_in, j * inputHeight + (i << 1) + 1);
            K p_1_0 = K::load(p_1_in, j * inputHeight + (i << 1));
            K p_1_1 = padding ? K::zero() : K::load(p_1_in, j * inputHeight + (i << 1) + 1);

            bb31_extension_t q_0_0 = bb31_extension_t::load(q_0_in, j * inputHeight + (i << 1));
            bb31_extension_t q_0_1 = padding ? bb31_extension_t::one() : bb31_extension_t::load(q_0_in, j * inputHeight + (i << 1) + 1);
            bb31_extension_t q_1_0 = bb31_extension_t::load(q_1_in, j * inputHeight + (i << 1));
            bb31_extension_t q_1_1 = padding ? bb31_extension_t::one() : bb31_extension_t::load(q_1_in, j * inputHeight + (i << 1) + 1);


            bb31_extension_t q_0_half = q_0_0 + q_0_1;
            bb31_extension_t q_1_half = q_1_0 + q_1_1;
            K p_0_half = p_0_0 + p_0_1;
            K p_1_half = p_1_0 + p_1_1;

            bb31_extension_t::store(out, j * outputHeight + i, eq * (q_0_0*(lambda * q_1_0 + p_1_0)+ q_1_0*p_0_0));
            bb31_extension_t::store(out, outputHeight * width + j * outputHeight + i , eq * (q_0_half*(lambda * q_1_half + p_1_half)+ q_1_half*p_0_half));
        }
        bb31_extension_t::store(out, 2 * outputHeight * width + i, eq);
    }
}

extern "C" void *logup_gkr_poly_baby_bear_kernel()
{
    return (void *)logup_gkr_poly<bb31_t, 32>;
}

extern "C" void *logup_gkr_poly_baby_bear_extension_kernel()
{
    return (void *)logup_gkr_poly<bb31_extension_t, 32>;
}
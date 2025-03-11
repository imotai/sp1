#include "logup_gkr_circuit.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"

template<typename K, size_t MEMORY_SIZE>
__global__ void gkr_circuit_transition(
    const K *p_in,
    const bb31_extension_t *q_in,
    bb31_extension_t *__restrict__ p_out,
    bb31_extension_t *__restrict__ q_out,
    size_t inputHeight,
    size_t width)
{
    size_t outputHeight = (inputHeight+1) / 2;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < outputHeight; i += blockDim.x * gridDim.x)
    {
        bool padding = (i == (outputHeight - 1)) && (inputHeight & 1);
        for (size_t j = blockDim.y * blockIdx.y + threadIdx.y; j < width; j += blockDim.y * gridDim.y)
        {
            K p_0 = K::load(p_in, j * inputHeight + (i << 1));
            K p_1 = padding ? K::zero() : K::load(p_in, j * inputHeight + (i << 1) + 1);

            bb31_extension_t q_0 = bb31_extension_t::load(q_in, j * inputHeight + (i << 1));
            bb31_extension_t q_1 = padding ? bb31_extension_t::one() : bb31_extension_t::load(q_in, j * inputHeight + (i << 1) + 1);

            bb31_extension_t::store(p_out, j * outputHeight + i, p_0 * q_1 + p_1 * q_0);
            bb31_extension_t::store(q_out, j * outputHeight + i, q_1 * q_0);
        }
    }
}

extern "C" void *gkr_circuit_transition_baby_bear_kernel()
{
    return (void *)gkr_circuit_transition<bb31_t, 32>;
}

extern "C" void *gkr_circuit_transition_baby_bear_extension_kernel()
{
    return (void *)gkr_circuit_transition<bb31_extension_t, 32>;
}
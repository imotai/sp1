#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"
#include "logup_gkr_circuit.cuh"

template <typename K, size_t MEMORY_SIZE>
__global__ void gkr_circuit_transition(const K *p_0_in, const K *p_1_in,
                                       const bb31_extension_t *q_0_in,
                                       const bb31_extension_t *q_1_in,
                                       bb31_extension_t *__restrict__ p_0_out,
                                       bb31_extension_t *__restrict__ p_1_out,
                                       bb31_extension_t *__restrict__ q_0_out,
                                       bb31_extension_t *__restrict__ q_1_out,
                                       size_t inputHeight, size_t width) {
  size_t outputHeight = (inputHeight + 1) / 2;
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < outputHeight;
       i += blockDim.x * gridDim.x) {
    bool padding = inputHeight == 1;
    for (size_t j = blockDim.y * blockIdx.y + threadIdx.y; j < width;
         j += blockDim.y * gridDim.y) {
      K p_0_0 = K::load(p_0_in, j * inputHeight + (i << 1));
      K p_0_1 =
          padding ? K::zero() : K::load(p_0_in, j * inputHeight + (i << 1) + 1);
      K p_1_0 = K::load(p_1_in, j * inputHeight + (i << 1));
      K p_1_1 =
          padding ? K::zero() : K::load(p_1_in, j * inputHeight + (i << 1) + 1);

      bb31_extension_t q_0_0 =
          bb31_extension_t::load(q_0_in, j * inputHeight + (i << 1));
      bb31_extension_t q_0_1 =
          padding
              ? bb31_extension_t::one()
              : bb31_extension_t::load(q_0_in, j * inputHeight + (i << 1) + 1);
      bb31_extension_t q_1_0 =
          bb31_extension_t::load(q_1_in, j * inputHeight + (i << 1));
      bb31_extension_t q_1_1 =
          padding
              ? bb31_extension_t::one()
              : bb31_extension_t::load(q_1_in, j * inputHeight + (i << 1) + 1);

      bb31_extension_t::store(p_0_out, j * outputHeight + i,
                              p_0_0 * q_1_0 + p_1_0 * q_0_0);
      bb31_extension_t::store(p_1_out, j * outputHeight + i,
                               p_0_1 * q_1_1 + p_1_1 * q_0_1);
      bb31_extension_t::store(q_0_out, j * outputHeight + i, q_0_0 * q_1_0);
      bb31_extension_t::store(q_1_out, j * outputHeight + i,
                                       q_0_1 * q_1_1);
    }
  }
}

extern "C" void *gkr_circuit_transition_baby_bear_kernel() {
  return (void *)gkr_circuit_transition<bb31_t, 32>;
}

extern "C" void *gkr_circuit_transition_baby_bear_extension_kernel() {
  return (void *)gkr_circuit_transition<bb31_extension_t, 32>;
}
#include <cuda/atomic>
#include "challenger.cuh"
#include "../poseidon2/poseidon2_bb31_16.cuh"
#include "../poseidon2/poseidon2.cuh"
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"



__global__ void grindKernel(DuplexChallenger challenger, bb31_t *result, size_t bits, size_t n, bool *found_flag) {
    found_flag[0] = false;
    challenger.grind(bits, result, found_flag, n);
}

extern "C" void *grind_baby_bear()
{
    return (void *)grindKernel;
}
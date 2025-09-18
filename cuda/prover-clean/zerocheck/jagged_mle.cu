#include "../config.cuh"
#include "../jagged.cuh"
#include "jagged_mle.cuh"

template <typename F>
__global__ void fixLastVariableJagged(
    const JaggedMle<DenseBuffer<F>> inputJaggedMle,
    JaggedMle<DenseBuffer<ext_t>> outputJaggedMle, 
    uint32_t length,
    ext_t alpha
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length;
         i += blockDim.x * gridDim.x) {
        inputJaggedMle.fixLastVariableTwoPadding(outputJaggedMle, i, alpha);
    }
}

template <typename F> 
__global__ void jaggedEvalChunked(
    const JaggedMle<DenseBuffer<F>> inputJaggedMle,
    const ext_t* __restrict__ row_coefficient, 
    const ext_t* __restrict__ col_coefficient,
    uint32_t L,
    uint32_t num_cols,
    ext_t* __restrict__ output_evals
) {
    inputJaggedMle.evaluate(row_coefficient, col_coefficient, L, num_cols, output_evals);
}

extern "C" void* fix_last_variable_jagged_felt() {
    return (void*)fixLastVariableJagged<felt_t>;
}
extern "C" void* fix_last_variable_jagged_ext() {
    return (void*)fixLastVariableJagged<ext_t>;
}

extern "C" void* jagged_eval_kernel_chunked_felt() {
    return (void*)jaggedEvalChunked<felt_t>;
}
extern "C" void* jagged_eval_kernel_chunked_ext() {
    return (void*)jaggedEvalChunked<ext_t>;
}
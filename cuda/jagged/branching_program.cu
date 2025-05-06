#include "branching_program.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"
#include <cstdio>

// The points are stored in column major order.
template<typename F>
__device__ static inline F getIthLeastSignificantValFromPoints(
    const F *points,
    size_t dim,
    size_t point_idx,
    size_t num_points,
    size_t i)
{
    if (dim <= i) {
        return F::zero();
    } else {
        return points[(dim - i - 1) * num_points + point_idx];
    }
}

template<typename EF>
__device__ static inline EF getIthLeastSignificantVal(
    const EF *point,
    size_t dim,
    size_t i)
{
    if (dim <= i) {
        return EF::zero();
    } else {
        return point[dim - i - 1];
    }
}

template<typename F, typename EF>
__device__ static inline EF getPrefixSumValue(
    int lambda_layer_idx,
    EF lambda,
    Range prefix_sum_layer_idx_range,
    Range rho_layer_idx_range,
    int layer_idx,
    size_t column_idx,
    size_t num_columns,
    const F *prefix_sum,
    size_t prefix_sum_length,
    const EF *rho,
    size_t rho_length)
{
    if (layer_idx == lambda_layer_idx) {
        return lambda;
    } else if (prefix_sum_layer_idx_range.in_range(layer_idx)) {
        return EF(getIthLeastSignificantValFromPoints<F>(
            prefix_sum,
            prefix_sum_length,
            column_idx,
            num_columns,
            layer_idx
        ));
    } else if (rho_layer_idx_range.in_range(layer_idx)) {
        return getIthLeastSignificantVal<EF>(
            rho,
            rho_length,
            layer_idx
        );
    }

    assert(false);
}

template<typename F, typename EF>
__device__ void computePartialLagrange(
    EF point_0,
    EF point_1,
    EF point_2,
    EF point_3,
    EF *output) 
{

    EF point_0_vals[2] = {(EF::one() - point_0), point_0};
    // Memoize the two-variable eq.
    EF two_vals_memoized[4];
    for (uint8_t i = 0; i < 2; i++) {
        EF prod = point_0_vals[i] * point_1;
        two_vals_memoized[i * 2 + 1] = prod;
        two_vals_memoized[i * 2] = point_0_vals[i]-prod;
    }

    // Memoize three-variate eq.
    EF three_vals_memoized[8];
    for (uint8_t i = 0; i < 4; i++) {
        EF prod = two_vals_memoized[i] * point_2;
        three_vals_memoized[i * 2 + 1] = prod;
        three_vals_memoized[i * 2] = two_vals_memoized[i]-prod;
    }

    // Write the output values.
    for (uint8_t i = 0; i < 8; i++) {
        EF prod = three_vals_memoized[i] * point_3;
        output[i * 2 + 1] = prod;
        output[i * 2] = three_vals_memoized[i]-prod;
    }
}

template<typename F, typename EF>
__device__ static inline EF getEqVal(
    size_t lambda_idx,
    size_t column_idx,
    size_t num_columns,
    const F* current_prefix_sums,
    const F* next_prefix_sums,
    size_t prefix_sum_length,
    int curr_prefix_eq_val_least_sig_bit,
    int next_prefix_eq_val_least_sig_bit,
    EF half)
{
    if (lambda_idx == 1) {
        return half;
    }

    assert(lambda_idx == 0);

    if (curr_prefix_eq_val_least_sig_bit != -1) {
        return EF(F::one() - getIthLeastSignificantValFromPoints(
            current_prefix_sums,
            prefix_sum_length,
            column_idx,
            num_columns,
            curr_prefix_eq_val_least_sig_bit));
    }

    if (next_prefix_eq_val_least_sig_bit != -1) {
        return EF(F::one() - getIthLeastSignificantValFromPoints(
            next_prefix_sums,
            prefix_sum_length,
            column_idx,
            num_columns,
            next_prefix_eq_val_least_sig_bit));
    }

    assert(false);
}

template<typename F, typename EF>
__global__ void branchingProgram(
    // The prefix sums.  The current and next prefix sums must be the same length.
    // Note that the number of layers is prefix_sum_length.
    const F *current_prefix_sums,
    const F *next_prefix_sums,
    size_t prefix_sum_length,

    // The z_row and z_index points.
    const EF *z_row,
    size_t z_row_length,
    const EF *z_index,
    size_t z_index_length,

    // The rho values that is set from sumcheck's fix_last_point.
    const EF *current_prefix_sum_rho,
    size_t current_prefix_sum_rho_length,
    const EF *next_prefix_sum_rho,
    size_t next_prefix_sum_rho_length,

    // The total number of columns.
    size_t num_columns,

    // The sumcheck round number.  If this is -1, then that means that none of the prefix sums values
    // should be substituted with the rho or lambda values.
    int8_t round_num,

    // The lambda points.
    const EF *lambdas,

    // The z_col_eq_vals.  This has length num_columns.
    const EF *z_col_eq_vals,

    // The intermediate_eq_full_evals.  This has length num_columns.
    const EF *intermediate_eq_full_evals,

    // The output.
    EF *__restrict__ output
)
{
    int num_layers = static_cast<int>(z_index_length) + 1;

    int curr_prefix_sum_lambda_layer_idx;
    int next_prefix_sum_lambda_layer_idx;
    Range curr_rho_layer_idx_range;
    Range next_rho_layer_idx_range;
    Range curr_prefix_sum_layer_idx_range;
    Range next_prefix_sum_layer_idx_range;
    int curr_prefix_eq_val_least_sig_bit;
    int next_prefix_eq_val_least_sig_bit;

    if (round_num == -1) {
        curr_prefix_eq_val_least_sig_bit = -1;
        next_prefix_eq_val_least_sig_bit = -1;
        curr_prefix_sum_lambda_layer_idx = -1;
        next_prefix_sum_lambda_layer_idx = -1;
        curr_rho_layer_idx_range = Range{-1, -1};
        next_rho_layer_idx_range = Range{-1, -1};
        curr_prefix_sum_layer_idx_range = Range{0, num_layers};
        next_prefix_sum_layer_idx_range = Range{0, num_layers};
    } else if (round_num < prefix_sum_length) {
        curr_prefix_eq_val_least_sig_bit = -1;
        next_prefix_eq_val_least_sig_bit = round_num;
        curr_prefix_sum_lambda_layer_idx = -1;
        next_prefix_sum_lambda_layer_idx = round_num;
        curr_rho_layer_idx_range = Range{-1, -1};
        next_rho_layer_idx_range = Range{0, next_prefix_sum_lambda_layer_idx};
        curr_prefix_sum_layer_idx_range = Range{0, num_layers};
        next_prefix_sum_layer_idx_range = Range{next_prefix_sum_lambda_layer_idx + 1, num_layers};
    } else {  // round_num >= prefix_sum_length
        curr_prefix_eq_val_least_sig_bit = round_num - prefix_sum_length;
        next_prefix_eq_val_least_sig_bit = -1;
        curr_prefix_sum_lambda_layer_idx = round_num - prefix_sum_length;
        next_prefix_sum_lambda_layer_idx = -1;
        curr_rho_layer_idx_range = Range{0, curr_prefix_sum_lambda_layer_idx};
        next_rho_layer_idx_range = Range{0, num_layers};
        curr_prefix_sum_layer_idx_range = Range{curr_prefix_sum_lambda_layer_idx + 1, num_layers};
        next_prefix_sum_layer_idx_range = Range{-1, -1};
    }

    for (size_t column_idx = blockDim.x * blockIdx.x + threadIdx.x; column_idx < num_columns; column_idx += blockDim.x * gridDim.x)
    {
        for (size_t lambda_idx = blockDim.y * blockIdx.y + threadIdx.y; lambda_idx < 2; lambda_idx += blockDim.y * gridDim.y) {
            EF lambda = lambdas[lambda_idx];
            EF state_by_state_results[MEMORY_STATE_COUNT] = {EF::zero(), EF::zero(), EF::zero(), EF::zero(), EF::zero()};
            state_by_state_results[SUCCESS_STATE] = EF::one();

            for (int layer_idx = num_layers - 1; layer_idx >= 0; layer_idx--)
            {
                EF partial_lagrange[16];

                EF point[4] = {
                    getIthLeastSignificantVal<EF>(z_row, z_row_length, layer_idx),
                    getIthLeastSignificantVal<EF>(z_index, z_index_length, layer_idx),
                    getPrefixSumValue<F, EF>(
                        curr_prefix_sum_lambda_layer_idx,
                        lambda,
                        curr_prefix_sum_layer_idx_range,
                        curr_rho_layer_idx_range,
                        layer_idx,
                        column_idx,
                        num_columns,
                        current_prefix_sums,
                        prefix_sum_length,
                        current_prefix_sum_rho,
                        current_prefix_sum_rho_length
                    ),
                    getPrefixSumValue<F, EF>(
                        next_prefix_sum_lambda_layer_idx,
                        lambda,
                        next_prefix_sum_layer_idx_range,
                        next_rho_layer_idx_range,
                        layer_idx,
                        column_idx,
                        num_columns,
                        next_prefix_sums,
                        prefix_sum_length,
                        next_prefix_sum_rho,
                        next_prefix_sum_rho_length
                    ),
                };

                computePartialLagrange<F, EF>(
                    point[0],
                    point[1],
                    point[2],
                    point[3],
                    partial_lagrange
                );

                EF new_state_by_state_results[MEMORY_STATE_COUNT] = {EF::zero(), EF::zero(), EF::zero(), EF::zero(), EF::zero()};

                #pragma unroll
                for (int input_memory_state = 0; input_memory_state < MEMORY_STATE_COUNT; input_memory_state++) {
                    if (input_memory_state == FAIL) {
                        continue;
                    }

                    EF accum_elems[MEMORY_STATE_COUNT] = {EF::zero(), EF::zero(), EF::zero(), EF::zero(), EF::zero()};
                    #pragma unroll
                    for (size_t bit_state = 0; bit_state < BIT_STATE_COUNT; bit_state++) {
                        EF value = partial_lagrange[bit_state];
                        MemoryState output_memory_state = TRANSITIONS[bit_state][input_memory_state];
                        if (output_memory_state != FAIL) {
                            accum_elems[output_memory_state] += value;
                        }
                    }

                    EF accum = EF::zero();
                    #pragma unroll
                    for (int output_memory_state = 0; output_memory_state < MEMORY_STATE_COUNT; output_memory_state++) {
                        accum += accum_elems[output_memory_state] * state_by_state_results[output_memory_state];
                    }

                    new_state_by_state_results[input_memory_state] = accum;
                }

                #pragma unroll
                for (int i = 0; i < MEMORY_STATE_COUNT; i++) {
                    state_by_state_results[i] = new_state_by_state_results[i];
                }
            }

            if (round_num != -1) {
                EF eq_eval = getEqVal<F, EF>(
                    lambda_idx,
                    column_idx,
                    num_columns,
                    current_prefix_sums,
                    next_prefix_sums,
                    prefix_sum_length,
                    curr_prefix_eq_val_least_sig_bit,
                    next_prefix_eq_val_least_sig_bit,
                    lambdas[1]) * intermediate_eq_full_evals[column_idx];
                EF z_col_eq_val = z_col_eq_vals[column_idx];
                EF::store(output, lambda_idx * num_columns + column_idx, state_by_state_results[INITIAL_MEMORY_STATE] * z_col_eq_val * eq_eval);
            } else {
                EF::store(output, lambda_idx * num_columns + column_idx, state_by_state_results[INITIAL_MEMORY_STATE]);
            }
        }
    }
}

__global__ void transition(
    size_t *__restrict__ output
) {
    for (size_t bit_state = 0; bit_state < BIT_STATE_COUNT; bit_state++) {
        for (size_t output_memory_state = 0; output_memory_state < MEMORY_STATE_COUNT; output_memory_state++) {
            output[bit_state * MEMORY_STATE_COUNT + output_memory_state] = TRANSITIONS[bit_state][output_memory_state];
        }
    }
}

extern "C" void *branching_program_kernel()
{
    return (void *)branchingProgram<bb31_t, bb31_extension_t>;
}


extern "C" void *transition_kernel()
{
    return (void *)transition<bb31_t, bb31_extension_t>;
}
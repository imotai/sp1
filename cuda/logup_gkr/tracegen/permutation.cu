#include "interaction.cuh"

#include "../../fields/bb31_extension_t.cuh"
#include "../../fields/bb31_t.cuh"

template <typename T, typename S> struct GKRInput {
  T numer;
  S denom;
};

template <typename F, typename EF>
__device__ __forceinline__ GKRInput<F, EF>
InteractionValue(size_t i, size_t RowIdx, Interactions<F> const interactions,
                 F *const preprocessed, F *const main, EF const alpha,
                 EF const beta, size_t height) {
  // Calculate the interaction index.
  size_t index = i;

  // Initialize the denominator and beta powers.
  EF denominator = alpha;
  EF beta_power = EF::one();

  // Add argument index to the denominator.
  EF argument_index = EF(interactions.arg_indices[index]);
  denominator += beta_power * argument_index;

  // Add the interaction values.
  for (size_t k = interactions.values_ptr[index];
       k < interactions.values_ptr[index + 1]; k++) {
    beta_power *= beta;
    EF acc = EF(interactions.values_constants[k]);
    for (size_t l = interactions.values_col_weights_ptr[k];
         l < interactions.values_col_weights_ptr[k + 1]; l++) {
      acc += EF(interactions.values_col_weights[l].get(preprocessed, main,
                                                       RowIdx, height));
    }
    denominator += beta_power * acc;
  }

  // Calculate the multiplicity values.
  bool is_send = interactions.is_send[index];
  F mult = interactions.mult_constants[index];

  for (size_t k = interactions.multiplicities_ptr[index];
       k < interactions.multiplicities_ptr[index + 1]; k++) {
    mult += interactions.mult_col_weights[k].get(preprocessed, main, RowIdx,
                                                 height);
  }

  if (!is_send) {
    mult = F::zero() - mult;
  }

  GKRInput value = {mult, denominator};

  return value;
}

template <typename F, typename EF>
__global__ void
PopulatePermutationRowsFlattened(Interactions<F> const interactions, F *numer_0,
                                 EF *denom_0, F *numer_1, EF *denom_1,
                                 F *const preprocessed, F *const main, EF alpha,
                                 EF beta, size_t height) {
  size_t RowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (2 * RowIdx >= height) {
    return;
  }

  size_t num_interactions = interactions.num_interactions;
  for (size_t i = 0; i < num_interactions; i++) {
    GKRInput even_result = InteractionValue(
        i, 2 * RowIdx, interactions, preprocessed, main, alpha, beta, height);
    GKRInput odd_result =
        InteractionValue(i, 2 * RowIdx + 1, interactions, preprocessed, main,
                         alpha, beta, height);
    // Assign the value to the extension field slot in the permutation trace.
    numer_0[i * (height / 2) + RowIdx] = even_result.numer;
    denom_0[i * (height / 2) + RowIdx] = even_result.denom;
    numer_1[i * (height / 2) + RowIdx] = odd_result.numer;
    denom_1[i * (height / 2) + RowIdx] = odd_result.denom;
  }
}

extern "C" void *gkr_tracegen_kernel() {
  return (void *)PopulatePermutationRowsFlattened<bb31_t, bb31_extension_t>;
}
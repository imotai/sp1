#include <stdint.h>

#include "execution.cuh"
#include "first_layer.cuh"
#include "interaction.cuh"
#include "round.cuh"

#include "../jagged.cuh"
#include "../config.cuh"

__global__ void proverCleanLogUpCircuitTransition(
    const JaggedMle<JaggedGkrLayer> inputJaggedMle,
    JaggedMle<JaggedGkrLayer> outputJaggedMle) {

    size_t height = inputJaggedMle.denseData.height;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height;
         i += blockDim.x * gridDim.x) {
        circuitTransitionTwoPadding(inputJaggedMle, outputJaggedMle, i);
    }
}

template <typename F, typename EF>
struct GkrInput {
    F numerator;
    EF denominator;

    __device__ __forceinline__ static GkrInput<F, EF> padding() {
        GkrInput<F, EF> values;
        values.numerator = F::zero();
        values.denominator = EF::one();
        return values;
    }
};

template <typename F, typename EF>
__device__ __forceinline__ GkrInput<F, EF> InteractionValue(
    size_t index,
    size_t rowIdx,
    Interactions<F> const interactions,
    F* const preprocessed,
    F* const main,
    EF const alpha,
    EF const beta,
    size_t height) {
    // Initialize the denominator and beta powers.
    EF denominator = alpha;
    EF beta_power = EF::one();

    // Add argument index to the denominator.
    EF argument_index = EF(interactions.arg_indices[index]);
    denominator += beta_power * argument_index;

    // Add the interaction values.
    for (size_t k = interactions.values_ptr[index]; k < interactions.values_ptr[index + 1]; k++) {
        beta_power *= beta;
        EF acc = EF(interactions.values_constants[k]);
        for (size_t l = interactions.values_col_weights_ptr[k];
             l < interactions.values_col_weights_ptr[k + 1];
             l++) {
            acc += EF(interactions.values_col_weights[l].get(preprocessed, main, rowIdx, height));
        }
        denominator += beta_power * acc;
    }

    // Calculate the multiplicity values.
    bool is_send = interactions.is_send[index];
    F mult = interactions.mult_constants[index];

    for (size_t k = interactions.multiplicities_ptr[index];
         k < interactions.multiplicities_ptr[index + 1];
         k++) {
        mult += interactions.mult_col_weights[k].get(preprocessed, main, rowIdx, height);
    }

    if (!is_send) {
        mult = F::zero() - mult;
    }

    GkrInput<F, EF> value;
    value.numerator = mult;
    value.denominator = denominator;

    return value;
}

// TODO: This has not been tested or refactored, since the input structure will change significantly
// soon.
template <typename F, typename EF>
__global__ void proverCleanPopulateLastCircuitLayer(
    Interactions<F> interactions,
    const uint32_t* startIndices,
    uint32_t* colIndex,
    F* numeratorValues,
    EF* denominatorValues,
    F* const preprocessed,
    F* const main,
    EF alpha,
    EF beta,
    size_t interactionOffset,
    size_t traceHeight,
    size_t halfTraceHeight,
    size_t outputHeight,
    size_t dimension,
    bool is_padding) {
    size_t numInteractions = interactions.num_interactions;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < halfTraceHeight;
         i += blockDim.x * gridDim.x) {
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;
        size_t parity = i & 1;

        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < numInteractions;
             j += blockDim.y * gridDim.y) {
            size_t colIdx = j + interactionOffset;
            size_t startIdx = startIndices[colIdx];

            size_t restrictedIndex = startIdx + i;

            if (is_padding) {
                FirstLayerCircuitValues values = FirstLayerCircuitValues::paddingValues();
                values.store(numeratorValues, denominatorValues, restrictedIndex, outputHeight);
                values.store(numeratorValues, denominatorValues, restrictedIndex + 1, outputHeight);
            } else {
                GkrInput<F, EF> zeroValue;
                GkrInput<F, EF> oneValue;
                zeroValue = InteractionValue(
                    j,
                    zeroIdx,
                    interactions,
                    preprocessed,
                    main,
                    alpha,
                    beta,
                    traceHeight);
                oneValue = InteractionValue(
                    j,
                    oneIdx,
                    interactions,
                    preprocessed,
                    main,
                    alpha,
                    beta,
                    traceHeight);
                FirstLayerCircuitValues values;
                values.numeratorZero = zeroValue.numerator;
                values.numeratorOne = oneValue.numerator;
                values.denominatorZero = zeroValue.denominator;
                values.denominatorOne = oneValue.denominator;
                values.store(numeratorValues, denominatorValues, restrictedIndex, outputHeight);
            }

            // Write the output interaction data and dimension. Do it only once per
            // pair of points.
            if (parity == 0) {
                uint32_t outputInteractionValue = colIdx + (dimension << 24);
                colIndex[restrictedIndex >> 1] = outputInteractionValue;
            }
        }
    }
}

__global__ void proverCleanExtractOutput(
    const JaggedMle<JaggedGkrLayer> inputJaggedMle,
    ext_t* __restrict__ numerator,
    ext_t* __restrict__ denominator,
    size_t gridHeight) {

    size_t height = inputJaggedMle.denseData.height;
    uint32_t* colIndex = inputJaggedMle.colIndex;
    uint32_t* startIndices = inputJaggedMle.startIndices;
    ext_t* layer = inputJaggedMle.denseData.layer;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < gridHeight;
         i += blockDim.x * gridDim.x) {

        CircuitValues values;
        // At this point, every other element is padding. So we need to combine every other pair of
        // points.
        if (i << 1 < height) {

            size_t zeroIdx = i << 2;
            size_t oneIdx = (i << 2) + 1;
            size_t colIdx = colIndex[i];

            CircuitValues valuesZero = CircuitValues::load(layer, zeroIdx, height);
            values.numeratorZero = valuesZero.numeratorZero * valuesZero.denominatorOne +
                                   valuesZero.numeratorOne * valuesZero.denominatorZero;

            values.denominatorZero = valuesZero.denominatorZero * valuesZero.denominatorOne;

            CircuitValues valuesOne = CircuitValues::load(layer, oneIdx, height);
            values.numeratorOne = valuesOne.numeratorZero * valuesOne.denominatorOne +
                                  valuesOne.numeratorOne * valuesOne.denominatorZero;
            values.denominatorOne = valuesOne.denominatorZero * valuesOne.denominatorOne;
        } else {
            values = CircuitValues::paddingValues();
        }

        // Store the values in the output MLEs
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;

        ext_t::store(numerator, zeroIdx, values.numeratorZero);
        ext_t::store(numerator, oneIdx, values.numeratorOne);
        ext_t::store(denominator, zeroIdx, values.denominatorZero);
        ext_t::store(denominator, oneIdx, values.denominatorOne);
    }
}

extern "C" void* prover_clean_logup_gkr_circuit_transition() {
    return (void*)proverCleanLogUpCircuitTransition;
}

extern "C" void* prover_clean_logup_gkr_populate_last_circuit_layer() {
    return (void*)proverCleanPopulateLastCircuitLayer<felt_t, ext_t>;
}


extern "C" void* prover_clean_logup_gkr_extract_output() { return (void*)proverCleanExtractOutput; }
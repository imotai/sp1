#include <stdint.h>

#include "execution.cuh"
#include "round.cuh"
#include "interaction.cuh"
#include "first_layer.cuh"

#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"

template <typename EF>
__global__ void logUpCircuitTransitionKernel(
    const EF *__restrict__ layer,
    const uint32_t *__restrict__ interactionData,
    const uint32_t *__restrict__ startIndices,
    EF *__restrict__ outputLayer,
    uint32_t *__restrict__ outputInteractionData,
    const uint32_t *__restrict__ nextLayerStartIndices,
    const size_t height,
    const size_t outputHeight)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x)
    {
        size_t interactionIdx = interactionData[i] & 0x00FFFFFF;
        size_t dimension = interactionData[i] >> 24;

        size_t rowIdx = i - startIndices[interactionIdx];

        CircuitValues<EF> values;

        CircuitValues<EF> valuesZero = CircuitValues<EF>::load(layer, i, 0UL, height);
        values.numeratorZero = valuesZero.numeratorZero * valuesZero.denominatorOne + valuesZero.numeratorOne * valuesZero.denominatorZero;
        values.denominatorZero = valuesZero.denominatorZero * valuesZero.denominatorOne;

        CircuitValues<EF> valuesOne = CircuitValues<EF>::load(layer, i, 1UL, height);
        values.numeratorOne = valuesOne.numeratorZero * valuesOne.denominatorOne + valuesOne.numeratorOne * valuesOne.denominatorZero;
        values.denominatorOne = valuesOne.denominatorZero * valuesOne.denominatorOne;

        // Store the restricted values
        size_t parity = rowIdx & 1;
        size_t restrictedRowIdx = rowIdx >> 1;
        size_t restrictedIndex = nextLayerStartIndices[interactionIdx] + restrictedRowIdx;
        // Store the restricted values
        values.store(outputLayer, restrictedIndex, parity, outputHeight);

        uint32_t outputDimension;
        // If the dimension is 0, we have exausted the real variables and therefore in the padding
        // region, thus we need to account for the value at 1 to be equal to the padding value.
        if (dimension == 1)
        {
            outputDimension = 1;
            CircuitValues<EF> paddingValues = CircuitValues<EF>::paddingValues();
            paddingValues.store(outputLayer, restrictedIndex, 1U, outputHeight);
        }
        else
        {
            outputDimension = dimension - 1;
        }

        // Write the output interaction data and dimension. Do it only once per pair of points.
        if (parity == 0)
        {
            uint32_t outputInteractionValue = interactionIdx + (outputDimension << 24);
            outputInteractionData[restrictedIndex] = outputInteractionValue;
        }
    }
}

extern "C" void *logup_gkr_circuit_transition_baby_bear_extension()
{
    return (void *)logUpCircuitTransitionKernel<bb31_extension_t>;
}

template <typename F, typename EF>
struct GkrInput
{
    F numerator;
    EF denominator;

    __device__ __forceinline__ static GkrInput<F, EF> padding()
    {
        GkrInput<F, EF> values;
        values.numerator = F::zero();
        values.denominator = EF::one();
        return values;
    }
};

template <typename F, typename EF>
__device__ __forceinline__ GkrInput<F, EF>
InteractionValue(size_t index, size_t rowIdx, Interactions<F> const interactions,
                 F *const preprocessed, F *const main, EF const alpha,
                 EF const beta, size_t height)
{
    // Initialize the denominator and beta powers.
    EF denominator = alpha;
    EF beta_power = EF::one();

    // Add argument index to the denominator.
    EF argument_index = EF(interactions.arg_indices[index]);
    denominator += beta_power * argument_index;

    // Add the interaction values.
    for (size_t k = interactions.values_ptr[index];
         k < interactions.values_ptr[index + 1]; k++)
    {
        beta_power *= beta;
        EF acc = EF(interactions.values_constants[k]);
        for (size_t l = interactions.values_col_weights_ptr[k];
             l < interactions.values_col_weights_ptr[k + 1]; l++)
        {
            acc += EF(interactions.values_col_weights[l].get(preprocessed, main,
                                                             rowIdx, height));
        }
        denominator += beta_power * acc;
    }

    // Calculate the multiplicity values.
    bool is_send = interactions.is_send[index];
    F mult = interactions.mult_constants[index];

    for (size_t k = interactions.multiplicities_ptr[index];
         k < interactions.multiplicities_ptr[index + 1]; k++)
    {
        mult += interactions.mult_col_weights[k].get(preprocessed, main, rowIdx,
                                                     height);
    }

    if (!is_send)
    {
        mult = F::zero() - mult;
    }

    GkrInput<F, EF> value;
    value.numerator = mult;
    value.denominator = denominator;

    return value;
}

template <typename F, typename EF>
__global__ void populateLastCircuitLayer(
    Interactions<F> interactions,
    const uint32_t *startIndices,
    uint32_t *interactionData,
    F *numeratorValues,
    EF *denominatorValues,
    F *const preprocessed,
    F *const main,
    EF alpha,
    EF beta,
    size_t interactionOffset,
    size_t traceHeight,
    size_t halfTraceHeight,
    size_t outputHeight,
    size_t dimension,
    bool is_padding)
{
    size_t numInteractions = interactions.num_interactions;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < halfTraceHeight; i += blockDim.x * gridDim.x)
    {
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;
        size_t parity = i & 1;
        size_t restrictedRowIdx = i >> 1;

        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < numInteractions; j += blockDim.y * gridDim.y)
        {
            size_t interactionIdx = j + interactionOffset;
            size_t startIdx = startIndices[interactionIdx];

            size_t restrictedIndex = startIdx + restrictedRowIdx;

            if (is_padding)
            {
                FirstLayerCircuitValues<F, EF> values = FirstLayerCircuitValues<F, EF>::paddingValues();
                values.store(numeratorValues, denominatorValues, restrictedIndex, 0UL, outputHeight);
                values.store(numeratorValues, denominatorValues, restrictedIndex, 1UL, outputHeight);
            }
            else
            {
                GkrInput<F, EF> zeroValue;
                GkrInput<F, EF> oneValue;
                zeroValue = InteractionValue(j, zeroIdx, interactions, preprocessed,
                                             main, alpha, beta, traceHeight);
                oneValue = InteractionValue(j, oneIdx, interactions, preprocessed,
                                            main, alpha, beta, traceHeight);
                FirstLayerCircuitValues<F, EF> values;
                values.numeratorZero = zeroValue.numerator;
                values.numeratorOne = oneValue.numerator;
                values.denominatorZero = zeroValue.denominator;
                values.denominatorOne = oneValue.denominator;
                values.store(numeratorValues, denominatorValues, restrictedIndex, parity, outputHeight);
            }

            // Write the output interaction data and dimension. Do it only once per pair of points.
            if (parity == 0)
            {
                uint32_t outputInteractionValue = interactionIdx + (dimension << 24);
                interactionData[restrictedIndex] = outputInteractionValue;
            }
        }
    }
}

extern "C" void *logup_gkr_populate_last_circuit_layer_baby_bear()
{
    return (void *)populateLastCircuitLayer<bb31_t, bb31_extension_t>;
}

template <typename EF>
__global__ void extractOutputKernel(
    const EF *__restrict__ layer,
    const uint32_t *__restrict__ interactionData,
    const uint32_t *__restrict__ startIndices,
    EF *__restrict__ numerator,
    EF *__restrict__ denominator,
    const size_t height,
    size_t gridHeight)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < gridHeight; i += blockDim.x * gridDim.x)
    {
        CircuitValues<EF> values;
        if (i < height)
        {
            size_t interactionIdx = interactionData[i] & 0x00FFFFFF;

            CircuitValues<EF> valuesZero = CircuitValues<EF>::load(layer, i, 0UL, height);
            values.numeratorZero = valuesZero.numeratorZero * valuesZero.denominatorOne + valuesZero.numeratorOne * valuesZero.denominatorZero;
            values.denominatorZero = valuesZero.denominatorZero * valuesZero.denominatorOne;

            CircuitValues<EF> valuesOne = CircuitValues<EF>::load(layer, i, 1UL, height);
            values.numeratorOne = valuesOne.numeratorZero * valuesOne.denominatorOne + valuesOne.numeratorOne * valuesOne.denominatorZero;
            values.denominatorOne = valuesOne.denominatorZero * valuesOne.denominatorOne;
        }
        else
        {
            values = CircuitValues<EF>::paddingValues();
        }

        // Store the values in the output MLEs
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;
        numerator[zeroIdx] = values.numeratorZero;
        numerator[oneIdx] = values.numeratorOne;
        denominator[zeroIdx] = values.denominatorZero;
        denominator[oneIdx] = values.denominatorOne;
    }
}

extern "C" void *logup_gkr_extract_output_baby_bear()
{
    return (void *)extractOutputKernel<bb31_extension_t>;
}
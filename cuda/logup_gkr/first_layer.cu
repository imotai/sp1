#include "first_layer.cuh"
#include "round.cuh"

#include "../reduce/reduction.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"

template <typename F, typename EF>
__global__ void logupGkrFixLastVariableFirstCircuitLayer(
    const F *__restrict__ inputNumerator,
    const EF *__restrict__ inputDenominator,
    const uint32_t *__restrict__ interactionData,
    const uint32_t *__restrict__ startIndices,
    EF alpha,
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

        FirstLayerCircuitValues<F, EF> valuesZero = FirstLayerCircuitValues<F, EF>::load(inputNumerator, inputDenominator, i, 0UL, height);
        FirstLayerCircuitValues<F, EF> valuesOne = FirstLayerCircuitValues<F, EF>::load(inputNumerator, inputDenominator, i, 1UL, height);
        CircuitValues<EF> values = FirstLayerCircuitValues<F, EF>::fix_last_variable(valuesZero, valuesOne, alpha);
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

extern "C" void *logup_gkr_fix_last_variable_first_layer_kernel_baby_bear()
{
    return (void *)logupGkrFixLastVariableFirstCircuitLayer<bb31_t, bb31_extension_t>;
}

template <typename F, typename EF>
__global__ void logupGkrSumAsPolyFirstCircuitLayer(
    EF *__restrict__ result,
    const F *__restrict__ inputNumerator,
    const EF *__restrict__ inputDenominator,
    const uint32_t *__restrict__ interactionData,
    const uint32_t *__restrict__ startIndices,
    const EF *__restrict__ eqRow,
    const EF *__restrict__ eqInteraction,
    const EF lambda,
    const size_t height)
{
    EF evalZero = EF::zero();
    EF evalHalf = EF::zero();
    EF eqSum = EF::zero();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x)
    {
        // The interaction data is a 32 bit integer such that the first 24 bits are the interaction
        // index and the last 8 bits are the dimension
        size_t interactionIdx = interactionData[i] & 0x00FFFFFF;

        // The start indices are determined by the interaction index and the start indices array
        size_t startIdx = startIndices[interactionIdx];
        // The row is simply the current index minus the start index
        size_t rowIdx = i - startIdx;

        size_t eqRowZeroIdx = rowIdx << 1;
        size_t eqRowOneIdx = eqRowZeroIdx + 1;
        EF eqValueZero = eqRow[eqRowZeroIdx] * eqInteraction[interactionIdx];
        EF eqValueOne = eqRow[eqRowOneIdx] * eqInteraction[interactionIdx];
        EF eqValueHalf = eqValueZero + eqValueOne;

        // Add the eqValue to the running aggregate
        eqSum += eqValueHalf;

        // Load the numerator and denominator values
        FirstLayerCircuitValues<F, EF> valuesZero = FirstLayerCircuitValues<F, EF>::load(inputNumerator, inputDenominator, i, 0UL, height);
        FirstLayerCircuitValues<F, EF> valuesOne = FirstLayerCircuitValues<F, EF>::load(inputNumerator, inputDenominator, i, 1UL, height);

        // Compute the values at the point 1 /2 (times a factor of 2)
        FirstLayerCircuitValues<F, EF> valuesHalf;
        valuesHalf.numeratorZero = valuesZero.numeratorZero + valuesOne.numeratorZero;
        valuesHalf.numeratorOne = valuesZero.numeratorOne + valuesOne.numeratorOne;
        valuesHalf.denominatorZero = valuesZero.denominatorZero + valuesOne.denominatorZero;
        valuesHalf.denominatorOne = valuesZero.denominatorOne + valuesOne.denominatorOne;

        // Compute the sumcheck sum values and add to the running aggregate
        evalZero += valuesZero.sumAsPoly(lambda, eqValueZero);
        evalHalf += valuesHalf.sumAsPoly(lambda, eqValueHalf);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    EF *shared = reinterpret_cast<EF *>(memory);

    AddOp<EF> op;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    EF evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared, op);
    EF evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared, op);
    EF eqSumBlockSum = partialBlockReduce(block, tile, eqSum, shared, op);

    if (threadIdx.x == 0)
    {
        EF::store(result, blockIdx.x, evalZeroblockSum);
        EF::store(result, gridDim.x + blockIdx.x, evalHalfblockSum);
        EF::store(result, 2 * gridDim.x + blockIdx.x, eqSumBlockSum);
    }
}

extern "C" void *logup_gkr_sum_as_poly_first_layer_kernel_baby_bear()
{
    return (void *)logupGkrSumAsPolyFirstCircuitLayer<bb31_t, bb31_extension_t>;
}

template <typename F, typename EF>
__global__ void logUpFirstLayerTransitionKernel(
    const F *__restrict__ inputNumerator,
    const EF *__restrict__ inputDenominator,
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

        FirstLayerCircuitValues<F, EF> valuesZero = FirstLayerCircuitValues<F, EF>::load(inputNumerator, inputDenominator, i, 0UL, height);
        values.numeratorZero = valuesZero.numeratorZero * valuesZero.denominatorOne + valuesZero.numeratorOne * valuesZero.denominatorZero;
        values.denominatorZero = valuesZero.denominatorZero * valuesZero.denominatorOne;

        FirstLayerCircuitValues<F, EF> valuesOne = FirstLayerCircuitValues<F, EF>::load(inputNumerator, inputDenominator, i, 1UL, height);
        values.numeratorOne = (valuesOne.denominatorOne * valuesOne.numeratorZero) + (valuesOne.denominatorZero * valuesOne.numeratorOne);
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

extern "C" void *logup_gkr_first_layer_transition_baby_bear()
{
    return (void *)logUpFirstLayerTransitionKernel<bb31_t, bb31_extension_t>;
}
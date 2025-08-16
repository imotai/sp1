#include <stdint.h>
#include "round.cuh"

#include "../reduce/reduction.cuh"

#include "../fields/kb31_extension_t.cuh"
#include "../fields/kb31_t.cuh"

template <typename EF>
__global__ void logupGkrFixLastVariableCircuitLayer(
    EF *__restrict__ layer,
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

        CircuitValues<EF> valuesZero = CircuitValues<EF>::load(layer, i, 0UL, height);
        CircuitValues<EF> valuesOne = CircuitValues<EF>::load(layer, i, 1UL, height);
        CircuitValues<EF> values = CircuitValues<EF>::fix_last_variable(valuesZero, valuesOne, alpha);
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



        // if (oddNumRows[interactionIdx]) {
        //     printf("InteractionIdx = %d\n", interactionIdx);
        // }
        size_t interactionHeight = (startIndices[interactionIdx + 1] - startIndices[interactionIdx]);

        size_t isOdd = interactionHeight & 1;

        bool isLast = (interactionHeight  -1) == rowIdx;
        

        if ((isOdd==1) && isLast)
        {
            // If the number of rows is odd, we need to set the last row to the padding value
            CircuitValues<EF> paddingValues = CircuitValues<EF>::paddingValues();
            paddingValues.store(outputLayer, restrictedIndex, 1U, outputHeight);
        }

        // Write the output interaction data and dimension. Do it only once per pair of points.
        if (parity == 0)
        {
            uint32_t outputInteractionValue = interactionIdx + (outputDimension << 24);
            outputInteractionData[restrictedIndex] = outputInteractionValue;
        }
    }
}

extern "C" void *logup_gkr_fix_last_variable_circuit_layer_kernel_koala_bear_extension()
{
    return (void *)logupGkrFixLastVariableCircuitLayer<kb31_extension_t>;
}

template <typename EF>
__global__ void logupGkrSumAsPolyCircuitLayer(
    EF *__restrict__ result,
    const EF *__restrict__ layer,
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
        CircuitValues<EF> valuesZero = CircuitValues<EF>::load(layer, i, 0UL, height);
        CircuitValues<EF> valuesOne = CircuitValues<EF>::load(layer, i, 1UL, height);

        // Compute the values at the point 1 /2 (times a factor of 2)
        CircuitValues<EF> valuesHalf;
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

extern "C" void *logup_gkr_sum_as_poly_layer_kernel_circuit_layer_koala_bear_extension()
{
    return (void *)logupGkrSumAsPolyCircuitLayer<kb31_extension_t>;
}

template <typename EF>
__global__ void FixLastVariableLastCircuitLayer(
    const EF *__restrict__ layer,
    EF alpha,
    EF *__restrict__ output,
    const size_t height)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x)
    {
        CircuitValues<EF> valuesZero = CircuitValues<EF>::load(layer, i, 0, height);
        CircuitValues<EF> valuesOne = CircuitValues<EF>::load(layer, i, 1, height);
        CircuitValues<EF> values = CircuitValues<EF>::fix_last_variable(valuesZero, valuesOne, alpha);

        // Store the restricted values
        output[i] = values.numeratorZero;
        output[height + i] = values.numeratorOne;
        output[2 * height + i] = values.denominatorZero;
        output[3 * height + i] = values.denominatorOne;
    }
}

extern "C" void *logup_gkr_fix_last_row_last_circuit_layer_kernel_circuit_layer_koala_bear_extension()
{
    return (void *)FixLastVariableLastCircuitLayer<kb31_extension_t>;
}

template <typename EF>
__global__ void partialSumAsPolyInteractionsLayer(
    EF *__restrict__ result,
    EF *__restrict__ layer,
    const EF *__restrict__ eqPoly,
    const EF lambda,
    const size_t height,
    const size_t outputHeight)
{
    EF evalZero = EF::zero();
    EF evalHalf = EF::zero();
    EF eqSum = EF::zero();
    bool padding = height & 1;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outputHeight; i += blockDim.x * gridDim.x)
    {
        // The indices for the values at (i, 0) and (i, 1)
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;

        // Load zero values
        CircuitValues<EF> valuesZero;
        valuesZero.numeratorZero = layer[zeroIdx];
        valuesZero.numeratorOne = layer[height + zeroIdx];
        valuesZero.denominatorZero = layer[2 * height + zeroIdx];
        valuesZero.denominatorOne = layer[3 * height + zeroIdx];

        // Load one values
        CircuitValues<EF> valuesOne;
        if (padding && i == outputHeight - 1)
        {
            valuesOne = CircuitValues<EF>::paddingValues();
        }
        else
        {
            valuesOne.numeratorZero = layer[oneIdx];
            valuesOne.numeratorOne = layer[height + oneIdx];
            valuesOne.denominatorZero = layer[2 * height + oneIdx];
            valuesOne.denominatorOne = layer[3 * height + oneIdx];
        }

        // Compute the values at one half
        CircuitValues<EF> valuesHalf;
        valuesHalf.numeratorZero = valuesZero.numeratorZero + valuesOne.numeratorZero;
        valuesHalf.numeratorOne = valuesZero.numeratorOne + valuesOne.numeratorOne;
        valuesHalf.denominatorZero = valuesZero.denominatorZero + valuesOne.denominatorZero;
        valuesHalf.denominatorOne = valuesZero.denominatorOne + valuesOne.denominatorOne;

        // Load the eq value
        EF eqValueZero = eqPoly[zeroIdx];
        EF eqValueOne = eqPoly[oneIdx];
        EF eqValueHalf = eqValueZero + eqValueOne;
        // Add the eq value to the running aggregate
        eqSum += eqValueHalf;

        // Compute the evaluations of the sumcheck polynomial at zero and one half and add to the
        // running aggregate
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

extern "C" void *logup_gkr_sum_as_poly_layer_kernel_interactions_layer_koala_bear_extension()
{
    return (void *)partialSumAsPolyInteractionsLayer<kb31_extension_t>;
}

template <typename EF>
__global__ void fixLastVariableInteractionsLayer(
    const EF *input,
    EF *__restrict__ output,
    EF alpha,
    size_t height,
    size_t outputHeight)
{
    bool padding = height & 1;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < outputHeight; i += blockDim.x * gridDim.x)
    {
        // The indices for the values at (i, 0) and (i, 1)
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;

        // Load zero values
        CircuitValues<EF> valuesZero;
        valuesZero.numeratorZero = input[zeroIdx];
        valuesZero.numeratorOne = input[height + zeroIdx];
        valuesZero.denominatorZero = input[2 * height + zeroIdx];
        valuesZero.denominatorOne = input[3 * height + zeroIdx];

        // Load one values
        CircuitValues<EF> valuesOne;
        if (padding && i == outputHeight - 1)
        {
            valuesOne = CircuitValues<EF>::paddingValues();
        }
        else
        {
            valuesOne.numeratorZero = input[oneIdx];
            valuesOne.numeratorOne = input[height + oneIdx];
            valuesOne.denominatorZero = input[2 * height + oneIdx];
            valuesOne.denominatorOne = input[3 * height + oneIdx];
        }

        CircuitValues<EF> values = CircuitValues<EF>::fix_last_variable(valuesZero, valuesOne, alpha);

        // Store the restricted values
        output[i] = values.numeratorZero;
        output[outputHeight + i] = values.numeratorOne;
        output[2 * outputHeight + i] = values.denominatorZero;
        output[3 * outputHeight + i] = values.denominatorOne;
    }
}

extern "C" void *logup_gkr_fix_last_variable_interactions_layer_kernel_koala_bear_extension()
{
    return (void *)fixLastVariableInteractionsLayer<kb31_extension_t>;
}

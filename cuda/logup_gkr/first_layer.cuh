
#pragma once

#include "round.cuh"

extern "C" void *logup_gkr_fix_last_variable_first_layer_kernel_baby_bear();
extern "C" void *logup_gkr_sum_as_poly_first_layer_kernel_baby_bear();
extern "C" void *logup_gkr_first_layer_transition_baby_bear();

template <typename F, typename EF>
struct FirstLayerCircuitValues
{
public:
    F numeratorZero;
    F numeratorOne;
    EF denominatorZero;
    EF denominatorOne;

public:
    static __device__ __forceinline__ FirstLayerCircuitValues<F, EF> load(
        F *numeratorValues,
        EF *denominatorValues,
        size_t i,
        size_t parity,
        size_t height)
    {
        FirstLayerCircuitValues<F, EF> values;
        // Load the numerator and denominator values
        //  numerator[i, b, 0] = numeratorValues[0, b, i]
        //  numerator[i, b, 1] = numeratorValues[1, b, i]
        //  denominator[i, b, 0] = denominatorValues[0, b, i]
        //  denominator[i, b, 1] = denominatorValues[1, b, i]
        //
        // Indexing into the layer tensor is done as follows:
        //
        //  layer[k, parity, n] = layer_ptr[k * 2 * height + parity * height + n]

        //  numerator[i, b, 0] = layer[0, b, i] = layer_ptr[0 * 2 * height + b * height + i]
        values.numeratorZero = numeratorValues[parity * height + i];
        //  numerator[i, b, 1] = layer[1, b, i] = layer_ptr[1 * 2 * height + b * height + i]
        values.numeratorOne = numeratorValues[2 * height + parity * height + i];

        //  denominator[i, b, 0] = layer[2, b, i] = layer_ptr[2 * 2 * height + b * height + i]
        values.denominatorZero = denominatorValues[parity * height + i];
        //  denominator[i, b, 1] = layer[3, b, i] = layer_ptr[3 * 2 * height + b * height + i]
        values.denominatorOne = denominatorValues[2 * height + parity * height + i];

        return values;
    }

    static __device__ __forceinline__ FirstLayerCircuitValues<F, EF> load(
        const F *numeratorValues,
        const EF *denominatorValues,
        size_t i,
        size_t parity,
        size_t height)
    {
        FirstLayerCircuitValues<F, EF> values;
        // Load the numerator and denominator values
        //  numerator[i, b, 0] = numeratorValues[0, b, i]
        //  numerator[i, b, 1] = numeratorValues[1, b, i]
        //  denominator[i, b, 0] = denominatorValues[0, b, i]
        //  denominator[i, b, 1] = denominatorValues[1, b, i]
        //
        // Indexing into the layer tensor is done as follows:
        //
        //  layer[k, parity, n] = layer_ptr[k * 2 * height + parity * height + n]

        //  numerator[i, b, 0] = layer[0, b, i] = layer_ptr[0 * 2 * height + b * height + i]
        values.numeratorZero = numeratorValues[parity * height + i];
        //  numerator[i, b, 1] = layer[1, b, i] = layer_ptr[1 * 2 * height + b * height + i]
        values.numeratorOne = numeratorValues[2 * height + parity * height + i];

        //  denominator[i, b, 0] = layer[2, b, i] = layer_ptr[2 * 2 * height + b * height + i]
        values.denominatorZero = denominatorValues[parity * height + i];
        //  denominator[i, b, 1] = layer[3, b, i] = layer_ptr[3 * 2 * height + b * height + i]
        values.denominatorOne = denominatorValues[2 * height + parity * height + i];

        return values;
    }

    static __device__ __forceinline__ FirstLayerCircuitValues<F, EF> paddingValues()
    {
        FirstLayerCircuitValues<F, EF> values;
        values.numeratorZero = F::zero();
        values.numeratorOne = F::zero();
        values.denominatorZero = EF::one();
        values.denominatorOne = EF::one();
        return values;
    }

    static __device__ __forceinline__ CircuitValues<EF> fix_last_variable(
        FirstLayerCircuitValues<F, EF> zeroValues,
        FirstLayerCircuitValues<F, EF> oneValues,
        EF alpha)
    {
        CircuitValues<EF> values;
        values.numeratorZero   = alpha.interpolateLinear(oneValues.numeratorZero,   zeroValues.numeratorZero);
        values.numeratorOne    = alpha.interpolateLinear(oneValues.numeratorOne,    zeroValues.numeratorOne);
        values.denominatorZero = alpha.interpolateLinear(oneValues.denominatorZero, zeroValues.denominatorZero);
        values.denominatorOne  = alpha.interpolateLinear(oneValues.denominatorOne,  zeroValues.denominatorOne);
        return values;
    }

    __device__ __forceinline__ void store(
        F *numeratorValues,
        EF *denominatorValues,
        size_t i,
        size_t parity,
        size_t height)
    {
        // Store the indices at entry [d, parity, restrictedIndex]. This tranlsates to the index
        //  of the outut layer given by: d * 2 * height + parity * height + restrictedIndex
        // where d = 0,1 for numerator_0, numerator_1, and values and d = 2,3 for denominator_0,
        // denominator_1 and values respectively.
        numeratorValues[parity * height + i] = numeratorZero;
        numeratorValues[2 * height + parity * height + i] = numeratorOne;
        denominatorValues[parity * height + i] = denominatorZero;
        denominatorValues[2 * height + parity * height + i] = denominatorOne;
    }

    /// Compute the sumcheck sum values
    __device__ __forceinline__ EF sumAsPoly(EF lambda, EF eqValue)
    {
        EF numerator = numeratorZero * denominatorOne + numeratorOne * denominatorZero;
        EF denominator = denominatorZero * denominatorOne;
        return eqValue * (numerator * lambda + denominator);
    }
};

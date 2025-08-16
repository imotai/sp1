#pragma once

extern "C" void *logup_gkr_sum_as_poly_layer_kernel_circuit_layer_koala_bear_extension();
extern "C" void *logup_gkr_fix_last_variable_circuit_layer_kernel_koala_bear_extension();
extern "C" void *logup_gkr_fix_last_row_last_circuit_layer_kernel_circuit_layer_koala_bear_extension();
extern "C" void *logup_gkr_sum_as_poly_layer_kernel_interactions_layer_koala_bear_extension();
extern "C" void *logup_gkr_fix_last_variable_interactions_layer_kernel_koala_bear_extension();

template <typename EF>
struct CircuitValues
{
public:
    EF numeratorZero;
    EF numeratorOne;
    EF denominatorZero;
    EF denominatorOne;

public:
    static __device__ __forceinline__ CircuitValues<EF> load(EF *layer, size_t i, size_t parity, size_t height)
    {
        CircuitValues<EF> values;
        // Load the numerator and denominator values
        //  numerator[i, b, 0] = layer[0, b, i]
        //  numerator[i, b, 1] = layer[1, b, i]
        //  denominator[i, b, 0] = layer[2, b, i]
        //  denominator[i, b, 1] = layer[3, b, i]
        //
        // Indexing into the layer tensor is done as follows:
        //
        //  layer[k, parity, n] = layer_ptr[k * 2 * height + parity * height + n]

        //  numerator[i, b, 0] = layer[0, b, i] = layer_ptr[0 * 2 * height + b * height + i]
        values.numeratorZero = layer[parity * height + i];
        //  numerator[i, b, 1] = layer[1, b, i] = layer_ptr[1 * 2 * height + b * height + i]
        values.numeratorOne = layer[2 * height + parity * height + i];

        //  denominator[i, b, 0] = layer[2, b, i] = layer_ptr[2 * 2 * height + b * height + i]
        values.denominatorZero = layer[4 * height + parity * height + i];
        //  denominator[i, b, 1] = layer[3, b, i] = layer_ptr[3 * 2 * height + b * height + i]
        values.denominatorOne = layer[6 * height + parity * height + i];

        return values;
    }

    static __device__ __forceinline__ CircuitValues<EF> load(const EF *layer, size_t i, size_t parity, size_t height)
    {
        CircuitValues<EF> values;
        // Load the numerator and denominator values
        //  numerator[i, b, 0] = layer[0, b, i]
        //  numerator[i, b, 1] = layer[1, b, i]
        //  denominator[i, b, 0] = layer[2, b, i]
        //  denominator[i, b, 1] = layer[3, b, i]
        //
        // Indexing into the layer tensor is done as follows:
        //
        //  layer[k, parity, n] = layer_ptr[k * 2 * height + parity * height + n]

        //  numerator[i, b, 0] = layer[0, b, i] = layer_ptr[0 * 2 * height + b * height + i]
        values.numeratorZero = layer[parity * height + i];
        //  numerator[i, b, 1] = layer[1, b, i] = layer_ptr[1 * 2 * height + b * height + i]
        values.numeratorOne = layer[2 * height + parity * height + i];

        //  denominator[i, b, 0] = layer[2, b, i] = layer_ptr[2 * 2 * height + b * height + i]
        values.denominatorZero = layer[4 * height + parity * height + i];
        //  denominator[i, b, 1] = layer[3, b, i] = layer_ptr[3 * 2 * height + b * height + i]
        values.denominatorOne = layer[6 * height + parity * height + i];

        return values;
    }

    static __device__ __forceinline__ CircuitValues<EF> paddingValues()
    {
        CircuitValues<EF> values;
        values.numeratorZero = EF::zero();
        values.numeratorOne = EF::zero();
        values.denominatorZero = EF::one();
        values.denominatorOne = EF::one();
        return values;
    }

    static __device__ __forceinline__ CircuitValues<EF> fix_last_variable(
        CircuitValues<EF> zeroValues,
        CircuitValues<EF> oneValues,
        EF alpha)
    {
        CircuitValues<EF> values;
        values.numeratorZero   = alpha.interpolateLinear(oneValues.numeratorZero,   zeroValues.numeratorZero);
        values.numeratorOne    = alpha.interpolateLinear(oneValues.numeratorOne,    zeroValues.numeratorOne);
        values.denominatorZero = alpha.interpolateLinear(oneValues.denominatorZero, zeroValues.denominatorZero);
        values.denominatorOne  = alpha.interpolateLinear(oneValues.denominatorOne,  zeroValues.denominatorOne);
        return values;
    }

    __device__ __forceinline__ void store(EF *layer, size_t i, size_t parity, size_t height)
    {
        // Store the indices at entry [d, parity, restrictedIndex]. This tranlsates to the index
        //  of the outut layer given by: d * 2 * height + parity * height + restrictedIndex
        // where d = 0,1 for numerator_0, numerator_1, and values and d = 2,3 for denominator_0,
        // denominator_1 and values respectively.
        layer[parity * height + i] = numeratorZero;
        layer[2 * height + parity * height + i] = numeratorOne;
        layer[4 * height + parity * height + i] = denominatorZero;
        layer[6 * height + parity * height + i] = denominatorOne;
    }

    /// Compute the sumcheck sum values
    __device__ __forceinline__ EF sumAsPoly(EF lambda, EF eqValue)
    {
        EF numerator = numeratorZero * denominatorOne + numeratorOne * denominatorZero;
        EF denominator = denominatorZero * denominatorOne;
        return eqValue * (numerator * lambda + denominator);
    }
};

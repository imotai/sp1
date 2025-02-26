#pragma once

#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"

template<typename K>
struct ConstraintFolder {
   public:
    const K *preprocessed;
    size_t preprocessed_width;
    const K *main;
    size_t main_width;
    size_t height;
    const bb31_t *publicValues;
    const bb31_extension_t *powersOfAlpha;
    size_t constraintIndex;
    bb31_extension_t accumulator;
    size_t rowIdx;
    size_t xValueIdx;

   public:
    __device__ ConstraintFolder() {}

    __inline__ __device__ K var_f(unsigned char variant, unsigned int idx) {
        switch (variant) {
            case 0:
                return K::zero();
            case 1:
                return K(idx);
            case 2:
                return K::load(preprocessed, xValueIdx * (preprocessed_width * height) + (idx * height + rowIdx));
            case 4:
                return K::load(main, xValueIdx * (main_width * height) + (idx * height + rowIdx));
            case 9:
                return K(bb31_t::load(publicValues, idx));
            default:
                // Case 3: next row for for preprocessed trace for univariate.
                // Case 5: next row for for main trace for univariate.
                // Case 6: isFirstRow for univariate.
                // Case 7: isLastRow for univariate.
                // Case 8: isTransition for univariate.
                // Case 10: globalCumulativeSum for univariate.
                assert(0);
        }
    }

    __inline__ __device__ bb31_extension_t var_ef(unsigned char variant, unsigned int idx) {
        switch (variant) {
            case 0:
                return bb31_extension_t::zero();
            default:
                // Case 1: Permutation trace row for univariate.
                // Case 2: Permutation trace next row for multivariate.
                // Case 3: Permutation challenge for univariate.
                // Case 4: Local cumulative sum for univariate.
                assert(0);
        }
    }
};
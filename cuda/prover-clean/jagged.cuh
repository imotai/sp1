#pragma once
#include "config.cuh"
#include <stdint.h>

/// A jagged MLE : a concatenation of a bunch of rows of varying length.
template <typename DenseData>
struct JaggedMle {
    using OutputDenseData = typename DenseData::OutputDenseData;

  public:
    /// Has the same length as q. colIndex[i] is the row that q[i] belongs to.
    uint32_t* colIndex;
    /// Has one more than the number of columns. startIndices[i] is the start index of the i-th row
    /// in q[i].
    uint32_t* startIndices;
    /// A contiguous vector of dense underlying data
    DenseData denseData;

    // i is between 0 and length(colIndex)
    __forceinline__ __device__ size_t
    fixLastVariableUnchecked(JaggedMle<OutputDenseData>& output, size_t i, ext_t alpha) const {
        size_t colIdx = this->colIndex[i];
        size_t startIdx = this->startIndices[colIdx];
        size_t interactionHeight = this->startIndices[colIdx + 1] - startIdx;

        size_t rowIdx = i - startIdx;

        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;
        size_t restrictedIndex = (output.startIndices[colIdx] << 1) + rowIdx;

        this->denseData.fixLastVariable(output.denseData, restrictedIndex, zeroIdx, oneIdx, alpha);

        return restrictedIndex;
    }

    // i is between 0 and length(colIndex). Assumes that length(colIndex) is even.
    __forceinline__ __device__ size_t
    fixLastVariableTwoPadding(JaggedMle<OutputDenseData>& output, size_t i, ext_t alpha) const {
        size_t colIdx = this->colIndex[i];
        size_t startIdx = this->startIndices[colIdx];
        size_t interactionHeight = this->startIndices[colIdx + 1] - startIdx;

        size_t rowIdx = i - startIdx;

        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;
        size_t restrictedIndex = (output.startIndices[colIdx] << 1) + rowIdx;

        this->denseData.fixLastVariable(output.denseData, restrictedIndex, zeroIdx, oneIdx, alpha);

        // If this row does not have a length that is a multiple of four, the next row will have an
        // odd length. So we need to add some extra padding to the next row.
        size_t remainderModFour = interactionHeight & 3;
        bool isLast = (interactionHeight - 1) == rowIdx;
        if (remainderModFour && isLast) {
            this->denseData.pad(output.denseData, restrictedIndex + 1);
            this->denseData.pad(output.denseData, restrictedIndex + 2);

            // We also need to   update the outputcolIndex.
            output.colIndex[(restrictedIndex >> 1) + 1] = colIdx;
        }

        if (rowIdx & 1) {
            output.colIndex[restrictedIndex >> 1] = colIdx;
        }

        return restrictedIndex;
    }
};

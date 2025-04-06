#include "poseidon2_bb31_16.cuh"
#include "poseidon2.cuh"
#include "poseidon2_baby_bear.cuh"
#include <stdio.h>

using HashParams = poseidon2_bb31_16::BabyBear;
using HasherState_t = poseidon2::BabyBearHasherState;

__global__ void
leafHashPoseidon2BabyBear16(
    bb31_t **inputs,
    bb31_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t *widths,
    size_t num_inputs,
    size_t tree_height)
{
    poseidon2::BabyBearHasher hasher;
    HasherState_t state;

    size_t matrixHeight = 1 << tree_height;
    for (size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < matrixHeight; idx += blockDim.x * gridDim.x)
    {
        for (size_t j = 0; j < num_inputs; j++)
        {
            state.absorbRow(hasher, inputs[j], idx, widths[j], matrixHeight);
        }
        size_t digestIdx = idx + (matrixHeight - 1);
        state.finalize(hasher, digests[digestIdx]);
    }
}

extern "C" void *leaf_hash_poseidon_2_baby_bear_16_kernel()
{
    return (void *)leafHashPoseidon2BabyBear16;
}

__global__ void compressPoseidon2BabyBear16(
    bb31_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t layer_height)
{
    poseidon2::BabyBearHasher hasher;

    size_t layerLength = 1 << layer_height;
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < layerLength; i += blockDim.x * gridDim.x)
    {
        size_t idx = i + (layerLength - 1);
        size_t leftIdx = (idx << 1) + 1;
        size_t rightIdx = leftIdx + 1;
        hasher.compress(digests[leftIdx], digests[rightIdx], digests[idx]);
    }
}

extern "C" void *compress_poseidon_2_baby_bear_16_kernel()
{
    return (void *)compressPoseidon2BabyBear16;
}

__global__ void computePathsBabyBear16(
    bb31_t (*paths)[HashParams::DIGEST_WIDTH],
    size_t *indices,
    size_t numIndices,
    bb31_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t tree_height)
{
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < numIndices; i += blockDim.x * gridDim.x)
    {
        size_t idx = (1 << tree_height) - 1 + indices[i];
        for (int k = 0; k < tree_height; k++)
        {
            size_t siblingIdx = ((idx - 1) ^ 1) + 1;
            size_t parentIdx = (idx - 1) >> 1;
            bb31_t *digest = digests[siblingIdx];
            bb31_t *path_digest = paths[i * tree_height + k];
#pragma unroll
            for (int j = 0; j < HashParams::DIGEST_WIDTH; j++)
            {
                path_digest[j] = digest[j];
            }
            idx = parentIdx;
        }
    }
}

extern "C" void *compute_paths_poseidon_2_baby_bear_16_kernel()
{
    return (void *)computePathsBabyBear16;
}

__global__ void computeOpeningsBabyBear16(
    bb31_t **__restrict__ inputs,
    bb31_t *__restrict__ outputs,
    size_t *indices,
    size_t numIndices,
    size_t numInputs,
    size_t *batchSizes,
    size_t *batchOffsets,
    size_t matrixHeight,
    size_t numOpeningValues)
{
    for (size_t batchIdx = (blockIdx.z * blockDim.z) + threadIdx.z; batchIdx < numInputs; batchIdx += blockDim.z * gridDim.z)
    {
        bb31_t *in = inputs[batchIdx];
        size_t offset = batchOffsets[batchIdx];
        size_t batchSize = batchSizes[batchIdx];
        for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < numIndices; i += blockDim.x * gridDim.x)
        {
            size_t rowIdx = indices[i];
            for (size_t j = (blockIdx.y * blockDim.y) + threadIdx.y; j < batchSize; j += blockDim.y * gridDim.y)
            {
                outputs[i * numOpeningValues + j + offset] = in[j * matrixHeight + rowIdx];
            }
        }
    }
}

extern "C" void *compute_openings_poseidon_2_baby_bear_16_kernel()
{
    return (void *)computeOpeningsBabyBear16;
}

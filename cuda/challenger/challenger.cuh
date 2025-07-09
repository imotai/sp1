#pragma once
#include "../fields/bb31_t.cuh"
#include "../poseidon2/poseidon2_bb31_16.cuh"
#include "../poseidon2/poseidon2_bn254_3.cuh"
#include "../poseidon2/poseidon2.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../fields/bn254_t.cuh"

extern "C" void *grind_baby_bear();


class DuplexChallenger
{
    static constexpr const int WIDTH = poseidon2_bb31_16::BabyBear::WIDTH;
    static constexpr const int RATE = poseidon2_bb31_16::constants::RATE;

    bb31_t *sponge_state;
    bb31_t *input_buffer;
    size_t input_buffer_size;
    bb31_t *output_buffer;
    size_t output_buffer_size;

    __device__ void duplexing()
    {
        // Assert input size doesn't exceed RATE
        assert(input_buffer_size <= RATE);

        // Copy input buffer elements to sponge state
        for (size_t i = 0; i < input_buffer_size; i++)
        {
            sponge_state[i] = input_buffer[i];
        }

        // Clear input buffer.
        input_buffer_size = 0;

        // Apply the permutation to the sponge state and store the output in the output buffer.
        poseidon2::BabyBearHasher hasher;
        hasher.permute(sponge_state, output_buffer);

        // Copy the output buffer to the sponge state.
        output_buffer_size = WIDTH;
        for (size_t i = 0; i < WIDTH; i++)
        {
            sponge_state[i] = output_buffer[i];
        }
    }

public:
    static constexpr const size_t NUM_ELEMENTS = WIDTH + 2 * RATE;

    __device__ __forceinline__ bb31_t getVal(size_t idx)
    {
        return sponge_state[idx % 16];
    }

    __device__ __forceinline__ DuplexChallenger load(bb31_t *shared)
    {
        DuplexChallenger challenger;
        challenger.sponge_state = shared;
        challenger.input_buffer = shared + WIDTH;
        challenger.output_buffer = shared + WIDTH + RATE;
        challenger.input_buffer_size = input_buffer_size;
        challenger.output_buffer_size = output_buffer_size;
        return challenger;
    }

    __device__ __inline_hint__ void observe(bb31_t *value)
    {
        // Clear the output buffer.
        output_buffer_size = 0;

        // Push value to the input buffer.
        input_buffer_size += 1;
        input_buffer[input_buffer_size - 1] = *value;

        if (input_buffer_size == RATE)
        {
            duplexing();
        }
    }

    __device__ __inline_hint__ void observe_ext(bb31_extension_t *value)
    {
#pragma unroll
        for (size_t i = 0; i < bb31_extension_t::D; i++)
        {
            observe(&value->value[i]);
        }
    }

    __device__ __inline_hint__ bb31_t sample()
    {
        bb31_t result;
        if (input_buffer_size != 0 || output_buffer_size == 0)
        {
            duplexing();
        }
        // Pop the last element of the buffer.
        result = output_buffer[output_buffer_size - 1];
        output_buffer_size -= 1;
        return result;
    }

    __device__ __inline_hint__ bb31_extension_t sample_ext()
    {
        bb31_extension_t result;
        for (size_t i = 0; i < bb31_extension_t::D; i++)
        {
            result.value[i] = sample();
        }
        return result;
    }

    __device__ __inline_hint__ size_t sample_bits(size_t bits)
    {
        bb31_t rand_f = sample();

        // Equivalent to "as_canonical_u32" in the Rust implementation.
        size_t rand_usize = rand_f.as_canonical_u32();
        return rand_usize & ((1 << bits) - 1);
    }

    __device__ __forceinline__ bool check_witness(size_t bits, bb31_t *witness)
    {
        observe(witness);
        return sample_bits(bits) == 0;
    }

    __device__ __forceinline__ void grind(size_t bits, bb31_t *result, bool* found_flag, size_t n)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        size_t original_buffer_size = input_buffer_size;
        size_t original_output_buffer_size = output_buffer_size;
        __shared__ bb31_t challenger_state[NUM_ELEMENTS];

        if (threadIdx.x == 0) {
            for (size_t j = 0; j < WIDTH; j++) {
                challenger_state[j] = sponge_state[j];
            }
            for (size_t j = WIDTH; j < WIDTH + RATE; j++) {
                challenger_state[j] = input_buffer[j - WIDTH];
            }
            for (size_t j = WIDTH + RATE; j < NUM_ELEMENTS; j++) {
                challenger_state[j] = output_buffer[j - WIDTH - RATE];
            }
        }

        // Ensure all threads see the shared memory initialized
        __syncthreads();

        // Local copy of challenger state for each thread in each iteration.
        bb31_t local_state[NUM_ELEMENTS];
        for (size_t i = idx; i < n && !*found_flag; i += blockDim.x * gridDim.x) {
            input_buffer_size = original_buffer_size;
            output_buffer_size = original_output_buffer_size;
            // Reset the local state to the shared state.
            for (size_t j = 0; j < NUM_ELEMENTS; j++) {
                local_state[j] = challenger_state[j];
            }
            DuplexChallenger temp_challenger = load(local_state);
            

            bb31_t witness = bb31_t((int)i);
            if (temp_challenger.check_witness(bits, &witness)) {
                result[0] = witness;
                atomicExch((int*)found_flag, 1);
                __threadfence();
                return;
            }

        }
    }
};

class MultiField32Challenger
{
    static constexpr const int WIDTH = poseidon2_bn254_3::Bn254::WIDTH;
    static constexpr const int RATE = poseidon2_bn254_3::constants::RATE;

    bn254_t *sponge_state;
    bb31_t *input_buffer;
    size_t input_buffer_size;
    bb31_t *output_buffer;
    size_t output_buffer_size;
    size_t num_duplex_elms;
    size_t num_f_elms;

    __device__ void duplexing()
    {
        // Assert input size doesn't exceed RATE
        assert(num_f_elms == 4);
        assert(input_buffer_size <= num_duplex_elms * RATE);

        // Copy input buffer elements to sponge state
        for (size_t i = 0; i < input_buffer_size; i += num_duplex_elms)
        {
            size_t end = min(input_buffer_size, i + num_duplex_elms);
            bn254_t reduced = poseidon2_bn254_3::reduceBabyBear(input_buffer + i, nullptr, end - i, 0);
            sponge_state[i / num_duplex_elms] = reduced;
        }

        // Clear input buffer.
        input_buffer_size = 0;

        // Apply the permutation to the sponge state and store the output in the output buffer.
        poseidon2::Bn254Hasher hasher;

        bn254_t next_state[WIDTH];
        for (size_t i = 0 ; i < WIDTH ; i++) {
            next_state[i].set_to_zero();
        }
        hasher.permute(sponge_state, next_state);

        // Copy the output buffer to the sponge state.
        output_buffer_size = WIDTH * num_f_elms;
        for (size_t i = 0; i < WIDTH; i++)
        {
            sponge_state[i] = next_state[i];
            bn254_t x = next_state[i];
            x.from();
            uint32_t v0 = (uint32_t)(((uint64_t)(x[0]) + (uint64_t(1) << 32) * (uint64_t)(x[1])) % 0x78000001);
            uint32_t v1 = (uint32_t)(((uint64_t)(x[2]) + (uint64_t(1) << 32) * (uint64_t)(x[3])) % 0x78000001);
            uint32_t v2 = (uint32_t)(((uint64_t)(x[4]) + (uint64_t(1) << 32) * (uint64_t)(x[5])) % 0x78000001);
            uint32_t v3 = (uint32_t)(((uint64_t)(x[6]) + (uint64_t(1) << 32) * (uint64_t)(x[7])) % 0x78000001);
            output_buffer[i * 4] = bb31_t::from_canonical_u32(v0);
            output_buffer[i * 4 + 1] = bb31_t::from_canonical_u32(v1);
            output_buffer[i * 4 + 2] = bb31_t::from_canonical_u32(v2);
            output_buffer[i * 4 + 3] = bb31_t::from_canonical_u32(v3);
        }
    }

public:
    __device__ __inline_hint__ void observe(bb31_t *value)
    {
        // Clear the output buffer.
        output_buffer_size = 0;

        // Push value to the input buffer.
        input_buffer_size += 1;
        input_buffer[input_buffer_size - 1] = *value;

        if (input_buffer_size == num_duplex_elms * RATE)
        {
            duplexing();
        }
    }

    __device__ __inline_hint__ void observe_ext(bb31_extension_t *value)
    {
#pragma unroll
        for (size_t i = 0; i < bb31_extension_t::D; i++)
        {
            observe(&value->value[i]);
        }
    }

    __device__ __inline_hint__ bb31_t sample()
    {
        bb31_t result;
        if (input_buffer_size != 0 || output_buffer_size == 0)
        {
            duplexing();
        }
        // Pop the last element of the buffer.
        result = output_buffer[output_buffer_size - 1];
        output_buffer_size -= 1;
        return result;
    }

    __device__ __inline_hint__ bb31_extension_t sample_ext()
    {
        bb31_extension_t result;
        for (size_t i = 0; i < bb31_extension_t::D; i++)
        {
            result.value[i] = sample();
        }
        return result;
    }

};
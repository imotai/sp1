#include "poseidon2_wide.cuh"

#include "fields/bb31_t.cuh"
#include "poseidon2/poseidon2_bb31_16.cuh"

namespace poseidon2_wide
{
    using namespace poseidon2_bb31_16::constants;

    constexpr static const uintptr_t NUM_EXTERNAL_ROUNDS = csl_sys::NUM_EXTERNAL_ROUNDS;
    constexpr static const uintptr_t NUM_INTERNAL_ROUNDS = csl_sys::NUM_INTERNAL_ROUNDS;

    __device__ __forceinline__ void populate_external_round(
        const bb31_t external_rounds_state[WIDTH * NUM_EXTERNAL_ROUNDS],
        bb31_t sbox[WIDTH * NUM_EXTERNAL_ROUNDS], size_t r, bb31_t next_state[WIDTH])
    {
        bb31_t round_state[WIDTH];
        if (r == 0)
        {
            // external_linear_layer_immut
            bb31_t temp_round_state[WIDTH];
            for (size_t i = 0; i < WIDTH; i++)
            {
                temp_round_state[i] = external_rounds_state[r * WIDTH + i];
            }
            poseidon2_bb31_16::BabyBear::externalLinearLayer(temp_round_state);
            for (size_t i = 0; i < WIDTH; i++)
            {
                round_state[i] = temp_round_state[i];
            }
        }
        else
        {
            for (size_t i = 0; i < WIDTH; i++)
            {
                round_state[i] = external_rounds_state[r * WIDTH + i];
            }
        }

        size_t round = r < NUM_EXTERNAL_ROUNDS / 2 ? r : r + NUM_INTERNAL_ROUNDS;
        bb31_t add_rc[WIDTH];
        for (size_t i = 0; i < WIDTH; i++)
        {
            add_rc[i] = round_state[i] + bb31_t(bb31_t::to_monty(RC_16_30_U32[round][i]));
        }

        bb31_t sbox_deg_3[WIDTH];
        bb31_t sbox_deg_7[WIDTH];
        for (size_t i = 0; i < WIDTH; i++)
        {
            sbox_deg_3[i] = add_rc[i] * add_rc[i] * add_rc[i];
            sbox_deg_7[i] = sbox_deg_3[i] * sbox_deg_3[i] * add_rc[i];
        }

        for (size_t i = 0; i < WIDTH; i++)
        {
            sbox[r * WIDTH + i] = sbox_deg_3[i];
        }

        for (size_t i = 0; i < WIDTH; i++)
        {
            next_state[i] = sbox_deg_7[i];
        }
        poseidon2_bb31_16::BabyBear::externalLinearLayer(next_state);
    }

    __device__ __forceinline__ void populate_internal_rounds(
        const bb31_t internal_rounds_state[WIDTH],
        bb31_t internal_rounds_s0[NUM_INTERNAL_ROUNDS - 1], bb31_t sbox[NUM_INTERNAL_ROUNDS],
        bb31_t ret_state[WIDTH])
    {
        bb31_t state[WIDTH];
        for (size_t i = 0; i < WIDTH; i++)
        {
            state[i] = internal_rounds_state[i];
        }

        bb31_t sbox_deg_3[NUM_INTERNAL_ROUNDS];
        for (size_t r = 0; r < NUM_INTERNAL_ROUNDS; r++)
        {
            size_t round = r + NUM_EXTERNAL_ROUNDS / 2;
            bb31_t add_rc = state[0] + bb31_t(bb31_t::to_monty(RC_16_30_U32[round][0]));

            sbox_deg_3[r] = add_rc * add_rc * add_rc;
            bb31_t sbox_deg_7 = sbox_deg_3[r] * sbox_deg_3[r] * add_rc;

            state[0] = sbox_deg_7;
            poseidon2_bb31_16::BabyBear::internalLinearLayer(state, MAT_INTERNAL_DIAG_M1, MONTY_INVERSE);

            if (r < NUM_INTERNAL_ROUNDS - 1)
            {
                internal_rounds_s0[r] = state[0];
            }
        }

        for (size_t i = 0; i < WIDTH; i++)
        {
            ret_state[i] = state[i];
        }

        // Store sbox values if pointer is not null
        for (size_t r = 0; r < NUM_INTERNAL_ROUNDS; r++)
        {
            sbox[r] = sbox_deg_3[r];
        }
    }

    __device__ __forceinline__ void populate_perm(
        const bb31_t input[WIDTH], bb31_t external_rounds_state[WIDTH * NUM_EXTERNAL_ROUNDS],
        bb31_t internal_rounds_state[WIDTH],
        bb31_t internal_rounds_s0[NUM_INTERNAL_ROUNDS - 1],
        bb31_t external_sbox[WIDTH * NUM_EXTERNAL_ROUNDS],
        bb31_t internal_sbox[NUM_INTERNAL_ROUNDS], bb31_t output_state[WIDTH])
    {
        for (size_t i = 0; i < WIDTH; i++)
        {
            external_rounds_state[i] = input[i];
        }

        for (size_t r = 0; r < NUM_EXTERNAL_ROUNDS / 2; r++)
        {
            bb31_t next_state[WIDTH];
            populate_external_round(external_rounds_state, external_sbox, r,
                                    next_state);
            if (r == NUM_EXTERNAL_ROUNDS / 2 - 1)
            {
                for (size_t i = 0; i < WIDTH; i++)
                {
                    internal_rounds_state[i] = next_state[i];
                }
            }
            else
            {
                for (size_t i = 0; i < WIDTH; i++)
                {
                    external_rounds_state[(r + 1) * WIDTH + i] = next_state[i];
                }
            }
        }

        bb31_t ret_state[WIDTH];
        populate_internal_rounds(internal_rounds_state, internal_rounds_s0,
                                 internal_sbox, ret_state);
        size_t row = NUM_EXTERNAL_ROUNDS / 2;
        for (size_t i = 0; i < WIDTH; i++)
        {
            external_rounds_state[row * WIDTH + i] = ret_state[i];
        }

        for (size_t r = NUM_EXTERNAL_ROUNDS / 2; r < NUM_EXTERNAL_ROUNDS; r++)
        {
            bb31_t next_state[WIDTH];
            populate_external_round(external_rounds_state, external_sbox, r,
                                    next_state);
            if (r == NUM_EXTERNAL_ROUNDS - 1)
            {
                for (size_t i = 0; i < WIDTH; i++)
                {
                    output_state[i] = next_state[i];
                }
            }
            else
            {
                for (size_t i = 0; i < WIDTH; i++)
                {
                    external_rounds_state[(r + 1) * WIDTH + i] = next_state[i];
                }
            }
        }
    }

    __device__ void event_to_row(const bb31_t input[WIDTH], bb31_t *input_row,
                                 size_t start, size_t stride,
                                 bool sbox_state)
    {
        bb31_t external_rounds_state[WIDTH * NUM_EXTERNAL_ROUNDS];
        bb31_t internal_rounds_state[WIDTH];
        bb31_t internal_rounds_s0[NUM_INTERNAL_ROUNDS - 1];
        bb31_t output_state[WIDTH];
        bb31_t external_sbox[WIDTH * NUM_EXTERNAL_ROUNDS];
        bb31_t internal_sbox[NUM_INTERNAL_ROUNDS];

        populate_perm(input, external_rounds_state, internal_rounds_state,
                      internal_rounds_s0, external_sbox, internal_sbox,
                      output_state);

        size_t cursor = 0;
        for (size_t i = 0; i < (WIDTH * NUM_EXTERNAL_ROUNDS); i++)
        {
            input_row[start + (cursor + i) * stride] = external_rounds_state[i];
        }

        cursor += WIDTH * NUM_EXTERNAL_ROUNDS;
        for (size_t i = 0; i < WIDTH; i++)
        {
            input_row[start + (cursor + i) * stride] = internal_rounds_state[i];
        }

        cursor += WIDTH;
        for (size_t i = 0; i < (NUM_INTERNAL_ROUNDS - 1); i++)
        {
            input_row[start + (cursor + i) * stride] = internal_rounds_s0[i];
        }

        cursor += NUM_INTERNAL_ROUNDS - 1;
        for (size_t i = 0; i < WIDTH; i++)
        {
            input_row[start + (cursor + i) * stride] = output_state[i];
        }

        if (sbox_state)
        {
            cursor += WIDTH;
            for (size_t i = 0; i < (WIDTH * NUM_EXTERNAL_ROUNDS); i++)
            {
                input_row[start + (cursor + i) * stride] = external_sbox[i];
            }

            cursor += WIDTH * NUM_EXTERNAL_ROUNDS;
            for (size_t i = 0; i < NUM_INTERNAL_ROUNDS; i++)
            {
                input_row[start + (cursor + i) * stride] = internal_sbox[i];
            }
        }
    }

} // namespace poseidon2_wide

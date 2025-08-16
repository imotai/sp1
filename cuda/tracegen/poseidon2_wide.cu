#include "poseidon2_wide.cuh"

#include "fields/kb31_t.cuh"
#include "poseidon2/poseidon2_kb31_16.cuh"

namespace poseidon2_wide
{
    using namespace poseidon2_kb31_16::constants;

    constexpr static const uintptr_t NUM_EXTERNAL_ROUNDS = poseidon2_kb31_16::constants::ROUNDS_F;
    constexpr static const uintptr_t NUM_INTERNAL_ROUNDS = poseidon2_kb31_16::constants::ROUNDS_P;

    __device__ __forceinline__ void populate_external_round(
        const kb31_t external_rounds_state[WIDTH * NUM_EXTERNAL_ROUNDS],
        size_t r, kb31_t next_state[WIDTH])
    {
        kb31_t round_state[WIDTH];
        if (r == 0)
        {
            // external_linear_layer_immut
            kb31_t temp_round_state[WIDTH];
            for (size_t i = 0; i < WIDTH; i++)
            {
                temp_round_state[i] = external_rounds_state[r * WIDTH + i];
            }
            poseidon2_kb31_16::KoalaBear::externalLinearLayer(temp_round_state);
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
        kb31_t add_rc[WIDTH];
        for (size_t i = 0; i < WIDTH; i++)
        {
            add_rc[i] = round_state[i] + kb31_t(kb31_t::to_monty(RC_16_30_U32[round][i]));
        }

        kb31_t sbox_deg_3[WIDTH];
        for (size_t i = 0; i < WIDTH; i++)
        {
            sbox_deg_3[i] = add_rc[i] * add_rc[i] * add_rc[i];
            // sbox_deg_7[i] = sbox_deg_3[i] * sbox_deg_3[i] * add_rc[i];
        }

        for (size_t i = 0; i < WIDTH; i++)
        {
            next_state[i] = sbox_deg_3[i];
        }
        poseidon2_kb31_16::KoalaBear::externalLinearLayer(next_state);
    }

    __device__ __forceinline__ void populate_internal_rounds(
        const kb31_t internal_rounds_state[WIDTH],
        kb31_t internal_rounds_s0[NUM_INTERNAL_ROUNDS - 1], 
        kb31_t ret_state[WIDTH])
    {
        kb31_t state[WIDTH];
        for (size_t i = 0; i < WIDTH; i++)
        {
            state[i] = internal_rounds_state[i];
        }

        kb31_t sbox_deg_3[NUM_INTERNAL_ROUNDS];
        for (size_t r = 0; r < NUM_INTERNAL_ROUNDS; r++)
        {
            size_t round = r + NUM_EXTERNAL_ROUNDS / 2;
            kb31_t add_rc = state[0] + kb31_t(kb31_t::to_monty(RC_16_30_U32[round][0]));

            sbox_deg_3[r] = add_rc * add_rc * add_rc;
            // kb31_t sbox_deg_7 = sbox_deg_3[r] * sbox_deg_3[r] * add_rc;

            state[0] = sbox_deg_3[r];
            poseidon2_kb31_16::KoalaBear::internalLinearLayer(state, MAT_INTERNAL_DIAG_M1, MONTY_INVERSE);

            if (r < NUM_INTERNAL_ROUNDS - 1)
            {
                internal_rounds_s0[r] = state[0];
            }
        }

        for (size_t i = 0; i < WIDTH; i++)
        {
            ret_state[i] = state[i];
        }
    }

    __device__ __forceinline__ void populate_perm(
        const kb31_t input[WIDTH], kb31_t external_rounds_state[WIDTH * NUM_EXTERNAL_ROUNDS],
        kb31_t internal_rounds_state[WIDTH],
        kb31_t internal_rounds_s0[NUM_INTERNAL_ROUNDS - 1],
        // kb31_t external_sbox[WIDTH * NUM_EXTERNAL_ROUNDS],
        kb31_t output_state[WIDTH])
    {
        for (size_t i = 0; i < WIDTH; i++)
        {
            external_rounds_state[i] = input[i];
        }

        for (size_t r = 0; r < NUM_EXTERNAL_ROUNDS / 2; r++)
        {
            kb31_t next_state[WIDTH];
            populate_external_round(external_rounds_state, r,
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

        kb31_t ret_state[WIDTH];
        populate_internal_rounds(internal_rounds_state, internal_rounds_s0,
                                  ret_state);
        size_t row = NUM_EXTERNAL_ROUNDS / 2;
        for (size_t i = 0; i < WIDTH; i++)
        {
            external_rounds_state[row * WIDTH + i] = ret_state[i];
        }

        for (size_t r = NUM_EXTERNAL_ROUNDS / 2; r < NUM_EXTERNAL_ROUNDS; r++)
        {
            kb31_t next_state[WIDTH];
            populate_external_round(external_rounds_state,  r,
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

    __device__ void event_to_row(const kb31_t input[WIDTH], kb31_t *input_row,
                                 size_t start, size_t stride
                                 )
    {
        kb31_t external_rounds_state[WIDTH * NUM_EXTERNAL_ROUNDS];
        kb31_t internal_rounds_state[WIDTH];
        kb31_t internal_rounds_s0[NUM_INTERNAL_ROUNDS - 1];
        kb31_t output_state[WIDTH];
        // kb31_t external_sbox[WIDTH * NUM_EXTERNAL_ROUNDS];
        // kb31_t internal_sbox[NUM_INTERNAL_ROUNDS];

        populate_perm(input, external_rounds_state, internal_rounds_state,
                      internal_rounds_s0, 
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
    }

} // namespace poseidon2_wide

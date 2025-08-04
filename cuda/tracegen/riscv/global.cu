#include "csl-cbindgen.hpp"

#include "fields/bb31_t.cuh"
#include "fields/bb31_extension_t.cuh"
#include "fields/bb31_septic_extension_t.cuh"

#include "poseidon2/poseidon2.cuh"
#include "poseidon2/poseidon2_bb31_16.cuh"

#include "tracegen/poseidon2_wide.cuh"

constexpr static const uintptr_t POSEIDON2_WIDTH = poseidon2_bb31_16::constants::WIDTH;

__device__ void populate_global_interaction(csl_sys::GlobalInteractionOperation<bb31_t> *cols, const csl_sys::GlobalInteractionEvent *event)
{
    // Initialize `m_trial` to the first 7 elements of the message.

#pragma unroll(1)
    for (uint32_t offset = 0; offset < 256; ++offset)
    {
        bb31_t m_trial[POSEIDON2_WIDTH];
        {
            m_trial[0] = bb31_t::from_canonical_u32(event->message[0]) + bb31_t::from_canonical_u32(uint32_t(event->kind) << 24);
            m_trial[1] = bb31_t::from_canonical_u32(event->message[1]);
            m_trial[2] = bb31_t::from_canonical_u32(event->message[2]);
            m_trial[3] = bb31_t::from_canonical_u32(event->message[3]);
            m_trial[4] = bb31_t::from_canonical_u32(event->message[4]);
            m_trial[5] = bb31_t::from_canonical_u32(event->message[5]);
            m_trial[6] = bb31_t::from_canonical_u32(event->message[6]);
            m_trial[7] = bb31_t::from_canonical_u32(event->message[7]) + bb31_t::from_canonical_u32(offset << 16);
            m_trial[8] = bb31_t::zero();
            m_trial[9] = bb31_t::zero();
            m_trial[10] = bb31_t::zero();
            m_trial[11] = bb31_t::zero();
            m_trial[12] = bb31_t::zero();
            m_trial[13] = bb31_t::zero();
            m_trial[14] = bb31_t::zero();
            m_trial[15] = bb31_t::zero();
        }
        // Set the 8th element of `x_trial` to the offset.

        // Compute the poseidon2 hash of `m_trial` to compute `m_hash`.
        bb31_t m_hash[POSEIDON2_WIDTH];
        poseidon2::BabyBearHasher::permute(m_trial, m_hash);

        // Convert the hash to a septic extension element.
        bb31_septic_extension_t x_trial = bb31_septic_extension_t::zero();
        for (uint32_t i = 0; i < 7; i++)
        {
            x_trial.value[i] = m_hash[i];
        }

        bb31_septic_extension_t y_sq = x_trial.curve_formula();
        bb31_t y_sq_pow_r = y_sq.pow_r();
        bb31_t is_square = y_sq_pow_r ^ 1006632960;
        if (is_square == bb31_t::one())
        {
            bb31_septic_extension_t y = y_sq.sqrt(y_sq_pow_r);
            if (y.is_exception())
            {
                continue;
            }
            if (y.is_receive() != event->is_receive)
            {
                y = bb31_septic_extension_t::zero() - y;
            }
            for (uint32_t idx = 0; idx < 8; idx++)
            {
                cols->offset_bits[idx] = bb31_t::from_canonical_u32((offset >> idx) & 1);
            }
            for (uintptr_t i = 0; i < 7; i++)
            {
                cols->x_coordinate._0[i] = x_trial.value[i];
                cols->y_coordinate._0[i] = y.value[i];
            }
            uint32_t range_check_value;
            if (event->is_receive)
            {
                range_check_value = y.value[6].as_canonical_u32() - 1;
            }
            else
            {
                range_check_value = y.value[6].as_canonical_u32() - (bb31_t::MOD + 1) / 2;
            }
            bb31_t top_4_bits = bb31_t::zero();
            for (uint32_t idx = 0; idx < 30; idx++)
            {
                cols->y6_bit_decomp[idx] = bb31_t::from_canonical_u32((range_check_value >> idx) & 1);
                if (idx >= 26)
                {
                    top_4_bits += cols->y6_bit_decomp[idx];
                }
            }
            top_4_bits -= bb31_t::from_canonical_u32(4);
            cols->range_check_witness = top_4_bits.reciprocal();

            bb31_t *input_row = reinterpret_cast<bb31_t *>(&cols->permutation);
            poseidon2_wide::event_to_row(m_trial, input_row, 0, 1, true);

            return;
        }
        // x_start += bb31_t::from_canonical_u32(1 << 16);
    }
    assert(false);
}

__device__ void populate_global_interaction_dummy(csl_sys::GlobalInteractionOperation<bb31_t> *cols)
{
    bb31_t m_trial[POSEIDON2_WIDTH];
    {
        m_trial[0] = bb31_t::zero();
        m_trial[1] = bb31_t::zero();
        m_trial[2] = bb31_t::zero();
        m_trial[3] = bb31_t::zero();
        m_trial[4] = bb31_t::zero();
        m_trial[5] = bb31_t::zero();
        m_trial[6] = bb31_t::zero();
        m_trial[7] = bb31_t::zero();
        m_trial[8] = bb31_t::zero();
        m_trial[9] = bb31_t::zero();
        m_trial[10] = bb31_t::zero();
        m_trial[11] = bb31_t::zero();
        m_trial[12] = bb31_t::zero();
        m_trial[13] = bb31_t::zero();
        m_trial[14] = bb31_t::zero();
        m_trial[15] = bb31_t::zero();
    }

    bb31_t *input_row = reinterpret_cast<bb31_t *>(&cols->permutation);
    poseidon2_wide::event_to_row(m_trial, input_row, 0, 1, true);
}

__global__ void riscv_global_generate_trace_decompress_kernel(
    bb31_t *trace,
    uintptr_t trace_height,
    const csl_sys::GlobalInteractionEvent *events,
    uintptr_t nb_events)
{
    static const size_t GLOBAL_COLUMNS =
        sizeof(csl_sys::GlobalCols<bb31_t>) / sizeof(bb31_t);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll(1)
    for (; i < trace_height; i += blockDim.x * gridDim.x)
    {
        // ok so we're on the ith row
        bb31_septic_curve_t sum = bb31_septic_curve_t();
        if (i == 0)
        {
            sum = bb31_septic_curve_t::start_point();
        }
        csl_sys::GlobalCols<bb31_t> cols;
        bb31_t *cols_arr = reinterpret_cast<bb31_t *>(&cols);
        for (int k = 0; k < GLOBAL_COLUMNS; k++)
        {
            cols_arr[k] = bb31_t::zero();
        }

        if (i < nb_events)
        {
            for (int k = 0; k < 8; k++)
            {
                cols.message[k] = bb31_t::from_canonical_u32(events[i].message[k]);
            }
            cols.is_receive = bb31_t::from_bool(events[i].is_receive);
            cols.kind = bb31_t::from_canonical_u8(events[i].kind);
            cols.is_send = bb31_t::one() - bb31_t::from_bool(events[i].is_receive);
            cols.is_real = bb31_t::one();
            cols.message_0_16bit_limb = bb31_t::from_canonical_u32(events[i].message[0] & 0xFFFF);
            cols.message_0_8bit_limb = bb31_t::from_canonical_u32((events[i].message[0] >> 16) & 0xFF);
            cols.index = bb31_t::from_canonical_u32(i);

            // Populate the interaction.
            populate_global_interaction(
                &cols.interaction,
                &events[i]);

            // Compute the running accumulator.
            cols.accumulation.cumulative_sum[0] = cols.interaction.x_coordinate;
            cols.accumulation.cumulative_sum[1] = cols.interaction.y_coordinate;
            bb31_septic_curve_t point = bb31_septic_curve_t(
                cols.interaction.x_coordinate._0,
                cols.interaction.y_coordinate._0);
            sum += point;
        }
        else
        {
            populate_global_interaction_dummy(&cols.interaction);
        }

        // Populate the initial digest.
        for (int k = 0; k < 7; k++)
        {
            cols.accumulation.initial_digest[0]._0[k] = sum.x.value[k];
            cols.accumulation.initial_digest[1]._0[k] = sum.y.value[k];
        }

        // Populate the trace.
        const bb31_t *arr = reinterpret_cast<bb31_t *>(&cols);
        for (size_t k = 0; k < GLOBAL_COLUMNS; ++k)
        {
            trace[i + k * trace_height] = arr[k];
        }
    }
}

__global__ void riscv_global_generate_trace_finalize_kernel(
    bb31_t *trace,
    uintptr_t trace_height,
    const bb31_septic_curve_t *cumulative_sums,
    uintptr_t nb_events)
{
    static const size_t GLOBAL_COLUMNS = sizeof(csl_sys::GlobalCols<bb31_t>) / sizeof(bb31_t);

    int i = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll(1)
    for (; i < trace_height; i += blockDim.x * gridDim.x)
    {
        csl_sys::GlobalCols<bb31_t> cols;
        bb31_t *temp_arr = reinterpret_cast<bb31_t *>(&cols);
        for (int j = 0; j < GLOBAL_COLUMNS; j++)
        {
            temp_arr[j] = trace[i + j * trace_height];
        }

        bb31_septic_curve_t sum = cumulative_sums[i];

        int event_idx = i;

        bb31_septic_extension_t point_x = bb31_septic_extension_t(cols.interaction.x_coordinate._0);
        bb31_septic_extension_t point_y = bb31_septic_extension_t(cols.interaction.y_coordinate._0) * (bb31_t::zero() - bb31_t::one());
        bb31_septic_curve_t point = bb31_septic_curve_t(point_x, point_y);

        for (int k = 0; k < 7; k++)
        {
            cols.accumulation.cumulative_sum[0]._0[k] = sum.x.value[k];
            cols.accumulation.cumulative_sum[1]._0[k] = sum.y.value[k];
        }

        sum += point;

        for (int k = 0; k < 7; k++)
        {
            cols.accumulation.initial_digest[0]._0[k] =
                sum.x.value[k];
            cols.accumulation.initial_digest[1]._0[k] =
                sum.y.value[k];
        }

        if (event_idx < nb_events)
        {
            for (int k = 0; k < 7; k++)
            {
                cols.accumulation.sum_checker._0[k] = bb31_t::zero();
            }
        }
        else
        {
            bb31_septic_curve_t dummy =
                bb31_septic_curve_t::dummy_point();
            for (int k = 0; k < 7; k++)
            {
                cols.interaction.x_coordinate._0[k] = dummy.x.value[k];
                cols.interaction.y_coordinate._0[k] = dummy.y.value[k];
            }
            bb31_septic_curve_t digest = bb31_septic_curve_t(
                cols.accumulation.cumulative_sum[0]._0,
                cols.accumulation.cumulative_sum[1]._0);
            bb31_septic_extension_t sum_checker_x =
                bb31_septic_curve_t::sum_checker_x(digest, dummy, digest);
            for (int k = 0; k < 7; k++)
            {
                cols.accumulation.sum_checker._0[k] = sum_checker_x.value[k];
            }
        }

        bb31_t *final_temp = reinterpret_cast<bb31_t *>(&cols);
        for (int j = 0; j < GLOBAL_COLUMNS; j++)
        {
            trace[i + j * trace_height] = final_temp[j];
        }
    }
}

namespace csl_sys
{
    extern KernelPtr riscv_global_generate_trace_decompress_kernel()
    {
        return (KernelPtr)::riscv_global_generate_trace_decompress_kernel;
    }
    extern KernelPtr riscv_global_generate_trace_finalize_kernel()
    {
        return (KernelPtr)::riscv_global_generate_trace_finalize_kernel;
    }
}

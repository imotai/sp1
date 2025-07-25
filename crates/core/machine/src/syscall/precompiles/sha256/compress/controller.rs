use crate::{air::SP1CoreAirBuilder, operations::SyscallAddrOperation, utils::next_multiple_of_32};

use super::ShaCompressControlChip;
use crate::utils::u32_to_half_word;
use core::borrow::Borrow;
use slop_air::{Air, BaseAir};
use slop_algebra::{AbstractField, PrimeField32};
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{ByteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, MachineAir},
    InteractionKind,
};
use std::{borrow::BorrowMut, iter::once};

impl ShaCompressControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

pub const NUM_SHA_COMPRESS_CONTROL_COLS: usize = size_of::<ShaCompressControlCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaCompressControlCols<T> {
    pub clk_high: T,
    pub clk_low: T,
    pub w_ptr: SyscallAddrOperation<T>,
    pub h_ptr: SyscallAddrOperation<T>,
    pub is_real: T,
    pub initial_state: [[T; WORD_SIZE / 2]; 8],
    pub final_state: [[T; WORD_SIZE / 2]; 8],
}

impl<F> BaseAir<F> for ShaCompressControlChip {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ShaCompressControlChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "ShaCompressControl".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.get_precompile_events(SyscallCode::SHA_COMPRESS).len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_multiple_of_32(nb_rows, size_log2);
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        let mut blu_events = vec![];
        for (_, event) in input.get_precompile_events(SyscallCode::SHA_COMPRESS).iter() {
            let event = if let PrecompileEvent::ShaCompress(event) = event {
                event
            } else {
                unreachable!()
            };
            let mut row = [F::zero(); NUM_SHA_COMPRESS_CONTROL_COLS];
            let cols: &mut ShaCompressControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.clk_high = F::from_canonical_u32((event.clk >> 24) as u32);
            cols.clk_low = F::from_canonical_u32((event.clk & 0xFFFFFF) as u32);
            cols.w_ptr.populate(&mut blu_events, event.w_ptr, 512);
            cols.h_ptr.populate(&mut blu_events, event.h_ptr, 64);
            cols.is_real = F::one();
            for i in 0..8 {
                let prev_value = event.h[i];
                let value = event.h_write_records[i].value;
                cols.initial_state[i] = u32_to_half_word(prev_value);
                cols.final_state[i] = u32_to_half_word((value as u32).wrapping_sub(prev_value));
            }
            rows.push(row);
        }

        let nb_rows = rows.len();
        let mut padded_nb_rows = nb_rows.next_multiple_of(32);
        if padded_nb_rows == 2 || padded_nb_rows == 1 {
            padded_nb_rows = 4;
        }
        for _ in nb_rows..padded_nb_rows {
            let row = [F::zero(); NUM_SHA_COMPRESS_CONTROL_COLS];
            rows.push(row);
        }
        output.add_byte_lookup_events(blu_events);

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_COMPRESS_CONTROL_COLS,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SHA_COMPRESS).is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for ShaCompressControlChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Initialize columns.
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaCompressControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let w_ptr =
            SyscallAddrOperation::<AB::F>::eval(builder, 512, local.w_ptr, local.is_real.into());
        let h_ptr =
            SyscallAddrOperation::<AB::F>::eval(builder, 64, local.h_ptr, local.is_real.into());

        // Receive the syscall.
        builder.receive_syscall(
            local.clk_high,
            local.clk_low,
            AB::F::from_canonical_u32(SyscallCode::SHA_COMPRESS.syscall_id()),
            w_ptr.map(Into::into),
            h_ptr.map(Into::into),
            local.is_real,
            InteractionScope::Local,
        );

        // Send the initial state.
        let send_values = once(local.clk_high.into())
            .chain(once(local.clk_low.into()))
            .chain(w_ptr.map(Into::into))
            .chain(h_ptr.map(Into::into))
            .chain(once(AB::Expr::from_canonical_u32(0)))
            .chain(
                local.initial_state.into_iter().flat_map(|word| word.into_iter()).map(Into::into),
            )
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::ShaCompress),
            InteractionScope::Local,
        );

        // Receive the final state.
        let receive_values = once(local.clk_high.into())
            .chain(once(local.clk_low.into()))
            .chain(w_ptr.map(Into::into))
            .chain(h_ptr.map(Into::into))
            .chain(once(AB::Expr::from_canonical_u32(80)))
            .chain(local.final_state.into_iter().flat_map(|word| word.into_iter()).map(Into::into))
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::ShaCompress),
            InteractionScope::Local,
        );
    }
}

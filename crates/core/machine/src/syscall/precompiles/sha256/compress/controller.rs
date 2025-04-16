use super::ShaCompressControlChip;
use core::borrow::Borrow;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{events::PrecompileEvent, syscalls::SyscallCode, ExecutionRecord, Program};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, MachineAir, SP1AirBuilder},
    InteractionKind, Word,
};
use std::borrow::BorrowMut;

impl ShaCompressControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

pub const NUM_SHA_COMPRESS_CONTROL_COLS: usize = size_of::<ShaCompressControlCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaCompressControlCols<T> {
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,
    pub h_ptr: T,
    pub is_real: T,
    pub initial_state: [Word<T>; 8],
    pub final_state: [Word<T>; 8],
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

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::SHA_COMPRESS).iter() {
            let event = if let PrecompileEvent::ShaCompress(event) = event {
                event
            } else {
                unreachable!()
            };
            let mut row = [F::zero(); NUM_SHA_COMPRESS_CONTROL_COLS];
            let cols: &mut ShaCompressControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.w_ptr = F::from_canonical_u32(event.w_ptr);
            cols.h_ptr = F::from_canonical_u32(event.h_ptr);
            cols.is_real = F::one();
            for i in 0..8 {
                let prev_value = event.h[i];
                let value = event.h_write_records[i].value;
                cols.initial_state[i] = Word::from(prev_value);
                cols.final_state[i] = Word::from(value.wrapping_sub(prev_value));
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
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Initialize columns.
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaCompressControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // Receive the syscall.
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_canonical_u32(SyscallCode::SHA_COMPRESS.syscall_id()),
            local.w_ptr,
            local.h_ptr,
            local.is_real,
            InteractionScope::Local,
        );

        // Send the initial state.
        builder.send(
            AirInteraction::new(
                vec![
                    local.shard.into(),
                    local.clk.into(),
                    local.w_ptr.into(),
                    local.h_ptr.into(),
                    AB::Expr::from_canonical_u32(0),
                ]
                .into_iter()
                .chain(
                    local
                        .initial_state
                        .into_iter()
                        .flat_map(|word| word.into_iter())
                        .map(Into::into),
                )
                .collect(),
                local.is_real.into(),
                InteractionKind::ShaCompress,
            ),
            InteractionScope::Local,
        );

        // Receive the final state.
        builder.receive(
            AirInteraction::new(
                vec![
                    local.shard.into(),
                    local.clk.into(),
                    local.w_ptr.into(),
                    local.h_ptr.into(),
                    AB::Expr::from_canonical_u32(80),
                ]
                .into_iter()
                .chain(
                    local.final_state.into_iter().flat_map(|word| word.into_iter()).map(Into::into),
                )
                .collect(),
                local.is_real.into(),
                InteractionKind::ShaCompress,
            ),
            InteractionScope::Local,
        );
    }
}

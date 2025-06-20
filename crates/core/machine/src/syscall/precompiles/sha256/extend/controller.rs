use super::ShaExtendControlChip;
use crate::{operations::SyscallAddrOperation, utils::next_multiple_of_32};
use core::borrow::Borrow;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{ByteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, MachineAir, SP1AirBuilder},
    InteractionKind,
};
use std::borrow::BorrowMut;

impl ShaExtendControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

pub const NUM_SHA_EXTEND_CONTROL_COLS: usize = size_of::<ShaExtendControlCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaExtendControlCols<T> {
    pub clk_high: T,
    pub clk_low: T,
    pub w_ptr: SyscallAddrOperation<T>,
    pub is_real: T,
}

impl<F> BaseAir<F> for ShaExtendControlChip {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ShaExtendControlChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "ShaExtendControl".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.get_precompile_events(SyscallCode::SHA_EXTEND).len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_multiple_of_32(nb_rows, size_log2);
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut blu_events = vec![];
        let mut rows = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::SHA_EXTEND).iter() {
            let event =
                if let PrecompileEvent::ShaExtend(event) = event { event } else { unreachable!() };
            let mut row = [F::zero(); NUM_SHA_EXTEND_CONTROL_COLS];
            let cols: &mut ShaExtendControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.clk_high = F::from_canonical_u32((event.clk >> 24) as u32);
            cols.clk_low = F::from_canonical_u32((event.clk & 0xFFFFFF) as u32);
            cols.w_ptr.populate(&mut blu_events, event.w_ptr, 256);
            cols.is_real = F::one();
            rows.push(row);
        }

        output.add_byte_lookup_events(blu_events);

        let nb_rows = rows.len();
        let mut padded_nb_rows = nb_rows.next_multiple_of(32);
        if padded_nb_rows == 2 || padded_nb_rows == 1 {
            padded_nb_rows = 4;
        }
        for _ in nb_rows..padded_nb_rows {
            let row = [F::zero(); NUM_SHA_EXTEND_CONTROL_COLS];
            rows.push(row);
        }

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_EXTEND_CONTROL_COLS,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SHA_EXTEND).is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for ShaExtendControlChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Initialize columns.
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaExtendControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let w_ptr =
            SyscallAddrOperation::<AB::F>::eval(builder, 256, local.w_ptr, local.is_real.into());

        // Receive the syscall.
        builder.receive_syscall(
            local.clk_high,
            local.clk_low,
            AB::F::from_canonical_u32(SyscallCode::SHA_EXTEND.syscall_id()),
            w_ptr.clone(),
            [AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()],
            local.is_real,
            InteractionScope::Local,
        );

        // // Send the initial state.
        // builder.send(
        //     AirInteraction::new(
        //         vec![
        //             local.clk_high.into(),
        //             local.clk_low.into(),
        //             w_ptr.clone(),
        //             AB::Expr::from_canonical_u32(16),
        //         ],
        //         local.is_real.into(),
        //         InteractionKind::ShaExtend,
        //     ),
        //     InteractionScope::Local,
        // );

        // // Receive the final state.
        // builder.receive(
        //     AirInteraction::new(
        //         vec![
        //             local.clk_high.into(),
        //             local.clk_low.into(),
        //             w_ptr.clone(),
        //             AB::Expr::from_canonical_u32(64),
        //         ],
        //         local.is_real.into(),
        //         InteractionKind::ShaExtend,
        //     ),
        //     InteractionScope::Local,
        // );
    }
}

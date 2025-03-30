use super::ShaExtendControlChip;
use core::borrow::Borrow;
use p3_air::{Air, BaseAir};
use p3_field::AbstractField;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core_executor::syscalls::SyscallCode;
use sp1_core_executor::{events::PrecompileEvent, ExecutionRecord, Program};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{AirInteraction, MachineAir};
use sp1_stark::air::{InteractionScope, SP1AirBuilder};
use sp1_stark::InteractionKind;
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
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,
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

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::SHA_EXTEND).iter() {
            let event =
                if let PrecompileEvent::ShaExtend(event) = event { event } else { unreachable!() };
            let mut row = [F::zero(); NUM_SHA_EXTEND_CONTROL_COLS];
            let cols: &mut ShaExtendControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.w_ptr = F::from_canonical_u32(event.w_ptr);
            cols.is_real = F::one();
            rows.push(row);
        }

        let nb_rows = rows.len();
        let mut padded_nb_rows = nb_rows.next_power_of_two();
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

        // Receive the syscall.
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_canonical_u32(SyscallCode::SHA_EXTEND.syscall_id()),
            local.w_ptr,
            AB::Expr::zero(),
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
                    AB::Expr::from_canonical_u32(16),
                ],
                local.is_real.into(),
                InteractionKind::ShaExtend,
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
                    AB::Expr::from_canonical_u32(64),
                ],
                local.is_real.into(),
                InteractionKind::ShaExtend,
            ),
            InteractionScope::Local,
        );
    }
}

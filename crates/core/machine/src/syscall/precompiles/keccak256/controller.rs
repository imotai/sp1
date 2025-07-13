use crate::{
    air::SP1CoreAirBuilder, memory::MemoryAccessCols, operations::SyscallAddrOperation,
    utils::next_multiple_of_32,
};

use super::{KeccakPermuteControlChip, STATE_NUM_WORDS};
use core::borrow::Borrow;
use slop_air::{Air, BaseAir};
use slop_algebra::{AbstractField, PrimeField32};
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{ByteRecord, MemoryRecordEnum, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, MachineAir},
    InteractionKind, Word,
};
use std::borrow::BorrowMut;

impl KeccakPermuteControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

pub const NUM_KECCAK_PERMUTE_CONTROL_COLS: usize = size_of::<KeccakPermuteControlCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct KeccakPermuteControlCols<T> {
    pub clk_high: T,
    pub clk_low: T,
    pub state_addr: SyscallAddrOperation<T>,
    pub is_real: T,
    pub initial_memory_access: [MemoryAccessCols<T>; 50],
    pub final_memory_access: [MemoryAccessCols<T>; 50],
    pub final_value: [Word<T>; 50],
}

impl<F> BaseAir<F> for KeccakPermuteControlChip {
    fn width(&self) -> usize {
        NUM_KECCAK_PERMUTE_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for KeccakPermuteControlChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "KeccakPermuteControl".to_string()
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let mut blu_events = vec![];
        for (_, event) in input.get_precompile_events(SyscallCode::KECCAK_PERMUTE).iter() {
            let event = if let PrecompileEvent::KeccakPermute(event) = event {
                event
            } else {
                unreachable!()
            };
            let mut row = [F::zero(); NUM_KECCAK_PERMUTE_CONTROL_COLS];
            let cols: &mut KeccakPermuteControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.state_addr.populate(&mut blu_events, event.state_addr, 200);
            for (j, read_record) in event.state_read_records.iter().enumerate() {
                cols.initial_memory_access[j]
                    .populate(MemoryRecordEnum::Read(*read_record), &mut blu_events);
            }
            for (j, write_record) in event.state_write_records.iter().enumerate() {
                cols.final_memory_access[j]
                    .populate(MemoryRecordEnum::Write(*write_record), &mut blu_events);
                cols.final_value[j] = Word::from(write_record.value);
            }
        }
        output.add_byte_lookup_events(blu_events);
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.get_precompile_events(SyscallCode::KECCAK_PERMUTE).len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_multiple_of_32(nb_rows, size_log2);
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        let mut blu_events = vec![];
        for (_, event) in input.get_precompile_events(SyscallCode::KECCAK_PERMUTE).iter() {
            let event = if let PrecompileEvent::KeccakPermute(event) = event {
                event
            } else {
                unreachable!()
            };
            let mut row = [F::zero(); NUM_KECCAK_PERMUTE_CONTROL_COLS];
            let cols: &mut KeccakPermuteControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.clk_high = F::from_canonical_u32((event.clk >> 24) as u32);
            cols.clk_low = F::from_canonical_u32((event.clk & 0xFFFFFF) as u32);
            cols.state_addr.populate(&mut blu_events, event.state_addr, 200);
            cols.is_real = F::one();
            for (j, read_record) in event.state_read_records.iter().enumerate() {
                cols.initial_memory_access[j]
                    .populate(MemoryRecordEnum::Read(*read_record), &mut blu_events);
            }
            for (j, write_record) in event.state_write_records.iter().enumerate() {
                cols.final_memory_access[j]
                    .populate(MemoryRecordEnum::Write(*write_record), &mut blu_events);
                cols.final_value[j] = Word::from(write_record.value);
            }

            rows.push(row);
        }

        let nb_rows = rows.len();
        let mut padded_nb_rows = nb_rows.next_multiple_of(32);
        if padded_nb_rows == 2 || padded_nb_rows == 1 {
            padded_nb_rows = 4;
        }
        for _ in nb_rows..padded_nb_rows {
            let row = [F::zero(); NUM_KECCAK_PERMUTE_CONTROL_COLS];
            rows.push(row);
        }

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_KECCAK_PERMUTE_CONTROL_COLS,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::KECCAK_PERMUTE).is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for KeccakPermuteControlChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Initialize columns.
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakPermuteControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let state_addr = SyscallAddrOperation::<AB::F>::eval(
            builder,
            200,
            local.state_addr,
            local.is_real.into(),
        );

        // Receive the syscall.
        builder.receive_syscall(
            local.clk_high,
            local.clk_low,
            AB::F::from_canonical_u32(SyscallCode::KECCAK_PERMUTE.syscall_id()),
            state_addr.clone(),
            AB::Expr::zero(),
            local.is_real,
            InteractionScope::Local,
        );

        // Send the initial state.
        builder.send(
            AirInteraction::new(
                vec![
                    local.clk_high.into(),
                    local.clk_low.into(),
                    state_addr.clone(),
                    AB::Expr::zero(),
                ]
                .into_iter()
                .chain(
                    local
                        .initial_memory_access
                        .into_iter()
                        .flat_map(|access| access.prev_value.into_iter())
                        .map(Into::into),
                )
                .collect(),
                local.is_real.into(),
                InteractionKind::Keccak,
            ),
            InteractionScope::Local,
        );

        // Receive the final state.
        builder.receive(
            AirInteraction::new(
                vec![
                    local.clk_high.into(),
                    local.clk_low.into(),
                    state_addr.clone(),
                    AB::Expr::from_canonical_u32(24),
                ]
                .into_iter()
                .chain(
                    local.final_value.into_iter().flat_map(|word| word.into_iter()).map(Into::into),
                )
                .collect(),
                local.is_real.into(),
                InteractionKind::Keccak,
            ),
            InteractionScope::Local,
        );

        // Evaluate the memory accesses.
        for i in 0..STATE_NUM_WORDS {
            builder.eval_memory_access_read(
                local.clk_high,
                local.clk_low,
                state_addr.clone() + AB::Expr::from_canonical_u32(4 * i as u32),
                local.initial_memory_access[i],
                local.is_real,
            );
            builder.eval_memory_access_write(
                local.clk_high,
                local.clk_low + AB::Expr::one(),
                state_addr.clone() + AB::Expr::from_canonical_u32(4 * i as u32),
                local.final_memory_access[i],
                local.final_value[i],
                local.is_real,
            );
        }
    }
}

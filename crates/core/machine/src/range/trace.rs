use std::borrow::BorrowMut;

use crate::utils::zeroed_f_vec;
use slop_algebra::PrimeField32;
use slop_matrix::dense::RowMajorMatrix;
use sp1_core_executor::{events::ByteRecord, ByteOpcode, ExecutionRecord, Program};
use sp1_stark::air::MachineAir;

use super::{
    columns::{RangeMultCols, NUM_RANGE_MULT_COLS, NUM_RANGE_PREPROCESSED_COLS},
    RangeChip,
};

pub const NUM_ROWS: usize = 1 << 17;

impl<F: PrimeField32> MachineAir<F> for RangeChip<F> {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Range".to_string()
    }

    fn num_rows(&self, _: &Self::Record) -> Option<usize> {
        Some(NUM_ROWS)
    }

    fn preprocessed_width(&self) -> usize {
        NUM_RANGE_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let trace = Self::trace();
        Some(trace)
    }

    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        let initial_timestamp_0 = ((input.public_values.initial_timestamp >> 32) & 0xFFFF) as u16;
        let initial_timestamp_3 = (input.public_values.initial_timestamp & 0xFFFF) as u16;
        let last_timestamp_0 = ((input.public_values.last_timestamp >> 32) & 0xFFFF) as u16;
        let last_timestamp_3 = (input.public_values.last_timestamp & 0xFFFF) as u16;
        let pc_start_0 = (input.public_values.pc_start & 0xFFFF) as u16;
        let pc_start_1 = ((input.public_values.pc_start >> 16) & 0xFFFF) as u16;
        let pc_start_2 = ((input.public_values.pc_start >> 32) & 0xFFFF) as u16;
        let next_pc_0 = (input.public_values.next_pc & 0xFFFF) as u16;
        let next_pc_1 = ((input.public_values.next_pc >> 16) & 0xFFFF) as u16;
        let next_pc_2 = ((input.public_values.next_pc >> 32) & 0xFFFF) as u16;

        output.add_bit_range_check(initial_timestamp_0, 16);
        output.add_bit_range_check(initial_timestamp_3, 16);
        output.add_bit_range_check(last_timestamp_0, 16);
        output.add_bit_range_check(last_timestamp_3, 16);
        output.add_bit_range_check(pc_start_0, 16);
        output.add_bit_range_check(pc_start_1, 16);
        output.add_bit_range_check(pc_start_2, 16);
        output.add_bit_range_check(next_pc_0, 16);
        output.add_bit_range_check(next_pc_1, 16);
        output.add_bit_range_check(next_pc_2, 16);
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut trace =
            RowMajorMatrix::new(zeroed_f_vec(NUM_RANGE_MULT_COLS * NUM_ROWS), NUM_RANGE_MULT_COLS);

        for (lookup, mult) in input.byte_lookups.iter() {
            if lookup.opcode != ByteOpcode::Range {
                continue;
            }
            let row = (lookup.a as usize) + (1 << lookup.b);
            let cols: &mut RangeMultCols<F> = trace.row_mut(row).borrow_mut();
            cols.multiplicity += F::from_canonical_usize(*mult);
        }

        trace
    }

    fn included(&self, _shard: &Self::Record) -> bool {
        true
    }
}

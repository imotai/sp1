use std::borrow::BorrowMut;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use sp1_core_executor::{ByteOpcode, ExecutionRecord, Program};
use sp1_stark::air::MachineAir;

use crate::utils::zeroed_f_vec;

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

    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {
        // Do nothing since this chip has no dependencies.
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

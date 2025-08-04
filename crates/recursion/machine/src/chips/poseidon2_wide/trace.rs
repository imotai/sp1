use slop_air::BaseAir;
use slop_algebra::{AbstractField, PrimeField32};
use slop_baby_bear::BabyBear;
use slop_matrix::dense::RowMajorMatrix;
use slop_maybe_rayon::prelude::*;
use sp1_core_machine::{
    operations::poseidon2::{trace::populate_perm, WIDTH},
    utils::next_multiple_of_32,
};
use sp1_recursion_executor::{
    ExecutionRecord, Instruction, Poseidon2Instr, Poseidon2Io, RecursionProgram,
};
use sp1_stark::air::MachineAir;
use std::{borrow::BorrowMut, mem::size_of};
use tracing::instrument;

use super::{columns::preprocessed::Poseidon2PreprocessedColsWide, Poseidon2WideChip};
use crate::chips::mem::MemoryAccessCols;

const PREPROCESSED_POSEIDON2_WIDTH: usize = size_of::<Poseidon2PreprocessedColsWide<u8>>();

impl<F: PrimeField32, const DEGREE: usize> MachineAir<F> for Poseidon2WideChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = RecursionProgram<F>;

    #[allow(clippy::uninlined_format_args)]
    fn name(&self) -> String {
        format!("Poseidon2WideDeg{}", DEGREE)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let height = input.program.shape.as_ref().and_then(|shape| shape.height(self));
        let events = &input.poseidon2_events;
        Some(next_multiple_of_32(events.len(), height))
    }

    #[instrument(name = "generate poseidon2 wide trace", level = "debug", skip_all, fields(rows = input.poseidon2_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _output: &mut ExecutionRecord<F>,
    ) -> RowMajorMatrix<F> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_trace only supports BabyBear field"
        );

        let events = unsafe {
            std::mem::transmute::<&Vec<Poseidon2Io<F>>, &Vec<Poseidon2Io<BabyBear>>>(
                &input.poseidon2_events,
            )
        };
        let padded_nb_rows = self.num_rows(input).unwrap();
        let num_columns = <Self as BaseAir<BabyBear>>::width(self);
        let mut values = vec![BabyBear::zero(); padded_nb_rows * num_columns];

        let populate_len = input.poseidon2_events.len() * num_columns;
        let (values_pop, values_dummy) = values.split_at_mut(populate_len);

        join(
            || {
                values_pop.par_chunks_mut(num_columns).zip_eq(events).for_each(|(row, &event)| {
                    populate_perm::<BabyBear, DEGREE>(event.input, Some(event.output), row);
                })
            },
            || {
                let mut dummy_row = vec![BabyBear::zero(); num_columns];
                populate_perm::<BabyBear, DEGREE>([BabyBear::zero(); WIDTH], None, &mut dummy_row);
                values_dummy
                    .par_chunks_mut(num_columns)
                    .for_each(|row| row.copy_from_slice(&dummy_row))
            },
        );

        RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            num_columns,
        )
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        PREPROCESSED_POSEIDON2_WIDTH
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let height = program.shape.as_ref().and_then(|shape| shape.height(self));
        Some(next_multiple_of_32(instrs_len, height))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_preprocessed_trace only supports BabyBear field"
        );

        // Allocating an intermediate `Vec` is faster.
        let instrs: Vec<&Poseidon2Instr<BabyBear>> = program
            .inner
            .iter() // Faster than using `rayon` for some reason. Maybe vectorization?
            .filter_map(|instruction| match instruction.inner() {
                Instruction::Poseidon2(instr) => Some(unsafe {
                    std::mem::transmute::<&Poseidon2Instr<F>, &Poseidon2Instr<BabyBear>>(
                        instr.as_ref(),
                    )
                }),
                _ => None,
            })
            .collect::<Vec<_>>();
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![BabyBear::zero(); padded_nb_rows * PREPROCESSED_POSEIDON2_WIDTH];

        let populate_len = instrs.len() * PREPROCESSED_POSEIDON2_WIDTH;
        values[..populate_len]
            .par_chunks_mut(PREPROCESSED_POSEIDON2_WIDTH)
            .zip_eq(instrs)
            .for_each(|(row, instr)| {
                // Set the memory columns. We read once, at the first iteration,
                // and write once, at the last iteration.
                *row.borrow_mut() = Poseidon2PreprocessedColsWide {
                    input: instr.addrs.input,
                    output: std::array::from_fn(|j| MemoryAccessCols {
                        addr: instr.addrs.output[j],
                        mult: instr.mults[j],
                    }),
                    is_real: BabyBear::one(),
                }
            });

        Some(RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            PREPROCESSED_POSEIDON2_WIDTH,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::chips::{poseidon2_wide::Poseidon2WideChip, test_fixtures};
    use slop_matrix::Matrix;
    use sp1_recursion_executor::ExecutionRecord;
    use sp1_stark::air::MachineAir;

    const DEGREE_3: usize = 3;
    const DEGREE_9: usize = 9;

    #[tokio::test]
    async fn test_generate_trace_deg_3() {
        let shard = test_fixtures::shard().await;
        let chip = Poseidon2WideChip::<DEGREE_3>;
        let trace = chip.generate_trace(shard, &mut ExecutionRecord::default());
        assert!(trace.height() > test_fixtures::MIN_ROWS);
    }

    #[tokio::test]
    async fn test_generate_trace_deg_9() {
        let shard = test_fixtures::shard().await;
        let chip = Poseidon2WideChip::<DEGREE_9>;
        let trace = chip.generate_trace(shard, &mut ExecutionRecord::default());
        assert!(trace.height() > test_fixtures::MIN_ROWS);
    }
    #[tokio::test]
    async fn test_generate_preprocessed_trace_deg_3() {
        let program = &test_fixtures::program_with_input().await.0;
        let chip = Poseidon2WideChip::<DEGREE_3>;
        let trace = chip.generate_preprocessed_trace(program).unwrap();
        assert!(trace.height() > test_fixtures::MIN_ROWS);
    }

    #[tokio::test]
    async fn test_generate_preprocessed_trace_deg_9() {
        let program = &test_fixtures::program_with_input().await.0;
        let chip = Poseidon2WideChip::<DEGREE_9>;
        let trace = chip.generate_preprocessed_trace(program).unwrap();
        assert!(trace.height() > test_fixtures::MIN_ROWS);
    }
}

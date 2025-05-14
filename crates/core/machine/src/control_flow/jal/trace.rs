use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ExecutionRecord, Program,
};
use sp1_stark::air::MachineAir;

use crate::utils::{next_multiple_of_32, zeroed_f_vec};

use super::{JalChip, JalColumns, NUM_JAL_COLS};

impl<F: PrimeField32> MachineAir<F> for JalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Jal".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.jal_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.jal_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <JalChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_JAL_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_JAL_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_JAL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut JalColumns<F> = row.borrow_mut();

                    if idx < input.jal_events.len() {
                        let event = &input.jal_events[idx];
                        let mut instruction = *input.program.fetch(event.0.pc);
                        cols.is_real = F::one();
                        instruction.op_b = event.0.pc.wrapping_add(instruction.op_b);
                        instruction.op_c = event.0.pc.wrapping_add(4);
                        if instruction.op_a == 0 {
                            instruction.op_c = 0;
                        }
                        cols.state.populate(
                            &mut blu,
                            input.public_values.execution_shard as u32,
                            event.0.clk,
                            event.0.pc,
                        );
                        cols.adapter.populate(&mut blu, &instruction, event.1);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_JAL_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.jal_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

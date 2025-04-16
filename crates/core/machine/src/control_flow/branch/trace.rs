use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{BranchEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
};
use sp1_stark::air::MachineAir;

use crate::utils::{next_multiple_of_32, zeroed_f_vec};

use super::{BranchChip, BranchColumns, NUM_BRANCH_COLS};

impl<F: PrimeField32> MachineAir<F> for BranchChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Branch".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.branch_events.len()) / num_cpus::get(), 1);
        let nb_rows = input.branch_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_multiple_of_32(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_BRANCH_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_BRANCH_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_BRANCH_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut BranchColumns<F> = row.borrow_mut();

                    if idx < input.branch_events.len() {
                        let event = &input.branch_events[idx];
                        self.event_to_row(&event.0, cols, &mut blu);
                        let mut instruction = *input.program.fetch(event.0.pc);
                        instruction.op_c = event.0.pc.wrapping_add(instruction.op_c);
                        cols.state.populate(
                            &mut blu,
                            input.public_values.execution_shard,
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
        RowMajorMatrix::new(values, NUM_BRANCH_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.branch_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl BranchChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &BranchEvent,
        cols: &mut BranchColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        cols.is_beq = F::from_bool(matches!(event.opcode, Opcode::BEQ));
        cols.is_bne = F::from_bool(matches!(event.opcode, Opcode::BNE));
        cols.is_blt = F::from_bool(matches!(event.opcode, Opcode::BLT));
        cols.is_bge = F::from_bool(matches!(event.opcode, Opcode::BGE));
        cols.is_bltu = F::from_bool(matches!(event.opcode, Opcode::BLTU));
        cols.is_bgeu = F::from_bool(matches!(event.opcode, Opcode::BGEU));

        let a_eq_b = event.a == event.b;

        let use_signed_comparison = matches!(event.opcode, Opcode::BLT | Opcode::BGE);

        let a_lt_b = if use_signed_comparison {
            (event.a as i32) < (event.b as i32)
        } else {
            event.a < event.b
        };

        let branching = match event.opcode {
            Opcode::BEQ => a_eq_b,
            Opcode::BNE => !a_eq_b,
            Opcode::BLT | Opcode::BLTU => a_lt_b,
            Opcode::BGE | Opcode::BGEU => !a_lt_b,
            _ => unreachable!(),
        };

        cols.compare_operation.populate_signed(
            blu,
            a_lt_b as u32,
            event.a,
            event.b,
            use_signed_comparison,
            true,
        );

        cols.next_pc = F::from_canonical_u32(event.next_pc);

        if branching {
            cols.is_branching = F::one();
        }
    }
}

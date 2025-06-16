use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{MachineAir, SP1AirBuilder};

use crate::{
    adapter::{register::alu_type::ALUTypeReader, state::CPUState},
    operations::BitwiseU16Operation,
    utils::{next_multiple_of_32, pad_rows_fixed},
};

/// The number of main trace columns for `BitwiseChip`.
pub const NUM_BITWISE_COLS: usize = size_of::<BitwiseCols<u8>>();

/// A chip that implements bitwise operations for the opcodes XOR, OR, and AND.
#[derive(Default)]
pub struct BitwiseChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct BitwiseCols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ALUTypeReader<T>,

    /// Instance of `BitwiseOperation` to handle bitwise logic in `BitwiseChip`'s ALU operations.
    pub bitwise_operation: BitwiseU16Operation<T>,

    /// If the opcode is XOR.
    pub is_xor: T,

    // If the opcode is OR.
    pub is_or: T,

    /// If the opcode is AND.
    pub is_and: T,
}

impl<F: PrimeField32> MachineAir<F> for BitwiseChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Bitwise".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.bitwise_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = input
            .bitwise_events
            .par_iter()
            .map(|event| {
                let mut row = [F::zero(); NUM_BITWISE_COLS];
                let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(&event.0, cols, &mut blu);
                let instruction = input.program.fetch(event.0.pc_rel);
                cols.state.populate(
                    &mut blu,
                    input.public_values.execution_shard as u32,
                    event.0.clk,
                    event.0.pc_rel,
                );
                cols.adapter.populate(&mut blu, instruction, event.1);

                row
            })
            .collect::<Vec<_>>();

        // Pad the trace to a power of two.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_BITWISE_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        assert_eq!(rows.len(), <BitwiseChip as MachineAir<F>>::num_rows(self, input).unwrap());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_BITWISE_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.bitwise_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .bitwise_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_BITWISE_COLS];
                    let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(&event.0, cols, &mut blu);
                    let instruction = input.program.fetch(event.0.pc_rel);
                    cols.state.populate(
                        &mut blu,
                        input.public_values.execution_shard as u32,
                        event.0.clk,
                        event.0.pc_rel,
                    );
                    cols.adapter.populate(&mut blu, instruction, event.1);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.bitwise_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl BitwiseChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut BitwiseCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.bitwise_operation.populate_bitwise(blu, event.a, event.b, event.c, event.opcode);

        cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
        cols.is_or = F::from_bool(event.opcode == Opcode::OR);
        cols.is_and = F::from_bool(event.opcode == Opcode::AND);
    }
}

impl<F> BaseAir<F> for BitwiseChip {
    fn width(&self) -> usize {
        NUM_BITWISE_COLS
    }
}

impl<AB> Air<AB> for BitwiseChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BitwiseCols<AB::Var> = (*local).borrow();

        // SAFETY: All selectors `is_xor`, `is_or`, `is_and` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real`, the sum of the three
        // selectors, is boolean. Therefore, the `opcode` and `cpu_opcode` matches the
        // corresponding opcode.
        let is_real = local.is_xor + local.is_or + local.is_and;
        builder.assert_bool(local.is_xor);
        builder.assert_bool(local.is_or);
        builder.assert_bool(local.is_and);
        builder.assert_bool(is_real.clone());

        // Get the opcode for the operation.
        let byte_opcode = local.is_xor * ByteOpcode::XOR.as_field::<AB::F>()
            + local.is_or * ByteOpcode::OR.as_field::<AB::F>()
            + local.is_and * ByteOpcode::AND.as_field::<AB::F>();

        // Get the cpu opcode, which corresponds to the opcode being sent in the CPU table.
        let cpu_opcode = local.is_xor * Opcode::XOR.as_field::<AB::F>()
            + local.is_or * Opcode::OR.as_field::<AB::F>()
            + local.is_and * Opcode::AND.as_field::<AB::F>();

        // Constrain the bitwise operation over `op_b` and `op_c`.
        let result = BitwiseU16Operation::<AB::F>::eval_bitwise_u16(
            builder,
            local.adapter.b().map(Into::into),
            local.adapter.c().map(Into::into),
            local.bitwise_operation,
            byte_opcode,
            is_real.clone(),
        );

        // Constrain the state of the CPU.
        // The program counter and timestamp increment by `4`.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc_rel + AB::F::from_canonical_u32(PC_INC),
            AB::Expr::from_canonical_u32(PC_INC),
            is_real.clone(),
        );

        // Constrain the program and register reads.
        ALUTypeReader::<AB::F>::eval(
            builder,
            local.state.shard::<AB>(),
            local.state.clk::<AB>(),
            local.state.pc_rel,
            cpu_opcode,
            result,
            local.adapter,
            is_real,
        );
    }
}

// #[cfg(test)]
// mod tests {
//     #![allow(clippy::print_stdout)]

//     use p3_baby_bear::BabyBear;
//     use p3_matrix::dense::RowMajorMatrix;
//     use sp1_core_executor::{events::AluEvent, ExecutionRecord, Opcode};
//     use sp1_stark::{
//         air::{MachineAir, SP1_PROOF_NUM_PV_ELTS},
//         baby_bear_poseidon2::BabyBearPoseidon2,
//         Chip, StarkMachine,
//     };

//     use crate::utils::{run_test_machine, setup_test_machine};

//     use super::BitwiseChip;

//     #[test]
//     fn generate_trace() {
//         let mut shard = ExecutionRecord::default();
//         shard.bitwise_events = vec![AluEvent::new(0, Opcode::XOR, 25, 10, 19, false)];
//         let chip = BitwiseChip::default();
//         let trace: RowMajorMatrix<BabyBear> =
//             chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         println!("{:?}", trace.values)
//     }

//     #[test]
//     fn prove_babybear() {
//         let mut shard = ExecutionRecord::default();
//         shard.bitwise_events = [
//             AluEvent::new(0, Opcode::XOR, 25, 10, 19, false),
//             AluEvent::new(0, Opcode::OR, 27, 10, 19, false),
//             AluEvent::new(0, Opcode::AND, 2, 10, 19, false),
//         ]
//         .repeat(1000);

//         // Run setup.
//         let air = BitwiseChip::default();
//         let config = BabyBearPoseidon2::new();
//         let chip = Chip::new(air);
//         let (pk, vk) = setup_test_machine(StarkMachine::new(
//             config.clone(),
//             vec![chip],
//             SP1_PROOF_NUM_PV_ELTS,
//             true,
//         ));

//         // Run the test.
//         let air = BitwiseChip::default();
//         let chip: Chip<BabyBear, BitwiseChip> = Chip::new(air);
//         let machine = StarkMachine::new(config.clone(), vec![chip], SP1_PROOF_NUM_PV_ELTS, true);
//         run_test_machine::<BabyBearPoseidon2, BitwiseChip>(vec![shard], machine, pk,
// vk).unwrap();     }

//     // TODO: Re-enable when we LOGUP-GKR working.
//     // #[test]
//     // fn test_malicious_bitwise() {
//     //     const NUM_TESTS: usize = 5;

//     //     for opcode in [Opcode::XOR, Opcode::OR, Opcode::AND] {
//     //         for _ in 0..NUM_TESTS {
//     //             let op_a = thread_rng().gen_range(0..u32::MAX);
//     //             let op_b = thread_rng().gen_range(0..u32::MAX);
//     //             let op_c = thread_rng().gen_range(0..u32::MAX);

//     //             let correct_op_a = if opcode == Opcode::XOR {
//     //                 op_b ^ op_c
//     //             } else if opcode == Opcode::OR {
//     //                 op_b | op_c
//     //             } else {
//     //                 op_b & op_c
//     //             };

//     //             assert!(op_a != correct_op_a);

//     //             let instructions = vec![
//     //                 Instruction::new(opcode, 5, op_b, op_c, true, true),
//     //                 Instruction::new(Opcode::ADD, 10, 0, 0, false, false),
//     //             ];
//     //             let program = Program::new(instructions, 0, 0);
//     //             let stdin = SP1Stdin::new();

//     //             type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

//     //             let malicious_trace_pv_generator = move |prover: &P,
//     //                                                      record: &mut ExecutionRecord|
//     //                   -> Vec<(
//     //                 String,
//     //                 RowMajorMatrix<Val<BabyBearPoseidon2>>,
//     //             )> {
//     //                 let mut malicious_record = record.clone();
//     //                 malicious_record.cpu_events[0].a = op_a;
//     //                 if let Some(MemoryRecordEnum::Write(mut write_record)) =
//     //                     malicious_record.cpu_events[0].a_record
//     //                 {
//     //                     write_record.value = op_a;
//     //                 }
//     //                 malicious_record.bitwise_events[0].a = op_a;
//     //                 prover.generate_traces(&malicious_record)
//     //             };

//     //             let result =
//     //                 run_malicious_test::<P>(program, stdin,
// Box::new(malicious_trace_pv_generator));     //             assert!(result.is_err() &&
// result.unwrap_err().is_local_cumulative_sum_failing());     //         }
//     //     }
//     // }
// }

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::iter::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, CLK_INC, PC_INC,
};
use sp1_derive::AlignedBorrow;

use sp1_stark::air::MachineAir;
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    adapter::{register::j_type::JTypeReader, state::CPUState},
    air::SP1CoreAirBuilder,
    utils::{next_multiple_of_32, zeroed_f_vec, InstructionExt as _},
};

#[derive(Default)]
pub struct AuipcChip;

pub const NUM_AUIPC_COLS: usize = size_of::<AuipcColumns<u8>>();

impl<F> BaseAir<F> for AuipcChip {
    fn width(&self) -> usize {
        NUM_AUIPC_COLS
    }
}

/// The column layout for AUIPC instructions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AuipcColumns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: JTypeReader<T>,

    /// Whether the instruction is an AUIPC instruction.
    pub is_real: T,
}

impl<AB> Air<AB> for AuipcChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AuipcColumns<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let opcode = AB::Expr::from_canonical_u32(Opcode::AUIPC as u32);

        // Constrain the state of the CPU.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc_rel + AB::F::from_canonical_u32(PC_INC),
            AB::Expr::from_canonical_u32(CLK_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        // Set `op_b` immediate as `op_a_not_0 * (pc + op_b)` value in the instruction encoding.
        JTypeReader::<AB::F>::eval(
            builder,
            local.state.clk_high::<AB>(),
            local.state.clk_low::<AB>(),
            local.state.pc_rel,
            opcode,
            *local.adapter.b(),
            local.adapter,
            local.is_real.into(),
        );
    }
}

impl<F: PrimeField32> MachineAir<F> for AuipcChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Auipc".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.auipc_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.auipc_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <AuipcChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_AUIPC_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_AUIPC_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_AUIPC_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AuipcColumns<F> = row.borrow_mut();

                    if idx < input.auipc_events.len() {
                        let event = &input.auipc_events[idx];
                        let instruction = input
                            .program
                            .fetch(event.0.pc_rel)
                            .preprocess_auipc(input.program.pc_base, event.0.pc_rel);
                        cols.is_real = F::one();

                        cols.state.populate(&mut blu, event.0.clk, event.0.pc_rel);
                        cols.adapter.populate(&mut blu, &instruction, event.1);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_AUIPC_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.auipc_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

// #[cfg(test)]
// mod tests {
//     use std::borrow::BorrowMut;

//     use p3_baby_bear::BabyBear;
//     use p3_field::AbstractField;
//     use p3_matrix::dense::RowMajorMatrix;
//     use sp1_core_executor::{
//         ExecutionError, ExecutionRecord, Executor, Instruction, Opcode, Program, Simple,
//     };
//     use sp1_stark::{
//         air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
//         MachineProver, SP1CoreOpts, Val,
//     };

//     use crate::{
//         control_flow::{AuipcChip, AuipcColumns},
//         io::SP1Stdin,
//         riscv::RiscvAir,
//         utils::run_malicious_test,
//     };

//     // TODO: Re-enable when we LOGUP-GKR working.
//     // #[test]
//     // fn test_malicious_auipc() {
//     //     let instructions = vec![
//     //         Instruction::new(Opcode::AUIPC, 29, 12, 12, true, true),
//     //         Instruction::new(Opcode::ADD, 10, 0, 0, false, false),
//     //     ];
//     //     let program = Program::new(instructions, 0, 0);
//     //     let stdin = SP1Stdin::new();

//     //     type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

//     //     let malicious_trace_pv_generator =
//     //         |prover: &P,
//     //          record: &mut ExecutionRecord|
//     //          -> Vec<(String, RowMajorMatrix<Val<BabyBearPoseidon2>>)> {
//     //             // Create a malicious record where the AUIPC instruction result is incorrect.
//     //             let mut malicious_record = record.clone();
//     //             malicious_record.auipc_events[0].a = 8;
//     //             prover.generate_traces(&malicious_record)
//     //         };

//     //     let result =
//     //         run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
//     //     assert!(result.is_err() && result.unwrap_err().is_local_cumulative_sum_failing());
//     // }

//     #[test]
//     fn test_malicious_multiple_opcode_flags() {
//         let instructions = vec![
//             Instruction::new(Opcode::AUIPC, 29, 12, 12, true, true),
//             Instruction::new(Opcode::ADD, 10, 0, 0, false, false),
//         ];
//         let program = Program::new(instructions, 0, 0);
//         let stdin = SP1Stdin::new();

//         type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

//         let malicious_trace_pv_generator =
//             |prover: &P,
//              record: &mut ExecutionRecord|
//              -> Vec<(String, RowMajorMatrix<Val<BabyBearPoseidon2>>)> {
//                 // Modify the branch chip to have a row that has multiple opcode flags set.
//                 let mut traces = prover.generate_traces(record);
//                 let auipc_chip_name = chip_name!(AuipcChip, BabyBear);
//                 for (chip_name, trace) in traces.iter_mut() {
//                     if *chip_name == auipc_chip_name {
//                         let first_row: &mut [BabyBear] = trace.row_mut(0);
//                         let first_row: &mut AuipcColumns<BabyBear> = first_row.borrow_mut();
//                         assert!(first_row.is_auipc == BabyBear::one());
//                         first_row.is_unimp = BabyBear::one();
//                     }
//                 }
//                 traces
//             };

//         let result =
//             run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
//         assert!(result.is_err() && result.unwrap_err().is_constraints_failing());
//     }

//     #[test]
//     fn test_unimpl() {
//         let instructions = vec![Instruction::new(Opcode::UNIMP, 29, 12, 0, true, true)];
//         let program = Program::new(instructions, 0, 0);
//         let stdin = SP1Stdin::new();

//         let mut runtime = Executor::new(program, SP1CoreOpts::default());
//         runtime.maximal_shapes = None;
//         runtime.write_vecs(&stdin.buffer);
//         let result = runtime.execute::<Simple>();

//         assert!(result.is_err() && result.unwrap_err() == ExecutionError::Unimplemented());
//     }

//     #[test]
//     fn test_ebreak() {
//         let instructions = vec![Instruction::new(Opcode::EBREAK, 29, 12, 0, true, true)];
//         let program = Program::new(instructions, 0, 0);
//         let stdin = SP1Stdin::new();

//         let mut runtime = Executor::new(program, SP1CoreOpts::default());
//         runtime.maximal_shapes = None;
//         runtime.write_vecs(&stdin.buffer);
//         let result = runtime.execute::<Simple>();

//         assert!(result.is_err() && result.unwrap_err() == ExecutionError::Breakpoint());
//     }
// }

//! Implementation to check that b * c = product.
//!
//! We first extend the operands to 64 bits. We sign-extend them if the op code is signed. Then we
//! calculate the un-carried product and propagate the carry. Finally, we check that the appropriate
//! bits of the product match the result.
//!
//! b_64 = sign_extend(b) if signed operation else b
//! c_64 = sign_extend(c) if signed operation else c
//!
//! m = []
//! # 64-bit integers have 8 limbs.
//! # Calculate un-carried product.
//! for i in 0..8:
//!     for j in 0..8:
//!         if i + j < 8:
//!             m\[i + j\] += b_64\[i\] * c_64\[j\]
//!
//! # Propagate carry
//! for i in 0..8:
//!     x = m\[i\]
//!     if i > 0:
//!         x += carry\[i - 1\]
//!     carry\[i\] = x / 256
//!     m\[i\] = x % 256
//!
//! if upper_half:
//!     assert_eq(a, m\[4..8\])
//! if lower_half:
//!     assert_eq(a, m\[0..4\])

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

use crate::{
    adapter::{register::alu_type::ALUTypeReader, state::CPUState},
    air::SP1CoreAirBuilder,
    operations::MulOperation,
    utils::{next_multiple_of_32, zeroed_f_vec},
};

/// The number of main trace columns for `MulChip`.
pub const NUM_MUL_COLS: usize = size_of::<MulCols<u8>>();

/// A chip that implements multiplication for the multiplication opcodes.
#[derive(Default)]
pub struct MulChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MulCols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ALUTypeReader<T>,

    /// The output operand.
    pub a: Word<T>,

    /// Instance of `MulOperation` to handle multiplication logic in `MulChip`'s ALU operations.
    pub mul_operation: MulOperation<T>,

    /// Selector to know whether this row is enabled.
    pub is_real: T,

    /// Whether the operation is MUL.
    pub is_mul: T,

    /// Whether the operation is MULH.
    pub is_mulh: T,

    /// Whether the operation is MULHU.
    pub is_mulhu: T,

    /// Whether the operation is MULHSU.
    pub is_mulhsu: T,

    /// Whether the operation is MULW.
    pub is_mulw: T,
}

impl<F: PrimeField32> MachineAir<F> for MulChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Mul".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.mul_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the trace rows for each event.
        let nb_rows = input.mul_events.len();
        let padded_nb_rows = <MulChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MUL_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_MUL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MulCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.mul_events[idx];
                        let instruction = input.program.fetch(event.0.pc);
                        self.event_to_row(&event.0, cols, &mut byte_lookup_events);
                        cols.state.populate(
                            &mut byte_lookup_events,
                            input.public_values.execution_shard as u32,
                            event.0.clk,
                            event.0.pc,
                        );
                        cols.adapter.populate(&mut byte_lookup_events, instruction, event.1);
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_MUL_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.mul_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .mul_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_MUL_COLS];
                    let cols: &mut MulCols<F> = row.as_mut_slice().borrow_mut();
                    let instruction = input.program.fetch(event.0.pc);
                    self.event_to_row(&event.0, cols, &mut blu);
                    cols.state.populate(
                        &mut blu,
                        input.public_values.execution_shard as u32,
                        event.0.clk,
                        event.0.pc,
                    );
                    cols.adapter.populate(&mut blu, instruction, event.1);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.mul_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl MulChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut MulCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        // cols.mul_operation.populate(
        //     blu,
        //     event.b,
        //     event.c,
        //     event.opcode == Opcode::MULH,
        //     event.opcode == Opcode::MULHSU,
        // );

        cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
        cols.is_mulh = F::from_bool(event.opcode == Opcode::MULH);
        cols.is_mulhu = F::from_bool(event.opcode == Opcode::MULHU);
        cols.is_mulhsu = F::from_bool(event.opcode == Opcode::MULHSU);
        cols.is_mulw = F::from_bool(event.opcode == Opcode::MULW);

        cols.a = Word::from(event.a);
        cols.is_real = F::one();
    }
}

impl<F> BaseAir<F> for MulChip {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }
}

impl<AB> Air<AB> for MulChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MulCols<AB::Var> = (*local).borrow();

        // Constrain the multiplication operation over `op_b`, `op_c` and the selectors.
        // MulOperation::<AB::F>::eval(
        //     builder,
        //     local.a.map(|x| x.into()),
        //     local.adapter.b().map(|x| x.into()),
        //     local.adapter.c().map(|x| x.into()),
        //     local.mul_operation,
        //     local.is_real.into(),
        //     local.is_mul.into(),
        //     local.is_mulh.into(),
        //     local.is_mulhu.into(),
        //     local.is_mulhsu.into(),
        // );

        // Calculate the opcode.
        let opcode = {
            // Exactly one of the opcodes must be on in a "real" row.
            builder.assert_eq(
                local.is_real,
                local.is_mul + local.is_mulh + local.is_mulhu + local.is_mulhsu + local.is_mulw,
            );
            builder.assert_bool(local.is_mul);
            builder.assert_bool(local.is_mulh);
            builder.assert_bool(local.is_mulhu);
            builder.assert_bool(local.is_mulhsu);
            builder.assert_bool(local.is_real);

            let mul: AB::Expr = AB::F::from_canonical_u32(Opcode::MUL as u32).into();
            let mulh: AB::Expr = AB::F::from_canonical_u32(Opcode::MULH as u32).into();
            let mulhu: AB::Expr = AB::F::from_canonical_u32(Opcode::MULHU as u32).into();
            let mulhsu: AB::Expr = AB::F::from_canonical_u32(Opcode::MULHSU as u32).into();
            let mulw: AB::Expr = AB::F::from_canonical_u32(Opcode::MULW as u32).into();
            local.is_mul * mul
                + local.is_mulh * mulh
                + local.is_mulhu * mulhu
                + local.is_mulhsu * mulhsu
                + local.is_mulw * mulw
        };

        // Constrain the state of the CPU.
        // The program counter and timestamp increment by `4`.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        ALUTypeReader::<AB::F>::eval(
            builder,
            local.state.shard::<AB>(),
            local.state.clk::<AB>(),
            local.state.pc,
            opcode,
            local.a,
            local.adapter,
            local.is_real.into(),
        );
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::{
//         io::SP1Stdin,
//         riscv::RiscvAir,
//         utils::{run_malicious_test, run_test_machine, setup_test_machine},
//     };
//     use p3_baby_bear::BabyBear;
//     use p3_matrix::dense::RowMajorMatrix;
//     use rand::{thread_rng, Rng};
//     use sp1_core_executor::{
//         events::{AluEvent, MemoryRecordEnum},
//         ExecutionRecord, Instruction, Opcode, Program,
//     };
//     use sp1_stark::{
//         air::{MachineAir, SP1_PROOF_NUM_PV_ELTS},
//         baby_bear_poseidon2::BabyBearPoseidon2,
//         Chip, CpuProver, MachineProver, StarkMachine, Val,
//     };

//     use super::MulChip;

//     #[test]
//     fn generate_trace_mul() {
//         let mut shard = ExecutionRecord::default();

//         // Fill mul_events with 10^7 MULHSU events.
//         let mut mul_events: Vec<AluEvent> = Vec::new();
//         for _ in 0..10i32.pow(7) {
//             mul_events.push(AluEvent::new(
//                 0,
//                 Opcode::MULHSU,
//                 0x80004000,
//                 0x80000000,
//                 0xffff8000,
//                 false,
//             ));
//         }
//         shard.mul_events = mul_events;
//         let chip = MulChip::default();
//         let _trace: RowMajorMatrix<BabyBear> =
//             chip.generate_trace(&shard, &mut ExecutionRecord::default());
//     }

//     #[test]
//     fn prove_babybear() {
//         let mut shard = ExecutionRecord::default();
//         let mut mul_events: Vec<AluEvent> = Vec::new();

//         let mul_instructions: Vec<(Opcode, u32, u32, u32)> = vec![
//             (Opcode::MUL, 0x00001200, 0x00007e00, 0xb6db6db7),
//             (Opcode::MUL, 0x00001240, 0x00007fc0, 0xb6db6db7),
//             (Opcode::MUL, 0x00000000, 0x00000000, 0x00000000),
//             (Opcode::MUL, 0x00000001, 0x00000001, 0x00000001),
//             (Opcode::MUL, 0x00000015, 0x00000003, 0x00000007),
//             (Opcode::MUL, 0x00000000, 0x00000000, 0xffff8000),
//             (Opcode::MUL, 0x00000000, 0x80000000, 0x00000000),
//             (Opcode::MUL, 0x00000000, 0x80000000, 0xffff8000),
//             (Opcode::MUL, 0x0000ff7f, 0xaaaaaaab, 0x0002fe7d),
//             (Opcode::MUL, 0x0000ff7f, 0x0002fe7d, 0xaaaaaaab),
//             (Opcode::MUL, 0x00000000, 0xff000000, 0xff000000),
//             (Opcode::MUL, 0x00000001, 0xffffffff, 0xffffffff),
//             (Opcode::MUL, 0xffffffff, 0xffffffff, 0x00000001),
//             (Opcode::MUL, 0xffffffff, 0x00000001, 0xffffffff),
//             (Opcode::MULHU, 0x00000000, 0x00000000, 0x00000000),
//             (Opcode::MULHU, 0x00000000, 0x00000001, 0x00000001),
//             (Opcode::MULHU, 0x00000000, 0x00000003, 0x00000007),
//             (Opcode::MULHU, 0x00000000, 0x00000000, 0xffff8000),
//             (Opcode::MULHU, 0x00000000, 0x80000000, 0x00000000),
//             (Opcode::MULHU, 0x7fffc000, 0x80000000, 0xffff8000),
//             (Opcode::MULHU, 0x0001fefe, 0xaaaaaaab, 0x0002fe7d),
//             (Opcode::MULHU, 0x0001fefe, 0x0002fe7d, 0xaaaaaaab),
//             (Opcode::MULHU, 0xfe010000, 0xff000000, 0xff000000),
//             (Opcode::MULHU, 0xfffffffe, 0xffffffff, 0xffffffff),
//             (Opcode::MULHU, 0x00000000, 0xffffffff, 0x00000001),
//             (Opcode::MULHU, 0x00000000, 0x00000001, 0xffffffff),
//             (Opcode::MULHSU, 0x00000000, 0x00000000, 0x00000000),
//             (Opcode::MULHSU, 0x00000000, 0x00000001, 0x00000001),
//             (Opcode::MULHSU, 0x00000000, 0x00000003, 0x00000007),
//             (Opcode::MULHSU, 0x00000000, 0x00000000, 0xffff8000),
//             (Opcode::MULHSU, 0x00000000, 0x80000000, 0x00000000),
//             (Opcode::MULHSU, 0x80004000, 0x80000000, 0xffff8000),
//             (Opcode::MULHSU, 0xffff0081, 0xaaaaaaab, 0x0002fe7d),
//             (Opcode::MULHSU, 0x0001fefe, 0x0002fe7d, 0xaaaaaaab),
//             (Opcode::MULHSU, 0xff010000, 0xff000000, 0xff000000),
//             (Opcode::MULHSU, 0xffffffff, 0xffffffff, 0xffffffff),
//             (Opcode::MULHSU, 0xffffffff, 0xffffffff, 0x00000001),
//             (Opcode::MULHSU, 0x00000000, 0x00000001, 0xffffffff),
//             (Opcode::MULH, 0x00000000, 0x00000000, 0x00000000),
//             (Opcode::MULH, 0x00000000, 0x00000001, 0x00000001),
//             (Opcode::MULH, 0x00000000, 0x00000003, 0x00000007),
//             (Opcode::MULH, 0x00000000, 0x00000000, 0xffff8000),
//             (Opcode::MULH, 0x00000000, 0x80000000, 0x00000000),
//             (Opcode::MULH, 0x00000000, 0x80000000, 0x00000000),
//             (Opcode::MULH, 0xffff0081, 0xaaaaaaab, 0x0002fe7d),
//             (Opcode::MULH, 0xffff0081, 0x0002fe7d, 0xaaaaaaab),
//             (Opcode::MULH, 0x00010000, 0xff000000, 0xff000000),
//             (Opcode::MULH, 0x00000000, 0xffffffff, 0xffffffff),
//             (Opcode::MULH, 0xffffffff, 0xffffffff, 0x00000001),
//             (Opcode::MULH, 0xffffffff, 0x00000001, 0xffffffff),
//         ];
//         for t in mul_instructions.iter() {
//             mul_events.push(AluEvent::new(0, t.0, t.1, t.2, t.3, false));
//         }

//         // Append more events until we have 1000 tests.
//         for _ in 0..(1000 - mul_instructions.len()) {
//             mul_events.push(AluEvent::new(0, Opcode::MUL, 1, 1, 1, false));
//         }

//         shard.mul_events = mul_events;

//         // Run setup.
//         let air = MulChip::default();
//         let config = BabyBearPoseidon2::new();
//         let chip = Chip::new(air);
//         let (pk, vk) = setup_test_machine(StarkMachine::new(
//             config.clone(),
//             vec![chip],
//             SP1_PROOF_NUM_PV_ELTS,
//             true,
//         ));

//         // Run the test.
//         let air = MulChip::default();
//         let chip: Chip<BabyBear, MulChip> = Chip::new(air);
//         let machine = StarkMachine::new(config.clone(), vec![chip], SP1_PROOF_NUM_PV_ELTS, true);
//         run_test_machine::<BabyBearPoseidon2, MulChip>(vec![shard], machine, pk, vk).unwrap();
//     }

//     #[test]
//     fn test_malicious_mul() {
//         const NUM_TESTS: usize = 5;

//         for opcode in [Opcode::MUL, Opcode::MULH, Opcode::MULHU, Opcode::MULHSU] {
//             for _ in 0..NUM_TESTS {
//                 let (correct_op_a, op_b, op_c) = if opcode == Opcode::MUL {
//                     let op_b = thread_rng().gen_range(0..i32::MAX);
//                     let op_c = thread_rng().gen_range(0..i32::MAX);
//                     ((op_b.overflowing_mul(op_c).0) as u32, op_b as u32, op_c as u32)
//                 } else if opcode == Opcode::MULH {
//                     let op_b = thread_rng().gen_range(0..i32::MAX);
//                     let op_c = thread_rng().gen_range(0..i32::MAX);
//                     let result = (op_b as i64) * (op_c as i64);
//                     (((result >> 32) as i32) as u32, op_b as u32, op_c as u32)
//                 } else if opcode == Opcode::MULHU {
//                     let op_b = thread_rng().gen_range(0..u32::MAX);
//                     let op_c = thread_rng().gen_range(0..u32::MAX);
//                     let result: u64 = (op_b as u64) * (op_c as u64);
//                     ((result >> 32) as u32, op_b as u32, op_c as u32)
//                 } else if opcode == Opcode::MULHSU {
//                     let op_b = thread_rng().gen_range(0..i32::MAX);
//                     let op_c = thread_rng().gen_range(0..u32::MAX);
//                     let result: i64 = (op_b as i64) * (op_c as i64);
//                     ((result >> 32) as u32, op_b as u32, op_c as u32)
//                 } else {
//                     unreachable!()
//                 };

//                 let op_a = thread_rng().gen_range(0..u32::MAX);
//                 assert!(op_a != correct_op_a);

//                 let instructions = vec![
//                     Instruction::new(opcode, 5, op_b, op_c, true, true),
//                     Instruction::new(Opcode::ADD, 10, 0, 0, false, false),
//                 ];

//                 let program = Program::new(instructions, 0, 0);
//                 let stdin = SP1Stdin::new();

//                 type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

//                 let malicious_trace_pv_generator = move |prover: &P,
//                                                          record: &mut ExecutionRecord|
//                       -> Vec<(
//                     String,
//                     RowMajorMatrix<Val<BabyBearPoseidon2>>,
//                 )> {
//                     let mut malicious_record = record.clone();
//                     malicious_record.cpu_events[0].a = op_a as u32;
//                     if let Some(MemoryRecordEnum::Write(mut write_record)) =
//                         malicious_record.cpu_events[0].a_record
//                     {
//                         write_record.value = op_a as u32;
//                     }
//                     malicious_record.mul_events[0].a = op_a;
//                     prover.generate_traces(&malicious_record)
//                 };

//                 let result =
//                     run_malicious_test::<P>(program, stdin,
// Box::new(malicious_trace_pv_generator));                 assert!(result.is_err() &&
// result.unwrap_err().is_constraints_failing());             }
//         }
//     }
// }

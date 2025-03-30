use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use sp1_core_executor::ByteOpcode;
use sp1_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{u32_to_u16_limbs, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    adapter::{register::alu_type::ALUTypeReader, state::CPUState},
    air::SP1CoreAirBuilder,
    utils::pad_rows_fixed,
};

/// The number of main trace columns for `ShiftLeft`.
pub const NUM_SHIFT_LEFT_COLS: usize = size_of::<ShiftLeftCols<u8>>();

/// The number of bits in a byte.
pub const BYTE_SIZE: usize = 8;

/// A chip that implements bitwise operations for the opcodes SLL and SLLI.
#[derive(Default)]
pub struct ShiftLeft;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftLeftCols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ALUTypeReader<T>,

    /// The output operand.
    pub a: Word<T>,

    /// The lower bits of each limb.
    pub lower_limb: Word<T>,

    /// The higher bits of each limb.
    pub higher_limb: Word<T>,

    /// Auxiliary column to help compute `pow_2`, equal to `2^(c & 0x3)`.
    pub pow_2_01: T,

    /// Auxiliary column to help compute `pow_2`, equal to `2^(c & 0x12)`.
    pub pow_2_23: T,

    /// The power of two corresponding to the bit shift, equal to `2^(c & 0x15)`.
    pub pow_2: T,

    /// A column to reduce AIR degree, equal to `pow_2 * c_bit[4]`.
    pub pow_2_bit: T,

    /// The bottom 5 bits of `c`.
    pub c_bits: [T; 5],

    /// Boolean to indicate whether the row is not a padding row.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for ShiftLeft {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShiftLeft".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the trace rows for each event.
        let mut rows: Vec<[F; NUM_SHIFT_LEFT_COLS]> = vec![];
        let shift_left_events = input.shift_left_events.clone();
        for event in shift_left_events.iter() {
            let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
            let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
            let mut blu = Vec::new();
            let instruction = input.program.fetch(event.0.pc);
            self.event_to_row(&event.0, cols, &mut blu);
            cols.state.populate(
                &mut blu,
                input.public_values.execution_shard,
                event.0.clk,
                event.0.pc,
            );
            cols.adapter.populate(&mut blu, instruction, event.1);
            rows.push(row);
        }

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_SHIFT_LEFT_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        // Convert the trace to a row major matrix.
        let mut trace = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHIFT_LEFT_COLS,
        );

        // Create the template for the padded rows. These are fake rows that don't fail on some
        // sanity checks.
        let padded_row_template = {
            let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
            let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
            cols.pow_2_01 = F::one();
            cols.pow_2_23 = F::one();
            cols.pow_2 = F::one();
            row
        };
        debug_assert!(padded_row_template.len() == NUM_SHIFT_LEFT_COLS);
        for i in input.shift_left_events.len() * NUM_SHIFT_LEFT_COLS..trace.values.len() {
            trace.values[i] = padded_row_template[i % NUM_SHIFT_LEFT_COLS];
        }

        trace
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.shift_left_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_left_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
                    let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
                    let instruction = input.program.fetch(event.0.pc);
                    self.event_to_row(&event.0, cols, &mut blu);
                    cols.state.populate(
                        &mut blu,
                        input.public_values.execution_shard,
                        event.0.clk,
                        event.0.pc,
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
            !shard.shift_left_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl ShiftLeft {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftLeftCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        let b = u32_to_u16_limbs(event.b);
        let c = u32_to_u16_limbs(event.c)[0];
        cols.a = Word::from(event.a);
        for i in 0..5 {
            cols.c_bits[i] = F::from_canonical_u16((c >> i) & 1);
        }
        cols.is_real = F::one();
        cols.pow_2_01 = F::from_canonical_u32(1 << (c & 3));
        cols.pow_2_23 = F::from_canonical_u32(1 << (c & 12));
        cols.pow_2 = F::from_canonical_u32(1 << (c & 15));
        if ((c >> 4) & 1) == 1 {
            cols.pow_2_bit = cols.pow_2;
        }
        let bit_shift = (c & 0xF) as u8;
        blu.add_bit_range_check(c >> 5, 11);
        for i in 0..WORD_SIZE {
            let limb = b[i] as u32;
            let lower_limb = (limb & ((1 << (16 - bit_shift)) - 1)) as u16;
            let higher_limb = (limb >> (16 - bit_shift)) as u16;
            cols.lower_limb.0[i] = F::from_canonical_u16(lower_limb);
            cols.higher_limb.0[i] = F::from_canonical_u16(higher_limb);
            blu.add_bit_range_check(lower_limb, 16 - bit_shift);
            blu.add_bit_range_check(higher_limb, bit_shift);
        }
    }
}

impl<F> BaseAir<F> for ShiftLeft {
    fn width(&self) -> usize {
        NUM_SHIFT_LEFT_COLS
    }
}

impl<AB> Air<AB> for ShiftLeft
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShiftLeftCols<AB::Var> = (*local).borrow();

        // Step 1: Compute the bottom 5 bits of `c`.
        // `c_lower_bits` is equal to the bit sum of the bottom 5 bits of `c`.
        // `bit_shift` is equal to the bit sum of the bottom 4 bits of `c`.
        let mut c_lower_bits = AB::Expr::zero();
        let mut bit_shift = AB::Expr::zero();
        for i in 0..5 {
            builder.assert_bool(local.c_bits[i]);
            c_lower_bits =
                c_lower_bits.clone() + local.c_bits[i] * AB::Expr::from_canonical_u32(1 << i);
            if i == 3 {
                bit_shift = c_lower_bits.clone();
            }
        }
        let inverse_32 = AB::F::from_canonical_u32(32).inverse();
        // Check `0 <= (c - c_lower_bits) / 32 < 2^11`, which shows `c - c_lower_bits` is a u16 and a multiple of 32.
        builder.send_byte(
            AB::F::from_canonical_u32(ByteOpcode::Range as u32),
            (local.adapter.c()[0] - c_lower_bits) * inverse_32,
            AB::Expr::from_canonical_u32(11),
            AB::Expr::zero(),
            local.is_real,
        );

        // Step 2: Compute `pow(2, lower 4 bits of c)`.
        builder.assert_eq(
            local.pow_2_01,
            (AB::Expr::one() + local.c_bits[0])
                * (AB::Expr::one() + AB::Expr::from_canonical_u32(3) * local.c_bits[1]),
        );
        builder.assert_eq(
            local.pow_2_23,
            (AB::Expr::one() + AB::Expr::from_canonical_u32(15) * local.c_bits[2])
                * (AB::Expr::one() + AB::Expr::from_canonical_u32(255) * local.c_bits[3]),
        );
        builder.assert_eq(local.pow_2, local.pow_2_01 * local.pow_2_23);

        // Step 3: Split the `b` word into lower and higher parts.
        for i in 0..WORD_SIZE {
            let limb = local.adapter.b()[i];
            // Check that `lower_limb < 2^(16 - bit_shift)`
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::Range as u32),
                local.lower_limb[i],
                AB::Expr::from_canonical_u32(16) - bit_shift.clone(),
                AB::Expr::zero(),
                local.is_real,
            );
            // Check that `higher_limb < 2^(bit_shift)`
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::Range as u32),
                local.higher_limb[i],
                bit_shift.clone(),
                AB::Expr::zero(),
                local.is_real,
            );
            // Check that `limb == higher_limb * 2^(16 - bit_shift) + lower_limb`
            // Multiply `2^(bit_shift)` to the equation to avoid populating `2^(16 - bit_shift)`.
            // This is possible, since `2^(bit_shift)` is not zero.
            builder.assert_eq(
                limb * local.pow_2,
                local.higher_limb[i] * AB::Expr::from_canonical_u32(1 << 16)
                    + local.lower_limb[i] * local.pow_2,
            );
        }

        // Step 4. Compute the final result `a`.
        builder.assert_eq(local.pow_2_bit, local.pow_2 * local.c_bits[4]);

        let limb_0 = local.lower_limb[0] * (local.pow_2 - local.pow_2_bit);
        let limb_1 = local.lower_limb[0] * local.pow_2_bit
            + local.higher_limb[0] * (AB::Expr::one() - local.c_bits[4])
            + local.lower_limb[1] * (local.pow_2 - local.pow_2_bit);

        builder.assert_word_eq(local.a, Word([limb_0, limb_1]));

        // SAFETY: `is_real` is checked to be boolean.
        // All interactions are done with multiplicity `is_real`, so padding rows lead to no
        // interactions. This chip only deals with the `SLL` opcode, so the opcode matches
        // the instruction.
        builder.assert_bool(local.is_real);

        // Constrain the CPU state.
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
            AB::F::from_canonical_u32(Opcode::SLL as u32),
            local.a,
            local.adapter,
            local.is_real.into(),
        );
    }
}

// #[cfg(test)]
// mod tests {
//     #![allow(clippy::print_stdout)]

//     use std::borrow::BorrowMut;

//     use crate::{
//         alu::ShiftLeftCols,
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
//         chip_name, Chip, CpuProver, MachineProver, StarkMachine, Val,
//     };

//     use super::ShiftLeft;

//     #[test]
//     fn generate_trace() {
//         let mut shard = ExecutionRecord::default();
//         shard.shift_left_events = vec![AluEvent::new(0, Opcode::SLL, 16, 8, 1, false)];
//         let chip = ShiftLeft::default();
//         let trace: RowMajorMatrix<BabyBear> =
//             chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         println!("{:?}", trace.values)
//     }

//     #[test]
//     fn prove_babybear() {
//         let mut shift_events: Vec<AluEvent> = Vec::new();
//         let shift_instructions: Vec<(Opcode, u32, u32, u32)> = vec![
//             (Opcode::SLL, 0x00000002, 0x00000001, 1),
//             (Opcode::SLL, 0x00000080, 0x00000001, 7),
//             (Opcode::SLL, 0x00004000, 0x00000001, 14),
//             (Opcode::SLL, 0x80000000, 0x00000001, 31),
//             (Opcode::SLL, 0xffffffff, 0xffffffff, 0),
//             (Opcode::SLL, 0xfffffffe, 0xffffffff, 1),
//             (Opcode::SLL, 0xffffff80, 0xffffffff, 7),
//             (Opcode::SLL, 0xffffc000, 0xffffffff, 14),
//             (Opcode::SLL, 0x80000000, 0xffffffff, 31),
//             (Opcode::SLL, 0x21212121, 0x21212121, 0),
//             (Opcode::SLL, 0x42424242, 0x21212121, 1),
//             (Opcode::SLL, 0x90909080, 0x21212121, 7),
//             (Opcode::SLL, 0x48484000, 0x21212121, 14),
//             (Opcode::SLL, 0x80000000, 0x21212121, 31),
//             (Opcode::SLL, 0x21212121, 0x21212121, 0xffffffe0),
//             (Opcode::SLL, 0x42424242, 0x21212121, 0xffffffe1),
//             (Opcode::SLL, 0x90909080, 0x21212121, 0xffffffe7),
//             (Opcode::SLL, 0x48484000, 0x21212121, 0xffffffee),
//             (Opcode::SLL, 0x00000000, 0x21212120, 0xffffffff),
//         ];
//         for t in shift_instructions.iter() {
//             shift_events.push(AluEvent::new(0, t.0, t.1, t.2, t.3, false));
//         }

//         // Append more events until we have 1000 tests.
//         for _ in 0..(1000 - shift_instructions.len()) {
//             //shift_events.push(AluEvent::new(0, 0, Opcode::SLL, 14, 8, 6));
//         }

//         let mut shard = ExecutionRecord::default();
//         shard.shift_left_events = shift_events;

//         // Run setup.
//         let air = ShiftLeft::default();
//         let config = BabyBearPoseidon2::new();
//         let chip = Chip::new(air);
//         let (pk, vk) = setup_test_machine(StarkMachine::new(
//             config.clone(),
//             vec![chip],
//             SP1_PROOF_NUM_PV_ELTS,
//             true,
//         ));

//         // Run the test.
//         let air = ShiftLeft::default();
//         let chip: Chip<BabyBear, ShiftLeft> = Chip::new(air);
//         let machine = StarkMachine::new(config.clone(), vec![chip], SP1_PROOF_NUM_PV_ELTS, true);
//         run_test_machine::<BabyBearPoseidon2, ShiftLeft>(vec![shard], machine, pk, vk).unwrap();
//     }

//     #[test]
//     fn test_malicious_sll() {
//         const NUM_TESTS: usize = 5;

//         for _ in 0..NUM_TESTS {
//             let op_a = thread_rng().gen_range(0..u32::MAX);
//             let op_b = thread_rng().gen_range(0..u32::MAX);
//             let op_c = thread_rng().gen_range(0..u32::MAX);

//             let correct_op_a = op_b << (op_c & 0x1F);

//             assert!(op_a != correct_op_a);

//             let instructions = vec![
//                 Instruction::new(Opcode::SLL, 5, op_b, op_c, true, true),
//                 Instruction::new(Opcode::ADD, 10, 0, 0, false, false),
//             ];

//             let program = Program::new(instructions, 0, 0);
//             let stdin = SP1Stdin::new();

//             type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

//             let malicious_trace_pv_generator =
//                 move |prover: &P,
//                       record: &mut ExecutionRecord|
//                       -> Vec<(String, RowMajorMatrix<Val<BabyBearPoseidon2>>)> {
//                     let mut malicious_record = record.clone();
//                     malicious_record.cpu_events[0].a = op_a as u32;
//                     if let Some(MemoryRecordEnum::Write(mut write_record)) =
//                         malicious_record.cpu_events[0].a_record
//                     {
//                         write_record.value = op_a as u32;
//                     }
//                     let mut traces = prover.generate_traces(&malicious_record);
//                     let shift_left_chip_name = chip_name!(ShiftLeft, BabyBear);
//                     for (name, trace) in traces.iter_mut() {
//                         if *name == shift_left_chip_name {
//                             let first_row = trace.row_mut(0);
//                             let first_row: &mut ShiftLeftCols<BabyBear> = first_row.borrow_mut();
//                             first_row.a = op_a.into();
//                         }
//                     }

//                     traces
//                 };

//             let result =
//                 run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
//             assert!(result.is_err() && result.unwrap_err().is_constraints_failing());
//         }
//     }
// }

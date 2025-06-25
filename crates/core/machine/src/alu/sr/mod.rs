use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_CLK_INC, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{u32_to_u16_limbs, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    adapter::{
        register::alu_type::{ALUTypeReader, ALUTypeReaderInput},
        state::CPUState,
    },
    air::{SP1CoreAirBuilder, SP1Operation},
    operations::{U16MSBOperation, U16MSBOperationInput},
    utils::{next_multiple_of_32, zeroed_f_vec},
};

/// The number of main trace columns for `ShiftRightChip`.
pub const NUM_SHIFT_RIGHT_COLS: usize = size_of::<ShiftRightCols<u8>>();

/// A chip that implements bitwise operations for the opcodes SRL and SRA.
#[derive(Default)]
pub struct ShiftRightChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftRightCols<T> {
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

    /// Auxiliary column to help compute `pow_2`, equal to `2^(3 - (c & 3))`.
    pub pow_2_01: T,

    /// Auxiliary column to help compute `pow_2`, equal to `2^(12 - (c & 12))`.
    pub pow_2_23: T,

    /// The power of two corresponding to the bit shift, equal to `2^(16 - (c & 15))`.
    pub pow_2: T,

    /// A column to reduce AIR degree, equal to `pow_2 * c_bit[4]`
    pub pow_2_bit: T,

    /// The bottom 5 bits of `c`.
    pub c_bits: [T; 5],

    /// The most significant bit of `b`.
    pub b_msb: U16MSBOperation<T>,

    /// If the opcode is SRL.
    pub is_srl: T,

    /// If the opcode is SRA.
    pub is_sra: T,
}

impl<F: PrimeField32> MachineAir<F> for ShiftRightChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShiftRight".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.shift_right_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the trace rows for each event.
        let nb_rows = input.shift_right_events.len();
        let padded_nb_rows = <ShiftRightChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SHIFT_RIGHT_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_SHIFT_RIGHT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SHIFT_RIGHT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut ShiftRightCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.shift_right_events[idx];
                        let instruction = input.program.fetch(event.0.pc);
                        self.event_to_row(&event.0, cols, &mut byte_lookup_events);
                        cols.state.populate(&mut byte_lookup_events, event.0.clk, event.0.pc);
                        cols.adapter.populate(&mut byte_lookup_events, instruction, event.1);
                    } else {
                        cols.pow_2_01 = F::from_canonical_u32(1 << 3);
                        cols.pow_2_23 = F::from_canonical_u32(1 << 12);
                        cols.pow_2 = F::from_canonical_u32(1 << 16);
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_SHIFT_RIGHT_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.shift_right_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_right_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_SHIFT_RIGHT_COLS];
                    let cols: &mut ShiftRightCols<F> = row.as_mut_slice().borrow_mut();
                    let instruction = input.program.fetch(event.0.pc);
                    self.event_to_row(&event.0, cols, &mut blu);
                    cols.state.populate(&mut blu, event.0.clk, event.0.pc);
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
            !shard.shift_right_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl ShiftRightChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftRightCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        let b = u32_to_u16_limbs(event.b);
        let c = u32_to_u16_limbs(event.c)[0];
        cols.a = Word::from(event.a);
        for i in 0..5 {
            cols.c_bits[i] = F::from_canonical_u16((c >> i) & 1);
        }
        cols.is_srl = F::from_bool(event.opcode == Opcode::SRL);
        cols.is_sra = F::from_bool(event.opcode == Opcode::SRA);
        cols.pow_2_01 = F::from_canonical_u32(1 << (3 - (c & 3)));
        cols.pow_2_23 = F::from_canonical_u32(1 << (12 - (c & 12)));
        cols.pow_2 = F::from_canonical_u32(1 << (16 - (c & 15)));
        if ((c >> 4) & 1) == 1 {
            cols.pow_2_bit = cols.pow_2;
        }
        let bit_shift = (c & 0xF) as u8;
        blu.add_bit_range_check(c >> 5, 11);
        for i in 0..WORD_SIZE {
            let limb = b[i] as u32;
            let lower_limb = (limb & ((1 << bit_shift) - 1)) as u16;
            let higher_limb = (limb >> bit_shift) as u16;
            cols.lower_limb.0[i] = F::from_canonical_u16(lower_limb);
            cols.higher_limb.0[i] = F::from_canonical_u16(higher_limb);
            blu.add_bit_range_check(lower_limb, bit_shift);
            blu.add_bit_range_check(higher_limb, 16 - bit_shift);
        }
        if event.opcode == Opcode::SRA {
            cols.b_msb.populate_msb(blu, b[1]);
        }
    }
}

impl<F> BaseAir<F> for ShiftRightChip {
    fn width(&self) -> usize {
        NUM_SHIFT_RIGHT_COLS
    }
}

impl<AB> Air<AB> for ShiftRightChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShiftRightCols<AB::Var> = (*local).borrow();

        let is_real = local.is_srl + local.is_sra;

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
        // Check `0 <= (c - c_lower_bits) / 32 < 2^11`, which shows `c - c_lower_bits` is a u16 and
        // a multiple of 32.
        builder.send_byte(
            AB::F::from_canonical_u32(ByteOpcode::Range as u32),
            (local.adapter.c()[0] - c_lower_bits) * inverse_32,
            AB::Expr::from_canonical_u32(11),
            AB::Expr::zero(),
            is_real.clone(),
        );

        // Step 2: Compute `pow(2, 16 - (lower 4 bits of c))`.
        builder.assert_eq(
            local.pow_2_01,
            (AB::Expr::from_canonical_u32(2) - local.c_bits[0])
                * (AB::Expr::from_canonical_u32(4)
                    - AB::Expr::from_canonical_u32(3) * local.c_bits[1]),
        );
        builder.assert_eq(
            local.pow_2_23,
            (AB::Expr::from_canonical_u32(16) - AB::Expr::from_canonical_u32(15) * local.c_bits[2])
                * (AB::Expr::from_canonical_u32(256)
                    - AB::Expr::from_canonical_u32(255) * local.c_bits[3]),
        );
        builder.assert_eq(
            local.pow_2,
            AB::Expr::from_canonical_u32(2) * local.pow_2_01 * local.pow_2_23,
        );

        // Step 3: Split the `b` word into lower and higher parts.
        for i in 0..WORD_SIZE {
            let limb = local.adapter.b()[i];
            // Check that `lower_limb < 2^(bit_shift)`
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::Range as u32),
                local.lower_limb[i],
                bit_shift.clone(),
                AB::Expr::zero(),
                is_real.clone(),
            );
            // Check that `higher_limb < 2^(16 - bit_shift)`
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::Range as u32),
                local.higher_limb[i],
                AB::Expr::from_canonical_u32(16) - bit_shift.clone(),
                AB::Expr::zero(),
                is_real.clone(),
            );
            // Check that `limb == higher_limb * 2^(bit_shift) + lower_limb`
            // Multiply `2^(16 - bit_shift)` to the equation to avoid populating `2^(bit_shift)`.
            // This is possible, since `2^(16 - bit_shift)` is not zero.
            builder.assert_eq(
                limb * local.pow_2,
                local.higher_limb[i] * AB::Expr::from_canonical_u32(1 << 16)
                    + local.lower_limb[i] * local.pow_2,
            );
        }

        // Step 4. Compute the MSB of `b`.
        <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
            builder,
            U16MSBOperationInput::<AB>::new(
                local.adapter.b().0[1].into(),
                local.b_msb,
                local.is_sra.into(),
            ),
        );
        // The sign of `b` should be considered positive if the opcode is SRL.
        builder.when_not(local.is_sra).assert_zero(local.b_msb.msb);

        // Step 5. Compute the final result `a`.
        builder.assert_eq(local.pow_2_bit, local.pow_2 * local.c_bits[4]);

        let limb_0 = local.higher_limb[0] * (AB::Expr::one() - local.c_bits[4])
            + local.lower_limb[1] * (local.pow_2 - local.pow_2_bit)
            + local.higher_limb[1] * local.c_bits[4]
            + (AB::Expr::from_canonical_u32(1 << 16) * local.c_bits[4] - local.pow_2_bit)
                * local.b_msb.msb;
        let limb_1 = (local.higher_limb[1]
            + AB::Expr::from_canonical_u32(1 << 16) * local.b_msb.msb)
            * (AB::Expr::one() - local.c_bits[4])
            - (local.pow_2 - local.pow_2_bit) * local.b_msb.msb
            + local.c_bits[4] * local.b_msb.msb * AB::Expr::from_canonical_u16(u16::MAX);

        builder.assert_word_eq(local.a, Word([limb_0, limb_1]));

        // SAFETY: All selectors `is_srl`, `is_sra` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real = is_srl + is_sra` is
        // boolean. All interactions are done with multiplicity `is_real`.
        // Therefore, the `opcode` matches the corresponding opcode.

        // Check that the operation flags are boolean.
        builder.assert_bool(local.is_srl);
        builder.assert_bool(local.is_sra);
        builder.assert_bool(is_real.clone());

        let opcode = local.is_srl * AB::F::from_canonical_u32(Opcode::SRL as u32)
            + local.is_sra * AB::F::from_canonical_u32(Opcode::SRA as u32);

        // Constrain the CPU state.
        // The program counter and timestamp increment by `4`.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_CLK_INC),
            is_real.clone(),
        );

        // Constrain the program and register reads.
        let alu_reader_input = ALUTypeReaderInput::<AB, AB::Expr>::new(
            local.state.clk_high::<AB>(),
            local.state.clk_low::<AB>(),
            local.state.pc,
            opcode,
            local.a.map(|x| x.into()),
            local.adapter,
            is_real,
        );
        ALUTypeReader::<AB::F>::eval(builder, alu_reader_input);
    }
}

// #[cfg(test)]
// mod tests {
//     #![allow(clippy::print_stdout)]

//     use std::borrow::BorrowMut;

//     use crate::{
//         alu::ShiftRightCols,
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

//     use super::ShiftRightChip;

//     #[test]
//     fn generate_trace() {
//         let mut shard = ExecutionRecord::default();
//         shard.shift_right_events = vec![AluEvent::new(0, Opcode::SRL, 6, 12, 1, false)];
//         let chip = ShiftRightChip::default();
//         let trace: RowMajorMatrix<BabyBear> =
//             chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         println!("{:?}", trace.values)
//     }

//     #[test]
//     fn prove_babybear() {
//         let shifts = vec![
//             (Opcode::SRL, 0xffff8000, 0xffff8000, 0),
//             (Opcode::SRL, 0x7fffc000, 0xffff8000, 1),
//             (Opcode::SRL, 0x01ffff00, 0xffff8000, 7),
//             (Opcode::SRL, 0x0003fffe, 0xffff8000, 14),
//             (Opcode::SRL, 0x0001ffff, 0xffff8001, 15),
//             (Opcode::SRL, 0xffffffff, 0xffffffff, 0),
//             (Opcode::SRL, 0x7fffffff, 0xffffffff, 1),
//             (Opcode::SRL, 0x01ffffff, 0xffffffff, 7),
//             (Opcode::SRL, 0x0003ffff, 0xffffffff, 14),
//             (Opcode::SRL, 0x00000001, 0xffffffff, 31),
//             (Opcode::SRL, 0x21212121, 0x21212121, 0),
//             (Opcode::SRL, 0x10909090, 0x21212121, 1),
//             (Opcode::SRL, 0x00424242, 0x21212121, 7),
//             (Opcode::SRL, 0x00008484, 0x21212121, 14),
//             (Opcode::SRL, 0x00000000, 0x21212121, 31),
//             (Opcode::SRL, 0x21212121, 0x21212121, 0xffffffe0),
//             (Opcode::SRL, 0x10909090, 0x21212121, 0xffffffe1),
//             (Opcode::SRL, 0x00424242, 0x21212121, 0xffffffe7),
//             (Opcode::SRL, 0x00008484, 0x21212121, 0xffffffee),
//             (Opcode::SRL, 0x00000000, 0x21212121, 0xffffffff),
//             (Opcode::SRA, 0x00000000, 0x00000000, 0),
//             (Opcode::SRA, 0xc0000000, 0x80000000, 1),
//             (Opcode::SRA, 0xff000000, 0x80000000, 7),
//             (Opcode::SRA, 0xfffe0000, 0x80000000, 14),
//             (Opcode::SRA, 0xffffffff, 0x80000001, 31),
//             (Opcode::SRA, 0x7fffffff, 0x7fffffff, 0),
//             (Opcode::SRA, 0x3fffffff, 0x7fffffff, 1),
//             (Opcode::SRA, 0x00ffffff, 0x7fffffff, 7),
//             (Opcode::SRA, 0x0001ffff, 0x7fffffff, 14),
//             (Opcode::SRA, 0x00000000, 0x7fffffff, 31),
//             (Opcode::SRA, 0x81818181, 0x81818181, 0),
//             (Opcode::SRA, 0xc0c0c0c0, 0x81818181, 1),
//             (Opcode::SRA, 0xff030303, 0x81818181, 7),
//             (Opcode::SRA, 0xfffe0606, 0x81818181, 14),
//             (Opcode::SRA, 0xffffffff, 0x81818181, 31),
//         ];
//         let mut shift_events: Vec<AluEvent> = Vec::new();
//         for t in shifts.iter() {
//             shift_events.push(AluEvent::new(0, t.0, t.1, t.2, t.3, false));
//         }
//         let mut shard = ExecutionRecord::default();
//         shard.shift_right_events = shift_events;

//         // Run setup.
//         let air = ShiftRightChip::default();
//         let config = BabyBearPoseidon2::new();
//         let chip = Chip::new(air);
//         let (pk, vk) = setup_test_machine(StarkMachine::new(
//             config.clone(),
//             vec![chip],
//             SP1_PROOF_NUM_PV_ELTS,
//             true,
//         ));

//         // Run the test.
//         let air = ShiftRightChip::default();
//         let chip: Chip<BabyBear, ShiftRightChip> = Chip::new(air);
//         let machine = StarkMachine::new(config.clone(), vec![chip], SP1_PROOF_NUM_PV_ELTS, true);
//         run_test_machine::<BabyBearPoseidon2, ShiftRightChip>(vec![shard], machine, pk, vk)
//             .unwrap();
//     }

//     #[test]
//     fn test_malicious_sr() {
//         const NUM_TESTS: usize = 5;

//         for opcode in [Opcode::SRL, Opcode::SRA] {
//             for _ in 0..NUM_TESTS {
//                 let (correct_op_a, op_b, op_c) = if opcode == Opcode::SRL {
//                     let op_b = thread_rng().gen_range(0..u32::MAX);
//                     let op_c = thread_rng().gen_range(0..u32::MAX);
//                     (op_b >> (op_c & 0x1F), op_b, op_c)
//                 } else if opcode == Opcode::SRA {
//                     let op_b = thread_rng().gen_range(0..i32::MAX);
//                     let op_c = thread_rng().gen_range(0..u32::MAX);
//                     ((op_b >> (op_c & 0x1F)) as u32, op_b as u32, op_c)
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
//                     let mut traces = prover.generate_traces(&malicious_record);
//                     let shift_right_chip_name = chip_name!(ShiftRightChip, BabyBear);
//                     for (name, trace) in traces.iter_mut() {
//                         if *name == shift_right_chip_name {
//                             let first_row = trace.row_mut(0);
//                             let first_row: &mut ShiftRightCols<BabyBear> =
// first_row.borrow_mut();                             first_row.a = op_a.into();
//                         }
//                     }
//                     traces
//                 };

//                 let result =
//                     run_malicious_test::<P>(program, stdin,
// Box::new(malicious_trace_pv_generator));                 assert!(result.is_err() &&
// result.unwrap_err().is_constraints_failing());             }
//         }
//     }
// }

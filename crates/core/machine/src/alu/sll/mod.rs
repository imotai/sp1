use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{u32_to_u16_limbs, u64_to_u16_limbs, WORD_BYTE_SIZE, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    adapter::{register::alu_type::ALUTypeReader, state::CPUState},
    air::SP1CoreAirBuilder,
    operations::{U16MSBOperation, U16toU8Operation},
    utils::{next_multiple_of_32, pad_rows_fixed},
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

    /// The lowerst byte of `c`.
    pub c_bits: [T; 8],

    /// v01 = (c0 + 1) * (3c1 + 1)
    pub v_01: T,

    /// v012 = (c0 + 1) * (3c1 + 1) * (15c2 + 1)
    pub v_012: T,

    /// v012 * c3
    pub v_0123: T,

    /// Flags representing c4 + 2c5.
    pub shift_u16: [T; 4],

    /// The lower bytes of b.
    pub b_lower_bytes: U16toU8Operation<T>,

    /// The overflows of each byte of b after the shift.
    pub top_bits: [T; WORD_BYTE_SIZE],

    /// The result of intermediate shift.
    pub byte_result: [T; WORD_SIZE],

    /// The most significant byte of the result of SLLW.
    pub sllw_msb: U16MSBOperation<T>,

    /// Boolean to indicate whether the row is not a padding row.
    // pub is_real: T,

    /// If the opcode is SLL.
    pub is_sll: T,

    /// If the opcode is SLLW.
    pub is_sllw: T,

    /// If the opcode is SLLIW.
    pub is_slliw: T,
}

impl<F: PrimeField32> MachineAir<F> for ShiftLeft {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShiftLeft".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.shift_left_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
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
                input.public_values.execution_shard as u32,
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

        assert_eq!(rows.len(), <ShiftLeft as MachineAir<F>>::num_rows(self, input).unwrap());

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
            cols.v_01 = F::one();
            cols.v_012 = F::one();
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
                        input.public_values.execution_shard as u32,
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
        let c = u64_to_u16_limbs(event.c)[0];

        if event.opcode == Opcode::SLLW || event.opcode == Opcode::SLLIW {
            let sllw_val = ((event.b as i64) << (c & 0x1f)) as u32;
            let sllw_limbs = u32_to_u16_limbs(sllw_val);
            cols.sllw_msb.populate_msb(blu, sllw_limbs[1]);
        }

        let b_bytes = event.b.to_le_bytes();
        cols.a = Word::from(event.a);
        cols.is_sll = F::from_bool(event.opcode == Opcode::SLL);
        cols.is_sllw = F::from_bool(event.opcode == Opcode::SLLW);
        cols.is_slliw = F::from_bool(event.opcode == Opcode::SLLIW);
        let is_sll = event.opcode == Opcode::SLL;

        let mut bits = [0u8; 8];
        for i in 0..8 {
            bits[i] = ((c >> i) & 1) as u8;
            cols.c_bits[i] = F::from_canonical_u8(bits[i]);
        }
        let c_low_byte = c & 0xFF;

        blu.add_u8_range_checks(&[((c - c_low_byte) / 256) as u8]);
        let v01 = (bits[0] + 1) * (3 * bits[1] + 1);
        cols.v_01 = F::from_canonical_u8(v01);
        let v012 = v01 * (15 * bits[2] + 1);
        cols.v_012 = F::from_canonical_u8(v012);
        let v0123 = v012 * bits[3];
        cols.v_0123 = F::from_canonical_u8(v0123);
        let v012_1_3 = v012 - v0123;

        let shift_amount = bits[4] + 2 * bits[5] * is_sll as u8;

        let mut shift = [0u16; 4];
        for i in 0..4 {
            if i == shift_amount as usize {
                shift[i] = 1;
            }
        }

        let mut top_bits = [0u16; 8];
        for i in 0..8 {
            top_bits[i] =
                b_bytes[i] as u16 >> (8 - bits[0] as u16 - 2 * bits[1] as u16 - 4 * bits[2] as u16);
            cols.top_bits[i] = F::from_canonical_u16(top_bits[i]);
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::SR,
                a: top_bits[i] as u16,
                b: b_bytes[i],
                c: 8 - bits[0] - 2 * bits[1] - 4 * bits[2],
            });
        }

        cols.shift_u16 = shift.map(|x| F::from_canonical_u16(x));

        cols.b_lower_bytes.populate_u16_to_u8_unsafe(blu, event.b);

        let mut v0123_coef = [0i32; 4];
        let mut v012_1_3_coef = [0i32; 4];
        let mut remaining_deg2 = [0i32; 8];
        for i in 0..8 {
            remaining_deg2[i] = -256 * top_bits[i] as i32;
            if i >= 1 {
                remaining_deg2[i] += top_bits[i - 1] as i32;
            }
        }

        for i in 0..4 {
            if i >= 1 {
                v0123_coef[i] += b_bytes[2 * i - 1] as i32;
            }
            v0123_coef[i] += 256 * b_bytes[2 * i] as i32;
            v012_1_3_coef[i] = b_bytes[2 * i] as i32 + 256 * b_bytes[2 * i + 1] as i32;
        }

        let mut u16_result = [0i64; 4];
        for i in 0..4 {
            let mut val = 0i64;
            val += (v0123_coef[i] as i64) * (v0123 as i64);
            val += (v012_1_3_coef[i] as i64) * (v012_1_3 as i64);
            if i >= 1 {
                val += (remaining_deg2[2 * i - 1] as i64) * (bits[3] as i64);
            }
            val += 256 * (remaining_deg2[2 * i] as i64) * (bits[3] as i64);
            val += (remaining_deg2[2 * i] as i64) * (1 - bits[3] as i64);
            val += 256 * (remaining_deg2[2 * i + 1] as i64) * (1 - bits[3] as i64);
            u16_result[i] = val;
            cols.byte_result[i] = F::from_canonical_u16(u16_result[i] as u16);
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

        // SAFETY: `is_real` is checked to be boolean.
        // All interactions are done with multiplicity `is_real`, so padding rows lead to no
        // interactions. This chip only deals with the `SLL` opcode, so the opcode matches
        // the instruction.
        let is_real = local.is_sll + local.is_sllw + local.is_slliw;
        builder.assert_bool(is_real.clone());
        builder.assert_bool(local.is_sll);
        builder.assert_bool(local.is_sllw);
        builder.assert_bool(local.is_slliw);

        // Check that local.c_bits is the bit representation for the low byte of c.
        for i in 0..8 {
            builder.assert_bool(local.c_bits[i]);
        }
        let base = AB::F::from_canonical_u32(1 << 8);
        let mut c_low_byte = AB::Expr::zero();
        for i in 0..8 {
            c_low_byte = c_low_byte + local.c_bits[i] * AB::F::from_canonical_u32(1 << i);
        }
        builder.slice_range_check_u8(
            &[(local.adapter.c()[0] - c_low_byte) * base.inverse()],
            is_real.clone(),
        );

        let b_bytes = U16toU8Operation::<AB::F>::eval_u16_to_u8_unsafe(
            builder,
            local.adapter.b().0.map(|x| x.into()),
            local.b_lower_bytes,
        );

        let eight: AB::Expr = AB::F::from_canonical_u32(8).into();

        let value = eight
            - local.c_bits[0]
            - local.c_bits[1] * AB::F::from_canonical_u32(2)
            - local.c_bits[2] * AB::F::from_canonical_u32(4);

        for i in 0..WORD_BYTE_SIZE {
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::SR as u32),
                local.top_bits[i],
                b_bytes[i].clone(),
                value.clone(),
                is_real.clone(),
            );
        }

        for i in 0..WORD_SIZE {
            builder.when(local.shift_u16[i]).assert_eq(
                local.c_bits[4] + local.c_bits[5] * AB::F::from_canonical_u32(2) * local.is_sll,
                AB::Expr::from_canonical_u32(i as u32),
            );
            builder.assert_bool(local.shift_u16[i]);
        }

        builder.when(is_real.clone()).assert_eq(
            local.shift_u16[0] + local.shift_u16[1] + local.shift_u16[2] + local.shift_u16[3],
            AB::Expr::from_canonical_u32(1),
        );

        let base = AB::Expr::from_canonical_u32(256);

        let v012_1_3 = local.v_012 - local.v_0123;
        let one = AB::F::from_canonical_u32(1);
        let fifteen = AB::F::from_canonical_u32(15);
        let three = AB::F::from_canonical_u32(3);
        builder.assert_eq(local.v_01, (local.c_bits[0] + one) * (local.c_bits[1] * three + one));
        builder.assert_eq(local.v_012, local.v_01 * (local.c_bits[2] * fifteen + one));
        builder.assert_eq(local.v_0123, local.v_012 * local.c_bits[3]);

        let mut v0123_coef = std::array::from_fn::<AB::Expr, 4, _>(|_| AB::Expr::zero());
        let mut v012_1_3_coef = std::array::from_fn::<AB::Expr, 4, _>(|_| AB::Expr::zero());
        let mut remaining_deg2 = std::array::from_fn::<AB::Expr, 8, _>(|_| AB::Expr::zero());
        for i in 0..8 {
            remaining_deg2[i] -= base.clone() * local.top_bits[i];
            if i >= 1 {
                remaining_deg2[i] += local.top_bits[i - 1].into();
            }
        }
        for i in 0..4 {
            if i >= 1 {
                v0123_coef[i] += b_bytes[2 * i - 1].clone();
            }
            v0123_coef[i] += base.clone() * b_bytes[2 * i].clone();
            v012_1_3_coef[i] = b_bytes[2 * i].clone() + base.clone() * b_bytes[2 * i + 1].clone();
        }

        let one_expr = AB::Expr::from_canonical_u32(1);
        for i in 0..4 {
            let mut result = v0123_coef[i].clone() * local.v_0123;
            result += v012_1_3_coef[i].clone() * v012_1_3.clone();
            if i >= 1 {
                result += remaining_deg2[2 * i - 1].clone() * local.c_bits[3];
            }
            result += base.clone() * remaining_deg2[2 * i].clone() * local.c_bits[3];
            result += remaining_deg2[2 * i].clone() * (one_expr.clone() - local.c_bits[3]);
            result += base.clone()
                * remaining_deg2[2 * i + 1].clone()
                * (one_expr.clone() - local.c_bits[3]);
            builder.assert_eq(local.byte_result[i], result);
        }

        for i in 0..WORD_SIZE {
            for j in 0..WORD_SIZE {
                if j < i {
                    builder
                        .when(local.is_sll)
                        .assert_eq(local.shift_u16[i] * local.a[j], AB::Expr::zero());
                } else {
                    builder.when(local.is_sll).assert_eq(
                        local.shift_u16[i] * (local.a[j] - local.byte_result[j - i]),
                        AB::Expr::zero(),
                    );
                }
            }
        }

        for i in 0..WORD_SIZE / 2 {
            for j in 0..WORD_SIZE / 2 {
                if j < i {
                    builder
                        .when(local.is_sllw + local.is_slliw)
                        .assert_eq(local.shift_u16[i] * local.a[j], AB::Expr::zero());
                } else {
                    builder.when(local.is_sllw + local.is_slliw).assert_eq(
                        local.shift_u16[i] * (local.a[j] - local.byte_result[j - i]),
                        AB::Expr::zero(),
                    );
                }
            }
        }
        let u16_max = AB::F::from_canonical_u32((1 << 16) - 1);
        for i in WORD_SIZE / 2..WORD_SIZE {
            builder
                .when(local.is_sllw + local.is_slliw)
                .assert_eq(local.sllw_msb.msb * u16_max, local.a[i]);
        }

        U16MSBOperation::<AB::F>::eval_msb(
            builder,
            local.a[1].into(),
            local.sllw_msb,
            local.is_sllw + local.is_slliw,
        );

        let opcode = local.is_sll * AB::F::from_canonical_u32(Opcode::SLL as u32)
            + local.is_sllw * AB::F::from_canonical_u32(Opcode::SLLW as u32)
            + local.is_slliw * AB::F::from_canonical_u32(Opcode::SLLIW as u32);

        // Constrain the CPU state.
        // The program counter and timestamp increment by `4`.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            is_real.clone(),
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
            is_real.clone(),
        );
    }
}

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

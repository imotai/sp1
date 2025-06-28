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
    ByteOpcode, ExecutionRecord, Opcode, Program, CLK_INC, PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{u32_to_u16_limbs, u64_to_u16_limbs, WORD_BYTE_SIZE, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    adapter::{
        register::alu_type::{ALUTypeReader, ALUTypeReaderInput},
        state::{CPUState, CPUStateInput},
    },
    air::{SP1CoreAirBuilder, SP1Operation},
    operations::{U16MSBOperation, U16MSBOperationInput, U16toU8Operation},
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

    /// The most significant byte of the result of SRLW/SRAW/SRLIW/SRAIW
    pub srw_msb: U16MSBOperation<T>,

    /// The input operand (truncated when SRAW/SRAIW/SRLIW/SRLW)
    pub b: Word<T>,

    /// The bottom 8 bits of `c`.
    pub c_bits: [T; 8],

    /// The most significant bit of `b`.
    pub b_msb: U16MSBOperation<T>,

    /// SRA msb * v0123
    pub sra_msb_v0123: T,

    /// v0123
    pub v_0123: T,

    /// v012
    pub v_012: T,

    /// v01
    pub v_01: T,

    /// The lower bytes of `b`.
    pub b_lower_bytes: U16toU8Operation<T>,

    /// The top bits of `b`.
    pub top_bits: [T; WORD_BYTE_SIZE],

    /// The result of the byte-shift.
    pub byte_result: [T; WORD_SIZE],

    /// The shift amount.
    pub shift_u16: [T; 4],

    /// If the opcode is SRL.
    pub is_srl: T,

    /// If the opcode is SRA.
    pub is_sra: T,

    /// If the opcode is SRLW.
    pub is_srlw: T,

    /// If the opcode is SRAW.
    pub is_sraw: T,

    /// If the opcode is SRLIW.
    pub is_srliw: T,

    /// If the opcode is SRAIW.
    pub is_sraiw: T,
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
                        let instruction = input.program.fetch(event.0.pc_rel);
                        self.event_to_row(&event.0, cols, &mut byte_lookup_events);
                        cols.state.populate(&mut byte_lookup_events, event.0.clk, event.0.pc_rel);
                        cols.adapter.populate(&mut byte_lookup_events, instruction, event.1);
                    } else {
                        cols.v_01 = F::from_canonical_u32(16);
                        cols.v_012 = F::from_canonical_u32(256);
                        cols.v_0123 = F::from_canonical_u32(65536);
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
                    let instruction = input.program.fetch(event.0.pc_rel);
                    self.event_to_row(&event.0, cols, &mut blu);
                    cols.state.populate(&mut blu, event.0.clk, event.0.pc_rel);
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
        let mut b = u64_to_u16_limbs(event.b);
        let c = u64_to_u16_limbs(event.c)[0];
        cols.a = Word::from(event.a);
        cols.is_srl = F::from_bool(event.opcode == Opcode::SRL);
        cols.is_sra = F::from_bool(event.opcode == Opcode::SRA);
        cols.is_srlw = F::from_bool(event.opcode == Opcode::SRLW);
        cols.is_sraw = F::from_bool(event.opcode == Opcode::SRAW);
        cols.is_srliw = F::from_bool(event.opcode == Opcode::SRLIW);
        cols.is_sraiw = F::from_bool(event.opcode == Opcode::SRAIW);
        let is_word = event.opcode == Opcode::SRLW
            || event.opcode == Opcode::SRAW
            || event.opcode == Opcode::SRLIW
            || event.opcode == Opcode::SRAIW;
        if is_word {
            b[2] = 0;
            b[3] = 0;
        }
        let b_u64 = if is_word { event.b & 0xFFFF_FFFF } else { event.b };
        cols.b = Word::from(b_u64);
        let not_word = !is_word;
        let mut bits = [0u8; 8];
        for i in 0..8 {
            bits[i] = ((c >> i) & 1) as u8;
            cols.c_bits[i] = F::from_canonical_u8(bits[i]);
        }
        let c_low_byte = c & 0xFF;
        blu.add_u8_range_checks(&[((c - c_low_byte) / 256) as u8]);

        let v01 = 2 * ((1 - bits[0] as u32) + 1) * (3 * (1 - bits[1] as u32) + 1);
        cols.v_01 = F::from_canonical_u32(v01);
        let v012 = v01 * (15 * (1 - bits[2] as u32) + 1);
        cols.v_012 = F::from_canonical_u32(v012);
        let v0123 = v012 * (255 * (1 - bits[3] as u32) + 1);
        cols.v_0123 = F::from_canonical_u32(v0123);
        let v012b_3 = (256 * v012 - v0123) / 255;
        assert!((256 * v012 - v0123) % 255 == 0);

        cols.b_lower_bytes.populate_u16_to_u8_unsafe(blu, b_u64);

        if event.opcode == Opcode::SRA {
            cols.b_msb.populate_msb(blu, b[3]);
        }
        if event.opcode == Opcode::SRAW || event.opcode == Opcode::SRAIW {
            cols.b_msb.populate_msb(blu, b[1]);
        }
        cols.sra_msb_v0123 = cols.b_msb.msb * cols.v_0123; // if not SRA, b_msb.msb == 0

        if event.opcode == Opcode::SRLW || event.opcode == Opcode::SRLIW {
            let srlw_val = (event.b as u32) >> ((event.c & 0x1f) as u32);
            let srlw_limbs = u32_to_u16_limbs(srlw_val);
            cols.srw_msb.populate_msb(blu, srlw_limbs[1]);
        }
        if event.opcode == Opcode::SRAW || event.opcode == Opcode::SRAIW {
            let sraw_val = (event.b as i32).wrapping_shr(((event.c as i64 & 0x1f) as i32) as u32);
            let sraw_limbs = u32_to_u16_limbs(sraw_val as u32);
            cols.srw_msb.populate_msb(blu, sraw_limbs[1]);
        }

        let b_bytes = b_u64.to_le_bytes();

        let mut top_bits = [0u8; 8];
        for i in 0..8 {
            top_bits[i] = b_bytes[i] >> (bits[0] + 2 * bits[1] + 4 * bits[2]);
            cols.top_bits[i] = F::from_canonical_u8(top_bits[i]);
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::SR,
                a: top_bits[i] as u16,
                b: b_bytes[i],
                c: bits[0] + 2 * bits[1] + 4 * bits[2],
            });
        }

        let mut v0123_coef = [0i32; 4];
        let mut v012_1_3_coef = [0i32; 4];
        let mut remaining_deg2 = [0i32; 8];
        for i in 0..8 {
            remaining_deg2[i] = top_bits[i] as i32;
            if i + 1 < 8 {
                remaining_deg2[i] -= 256 * top_bits[i + 1] as i32;
            }
        }
        for i in 0..4 {
            if i < 3 {
                v0123_coef[i] += b_bytes[2 * i + 2] as i32;
                v0123_coef[i] += 256 * b_bytes[2 * i + 3] as i32;
                v012_1_3_coef[i] += 256 * b_bytes[2 * i + 2] as i32;
            }
            if i < 4 {
                v012_1_3_coef[i] += b_bytes[2 * i + 1] as i32;
            }
        }

        let mut u16_result = [0i64; 4];
        for i in 0..4 {
            u16_result[i] += v0123_coef[i] as i64 * v012b_3 as i64;
            u16_result[i] += v012_1_3_coef[i] as i64 * ((v012 as u32) - v012b_3) as i64;
            u16_result[i] += remaining_deg2[2 * i + 1] as i64 * bits[3] as i64;
            if i < 3 {
                u16_result[i] += 256 * remaining_deg2[2 * i + 2] as i64 * bits[3] as i64;
            }
            u16_result[i] += remaining_deg2[2 * i] as i64 * (1 - bits[3] as i64);
            u16_result[i] += 256 * remaining_deg2[2 * i + 1] as i64 * (1 - bits[3] as i64);
            cols.byte_result[i] = F::from_canonical_u16(u16_result[i] as u16);
        }

        let shift_amount = bits[4] + 2 * bits[5] * (not_word as u8);

        let mut shift = [0u16; 4];
        for i in 0..4 {
            if i == shift_amount as usize {
                shift[i] = 1;
            }
        }

        cols.shift_u16 = shift.map(|x| F::from_canonical_u16(x));
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

        let is_real = local.is_srl
            + local.is_sra
            + local.is_srlw
            + local.is_sraw
            + local.is_srliw
            + local.is_sraiw;

        // SAFETY: All selectors `is_srl`, `is_sra` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real = is_srl + is_sra` is
        // boolean. All interactions are done with multiplicity `is_real`.
        // Therefore, the `opcode` matches the corresponding opcode.

        // Check that the operation flags are boolean.
        builder.assert_bool(local.is_srl);
        builder.assert_bool(local.is_sra);
        builder.assert_bool(local.is_srlw);
        builder.assert_bool(local.is_sraw);
        builder.assert_bool(local.is_srliw);
        builder.assert_bool(local.is_sraiw);
        builder.assert_bool(is_real.clone());

        let one = AB::Expr::one();

        let is_word = local.is_srlw + local.is_sraw + local.is_srliw + local.is_sraiw;
        let not_word = one.clone() - is_word.clone();

        let opcode = local.is_srl * AB::F::from_canonical_u32(Opcode::SRL as u32)
            + local.is_sra * AB::F::from_canonical_u32(Opcode::SRA as u32)
            + local.is_srlw * AB::F::from_canonical_u32(Opcode::SRLW as u32)
            + local.is_sraw * AB::F::from_canonical_u32(Opcode::SRAW as u32)
            + local.is_srliw * AB::F::from_canonical_u32(Opcode::SRLIW as u32)
            + local.is_sraiw * AB::F::from_canonical_u32(Opcode::SRAIW as u32);

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
        let two = AB::F::from_canonical_u32(2);
        let three = AB::F::from_canonical_u32(3);
        let fifteen = AB::F::from_canonical_u32(15);
        let two_fifty_five = AB::F::from_canonical_u32(255);
        let two_fifty_six = AB::F::from_canonical_u32(256);

        builder.assert_eq(
            local.v_01,
            (((one.clone() - local.c_bits[0]) + one.clone()) * two)
                * ((one.clone() - local.c_bits[1]) * three + one.clone()),
        );
        builder.assert_eq(
            local.v_012,
            local.v_01 * ((one.clone() - local.c_bits[2]) * fifteen + one.clone()),
        );
        builder.assert_eq(
            local.v_0123,
            local.v_012 * ((one.clone() - local.c_bits[3]) * two_fifty_five + one.clone()),
        );

        let v012b_3 = (local.v_012 * two_fifty_six - local.v_0123) * two_fifty_five.inverse();

        let b_bytes = U16toU8Operation::<AB::F>::eval_u16_to_u8_unsafe(
            builder,
            local.b.0.map(|x| x.into()),
            local.b_lower_bytes,
        );

        for i in 0..WORD_SIZE / 2 {
            builder.assert_eq(local.b.0[i], local.adapter.b()[i]);
        }
        for i in WORD_SIZE / 2..WORD_SIZE {
            builder.assert_eq(
                local.b.0[i],
                local.adapter.b()[i] * not_word.clone() + is_word.clone() * AB::F::zero(),
            );
        }

        let value = local.c_bits[0]
            + local.c_bits[1] * AB::F::from_canonical_u32(2)
            + local.c_bits[2] * AB::F::from_canonical_u32(4);

        for i in 0..WORD_BYTE_SIZE {
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::SR as u32),
                local.top_bits[i],
                b_bytes[i].clone(),
                value.clone(),
                is_real.clone(),
            );
        }

        let mut v0123_coef = std::array::from_fn::<AB::Expr, WORD_SIZE, _>(|_| AB::Expr::zero());
        let mut v012_1_3_coef = std::array::from_fn::<AB::Expr, WORD_SIZE, _>(|_| AB::Expr::zero());
        let mut remaining_deg2 =
            std::array::from_fn::<AB::Expr, WORD_BYTE_SIZE, _>(|_| AB::Expr::zero());
        for i in 0..WORD_BYTE_SIZE {
            remaining_deg2[i] = local.top_bits[i].into();
            if i + 1 < WORD_BYTE_SIZE {
                remaining_deg2[i] =
                    remaining_deg2[i].clone() - local.top_bits[i + 1] * two_fifty_six;
            }
        }
        for i in 0..WORD_SIZE {
            if i < WORD_SIZE - 1 {
                v0123_coef[i] = v0123_coef[i].clone() + b_bytes[2 * i + 2].clone();
                v0123_coef[i] = v0123_coef[i].clone() + b_bytes[2 * i + 3].clone() * two_fifty_six;
                v012_1_3_coef[i] =
                    v012_1_3_coef[i].clone() + b_bytes[2 * i + 2].clone() * two_fifty_six;
            }
            if i < WORD_SIZE {
                v012_1_3_coef[i] = v012_1_3_coef[i].clone() + b_bytes[2 * i + 1].clone();
            }
        }

        for i in 0..WORD_SIZE {
            let mut result = v0123_coef[i].clone() * v012b_3.clone();
            result = result + v012_1_3_coef[i].clone() * (local.v_012 - v012b_3.clone());
            result = result + remaining_deg2[2 * i + 1].clone() * local.c_bits[3];
            if i < WORD_SIZE - 1 {
                result =
                    result + remaining_deg2[2 * i + 2].clone() * local.c_bits[3] * two_fifty_six;
            }
            result = result + remaining_deg2[2 * i].clone() * (one.clone() - local.c_bits[3]);
            result = result
                + remaining_deg2[2 * i + 1].clone()
                    * (one.clone() - local.c_bits[3])
                    * two_fifty_six;
            builder.assert_eq(local.byte_result[i], result);
        }

        <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
            builder,
            U16MSBOperationInput::<AB>::new(local.b.0[3].into(), local.b_msb, local.is_sra.into()),
        );
        <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
            builder,
            U16MSBOperationInput::<AB>::new(
                local.b.0[1].into(),
                local.b_msb,
                local.is_sraw.into() + local.is_sraiw.into(),
            ),
        );
        builder.assert_eq(local.sra_msb_v0123, local.b_msb.msb * local.v_0123);

        <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
            builder,
            U16MSBOperationInput::<AB>::new(local.a.0[1].into(), local.srw_msb, is_word.clone()),
        );
        builder
            .when(local.is_srlw + local.is_srl + local.is_srliw)
            .assert_eq(local.b_msb.msb, AB::Expr::zero());

        for i in 0..WORD_SIZE {
            builder.when(local.shift_u16[i]).assert_eq(
                local.c_bits[4] + local.c_bits[5] * AB::F::from_canonical_u32(2) * not_word.clone(),
                AB::Expr::from_canonical_u32(i as u32),
            );
            builder.assert_bool(local.shift_u16[i]);
        }

        builder.when(is_real.clone()).assert_eq(
            local.shift_u16[0] + local.shift_u16[1] + local.shift_u16[2] + local.shift_u16[3],
            AB::Expr::from_canonical_u32(1),
        );

        let base = AB::F::from_canonical_u32(65536);
        let base_minus_one = AB::F::from_canonical_u32(65535);

        // If the opcode is SRL/SRA:
        for i in 0..WORD_SIZE {
            for j in 0..(WORD_SIZE - 1 - i) {
                builder.when(not_word.clone()).assert_eq(
                    local.shift_u16[i] * (local.a[j] - local.byte_result[i + j]),
                    AB::Expr::zero(),
                );
            }
            builder.when(not_word.clone()).assert_eq(
                local.shift_u16[i]
                    * (local.a[WORD_SIZE - 1 - i]
                        - local.byte_result[WORD_SIZE - 1]
                        - (local.b_msb.msb * base - local.sra_msb_v0123)),
                AB::Expr::zero(),
            );
            for j in (WORD_SIZE - i)..WORD_SIZE {
                builder.when(not_word.clone()).assert_eq(
                    local.shift_u16[i] * (local.a[j] - local.b_msb.msb * base_minus_one),
                    AB::Expr::zero(),
                );
            }
        }

        // If the opcode is SRLW/SRAW/SRLIW/SRAIW:
        builder
            .when(is_word.clone())
            .assert_eq(local.shift_u16[0] * (local.a[0] - local.byte_result[0]), AB::Expr::zero());
        builder.when(is_word.clone()).assert_eq(
            local.shift_u16[1]
                * (local.a[0]
                    - local.byte_result[1]
                    - (local.b_msb.msb * base - local.sra_msb_v0123)),
            AB::Expr::zero(),
        );
        builder.when(is_word.clone()).assert_eq(
            local.shift_u16[1] * (local.a[1] - local.b_msb.msb * base_minus_one),
            AB::Expr::zero(),
        );
        for i in WORD_SIZE / 2..WORD_SIZE {
            builder.when(is_word.clone()).assert_eq(local.a[i], local.srw_msb.msb * base_minus_one);
        }

        // Constrain the CPU state.
        // The program counter and timestamp increment by `4`.
        <CPUState<AB::F> as SP1Operation<AB>>::eval(
            builder,
            CPUStateInput {
                cols: local.state,
                next_pc: local.state.pc_rel + AB::F::from_canonical_u32(PC_INC),
                clk_increment: AB::Expr::from_canonical_u32(CLK_INC),
                is_real: is_real.clone(),
            },
        );

        // Constrain the program and register reads.
        let alu_reader_input = ALUTypeReaderInput::<AB, AB::Expr>::new(
            local.state.clk_high::<AB>(),
            local.state.clk_low::<AB>(),
            local.state.pc_rel,
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

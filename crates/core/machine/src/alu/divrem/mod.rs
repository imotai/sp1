//! Division and remainder verification.
//!
//! This module implements the verification logic for division and remainder operations. It ensures
//! that for any given inputs b and c and outputs quotient and remainder, the equation
//!
//! b = c * quotient + remainder
//!
//! holds true, while also ensuring that the signs of `b` and `remainder` match.
//!
//! A critical aspect of this implementation is the use of 64-bit arithmetic for result calculation.
//! This choice is driven by the need to make the solution unique: in 32-bit arithmetic,
//! `c * quotient + remainder` could overflow, leading to results that are congruent modulo 2^{32}
//! and thus not uniquely defined. The 64-bit approach avoids this overflow, ensuring that each
//! valid input combination maps to a unique result.
//!
//! Implementation:
//!
//! # Use the multiplication ALU table. result is 64 bits.
//! result = quotient * c.
//!
//! # Add sign-extended remainder to result. Propagate carry to handle overflow within bytes.
//! base = pow(2, 8)
//! carry = 0
//! for i in range(8):
//!     x = result\[i\] + remainder\[i\] + carry
//!     result\[i\] = x % base
//!     carry = x // base
//!
//! # The number represented by c * quotient + remainder in 64 bits must equal b in 32 bits.
//!
//! # Assert the lower 32 bits of result match b.
//! assert result[0..4] == b[0..4]
//!
//! # Assert the upper 32 bits of result match the sign of b.
//! if (b == -2^{31}) and (c == -1):
//!     # This is the only exception as this is the only case where it overflows.
//!     assert result[4..8] == [0, 0, 0, 0]
//! elif b < 0:
//!     assert result[4..8] == [0xff, 0xff, 0xff, 0xff]
//! else:
//!     assert result[4..8] == [0, 0, 0, 0]
//!
//! # Check a = quotient or remainder.
//! assert a == (quotient if opcode == division else remainder)
//!
//! # remainder and b must have the same sign.
//! if remainder < 0:
//!     assert b <= 0
//! if remainder > 0:
//!     assert b >= 0
//!
//! # abs(remainder) < abs(c)
//! if c < 0:
//!    assert c < remainder <= 0
//! elif c > 0:
//!    assert 0 <= remainder < c
//!
//! if is_c_0:
//!    # if division by 0, then quotient = 0xffffffff per RISC-V spec. This needs special care since
//!    # b = 0 * quotient + b is satisfied by any quotient.
//!    assert quotient = 0xffffffff

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use slop_air::{Air, AirBuilder, BaseAir};
use slop_algebra::{AbstractField, PrimeField32};
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    get_msb, get_quotient_and_remainder, is_signed_operation, is_signed_word_operation,
    is_unsigned_operation, is_unsigned_word_operation, is_word_operation, ExecutionRecord, Opcode,
    Program, CLK_INC, PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{air::MachineAir, Word};

use crate::{
    adapter::{
        register::alu_type::{ALUTypeReader, ALUTypeReaderInput},
        state::{CPUState, CPUStateInput},
    },
    air::{SP1CoreAirBuilder, SP1Operation, WordAirBuilder},
    operations::{
        AddOperation, IsEqualWordOperation, IsEqualWordOperationInput, IsZeroWordOperation,
        IsZeroWordOperationInput, LtOperationUnsigned, LtOperationUnsignedInput, MulOperation,
        U16MSBOperation, U16MSBOperationInput,
    },
    utils::{next_multiple_of_32, pad_rows_fixed},
};

/// The number of main trace columns for `DivRemChip`.
pub const NUM_DIVREM_COLS: usize = size_of::<DivRemCols<u8>>();

/// The size of a 128-bit in limbs.
const LONG_WORD_SIZE: usize = 2 * WORD_SIZE;

/// A chip that implements division for the opcodes DIV/REM.
#[derive(Default)]
pub struct DivRemChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct DivRemCols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ALUTypeReader<T>,

    /// The output operand.
    pub a: Word<T>,

    /// The input operand (b sign extended if word operation).
    pub b: Word<T>,

    /// The input operand (c sign extended if word operation).
    pub c: Word<T>,

    /// Results of dividing `b` by `c`.
    pub quotient: Word<T>,

    /// The quotient used in the computation of `c * quotient + remainder`
    /// (truncated in the case of unsigned word operation).
    pub quotient_comp: Word<T>,

    /// The remainder used in the computation of `c * quotient + remainder`
    /// (truncated in the case of unsigned word operation).
    pub remainder_comp: Word<T>,

    /// Remainder when dividing `b` by `c`.
    pub remainder: Word<T>,

    /// `abs(remainder)`, used to check `abs(remainder) < abs(c)`.
    pub abs_remainder: Word<T>,

    /// `abs(c)`, used to check `abs(remainder) < abs(c)`.
    pub abs_c: Word<T>,

    /// `max(abs(c), 1)`, used to check `abs(remainder) < abs(c)`.
    pub max_abs_c_or_1: Word<T>,

    /// The result of `c * quotient`.
    pub c_times_quotient: [T; LONG_WORD_SIZE],

    /// Instance of `MulOperation` for the lower half of `c * quotient`.
    pub c_times_quotient_lower: MulOperation<T>,

    /// Instance of `MulOperation` for the upper half of `c * quotient`.
    pub c_times_quotient_upper: MulOperation<T>,

    /// Instance of `AddOperation` to get the negative of `c`
    pub c_neg_operation: AddOperation<T>,

    /// Instance of `AddOperation` to get the negative of `remainder`.
    pub rem_neg_operation: AddOperation<T>,

    /// Instance of `LtOperation` to check if abs(remainder) < abs(c).
    pub remainder_lt_operation: LtOperationUnsigned<T>,

    /// Carry propagated when adding `remainder` by `c * quotient`.
    pub carry: [T; LONG_WORD_SIZE],

    /// Flag to indicate division by 0.
    pub is_c_0: IsZeroWordOperation<T>,

    /// Flag to indicate whether the opcode is DIV.
    pub is_div: T,

    /// Flag to indicate whether the opcode is DIVU.
    pub is_divu: T,

    /// Flag to indicate whether the opcode is REM.
    pub is_rem: T,

    /// Flag to indicate whether the opcode is REMU.
    pub is_remu: T,

    /// Flag to indicate whether the opcode is DIVW.
    pub is_divw: T,

    /// Flag to indicate whether the opcode is REMW.
    pub is_remw: T,

    /// Flag to indicate whether the opcode is DIVUW.
    pub is_divuw: T,

    /// Flag to indicate whether the opcode is REMUW.
    pub is_remuw: T,

    /// The base opcode for the divrem instruction.
    pub base_op_code: T,

    /// Flag to indicate whether the division operation overflows.
    ///
    /// Overflow occurs in a specific case of signed 32-bit integer division: when `b` is the
    /// minimum representable value (`-2^31`, the smallest negative number) and `c` is `-1`. In
    /// this case, the division result exceeds the maximum positive value representable by a
    /// 32-bit signed integer.
    pub is_overflow: T,

    /// Flag for whether the value of `b` matches the unique overflow case `b = -2^31` and `c =
    /// -1`.
    pub is_overflow_b: IsEqualWordOperation<T>,

    /// Flag for whether the value of `c` matches the unique overflow case `b = -2^31` and `c =
    /// -1`.
    pub is_overflow_c: IsEqualWordOperation<T>,

    /// The most significant bit of `b`.
    pub b_msb: U16MSBOperation<T>,

    /// The most significant bit of remainder.
    pub rem_msb: U16MSBOperation<T>,

    /// The most significant bit of `c`.
    pub c_msb: U16MSBOperation<T>,

    /// The most significant bit of `quotient`.
    pub quot_msb: U16MSBOperation<T>,

    /// Flag to indicate whether `b` is negative.
    pub b_neg: T,

    /// Flag to indicate whether `b` is negative and not is_overflow.
    pub b_neg_not_overflow: T,

    /// Flag to indicate whether `b` is not negative and not is_overflow.
    pub b_not_neg_not_overflow: T,

    /// Flag to indicate whether is_real and not word operation.
    pub is_real_not_word: T,

    /// Flag to indicate whether `rem_neg` is negative.
    pub rem_neg: T,

    /// Flag to indicate whether `c` is negative.
    pub c_neg: T,

    /// Selector to determine whether an ALU Event is sent for absolute value computation of `c`.
    pub abs_c_alu_event: T,

    /// Selector to determine whether an ALU Event is sent for absolute value computation of `rem`.
    pub abs_rem_alu_event: T,

    /// Selector to know whether this row is enabled.
    pub is_real: T,

    /// Column to modify multiplicity for remainder range check event.
    pub remainder_check_multiplicity: T,
}

impl<F: PrimeField32> MachineAir<F> for DivRemChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "DivRem".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.divrem_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the trace rows for each event.
        let mut rows: Vec<[F; NUM_DIVREM_COLS]> = vec![];
        let divrem_events = input.divrem_events.clone();
        for event_record in divrem_events.iter() {
            let event = event_record.0;
            let alu_record = event_record.1;

            assert!(
                event.opcode == Opcode::DIVU
                    || event.opcode == Opcode::REMU
                    || event.opcode == Opcode::REM
                    || event.opcode == Opcode::DIV
                    || event.opcode == Opcode::DIVW
                    || event.opcode == Opcode::REMW
                    || event.opcode == Opcode::DIVUW
                    || event.opcode == Opcode::REMUW
            );

            let mut row = [F::zero(); NUM_DIVREM_COLS];
            let cols: &mut DivRemCols<F> = row.as_mut_slice().borrow_mut();

            {
                let mut blu = vec![];
                cols.state.populate(&mut blu, event.clk, event.pc);
                cols.adapter.populate(&mut blu, alu_record);
                output.add_byte_lookup_events(blu);
            }

            // Initialize cols with basic operands and flags derived from the current event.
            {
                cols.a = Word::from(event.a);
                if is_signed_word_operation(event.opcode) {
                    cols.b = Word::from(event.b as i32 as i64 as u64);
                    cols.c = Word::from(event.c as i32 as i64 as u64);
                } else if is_unsigned_word_operation(event.opcode) {
                    cols.b = Word::from(event.b as u32 as u64);
                    cols.c = Word::from(event.c as u32 as u64);
                } else {
                    cols.b = Word::from(event.b);
                    cols.c = Word::from(event.c);
                }

                cols.is_real = F::one();

                cols.is_divu = F::from_bool(event.opcode == Opcode::DIVU);
                cols.is_remu = F::from_bool(event.opcode == Opcode::REMU);
                cols.is_div = F::from_bool(event.opcode == Opcode::DIV);
                cols.is_rem = F::from_bool(event.opcode == Opcode::REM);
                cols.is_divw = F::from_bool(event.opcode == Opcode::DIVW);
                cols.is_divuw = F::from_bool(event.opcode == Opcode::DIVUW);
                cols.is_remw = F::from_bool(event.opcode == Opcode::REMW);
                cols.is_remuw = F::from_bool(event.opcode == Opcode::REMUW);

                let (divw_base, divw_imm) = Opcode::DIVW.base_opcode();
                let divw_imm = divw_imm.expect("DIVW immediate opcode not found");
                let (remw_base, remw_imm) = Opcode::REMW.base_opcode();
                let remw_imm = remw_imm.expect("REMW immediate opcode not found");
                let (divuw_base, divuw_imm) = Opcode::DIVUW.base_opcode();
                let divuw_imm = divuw_imm.expect("DIVUW immediate opcode not found");
                let (remuw_base, remuw_imm) = Opcode::REMUW.base_opcode();
                let remuw_imm = remuw_imm.expect("REMUW immediate opcode not found");

                let is_imm_c = cols.adapter.imm_c.is_one();

                let divw_base_opcode =
                    F::from_canonical_u32(if is_imm_c { divw_imm } else { divw_base });
                let remw_base_opcode =
                    F::from_canonical_u32(if is_imm_c { remw_imm } else { remw_base });
                let divuw_base_opcode =
                    F::from_canonical_u32(if is_imm_c { divuw_imm } else { divuw_base });
                let remuw_base_opcode =
                    F::from_canonical_u32(if is_imm_c { remuw_imm } else { remuw_base });

                cols.base_op_code = match event.opcode {
                    Opcode::DIVU => F::from_canonical_u32(Opcode::DIVU.base_opcode().0),
                    Opcode::REMU => F::from_canonical_u32(Opcode::REMU.base_opcode().0),
                    Opcode::DIV => F::from_canonical_u32(Opcode::DIV.base_opcode().0),
                    Opcode::REM => F::from_canonical_u32(Opcode::REM.base_opcode().0),
                    Opcode::DIVW => divw_base_opcode,
                    Opcode::REMW => remw_base_opcode,
                    Opcode::DIVUW => divuw_base_opcode,
                    Opcode::REMUW => remuw_base_opcode,
                    _ => unreachable!(),
                };

                let not_word_operation =
                    F::one() - cols.is_divw - cols.is_remw - cols.is_divuw - cols.is_remuw;
                cols.is_real_not_word = cols.is_real * not_word_operation;
                cols.is_c_0.populate(event.c);
            }

            let (quotient, remainder) = get_quotient_and_remainder(event.b, event.c, event.opcode);

            cols.quotient = Word::from(quotient);
            cols.remainder = Word::from(remainder);
            let b = if is_signed_word_operation(event.opcode) {
                event.b as i32 as i64 as u64
            } else if is_unsigned_word_operation(event.opcode) {
                event.b as u32 as u64
            } else {
                event.b
            };
            let c = if is_signed_word_operation(event.opcode) {
                event.c as i32 as i64 as u64
            } else if is_unsigned_word_operation(event.opcode) {
                event.c as u32 as u64
            } else {
                event.c
            };

            // Calculate flags for sign detection.
            {
                if is_signed_operation(event.opcode) {
                    cols.rem_neg = F::from_canonical_u8(get_msb(remainder));
                    cols.b_neg = F::from_canonical_u8(get_msb(event.b));
                    cols.c_neg = F::from_canonical_u8(get_msb(event.c));
                    cols.is_overflow =
                        F::from_bool(event.b as i64 == i64::MIN && event.c as i64 == -1);
                    cols.abs_remainder = Word::from((remainder as i64).abs() as u64);
                    cols.abs_c = Word::from((event.c as i64).abs() as u64);
                    cols.max_abs_c_or_1 = Word::from(u64::max(1, (event.c as i64).abs() as u64));
                } else if is_signed_word_operation(event.opcode) {
                    cols.rem_neg = F::from_canonical_u8(get_msb((remainder as i32) as i64 as u64));
                    cols.b_neg = F::from_canonical_u8(get_msb((event.b as i32) as i64 as u64));
                    cols.c_neg = F::from_canonical_u8(get_msb((event.c as i32) as i64 as u64));
                    cols.is_overflow =
                        F::from_bool(event.b as i32 == i32::MIN && event.c as i32 == -1);
                    cols.abs_remainder = Word::from((remainder as i64).abs() as u64);
                    cols.abs_c = Word::from((c as i64).abs() as u64);
                    cols.max_abs_c_or_1 = Word::from(u64::max(1, (c as i64).abs() as u64));
                } else if is_unsigned_word_operation(event.opcode) {
                    cols.abs_remainder = Word::from(remainder as u32);
                    cols.abs_c = Word::from(event.c as u32);
                    cols.max_abs_c_or_1 = Word::from(u32::max(1, event.c as u32));
                } else {
                    cols.abs_remainder = cols.remainder;
                    cols.abs_c = Word::from(event.c);
                    cols.max_abs_c_or_1 = Word::from(u64::max(1, event.c));
                }

                if is_word_operation(event.opcode) {
                    cols.is_overflow_b.populate((event.b as u32) as u64, i32::MIN as u32 as u64);
                    cols.is_overflow_c.populate((event.c as u32) as u64, -1i32 as u32 as u64);
                } else {
                    cols.is_overflow_b.populate(event.b, i64::MIN as u64);
                    cols.is_overflow_c.populate(event.c, -1i64 as u64);
                }

                cols.b_neg_not_overflow = cols.b_neg * (F::one() - cols.is_overflow);
                cols.b_not_neg_not_overflow =
                    (F::one() - cols.b_neg) * (F::one() - cols.is_overflow);

                // Set the `alu_event` flags.
                cols.abs_c_alu_event = cols.c_neg * cols.is_real;
                cols.abs_rem_alu_event = cols.rem_neg * cols.is_real;

                // Populate the c_neg_operation and rem_neg_operation.
                {
                    let mut blu_events = vec![];
                    if cols.abs_c_alu_event.is_one() {
                        cols.c_neg_operation.populate(
                            &mut blu_events,
                            cols.c.to_u64(),
                            cols.abs_c.to_u64(),
                        );
                    }
                    if cols.abs_rem_alu_event.is_one() {
                        cols.rem_neg_operation.populate(
                            &mut blu_events,
                            cols.remainder.to_u64(),
                            cols.abs_remainder.to_u64(),
                        );
                    }
                    output.add_byte_lookup_events(blu_events);
                }

                // Insert the MSB lookup events.
                {
                    let mut blu_events: Vec<ByteLookupEvent> = vec![];

                    if is_word_operation(event.opcode) {
                        cols.b_msb.populate_msb(&mut blu_events, (event.b >> 16) as u16);
                        cols.c_msb.populate_msb(&mut blu_events, (event.c >> 16) as u16);
                        cols.rem_msb.populate_msb(&mut blu_events, (remainder >> 16) as u16);
                        cols.quot_msb.populate_msb(&mut blu_events, (quotient >> 16) as u16);
                    } else {
                        cols.b_msb.populate_msb(&mut blu_events, (b >> 48) as u16);
                        cols.c_msb.populate_msb(&mut blu_events, (c >> 48) as u16);
                        cols.rem_msb.populate_msb(&mut blu_events, (remainder >> 48) as u16);
                    }

                    output.add_byte_lookup_events(blu_events);
                }
            }

            // Calculate the modified multiplicity
            {
                let mut blu_events = vec![];
                cols.remainder_check_multiplicity = cols.is_real * (F::one() - cols.is_c_0.result);
                if cols.remainder_check_multiplicity.is_one() {
                    cols.remainder_lt_operation.populate_unsigned(
                        &mut blu_events,
                        1u64,
                        cols.abs_remainder.to_u64(),
                        cols.max_abs_c_or_1.to_u64(),
                    );
                }

                output.add_byte_lookup_events(blu_events);
            }

            // Calculate c * quotient + remainder.
            {
                let mut blu_events = vec![];
                let c_times_quotient_byte = {
                    if is_signed_operation(event.opcode) {
                        (((quotient as i64) as i128) * ((event.c as i64) as i128)).to_le_bytes()
                    } else if is_signed_word_operation(event.opcode) {
                        (((quotient as i32) * (event.c as i32)) as i128).to_le_bytes()
                    } else if is_unsigned_word_operation(event.opcode) {
                        (((quotient as u32) * (event.c as u32)) as u128).to_le_bytes()
                    } else {
                        ((quotient as u128) * (event.c as u128)).to_le_bytes()
                    }
                };
                let c_times_quotient_u16: [u16; LONG_WORD_SIZE] = core::array::from_fn(|i| {
                    u16::from_le_bytes([
                        c_times_quotient_byte[2 * i],
                        c_times_quotient_byte[2 * i + 1],
                    ])
                });

                cols.c_times_quotient = c_times_quotient_u16.map(F::from_canonical_u16);

                // Quotient needs to be truncated in the case of unsigned word operation for the
                // following computation because unsigned word operations still sign extend the u32
                // result.
                let quotient_u32 = if is_unsigned_word_operation(event.opcode) {
                    quotient as u32 as u64
                } else {
                    quotient
                };
                cols.quotient_comp = Word::from(quotient_u32);

                if is_signed_word_operation(event.opcode) {
                    cols.c_times_quotient_lower.populate(
                        &mut blu_events,
                        quotient_u32,
                        c,
                        false,
                        false,
                        true,
                    );
                } else {
                    cols.c_times_quotient_lower.populate(
                        &mut blu_events,
                        quotient_u32,
                        c,
                        false,
                        false,
                        false,
                    );
                }
                if is_signed_operation(event.opcode) {
                    cols.c_times_quotient_upper.populate(
                        &mut blu_events,
                        quotient,
                        c,
                        true,
                        false,
                        false,
                    );
                } else if is_unsigned_operation(event.opcode) {
                    cols.c_times_quotient_upper.populate(
                        &mut blu_events,
                        quotient,
                        c,
                        false,
                        false,
                        false,
                    );
                }

                output.add_byte_lookup_events(blu_events);

                let remainder_bytes = {
                    if is_signed_operation(event.opcode) {
                        ((remainder as i64) as i128).to_le_bytes()
                    } else if is_signed_word_operation(event.opcode) {
                        ((remainder as i32) as i128).to_le_bytes()
                    } else if is_unsigned_word_operation(event.opcode) {
                        ((remainder as u32) as u128).to_le_bytes()
                    } else {
                        (remainder as u128).to_le_bytes()
                    }
                };
                let remainder_u16: [u16; LONG_WORD_SIZE] = core::array::from_fn(|i| {
                    u16::from_le_bytes([remainder_bytes[2 * i], remainder_bytes[2 * i + 1]])
                });

                // Remainder needs to be truncated/sign extended for c * quotient + remainder
                // computation.
                if is_word_operation(event.opcode) {
                    cols.remainder_comp = Word([
                        F::from_canonical_u16(remainder_u16[0]),
                        F::from_canonical_u16(remainder_u16[1]),
                        F::from_canonical_u16(remainder_u16[2]),
                        F::from_canonical_u16(remainder_u16[3]),
                    ]);
                } else {
                    cols.remainder_comp = cols.remainder;
                }

                // Add remainder to product.
                let mut carry = [0u32; 8];
                let base = 1 << 16;
                for i in 0..LONG_WORD_SIZE {
                    let mut x = c_times_quotient_u16[i] as u32 + remainder_u16[i] as u32;
                    if i > 0 {
                        x += carry[i - 1];
                    }
                    carry[i] = x / base;
                    cols.carry[i] = F::from_canonical_u32(carry[i]);
                }
                // Range check.
                {
                    output.add_u16_range_checks(&[
                        (quotient & 0xFFFF) as u16,
                        (quotient >> 16) as u16,
                        (quotient >> 32) as u16,
                        (quotient >> 48) as u16,
                    ]);
                    output.add_u16_range_checks(&[
                        (remainder & 0xFFFF) as u16,
                        (remainder >> 16) as u16,
                        (remainder >> 32) as u16,
                        (remainder >> 48) as u16,
                    ]);
                    output.add_u16_range_checks(&c_times_quotient_u16);
                }
            }

            rows.push(row);
        }

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_DIVREM_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        assert_eq!(rows.len(), <DivRemChip as MachineAir<F>>::num_rows(self, input).unwrap());

        // Convert the trace to a row major matrix.
        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_DIVREM_COLS);

        // Create the template for the padded rows. These are fake rows that don't fail on some
        // sanity checks.
        let padded_row_template = {
            let mut row = [F::zero(); NUM_DIVREM_COLS];
            let cols: &mut DivRemCols<F> = row.as_mut_slice().borrow_mut();
            // 0 divided by 1. quotient = remainder = 0.
            cols.is_divu = F::one();
            cols.adapter.op_c_memory.prev_value = Word::from(1u64);
            cols.abs_c[0] = F::one();
            cols.c[0] = F::one();
            cols.max_abs_c_or_1[0] = F::one();

            cols.is_c_0.populate(1);

            row
        };
        debug_assert!(padded_row_template.len() == NUM_DIVREM_COLS);
        for i in input.divrem_events.len() * NUM_DIVREM_COLS..trace.values.len() {
            trace.values[i] = padded_row_template[i % NUM_DIVREM_COLS];
        }

        trace
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.divrem_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<F> BaseAir<F> for DivRemChip {
    fn width(&self) -> usize {
        NUM_DIVREM_COLS
    }
}

impl<AB> Air<AB> for DivRemChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &DivRemCols<AB::Var> = (*local).borrow();
        let base = AB::F::from_canonical_u32(1 << 16);
        let one: AB::Expr = AB::F::one().into();
        let zero: AB::Expr = AB::F::zero().into();
        let is_word_operation = local.is_divw + local.is_remw + local.is_divuw + local.is_remuw;
        let is_signed_word_operation = local.is_divw + local.is_remw;
        let u16_max = AB::F::from_canonical_u16(u16::MAX);
        builder.assert_eq(
            local.is_real_not_word,
            local.is_real * (one.clone() - is_word_operation.clone()),
        );

        // Calculate whether b, remainder, and c are negative.
        {
            // Negative if and only if opcode is signed & MSB = 1.
            let is_signed_type = local.is_div + local.is_rem + local.is_divw + local.is_remw;
            let msb_sign_pairs = [
                (local.b_msb.msb, local.b_neg),
                (local.rem_msb.msb, local.rem_neg),
                (local.c_msb.msb, local.c_neg),
            ];

            for msb_sign_pair in msb_sign_pairs.iter() {
                let msb = msb_sign_pair.0;
                let is_negative = msb_sign_pair.1;
                builder.assert_eq(msb * is_signed_type.clone(), is_negative);
            }
        }

        // Assert that the truncated/sign extended b and c align with the original b and c.
        {
            for i in 0..WORD_SIZE / 2 {
                builder.assert_eq(local.adapter.b()[i], local.b[i]);
                builder.assert_eq(local.adapter.c()[i], local.c[i]);
            }
            for i in WORD_SIZE / 2..WORD_SIZE {
                builder.assert_eq(
                    local.b[i],
                    local.adapter.b()[i] * (one.clone() - is_word_operation.clone())
                        + local.b_neg * is_word_operation.clone() * u16_max,
                );
                builder.assert_eq(
                    local.c[i],
                    local.adapter.c()[i] * (one.clone() - is_word_operation.clone())
                        + local.c_neg * is_word_operation.clone() * u16_max,
                );
            }
        }

        // Use the mul operation to compute c * quotient and compare it to local.c_times_quotient.
        {
            let lower_half: [AB::Expr; 4] = [
                local.c_times_quotient[0].into(),
                local.c_times_quotient[1].into(),
                local.c_times_quotient[2].into(),
                local.c_times_quotient[3].into(),
            ];

            // The lower 8 bytes of c_times_quotient must match the lower 8 bytes of (c * quotient).
            MulOperation::<AB::F>::eval(
                builder,
                Word(lower_half),
                local.quotient_comp.map(|x| x.into()),
                local.c.map(|x| x.into()),
                local.c_times_quotient_lower,
                local.is_real.into(),
                one.clone() - is_signed_word_operation.clone(),
                AB::Expr::zero(),
                is_signed_word_operation.clone(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            );

            for i in 0..WORD_SIZE / 2 {
                builder.assert_eq(local.quotient_comp[i], local.quotient[i]);
            }

            for i in WORD_SIZE / 2..WORD_SIZE {
                builder
                    .when(local.is_divuw + local.is_remuw)
                    .assert_eq(local.quotient_comp[i], AB::Expr::zero());
                builder.when(local.is_divw + local.is_remw).assert_eq(
                    local.quotient_comp[i],
                    local.quot_msb.msb * AB::F::from_canonical_u16(u16::MAX),
                );
                builder
                    .when(one.clone() - is_word_operation.clone())
                    .assert_eq(local.quotient_comp[i], local.quotient[i]);
            }

            // SAFETY: Since exactly one flag is turned on, `is_mulh` and `is_mulhu` are correct.
            let is_mulh = local.is_div + local.is_rem;
            let is_mulhu = local.is_divu + local.is_remu;

            let upper_half: [AB::Expr; 4] = [
                local.c_times_quotient[4].into(),
                local.c_times_quotient[5].into(),
                local.c_times_quotient[6].into(),
                local.c_times_quotient[7].into(),
            ];

            // The upper 8 bytes of c_times_quotient must match the upper 8 bytes of (c * quotient).
            // Only required for non-word operations.
            MulOperation::<AB::F>::eval(
                builder,
                Word(upper_half),
                local.quotient.map(|x| x.into()),
                local.c.map(|x| x.into()),
                local.c_times_quotient_upper,
                local.is_real_not_word.into(),
                AB::Expr::zero(),
                is_mulh,
                AB::Expr::zero(),
                is_mulhu,
                AB::Expr::zero(),
            );
        }

        // Calculate is_overflow. is_overflow_word = is_equal(b as u32, -2^{31}) * is_equal(c as
        // u32, -1i32 as u32) * is_signed is_overflow_not_word = is_equal(b, -2^{63}) *
        // is_equal(c, -1i64 as u64) * is_signed
        {
            <IsEqualWordOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                IsEqualWordOperationInput::new(
                    local.adapter.b().map(|x| x.into()),
                    Word::from(i64::MIN as u64).map(|x: AB::F| x.into()),
                    local.is_overflow_b,
                    local.is_real_not_word.into(),
                ),
            );

            <IsEqualWordOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                IsEqualWordOperationInput::new(
                    local.adapter.c().map(|x| x.into()),
                    Word::from(-1i64 as u64).map(|x: AB::F| x.into()),
                    local.is_overflow_c,
                    local.is_real_not_word.into(),
                ),
            );

            let mut truncated_b = local.adapter.b().map(|x| x.into());
            let mut truncated_c = local.adapter.c().map(|x| x.into());
            truncated_b[2] = AB::Expr::zero();
            truncated_c[2] = AB::Expr::zero();
            truncated_b[3] = AB::Expr::zero();
            truncated_c[3] = AB::Expr::zero();

            IsEqualWordOperation::<AB::F>::eval(
                builder,
                IsEqualWordOperationInput::new(
                    truncated_b,
                    Word::from(i32::MIN as u32 as u64).map(|x: AB::F| x.into()),
                    local.is_overflow_b,
                    is_word_operation.clone(),
                ),
            );

            IsEqualWordOperation::<AB::F>::eval(
                builder,
                IsEqualWordOperationInput::new(
                    truncated_c,
                    Word::from(-1i32 as u32 as u64).map(|x: AB::F| x.into()),
                    local.is_overflow_c,
                    is_word_operation.clone(),
                ),
            );

            let is_signed = local.is_div + local.is_rem + local.is_divw + local.is_remw;

            builder.assert_eq(
                local.is_overflow,
                local.is_overflow_b.is_diff_zero.result
                    * local.is_overflow_c.is_diff_zero.result
                    * is_signed,
            );
        }

        // Add remainder to product c * quotient, and compare it to b.
        {
            let sign_extension = local.rem_neg * AB::F::from_canonical_u16(u16::MAX);
            let mut c_times_quotient_plus_remainder: Vec<AB::Expr> =
                vec![AB::Expr::zero(); LONG_WORD_SIZE];

            // Add remainder to c_times_quotient and propagate carry.
            for i in 0..LONG_WORD_SIZE {
                c_times_quotient_plus_remainder[i] = local.c_times_quotient[i].into();

                // Add remainder.
                if i < WORD_SIZE {
                    c_times_quotient_plus_remainder[i] =
                        c_times_quotient_plus_remainder[i].clone() + local.remainder_comp[i].into();
                } else {
                    // If rem is negative, add 0xff to the upper 4 bytes.
                    c_times_quotient_plus_remainder[i] =
                        c_times_quotient_plus_remainder[i].clone() + sign_extension.clone();
                }

                // Propagate carry.
                // SAFETY: Since carry is a boolean and `c_times_quotient_plus_remainder` are u16s,
                // the results are guaranteed to be correct by the constraints.
                c_times_quotient_plus_remainder[i] =
                    c_times_quotient_plus_remainder[i].clone() - local.carry[i] * base;
                if i > 0 {
                    c_times_quotient_plus_remainder[i] =
                        c_times_quotient_plus_remainder[i].clone() + local.carry[i - 1].into();
                }
            }

            // Compare c_times_quotient_plus_remainder to b by checking each limb.
            for i in 0..LONG_WORD_SIZE {
                if i < WORD_SIZE / 2 {
                    // The lower 8 bytes of the result must match the corresponding bytes in b.
                    builder.assert_eq(
                        local.adapter.b()[i],
                        c_times_quotient_plus_remainder[i].clone(),
                    );
                } else if i < WORD_SIZE {
                    // The upper 8 bytes of the result must match the corresponding bytes in b.
                    builder.when(one.clone() - is_word_operation.clone()).assert_eq(
                        local.adapter.b()[i],
                        c_times_quotient_plus_remainder[i].clone(),
                    );

                    builder
                        .when(is_word_operation.clone())
                        .when(local.b_neg_not_overflow)
                        .assert_eq(
                            c_times_quotient_plus_remainder[i].clone(),
                            AB::F::from_canonical_u16(u16::MAX),
                        );

                    builder
                        .when(is_word_operation.clone())
                        .when(local.b_not_neg_not_overflow)
                        .assert_eq(c_times_quotient_plus_remainder[i].clone(), AB::F::zero());

                    // Since c * quotient is calculated using MULW, the result is sign extended up
                    // to WORD_SIZE in the overflow case.
                    builder.when(is_word_operation.clone()).when(local.is_overflow).assert_eq(
                        c_times_quotient_plus_remainder[i].clone(),
                        AB::F::from_canonical_u16(u16::MAX),
                    );
                } else {
                    // The upper 8 bytes must reflect the sign of b in two's complement:
                    // - All 1s (0xff) for negative b.
                    // - All 0s for non-negative b.
                    // let not_overflow = one.clone() - local.is_overflow;
                    builder.when(local.b_neg_not_overflow).assert_eq(
                        c_times_quotient_plus_remainder[i].clone(),
                        AB::F::from_canonical_u16(u16::MAX),
                    );
                    builder
                        .when(local.b_not_neg_not_overflow)
                        .assert_zero(c_times_quotient_plus_remainder[i].clone());

                    // The only exception to the upper-8-byte check is the overflow case.
                    builder
                        .when(local.is_overflow * (one.clone() - is_word_operation.clone()))
                        .assert_zero(c_times_quotient_plus_remainder[i].clone());
                }
            }

            // Constrain that the remainder used for the calculation is sign-extended/truncated
            // correctly (in case of word operation).
            for i in 0..WORD_SIZE {
                if i < WORD_SIZE / 2 {
                    builder
                        .when(is_word_operation.clone())
                        .assert_eq(local.remainder_comp[i], local.remainder[i]);
                } else {
                    builder.when(is_word_operation.clone()).assert_eq(
                        local.remainder_comp[i],
                        local.rem_neg * AB::F::from_canonical_u16(u16::MAX),
                    );
                }
                builder
                    .when(one.clone() - is_word_operation.clone())
                    .assert_eq(local.remainder_comp[i], local.remainder[i]);
            }
        }

        // `a` must equal remainder or quotient depending on the opcode.
        for i in 0..WORD_SIZE {
            builder
                .when(local.is_divu + local.is_div + local.is_divw + local.is_divuw)
                .assert_eq(local.quotient[i], local.a[i]);
            builder
                .when(local.is_remu + local.is_rem + local.is_remw + local.is_remuw)
                .assert_eq(local.remainder[i], local.a[i]);
        }

        for i in WORD_SIZE / 2..WORD_SIZE {
            builder.when(is_word_operation.clone()).assert_eq(
                local.quot_msb.msb * AB::F::from_canonical_u16(u16::MAX),
                local.quotient[i],
            );
            builder.when(is_word_operation.clone()).assert_eq(
                local.rem_msb.msb * AB::F::from_canonical_u16(u16::MAX),
                local.remainder[i],
            );
        }

        // remainder and b must have the same sign. Due to the intricate nature of sign logic in ZK,
        // we will check a slightly stronger condition:
        //
        // 1. If remainder < 0, then b < 0.
        // 2. If remainder > 0, then b >= 0.
        {
            // A number is 0 if and only if the sum of the two limbs equals to 0.
            let mut rem_limb_sum = zero.clone();
            for i in 0..WORD_SIZE {
                rem_limb_sum = rem_limb_sum.clone() + local.remainder[i].into();
            }

            // 1. If remainder < 0, then b < 0.
            builder
                .when(local.rem_neg) // rem is negative.
                .assert_one(local.b_neg); // b is negative.

            // 2. If remainder > 0, then b >= 0.
            builder
                .when(rem_limb_sum.clone()) // remainder is nonzero.
                .when(one.clone() - local.rem_neg) // rem is not negative.
                .assert_zero(local.b_neg); // b is not negative.
        }

        // When division by 0, quotient must be u64::MAX per RISC-V spec.
        {
            // Calculate whether c is 0.
            <IsZeroWordOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                IsZeroWordOperationInput::new(
                    local.adapter.c().map(|x| x.into()),
                    local.is_c_0,
                    local.is_real.into(),
                ),
            );

            // If is_c_0 is true, then quotient must be 0xffffffff_ffffffff = u64::MAX.
            for i in 0..WORD_SIZE {
                builder
                    .when(local.is_c_0.result)
                    .assert_eq(local.quotient[i], AB::F::from_canonical_u16(u16::MAX));
            }
        }

        // Range check remainder. (i.e., |remainder| < |c| when not is_c_0)
        {
            // For each of `c` and `rem`, assert that the absolute value is equal to the original
            // value, if the original value is non-negative or the minimum i64.
            for i in 0..WORD_SIZE {
                // For c, simply check that abs_c equals c when c is not negative
                builder.when_not(local.c_neg).assert_eq(local.c[i], local.abs_c[i]);

                // For remainder, handle both cases with a single condition
                builder
                    .when_not(local.rem_neg)
                    .assert_eq(local.remainder_comp[i], local.abs_remainder[i]);
            }
            // In the case that `c` or `rem` is negative, instead check that their sum is zero.
            AddOperation::<AB::F>::eval(
                builder,
                local.c.map(|x| x.into()),
                local.abs_c.map(|x| x.into()),
                local.c_neg_operation,
                local.abs_c_alu_event.into(),
            );
            builder.when(local.abs_c_alu_event).assert_word_eq(
                Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]),
                local.c_neg_operation.value,
            );

            AddOperation::<AB::F>::eval(
                builder,
                local.remainder.map(|x| x.into()),
                local.abs_remainder.map(|x| x.into()),
                local.rem_neg_operation,
                local.abs_rem_alu_event.into(),
            );
            builder.when(local.abs_rem_alu_event).assert_word_eq(
                Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]),
                local.rem_neg_operation.value,
            );

            // Check that the absolute value selector columns are computed correctly.
            // This enforces the send multiplicities are zero when `is_real == 0`.
            builder.assert_eq(local.abs_c_alu_event, local.c_neg * local.is_real);
            builder.assert_eq(local.abs_rem_alu_event, local.rem_neg * local.is_real);

            // max(abs(c), 1) = abs(c) * (1 - is_c_0) + 1 * is_c_0
            let max_abs_c_or_1: Word<AB::Expr> = {
                let mut v = vec![zero.clone(); WORD_SIZE];

                // Set the least significant byte to 1 if is_c_0 is true.
                v[0] = local.is_c_0.result * one.clone()
                    + (one.clone() - local.is_c_0.result) * local.abs_c[0];

                // Set the remaining bytes to 0 if is_c_0 is true.
                for i in 1..WORD_SIZE {
                    v[i] = (one.clone() - local.is_c_0.result) * local.abs_c[i];
                }
                Word(v.try_into().unwrap_or_else(|_| panic!("Incorrect length")))
            };
            for i in 0..WORD_SIZE {
                builder.assert_eq(local.max_abs_c_or_1[i], max_abs_c_or_1[i].clone());
            }

            // Handle cases:
            // - If is_real == 0 then remainder_check_multiplicity == 0 is forced.
            // - If is_real == 1 then is_c_0_result must be the expected one, so
            //   remainder_check_multiplicity = (1 - is_c_0_result) * is_real.
            builder.assert_eq(
                (AB::Expr::one() - local.is_c_0.result) * local.is_real,
                local.remainder_check_multiplicity,
            );

            // Dispatch abs(remainder) < max(abs(c), 1), this is equivalent to abs(remainder) <
            // abs(c) if not division by 0.
            <LtOperationUnsigned<AB::F> as SP1Operation<AB>>::eval(
                builder,
                LtOperationUnsignedInput::<AB>::new(
                    local.abs_remainder.map(Into::into),
                    local.max_abs_c_or_1.map(Into::into),
                    local.remainder_lt_operation,
                    local.remainder_check_multiplicity.into(),
                ),
            );
            builder
                .when(local.remainder_check_multiplicity)
                .assert_eq(one.clone(), local.remainder_lt_operation.u16_compare_operation.bit);
        }

        // Check that the MSBs are correct.
        {
            //If not word operation, we check the last limb.
            <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                U16MSBOperationInput::<AB>::new(
                    local.adapter.b()[WORD_SIZE - 1].into(),
                    local.b_msb,
                    local.is_real_not_word.into(),
                ),
            );
            <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                U16MSBOperationInput::<AB>::new(
                    local.adapter.c()[WORD_SIZE - 1].into(),
                    local.c_msb,
                    local.is_real_not_word.into(),
                ),
            );
            <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                U16MSBOperationInput::<AB>::new(
                    local.remainder[WORD_SIZE - 1].into(),
                    local.rem_msb,
                    local.is_real_not_word.into(),
                ),
            );

            //If word operation, we check the second limb.
            <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                U16MSBOperationInput::<AB>::new(
                    local.adapter.b()[WORD_SIZE / 2 - 1].into(),
                    local.b_msb,
                    is_word_operation.clone(),
                ),
            );
            <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                U16MSBOperationInput::<AB>::new(
                    local.adapter.c()[WORD_SIZE / 2 - 1].into(),
                    local.c_msb,
                    is_word_operation.clone(),
                ),
            );
            <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                U16MSBOperationInput::<AB>::new(
                    local.remainder[WORD_SIZE / 2 - 1].into(),
                    local.rem_msb,
                    is_word_operation.clone(),
                ),
            );

            // If word operation, we check the second limb of quotient.
            <U16MSBOperation<AB::F> as SP1Operation<AB>>::eval(
                builder,
                U16MSBOperationInput::<AB>::new(
                    local.quotient[WORD_SIZE / 2 - 1].into(),
                    local.quot_msb,
                    is_word_operation.clone(),
                ),
            );
        }

        // Range check all the u16 limbs and boolean carries.
        {
            builder.slice_range_check_u16(&local.quotient.0, local.is_real);
            builder.slice_range_check_u16(&local.remainder.0, local.is_real);

            local.carry.iter().for_each(|carry| {
                builder.assert_bool(*carry);
            });

            builder.slice_range_check_u16(&local.c_times_quotient, local.is_real);
        }

        // Check that the flags are boolean.
        {
            let bool_flags = [
                local.is_div,
                local.is_divu,
                local.is_rem,
                local.is_remu,
                local.is_divw,
                local.is_remw,
                local.is_divuw,
                local.is_remuw,
                local.is_overflow,
                local.is_real_not_word,
                local.b_neg,
                local.b_neg_not_overflow,
                local.b_not_neg_not_overflow,
                local.rem_neg,
                local.c_neg,
                local.is_real,
                local.abs_c_alu_event,
                local.abs_rem_alu_event,
            ];

            for flag in bool_flags.iter() {
                builder.assert_bool(*flag);
            }
        }

        // Receive the arguments.
        {
            // Exactly one of the opcode flags must be on.
            // SAFETY: All selectors `is_divu`, `is_remu`, `is_div`, `is_rem` are checked to be
            // boolean. Each row has exactly one selector turned on, as their sum is
            // checked to be one. Therefore, the `opcode` matches the corresponding
            // opcode of the instruction.
            builder.assert_eq(
                one.clone(),
                local.is_divu
                    + local.is_remu
                    + local.is_div
                    + local.is_rem
                    + local.is_divw
                    + local.is_remw
                    + local.is_divuw
                    + local.is_remuw,
            );

            // Get the opcode for the operation.
            let opcode = {
                let divu: AB::Expr = AB::F::from_canonical_u32(Opcode::DIVU as u32).into();
                let remu: AB::Expr = AB::F::from_canonical_u32(Opcode::REMU as u32).into();
                let div: AB::Expr = AB::F::from_canonical_u32(Opcode::DIV as u32).into();
                let rem: AB::Expr = AB::F::from_canonical_u32(Opcode::REM as u32).into();
                let divw: AB::Expr = AB::F::from_canonical_u32(Opcode::DIVW as u32).into();
                let remw: AB::Expr = AB::F::from_canonical_u32(Opcode::REMW as u32).into();
                let divuw: AB::Expr = AB::F::from_canonical_u32(Opcode::DIVUW as u32).into();
                let remuw: AB::Expr = AB::F::from_canonical_u32(Opcode::REMUW as u32).into();

                local.is_divu * divu
                    + local.is_remu * remu
                    + local.is_div * div
                    + local.is_rem * rem
                    + local.is_divw * divw
                    + local.is_remw * remw
                    + local.is_divuw * divuw
                    + local.is_remuw * remuw
            };

            // Compute instruction field constants for each opcode
            let funct3 = local.is_divu
                * AB::Expr::from_canonical_u8(Opcode::DIVU.funct3().unwrap())
                + local.is_remu * AB::Expr::from_canonical_u8(Opcode::REMU.funct3().unwrap())
                + local.is_div * AB::Expr::from_canonical_u8(Opcode::DIV.funct3().unwrap())
                + local.is_rem * AB::Expr::from_canonical_u8(Opcode::REM.funct3().unwrap())
                + local.is_divw * AB::Expr::from_canonical_u8(Opcode::DIVW.funct3().unwrap())
                + local.is_remw * AB::Expr::from_canonical_u8(Opcode::REMW.funct3().unwrap())
                + local.is_divuw * AB::Expr::from_canonical_u8(Opcode::DIVUW.funct3().unwrap())
                + local.is_remuw * AB::Expr::from_canonical_u8(Opcode::REMUW.funct3().unwrap());
            let funct7 = local.is_divu
                * AB::Expr::from_canonical_u8(Opcode::DIVU.funct7().unwrap())
                + local.is_remu * AB::Expr::from_canonical_u8(Opcode::REMU.funct7().unwrap())
                + local.is_div * AB::Expr::from_canonical_u8(Opcode::DIV.funct7().unwrap())
                + local.is_rem * AB::Expr::from_canonical_u8(Opcode::REM.funct7().unwrap())
                + local.is_divw * AB::Expr::from_canonical_u8(Opcode::DIVW.funct7().unwrap())
                + local.is_remw * AB::Expr::from_canonical_u8(Opcode::REMW.funct7().unwrap())
                + local.is_divuw * AB::Expr::from_canonical_u8(Opcode::DIVUW.funct7().unwrap())
                + local.is_remuw * AB::Expr::from_canonical_u8(Opcode::REMUW.funct7().unwrap());

            let base_opcode = local.base_op_code.into();

            // Constrain the state of the CPU.
            // The program counter and timestamp increment by `4` and `8`.
            <CPUState<AB::F> as SP1Operation<AB>>::eval(
                builder,
                CPUStateInput {
                    cols: local.state,
                    next_pc: [
                        local.state.pc[0] + AB::F::from_canonical_u32(PC_INC),
                        local.state.pc[1].into(),
                        local.state.pc[2].into(),
                    ],
                    clk_increment: AB::Expr::from_canonical_u32(CLK_INC),
                    is_real: local.is_real.into(),
                },
            );

            // Constrain the program and register reads.
            let alu_reader_input = ALUTypeReaderInput::<AB, AB::Expr>::new(
                local.state.clk_high::<AB>(),
                local.state.clk_low::<AB>(),
                local.state.pc,
                opcode,
                [base_opcode, funct3, funct7],
                local.a.map(|x| x.into()),
                local.adapter,
                local.is_real.into(),
            );
            ALUTypeReader::<AB::F>::eval(builder, alu_reader_input);
        }
    }
}

// #[cfg(test)]
// mod tests {
//     #![allow(clippy::print_stdout)]

//     use crate::{
//         io::SP1Stdin,
//         riscv::RiscvAir,
//         utils::{run_malicious_test, run_test_machine, setup_test_machine},
//     };
//     use slop_baby_bear::BabyBear;
//     use slop_matrix::dense::RowMajorMatrix;
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

//     use super::DivRemChip;

//     #[test]
//     fn generate_trace() {
//         let mut shard = ExecutionRecord::default();
//         shard.divrem_events = vec![AluEvent::new(0, Opcode::DIVU, 2, 17, 3, false)];
//         let chip = DivRemChip::default();
//         let trace: RowMajorMatrix<BabyBear> =
//             chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         println!("{:?}", trace.values)
//     }

//     fn neg(a: u32) -> u32 {
//         u32::MAX - a + 1
//     }

//     #[test]
//     fn prove_babybear() {
//         let mut divrem_events: Vec<AluEvent> = Vec::new();

//         let divrems: Vec<(Opcode, u32, u32, u32)> = vec![
//             (Opcode::DIVU, 3, 20, 6),
//             (Opcode::DIVU, 715827879, neg(20), 6),
//             (Opcode::DIVU, 0, 20, neg(6)),
//             (Opcode::DIVU, 0, neg(20), neg(6)),
//             (Opcode::DIVU, 1 << 31, 1 << 31, 1),
//             (Opcode::DIVU, 0, 1 << 31, neg(1)),
//             (Opcode::DIVU, u32::MAX, 1 << 31, 0),
//             (Opcode::DIVU, u32::MAX, 1, 0),
//             (Opcode::DIVU, u32::MAX, 0, 0),
//             (Opcode::REMU, 4, 18, 7),
//             (Opcode::REMU, 6, neg(20), 11),
//             (Opcode::REMU, 23, 23, neg(6)),
//             (Opcode::REMU, neg(21), neg(21), neg(11)),
//             (Opcode::REMU, 5, 5, 0),
//             (Opcode::REMU, neg(1), neg(1), 0),
//             (Opcode::REMU, 0, 0, 0),
//             (Opcode::REM, 7, 16, 9),
//             (Opcode::REM, neg(4), neg(22), 6),
//             (Opcode::REM, 1, 25, neg(3)),
//             (Opcode::REM, neg(2), neg(22), neg(4)),
//             (Opcode::REM, 0, 873, 1),
//             (Opcode::REM, 0, 873, neg(1)),
//             (Opcode::REM, 5, 5, 0),
//             (Opcode::REM, neg(5), neg(5), 0),
//             (Opcode::REM, 0, 0, 0),
//             (Opcode::REM, 0, 0x80000001, neg(1)),
//             (Opcode::DIV, 3, 18, 6),
//             (Opcode::DIV, neg(6), neg(24), 4),
//             (Opcode::DIV, neg(2), 16, neg(8)),
//             (Opcode::DIV, neg(1), 0, 0),
//             (Opcode::DIV, 1 << 31, 1 << 31, neg(1)),
//             (Opcode::REM, 0, 1 << 31, neg(1)),
//         ];
//         for t in divrems.iter() {
//             divrem_events.push(AluEvent::new(0, t.0, t.1, t.2, t.3, false));
//         }

//         // Append more events until we have 1000 tests.
//         for _ in 0..(1000 - divrems.len()) {
//             divrem_events.push(AluEvent::new(0, Opcode::DIVU, 1, 1, 1, false));
//         }

//         let mut shard = ExecutionRecord::default();
//         shard.divrem_events = divrem_events;

//         // Run setup.
//         let air = DivRemChip::default();
//         let config = BabyBearPoseidon2::new();
//         let chip = Chip::new(air);
//         let (pk, vk) = setup_test_machine(StarkMachine::new(
//             config.clone(),
//             vec![chip],
//             SP1_PROOF_NUM_PV_ELTS,
//             true,
//         ));

//         // Run the test.
//         let air = DivRemChip::default();
//         let chip: Chip<BabyBear, DivRemChip> = Chip::new(air);
//         let machine = StarkMachine::new(config.clone(), vec![chip], SP1_PROOF_NUM_PV_ELTS, true);
//         run_test_machine::<BabyBearPoseidon2, DivRemChip>(vec![shard], machine, pk, vk).unwrap();
//     }

//     #[test]
//     fn test_malicious_divrem() {
//         const NUM_TESTS: usize = 5;

//         for opcode in [Opcode::DIV, Opcode::DIVU, Opcode::REM, Opcode::REMU] {
//             for _ in 0..NUM_TESTS {
//                 let (correct_op_a, op_b, op_c) = if opcode == Opcode::DIV {
//                     let op_b = thread_rng().gen_range(0..i32::MAX);
//                     let op_c = thread_rng().gen_range(0..i32::MAX);
//                     ((op_b / op_c) as u32, op_b as u32, op_c as u32)
//                 } else if opcode == Opcode::DIVU {
//                     let op_b = thread_rng().gen_range(0..u32::MAX);
//                     let op_c = thread_rng().gen_range(0..u32::MAX);
//                     (op_b / op_c, op_b as u32, op_c as u32)
//                 } else if opcode == Opcode::REM {
//                     let op_b = thread_rng().gen_range(0..i32::MAX);
//                     let op_c = thread_rng().gen_range(0..i32::MAX);
//                     ((op_b % op_c) as u32, op_b as u32, op_c as u32)
//                 } else if opcode == Opcode::REMU {
//                     let op_b = thread_rng().gen_range(0..u32::MAX);
//                     let op_c = thread_rng().gen_range(0..u32::MAX);
//                     (op_b % op_c, op_b as u32, op_c as u32)
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
//                     malicious_record.cpu_events[0].a = op_a;
//                     if let Some(MemoryRecordEnum::Write(mut write_record)) =
//                         malicious_record.cpu_events[0].a_record
//                     {
//                         write_record.value = op_a;
//                     }
//                     malicious_record.divrem_events[0].a = op_a;
//                     prover.generate_traces(&malicious_record)
//                 };

//                 let result =
//                     run_malicious_test::<P>(program, stdin,
// Box::new(malicious_trace_pv_generator));                 assert!(result.is_err() &&
// result.unwrap_err().is_constraints_failing());             }
//         }
//     }
// }

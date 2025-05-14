use p3_field::AbstractField;
use sp1_core_executor::{events::ByteRecord, Opcode};
use sp1_stark::{air::SP1AirBuilder, Word};

use p3_field::Field;
use sp1_derive::AlignedBorrow;

use super::{BitwiseOperation, U16toU8Operation};

/// A set of columns needed to compute the bitwise operation over two u16 limbs.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct BitwiseU16Operation<T> {
    /// Lower byte of two limbs of `b`.
    pub b_low_bytes: U16toU8Operation<T>,

    /// Lower byte of two limbs of `c`.
    pub c_low_bytes: U16toU8Operation<T>,

    /// The bitwise operation over bytes.
    pub bitwise_operation: BitwiseOperation<T>,
}

impl<F: Field> BitwiseU16Operation<F> {
    pub fn populate_bitwise(
        &mut self,
        record: &mut impl ByteRecord,
        a_u64: u64,
        b_u64: u64,
        c_u64: u64,
        opcode: Opcode,
    ) {
        self.b_low_bytes.populate_u16_to_u8_unsafe(record, b_u64);
        self.c_low_bytes.populate_u16_to_u8_unsafe(record, c_u64);
        self.bitwise_operation.populate_bitwise(record, a_u64, b_u64, c_u64, opcode);
    }

    /// Evaluate the bitwise operation over two `Word`s of two u16 limbs.
    /// Assumes that the two words are valid `Word`s of two u16 limbs.
    /// Assumes that `opcode` is a valid byte opcode.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, the return value is constrained to be correct.
    pub fn eval_bitwise_u16<AB: SP1AirBuilder>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: BitwiseU16Operation<AB::Var>,
        opcode: AB::Expr,
        is_real: AB::Expr,
    ) -> Word<AB::Expr> {
        builder.assert_bool(is_real.clone());

        // Convert the two words to bytes using the unsafe API.
        // SAFETY: This is safe because the `BitwiseOperation` will range check the bytes.
        let b_bytes =
            U16toU8Operation::<AB::F>::eval_u16_to_u8_unsafe(builder, b.0, cols.b_low_bytes);
        let c_bytes =
            U16toU8Operation::<AB::F>::eval_u16_to_u8_unsafe(builder, c.0, cols.c_low_bytes);
        // SAFETY: This is safe because `is_real` is constrained to be boolean.
        BitwiseOperation::<AB::F>::eval_bitwise(
            builder,
            b_bytes,
            c_bytes,
            cols.bitwise_operation,
            opcode,
            is_real,
        );

        // Combine the byte results into two u16 limbs.
        let result_limb0 = cols.bitwise_operation.result[0]
            + cols.bitwise_operation.result[1] * AB::F::from_canonical_u32(1 << 8);
        let result_limb1 = cols.bitwise_operation.result[2]
            + cols.bitwise_operation.result[3] * AB::F::from_canonical_u32(1 << 8);
        let result_limb2 = cols.bitwise_operation.result[4]
            + cols.bitwise_operation.result[5] * AB::F::from_canonical_u32(1 << 8);
        let result_limb3 = cols.bitwise_operation.result[6]
            + cols.bitwise_operation.result[7] * AB::F::from_canonical_u32(1 << 8);
        Word([result_limb0, result_limb1, result_limb2, result_limb3])
    }
}

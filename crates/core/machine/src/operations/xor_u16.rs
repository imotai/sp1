use p3_field::AbstractField;
use sp1_core_executor::events::ByteRecord;
use sp1_stark::{air::SP1AirBuilder, Word};

use p3_field::Field;
use sp1_derive::AlignedBorrow;

use super::{U16toU8Operation, XorOperation};

/// A set of columns needed to compute the xor operation over two u16 limbs.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct XorU16Operation<T> {
    /// Lower byte of two limbs of `b`.
    pub b_low_bytes: U16toU8Operation<T>,

    /// Lower byte of two limbs of `c`.
    pub c_low_bytes: U16toU8Operation<T>,

    /// The bitwise operation over bytes.
    pub xor_operation: XorOperation<T>,
}

impl<F: Field> XorU16Operation<F> {
    pub fn populate_xor_u16(
        &mut self,
        record: &mut impl ByteRecord,
        b_u32: u32,
        c_u32: u32,
    ) -> u32 {
        let expected = b_u32 ^ c_u32;
        self.b_low_bytes.populate_u16_to_u8_unsafe(record, b_u32);
        self.c_low_bytes.populate_u16_to_u8_unsafe(record, c_u32);
        self.xor_operation.populate(record, b_u32, c_u32);
        expected
    }

    /// Evaluate the xor operation over two `Word`s of two u16 limbs.
    /// Assumes that the two words are valid `Word`s of two u16 limbs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, the return value is constrained to be correct.
    pub fn eval_xor_u16<AB: SP1AirBuilder>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: XorU16Operation<AB::Var>,
        is_real: AB::Var,
    ) -> Word<AB::Expr> {
        builder.assert_bool(is_real);

        // Convert the two words to bytes using the unsafe API.
        // SAFETY: This is safe because the `XorOperation` will range check the bytes.
        let b_bytes =
            U16toU8Operation::<AB::F>::eval_u16_to_u8_unsafe(builder, b.0, cols.b_low_bytes);
        let c_bytes =
            U16toU8Operation::<AB::F>::eval_u16_to_u8_unsafe(builder, c.0, cols.c_low_bytes);
        // SAFETY: This is safe because `is_real` is constrained to be boolean.
        XorOperation::<AB::F>::eval(builder, b_bytes, c_bytes, cols.xor_operation, is_real);

        // Combine the byte results into two u16 limbs.
        let result_limb0 = cols.xor_operation.value[0]
            + cols.xor_operation.value[1] * AB::F::from_canonical_u32(1 << 8);
        let result_limb1 = cols.xor_operation.value[2]
            + cols.xor_operation.value[3] * AB::F::from_canonical_u32(1 << 8);
        Word([result_limb0, result_limb1])
    }
}

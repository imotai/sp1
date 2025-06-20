use p3_field::AbstractField;
use sp1_core_executor::events::ByteRecord;
use sp1_stark::{air::SP1AirBuilder, Word};

use p3_field::Field;
use sp1_derive::AlignedBorrow;

use crate::{
    air::{SP1Operation, SP1OperationBuilder},
    operations::{U16toU8OperationUnsafe, U16toU8OperationUnsafeInput},
};

use super::{AndOperation, U16toU8Operation};

/// A set of columns needed to compute the and operation over two u16 limbs.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AndU16Operation<T> {
    /// Lower byte of two limbs of `b`.
    pub b_low_bytes: U16toU8Operation<T>,

    /// Lower byte of two limbs of `c`.
    pub c_low_bytes: U16toU8Operation<T>,

    /// The bitwise operation over bytes.
    pub and_operation: AndOperation<T>,
}

impl<F: Field> AndU16Operation<F> {
    pub fn populate_and_u16(
        &mut self,
        record: &mut impl ByteRecord,
        b_u64: u64,
        c_u64: u64,
    ) -> u64 {
        self.b_low_bytes.populate_u16_to_u8_unsafe(record, b_u64);
        self.c_low_bytes.populate_u16_to_u8_unsafe(record, c_u64);
        self.and_operation.populate(record, b_u64, c_u64)
    }

    /// Evaluate the and operation over two `Word`s of two u16 limbs.
    /// Assumes that the two words are valid `Word`s of two u16 limbs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, the return value is constrained to be correct.
    pub fn eval_and_u16<AB: SP1AirBuilder + SP1OperationBuilder<U16toU8OperationUnsafe>>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: AndU16Operation<AB::Var>,
        is_real: AB::Var,
    ) -> Word<AB::Expr> {
        builder.assert_bool(is_real);

        // Convert the two words to bytes using the unsafe API.
        // SAFETY: This is safe because the `AndOperation` will range check the bytes.
        let b_input = U16toU8OperationUnsafeInput::new(b.0, cols.b_low_bytes);
        let b_bytes = U16toU8OperationUnsafe::eval(builder, b_input);
        let c_input = U16toU8OperationUnsafeInput::new(c.0, cols.c_low_bytes);
        let c_bytes = U16toU8OperationUnsafe::eval(builder, c_input);
        // SAFETY: This is safe because `is_real` is constrained to be boolean.
        AndOperation::<AB::F>::eval(builder, b_bytes, c_bytes, cols.and_operation, is_real);

        // Combine the byte results into two u16 limbs.
        let result_limb0 = cols.and_operation.value[0]
            + cols.and_operation.value[1] * AB::F::from_canonical_u32(1 << 8);
        let result_limb1 = cols.and_operation.value[2]
            + cols.and_operation.value[3] * AB::F::from_canonical_u32(1 << 8);
        let result_limb2 = cols.and_operation.value[4]
            + cols.and_operation.value[5] * AB::F::from_canonical_u32(1 << 8);
        let result_limb3 = cols.and_operation.value[6]
            + cols.and_operation.value[7] * AB::F::from_canonical_u32(1 << 8);
        Word([result_limb0, result_limb1, result_limb2, result_limb3])
    }
}

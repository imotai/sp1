use slop_algebra::AbstractField;
use sp1_core_executor::ExecutionRecord;
use sp1_stark::{air::SP1AirBuilder, Word};

use slop_algebra::Field;
use sp1_derive::AlignedBorrow;

use crate::{
    air::{SP1Operation, SP1OperationBuilder},
    operations::{U16toU8OperationUnsafe, U16toU8OperationUnsafeInput},
};

use super::{OrOperation, U16toU8Operation};

/// A set of columns needed to compute the or operation over two u16 limbs.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct OrU16Operation<T> {
    /// Lower byte of two limbs of `b`.
    pub b_low_bytes: U16toU8Operation<T>,

    /// Lower byte of two limbs of `c`.
    pub c_low_bytes: U16toU8Operation<T>,

    /// The bitwise operation over bytes.
    pub or_operation: OrOperation<T>,
}

impl<F: Field> OrU16Operation<F> {
    pub fn populate_or_u16(&mut self, record: &mut ExecutionRecord, b_u32: u32, c_u32: u32) {
        self.b_low_bytes.populate_u16_to_u8_unsafe(record, b_u32);
        self.c_low_bytes.populate_u16_to_u8_unsafe(record, c_u32);
        self.or_operation.populate(record, b_u32, c_u32);
    }

    /// Evaluate the or operation over two `Word`s of two u16 limbs.
    /// Assumes that the two words are valid `Word`s of two u16 limbs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, the return value is constrained to be correct.
    pub fn eval_or_u16<AB: SP1AirBuilder + SP1OperationBuilder<U16toU8OperationUnsafe>>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: OrU16Operation<AB::Var>,
        is_real: AB::Var,
    ) -> Word<AB::Expr> {
        builder.assert_bool(is_real);

        // Convert the two words to bytes using the unsafe API.
        // SAFETY: This is safe because the `OrOperation` will range check the bytes.
        let b_input = U16toU8OperationUnsafeInput::new(b.0, cols.b_low_bytes);
        let b_bytes = U16toU8OperationUnsafe::eval(builder, b_input);
        let c_input = U16toU8OperationUnsafeInput::new(c.0, cols.c_low_bytes);
        let c_bytes = U16toU8OperationUnsafe::eval(builder, c_input);
        // SAFETY: This is safe because `is_real` is constrained to be boolean.
        OrOperation::<AB::F>::eval(builder, b_bytes, c_bytes, cols.or_operation, is_real);

        // Combine the byte results into two u16 limbs.
        let result_limb0 = cols.or_operation.value[0]
            + cols.or_operation.value[1] * AB::F::from_canonical_u32(1 << 8);
        let result_limb1 = cols.or_operation.value[2]
            + cols.or_operation.value[3] * AB::F::from_canonical_u32(1 << 8);
        Word([result_limb0, result_limb1])
    }
}

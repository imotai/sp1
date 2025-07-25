use slop_algebra::{AbstractField, Field};
use sp1_core_executor::{events::ByteRecord, ByteOpcode, ExecutionRecord};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_BYTE_SIZE;
use sp1_stark::air::SP1AirBuilder;

/// A set of columns needed to compute the or operation over four bytes.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct OrOperation<T> {
    /// The result of `x | y`.
    pub value: [T; WORD_BYTE_SIZE],
}

impl<F: Field> OrOperation<F> {
    pub fn populate(&mut self, record: &mut ExecutionRecord, x: u64, y: u64) -> u64 {
        let expected = x | y;
        let x_bytes = x.to_le_bytes();
        let y_bytes = y.to_le_bytes();
        for i in 0..WORD_BYTE_SIZE {
            self.value[i] = F::from_canonical_u8(x_bytes[i] | y_bytes[i]);
            record.lookup_or(x_bytes[i], y_bytes[i]);
        }
        expected
    }

    /// Evaluate the or operation over four bytes.
    /// Assumes that `is_real` is boolean.
    /// If `is_real` is true, constrains that the inputs are valid bytes.
    /// If `is_real` is true, constrains that the `value` is the correct result.
    pub fn eval<AB: SP1AirBuilder>(
        builder: &mut AB,
        a: [AB::Expr; WORD_BYTE_SIZE],
        b: [AB::Expr; WORD_BYTE_SIZE],
        cols: OrOperation<AB::Var>,
        is_real: AB::Var,
    ) {
        // The byte table will constrain that, if `is_real` is true,
        //  - `a[i], b[i]` are bytes.
        //  - `value[i] = a[i] | b[i]`.
        for i in 0..WORD_BYTE_SIZE {
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::OR as u32),
                cols.value[i],
                a[i].clone(),
                b[i].clone(),
                is_real,
            );
        }
    }
}

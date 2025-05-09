use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;

use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use sp1_primitives::consts::WORD_BYTE_SIZE;
use sp1_stark::air::SP1AirBuilder;

/// A set of columns needed to compute the and operation over four bytes.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AndOperation<T> {
    /// The result of `x & y`.
    pub value: [T; WORD_BYTE_SIZE],
}

impl<F: Field> AndOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, x: u32, y: u32) -> u32 {
        let expected = x & y;
        let x_bytes = x.to_le_bytes();
        let y_bytes = y.to_le_bytes();
        for i in 0..WORD_BYTE_SIZE {
            let and = x_bytes[i] & y_bytes[i];
            self.value[i] = F::from_canonical_u8(and);

            let byte_event = ByteLookupEvent {
                opcode: ByteOpcode::AND,
                a: and as u16,
                b: x_bytes[i],
                c: y_bytes[i],
            };
            record.add_byte_lookup_event(byte_event);
        }
        expected
    }

    /// Evaluate the and operation over four bytes.
    /// Assumes that `is_real` is boolean.
    /// If `is_real` is true, constrains that the inputs are valid bytes.
    /// If `is_real` is true, constrains that the `value` is the correct result.
    #[allow(unused_variables)]
    pub fn eval<AB: SP1AirBuilder>(
        builder: &mut AB,
        a: [AB::Expr; WORD_BYTE_SIZE],
        b: [AB::Expr; WORD_BYTE_SIZE],
        cols: AndOperation<AB::Var>,
        is_real: AB::Var,
    ) {
        // The byte table will constrain that, if `is_real` is true,
        //  - `a[i], b[i]` are bytes.
        //  - `value[i] = a[i] & b[i]`.
        for i in 0..WORD_BYTE_SIZE {
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::AND as u32),
                cols.value[i],
                a[i].clone(),
                b[i].clone(),
                is_real,
            );
        }
    }
}

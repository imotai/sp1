use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, Opcode,
};
use sp1_primitives::consts::WORD_BYTE_SIZE;
use sp1_stark::air::SP1AirBuilder;

use p3_field::Field;
use sp1_derive::AlignedBorrow;

/// A set of columns needed to compute the bitwise operation over four bytes.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct BitwiseOperation<T> {
    /// The result of the bitwise operation in bytes.
    pub result: [T; WORD_BYTE_SIZE],
}

impl<F: Field> BitwiseOperation<F> {
    pub fn populate_bitwise(
        &mut self,
        record: &mut impl ByteRecord,
        a_u64: u64,
        b_u64: u64,
        c_u64: u64,
        opcode: Opcode,
    ) {
        let a = a_u64.to_le_bytes();
        let b = b_u64.to_le_bytes();
        let c = c_u64.to_le_bytes();

        // self.result = a.map(|x| F::from_canonical_u8(x));
        // TODO: u64

        for ((b_a, b_b), b_c) in a.into_iter().zip(b).zip(c) {
            let byte_event =
                ByteLookupEvent { opcode: ByteOpcode::from(opcode), a: b_a as u16, b: b_b, c: b_c };
            record.add_byte_lookup_event(byte_event);
        }
    }

    /// Evaluate the bitwise operation over four bytes.
    /// Assumes that `is_real` is boolean.
    /// If `is_real` is true, constrains that the inputs are valid bytes.
    /// If `is_real` is true, constrains that the `result` is the correct result.
    pub fn eval_bitwise<AB: SP1AirBuilder>(
        builder: &mut AB,
        a: [AB::Expr; WORD_BYTE_SIZE],
        b: [AB::Expr; WORD_BYTE_SIZE],
        cols: BitwiseOperation<AB::Var>,
        opcode: AB::Expr,
        is_real: AB::Expr,
    ) {
        // The byte table will constrain that, if `is_real` is true,
        //  - `a[i], b[i]` are bytes.
        //  - `result[i] = op(a[i], b[i])`.
        for i in 0..WORD_BYTE_SIZE {
            builder.send_byte(
                opcode.clone(),
                cols.result[i],
                a[i].clone(),
                b[i].clone(),
                is_real.clone(),
            );
        }
    }
}

use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use sp1_stark::air::SP1AirBuilder;

use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;

/// Operation columns for computing the most significant bit of a u16.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct U16MSBOperation<T> {
    /// The result of the msb operation.
    pub msb: T,
}

impl<F: Field> U16MSBOperation<F> {
    pub fn populate_msb(&mut self, record: &mut impl ByteRecord, a_u16: u16, is_real: bool) -> u32 {
        let msb = (a_u16 >> 15) & 1;
        self.msb = F::from_canonical_u16(msb);
        let diff = a_u16.wrapping_mul(2u16);
        if is_real {
            record.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::Range,
                a: diff,
                b: 16,
                c: 0,
            });
        }
        msb as u32
    }

    /// Evaluate the `U16MSBOperation` on the given inputs.
    /// Assumes that `a` is a valid u16.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, it constrains that the result is the msb of `a`.
    pub fn eval_msb<AB: SP1AirBuilder>(
        builder: &mut AB,
        a: AB::Expr,
        cols: U16MSBOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        builder.assert_bool(is_real.clone());
        builder.assert_bool(cols.msb);
        let two = AB::Expr::from_canonical_u32(2);
        let base = AB::Expr::from_canonical_u32(1 << 16);
        let diff = two * a - cols.msb * base;
        // Constrains that `2 * a - msb * 2^16` is in u16 range, and `msb` is boolean.
        // If `0 <= a < 2^15`, then `msb` must be 0.
        // If `2^15 <= a < 2^16`, then `msb` must be 1.
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::Range as u8),
            diff,
            AB::Expr::from_canonical_u32(16),
            AB::Expr::zero(),
            is_real,
        );
    }
}

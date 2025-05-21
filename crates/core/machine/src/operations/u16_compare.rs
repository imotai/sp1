use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use sp1_stark::air::SP1AirBuilder;

use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct U16CompareOperation<T> {
    /// The result of the compare operation (1 if a < b, 0 if a >= b)
    pub bit: T,
}

impl<F: Field> U16CompareOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u16: u16, b_u16: u16, c_u16: u16) {
        self.bit = F::from_canonical_u16(a_u16);
        let diff = b_u16.wrapping_sub(c_u16);
        // record.add_byte_lookup_event(ByteLookupEvent {
        //     opcode: ByteOpcode::Range,
        //     a: diff as u16,
        //     b: 16,
        //     c: 0,
        // });
    }

    /// Evaluate the compare operation.
    /// Assumes that `a`, `b` are both u16.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, constrains that the result is correctly computed.
    pub fn eval_compare_u16<AB: SP1AirBuilder>(
        builder: &mut AB,
        a: AB::Expr,
        b: AB::Expr,
        cols: U16CompareOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        builder.assert_bool(is_real.clone());
        builder.assert_bool(cols.bit);
        let base = AB::Expr::from_canonical_u32(1 << 16);
        let diff = a - b + cols.bit * base;
        // Since `a, b` are both u16, `a - b` will be in `(-2^16, 2^16)`.
        // If `a < b`, then `bit` must be one for `diff` to be in `[0, 2^16)`.
        // If `a >= b`, then `bit` must be zero for `diff` to be in `[0, 2^16)`.
        // With correct `bit`, the `diff` value will be in u16 range.
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::Range as u8),
            diff,
            AB::Expr::from_canonical_u32(16),
            AB::Expr::zero(),
            is_real.clone(),
        );
    }
}

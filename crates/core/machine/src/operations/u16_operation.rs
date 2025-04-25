use sp1_core_executor::events::ByteRecord;
use sp1_primitives::consts::{u32_to_u16_limbs, WORD_BYTE_SIZE, WORD_SIZE};
use sp1_stark::air::SP1AirBuilder;

use crate::air::WordAirBuilder;
use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;

/// A set of columns for a u16 to u8 adapter.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct U16toU8Operation<T> {
    low_bytes: [T; WORD_SIZE], // least significant byte of each u16 limb
}

impl<F: Field> U16toU8Operation<F> {
    pub fn populate_u16_to_u8_unsafe(&mut self, _: &mut impl ByteRecord, a_u32: u32) {
        let a_limbs = u32_to_u16_limbs(a_u32);
        let mut ret = [0u8; WORD_SIZE];
        for i in 0..WORD_SIZE {
            ret[i] = (a_limbs[i] % 256) as u8;
        }
        self.low_bytes = ret.map(|x| F::from_canonical_u8(x));
    }

    pub fn populate_u16_to_u8_safe(&mut self, record: &mut impl ByteRecord, a_u32: u32) {
        let a_limbs = u32_to_u16_limbs(a_u32);
        let mut ret = [0u8; WORD_SIZE];
        for i in 0..WORD_SIZE {
            ret[i] = (a_limbs[i] % 256) as u8;
            let upper = ((a_limbs[i] - ret[i] as u16) / 256) as u8;
            record.add_u8_range_check(ret[i], upper);
        }
        self.low_bytes = ret.map(|x| F::from_canonical_u8(x));
    }

    /// Converts two u16 limbs into four u8 limbs.
    /// This function assumes that the u8 limbs will be range checked.
    pub fn eval_u16_to_u8_unsafe<AB: SP1AirBuilder>(
        _: &mut AB,
        u16_values: [AB::Expr; WORD_SIZE],
        cols: U16toU8Operation<AB::Var>,
    ) -> [AB::Expr; WORD_BYTE_SIZE] {
        let mut ret = [AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()];
        let divisor = AB::F::from_canonical_u32(1 << 8).inverse();

        for i in 0..WORD_SIZE {
            ret[i * 2] = cols.low_bytes[i].into();
            ret[i * 2 + 1] = (u16_values[i].clone() - ret[i * 2].clone()) * divisor;
        }

        ret
    }

    /// Converts two u16 limbs into four u8 limbs.
    /// This function range checks the four u8 limbs.
    pub fn eval_u16_to_u8_safe<AB: SP1AirBuilder>(
        builder: &mut AB,
        u16_values: [AB::Expr; WORD_SIZE],
        cols: U16toU8Operation<AB::Var>,
        is_real: AB::Expr,
    ) -> [AB::Expr; WORD_BYTE_SIZE] {
        let ret = U16toU8Operation::<AB::F>::eval_u16_to_u8_unsafe(builder, u16_values, cols);
        builder.slice_range_check_u8(&ret, is_real);
        ret
    }
}

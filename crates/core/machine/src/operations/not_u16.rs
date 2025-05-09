use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{u32_to_u16_limbs, WORD_SIZE};
use sp1_stark::{air::SP1AirBuilder, Word};

/// A set of columns needed to compute the not of a word.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct NotU16Operation<T> {
    /// The result of `!x`.
    pub value: Word<T>,
}

impl<F: Field> NotU16Operation<F> {
    pub fn populate(&mut self, x: u32) -> u32 {
        let expected = !x;
        let x_limbs = u32_to_u16_limbs(x);
        for i in 0..WORD_SIZE {
            self.value[i] = F::from_canonical_u16(!x_limbs[i]);
        }
        expected
    }

    /// Evaluate the not operation over a `Word` of two u16 limbs.
    /// Assumes that the input is a valid `Word` of two u16 limbs.
    /// If `is_real` is non-zero, constrains that the `value` is correct.
    #[allow(unused_variables)]
    pub fn eval<AB: SP1AirBuilder>(
        builder: &mut AB,
        a: Word<AB::Expr>,
        cols: NotU16Operation<AB::Var>,
        is_real: impl Into<AB::Expr> + Copy,
    ) {
        // For any u16 limb b, b + !b = 0xFFFF.
        for i in 0..WORD_SIZE {
            builder
                .when(is_real)
                .assert_eq(cols.value[i] + a[i].clone(), AB::F::from_canonical_u16(u16::MAX));
        }
    }
}

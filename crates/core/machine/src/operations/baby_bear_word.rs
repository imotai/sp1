use p3_air::AirBuilder;
use p3_field::{AbstractField, Field, PrimeField32};
use serde::{Deserialize, Serialize};
use sp1_core_executor::events::ByteRecord;
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{BaseAirBuilder, SP1AirBuilder},
    Word,
};

use crate::air::{SP1Operation, SP1OperationBuilder};

use super::{U16CompareOperation, U16CompareOperationInput};

/// A set of columns needed to range check a BabyBear word.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct BabyBearWordRangeChecker<T> {
    /// Most significant limb is less than 15 * 2^11 = 30720.
    pub most_sig_limb_lt_30720: U16CompareOperation<T>,
}

impl<F: PrimeField32> BabyBearWordRangeChecker<F> {
    pub fn populate(&mut self, value: Word<F>, record: &mut impl ByteRecord) {
        let ms_limb = value[1].as_canonical_u32() as u16;
        self.most_sig_limb_lt_30720.populate(record, (ms_limb < 30720) as u16, ms_limb, 30720u16);
    }
}

impl<F: Field> BabyBearWordRangeChecker<F> {
    /// Constrains that `value` represents a value less than the BabyBear modulus.
    /// Assumes that `value` is a valid `Word` of two u16 limbs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, constrains that `value` is a valid BabyBear word.
    pub fn range_check<AB>(
        builder: &mut AB,
        value: Word<AB::Var>,
        cols: BabyBearWordRangeChecker<AB::Var>,
        is_real: AB::Expr,
    ) where
        AB: SP1AirBuilder + SP1OperationBuilder<U16CompareOperation<<AB as AirBuilder>::F>>,
    {
        builder.assert_bool(is_real.clone());

        // Note that BabyBear modulus is 2^31 - 2^27 + 1 = 15 * 2^27 + 1.
        // First, check if the most significant limb is less than 15 * 2^11 = 30720.
        <U16CompareOperation<AB::F> as SP1Operation<AB>>::eval(
            builder,
            U16CompareOperationInput::<AB>::new(
                value[1].into(),
                AB::Expr::from_canonical_u16(30720),
                cols.most_sig_limb_lt_30720,
                is_real.clone(),
            ),
        );

        // If the range check bit is off, the most significant limb is >= 15 * 2^11 = 30720.
        // To be a valid BabyBear word, the most significant limb must be 15 * 2^11 = 30720.
        builder
            .when(is_real.clone())
            .when_not(cols.most_sig_limb_lt_30720.bit)
            .assert_eq(value[1], AB::Expr::from_canonical_u16(30720));

        // Moreover, if the most significant limb = 15 * 2^11, then the other limb must be zero.
        builder
            .when(is_real.clone())
            .when_not(cols.most_sig_limb_lt_30720.bit)
            .assert_zero(value[0]);
    }
}

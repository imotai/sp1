use p3_air::AirBuilder;
use p3_field::Field;
use serde::{Deserialize, Serialize};
use sp1_derive::{AlignedBorrow, InputExpr, InputParams, IntoShape, SP1OperationBuilder};
use sp1_primitives::consts::u32_to_u16_limbs;
use sp1_stark::{air::SP1AirBuilder, Word};

use crate::{
    air::{SP1Operation, SP1OperationBuilder},
    operations::IsZeroOperation,
};

use super::{IsZeroWordOperation, IsZeroWordOperationInput};

/// A set of columns needed to compute the equality of two words.
#[derive(
    AlignedBorrow,
    Default,
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    IntoShape,
    SP1OperationBuilder,
)]
#[repr(C)]
pub struct IsEqualWordOperation<T> {
    /// An operation to check whether the differences in limbs are all 0 (i.e., `a[0] - b[0]`,
    /// `a[1] - b[1]`). The result of `IsEqualWordOperation` is
    /// `is_diff_zero.result`.
    pub is_diff_zero: IsZeroWordOperation<T>,
}

impl<F: Field> IsEqualWordOperation<F> {
    pub fn populate(&mut self, a_u32: u32, b_u32: u32) -> u32 {
        let a = u32_to_u16_limbs(a_u32);
        let b = u32_to_u16_limbs(b_u32);
        let diff = [
            F::from_canonical_u16(a[0]) - F::from_canonical_u16(b[0]),
            F::from_canonical_u16(a[1]) - F::from_canonical_u16(b[1]),
        ];
        self.is_diff_zero.populate_from_field_element(Word(diff));
        (a_u32 == b_u32) as u32
    }

    /// Evaluate the `IsEqualWordOperation` on the given inputs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, it constrains that the result is `a == b`.
    fn eval_is_equal_word<
        AB: SP1AirBuilder
            + SP1OperationBuilder<IsZeroOperation<<AB as AirBuilder>::F>>
            + SP1OperationBuilder<IsZeroWordOperation<<AB as AirBuilder>::F>>,
    >(
        builder: &mut AB,
        a: Word<AB::Expr>,
        b: Word<AB::Expr>,
        cols: IsEqualWordOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        builder.assert_bool(is_real.clone());

        // Calculate differences in limbs.
        let diff = Word([a[0].clone() - b[0].clone(), a[1].clone() - b[1].clone()]);

        // Check if the difference is 0.
        <IsZeroWordOperation<AB::F> as SP1Operation<AB>>::eval(
            builder,
            IsZeroWordOperationInput::new(diff, cols.is_diff_zero, is_real.clone()),
        );
    }
}

#[derive(Clone, InputExpr, InputParams)]
pub struct IsEqualWordOperationInput<AB: SP1AirBuilder> {
    pub a: Word<AB::Expr>,
    pub b: Word<AB::Expr>,
    pub cols: IsEqualWordOperation<AB::Var>,
    pub is_real: AB::Expr,
}

impl<AB: SP1AirBuilder> IsEqualWordOperationInput<AB> {
    pub fn new(
        a: Word<AB::Expr>,
        b: Word<AB::Expr>,
        cols: IsEqualWordOperation<AB::Var>,
        is_real: AB::Expr,
    ) -> Self {
        Self { a, b, cols, is_real }
    }
}

impl<
        AB: SP1AirBuilder
            + SP1OperationBuilder<IsZeroOperation<<AB as AirBuilder>::F>>
            + SP1OperationBuilder<IsZeroWordOperation<<AB as AirBuilder>::F>>,
    > SP1Operation<AB> for IsEqualWordOperation<AB::F>
{
    type Input = IsEqualWordOperationInput<AB>;
    type Output = ();

    fn lower(builder: &mut AB, input: Self::Input) {
        Self::eval_is_equal_word(builder, input.a, input.b, input.cols, input.is_real);
    }
}

//! An operation to check if the input word is 0.
//!
//! This is bijective (i.e., returns 1 if and only if the input is 0). It is also worth noting that
//! this operation doesn't do a range check.
use p3_air::AirBuilder;
use p3_field::Field;
use serde::{Deserialize, Serialize};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{air::SP1AirBuilder, Word};

use crate::air::{SP1Operation, SP1OperationBuilder};

use super::IsZeroOperation;

/// A set of columns needed to compute whether the given word is 0.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct IsZeroWordOperation<T> {
    /// `IsZeroOperation` to check if each limb in the input word is zero.
    pub is_zero_limb: [IsZeroOperation<T>; WORD_SIZE],

    /// A boolean flag indicating whether the word is zero. This equals `is_zero_limb[0] * ... *
    /// is_zero_limb[WORD_SIZE - 1]`.
    pub result: T,
}

impl<F: Field> IsZeroWordOperation<F> {
    pub fn populate(&mut self, a_u32: u32) -> u32 {
        self.populate_from_field_element(Word::from(a_u32))
    }

    pub fn populate_from_field_element(&mut self, a: Word<F>) -> u32 {
        let mut is_zero = true;
        for i in 0..WORD_SIZE {
            is_zero &= self.is_zero_limb[i].populate_from_field_element(a[i]) == 1;
        }
        self.result = F::from_bool(is_zero);
        is_zero as u32
    }

    /// Evaluate the `IsZeroWordOperation` on the given inputs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, it constrains that the result is `a == 0`.
    fn eval_zero_word<
        AB: SP1AirBuilder + SP1OperationBuilder<IsZeroOperation<<AB as AirBuilder>::F>>,
    >(
        builder: &mut AB,
        a: Word<AB::Expr>,
        cols: IsZeroWordOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Calculate whether each limb is 0.
        for i in 0..WORD_SIZE {
            IsZeroOperation::<AB::F>::eval(
                builder,
                (a[i].clone(), cols.is_zero_limb[i], is_real.clone()),
            )
        }

        builder.assert_bool(is_real.clone());
        builder.assert_bool(cols.result);
        builder
            .when(is_real.clone())
            .assert_eq(cols.result, cols.is_zero_limb[0].result * cols.is_zero_limb[1].result);
    }
}

impl<AB: SP1AirBuilder + SP1OperationBuilder<IsZeroOperation<<AB as AirBuilder>::F>>>
    SP1Operation<AB> for IsZeroWordOperation<AB::F>
{
    type Input = (Word<AB::Expr>, IsZeroWordOperation<AB::Var>, AB::Expr);
    type Output = ();

    fn lower(builder: &mut AB, input: Self::Input) {
        let (a, cols, is_real) = input;
        Self::eval_zero_word(builder, a, cols, is_real);
    }
}

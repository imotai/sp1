use itertools::izip;
use sp1_core_executor::events::ByteRecord;
use sp1_stark::{air::SP1AirBuilder, Word};

use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{u32_to_u16_limbs, WORD_SIZE};

use super::{U16CompareOperation, U16MSBOperation};

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LtOperationUnsigned<T> {
    /// Instance of the U16CompareOperation.
    pub u16_compare_operation: U16CompareOperation<T>,
    /// Boolean flag to indicate which byte pair differs if the operands are not equal.
    pub u16_flags: [T; WORD_SIZE],
    /// An inverse of differing byte if b_comp != c_comp.
    pub not_eq_inv: T,
    /// The comparison limbs to be looked up.
    pub comparison_limbs: [T; 2],
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LtOperationSigned<T> {
    /// The result of the SLTU operation.
    pub result: LtOperationUnsigned<T>,
    /// The most significant bit of operand b if `is_signed` is true.
    pub b_msb: U16MSBOperation<T>,
    /// The most significant bit of operand c if `is_signed` is true.
    pub c_msb: U16MSBOperation<T>,
}

impl<F: Field> LtOperationSigned<F> {
    pub fn populate_signed(
        &mut self,
        record: &mut impl ByteRecord,
        a_u32: u32,
        b_u32: u32,
        c_u32: u32,
        is_signed: bool,
        is_real: bool,
    ) {
        let b_comp = u32_to_u16_limbs(b_u32);
        let c_comp = u32_to_u16_limbs(c_u32);
        if is_signed {
            self.b_msb.populate_msb(record, b_comp[1], is_real);
            self.c_msb.populate_msb(record, c_comp[1], is_real);
            // (a as i32) < (b as i32) if and only if (a ^ (1 << 31)) < (b ^ (1 << 31))
            self.result.populate_unsigned(
                record,
                a_u32,
                b_u32 ^ (1 << 31),
                c_u32 ^ (1 << 31),
                is_real,
            );
        } else {
            self.result.populate_unsigned(record, a_u32, b_u32, c_u32, is_real);
        }
    }

    /// Evaluate the signed LT operation.
    /// Assumes that `b`, `c` are valid `Word`s of two u16 limbs.
    /// Constrains that `is_signed` is boolean.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, constrains that the result is the signed LT of `b` and `c`.
    pub fn eval_lt_signed<AB: SP1AirBuilder>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: LtOperationSigned<AB::Var>,
        is_signed: AB::Expr,
        is_real: AB::Expr,
    ) {
        builder.assert_bool(is_signed.clone());
        builder.assert_bool(is_real.clone());
        // If `is_real` is false, assert that `is_signed` is zero.
        builder.when_not(is_real.clone()).assert_zero(is_signed.clone());

        // Constrain the MSB of `b` and `c` if `is_signed` is true.
        // This will be used to determine the sign of `b` and `c`.
        U16MSBOperation::<AB::F>::eval_msb(
            builder,
            b.0[WORD_SIZE - 1].clone(),
            cols.b_msb,
            is_signed.clone(),
        );
        U16MSBOperation::<AB::F>::eval_msb(
            builder,
            c.0[WORD_SIZE - 1].clone(),
            cols.c_msb,
            is_signed.clone(),
        );

        // Constrain `b` and `c` to be considered positive if `is_signed` is false.
        builder.when_not(is_signed.clone()).assert_zero(cols.b_msb.msb);
        builder.when_not(is_signed.clone()).assert_zero(cols.c_msb.msb);

        let mut b_compare = b;
        let mut c_compare = c;

        let base = AB::Expr::from_canonical_u32(1 << 16);

        // XOR `1 << 31` to `b` and `c` if `is_signed` is true.
        // If `is_signed` is false, the `msb` values are constrained to be zero.
        // If `is_signed` is true, the `msb` values are constrained by `U16MSBOperation`.
        // In both cases, `b_compare` and `c_compare` are correct.
        b_compare[WORD_SIZE - 1] = b_compare[WORD_SIZE - 1].clone()
            + is_signed.clone() * AB::Expr::from_canonical_u32(1 << 15)
            - base.clone() * cols.b_msb.msb;
        c_compare[WORD_SIZE - 1] = c_compare[WORD_SIZE - 1].clone()
            + is_signed.clone() * AB::Expr::from_canonical_u32(1 << 15)
            - base.clone() * cols.c_msb.msb;

        // Now apply the unsigned LT operation.
        LtOperationUnsigned::<AB::F>::eval_lt_unsigned(
            builder,
            b_compare,
            c_compare,
            cols.result,
            is_real.clone(),
        );
    }
}

impl<F: Field> LtOperationUnsigned<F> {
    pub fn populate_unsigned(
        &mut self,
        record: &mut impl ByteRecord,
        a_u32: u32,
        b_u32: u32,
        c_u32: u32,
        is_real: bool,
    ) {
        let a_limbs = u32_to_u16_limbs(a_u32);
        let b_limbs = u32_to_u16_limbs(b_u32);
        let c_limbs = u32_to_u16_limbs(c_u32);

        let a_u16 = a_limbs[0] as u16;

        let mut comparison_limbs = [0u16; 2];
        for (b_limb, c_limb, flag) in
            izip!(b_limbs.iter().rev(), c_limbs.iter().rev(), self.u16_flags.iter_mut().rev())
        {
            if b_limb != c_limb {
                *flag = F::one();
                comparison_limbs[0] = *b_limb;
                comparison_limbs[1] = *c_limb;
                let b_limb = F::from_canonical_u16(*b_limb);
                let c_limb = F::from_canonical_u16(*c_limb);
                if is_real {
                    self.not_eq_inv = (b_limb - c_limb).inverse();
                }
                self.comparison_limbs = [b_limb, c_limb];
                break;
            }
        }
        self.u16_compare_operation.populate(
            record,
            a_u16,
            comparison_limbs[0],
            comparison_limbs[1],
            is_real,
        );
    }

    /// Evaluate that LT operation.
    /// Assumes that `b`, `c` are either valid `Word`s of two u16 limbs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, constrains that the result is the LT of `b` and `c`.
    pub fn eval_lt_unsigned<AB: SP1AirBuilder>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: LtOperationUnsigned<AB::Var>,
        is_real: AB::Expr,
    ) {
        builder.assert_bool(is_real.clone());

        // Verify that the limb equality flags are set correctly, i.e. all are boolean and only
        // at most a single byte flag is set.
        let sum_flags = cols.u16_flags[0] + cols.u16_flags[1];
        builder.assert_bool(cols.u16_flags[0]);
        builder.assert_bool(cols.u16_flags[1]);
        builder.assert_bool(sum_flags.clone());

        let is_comp_eq = AB::Expr::one() - sum_flags;

        // A flag to indicate whether an equality check is necessary.
        // This is for all limbs from most significant until the first inequality.
        let mut is_inequality_visited = AB::Expr::zero();

        // Iterate over the limbs in reverse order and select the differing limbs using the limb
        // flag columns values.
        let mut b_comparison_limb = AB::Expr::zero();
        let mut c_comparison_limb = AB::Expr::zero();
        for (b_limb, c_limb, &flag) in
            izip!(b.0.iter().rev(), c.0.iter().rev(), cols.u16_flags.iter().rev())
        {
            // Once the byte flag was set to one, we turn off the equality check flag.
            // We can do this by calculating the sum of the flags since only one is set to `1`.
            is_inequality_visited = is_inequality_visited.clone() + flag.into();

            // If inequality is not visited, assert that the limbs are equal.
            builder
                .when_not(is_inequality_visited.clone())
                .assert_eq(b_limb.clone(), c_limb.clone());

            b_comparison_limb = b_comparison_limb.clone() + b_limb.clone() * flag.into();
            c_comparison_limb = c_comparison_limb.clone() + c_limb.clone() * flag.into();
        }

        let (b_comp_limb, c_comp_limb) = (cols.comparison_limbs[0], cols.comparison_limbs[1]);
        builder.assert_eq(b_comparison_limb, b_comp_limb);
        builder.assert_eq(c_comparison_limb, c_comp_limb);

        // Using the values above, we can constrain the `is_comp_eq` flag. We already asserted
        // in the loop that when `is_comp_eq == 1` then all limbs are equal. It is left to
        // verify that when `is_comp_eq == 0` the comparison limbs are indeed not equal.
        // This is done using the inverse hint `not_eq_inv`, when `is_real` is true.
        builder
            .when_not(is_comp_eq)
            .assert_eq(cols.not_eq_inv * (b_comp_limb - c_comp_limb), is_real.clone());

        // Compare the two comparison limbs.
        U16CompareOperation::<AB::F>::eval_compare_u16(
            builder,
            b_comp_limb.into(),
            c_comp_limb.into(),
            cols.u16_compare_operation,
            is_real.clone(),
        );
    }
}

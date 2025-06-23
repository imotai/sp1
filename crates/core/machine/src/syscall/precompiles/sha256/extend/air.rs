use p3_air::{Air, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_core_executor::ByteOpcode;
use sp1_stark::{
    air::{AirInteraction, InteractionScope},
    InteractionKind, Word,
};

use super::{ShaExtendChip, ShaExtendCols, NUM_SHA_EXTEND_COLS};
use crate::{
    air::SP1CoreAirBuilder,
    operations::{
        Add4Operation, ClkOperation, FixedRotateRightOperation, FixedShiftRightOperation,
        XorU32Operation,
    },
};

use core::borrow::Borrow;
use std::iter::once;

impl<F> BaseAir<F> for ShaExtendChip {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_COLS
    }
}

impl<AB> Air<AB> for ShaExtendChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Initialize columns.
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaExtendCols<AB::Var> = (*local).borrow();

        let i_start = AB::F::from_canonical_u32(16);

        ClkOperation::<AB::F>::eval(
            builder,
            local.clk_low.into(),
            local.i - i_start,
            local.next_clk,
            local.is_real.into(),
        );

        // Read w[i-15].
        builder.eval_memory_access_read(
            local.clk_high + local.next_clk.is_overflow,
            local.next_clk.next_clk_low::<AB>(),
            &local.w_i_minus_15_ptr.map(Into::into),
            local.w_i_minus_15,
            local.is_real,
        );

        // Read w[i-2].
        builder.eval_memory_access_read(
            local.clk_high + local.next_clk.is_overflow,
            local.next_clk.next_clk_low::<AB>(),
            &local.w_i_minus_2_ptr.map(Into::into),
            local.w_i_minus_2,
            local.is_real,
        );

        // Read w[i-16].
        builder.eval_memory_access_read(
            local.clk_high + local.next_clk.is_overflow,
            local.next_clk.next_clk_low::<AB>(),
            &local.w_i_minus_16_ptr.map(Into::into),
            local.w_i_minus_16,
            local.is_real,
        );

        // Read w[i-7].
        builder.eval_memory_access_read(
            local.clk_high + local.next_clk.is_overflow,
            local.next_clk.next_clk_low::<AB>(),
            &local.w_i_minus_7_ptr.map(Into::into),
            local.w_i_minus_7,
            local.is_real,
        );

        // Compute `s0`.
        // w[i-15] rightrotate 7.
        let w_i_minus_15_prev_value_half_word =
            [local.w_i_minus_15.prev_value[0], local.w_i_minus_15.prev_value[1]];
        builder.assert_zero(local.w_i_minus_15.prev_value[2]);
        builder.assert_zero(local.w_i_minus_15.prev_value[3]);

        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            w_i_minus_15_prev_value_half_word,
            7,
            local.w_i_minus_15_rr_7,
            local.is_real,
        );
        // w[i-15] rightrotate 18.
        let w_i_minus_15_prev_value_half_word =
            [local.w_i_minus_15.prev_value[0], local.w_i_minus_15.prev_value[1]];
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            w_i_minus_15_prev_value_half_word,
            18,
            local.w_i_minus_15_rr_18,
            local.is_real,
        );
        // w[i-15] rightshift 3.
        let w_i_minus_15_prev_value_half_word =
            [local.w_i_minus_15.prev_value[0], local.w_i_minus_15.prev_value[1]];
        FixedShiftRightOperation::<AB::F>::eval(
            builder,
            w_i_minus_15_prev_value_half_word,
            3,
            local.w_i_minus_15_rs_3,
            local.is_real,
        );
        // (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18)
        let s0_intermediate_result = XorU32Operation::<AB::F>::eval_xor_u32(
            builder,
            local.w_i_minus_15_rr_7.value.map(|x| x.into()),
            local.w_i_minus_15_rr_18.value.map(|x| x.into()),
            local.s0_intermediate,
            local.is_real,
        );
        // s0 := (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift 3)
        let s0_result = XorU32Operation::<AB::F>::eval_xor_u32(
            builder,
            s0_intermediate_result,
            local.w_i_minus_15_rs_3.value.map(|x| x.into()),
            local.s0,
            local.is_real,
        );

        // Compute `s1`.
        // w[i-2] rightrotate 17.
        let w_i_minus_2_prev_value_half_word =
            [local.w_i_minus_2.prev_value[0], local.w_i_minus_2.prev_value[1]];
        builder.assert_zero(local.w_i_minus_2.prev_value[2]);
        builder.assert_zero(local.w_i_minus_2.prev_value[3]);

        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            w_i_minus_2_prev_value_half_word,
            17,
            local.w_i_minus_2_rr_17,
            local.is_real,
        );
        // w[i-2] rightrotate 19.
        let w_i_minus_2_prev_value_half_word =
            [local.w_i_minus_2.prev_value[0], local.w_i_minus_2.prev_value[1]];
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            w_i_minus_2_prev_value_half_word,
            19,
            local.w_i_minus_2_rr_19,
            local.is_real,
        );
        // w[i-2] rightshift 10.
        let w_i_minus_2_prev_value_half_word =
            [local.w_i_minus_2.prev_value[0], local.w_i_minus_2.prev_value[1]];
        FixedShiftRightOperation::<AB::F>::eval(
            builder,
            w_i_minus_2_prev_value_half_word,
            10,
            local.w_i_minus_2_rs_10,
            local.is_real,
        );
        // (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19)
        let s1_intermediate_result = XorU32Operation::<AB::F>::eval_xor_u32(
            builder,
            local.w_i_minus_2_rr_17.value.map(|x| x.into()),
            local.w_i_minus_2_rr_19.value.map(|x| x.into()),
            local.s1_intermediate,
            local.is_real,
        );
        // s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)
        let s1_result = XorU32Operation::<AB::F>::eval_xor_u32(
            builder,
            s1_intermediate_result,
            local.w_i_minus_2_rs_10.value.map(|x| x.into()),
            local.s1,
            local.is_real,
        );

        // s2 := w[i-16] + s0 + w[i-7] + s1.
        let w_i_minus_16_prev_value_half_word =
            [local.w_i_minus_16.prev_value[0], local.w_i_minus_16.prev_value[1]];
        builder.assert_zero(local.w_i_minus_16.prev_value[2]);
        builder.assert_zero(local.w_i_minus_16.prev_value[3]);

        let w_i_minus_7_prev_value_half_word =
            [local.w_i_minus_7.prev_value[0], local.w_i_minus_7.prev_value[1]];
        builder.assert_zero(local.w_i_minus_7.prev_value[2]);
        builder.assert_zero(local.w_i_minus_7.prev_value[3]);

        Add4Operation::<AB::F>::eval(
            builder,
            w_i_minus_16_prev_value_half_word.map(|x| x.into()),
            s0_result,
            w_i_minus_7_prev_value_half_word.map(|x| x.into()),
            s1_result,
            local.is_real,
            local.s2,
        );

        // // Write `s2` to `w[i]`.
        let s2_value_word = Word([
            local.s2.value[0].into(),
            local.s2.value[1].into(),
            AB::Expr::zero(),
            AB::Expr::zero(),
        ]);
        builder.eval_memory_access_write(
            local.clk_high + local.next_clk.is_overflow,
            local.next_clk.next_clk_low::<AB>(),
            &local.w_i_ptr.map(Into::into),
            local.w_i,
            s2_value_word,
            local.is_real,
        );

        // Receive the state.
        let receive_values = once(local.clk_high.into())
            .chain(once(local.clk_low.into()))
            .chain(local.w_ptr.map(|x| x.into()))
            .chain(once(local.i.into()))
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::ShaExtend),
            InteractionScope::Local,
        );

        // Send the next state.
        let send_values = once(local.clk_high.into())
            .chain(once(local.clk_low.into()))
            .chain(local.w_ptr.map(|x| x.into()))
            .chain(once(local.i + AB::Expr::one()))
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::ShaExtend),
            InteractionScope::Local,
        );

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::one(),
            local.i - AB::Expr::from_canonical_u32(16),
            AB::Expr::from_canonical_u32(48),
            local.is_real,
        );

        // Assert that is_real is a bool.
        builder.assert_bool(local.is_real);
    }
}

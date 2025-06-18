use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, SP1AirBuilder},
    InteractionKind, Word,
};

use super::{
    columns::{ShaCompressCols, NUM_SHA_COMPRESS_COLS},
    ShaCompressChip, SHA_COMPRESS_K,
};
use crate::{
    air::{MemoryAirBuilder, SP1CoreAirBuilder, WordAirBuilder},
    operations::{
        Add5Operation, AddOperation, AndU16Operation, FixedRotateRightOperation, NotU16Operation,
        XorU16Operation,
    },
};
use sp1_stark::air::BaseAirBuilder;

impl<F> BaseAir<F> for ShaCompressChip {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_COLS
    }
}

impl<AB> Air<AB> for ShaCompressChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaCompressCols<AB::Var> = (*local).borrow();

        self.eval_control_flow_flags(builder, local);

        self.eval_memory(builder, local);

        self.eval_compression_ops(builder, local);

        self.eval_finalize_ops(builder, local);
    }
}

impl ShaCompressChip {
    fn eval_control_flow_flags<AB: SP1CoreAirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        // Verify that all of the octet columns are bool, and exactly one is true.
        let mut octet_sum = AB::Expr::zero();
        for i in 0..8 {
            builder.assert_bool(local.octet[i]);
            octet_sum = octet_sum.clone() + local.octet[i].into();
        }
        builder.assert_one(octet_sum);

        // Verify that all of the octet_num columns are bool, and exactly one is true.
        let mut octet_num_sum = AB::Expr::zero();
        for i in 0..10 {
            builder.assert_bool(local.octet_num[i]);
            octet_num_sum = octet_num_sum.clone() + local.octet_num[i].into();
        }
        builder.assert_one(octet_num_sum);

        // Assert that the is_initialize flag is correct.
        builder.assert_eq(local.is_initialize, local.octet_num[0] * local.is_real);

        // Assert that the is_compression flag is correct.
        builder.assert_eq(
            local.is_compression,
            (local.octet_num[1]
                + local.octet_num[2]
                + local.octet_num[3]
                + local.octet_num[4]
                + local.octet_num[5]
                + local.octet_num[6]
                + local.octet_num[7]
                + local.octet_num[8])
                * local.is_real,
        );

        // Assert that the is_finalize flag is correct.
        builder.assert_eq(local.is_finalize, local.octet_num[9] * local.is_real);

        // Receive state.
        builder.receive(
            AirInteraction::new(
                vec![local.clk_high, local.clk_low, local.w_ptr, local.h_ptr, local.index]
                    .into_iter()
                    .chain(
                        [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h]
                            .into_iter()
                            .flat_map(|word| word.into_iter()),
                    )
                    .map(Into::into)
                    .collect(),
                local.is_real.into(),
                InteractionKind::ShaCompress,
            ),
            InteractionScope::Local,
        );

        // Send state, for initialize and finalize.
        builder.send(
            AirInteraction::new(
                vec![
                    local.clk_high.into(),
                    local.clk_low.into(),
                    local.w_ptr.into(),
                    local.h_ptr.into(),
                    local.index.into() + AB::Expr::one(),
                ]
                .into_iter()
                .chain(
                    [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h]
                        .into_iter()
                        .flat_map(|word| word.into_iter())
                        .map(Into::into),
                )
                .collect(),
                local.is_initialize + local.is_finalize,
                InteractionKind::ShaCompress,
            ),
            InteractionScope::Local,
        );

        // Send state, for compression.
        // h := g
        // g := f
        // f := e
        // e := d + temp1
        // d := c
        // c := b
        // b := a
        // a := temp1 + temp2
        builder.send(
            AirInteraction::new(
                vec![
                    local.clk_high.into(),
                    local.clk_low.into(),
                    local.w_ptr.into(),
                    local.h_ptr.into(),
                    local.index.into() + AB::Expr::one(),
                ]
                .into_iter()
                .chain(
                    [
                        local.temp1_add_temp2.value,
                        local.a,
                        local.b,
                        local.c,
                        local.d_add_temp1.value,
                        local.e,
                        local.f,
                        local.g,
                    ]
                    .into_iter()
                    .flat_map(|word| word.into_iter())
                    .map(Into::into),
                )
                .collect(),
                local.is_compression.into(),
                InteractionKind::ShaCompress,
            ),
            InteractionScope::Local,
        );

        // Assert that is_real is a bool.
        builder.assert_bool(local.is_real);
    }

    /// Constrains that memory address is correct and that memory is correctly written/read.
    fn eval_memory<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &ShaCompressCols<AB::Var>) {
        builder.eval_memory_access_write(
            local.clk_high,
            local.clk_low + local.is_finalize,
            local.mem_addr,
            local.mem,
            local.mem_value,
            local.is_initialize + local.is_compression + local.is_finalize,
        );

        // Calculate the current cycle_num.
        let mut cycle_num = AB::Expr::zero();
        for i in 0..10 {
            cycle_num = cycle_num.clone() + local.octet_num[i] * AB::Expr::from_canonical_usize(i);
        }

        // Calculate the current step of the cycle 8.
        let mut cycle_step = AB::Expr::zero();
        for i in 0..8 {
            cycle_step = cycle_step.clone() + local.octet[i] * AB::Expr::from_canonical_usize(i);
        }

        // Check the index is correct.
        builder.assert_eq(
            local.index,
            cycle_step.clone() + cycle_num.clone() * AB::Expr::from_canonical_u32(8),
        );

        // Verify correct mem address for initialize phase
        builder.when(local.is_initialize).assert_eq(
            local.mem_addr,
            local.h_ptr + cycle_step.clone() * AB::Expr::from_canonical_u32(4),
        );

        // Verify correct mem address for compression phase
        builder.when(local.is_compression).assert_eq(
            local.mem_addr,
            local.w_ptr
                + (((cycle_num - AB::Expr::one()) * AB::Expr::from_canonical_u32(8))
                    + cycle_step.clone())
                    * AB::Expr::from_canonical_u32(4),
        );

        // Verify correct mem address for finalize phase
        builder.when(local.is_finalize).assert_eq(
            local.mem_addr,
            local.h_ptr + cycle_step.clone() * AB::Expr::from_canonical_u32(4),
        );

        // In the initialize phase, verify that local.a, local.b, ... is correctly read from memory
        // and does not change
        let vars = [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h];
        for (i, var) in vars.iter().enumerate() {
            builder
                .when(local.is_initialize * local.octet[i])
                .assert_word_eq(*var, local.mem.prev_value);
            builder
                .when(local.is_initialize * local.octet[i])
                .assert_word_eq(*var, local.mem_value);
        }

        // During initialize and compression, verify that memory is read only and does not change.
        builder
            .when(local.is_initialize + local.is_compression)
            .assert_word_eq(local.mem.prev_value, local.mem_value);

        // In the finalize phase, verify that the correct value is written to memory.
        builder.when(local.is_finalize).assert_word_eq(local.mem_value, local.finalize_add.value);
    }

    fn eval_compression_ops<AB: SP1CoreAirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        // Constrain k column which loops over 64 constant values.
        for i in 0..64 {
            let octet_num = i / 8;
            let inner_index = i % 8;
            builder
                .when(local.octet_num[octet_num + 1] * local.octet[inner_index])
                .assert_all_eq(local.k, Word::<AB::F>::from(SHA_COMPRESS_K[i]));
        }

        // S1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25).
        // Calculate e rightrotate 6.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.e,
            6,
            local.e_rr_6,
            local.is_compression,
        );
        // Calculate e rightrotate 11.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.e,
            11,
            local.e_rr_11,
            local.is_compression,
        );
        // Calculate e rightrotate 25.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.e,
            25,
            local.e_rr_25,
            local.is_compression,
        );
        // Calculate (e rightrotate 6) xor (e rightrotate 11).
        let s1_intermediate = XorU16Operation::<AB::F>::eval_xor_u16(
            builder,
            local.e_rr_6.value.map(|x| x.into()),
            local.e_rr_11.value.map(|x| x.into()),
            local.s1_intermediate,
            local.is_compression,
        );
        // Calculate S1 := ((e rightrotate 6) xor (e rightrotate 11)) xor (e rightrotate 25).
        let s1 = XorU16Operation::<AB::F>::eval_xor_u16(
            builder,
            s1_intermediate,
            local.e_rr_25.value.map(|x| x.into()),
            local.s1,
            local.is_compression,
        );

        // Calculate ch := (e and f) xor ((not e) and g).
        // Calculate e and f.
        let e_and_f = AndU16Operation::<AB::F>::eval_and_u16(
            builder,
            local.e.map(|x| x.into()),
            local.f.map(|x| x.into()),
            local.e_and_f,
            local.is_compression,
        );
        // Calculate not e.
        NotU16Operation::<AB::F>::eval(
            builder,
            local.e.map(|x| x.into()),
            local.e_not,
            local.is_compression,
        );
        // Calculate (not e) and g.
        let e_not_and_g = AndU16Operation::<AB::F>::eval_and_u16(
            builder,
            local.e_not.value.map(|x| x.into()),
            local.g.map(|x| x.into()),
            local.e_not_and_g,
            local.is_compression,
        );
        // Calculate ch := (e and f) xor ((not e) and g).
        let ch = XorU16Operation::<AB::F>::eval_xor_u16(
            builder,
            e_and_f,
            e_not_and_g,
            local.ch,
            local.is_compression,
        );

        // Calculate temp1 := h + S1 + ch + k[i] + w[i].
        Add5Operation::<AB::F>::eval(
            builder,
            &[
                local.h.map(|x| x.into()),
                s1,
                ch,
                local.k.map(|x| x.into()),
                local.mem_value.map(|x| x.into()),
            ],
            local.is_compression,
            local.temp1,
        );

        // Calculate S0 := (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22).
        // Calculate a rightrotate 2.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.a,
            2,
            local.a_rr_2,
            local.is_compression,
        );
        // Calculate a rightrotate 13.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.a,
            13,
            local.a_rr_13,
            local.is_compression,
        );
        // Calculate a rightrotate 22.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.a,
            22,
            local.a_rr_22,
            local.is_compression,
        );
        // Calculate (a rightrotate 2) xor (a rightrotate 13).
        let s0_intermediate = XorU16Operation::<AB::F>::eval_xor_u16(
            builder,
            local.a_rr_2.value.map(|x| x.into()),
            local.a_rr_13.value.map(|x| x.into()),
            local.s0_intermediate,
            local.is_compression,
        );
        // Calculate S0 := ((a rightrotate 2) xor (a rightrotate 13)) xor (a rightrotate 22).
        let s0 = XorU16Operation::<AB::F>::eval_xor_u16(
            builder,
            s0_intermediate,
            local.a_rr_22.value.map(|x| x.into()),
            local.s0,
            local.is_compression,
        );

        // Calculate maj := (a and b) xor (a and c) xor (b and c).
        // Calculate a and b.
        let a_and_b = AndU16Operation::<AB::F>::eval_and_u16(
            builder,
            local.a.map(|x| x.into()),
            local.b.map(|x| x.into()),
            local.a_and_b,
            local.is_compression,
        );
        // Calculate a and c.
        let a_and_c = AndU16Operation::<AB::F>::eval_and_u16(
            builder,
            local.a.map(|x| x.into()),
            local.c.map(|x| x.into()),
            local.a_and_c,
            local.is_compression,
        );
        // Calculate b and c.
        let b_and_c = AndU16Operation::<AB::F>::eval_and_u16(
            builder,
            local.b.map(|x| x.into()),
            local.c.map(|x| x.into()),
            local.b_and_c,
            local.is_compression,
        );
        // Calculate (a and b) xor (a and c).
        let maj_intermediate = XorU16Operation::<AB::F>::eval_xor_u16(
            builder,
            a_and_b,
            a_and_c,
            local.maj_intermediate,
            local.is_compression,
        );
        // Calculate maj := ((a and b) xor (a and c)) xor (b and c).
        let maj = XorU16Operation::<AB::F>::eval_xor_u16(
            builder,
            maj_intermediate,
            b_and_c,
            local.maj,
            local.is_compression,
        );

        // Calculate temp2 := s0 + maj.
        AddOperation::<AB::F>::eval(builder, s0, maj, local.temp2, local.is_compression.into());

        // Calculate d + temp1 for the new value of e.
        AddOperation::<AB::F>::eval(
            builder,
            local.d.map(|x| x.into()),
            local.temp1.value.map(|x| x.into()),
            local.d_add_temp1,
            local.is_compression.into(),
        );

        // Calculate temp1 + temp2 for the new value of a.
        AddOperation::<AB::F>::eval(
            builder,
            local.temp1.value.map(|x| x.into()),
            local.temp2.value.map(|x| x.into()),
            local.temp1_add_temp2,
            local.is_compression.into(),
        );
    }

    fn eval_finalize_ops<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        // In the finalize phase, need to execute h[0] + a, h[1] + b, ..., h[7] + h, for each of the
        // phase's 8 rows.
        // We can get the needed operand (a,b,c,...,h) by doing an inner product between octet and
        // [a,b,c,...,h] which will act as a selector.
        let add_operands = [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h];
        let zero = AB::Expr::zero();
        let mut filtered_operand = Word([zero.clone(), zero.clone()]);
        for (i, operand) in local.octet.iter().zip(add_operands.iter()) {
            for j in 0..WORD_SIZE {
                filtered_operand.0[j] = filtered_operand.0[j].clone() + *i * operand.0[j];
            }
        }

        builder
            .when(local.is_finalize)
            .assert_word_eq(filtered_operand, local.finalized_operand.map(|x| x.into()));

        // finalize_add.result = h[i] + finalized_operand
        AddOperation::<AB::F>::eval(
            builder,
            local.mem.prev_value.map(|x| x.into()),
            local.finalized_operand.map(|x| x.into()),
            local.finalize_add,
            local.is_finalize.into(),
        );

        // Memory write is constrained in constrain_memory.
    }
}

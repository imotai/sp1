use crate::adapter::{register::jalr_type::JalrTypeReader, state::CPUState};
use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_core_executor::{Opcode, PC_INC};
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::air::{BaseAirBuilder, SP1AirBuilder};
use std::borrow::Borrow;

use crate::operations::{AddOperation, BabyBearWordRangeChecker};

use super::{JalrChip, JalrColumns};

impl<AB> Air<AB> for JalrChip
where
    AB: SP1AirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &JalrColumns<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let opcode = Opcode::JALR.as_field::<AB::F>();

        // We constrain `next_pc_rel` to be the sum of `op_b` and `op_c`.
        AddOperation::<AB::F>::eval(
            builder,
            local.adapter.b().map(|x| x.into()),
            local.adapter.c().map(|x| x.into()),
            local.add_operation,
            local.is_real.into(),
        );

        let next_pc_rel = local.add_operation.value;

        // Constrain the state of the CPU.
        // The `next_pc_rel` is constrained by the AIR.
        // The clock is incremented by `4`.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            next_pc_rel.reduce::<AB>(),
            AB::Expr::from_canonical_u32(PC_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        JalrTypeReader::<AB::F>::eval(
            builder,
            local.state.shard::<AB>(),
            local.state.clk::<AB>(),
            local.state.pc_rel,
            opcode,
            local.op_a_value,
            local.adapter,
            local.is_real.into(),
        );

        // Verify that pc_abs + 4 is saved in op_a. We retrieve the value from the upper 3 limbs of
        // op_b, where it has been preprocessed. The following assertions also verify that
        // `local.op_a_value` is a well-formed word, since `local.adapter.op_b[1..=3]` and `0`
        // represent u16s.
        // When op_a is set to register X0, the RISC-V spec states that the jump instruction will
        // not have a return destination address (it is effectively a GOTO command).  In this case,
        // we shouldn't verify the return address.
        // If `op_a_0` is set, the `op_a_value` will be constrained to be zero.
        let mut builder_cond = builder.when(local.is_real);
        let mut builder_cond = builder_cond.when_not(local.adapter.op_a_0);
        for i in 0..(WORD_SIZE - 1) {
            builder_cond.assert_eq(local.op_a_value[i], local.adapter.op_b[i + 1]);
        }
        builder_cond.assert_zero(local.op_a_value[WORD_SIZE - 1]);

        // SAFETY: `is_real` is already checked to be boolean.
        // `next_pc_rel` is checked to a valid word.
        // This is due to the ADDOperation checking outputs are valid words.
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            next_pc_rel,
            local.next_pc_range_checker,
            local.is_real.into(),
        );
    }
}

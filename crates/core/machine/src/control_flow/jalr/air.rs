use crate::adapter::{register::i_type::ITypeReader, state::CPUState};
use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_core_executor::{Opcode, DEFAULT_PC_INC};
use sp1_stark::air::{BaseAirBuilder, SP1AirBuilder};
use std::borrow::Borrow;

use crate::{
    air::WordAirBuilder,
    operations::{AddOperation, BabyBearWordRangeChecker},
};

use super::{JalrChip, JalrColumns};

impl<AB> Air<AB> for JalrChip
where
    AB: SP1AirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        // let main = builder.main();
        // let local = main.row_slice(0);
        // let local: &JalrColumns<AB::Var> = (*local).borrow();

        // builder.assert_bool(local.is_real);

        // let opcode = Opcode::JALR.as_field::<AB::F>();

        // // We constrain `next_pc` to be the sum of `op_b` and `op_c`.
        // AddOperation::<AB::F>::eval(
        //     builder,
        //     local.adapter.b().map(|x| x.into()),
        //     local.adapter.c().map(|x| x.into()),
        //     local.add_operation,
        //     local.is_real.into(),
        // );

        // let next_pc = local.add_operation.value;

        // // Constrain the state of the CPU.
        // // The `next_pc` is constrained by the AIR.
        // // The clock is incremented by `4`.
        // CPUState::<AB::F>::eval(
        //     builder,
        //     local.state,
        //     next_pc.reduce::<AB>(),
        //     AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
        //     local.is_real.into(),
        // );

        // // Constrain the program and register reads.
        // ITypeReader::<AB::F>::eval(
        //     builder,
        //     local.state.shard::<AB>(),
        //     local.state.clk::<AB>(),
        //     local.state.pc,
        //     opcode,
        //     local.op_a_value,
        //     local.adapter,
        //     local.is_real.into(),
        // );

        // // Verify that the local.pc + 4 is saved in op_a for both jump instructions.
        // // When op_a is set to register X0, the RISC-V spec states that the jump instruction will
        // // not have a return destination address (it is effectively a GOTO command).  In this case,
        // // we shouldn't verify the return address.
        // // If `op_a_0` is set, the `op_a_value` will be constrained to be zero.
        // builder.when(local.is_real).when_not(local.adapter.op_a_0).assert_eq(
        //     local.op_a_value.reduce::<AB>(),
        //     local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
        // );

        // // Constrain the op_a value to be a valid word.
        // builder.slice_range_check_u16(&local.op_a_value.0, local.is_real);

        // // Constrain the op_a word to represent the BabyBear value canonically.
        // BabyBearWordRangeChecker::<AB::F>::range_check(
        //     builder,
        //     local.op_a_value,
        //     local.op_a_range_checker,
        //     local.is_real.into(),
        // );

        // // SAFETY: `is_real` is already checked to be boolean.
        // // `next_pc` is checked to a valid word.
        // // This is due to the ADDOperation checking outputs are valid words.
        // BabyBearWordRangeChecker::<AB::F>::range_check(
        //     builder,
        //     next_pc,
        //     local.next_pc_range_checker,
        //     local.is_real.into(),
        // );
    }
}

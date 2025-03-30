use std::borrow::Borrow;

use crate::adapter::register::j_type::JTypeReader;
use crate::adapter::state::CPUState;
use p3_air::Air;
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_core_executor::{Opcode, DEFAULT_PC_INC};
use sp1_stark::air::SP1AirBuilder;

use super::{JalChip, JalColumns};

impl<AB> Air<AB> for JalChip
where
    AB: SP1AirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &JalColumns<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let opcode = Opcode::JAL.as_field::<AB::F>();

        // Constrain the state of the CPU.
        // The `next_pc` is constrained by the AIR.
        // The clock is incremented by `4`.
        // Set `op_b` immediate as `pc + op_b` value in the instruction encoding.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.adapter.b().reduce::<AB>(),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        // Verify that the local.pc + 4 is saved in op_a for both jump instructions.
        // When op_a is set to register X0, the RISC-V spec states that the jump instruction will
        // not have a return destination address (it is effectively a GOTO command).  In this case,
        // we shouldn't verify the return address.
        // Set `op_c` immediate as `op_a_not_0 * (pc + 4)` in the instruction encoding.
        JTypeReader::<AB::F>::eval(
            builder,
            local.state.shard::<AB>(),
            local.state.clk::<AB>(),
            local.state.pc,
            opcode,
            *local.adapter.c(),
            local.adapter,
            local.is_real.into(),
        );
    }
}

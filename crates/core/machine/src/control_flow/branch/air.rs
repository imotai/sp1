use std::borrow::Borrow;

use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_core_executor::{Opcode, DEFAULT_CLK_INC};

use crate::{
    adapter::{register::i_type::ITypeReader, state::CPUState},
    air::{SP1CoreAirBuilder, SP1Operation},
    operations::{LtOperationSigned, LtOperationSignedInput},
};

use super::{BranchChip, BranchColumns};

/// Verifies all the branching related columns.
///
/// It does this in few parts:
/// 1. It verifies that the next pc is correct based on the branching column.  That column is a
///    boolean that indicates whether the branch condition is true.
/// 2. It verifies the correct value of branching based on the opcode and the comparison operation.
impl<AB> Air<AB> for BranchChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BranchColumns<AB::Var> = (*local).borrow();

        // SAFETY: All selectors `is_beq`, `is_bne`, `is_blt`, `is_bge`, `is_bltu`, `is_bgeu` are
        // checked to be boolean. Each "real" row has exactly one selector turned on, as
        // `is_real`, the sum of the six selectors, is boolean. Therefore, the `opcode`
        // matches the corresponding opcode.
        builder.assert_bool(local.is_beq);
        builder.assert_bool(local.is_bne);
        builder.assert_bool(local.is_blt);
        builder.assert_bool(local.is_bge);
        builder.assert_bool(local.is_bltu);
        builder.assert_bool(local.is_bgeu);
        let is_real = local.is_beq
            + local.is_bne
            + local.is_blt
            + local.is_bge
            + local.is_bltu
            + local.is_bgeu;
        builder.assert_bool(is_real.clone());

        let opcode = local.is_beq * Opcode::BEQ.as_field::<AB::F>()
            + local.is_bne * Opcode::BNE.as_field::<AB::F>()
            + local.is_blt * Opcode::BLT.as_field::<AB::F>()
            + local.is_bge * Opcode::BGE.as_field::<AB::F>()
            + local.is_bltu * Opcode::BLTU.as_field::<AB::F>()
            + local.is_bgeu * Opcode::BGEU.as_field::<AB::F>();

        // Constrain the state of the CPU.
        // The `next_pc` is constrained by the AIR.
        // The clock is incremented by `4`.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.next_pc.into(),
            AB::Expr::from_canonical_u32(DEFAULT_CLK_INC),
            is_real.clone(),
        );

        // Constrain the program and register reads.
        ITypeReader::<AB::F>::eval_op_a_immutable(
            builder,
            local.state.clk_high::<AB>(),
            local.state.clk_low::<AB>(),
            local.state.pc,
            opcode,
            local.adapter,
            is_real.clone(),
        );

        // SAFETY: `use_signed_comparison` is boolean, since at most one selector is turned on.
        let use_signed_comparison = local.is_blt + local.is_bge;
        <LtOperationSigned<AB::F> as SP1Operation<AB>>::eval(
            builder,
            LtOperationSignedInput::<AB>::new(
                local.adapter.prev_a().map(Into::into),
                local.adapter.b().map(Into::into),
                local.compare_operation,
                use_signed_comparison.clone(),
                is_real.clone(),
            ),
        );

        // From the `LtOperationSigned`, derive whether `a == b`, `a < b`, or `a > b`.
        let is_eq = AB::Expr::one()
            - (local.compare_operation.result.u16_flags[0]
                + local.compare_operation.result.u16_flags[1]);
        let is_less_than = local.compare_operation.result.u16_compare_operation.bit;

        // Constrain the branching column with the comparison results and opcode flags.
        let mut branching: AB::Expr = AB::Expr::zero();
        branching = branching.clone() + local.is_beq * is_eq.clone();
        branching = branching.clone() + local.is_bne * (AB::Expr::one() - is_eq);
        branching =
            branching.clone() + (local.is_bge + local.is_bgeu) * (AB::Expr::one() - is_less_than);
        branching = branching.clone() + (local.is_blt + local.is_bltu) * is_less_than;

        builder.when(is_real.clone()).assert_eq(local.is_branching, branching.clone());

        // Constrain the next_pc using the branching column.
        // Set `op_c` immediate as `pc + op_c` value in the instruction encoding.
        let mut next_pc: AB::Expr = AB::Expr::zero();
        next_pc = next_pc.clone() + local.is_branching * local.adapter.c().reduce::<AB>();
        next_pc = next_pc.clone()
            + (AB::Expr::one() - local.is_branching)
                * (local.state.pc + AB::Expr::from_canonical_u16(4));

        builder.when(is_real.clone()).assert_eq(local.next_pc, next_pc);
    }
}

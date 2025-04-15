use p3_field::{AbstractField, Field, PrimeField32};
use sp1_core_executor::{
    events::{ByteRecord, MemoryAccessPosition},
    ITypeRecord, Instruction,
};
use sp1_derive::AlignedBorrow;

use sp1_stark::{air::SP1AirBuilder, Word};

use crate::{
    air::{SP1CoreAirBuilder, WordAirBuilder},
    cpu::columns::InstructionCols,
    memory::MemoryAccessCols,
};

/// A set of columns to read operations with op_a and op_b being registers and op_c being an
/// immediate.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ITypeReader<T> {
    pub op_a: T,
    pub op_a_memory: MemoryAccessCols<T>,
    pub op_a_0: T,
    pub op_b: T,
    pub op_b_memory: MemoryAccessCols<T>,
    pub op_c_imm: Word<T>,
}

impl<F: PrimeField32> ITypeReader<F> {
    pub fn populate(
        &mut self,
        blu_events: &mut impl ByteRecord,
        instruction: &Instruction,
        record: ITypeRecord,
    ) {
        self.op_a = F::from_canonical_u8(instruction.op_a);
        self.op_a_memory.populate(record.a, blu_events);
        self.op_a_0 = F::from_bool(instruction.op_a == 0);
        self.op_b = F::from_canonical_u32(instruction.op_b);
        self.op_b_memory.populate(record.b, blu_events);
        self.op_c_imm = Word::from(instruction.op_c);
    }
}

impl<T> ITypeReader<T> {
    pub fn prev_a(&self) -> &Word<T> {
        &self.op_a_memory.prev_value
    }

    pub fn b(&self) -> &Word<T> {
        &self.op_b_memory.prev_value
    }

    pub fn c(&self) -> &Word<T> {
        &self.op_c_imm
    }
}

impl<F: Field> ITypeReader<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: SP1CoreAirBuilder>(
        builder: &mut AB,
        shard: impl Into<AB::Expr> + Clone,
        clk: AB::Expr,
        pc: AB::Var,
        opcode: impl Into<AB::Expr>,
        op_a_write_value: Word<impl Into<AB::Expr> + Clone>,
        cols: ITypeReader<AB::Var>,
        is_real: AB::Expr,
    ) {
        builder.assert_bool(is_real.clone());
        let instruction = InstructionCols {
            opcode: opcode.into(),
            op_a: cols.op_a.into(),
            op_b: Word::extend_expr::<AB>(cols.op_b.into()),
            op_c: cols.op_c_imm.map(Into::into),
            op_a_0: cols.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(pc, instruction, is_real.clone());
        // Assert that `op_a` is zero if `op_a_0` is true.
        builder.when(cols.op_a_0).assert_word_eq(op_a_write_value.clone(), Word::zero::<AB>());
        builder.eval_memory_access_write(
            shard.clone(),
            clk.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            cols.op_a,
            cols.op_a_memory,
            op_a_write_value,
            is_real.clone(),
        );
        builder.eval_memory_access_read(
            shard.clone(),
            clk.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            cols.op_b,
            cols.op_b_memory,
            is_real,
        );
    }

    pub fn eval_op_a_immutable<AB: SP1AirBuilder>(
        builder: &mut AB,
        shard: impl Into<AB::Expr> + Clone,
        clk: AB::Expr,
        pc: AB::Var,
        opcode: impl Into<AB::Expr>,
        cols: ITypeReader<AB::Var>,
        is_real: AB::Expr,
    ) {
        Self::eval(builder, shard, clk, pc, opcode, cols.op_a_memory.prev_value, cols, is_real);
    }
}

use p3_air::AirBuilder;
use p3_field::{AbstractField, Field, PrimeField32};
use sp1_core_executor::{
    events::{ByteRecord, MemoryAccessPosition},
    ALUTypeRecord, Instruction,
};
use sp1_derive::AlignedBorrow;

use sp1_stark::{air::SP1AirBuilder, Word};

use crate::{
    air::{SP1CoreAirBuilder, WordAirBuilder},
    cpu::columns::InstructionCols,
    memory::MemoryAccessInShardCols,
};

/// A set of columns to read operations with op_a and op_b being registers and op_c being a register
/// or immediate.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ALUTypeReader<T> {
    pub op_a: T,
    pub op_a_memory: MemoryAccessInShardCols<T>,
    pub op_a_0: T,
    pub op_b: T,
    pub op_b_memory: MemoryAccessInShardCols<T>,
    pub op_c: Word<T>,
    pub op_c_memory: MemoryAccessInShardCols<T>,
    pub imm_c: T,
}

impl<T> ALUTypeReader<T> {
    pub fn prev_a(&self) -> &Word<T> {
        &self.op_a_memory.prev_value
    }

    pub fn b(&self) -> &Word<T> {
        &self.op_b_memory.prev_value
    }

    pub fn c(&self) -> &Word<T> {
        &self.op_c_memory.prev_value
    }
}

impl<F: PrimeField32> ALUTypeReader<F> {
    pub fn populate(
        &mut self,
        blu_events: &mut impl ByteRecord,
        instruction: &Instruction,
        record: ALUTypeRecord,
    ) {
        self.op_a = F::from_canonical_u8(instruction.op_a);
        self.op_a_memory.populate(record.a, blu_events);
        self.op_a_0 = F::from_bool(instruction.op_a == 0);
        self.op_b = F::from_canonical_u32(instruction.op_b);
        self.op_b_memory.populate(record.b, blu_events);
        self.op_c = Word::from(instruction.op_c);
        let imm_c = record.c.is_none();
        self.imm_c = F::from_bool(imm_c);
        if imm_c {
            self.op_c_memory.prev_value = self.op_c;
        } else {
            self.op_c_memory.populate(record.c.unwrap(), blu_events);
        }
    }
}

impl<F: Field> ALUTypeReader<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: SP1CoreAirBuilder>(
        builder: &mut AB,
        clk_high: AB::Expr,
        clk_low: AB::Expr,
        pc: AB::Var,
        opcode: impl Into<AB::Expr>,
        op_a_write_value: Word<impl Into<AB::Expr> + Clone>,
        cols: ALUTypeReader<AB::Var>,
        is_real: AB::Expr,
    ) {
        builder.assert_bool(is_real.clone());
        // Assert that `imm_c` is zero if the operation is not real.
        // This is to ensure that the `op_c` read multiplicity is zero on padding rows.
        builder.when_not(is_real.clone()).assert_eq(cols.imm_c, AB::Expr::zero());
        let instruction = InstructionCols {
            opcode: opcode.into(),
            op_a: cols.op_a.into(),
            op_b: Word::extend_expr::<AB>(cols.op_b.into()),
            op_c: cols.op_c.map(Into::into),
            op_a_0: cols.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: cols.imm_c.into(),
        };
        builder.send_program(pc, instruction, is_real.clone());
        // Assert that `op_a` is zero if `op_a_0` is true.
        builder.when(cols.op_a_0).assert_word_eq(op_a_write_value.clone(), Word::zero::<AB>());
        builder.eval_memory_access_in_shard_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            cols.op_a,
            cols.op_a_memory,
            op_a_write_value,
            is_real.clone(),
        );
        builder.eval_memory_access_in_shard_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            cols.op_b,
            cols.op_b_memory,
            is_real.clone(),
        );
        // Read the `op_c[0]` register only when `imm_c` is zero and `is_real` is true.
        builder.eval_memory_access_in_shard_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
            cols.op_c[0],
            cols.op_c_memory,
            is_real - cols.imm_c,
        );
        // If `op_c` is an immediate, assert that `op_c` value is copied into
        // `op_c_memory.prev_value`.
        builder.when(cols.imm_c).assert_word_eq(cols.op_c_memory.prev_value, cols.op_c);
    }

    pub fn eval_op_a_immutable<AB: SP1AirBuilder>(
        builder: &mut AB,
        clk_high: AB::Expr,
        clk_low: AB::Expr,
        pc: AB::Var,
        opcode: impl Into<AB::Expr>,
        cols: ALUTypeReader<AB::Var>,
        is_real: AB::Expr,
    ) {
        Self::eval(
            builder,
            clk_high,
            clk_low,
            pc,
            opcode,
            cols.op_a_memory.prev_value,
            cols,
            is_real,
        );
    }
}

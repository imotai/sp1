use serde::{Deserialize, Serialize};

use crate::Opcode;

use super::MemoryRecordEnum;

/// Alu Instruction Event.
///
/// This object encapsulated the information needed to prove a RISC-V ALU operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct AluEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc_rel: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u64,
    /// The second operand value.
    pub b: u64,
    /// The third operand value.
    pub c: u64,
    /// Whether the first operand is register 0.
    pub op_a_0: bool,
}

impl AluEvent {
    /// Create a new [`AluEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc_rel: u32,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        op_a_0: bool,
    ) -> Self {
        Self { clk, pc_rel, opcode, a, b, c, op_a_0 }
    }
}

/// Memory Instruction Event.
///
/// This object encapsulated the information needed to prove a RISC-V memory operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct MemInstrEvent {
    /// The shard.
    pub shard: u32,
    /// The clk.
    pub clk: u64,
    /// The program counter.
    pub pc_rel: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u64,
    /// The second operand value.
    pub b: u64,
    /// The third operand value.
    pub c: u64,
    /// Whether the first operand is register 0.
    pub op_a_0: bool,
    /// The memory access record for memory operations.
    pub mem_access: MemoryRecordEnum,
}

impl MemInstrEvent {
    /// Create a new [`MemInstrEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shard: u32,
        clk: u64,
        pc_rel: u32,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        op_a_0: bool,
        mem_access: MemoryRecordEnum,
    ) -> Self {
        Self { shard, clk, pc_rel, opcode, a, b, c, op_a_0, mem_access }
    }
}

/// Branch Instruction Event.
///
/// This object encapsulated the information needed to prove a RISC-V branch operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct BranchEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc_rel: u32,
    /// The next program counter.
    pub next_pc_rel: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u64,
    /// The second operand value.
    pub b: u64,
    /// The third operand value.
    pub c: u64,
    /// Whether the first operand is register 0.
    pub op_a_0: bool,
}

impl BranchEvent {
    /// Create a new [`BranchEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc_rel: u32,
        next_pc_rel: u32,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        op_a_0: bool,
    ) -> Self {
        Self { clk, pc_rel, next_pc_rel, opcode, a, b, c, op_a_0 }
    }
}

/// Jump Instruction Event.
///
/// This object encapsulated the information needed to prove a RISC-V jump operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct JumpEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc_rel: u32,
    /// The next program counter.
    pub next_pc_rel: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u64,
    /// The second operand value.
    pub b: u64,
    /// The third operand value.
    pub c: u64,
    /// Whether the first operand is register 0.
    pub op_a_0: bool,
}

impl JumpEvent {
    /// Create a new [`JumpEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc_rel: u32,
        next_pc_rel: u32,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        op_a_0: bool,
    ) -> Self {
        Self { clk, pc_rel, next_pc_rel, opcode, a, b, c, op_a_0 }
    }
}
/// AUIPC Instruction Event.
///
/// This object encapsulated the information needed to prove a RISC-V AUIPC operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct AUIPCEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc_rel: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u64,
    /// The second operand value.
    pub b: u64,
    /// The third operand value.
    pub c: u64,
    /// Whether the first operand is register 0.
    pub op_a_0: bool,
}

impl AUIPCEvent {
    /// Create a new [`AUIPCEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc_rel: u32,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        op_a_0: bool,
    ) -> Self {
        Self { clk, pc_rel, opcode, a, b, c, op_a_0 }
    }
}

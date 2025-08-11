#![allow(clippy::fn_to_numeric_cast)]

use super::{TranspilerBackend, CONTEXT, JUMP_TABLE, PC_OFFSET, TEMP_A, TEMP_B};
use crate::{DebugFn, EcallHandler, JitContext, RiscOperand, RiscRegister, SP1RiscvTranspiler};
use dynasmrt::{
    dynasm,
    x64::{Assembler, Rq},
    DynasmApi, DynasmLabelApi,
};
use std::mem::offset_of;

impl SP1RiscvTranspiler for TranspilerBackend {
    fn new(
        program_size: usize,
        memory_size: usize,
        trace_buf_size: usize,
        pc_start: u32,
        pc_base: u32,
    ) -> Result<Self, std::io::Error> {
        let inner = Assembler::new()?;

        let mut this = Self {
            inner,
            jump_table: Vec::with_capacity(program_size),
            memory_size,
            trace_buf_size,
            has_instructions: false,
            pc_base,
            pc_start,
            // Register a dummy ecall handler.
            ecall_handler: super::ecallk as _,
        };

        // Handle calling conventions and save anything were gonna clobber.
        this.prologue();

        Ok(this)
    }

    fn register_ecall_handler(&mut self, handler: EcallHandler) {
        self.ecall_handler = handler;
    }

    fn start_instr(&mut self) {
        // We dont want to compile without a single jumpdest, otherwise we will sigsegv.
        self.has_instructions = true;

        let offset = self.inner.offset();
        self.jump_table.push(offset.0);
    }

    fn jal(&mut self, rd: RiscRegister, imm: u32) {
        // Store the current pc + 4 into
        self.load_pc_into_register(TEMP_A);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 1. Add 4 to the current PC in temp_a
            // ------------------------------------
            add Rd(TEMP_A), 4
        }

        // Store the current PC + 4 into the destination register.
        self.emit_risc_register_store(TEMP_A, rd);

        // Adjust the PC store in the context by the immediate.
        self.bump_pc(imm);

        // Jump the ASM code corresponding to the PC.
        self.jump_to_pc();
    }

    fn jalr(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Compute the PC to store into rd.
        // ------------------------------------
        self.load_pc_into_register(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            add Rd(TEMP_B), 4
        }

        // ------------------------------------
        // 2. Load rs1, add imm, and store it as PC
        // ------------------------------------

        self.emit_risc_operand_load(rs1.into(), TEMP_A);

        dynasm! {
            self;
            .arch x64;

            add Rd(TEMP_A), imm as i32;
            mov DWORD [Rq(CONTEXT) + PC_OFFSET], Rd(TEMP_A)
        }

        // ------------------------------------
        // 3. Store the computed PC into rd
        // ------------------------------------

        self.emit_risc_register_store(TEMP_B, rd);

        // ------------------------------------
        // 3. Jump to the target pc.
        // ------------------------------------

        self.jump_to_pc();
    }

    fn load_riscv_operand(&mut self, op: RiscOperand, dst: Self::ScratchRegister) {
        self.emit_risc_operand_load(op, dst as u8);
    }

    fn store_riscv_register(&mut self, value: Self::ScratchRegister, rd: RiscRegister) {
        self.emit_risc_register_store(value as u8, rd);
    }

    fn set_pc(&mut self, target: u32) {
        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // Set the PC in the context to the target.
            // ------------------------------------
            mov DWORD [Rq(CONTEXT) + PC_OFFSET], target as i32
        }
    }

    fn bump_clk(&mut self, amt: u32) {
        let clk_offset = offset_of!(JitContext, clk) as i32;

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // Add the amount to the clk field in the context.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + clk_offset], amt as i32
        }
    }

    fn exit_if_clk_exceeds(&mut self, max_cycles: u64) {
        let clk_offset = offset_of!(JitContext, clk) as i32;

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 1. Load clk into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), [Rq(CONTEXT) + clk_offset];

            // ------------------------------------
            // 2. Load max_cycles into TEMP_B
            // ------------------------------------
            mov Rq(TEMP_B), QWORD max_cycles as i64;

            // ------------------------------------
            // 3. Check if clk >= max_cycles
            // ------------------------------------
            cmp Rq(TEMP_A), Rq(TEMP_B);

            // ------------------------------------
            // 4. If clk >= max_cycles, return
            // ------------------------------------
            jae ->exit
        }
    }

    fn inspect_register(&mut self, reg: RiscRegister, handler: DebugFn) {
        // Load into the argument register for the function call.
        self.emit_risc_operand_load(RiscOperand::Register(reg), Rq::RDI as u8);

        // Call the handler with the value of the register.
        self.call_extern_fn_raw(handler as _);
    }

    fn inspect_immediate(&mut self, imm: u32, handler: DebugFn) {
        dynasm! {
            self;
            .arch x64;

            mov rdi, imm as i32
        }

        self.call_extern_fn_raw(handler as _);
    }

    fn auipc(&mut self, rd: RiscRegister, imm: u32) {
        // rd <- pc + imm
        // pc <- pc + 4
        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 1. Copy the current PC into TEMP_A
            // ------------------------------------
            mov Rd(TEMP_A), [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Increment the PC by the immediate.
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Increment the PC to the next instruction.
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }

        // Store the result in the destination register.
        self.emit_risc_register_store(TEMP_A, rd);
    }

    fn beq(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        // Compare the registers
        dynasm! {
            self;
            .arch x64;

            // Check if rs1 == rs2
            cmp Rd(TEMP_A), Rd(TEMP_B);
            // If rs1 != rs2, jump to not_branched, since that would imply !(rs1 == rs2)
            jne >not_branched;

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rd(TEMP_A), DWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rd(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2;
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bge(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;

            // Check if rs1 == rs2
            cmp Rd(TEMP_A), Rd(TEMP_B);
            // If rs1 < rs2, jump to not_branched, since that would imply !(rs1 >= rs2)
            jl >not_branched;

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rd(TEMP_A), DWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rd(TEMP_A), pc_base;
            shr Rd(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bgeu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;

            cmp Rd(TEMP_A), Rd(TEMP_B);
            // If rs1 < rs2, jump to not_branched, since that would imply !(rs1 >= rs2)
            jb >not_branched;

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rd(TEMP_A), DWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rd(TEMP_A), pc_base;
            shr Rd(TEMP_A), 2;
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn blt(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // Compare the two registers.
            //
            cmp Rd(TEMP_A), Rd(TEMP_B);   // signed compare
            jge >not_branched;            // rs1 ≥ rs2  →  skip

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rd(TEMP_A), DWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rd(TEMP_A), pc_base;
            shr Rd(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bltu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;
            cmp Rd(TEMP_A), Rd(TEMP_B);   // unsigned compare
            jae >not_branched;             // rs1 ≥ rs2 (unsigned) → skip

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rd(TEMP_A), DWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rd(TEMP_A), pc_base;
            shr Rd(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // 1. Bump the pc by 4
            add DWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bne(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;
            cmp Rd(TEMP_A), Rd(TEMP_B);   // sets ZF
            je  >not_branched;            // rs1 == rs2  →  skip

            // ------------------------------------
            // Branched:
            // 0. Bump the pc in the context by the immediate.
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rd(TEMP_A), DWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rd(TEMP_A), pc_base;
            shr Rd(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add DWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn ecall(&mut self) {
        // Load the JitContext pointer into the argument register.
        dynasm! {
            self;
            .arch x64;
            mov rdi, Rq(CONTEXT)
        };

        self.call_extern_fn_raw(self.ecall_handler as _);

        // Ecall semantics:
        // todo: document
        self.emit_risc_register_store(Rq::RAX as u8, RiscRegister::X5);

        self.jump_to_pc();
    }

    fn lb(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load in the base address and the phy sical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load byte → sign-extend to 32 bits
            // ------------------------------------
            movsx Rd(TEMP_B), BYTE [Rq(TEMP_B)]
        }

        // 4. Write back to destination register
        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lbu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load in the base address
        //    and the physical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to
            //    the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load byte → zero-extend to 32 bits
            // ------------------------------------
            movzx Rd(TEMP_B), BYTE [Rq(TEMP_B)]
        }

        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lh(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load in the base address
        //    and the physical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load half-word → sign-extend to 32 bits
            // ------------------------------------
            movsx Rd(TEMP_B), WORD [Rq(TEMP_B)]
        }

        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lhu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load in the base address
        //    and the physical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load half-word → zero-extend to 32 bits
            // ------------------------------------
            movzx Rd(TEMP_B), WORD [Rq(TEMP_B)]
        }

        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lw(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load the base (risc32) address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            //
            // We use 64 bit arithmetic since were dealing with native memory.
            // If there was an overflow it wouldve been handled correctly by the previous add.
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 3. Load the word from physical memory into TEMP_B
            // ------------------------------------
            mov Rd(TEMP_B), DWORD [Rq(TEMP_B)]
        }

        // ------------------------------------
        // 4. Store the result in the destination register.
        // ------------------------------------
        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn sb(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load the base (risc32) address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A)
        }

        // ------------------------------------
        // 4. Load the word from the RISC register into TEMP_A
        // ------------------------------------
        self.emit_risc_operand_load(rs2.into(), TEMP_A);

        // ------------------------------------
        // 5. Store the word into physical memory
        // ------------------------------------
        dynasm! {
            self;
            .arch x64;

            mov [Rq(TEMP_B)], Rb(TEMP_A)
        }
    }

    fn sh(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load the base (risc32) address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A)
        }

        // ------------------------------------
        // 4. Load the word from the RISC register into TEMP_A
        // ------------------------------------
        self.emit_risc_operand_load(rs2.into(), TEMP_A);

        // ------------------------------------
        // 5. Store the word into physical memory
        // ------------------------------------
        dynasm! {
            self;
            .arch x64;

            mov [Rq(TEMP_B)], Rw(TEMP_A)
        }
    }

    fn sw(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        // ------------------------------------
        // 1. Load the base (risc32) address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rd(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A)
        }

        // ------------------------------------
        // 4. Load the word from the RISC register into TEMP_A
        // ------------------------------------
        self.emit_risc_operand_load(rs2.into(), TEMP_A);

        // ------------------------------------
        // 5. Store the word into physical memory
        // ------------------------------------
        dynasm! {
            self;
            .arch x64;

            mov [Rq(TEMP_B)], Rd(TEMP_A)
        }
    }
}

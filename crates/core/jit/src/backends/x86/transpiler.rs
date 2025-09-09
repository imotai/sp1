#![allow(clippy::fn_to_numeric_cast)]

use super::{TranspilerBackend, CONTEXT, TEMP_A, TEMP_B};
use crate::{
    DebugFn, EcallHandler, ExternFn, JitContext, JitFunction, RiscOperand, RiscRegister,
    RiscvTranspiler,
};
use dynasmrt::{
    dynasm,
    x64::{Assembler, Rq},
    DynasmApi, DynasmLabelApi,
};
use std::{io, mem::offset_of};

impl RiscvTranspiler for TranspilerBackend {
    fn new(
        program_size: usize,
        memory_size: usize,
        trace_buf_size: usize,
        pc_start: u64,
        pc_base: u64,
        clk_bump: u64,
    ) -> Result<Self, std::io::Error> {
        if pc_start < pc_base {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "pc_start must be greater than pc_base",
            ));
        }

        let inner = Assembler::new()?;

        let mut this = Self {
            inner,
            jump_table: Vec::with_capacity(program_size),
            // Double the size of memory.
            // We are going to store entries of the form (clk, word).
            memory_size: memory_size * 2,
            trace_buf_size,
            has_instructions: false,
            pc_base,
            pc_start,
            // Register a dummy ecall handler.
            ecall_handler: super::ecallk as _,
            control_flow_instruction_inserted: false,
            instruction_started: false,
            clk_bump,
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

        // Push the offset of the jumpdest for this instruction.
        let offset = self.inner.offset();
        self.jump_table.push(offset.0);

        // If the instruction has already started, then we are in a bad state.
        if self.instruction_started {
            panic!("start_instr called without calling end_instr");
        }

        // We are now "within" an instruction.
        self.instruction_started = true;
    }

    fn end_instr(&mut self) {
        // Add the base amount of cycles for the instruction.
        self.bump_clk();

        // If the instruction is branch, jal, jalr or ecall then we need to emit a jump to pc
        if self.control_flow_instruction_inserted {
            self.jump_to_pc();
        } else {
            self.bump_pc(4);
        }

        self.control_flow_instruction_inserted = false;
        self.instruction_started = false;
    }

    fn exit_if_clk_exceeds(&mut self, max_cycles: u64) {
        let clk_offset = offset_of!(JitContext, global_clk) as i32;

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

    fn finalize(mut self) -> io::Result<JitFunction> {
        self.epilogue();

        let code = self.inner.finalize().expect("failed to finalize x86 backend");

        debug_assert!(code.size() > 0, "Got empty x86 code buffer");

        JitFunction::new(
            code,
            self.jump_table,
            self.memory_size,
            self.trace_buf_size,
            self.pc_start,
        )
    }

    fn call_extern_fn(&mut self, fn_ptr: ExternFn) {
        // Load the JitContext pointer into the argument register.
        dynasm! {
            self;
            .arch x64;
            mov rdi, Rq(CONTEXT)
        };

        self.call_extern_fn_raw(fn_ptr as _);
    }

    fn inspect_register(&mut self, reg: RiscRegister, handler: DebugFn) {
        // Load into the argument register for the function call.
        self.emit_risc_operand_load(RiscOperand::Register(reg), Rq::RDI as u8);

        // Call the handler with the value of the register.
        self.call_extern_fn_raw(handler as _);
    }

    fn inspect_immediate(&mut self, imm: u64, handler: DebugFn) {
        dynasm! {
            self;
            .arch x64;

            mov rdi, imm as i32
        }

        self.call_extern_fn_raw(handler as _);
    }
}

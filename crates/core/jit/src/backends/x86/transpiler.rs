#![allow(clippy::fn_to_numeric_cast)]

use super::{ScratchRegisterX86, TranspilerBackend, CONTEXT, PC_OFFSET, TEMP_A, TEMP_B};
use crate::{
    DebugFn, EcallHandler, ExternFn, JitContext, JitFunction, RiscOperand, RiscRegister,
    SP1RiscvTranspiler,
};
use dynasmrt::{
    dynasm,
    x64::{Assembler, Rq},
    DynasmApi, DynasmLabelApi,
};
use std::io;
use std::mem::offset_of;

impl SP1RiscvTranspiler for TranspilerBackend {
    type ScratchRegister = ScratchRegisterX86;

    fn new(
        program_size: usize,
        memory_size: usize,
        trace_buf_size: usize,
        pc_start: u64,
        pc_base: u64,
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

    fn load_riscv_operand(&mut self, op: RiscOperand, dst: Self::ScratchRegister) {
        self.emit_risc_operand_load(op, dst as u8);
    }

    fn store_riscv_register(&mut self, value: Self::ScratchRegister, rd: RiscRegister) {
        self.emit_risc_register_store(value as u8, rd);
    }

    fn set_pc(&mut self, target: u64) {
        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // Set the PC in the context to the target.
            // ------------------------------------
            mov QWORD [Rq(CONTEXT) + PC_OFFSET], target as i32
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

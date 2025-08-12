#![allow(clippy::fn_to_numeric_cast)]

use crate::{
    EcallHandler, JitContext, RiscOperand, RiscRegister, ScratchRegister, TraceChunkRaw,
    TraceCollector,
};
use dynasmrt::{
    dynasm,
    x64::{Assembler, Rq},
    DynasmApi, DynasmLabelApi,
};
use std::{
    mem::offset_of,
    ops::{Deref, DerefMut},
};

mod backend;
#[cfg(test)]
mod tests;
mod transpiler;

/// The first scratch register.
const TEMP_A: u8 = Rq::RBX as u8;

/// The second scratch register.
const TEMP_B: u8 = Rq::RBP as u8;

/// The JitContext pointer.
const CONTEXT: u8 = Rq::R12 as u8;

/// The jump table pointer.
const JUMP_TABLE: u8 = Rq::R13 as u8;

/// The trace buffer pointer.
const TRACE_BUF: u8 = Rq::R14 as u8;

/// The saved stack pointer, used during external function calls.
const SAVED_STACK_PTR: u8 = Rq::R15 as u8;

/// The offset of the pc in the JitContext.
const PC_OFFSET: i32 = offset_of!(JitContext, pc) as i32;

/// The offset of the memory pointer in the JitContext.
const MEMORY_PTR_OFFSET: i32 = offset_of!(JitContext, memory) as i32;

/// The offset of the registers in the JitContext.
const REGISTERS_OFFSET: i32 = offset_of!(JitContext, registers) as i32;

/// The x86 backend for JIT transpipling RISC-V instructions to x86-64, according to the
/// [crate::SP1RiscvTranspiler] trait.
pub struct TranspilerBackend {
    inner: Assembler,
    /// A mapping of pc - pc_base => offset in the code buffer.
    jump_table: Vec<usize>,
    /// The size of the memory buffer to allocate.
    memory_size: usize,
    /// The size of the trace buffer to allocate.
    trace_buf_size: usize,
    /// Has at least one instruction been inserted.
    has_instructions: bool,
    /// The pc base.
    pc_base: u32,
    /// The pc start.
    pc_start: u32,
    /// The ecall handler.
    ecall_handler: EcallHandler,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScratchRegisterX86 {
    A = TEMP_A,
    B = TEMP_B,
}

impl ScratchRegister for ScratchRegisterX86 {
    const A: Self = Self::A;
    const B: Self = Self::B;
}

impl std::ops::BitAnd<u8> for ScratchRegisterX86 {
    type Output = u8;

    fn bitand(self, rhs: u8) -> Self::Output {
        (self as u8).bitand(rhs)
    }
}

impl TraceCollector for TranspilerBackend {
    fn trace_registers(&mut self) {
        for reg in RiscRegister::all_registers().iter() {
            let (xmm_index, xmm_offset) = Self::get_xmm_index(*reg);
            let value_byte_offset = *reg as u32 * 4;

            dynasm! {
                self;
                .arch x64;

                pextrd [Rq(TRACE_BUF) + value_byte_offset as i32], Rx(xmm_index), xmm_offset
            };
        }
    }

    /// Write the value in the RiscRegister into the memory buffer.
    ///
    /// Its assumed that the value is actually the result of the memory read.
    fn trace_mem_value(&mut self, src: RiscRegister) {
        const TAIL_START_OFFSET: i32 = std::mem::size_of::<TraceChunkRaw>() as i32;
        const NUM_MEM_READS_OFFSET: i32 = offset_of!(TraceChunkRaw, num_mem_reads) as i32;

        // Load the value, assumed to be of a memory read, into TEMP_A.
        self.emit_risc_operand_load(src.into(), TEMP_B);

        dynasm! {
            self;
            .arch x64;


            // ------------------------------------
            // 1. Load the num mem reads and convert to a byte offset.
            // ------------------------------------
            mov Rd(TEMP_A), DWORD [Rq(TRACE_BUF) + NUM_MEM_READS_OFFSET];
            shl Rd(TEMP_A), 2; // num_mem_reads * 4, the size of a u32
            add Rd(TEMP_A), TAIL_START_OFFSET;

            // ------------------------------------
            // 3. Store the value into the tail.
            // ------------------------------------
            mov [Rq(TRACE_BUF) + Rq(TEMP_A)], Rd(TEMP_B);

            // ------------------------------------
            // 2. Increment the num mem reads, since weve pushed into it.
            // ------------------------------------
            add DWORD [Rq(TRACE_BUF) + NUM_MEM_READS_OFFSET], 1
        }
    }

    /// Write the start pc of the trace chunk.
    fn trace_pc_start(&mut self) {
        const PC_START_OFFSET: i32 = offset_of!(TraceChunkRaw, pc_start) as i32;

        self.load_pc_into_register(TEMP_A);

        dynasm! {
            self;
            .arch x64;

            mov [Rq(TRACE_BUF) + PC_START_OFFSET], Rd(TEMP_A)
        }
    }

    /// Write the start clk of the trace chunk.
    fn trace_clk_start(&mut self) {
        const CLK_START_OFFSET: i32 = offset_of!(TraceChunkRaw, clk_start) as i32;

        dynasm! {
            self;
            .arch x64;

            mov Rq(TEMP_A), [Rq(CONTEXT) + CLK_START_OFFSET];
            mov [Rq(TRACE_BUF) + CLK_START_OFFSET], Rd(TEMP_A)
        }
    }
}

impl TranspilerBackend {
    /// Emit the prologue for the function.
    ///
    /// This is called before the first instruction is emitted.
    fn prologue(&mut self) {
        // Compute the offsets so we can store some pointers seperately.
        let jump_table_offset = offset_of!(JitContext, jump_table) as i32;
        let trace_buf_offset = offset_of!(JitContext, trace_buf) as i32;

        // Prologue
        //
        // Push all the callee-saved registers we clobber, to be restored when we exit.
        //
        // We also want to 0 out all the registers we use,
        // since were operting on the lower 32 bits of them, and upper zereos could pose problems.
        dynasm! {
            self;
            .arch x64;

            // Save the callee saved registers were gonna clobber.
            push Rq(TEMP_A);
            push Rq(TEMP_B);
            push Rq(CONTEXT);
            push Rq(JUMP_TABLE);
            push Rq(TRACE_BUF);
            push Rq(SAVED_STACK_PTR);

            // 0 the registers we are gonna use the lower half of.
            // we alreadyt saved these registers on the stack.
            xor Rq(TEMP_A), Rq(TEMP_A);
            xor Rq(TEMP_B), Rq(TEMP_B);
            xor Rq(CONTEXT), Rq(CONTEXT);
            xor Rq(JUMP_TABLE), Rq(JUMP_TABLE);
            xor Rq(TRACE_BUF), Rq(TRACE_BUF);
            xor Rq(SAVED_STACK_PTR), Rq(SAVED_STACK_PTR);

            // Save some useful pointers to non-volatile registers so we can use them in ASM easily.
            mov Rq(JUMP_TABLE), [rdi + jump_table_offset];
            mov Rq(TRACE_BUF), [rdi + trace_buf_offset];
            // Save the JitContext pointer to a non-volatile register.
            mov Rq(CONTEXT), rdi;

            // Zero all the XMM registers we care about, they are volatile.
            xorps xmm0, xmm0;
            xorps xmm1, xmm1;
            xorps xmm2, xmm2;
            xorps xmm3, xmm3;
            xorps xmm4, xmm4;
            xorps xmm5, xmm5;
            xorps xmm6, xmm6;
            xorps xmm7, xmm7;
            xorps xmm8, xmm8;
            xorps xmm9, xmm9;
            xorps xmm10, xmm10;
            xorps xmm11, xmm11;
            xorps xmm12, xmm12;
            xorps xmm13, xmm13;
            xorps xmm14, xmm14;
            xorps xmm15, xmm15
        };

        // For each register from the context, lets load it into a phyiscal register.
        self.load_registers_from_context();

        // Its possible that enter back into the function with a non-zero PC.
        self.jump_to_pc();
    }

    /// Restore all the registers callee-saved registers we clobbered
    ///
    /// To be called after the last instruction has been emitted.
    fn epilogue(&mut self) {
        if !self.has_instructions {
            panic!(
                "No instructions were emitted, 
                cannot finalize as this will break assumptions made in the jump table."
            );
        }

        // Start the global exit label.
        // Its possible that we need to hit this label due to reaching cycle limt.
        dynasm! {
            self;
            .arch x64;

            // Define the exit global label.
            ->exit:
        }

        // Ensure the registers are saved to the context.
        self.save_registers_to_context();

        dynasm! {
            self;
            .arch x64;

            // Restore the caller saved registers.
            pop Rq(SAVED_STACK_PTR);
            pop Rq(TRACE_BUF);
            pop Rq(JUMP_TABLE);
            pop Rq(CONTEXT);
            pop Rq(TEMP_B);
            pop Rq(TEMP_A);

            ret
        };
    }

    fn save_registers_to_context(&mut self) {
        for reg in RiscRegister::all_registers().iter() {
            let (xmm_index, xmm_offset) = Self::get_xmm_index(*reg);
            let value_byte_offset = *reg as u32 * 4;

            dynasm! {
                self;
                .arch x64;

                pextrd [Rq(CONTEXT) + REGISTERS_OFFSET + value_byte_offset as i32], Rx(xmm_index), xmm_offset
            };
        }
    }

    fn load_registers_from_context(&mut self) {
        // For each register from the context, lets load it into a phyiscal register.
        for reg in RiscRegister::all_registers().iter() {
            let (xmm_index, xmm_offset) = Self::get_xmm_index(*reg);
            let value_byte_offset = *reg as u32 * 4;

            dynasm! {
                self;
                .arch x64;

                pinsrd Rx(xmm_index), [Rq(CONTEXT) + REGISTERS_OFFSET + value_byte_offset as i32], xmm_offset
            };
        }
    }

    /// RiscV registers are mapped to XMM registers.
    ///
    /// We load the value from the XMM register into the general purpose register for the backend to operate on.
    /// We do this to avoid accidently clobbering the XMM registers.
    ///
    /// NOTE: This aliases the lower 32 bits of the register.
    fn emit_risc_operand_load(&mut self, op: RiscOperand, dst: u8) {
        match op {
            RiscOperand::Register(reg) => match reg {
                RiscRegister::X0 => {
                    dynasm! {
                        self;
                        .arch x64;

                        mov Rq(dst), 0_i32 // load 0 into dst
                    };
                }
                _ => {
                    let (xmm_index, xmm_offset) = Self::get_xmm_index(reg);

                    dynasm! {
                        self;
                        .arch x64;

                        pextrd Rq(dst), Rx(xmm_index), xmm_offset // load into low 32 bits of dst
                    };
                }
            },
            RiscOperand::Immediate(imm) => {
                dynasm! {
                    self;
                    .arch x64;

                    mov Rq(dst), imm
                };
            }
        }
    }

    /// Store the value from the general purpose register into the corresponding XMM register.
    ///
    /// Note: This aliases the lower 32 bits of the register.
    #[inline]
    fn emit_risc_register_store(&mut self, src: u8, dst: RiscRegister) {
        if dst == RiscRegister::X0 {
            // x0 is hardwired to 0 in RISC-V, ignore stores to it.
            return;
        }

        let (xmm_index, xmm_offset) = Self::get_xmm_index(dst);

        dynasm! {
            self;
            .arch x64;
            pinsrd Rx(xmm_index), Rq(src), xmm_offset
        };
    }

    /// Compute the XMM index and offset for the given register.
    ///
    /// We operate on the assumption there are 8 128 bit XMM registers we can use.
    /// We map a register to an index in the range `[0, 7]` and an offset in the range `[0, 3]`.
    const fn get_xmm_index(reg: RiscRegister) -> (u8, i8) {
        let reg = reg as u8;
        (reg / 2, (reg % 2) as i8)
    }

    /// Call an external function, assumes that the arguments are already in the correct registers.
    #[inline]
    fn call_extern_fn_raw(&mut self, fn_ptr: usize) {
        // Before the call, save all the registers to the context.
        self.save_registers_to_context();

        // We need to save the caller-saved registers before we make any calls,
        // then restore them after the call.
        dynasm! {
            self;
            .arch x64;

            // Save the original stack pointer
            mov Rq(SAVED_STACK_PTR), rsp;

            // Align the stack to 16 bytes for the call
            lea rsp, [rsp - 8]; // sub 8 from the rsp
            mov rax, rsp; // copy
            and rax, 15; // compute rsp % 16
            sub rsp, rax; // sub that from the rsp to ensure 16 byte alignment

            sub rsp, 256;

            // Save the XMM registers to the stack
            // These are always volatile, and we dont want them to be clobbered.
            movdqu [rsp], xmm0;
            movdqu [rsp + 16], xmm1;
            movdqu [rsp + 32], xmm2;
            movdqu [rsp + 48], xmm3;
            movdqu [rsp + 64], xmm4;
            movdqu [rsp + 80], xmm5;
            movdqu [rsp + 96], xmm6;
            movdqu [rsp + 112], xmm7;
            movdqu [rsp + 128], xmm8;
            movdqu [rsp + 144], xmm9;
            movdqu [rsp + 160], xmm10;
            movdqu [rsp + 176], xmm11;
            movdqu [rsp + 192], xmm12;
            movdqu [rsp + 208], xmm13;
            movdqu [rsp + 224], xmm14;
            movdqu [rsp + 240], xmm15;

            // Call the external function
            mov rax, QWORD fn_ptr as _;
            call rax;

            // Restore xmm0 and xmm1
            movdqu xmm0, [rsp];
            movdqu xmm1, [rsp + 16];
            movdqu xmm2, [rsp + 32];
            movdqu xmm3, [rsp + 48];
            movdqu xmm4, [rsp + 64];
            movdqu xmm5, [rsp + 80];
            movdqu xmm6, [rsp + 96];
            movdqu xmm7, [rsp + 112];
            movdqu xmm8, [rsp + 128];
            movdqu xmm9, [rsp + 144];
            movdqu xmm10, [rsp + 160];
            movdqu xmm11, [rsp + 176];
            movdqu xmm12, [rsp + 192];
            movdqu xmm13, [rsp + 208];
            movdqu xmm14, [rsp + 224];
            movdqu xmm15, [rsp + 240];

            // Restore the original stack pointer
            mov rsp, Rq(SAVED_STACK_PTR)
        }

        self.load_registers_from_context();
    }

    /// Load the pc from the context into the given register.
    #[inline]
    fn load_pc_into_register(&mut self, dst: u8) {
        let pc_offset = offset_of!(JitContext, pc) as i32;

        dynasm! {
            self;
            .arch x64;
            mov Rq(dst), QWORD [Rq(CONTEXT) + pc_offset]
        }
    }

    #[inline]
    fn load_memory_ptr(&mut self, src: u8) {
        dynasm! {
            self;
            .arch x64;
            mov Rq(src), QWORD [Rq(CONTEXT) + MEMORY_PTR_OFFSET]
        }
    }

    /// Bump the pc by the given amount.
    #[inline]
    fn bump_pc(&mut self, amt: u32) {
        let pc_offset = offset_of!(JitContext, pc) as i32;

        dynasm! {
            self;
            .arch x64;

            add QWORD [Rq(CONTEXT) + pc_offset], amt as i32
        }
    }

    /// Looks up into the jump table and executes a jump.
    #[inline]
    fn jump_to_pc(&mut self) {
        self.load_pc_into_register(TEMP_A);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;

            // If the PC we want to jump to is 0, jump to the exit label.
            cmp Rq(TEMP_A), 0;
            je ->exit;

            // add the pc to the jump table pointer and load
            // Scale by 8 since these are 8 byte pointers
            sub Rq(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2; // Divide by 4 to get the index
            mov Rq(TEMP_B), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8]; // Convert to a byte offset
            jmp Rq(TEMP_B)
        }
    }
}

impl Deref for TranspilerBackend {
    type Target = Assembler;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TranspilerBackend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// The backend implicity relies on the exsitence of 16 128 bit XMM registers.
///
/// If this is not the case, we throw an error at compile time.
#[cfg(not(target_feature = "sse"))]
compile_error!("SSE is required for the x86 backend");

/// The backend implicity relies on the target being little endian.
///
/// If this is not the case, we throw an error at compile time.
#[cfg(not(target_endian = "little"))]
compile_error!("Little endian is required for the x86 backend");

/// A dummy ecall handler that can be called by the JIT.
extern "C" fn ecallk(ctx: *mut JitContext) -> u32 {
    let ctx = unsafe { &mut *ctx };

    eprintln!("dummy ecall handler called with code: 0x{:x}", ctx.registers[5]);

    if ctx.registers[5] == 0 {
        ctx.pc = 0;
    } else {
        ctx.pc += 4;
    }

    ctx.clk += 256;

    0
}

pub mod backends;
mod macros;
pub mod risc;

pub use backends::*;
pub use risc::*;

/// A transpiler for risc32 instructions.
///
/// This trait is implemented for each target architecture supported by the JIT transpiler.
///
/// The transpiler is responsible for translating the risc32 instructions into the target
/// architecture's instruction set.
///
/// This transpiler should generate an entrypoint of the form: [`fn(*mut JitContext)`]
///
/// For each instruction, you will typically want to call [`SP1RiscvTranspiler::start_instr`]
/// before transpiling the instruction. This maps a "riscv instruction index" to some physical native address, as there
/// are multiple native instructions per riscv instruction.
///
/// You will also likely want to call [`SP1RiscvTranspiler::bump_clk`] to increment the clock counter, and
/// [`SP1RiscvTranspiler::set_pc`] to set the PC.
///
/// # Note
/// Some instructions will directly modify the PC, such as [`SP1RiscvTranspiler::jal`] and [`SP1RiscvTranspiler::jalr`], and all the branch instructions,
/// for these instructions, you would not want to call [`SP1RiscvTranspiler::set_pc`] as it will be called for you.
///
///
/// ```rust,no_run,ignore
/// pub fn add_program() {
///     let mut transpiler = SP1RiscvTranspiler::new(program_size, memory_size, trace_buf_size).unwrap();
///      
///     // Transpile the first instruction.
///     transpiler.start_instr();
///     transpiler.add(RiscOperand::Reg(RiscRegister::A), RiscOperand::Reg(RiscRegister::B), RiscRegister::C);
///     transpiler.bump_clk(4);
///     transpiler.set_pc(4);
///     
///     // Transpile the second instruction.
///     transpiler.start_instr();
///
///     transpiler.add(RiscOperand::Reg(RiscRegister::A), RiscOperand::Reg(RiscRegister::B), RiscRegister::C);
///     transpiler.bump_clk(4);
///     transpiler.set_pc(8);
///     
///     let mut func = transpiler.finalize();
///
///     // Call the function.
///     let traces = func.call();
///
///     // do stuff with the traces.
/// }
/// ```
pub trait SP1RiscvTranspiler: TraceCollector + ALUBackend + Sized {
    /// Create a new transpiler.
    ///
    /// The program is used for the jump-table and is not a hard limit on the size of the program.
    /// The memory size is the exact amount that will be allocated for the program.
    fn new(
        program_size: usize,
        memory_size: usize,
        trace_buf_size: usize,
        pc_start: u32,
        pc_base: u32,
    ) -> Result<Self, std::io::Error>;

    fn register_ecall_handler(&mut self, handler: EcallHandler);

    /// Populates a jump table entry for the current instruction being transpiled.
    ///
    /// Effectively should create a mapping from RISCV PC -> absolute address of the instruction.
    ///
    /// This method should be called for "each pc" in the program.
    fn start_instr(&mut self);

    /// Load a [RiscRegister] or an immedeatie into a native register.
    fn load_riscv_operand(&mut self, src: RiscOperand, dst: Self::ScratchRegister);

    /// Load a [ScratchRegister] into a [`RiscRegister`].
    fn store_riscv_register(&mut self, src: Self::ScratchRegister, dst: RiscRegister);

    /// Adds amt to the clk in the context.
    fn bump_clk(&mut self, amt: u32);

    /// Set the PC in the context.
    /// pc <- target
    fn set_pc(&mut self, target: u32);

    /// Exit if the clk is greater than some value.
    ///
    /// This should also set the PC to the "continution point" if exists.
    fn exit_if_clk_exceeds(&mut self, max_cycles: u64);

    /// Inspcet a [RiscRegister] using a function pointer.
    ///
    /// Implementors should ensure that [`RiscvTranspiler::start_instr`] is called before this.
    fn inspect_register(&mut self, reg: RiscRegister, handler: DebugFn);

    /// Print an immediate value.
    ///
    /// Implementors should ensure that [`RiscvTranspiler::start_instr`] is called before this.
    fn inspect_immediate(&mut self, imm: u32, handler: DebugFn);

    /// Load a byte from memory into a register.
    ///
    /// lb: rd = m8(rs1 + imm)
    fn lb(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Load a half word from memory into a register.
    ///
    /// lh: rd = m16(rs1 + imm)
    fn lh(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Load a word from memory into a register.
    ///
    /// lw: rd = m32(rs1 + imm)
    fn lw(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Load a byte from memory into a register, zero extended.
    ///
    /// lbu: rd = zx(m8(rs1 + imm))
    fn lbu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Load a half word from memory into a register, zero extended.    
    ///
    /// lhu: rd = zx(m16(rs1 + imm))
    fn lhu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Load a double word from memory into a register.
    ///
    /// ldu: rd = m64(rs1 + imm)
    fn ld(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Load a word from memory into a register, zero extended.
    ///
    /// lwu: rd = zx(m32(rs1 + imm))
    fn lwu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Store a byte into memory.
    ///
    /// sb: m8(rs1 + imm) = rs2[7:0]
    fn sb(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Store a half word into memory.
    ///
    /// sh: m16(rs1 + imm) = rs2[15:0]
    fn sh(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Store a word into memory.
    ///
    /// sw: m32(rs1 + imm) = rs2[31:0]
    fn sw(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Store a double word into memory.
    ///
    /// sd: m64(rs1 + imm) = rs2[63:0]
    fn sd(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Compare the values of two registers, and jump to an address if they are equal.
    ///
    /// beq: pc = pc + ((rs1 == rs2) ? imm : 4)
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn beq(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Compare the values of two registers, and jump to an address if they are not equal.
    ///
    /// bne: pc = pc + ((rs1 != rs2) ? imm : 4)
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn bne(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Compare the values of two registers, and jump to an address if the first is less than the second.
    ///
    /// blt: pc = pc + ((rs1 < rs2) ? imm : 4)
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn blt(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Compare the values of two registers, and jump to an address if the first is greater than or equal to the second.
    ///
    /// bge: pc = pc + ((rs1 >= rs2) ? imm : 4)
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn bge(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Compare the values of two registers, and jump to an address if the first is less than the second, unsigned.
    ///
    /// bltu: pc = pc + ((rs1 < rs2) ? imm : 4)
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn bltu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Compare the values of two registers, and jump to an address if the first is greater than or equal to the second, unsigned.
    ///
    /// bgeu: pc = pc + ((rs1 >= rs2) ? imm : 4)
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn bgeu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32);

    /// Jump to an address.
    ///
    /// jal: rd = pc + 4, pc = pc + imm
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn jal(&mut self, rd: RiscRegister, imm: u32);

    /// Jump to an address, and return to the previous address.
    ///
    /// jalr: rd = pc + 4, pc = rs1 + imm
    ///
    /// NOTE: During transpilatiom, this method will emit the PC bumps for you,
    /// typically however, you will want to explicty call [`SP1RiscvTranspiler::set_pc`] at the end of each instruction.
    fn jalr(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32);

    /// Advance to the next pc, storing the current (pc + imm) in a register.
    ///
    /// auipc: rd = pc + imm, pc = pc + 4
    fn auipc(&mut self, rd: RiscRegister, imm: u32);

    /// Transfer control to the operating system.
    fn ecall(&mut self);

    fn unimp(&mut self) {
        extern "C" fn unimp(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("Unimplemented instruction at pc: {}", ctx.pc);
        }

        self.call_extern_fn(unimp);
    }

    impl_risc_alu! {
        add,
        sub,
        mul,
        div,
        divu,
        rem,
        remu,
        mulh,
        mulhu,
        mulhsu,
        sll,
        srl,
        sra,
        slt,
        sltu,
        xor,
        or,
        and
    }
}

/// A trait the collects traces, in the form [TraceChunk].
///
/// This type is expected to follow the conventions as described in the [TraceChunk] documentation.
pub trait TraceCollector {
    /// Write the current state of the registers into the trace buf.
    ///
    /// For SP1 this is only called once in the beginning of a "chunk".
    fn trace_registers(&mut self);

    /// Write the value in the RiscRegister into the memory buffer.
    ///
    /// Its assumed that the value is actually the result of the memory read.
    fn trace_mem_value(&mut self, src: RiscRegister);

    /// Write the start pc of the trace chunk.
    fn trace_pc_start(&mut self);

    /// Write the start clk of the trace chunk.
    fn trace_clk_start(&mut self);
}

/// A function that accepts the memory pointer.
pub type ExternFn = extern "C" fn(*mut JitContext);

pub type EcallHandler = extern "C" fn(*mut JitContext) -> u32;

/// A debugging utility to inspect registers
pub type DebugFn = extern "C" fn(u32);

pub trait Debuggable {
    fn print_ctx(&mut self);
}

impl<T: SP1RiscvTranspiler> Debuggable for T {
    // Useful only for debugging.
    fn print_ctx(&mut self) {
        extern "C" fn print_ctx(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("pc: {}", ctx.pc);
            eprintln!("clk: {}", ctx.clk);
        }

        self.call_extern_fn(print_ctx);
    }
}

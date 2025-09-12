#![allow(clippy::items_after_statements)]

use crate::{memory::MAX_LOG_ADDR, HALT_PC};
use sp1_jit::{
    DebugBackend, JitFunction, MemoryView, RiscOperand, RiscRegister, RiscvTranspiler,
    TranspilerBackend,
};
use std::{collections::VecDeque, sync::Arc};

use crate::{Instruction, Opcode, Program, Register};
pub use sp1_jit::TraceChunkRaw;

mod ecall;
mod hint;
mod postprocess;
mod precompiles;
mod unconstrained;
mod write;

#[cfg(test)]
mod tests;

/// A minimal trace executor.
pub struct MinimalExecutor {
    program: Arc<Program>,
    compiled: JitFunction,
    input: VecDeque<Vec<u8>>,
}

impl MinimalExecutor {
    /// Create a new minimal executor and transpiles the program.
    ///
    /// If the trace size is not set, it will be set to 2^36 bytes.
    #[must_use]
    pub fn new(program: Arc<Program>, is_debug: bool, max_trace_size: Option<u64>) -> Self {
        tracing::debug!("transpiling program, debug={is_debug}, max_trace_size={max_trace_size:?}");

        let compiled = Self::transpile(program.as_ref(), is_debug, max_trace_size);

        Self { program, compiled, input: VecDeque::new() }
    }

    /// Transpile the program, saving the JIT function.
    fn transpile(program: &Program, debug: bool, max_trace_size: Option<u64>) -> JitFunction {
        let trace_buf_size =
            max_trace_size.map_or(0, |max_trace_size| (max_trace_size as usize * 10) / 9);

        let mut backend = TranspilerBackend::new(
            program.instructions.len(),
            2_u64.pow(MAX_LOG_ADDR as u32) as usize,
            trace_buf_size,
            program.pc_start_abs,
            program.pc_base,
            8,
        )
        .expect("Failed to create transpiler backend");

        backend.register_ecall_handler(ecall::sp1_ecall_handler);

        if debug {
            Self::transpile_instructions(DebugBackend::new(backend), program, max_trace_size)
        } else {
            Self::transpile_instructions(backend, program, max_trace_size)
        }
    }

    /// Create a new minimal executor with no tracing or debugging.
    #[must_use]
    pub fn simple(program: Arc<Program>) -> Self {
        Self::new(program, false, None)
    }

    /// Create a new minimal executor with tracing.
    ///
    /// If the trace size is not set, it will be set to 2^36.
    #[must_use]
    pub fn tracing(program: Arc<Program>, max_trace_size: Option<u64>) -> Self {
        Self::new(program, false, max_trace_size.or(Some(2_u64.pow(35))))
    }

    /// Create a new minimal executor with debugging.
    #[must_use]
    pub fn debug(program: Arc<Program>) -> Self {
        Self::new(program, true, None)
    }

    /// Add input to the executor.
    pub fn with_input(&mut self, input: &[u8]) {
        self.input.push_back(input.to_vec());
    }

    /// Execute the program. Returning a trace chunk if the program has not completed.
    pub fn execute_chunk(&mut self) -> Option<TraceChunkRaw> {
        if !self.input.is_empty() {
            self.compiled.set_input_buffer(std::mem::take(&mut self.input));
        }

        // SAFETY: The backend is assumed to output valid JIT functions.
        unsafe { self.compiled.call() }
    }

    /// Check if the program has halted.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.compiled.pc == HALT_PC
    }

    /// Get the current clock of the JIT function.
    ///
    /// This clock is incremented by 8 or 256 depending on the instruction.
    #[must_use]
    pub fn clk(&self) -> u64 {
        self.compiled.clk
    }

    /// Get the global clock of the JIT function.
    ///
    /// This clock is incremented by 1 per instruction.
    #[must_use]
    pub fn global_clk(&self) -> u64 {
        self.compiled.global_clk
    }

    /// Get the public values stream of the JIT function.
    #[must_use]
    pub fn public_values_stream(&self) -> &Vec<u8> {
        &self.compiled.public_values_stream
    }

    /// Consume self, and return the public values stream.
    #[must_use]
    pub fn into_public_values_stream(self) -> Vec<u8> {
        self.compiled.public_values_stream
    }

    /// Get the hints of the JIT function.
    #[must_use]
    pub fn hints(&self) -> &Vec<(u64, Vec<u8>)> {
        &self.compiled.hints
    }

    /// Get the lengths of all the hints.
    #[must_use]
    pub fn hint_lens(&self) -> Vec<usize> {
        self.compiled.hints.iter().map(|(_, hint)| hint.len()).collect()
    }

    /// Get a view of the current memory of the JIT function.
    #[must_use]
    pub fn memory(&self) -> MemoryView<'_> {
        MemoryView::new(&self.compiled.memory)
    }

    /// Reset the JIT function, to start from the beginning of the program.
    pub fn reset(&mut self) {
        self.compiled.reset();

        let _ = std::mem::take(&mut self.input);
    }

    fn transpile_instructions<B: RiscvTranspiler>(
        mut backend: B,
        program: &Program,
        max_trace_size: Option<u64>,
    ) -> JitFunction {
        let tracing = max_trace_size.is_some();

        for instruction in program.instructions.iter() {
            backend.start_instr();

            match instruction.opcode {
                Opcode::LB
                | Opcode::LH
                | Opcode::LW
                | Opcode::LBU
                | Opcode::LHU
                | Opcode::LD
                | Opcode::LWU => {
                    Self::transpile_load_instruction(&mut backend, instruction, tracing);
                    if let Some(max_trace_size) = max_trace_size {
                        backend.exit_if_trace_exceeds(max_trace_size);
                    }
                }
                Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD => {
                    Self::transpile_store_instruction(&mut backend, instruction, tracing);
                    if let Some(max_trace_size) = max_trace_size {
                        backend.exit_if_trace_exceeds(max_trace_size);
                    }
                }
                Opcode::BEQ
                | Opcode::BNE
                | Opcode::BLT
                | Opcode::BGE
                | Opcode::BLTU
                | Opcode::BGEU => {
                    Self::transpile_branch_instruction(&mut backend, instruction);
                }
                Opcode::JAL | Opcode::JALR => {
                    Self::transpile_jump_instruction(&mut backend, instruction);
                }
                Opcode::ADD
                | Opcode::ADDI
                | Opcode::SUB
                | Opcode::XOR
                | Opcode::OR
                | Opcode::AND
                | Opcode::SLL
                | Opcode::SRL
                | Opcode::SRA
                | Opcode::SLT
                | Opcode::SLTU
                | Opcode::MUL
                | Opcode::MULH
                | Opcode::MULHU
                | Opcode::MULHSU
                | Opcode::DIV
                | Opcode::DIVU
                | Opcode::REM
                | Opcode::REMU
                | Opcode::ADDW
                | Opcode::SUBW
                | Opcode::SLLW
                | Opcode::SRLW
                | Opcode::SRAW
                | Opcode::DIVUW
                | Opcode::DIVW
                | Opcode::MULW
                | Opcode::REMUW
                | Opcode::REMW
                    if instruction.is_alu_instruction() =>
                {
                    Self::transpile_alu_instruction(&mut backend, instruction);
                }
                Opcode::AUIPC => {
                    let (rd, imm) = instruction.u_type();
                    backend.auipc(rd.into(), imm);
                }
                Opcode::LUI => {
                    let (rd, imm) = instruction.u_type();
                    backend.lui(rd.into(), imm);
                }
                Opcode::ECALL => {
                    backend.ecall();
                    if let Some(max_trace_size) = max_trace_size {
                        backend.exit_if_trace_exceeds(max_trace_size);
                    }
                }
                Opcode::EBREAK | Opcode::UNIMP => {
                    backend.unimp();
                }
                _ => panic!("Invalid instruction: {:?}", instruction.opcode),
            }

            backend.end_instr();
        }

        let mut finalized = backend.finalize().expect("Failed to finalize function");
        finalized.with_initial_memory_image(program.memory_image.clone());

        finalized
    }

    fn transpile_load_instruction<B: RiscvTranspiler>(
        backend: &mut B,
        instruction: &Instruction,
        tracing: bool,
    ) {
        let (rd, rs1, imm) = instruction.i_type();

        // For each load, we want to trace the value at the address as well as the previous clock
        // at that address.
        if tracing {
            backend.trace_mem_value(rs1.into(), imm);
        }

        match instruction.opcode {
            Opcode::LB => backend.lb(rd.into(), rs1.into(), imm),
            Opcode::LH => backend.lh(rd.into(), rs1.into(), imm),
            Opcode::LW => backend.lw(rd.into(), rs1.into(), imm),
            Opcode::LBU => backend.lbu(rd.into(), rs1.into(), imm),
            Opcode::LHU => backend.lhu(rd.into(), rs1.into(), imm),
            Opcode::LD => backend.ld(rd.into(), rs1.into(), imm),
            Opcode::LWU => backend.lwu(rd.into(), rs1.into(), imm),
            _ => unreachable!("Invalid load opcode: {:?}", instruction.opcode),
        }
    }

    fn transpile_store_instruction<B: RiscvTranspiler>(
        backend: &mut B,
        instruction: &Instruction,
        tracing: bool,
    ) {
        let (rs1, rs2, imm) = instruction.s_type();

        // For stores, its the same logic as a load, we want the last known clk and value at the
        // address.
        if tracing {
            backend.trace_mem_value(rs2.into(), imm);
        }

        // Note: We switch around rs1 and rs2 operaneds to align with the executor.
        match instruction.opcode {
            Opcode::SB => backend.sb(rs2.into(), rs1.into(), imm),
            Opcode::SH => backend.sh(rs2.into(), rs1.into(), imm),
            Opcode::SW => backend.sw(rs2.into(), rs1.into(), imm),
            Opcode::SD => backend.sd(rs2.into(), rs1.into(), imm),
            _ => unreachable!("Invalid store opcode: {:?}", instruction.opcode),
        }
    }

    fn transpile_branch_instruction<B: RiscvTranspiler>(
        backend: &mut B,
        instruction: &Instruction,
    ) {
        let (rs1, rs2, imm) = instruction.b_type();
        match instruction.opcode {
            Opcode::BEQ => backend.beq(rs1.into(), rs2.into(), imm),
            Opcode::BNE => backend.bne(rs1.into(), rs2.into(), imm),
            Opcode::BLT => backend.blt(rs1.into(), rs2.into(), imm),
            Opcode::BGE => backend.bge(rs1.into(), rs2.into(), imm),
            Opcode::BLTU => backend.bltu(rs1.into(), rs2.into(), imm),
            Opcode::BGEU => backend.bgeu(rs1.into(), rs2.into(), imm),
            _ => unreachable!("Invalid branch opcode: {:?}", instruction.opcode),
        }
    }

    fn transpile_jump_instruction<B: RiscvTranspiler>(backend: &mut B, instruction: &Instruction) {
        match instruction.opcode {
            Opcode::JAL => {
                let (rd, imm) = instruction.j_type();
                backend.jal(rd.into(), imm);
            }
            Opcode::JALR => {
                let (rd, rs1, imm) = instruction.i_type();

                backend.jalr(rd.into(), rs1.into(), imm);
            }
            _ => unreachable!("Invalid jump opcode: {:?}", instruction.opcode),
        }
    }

    fn transpile_alu_instruction<B: RiscvTranspiler>(backend: &mut B, instruction: &Instruction) {
        let (rd, b, c): (RiscRegister, RiscOperand, RiscOperand) = if !instruction.imm_c {
            let (rd, rs1, rs2) = instruction.r_type();

            (rd.into(), rs1.into(), rs2.into())
        } else if !instruction.imm_b && instruction.imm_c {
            let (rd, rs1, imm) = instruction.i_type();

            (rd.into(), rs1.into(), imm.into())
        } else {
            debug_assert!(instruction.imm_b && instruction.imm_c);
            let (rd, b, c) =
                (Register::from_u8(instruction.op_a), instruction.op_b, instruction.op_c);

            (rd.into(), b.into(), c.into())
        };

        match instruction.opcode {
            Opcode::ADD | Opcode::ADDI => backend.add(rd, b, c),
            Opcode::SUB => backend.sub(rd, b, c),
            Opcode::XOR => backend.xor(rd, b, c),
            Opcode::OR => backend.or(rd, b, c),
            Opcode::AND => backend.and(rd, b, c),
            Opcode::SLL => backend.sll(rd, b, c),
            Opcode::SRL => backend.srl(rd, b, c),
            Opcode::SRA => backend.sra(rd, b, c),
            Opcode::SLT => backend.slt(rd, b, c),
            Opcode::SLTU => backend.sltu(rd, b, c),
            Opcode::MUL => backend.mul(rd, b, c),
            Opcode::MULH => backend.mulh(rd, b, c),
            Opcode::MULHU => backend.mulhu(rd, b, c),
            Opcode::MULHSU => backend.mulhsu(rd, b, c),
            Opcode::DIV => backend.div(rd, b, c),
            Opcode::DIVU => backend.divu(rd, b, c),
            Opcode::REM => backend.rem(rd, b, c),
            Opcode::REMU => backend.remu(rd, b, c),
            Opcode::ADDW => backend.addw(rd, b, c),
            Opcode::SUBW => backend.subw(rd, b, c),
            Opcode::SLLW => backend.sllw(rd, b, c),
            Opcode::SRLW => backend.srlw(rd, b, c),
            Opcode::SRAW => backend.sraw(rd, b, c),
            Opcode::MULW => backend.mulw(rd, b, c),
            Opcode::DIVUW => backend.divuw(rd, b, c),
            Opcode::DIVW => backend.divw(rd, b, c),
            Opcode::REMUW => backend.remuw(rd, b, c),
            Opcode::REMW => backend.remw(rd, b, c),
            _ => unreachable!("Invalid ALU opcode: {:?}", instruction.opcode),
        }
    }
}

impl From<Register> for RiscOperand {
    fn from(value: Register) -> Self {
        RiscOperand::Register(value.into())
    }
}

impl From<Register> for RiscRegister {
    fn from(value: Register) -> Self {
        match value {
            Register::X0 => RiscRegister::X0,
            Register::X1 => RiscRegister::X1,
            Register::X2 => RiscRegister::X2,
            Register::X3 => RiscRegister::X3,
            Register::X4 => RiscRegister::X4,
            Register::X5 => RiscRegister::X5,
            Register::X6 => RiscRegister::X6,
            Register::X7 => RiscRegister::X7,
            Register::X8 => RiscRegister::X8,
            Register::X9 => RiscRegister::X9,
            Register::X10 => RiscRegister::X10,
            Register::X11 => RiscRegister::X11,
            Register::X12 => RiscRegister::X12,
            Register::X13 => RiscRegister::X13,
            Register::X14 => RiscRegister::X14,
            Register::X15 => RiscRegister::X15,
            Register::X16 => RiscRegister::X16,
            Register::X17 => RiscRegister::X17,
            Register::X18 => RiscRegister::X18,
            Register::X19 => RiscRegister::X19,
            Register::X20 => RiscRegister::X20,
            Register::X21 => RiscRegister::X21,
            Register::X22 => RiscRegister::X22,
            Register::X23 => RiscRegister::X23,
            Register::X24 => RiscRegister::X24,
            Register::X25 => RiscRegister::X25,
            Register::X26 => RiscRegister::X26,
            Register::X27 => RiscRegister::X27,
            Register::X28 => RiscRegister::X28,
            Register::X29 => RiscRegister::X29,
            Register::X30 => RiscRegister::X30,
            Register::X31 => RiscRegister::X31,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use sp1_primitives::Elf;

    #[allow(clippy::cast_precision_loss)]
    fn run_program_and_compare_end_state(program: &Elf) {
        let program = Program::from(program).unwrap();
        let program = Arc::new(program);

        let mut interpreter =
            crate::executor::Executor::new(program.clone(), crate::SP1CoreOpts::default());
        let start = std::time::Instant::now();
        interpreter.run_fast().expect("Interpreter failed");
        let interpreter_time = start.elapsed();

        let mut executor = MinimalExecutor::new(program.clone(), false, Some(2_u64.pow(35)));
        let start = std::time::Instant::now();
        executor.execute_chunk();
        let jit_time = start.elapsed();

        // convert to mhz
        let cycles = executor.compiled.global_clk;
        let mhz = cycles as f64 / (jit_time.as_micros() as f64);
        eprintln!("cycles={cycles}");
        eprintln!("jit mhz={mhz} mhz");

        let interpreter_cycles = interpreter.state.global_clk;
        let interpreter_mhz = interpreter_cycles as f64 / (interpreter_time.as_micros() as f64);
        eprintln!("interpreter mhz={interpreter_mhz} mhz");

        let interpreter_registers = interpreter
            .state
            .memory
            .registers
            .registers
            .iter()
            .map(|r| r.map(|r| r.value).unwrap_or(0))
            .collect::<Vec<_>>();

        let jit_registers = executor.compiled.registers;

        for i in 0..32 {
            if interpreter_registers[i] != jit_registers[i] {
                eprintln!("interpreter registers: {interpreter_registers:?}");
                eprintln!("jit registers: {jit_registers:?}");
                panic!(
                    "JIT[{i}] = {} != interpreter[{i}] = {}",
                    jit_registers[i], interpreter_registers[i]
                );
            }
        }

        assert_eq!(
            executor.compiled.clk, interpreter.state.clk,
            "JIT and interpreter final clk mismatch"
        );

        assert_eq!(
            executor.compiled.global_clk, interpreter.state.global_clk,
            "JIT and interpreter final global_clk mismatch"
        );
    }

    #[test]
    fn test_run_keccak_with_input() {
        use bincode::serialize;
        use test_artifacts::KECCAK256_ELF;

        let program = Program::from(&KECCAK256_ELF).unwrap();
        let program = Arc::new(program);

        let mut executor = MinimalExecutor::new(program.clone(), false, None);
        // executor.debug();
        executor.with_input(&serialize(&5_usize).unwrap());
        for i in 0..5 {
            executor.with_input(&serialize(&vec![i; i]).unwrap());
        }
        executor.execute_chunk();

        let mut interpreter =
            crate::executor::Executor::new(program.clone(), crate::SP1CoreOpts::default());
        interpreter.write_stdin_slice(&serialize(&5_usize).unwrap());
        for i in 0..5 {
            interpreter.write_stdin_slice(&serialize(&vec![i; i]).unwrap());
        }
        interpreter.run_fast().expect("Interpreter failed");

        let interpreter_registers = interpreter
            .state
            .memory
            .registers
            .registers
            .iter()
            .map(|r| r.map(|r| r.value).unwrap_or(0))
            .collect::<Vec<_>>();

        let jit_registers = executor.compiled.registers;

        for i in 0..32 {
            if interpreter_registers[i] != jit_registers[i] {
                eprintln!("interpreter registers: {interpreter_registers:?}");
                eprintln!("jit registers: {jit_registers:?}");
                panic!("JIT and interpreter final registers mismatch at index {i}");
            }
        }

        assert_eq!(
            executor.compiled.pc, interpreter.state.pc,
            "JIT and interpreter final pc mismatch"
        );

        assert_eq!(
            executor.compiled.clk, interpreter.state.clk,
            "JIT and interpreter final clk mismatch"
        );

        assert_eq!(
            executor.compiled.global_clk, interpreter.state.global_clk,
            "JIT and interpreter final global_clk mismatch"
        );
    }

    #[test]
    fn test_chunk_stops_correctly() {
        use bincode::serialize;
        use sp1_jit::{MemValue, MinimalTrace};
        use test_artifacts::KECCAK256_ELF;

        let program = Program::from(&KECCAK256_ELF).unwrap();
        let program = Arc::new(program);

        // Set trace_size to allow a small number of memory reads per chunk
        // Using a small trace size to trigger cuts based on memory reads
        let max_mem_reads = 5;
        let trace_size = max_mem_reads * std::mem::size_of::<MemValue>() as u64;

        let mut executor = MinimalExecutor::new(program.clone(), false, Some(trace_size));
        executor.with_input(&serialize(&5_usize).unwrap());
        for i in 0..5 {
            executor.with_input(&serialize(&vec![i; i]).unwrap());
        }

        let trace = executor.execute_chunk().expect("expected a trace chunk, but got none");
        assert_eq!(trace.clk_start(), 1);
        assert_eq!(trace.pc_start(), program.pc_start_abs);

        // Now the cutting is based on memory reads at 90% threshold
        // We expect the chunk to stop when num_mem_reads reaches 90% of max_mem_reads
        let threshold_reads = (max_mem_reads * 9) / 10; // 90% of max_mem_reads
        assert!(
            trace.num_mem_reads() == threshold_reads,
            "Expected first chunk to have at least {} memory reads (90% threshold), got {}",
            threshold_reads,
            trace.num_mem_reads()
        );

        // Execute another chunk and verify it continues from where the first left off
        let trace2 = executor.execute_chunk().expect("expected a trace chunk, but got none");

        // The second chunk should start from where the first chunk ended
        assert!(
            trace2.clk_start() > trace.clk_start(),
            "Second chunk should start after first chunk"
        );

        // The second chunk should also cut at or above the 90% threshold
        assert!(
            trace2.num_mem_reads() == threshold_reads,
            "Expected second chunk to also have at least {} memory reads (90% threshold), got {}",
            threshold_reads,
            trace2.num_mem_reads()
        );
    }

    #[test]
    fn test_run_fibonacci() {
        use crate::programs::tests::FIBONACCI_ELF;

        run_program_and_compare_end_state(&FIBONACCI_ELF);
    }

    #[test]
    fn test_run_sha256() {
        use test_artifacts::SHA2_ELF;

        run_program_and_compare_end_state(&SHA2_ELF);
    }

    #[test]
    fn test_run_sha_extend() {
        use test_artifacts::SHA_EXTEND_ELF;

        run_program_and_compare_end_state(&SHA_EXTEND_ELF);
    }

    #[test]
    fn test_run_sha_compress() {
        use test_artifacts::SHA_COMPRESS_ELF;

        run_program_and_compare_end_state(&SHA_COMPRESS_ELF);
    }

    #[test]
    fn test_run_keccak_permute() {
        use test_artifacts::KECCAK_PERMUTE_ELF;

        run_program_and_compare_end_state(&KECCAK_PERMUTE_ELF);
    }

    #[test]
    fn test_run_secp256k1_add() {
        use test_artifacts::SECP256K1_ADD_ELF;

        run_program_and_compare_end_state(&SECP256K1_ADD_ELF);
    }

    #[test]
    fn test_run_secp256k1_double() {
        use test_artifacts::SECP256K1_DOUBLE_ELF;

        run_program_and_compare_end_state(&SECP256K1_DOUBLE_ELF);
    }

    #[test]
    fn test_run_secp256r1_add() {
        use test_artifacts::SECP256R1_ADD_ELF;

        run_program_and_compare_end_state(&SECP256R1_ADD_ELF);
    }

    #[test]
    fn test_run_secp256r1_double() {
        use test_artifacts::SECP256R1_DOUBLE_ELF;

        run_program_and_compare_end_state(&SECP256R1_DOUBLE_ELF);
    }

    #[test]
    fn test_run_bls12_381_add() {
        use test_artifacts::BLS12381_ADD_ELF;

        run_program_and_compare_end_state(&BLS12381_ADD_ELF);
    }

    #[test]
    fn test_ed_add() {
        use test_artifacts::ED_ADD_ELF;

        run_program_and_compare_end_state(&ED_ADD_ELF);
    }

    #[test]
    fn test_bn254_add() {
        use test_artifacts::BN254_ADD_ELF;

        run_program_and_compare_end_state(&BN254_ADD_ELF);
    }

    #[test]
    fn test_bn254_double() {
        use test_artifacts::BN254_DOUBLE_ELF;

        run_program_and_compare_end_state(&BN254_DOUBLE_ELF);
    }

    #[test]
    fn test_bn254_mul() {
        use test_artifacts::BN254_MUL_ELF;

        run_program_and_compare_end_state(&BN254_MUL_ELF);
    }

    #[test]
    fn test_uint256_mul() {
        use test_artifacts::UINT256_MUL_ELF;

        run_program_and_compare_end_state(&UINT256_MUL_ELF);
    }

    #[test]
    fn test_bls12_381_fp() {
        use test_artifacts::BLS12381_FP_ELF;

        run_program_and_compare_end_state(&BLS12381_FP_ELF);
    }

    #[test]
    fn test_bls12_381_fp2_mul() {
        use test_artifacts::BLS12381_FP2_MUL_ELF;

        run_program_and_compare_end_state(&BLS12381_FP2_MUL_ELF);
    }

    #[test]
    fn test_bls12_381_fp2_addsub() {
        use test_artifacts::BLS12381_FP2_ADDSUB_ELF;

        run_program_and_compare_end_state(&BLS12381_FP2_ADDSUB_ELF);
    }

    #[test]
    fn test_bn254_fp() {
        use test_artifacts::BN254_FP_ELF;

        run_program_and_compare_end_state(&BN254_FP_ELF);
    }

    #[test]
    fn test_bn254_fp2_addsub() {
        use test_artifacts::BN254_FP2_ADDSUB_ELF;

        run_program_and_compare_end_state(&BN254_FP2_ADDSUB_ELF);
    }

    #[test]
    fn test_bn254_fp2_mul() {
        use test_artifacts::BN254_FP2_MUL_ELF;

        run_program_and_compare_end_state(&BN254_FP2_MUL_ELF);
    }

    #[test]
    fn test_ed_decompress() {
        use test_artifacts::ED_DECOMPRESS_ELF;

        run_program_and_compare_end_state(&ED_DECOMPRESS_ELF);
    }

    #[test]
    fn test_ed25519_verify() {
        use test_artifacts::ED25519_ELF;

        run_program_and_compare_end_state(&ED25519_ELF);
    }

    #[test]
    fn test_ssz_withdrawls() {
        use test_artifacts::SSZ_WITHDRAWALS_ELF;

        run_program_and_compare_end_state(&SSZ_WITHDRAWALS_ELF);
    }

    // #[test]
    // fn test_compare_registers_at_each_timestamp() {
    //     const ELF: Elf = test_artifacts::SHA_EXTEND_ELF;

    //     let program = Program::from(&ELF).unwrap();
    //     let program = Arc::new(program);

    //     let jit_rx = sp1_jit::debug::init_debug_registers();
    //     let interpreter_rx = crate::executor::init_debug_registers();

    //     let jit_handle = std::thread::spawn({
    //         let program = program.clone();
    //         move || {
    //             let mut executor = MinimalExecutor::new(program.clone(), false, true, None);
    //             executor.execute_chunk();
    //             eprintln!("jit final registers={:?}", executor.compiled.registers);
    //             eprintln!("jit final clk={}", executor.compiled.clk);
    //         }
    //     });

    //     let interpreter_handle = std::thread::spawn({
    //         let program = program.clone();
    //         move || {
    //             let mut interpreter =
    //                 crate::executor::Executor::new(program.clone(), Default::default());
    //             interpreter.run_fast();

    //             eprintln!(
    //                 "interpreter final registers={:?}",
    //                 interpreter
    //                     .state
    //                     .memory
    //                     .registers
    //                     .registers
    //                     .iter()
    //                     .map(|r| r.map(|r| r.value).unwrap_or(0))
    //                     .collect::<Vec<_>>()
    //             );

    //             eprintln!("interpreter final clk={}", interpreter.state.clk);
    //         }
    //     });
    //     eprintln!("waiting for threads to finish");

    //     for (cycle, (jit, interpreter)) in
    //         jit_rx.into_iter().zip(interpreter_rx.into_iter()).enumerate()
    //     {
    //         let ((jit_pc, jit_clk, jit), (interpreter_pc, interpreter_clk, interpreter)) =
    //             match (jit, interpreter) {
    //                 (Some(jit), Some(interpreter)) => (jit, interpreter),
    //                 (Some(_), None) => {
    //                     eprintln!("jit=        {:?}", jit);
    //                     eprintln!("interpreter={:?}", interpreter);

    //                     panic!("😨 JIT finished, but interpreter did not");
    //                 }
    //                 (None, Some(_)) => {
    //                     eprintln!("jit=        {:?}", jit);
    //                     eprintln!("interpreter={:?}", interpreter);

    //                     panic!("😨 Interpreter finished, but JIT did not");
    //                 }
    //                 (None, None) => break,
    //             };

    //         if jit_clk != interpreter_clk {
    //             eprintln!("jit_clk={}", jit_clk);
    //             eprintln!("interpreter_clk={}", interpreter_clk);
    //             eprintln!("{},jit=        {:?}", jit_clk, jit);
    //             eprintln!("{},interpreter={:?}", interpreter_clk, interpreter);
    //             panic!("😨 Clk mismatch");
    //         }

    //         if jit_pc != interpreter_pc {
    //             eprintln!("jit_pc={}", jit_pc);
    //             eprintln!("interpreter_pc={}", interpreter_pc);
    //             eprintln!("{},jit=        {:?}", jit_clk, jit);
    //             eprintln!("{},interpreter={:?}", jit_clk, interpreter);
    //             panic!("😨 Pc mismatch");
    //         }

    //         for i in 0..32 {
    //             if jit[i] != interpreter[i] {
    //                 eprintln!("{},jit=        {:?}", jit_clk, jit);
    //                 eprintln!("{},interpreter={:?}", jit_clk, interpreter);
    //                 eprintln!(
    //                     "😨  REGISTER MISMATCH at index: {}, clk = {}, jit[{i}]={:?},
    // interpreter[{i}]={:?}",
    //                     i, jit_clk, jit[i], interpreter[i]
    //                 );
    //                 panic!("😨 Register mismatch");
    //             }
    //         }

    //         eprintln!("registers match at cycle {}", cycle);
    //     }
    // }
}

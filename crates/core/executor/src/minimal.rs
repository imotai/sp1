#![allow(clippy::items_after_statements)]

use crate::memory::MAX_LOG_ADDR;
use sp1_jit::{
    DebugBackend, JitFunction, RiscOperand, RiscRegister, SP1RiscvTranspiler, TraceChunk,
    TranspilerBackend,
};
use std::{collections::VecDeque, sync::Arc};

use crate::{Instruction, Opcode, Program, Register};

mod ecall;
mod hint;
mod precompiles;
mod unconstrained;
mod write;

/// A minimal trace executor.
pub struct MinimalExecutor {
    program: Arc<Program>,
    compiled: Option<JitFunction>,
    debug: bool,
    input: VecDeque<Vec<u8>>,
}

impl MinimalExecutor {
    /// Create a new minimal executor.
    #[must_use]
    pub const fn new(program: Arc<Program>) -> Self {
        Self { program, compiled: None, debug: false, input: VecDeque::new() }
    }

    /// Enable debug mode.
    ///
    /// This method will use a transpiler that prints debug info for each instruction.
    pub fn debug(&mut self) {
        self.debug = true;
    }

    /// Add input to the executor.
    pub fn with_input(&mut self, input: &[u8]) {
        self.input.push_back(input.to_vec());
    }

    /// Execute the program. Returning a trace chunk if the program has not completed.
    pub fn execute_chunk(&mut self) -> Option<TraceChunk> {
        loop {
            if let Some(ref mut compiled) = self.compiled {
                if !self.input.is_empty() {
                    compiled.set_input_buffer(std::mem::take(&mut self.input));
                }

                // SAFETY: The backend is assumed to output valid JIT functions.
                let trace_buf = unsafe { compiled.call()? };

                return Some(TraceChunk::copy_from_bytes(&trace_buf));
            }

            // If the JIT function is not compiled, compile it.
            self.transpile();
        }
    }

    /// Get the number of cycles executed by the JIT function.
    #[must_use]
    pub fn clk(&self) -> u64 {
        self.compiled.as_ref().map_or(0, |c| c.clk)
    }

    /// Reset the JIT function, to start from the beginning of the program.
    pub fn reset(&mut self) {
        if let Some(ref mut compiled) = self.compiled {
            compiled.reset();
        }

        let _ = std::mem::take(&mut self.input);
    }

    /// Transpile the program, saving the JIT function.
    pub fn transpile(&mut self) {
        let mut backend = TranspilerBackend::new(
            self.program.instructions.len(),
            2_u64.pow(MAX_LOG_ADDR as u32) as usize,
            u32::MAX as usize,
            self.program.pc_start_abs,
            self.program.pc_base,
        )
        .expect("Failed to create transpiler backend");

        backend.register_ecall_handler(ecall::sp1_ecall_handler);

        eprintln!("transpiling program");

        if self.debug {
            self.transpile_instructions(DebugBackend::new(backend));
        } else {
            self.transpile_instructions(backend);
        }
    }

    fn transpile_instructions<B: SP1RiscvTranspiler>(&mut self, mut backend: B) {
        backend.trace_clk_start();
        backend.trace_pc_start();
        for (i, instruction) in self.program.instructions.iter().enumerate() {
            backend.start_instr();

            let next_pc = ((i + 1) * 4) as u64 + self.program.pc_base;

            // Add the base amount of cycles for the instruction.
            backend.bump_clk(8);

            match instruction.opcode {
                Opcode::LB
                | Opcode::LH
                | Opcode::LW
                | Opcode::LBU
                | Opcode::LHU
                | Opcode::LD
                | Opcode::LWU => {
                    Self::transpile_load_instruction(&mut backend, instruction);
                }
                Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD => {
                    Self::transpile_store_instruction(&mut backend, instruction);
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
                }
                Opcode::EBREAK | Opcode::UNIMP => {
                    backend.unimp();
                }
                _ => panic!("Invalid instruction: {:?}", instruction.opcode),
            }

            // The following instructions will modify the PC directly,
            // so we don't need to set the PC here.
            if !(instruction.is_branch_instruction()
                || instruction.is_jump_instruction()
                || instruction.opcode == Opcode::AUIPC
                || instruction.is_ecall_instruction())
            {
                backend.set_pc(next_pc);
            }
        }

        let mut finalized = backend.finalize().expect("Failed to finalize function");
        finalized.with_initial_memory_image(self.program.memory_image.clone());

        self.compiled = Some(finalized);
    }

    fn transpile_load_instruction<B: SP1RiscvTranspiler>(
        backend: &mut B,
        instruction: &Instruction,
    ) {
        let (rd, rs1, imm) = instruction.i_type();

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

        backend.trace_mem_value(rd.into());
    }

    fn transpile_store_instruction<B: SP1RiscvTranspiler>(
        backend: &mut B,
        instruction: &Instruction,
    ) {
        let (rs1, rs2, imm) = instruction.s_type();

        // Note: We switch around rs1 and rs2 operaneds to align with the executor.
        match instruction.opcode {
            Opcode::SB => backend.sb(rs2.into(), rs1.into(), imm),
            Opcode::SH => backend.sh(rs2.into(), rs1.into(), imm),
            Opcode::SW => backend.sw(rs2.into(), rs1.into(), imm),
            Opcode::SD => backend.sd(rs2.into(), rs1.into(), imm),
            _ => unreachable!("Invalid store opcode: {:?}", instruction.opcode),
        }
    }

    fn transpile_branch_instruction<B: SP1RiscvTranspiler>(
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

    fn transpile_jump_instruction<B: SP1RiscvTranspiler>(
        backend: &mut B,
        instruction: &Instruction,
    ) {
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

    fn transpile_alu_instruction<B: SP1RiscvTranspiler>(
        backend: &mut B,
        instruction: &Instruction,
    ) {
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

    fn run_program_and_compare_end_state(program: Elf) {
        let program = Program::from(&program).unwrap();
        let program = Arc::new(program);

        let mut interpreter = crate::executor::Executor::new(program.clone(), Default::default());
        let start = std::time::Instant::now();
        interpreter.run_fast().expect("Interpreter failed");
        let interpreter_time = start.elapsed();

        let mut executor = MinimalExecutor::new(program.clone());
        let start = std::time::Instant::now();
        executor.execute_chunk().expect("JIT failed");
        let jit_time = start.elapsed();

        // convert to mhz
        let cycles = executor.compiled.as_ref().expect("JIT not compiled").clk;
        let mhz = cycles as f64 / (jit_time.as_micros() as f64);
        eprintln!("cycles={cycles}");
        eprintln!("jit mhz={mhz} mhz");

        let interpreter_cycles = interpreter.state.clk;
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

        let jit_registers = executor.compiled.as_ref().expect("JIT not compiled").registers;

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
            executor.compiled.as_ref().expect("JIT not compiled").clk,
            interpreter.state.clk,
            "JIT and interpreter final clk mismatch"
        );
    }

    #[test]
    fn test_run_keccak_with_input() {
        use bincode::serialize;
        use test_artifacts::KECCAK256_ELF;

        let program = Program::from(&KECCAK256_ELF).unwrap();
        let program = Arc::new(program);

        let mut executor = MinimalExecutor::new(program.clone());
        // executor.debug();
        executor.with_input(&serialize(&5_usize).unwrap());
        for i in 0..5 {
            executor.with_input(&serialize(&i).unwrap());
        }
        executor.execute_chunk().expect("JIT failed");

        let mut interpreter = crate::executor::Executor::new(program.clone(), Default::default());
        interpreter.write_stdin_slice(&serialize(&5_usize).unwrap());
        for i in 0..5 {
            interpreter.write_stdin_slice(&serialize(&i).unwrap());
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

        let jit_registers = executor.compiled.as_ref().expect("JIT not compiled").registers;

        for i in 0..32 {
            if interpreter_registers[i] != jit_registers[i] {
                eprintln!("interpreter registers: {interpreter_registers:?}");
                eprintln!("jit registers: {jit_registers:?}");
                panic!("JIT and interpreter final registers mismatch at index {i}");
            }
        }

        assert_eq!(
            executor.compiled.as_ref().expect("JIT not compiled").pc,
            interpreter.state.pc,
            "JIT and interpreter final pc mismatch"
        );

        assert_eq!(
            executor.compiled.as_ref().expect("JIT not compiled").clk,
            interpreter.state.clk,
            "JIT and interpreter final clk mismatch"
        );
    }

    #[test]
    fn test_run_fibonacci() {
        use crate::programs::tests::FIBONACCI_ELF;

        run_program_and_compare_end_state(FIBONACCI_ELF);
    }

    #[test]
    fn test_run_sha256() {
        use test_artifacts::SHA2_ELF;

        run_program_and_compare_end_state(SHA2_ELF);
    }

    #[test]
    fn test_run_sha_extend() {
        use test_artifacts::SHA_EXTEND_ELF;

        run_program_and_compare_end_state(SHA_EXTEND_ELF);
    }

    #[test]
    fn test_run_sha_compress() {
        use test_artifacts::SHA_COMPRESS_ELF;

        run_program_and_compare_end_state(SHA_COMPRESS_ELF);
    }

    #[test]
    fn test_run_keccak_permute() {
        use test_artifacts::KECCAK_PERMUTE_ELF;

        run_program_and_compare_end_state(KECCAK_PERMUTE_ELF);
    }

    #[test]
    fn test_run_secp256k1_add() {
        use test_artifacts::SECP256K1_ADD_ELF;

        run_program_and_compare_end_state(SECP256K1_ADD_ELF);
    }

    #[test]
    fn test_run_secp256k1_double() {
        use test_artifacts::SECP256K1_DOUBLE_ELF;

        run_program_and_compare_end_state(SECP256K1_DOUBLE_ELF);
    }

    #[test]
    fn test_run_secp256r1_add() {
        use test_artifacts::SECP256R1_ADD_ELF;

        run_program_and_compare_end_state(SECP256R1_ADD_ELF);
    }

    #[test]
    fn test_run_secp256r1_double() {
        use test_artifacts::SECP256R1_DOUBLE_ELF;

        run_program_and_compare_end_state(SECP256R1_DOUBLE_ELF);
    }

    #[test]
    fn test_run_bls12_381_add() {
        use test_artifacts::BLS12381_ADD_ELF;

        run_program_and_compare_end_state(BLS12381_ADD_ELF);
    }

    #[test]
    fn test_ed_add() {
        use test_artifacts::ED_ADD_ELF;

        run_program_and_compare_end_state(ED_ADD_ELF);
    }

    #[test]
    fn test_bn254_add() {
        use test_artifacts::BN254_ADD_ELF;

        run_program_and_compare_end_state(BN254_ADD_ELF);
    }

    #[test]
    fn test_bn254_double() {
        use test_artifacts::BN254_DOUBLE_ELF;

        run_program_and_compare_end_state(BN254_DOUBLE_ELF);
    }

    #[test]
    fn test_bn254_mul() {
        use test_artifacts::BN254_MUL_ELF;

        run_program_and_compare_end_state(BN254_MUL_ELF);
    }

    #[test]
    fn test_uint256_mul() {
        use test_artifacts::UINT256_MUL_ELF;

        run_program_and_compare_end_state(UINT256_MUL_ELF);
    }

    #[test]
    fn test_bls12_381_fp() {
        use test_artifacts::BLS12381_FP_ELF;

        run_program_and_compare_end_state(BLS12381_FP_ELF);
    }

    #[test]
    fn test_bls12_381_fp2_mul() {
        use test_artifacts::BLS12381_FP2_MUL_ELF;

        run_program_and_compare_end_state(BLS12381_FP2_MUL_ELF);
    }

    #[test]
    fn test_bls12_381_fp2_addsub() {
        use test_artifacts::BLS12381_FP2_ADDSUB_ELF;

        run_program_and_compare_end_state(BLS12381_FP2_ADDSUB_ELF);
    }

    #[test]
    fn test_bn254_fp() {
        use test_artifacts::BN254_FP_ELF;

        run_program_and_compare_end_state(BN254_FP_ELF);
    }

    #[test]
    fn test_bn254_fp2_addsub() {
        use test_artifacts::BN254_FP2_ADDSUB_ELF;

        run_program_and_compare_end_state(BN254_FP2_ADDSUB_ELF);
    }

    #[test]
    fn test_bn254_fp2_mul() {
        use test_artifacts::BN254_FP2_MUL_ELF;

        run_program_and_compare_end_state(BN254_FP2_MUL_ELF);
    }

    #[test]
    fn test_ed_decompress() {
        use test_artifacts::ED_DECOMPRESS_ELF;

        run_program_and_compare_end_state(ED_DECOMPRESS_ELF);
    }

    #[test]
    fn test_ed25519_verify() {
        use test_artifacts::ED25519_ELF;

        run_program_and_compare_end_state(ED25519_ELF);
    }

    #[test]
    fn test_ssz_withdrawls() {
        use test_artifacts::SSZ_WITHDRAWALS_ELF;

        run_program_and_compare_end_state(SSZ_WITHDRAWALS_ELF);
    }

    // #[test]
    // fn test_compare_registers_at_each_timestamp() {
    //     const ELF: Elf = test_artifacts::KECCAK_PERMUTE_ELF;

    //     let program = Program::from(&ELF).unwrap();
    //     let program = Arc::new(program);

    //     let jit_rx = sp1_jit::init_debug_registers();
    //     let interpreter_rx = crate::executor::init_debug_registers();

    //     let jit_handle = std::thread::spawn({
    //         let program = program.clone();
    //         move || {
    //             let mut executor = MinimalExecutor::new(program.clone());
    //             executor.debug();
    //             executor.execute_chunk();

    //             eprintln!(
    //                 "jit final registers={:?}",
    //                 executor.compiled.as_ref().expect("JIT not compiled").registers
    //             );

    //             eprintln!(
    //                 "jit final clk={}",
    //                 executor.compiled.as_ref().expect("JIT not compiled").clk
    //             );
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
    //     // jit_handle.join().unwrap();
    //     // interpreter_handle.join().unwrap();
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
    //                     "😨  REGISTER MISMATCH at index: {}, clk = {}, jit[{i}]={:?}, interpreter[{i}]={:?}",
    //                     i,
    //                     jit_clk,
    //                     jit[i],
    //                     interpreter[i]
    //                 );
    //                 panic!("😨 Register mismatch");
    //             }
    //         }

    //         eprintln!("registers match at cycle {}", cycle);
    //     }
    // }
}

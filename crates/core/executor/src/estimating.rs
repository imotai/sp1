use sp1_jit::MinimalTrace;
use std::sync::Arc;

use crate::{
    syscalls::SyscallCode,
    vm::{
        gas::ReportGenerator,
        memory::CompressedMemory,
        results::{CycleResult, EcallResult, LoadResult, StoreResult},
        syscall::{core_syscall_handler, SyscallRuntime},
        CoreVM,
    },
    ExecutionError, ExecutionReport, Instruction, Opcode, Program, Register,
};

/// A RISC-V VM that uses a [`MinimalTrace`] to create a [`ExecutionRecord`].
pub struct GasEstimatingVM<'a> {
    /// The core VM.
    pub core: CoreVM<'a>,
    /// The addresses that have been touched.
    pub touched_addresses: &'a mut CompressedMemory,
    /// The gas calculator for the VM.
    pub gas_calculator: ReportGenerator,
    /// The index of the hint lens the next shard will use.
    pub hint_lens_idx: usize,
}

impl GasEstimatingVM<'_> {
    /// Execute the program until it halts.
    pub fn execute(&mut self) -> Result<ExecutionReport, ExecutionError> {
        if self.core.is_done() {
            return Ok(self.gas_calculator.generate_report());
        }

        loop {
            match self.execute_instruction()? {
                CycleResult::Done(false) => {}
                CycleResult::TraceEnd | CycleResult::ShardBoundry | CycleResult::Done(true) => {
                    return Ok(self.gas_calculator.generate_report());
                }
            }
        }
    }

    /// Execute the next instruction at the current PC.
    pub fn execute_instruction(&mut self) -> Result<CycleResult, ExecutionError> {
        let instruction = self.core.fetch();
        if instruction.is_none() {
            unreachable!("Fetching the next instruction failed");
        }

        // SAFETY: The instruction is guaranteed to be valid as we checked for `is_none` above.
        let instruction = unsafe { *instruction.unwrap_unchecked() };

        match &instruction.opcode {
            Opcode::ADD
            | Opcode::ADDI
            | Opcode::SUB
            | Opcode::XOR
            | Opcode::OR
            | Opcode::AND
            | Opcode::SLL
            | Opcode::SLLW
            | Opcode::SRL
            | Opcode::SRA
            | Opcode::SRLW
            | Opcode::SRAW
            | Opcode::SLT
            | Opcode::SLTU
            | Opcode::MUL
            | Opcode::MULHU
            | Opcode::MULHSU
            | Opcode::MULH
            | Opcode::MULW
            | Opcode::DIVU
            | Opcode::REMU
            | Opcode::DIV
            | Opcode::REM
            | Opcode::DIVW
            | Opcode::ADDW
            | Opcode::SUBW
            | Opcode::DIVUW
            | Opcode::REMUW
            | Opcode::REMW => {
                self.execute_alu(&instruction);
            }
            Opcode::LB
            | Opcode::LBU
            | Opcode::LH
            | Opcode::LHU
            | Opcode::LW
            | Opcode::LWU
            | Opcode::LD => self.execute_load(&instruction)?,
            Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD => {
                self.execute_store(&instruction)?;
            }
            Opcode::JAL | Opcode::JALR => {
                self.execute_jump(&instruction);
            }
            Opcode::BEQ | Opcode::BNE | Opcode::BLT | Opcode::BGE | Opcode::BLTU | Opcode::BGEU => {
                self.execute_branch(&instruction);
            }
            Opcode::LUI | Opcode::AUIPC => {
                self.execute_utype(&instruction);
            }
            Opcode::ECALL => self.execute_ecall(&instruction)?,
            Opcode::EBREAK | Opcode::UNIMP => {
                unreachable!("Invalid opcode for `execute_instruction`: {:?}", instruction.opcode)
            }
        }

        Ok(self.core.advance())
    }
}

impl<'a> GasEstimatingVM<'a> {
    /// Create a new full-tracing VM from a minimal trace.
    #[tracing::instrument(name = "GasEstimatingVM::new", skip_all)]
    pub fn new<T: MinimalTrace>(
        trace: &'a T,
        program: Arc<Program>,
        touched_addresses: &'a mut CompressedMemory,
    ) -> Self {
        Self {
            core: CoreVM::new(trace, program),
            touched_addresses,
            hint_lens_idx: 0,
            gas_calculator: ReportGenerator::new(trace.clk_start()),
        }
    }

    /// Execute a load instruction.
    ///
    /// This method will update the local memory access for the memory read, the register read,
    /// and the register write.
    ///
    /// It will also emit the memory instruction event and the events for the load instruction.
    pub fn execute_load(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let LoadResult { addr, rd, mr_record, .. } = self.core.execute_load(instruction)?;

        // Ensure the address is aligned to 8 bytes.
        self.touched_addresses.insert(addr & !0b111, true);

        self.gas_calculator.handle_instruction(
            instruction,
            self.core.needs_bump_clk_high(),
            rd == Register::X0,
            self.core.needs_state_bump(instruction),
        );

        self.gas_calculator.handle_mem_event(addr, mr_record.prev_timestamp);

        Ok(())
    }

    /// Execute a store instruction.
    ///
    /// This method will update the local memory access for the memory read, the register read,
    /// and the register write.
    ///
    /// It will also emit the memory instruction event and the events for the store instruction.
    pub fn execute_store(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let StoreResult { addr, mw_record, .. } = self.core.execute_store(instruction)?;

        // Ensure the address is aligned to 8 bytes.
        self.touched_addresses.insert(addr & !0b111, true);

        self.gas_calculator.handle_instruction(
            instruction,
            self.core.needs_bump_clk_high(),
            false, // store instruction, no load of x0
            self.core.needs_state_bump(instruction),
        );

        self.gas_calculator.handle_mem_event(addr, mw_record.prev_timestamp);

        Ok(())
    }

    /// Execute an ALU instruction and emit the events.
    #[inline]
    pub fn execute_alu(&mut self, instruction: &Instruction) {
        let _ = self.core.execute_alu(instruction);

        self.gas_calculator.handle_instruction(
            instruction,
            self.core.needs_bump_clk_high(),
            false, // alu instruction, no load of x0
            self.core.needs_state_bump(instruction),
        );
    }

    /// Execute a jump instruction and emit the events.
    #[inline]
    pub fn execute_jump(&mut self, instruction: &Instruction) {
        let _ = self.core.execute_jump(instruction);

        self.gas_calculator.handle_instruction(
            instruction,
            self.core.needs_bump_clk_high(),
            false, // jump instruction, no load of x0
            self.core.needs_state_bump(instruction),
        );
    }

    /// Execute a branch instruction and emit the events.
    #[inline]
    pub fn execute_branch(&mut self, instruction: &Instruction) {
        let _ = self.core.execute_branch(instruction);

        self.gas_calculator.handle_instruction(
            instruction,
            self.core.needs_bump_clk_high(),
            false, // branch instruction, no load of x0
            self.core.needs_state_bump(instruction),
        );
    }

    /// Execute a U-type instruction and emit the events.   
    #[inline]
    pub fn execute_utype(&mut self, instruction: &Instruction) {
        let _ = self.core.execute_utype(instruction);

        self.gas_calculator.handle_instruction(
            instruction,
            self.core.needs_bump_clk_high(),
            false, // u-type instruction, no load of x0
            self.core.needs_state_bump(instruction),
        );
    }

    /// Execute an ecall instruction and emit the events.
    #[inline]
    pub fn execute_ecall(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let EcallResult { code, .. } =
            CoreVM::execute_ecall(self, instruction, core_syscall_handler)?;

        if code == SyscallCode::HINT_LEN {
            self.hint_lens_idx += 1;
        }

        if code.should_send() == 1 {
            if self.core.is_retained_syscall(code) {
                self.gas_calculator.handle_retained_syscall(code);
            } else {
                self.gas_calculator.syscall_sent(code);
            }
        }

        self.gas_calculator.handle_instruction(
            instruction,
            self.core.needs_bump_clk_high(),
            false, // ecall instruction, no load of x0
            self.core.needs_state_bump(instruction),
        );

        Ok(())
    }
}

impl<'a> SyscallRuntime<'a> for GasEstimatingVM<'a> {
    fn core(&self) -> &CoreVM<'a> {
        &self.core
    }

    fn core_mut(&mut self) -> &mut CoreVM<'a> {
        &mut self.core
    }

    fn mw_slice(&mut self, addr: u64, len: usize) -> Vec<crate::events::MemoryWriteRecord> {
        let records = self.core_mut().mw_slice(addr, len);

        for (i, record) in records.iter().enumerate() {
            self.gas_calculator.handle_mem_event(addr + i as u64 * 8, record.prev_timestamp);
        }

        records
    }

    fn mr_slice(&mut self, addr: u64, len: usize) -> Vec<crate::events::MemoryReadRecord> {
        let records = self.core_mut().mr_slice(addr, len);

        for (i, record) in records.iter().enumerate() {
            self.gas_calculator.handle_mem_event(addr + i as u64 * 8, record.prev_timestamp);
        }

        records
    }
}

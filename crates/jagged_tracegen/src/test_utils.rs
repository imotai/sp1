//! Common test utilities shared across test modules.
//! TODO: This should only be built in tests.

pub mod tracegen_setup {
    use sp1_core_executor::{ExecutionRecord, Program, SP1Context, SP1CoreOpts};
    use sp1_core_machine::{executor::MachineExecutor, io::SP1Stdin, riscv::RiscvAir};
    use sp1_hypercube::Machine;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    use csl_utils::Felt;

    pub const FIBONACCI_ELF: &[u8] =
        include_bytes!("../../prover_components/programs/fibonacci/riscv64im-succinct-zkvm-elf");

    pub const KECCAK_ELF: &[u8] =
        include_bytes!("../../prover_components/programs/keccak/riscv64im-succinct-zkvm-elf");

    pub const CORE_MAX_LOG_ROW_COUNT: u32 = 22;
    pub const LOG_STACKING_HEIGHT: u32 = 21;

    /// Which test program to execute for trace generation.
    #[derive(Debug, Clone, Copy, Default)]
    pub enum TestProgram {
        /// Fibonacci program with input 8000 (~96_000 cycles)
        #[default]
        Fibonacci,
        /// Keccak program (hash computation)
        Keccak,
    }

    impl TestProgram {
        /// Returns the ELF bytes for this program.
        pub fn elf(&self) -> &'static [u8] {
            match self {
                TestProgram::Fibonacci => FIBONACCI_ELF,
                TestProgram::Keccak => KECCAK_ELF,
            }
        }

        /// Returns the stdin for this program.
        pub fn stdin(&self) -> SP1Stdin {
            let mut stdin = SP1Stdin::new();
            match self {
                TestProgram::Fibonacci => {
                    stdin.write(&8_000u32);
                }
                TestProgram::Keccak => {
                    // Keccak program expects input data to hash
                    let input: Vec<u8> = vec![0u8; 1024];
                    stdin.write_slice(&input);
                }
            }
            stdin
        }

        /// Returns the program name for error messages.
        pub fn name(&self) -> String {
            match self {
                TestProgram::Fibonacci => "Fibonacci".to_string(),
                TestProgram::Keccak => "Keccak".to_string(),
            }
        }

        /// Returns the number of records to skip before returning the desired one.
        /// Some programs have initialization shards that aren't representative.
        pub fn records_to_skip(&self) -> usize {
            match self {
                TestProgram::Fibonacci => 0,
                TestProgram::Keccak => 1, // Skip first record (initialization)
            }
        }
    }

    /// Get a core trace for proving by executing a program and taking the first record.
    ///
    /// This implementation directly executes the specified ELF to generate
    /// execution records.
    ///
    /// Returns (machine, record, program) for use in core execution tracegen tests.
    ///
    /// Note: This generates ExecutionRecord, not recursion/compression records.
    pub async fn setup() -> (Machine<Felt, RiscvAir<Felt>>, ExecutionRecord, Arc<Program>) {
        setup_with_program(TestProgram::default()).await
    }

    /// Get a core trace for proving by executing the specified program.
    ///
    /// Returns (machine, record, program) for use in core execution tracegen tests.
    pub async fn setup_with_program(
        test_program: TestProgram,
    ) -> (Machine<Felt, RiscvAir<Felt>>, ExecutionRecord, Arc<Program>) {
        // 1. Load program from ELF
        let program = Arc::new(Program::from(test_program.elf()).unwrap_or_else(|_| {
            panic!("Failed to load {} ELF - file may be corrupted", test_program.name())
        }));

        // 2. Create stdin with program-specific input
        let stdin = test_program.stdin();

        // 3. Create executor and channel
        let opts = SP1CoreOpts { global_dependencies_opt: true, ..Default::default() };
        let executor = MachineExecutor::<Felt>::new(
            2 * 1024 * 1024 * 1024, // 2GB buffer
            2,                      // 2 workers
            opts,
        );
        let (records_tx, mut records_rx) = mpsc::unbounded_channel();

        // 4. Execute program (spawns async, sends records to channel)
        let context = SP1Context::default();
        executor
            .execute(program.clone(), stdin, context, records_tx)
            .await
            .unwrap_or_else(|_| panic!("{} program execution failed", test_program.name()));

        // 5. Skip initial records if needed, then collect the desired record
        for _ in 0..test_program.records_to_skip() {
            let _ = records_rx
                .recv()
                .await
                .expect("Not enough execution records - executor may have failed");
        }
        let (record, _permit) = records_rx
            .recv()
            .await
            .expect("No execution records received - executor may have failed");

        // 6. Get machine
        let machine = RiscvAir::<Felt>::machine();

        (machine, record, program)
    }
}

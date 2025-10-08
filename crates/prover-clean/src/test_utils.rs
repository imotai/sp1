//! Common test utilities shared across test modules.

#[cfg(test)]
pub mod tracegen_setup {
    use sp1_core_executor::{ExecutionRecord, Program, SP1Context, SP1CoreOpts};
    use sp1_core_machine::{executor::MachineExecutor, io::SP1Stdin, riscv::RiscvAir};
    use sp1_hypercube::Machine;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    use crate::config::Felt;

    pub const FIBONACCI_ELF: &[u8] =
        include_bytes!("../../prover/programs/fibonacci/riscv64im-succinct-zkvm-elf");

    pub const CORE_MAX_LOG_ROW_COUNT: u32 = 22;
    pub const CORE_MAX_TRACE_SIZE: u32 = 1 << 29;
    pub const LOG_STACKING_HEIGHT: u32 = 21;

    /// Setup core execution test data by executing fibonacci program.
    ///
    /// This implementation directly executes the Fibonacci ELF to generate
    /// execution records.
    ///
    /// Returns (machine, record, program) for use in core execution tracegen tests.
    ///
    /// Note: This generates ExecutionRecord, not recursion/compression records.
    pub async fn setup() -> (Machine<Felt, RiscvAir<Felt>>, ExecutionRecord, Arc<Program>) {
        // 1. Load program from ELF
        let program = Arc::new(
            Program::from(FIBONACCI_ELF)
                .expect("Failed to load Fibonacci ELF - file may be corrupted"),
        );

        // 2. Create stdin with fibonacci input
        let mut stdin = SP1Stdin::new();
        stdin.write(&800_000u32);

        // 3. Create executor and channel
        let opts = SP1CoreOpts::default();
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
            .expect("Fibonacci program execution failed");

        // 5. Collect first record
        let (record, _permit) = records_rx
            .recv()
            .await
            .expect("No execution records received - executor may have failed");

        // 6. Get machine
        let machine = RiscvAir::<Felt>::machine();

        (machine, record, program)
    }
}

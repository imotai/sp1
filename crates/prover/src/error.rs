use sp1_core_machine::executor::MachineExecutorError;
use sp1_stark::prover::MachineProverError;
use thiserror::Error;
use tokio::sync::oneshot;

#[derive(Debug, Error)]
pub enum SP1ProverError {
    // Core prover error.
    #[error("Machine prover error: {0}")]
    CoreProverError(MachineProverError),
    #[error("Compilation error")]
    CompilationError,
    // Core executor error.
    #[error("Core executor error: {0}")]
    CoreExecutorError(MachineExecutorError),
    // Recursion prover error.
    #[error("Recursion prover error: {0}")]
    RecursionProverError(MachineProverError),
    /// Recursion program error.
    #[error("Recursion program error: {0}")]
    RecursionProgramError(#[from] RecursionProgramError),
}

#[derive(Debug, Error)]
pub enum RecursionProgramError {
    #[error("Compilation error")]
    CompilationError(#[from] oneshot::error::RecvError),
    // Recursion witness error.
    #[error("Invalid record shape for shard chips")]
    InvalidRecordShape,
    #[error("Task was aborted")]
    TaskAborted,
}

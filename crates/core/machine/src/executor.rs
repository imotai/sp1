//! Execution utilities for tracing and generating execution records.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use slop_algebra::PrimeField32;
use sp1_core_executor::Program;
use sp1_core_executor::{
    events::MemoryRecord, ExecutionError, ExecutionRecord, SP1CoreOpts, TracingVM,
};
use sp1_hypercube::air::PROOF_NONCE_NUM_WORDS;
use sp1_jit::MinimalTrace;
use tracing::Level;

/// The output of the machine executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOutput {
    pub public_value_stream: Vec<u8>,
    pub cycles: u64,
}

/// Trace a single [`MinimalTrace`] (corresponding to a shard) and return the execution record.
///
/// This is the core tracing function that converts a minimal trace into a full execution record
/// with all the events needed for proving.
#[tracing::instrument(
    level = Level::DEBUG,
    name = "trace_chunk",
    skip_all,
)]
pub fn trace_chunk<F: PrimeField32>(
    program: Arc<Program>,
    opts: SP1CoreOpts,
    chunk: impl MinimalTrace,
    proof_nonce: [u32; PROOF_NONCE_NUM_WORDS],
    mut record: ExecutionRecord,
) -> Result<(bool, ExecutionRecord, [MemoryRecord; 32]), ExecutionError> {
    let mut vm = TracingVM::new(&chunk, program, opts, proof_nonce, &mut record);
    let status = vm.execute()?;
    tracing::trace!("chunk ended at clk: {}", vm.core.clk());
    tracing::trace!("chunk ended at pc: {}", vm.core.pc());

    let pv = vm.public_values();

    // Handle the case where `COMMIT` or `COMMIT_DEFERRED_PROOFS` happens across last two shards.
    //
    // todo: does this actually work in the new regieme? what if the shard is stopped due to the clk
    // limit? if so, does that mean this could be wrong? its unclear!
    if status.is_shard_boundry() && (pv.commit_syscall == 1 || pv.commit_deferred_syscall == 1) {
        tracing::trace!("commit syscall or commit deferred proofs across last two shards");

        loop {
            // Execute until we get a done status.
            if vm.execute()?.is_done() {
                let pv = *vm.public_values();

                // Update the record.
                vm.record.public_values.commit_syscall = 1;
                vm.record.public_values.commit_deferred_syscall = 1;
                vm.record.public_values.committed_value_digest = pv.committed_value_digest;
                vm.record.public_values.deferred_proofs_digest = pv.deferred_proofs_digest;

                break;
            }
        }
    }

    // Finalize the public values
    vm.record.finalize_public_values::<F>(true);

    let registers = *vm.core.registers();
    drop(vm);
    Ok((status.is_done(), record, registers))
}

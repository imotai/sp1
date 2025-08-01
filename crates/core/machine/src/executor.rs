use std::{
    fs::File as StdFile,
    io::{self, SeekFrom},
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use slop_algebra::PrimeField32;
use slop_futures::queue::WorkerQueue;
use sp1_core_executor::{
    subproof::NoOpSubproofVerifier, ExecutionError, ExecutionRecord, ExecutionReport,
    ExecutionState, Executor, Program, SP1Context, SP1CoreOpts,
};
use sp1_stark::{
    air::PublicValues,
    prover::{MemoryPermit, MemoryPermitting},
    Machine, MachineRecord,
};
use thiserror::Error;
use tokio::{fs::File, io::AsyncSeekExt, sync::mpsc};
use tracing::Span;

use crate::{io::SP1Stdin, riscv::RiscvAir, utils::concurrency::AsyncTurn};

pub struct MachineExecutor<F: PrimeField32> {
    num_record_workers: usize,
    opts: SP1CoreOpts,
    machine: Machine<F, RiscvAir<F>>,
    memory: MemoryPermitting,
    _marker: std::marker::PhantomData<F>,
}

impl<F: PrimeField32> MachineExecutor<F> {
    pub fn new(record_buffer_size: u64, num_record_workers: usize, opts: SP1CoreOpts) -> Self {
        let machine = RiscvAir::<F>::machine();
        let memory = MemoryPermitting::new(record_buffer_size);

        Self { num_record_workers, opts, machine, memory, _marker: PhantomData }
    }

    pub async fn execute(
        &self,
        program: Arc<Program>,
        stdin: SP1Stdin,
        context: SP1Context<'static>,
        record_tx: mpsc::UnboundedSender<(ExecutionRecord, Option<MemoryPermit>)>,
    ) -> Result<ExecutionOutput, MachineExecutorError> {
        // Spawn the record generation tasks.
        //
        // todo: memory permit this as we know the opcode counts up front.
        let mut record_worker_channels = Vec::with_capacity(self.num_record_workers);
        let mut handles = Vec::new();
        for _ in 0..self.num_record_workers {
            let (tx, mut rx) = mpsc::unbounded_channel::<RecordTask>();
            record_worker_channels.push(tx);
            let machine = self.machine.clone();
            let opts = self.opts.clone();
            let memory = self.memory.clone();
            let handle = tokio::task::spawn(async move {
                while let Some(task) = rx.recv().await {
                    let RecordTask {
                        index,
                        checkpoint_file,
                        done,
                        program,
                        record_gen_sync,
                        report,
                        state,
                        deferred,
                        record_tx,
                        // TODO: Use the span.
                        span: _,
                    } = task;

                    // Acquire a memory permit for the expected size of the records.
                    let expected_record_size = report.total_record_size();

                    // todo(n): This does not properly account for deferred records.
                    let memory_permit = memory
                        .acquire(expected_record_size)
                        .await
                        .expect("failed to acquire memory permit");

                    let (mut record, checkpoint_file) = tokio::task::spawn_blocking({
                        let opts = opts.clone();
                        move || {
                            tracing::debug_span!("trace checkpoint").in_scope(|| {
                                let (records, _) =
                                    trace_checkpoint(program.clone(), &checkpoint_file, opts);
                                (records, checkpoint_file)
                            })
                        }
                    })
                    .await
                    .expect("failed to trace checkpoint");

                    let mut checkpoint_file = File::from_std(checkpoint_file);
                    checkpoint_file
                        .seek(SeekFrom::Start(0))
                        .await
                        .expect("failed to seek to start of tempfile");

                    // Wait for our turn to update the state.
                    let _turn_guard = record_gen_sync.wait_for_turn(index).await;

                    // Update the public values & prover state for the shards which contain
                    // "cpu events".
                    let mut deferred_reccords = {
                        let mut state = state.lock().unwrap();

                        state.shard += 1;
                        state.execution_shard = record.public_values.execution_shard;
                        state.next_execution_shard = record.public_values.execution_shard + 1;
                        state.pc_start = record.public_values.pc_start;
                        state.next_pc = record.public_values.next_pc;
                        state.initial_timestamp = record.public_values.initial_timestamp;
                        state.last_timestamp = record.public_values.last_timestamp;

                        let initial_timestamp_high = (state.initial_timestamp >> 24) as u32;
                        let initial_timestamp_low = (state.initial_timestamp & 0xFFFFFF) as u32;
                        let last_timestamp_high = (state.last_timestamp >> 24) as u32;
                        let last_timestamp_low = (state.last_timestamp & 0xFFFFFF) as u32;
                        if initial_timestamp_high == last_timestamp_high {
                            state.is_timestamp_high_eq = 1;
                        } else {
                            state.is_timestamp_high_eq = 0;
                            state.inv_timestamp_high = (F::from_canonical_u32(last_timestamp_high)
                                - F::from_canonical_u32(initial_timestamp_high))
                            .inverse()
                            .as_canonical_u32();
                        }

                        if initial_timestamp_low == last_timestamp_low {
                            state.is_timestamp_low_eq = 1;
                        } else {
                            state.is_timestamp_low_eq = 0;
                            state.inv_timestamp_low = (F::from_canonical_u32(last_timestamp_low)
                                - F::from_canonical_u32(initial_timestamp_low))
                            .inverse()
                            .as_canonical_u32();
                        }

                        if state.committed_value_digest == [0u32; 8] {
                            state.committed_value_digest =
                                record.public_values.committed_value_digest;
                        }
                        if state.deferred_proofs_digest == [0u32; 8] {
                            state.deferred_proofs_digest =
                                record.public_values.deferred_proofs_digest;
                        }

                        record.public_values = *state;
                        state.prev_exit_code = record.public_values.exit_code;
                        state.initial_timestamp = record.public_values.last_timestamp;

                        // Defer events that are too expensive to include in every shard.
                        let mut deferred = deferred.lock().unwrap();
                        deferred.append(&mut record.defer(&opts.retained_events_presets));

                        let can_pack_global_memory = done
                            && record.estimated_trace_area
                                < opts.split_opts.combine_memory_threshold.0
                            && deferred.global_memory_initialize_events.len()
                                < opts.split_opts.combine_memory_threshold.1
                            && deferred.global_memory_finalize_events.len()
                                < opts.split_opts.combine_memory_threshold.1;

                        // See if any deferred shards are ready to be committed to.
                        let mut deferred_reccords = deferred.split(
                            done,
                            can_pack_global_memory.then_some(&mut *record),
                            opts.split_opts,
                        );
                        tracing::debug!("deferred {} records", deferred_reccords.len());

                        // Update the public values & prover state for the shards which do not
                        // contain "cpu events" before committing to them.
                        state.execution_shard = state.next_execution_shard;
                        for record in deferred_reccords.iter_mut() {
                            state.shard += 1;
                            state.previous_init_addr_word =
                                record.public_values.previous_init_addr_word;
                            state.last_init_addr_word = record.public_values.last_init_addr_word;
                            state.previous_finalize_addr_word =
                                record.public_values.previous_finalize_addr_word;
                            state.last_finalize_addr_word =
                                record.public_values.last_finalize_addr_word;
                            state.pc_start = state.next_pc;
                            state.last_timestamp = state.initial_timestamp;
                            state.is_timestamp_high_eq = 1;
                            state.is_timestamp_low_eq = 1;
                            state.next_execution_shard = state.execution_shard;
                            record.public_values = *state;
                        }

                        deferred_reccords
                    };

                    // Generate the dependencies.
                    tokio::task::spawn_blocking({
                        let machine = machine.clone();
                        move || {
                            let mut record = vec![*record];
                            machine.generate_dependencies(&mut record, None);
                            machine.generate_dependencies(&mut deferred_reccords, None);

                            // Send the records to the output channel.
                            record_tx.send((record.pop().unwrap(), Some(memory_permit))).unwrap();

                            // If there are deferred records, send them to the output channel.
                            for record in deferred_reccords {
                                record_tx.send((record, None)).unwrap();
                            }
                        }
                    })
                    .await
                    .expect("failed to send records");
                }
            });

            handles.push(handle);
        }

        // Initialize the record generation state.
        let record_gen_sync = AsyncTurn::new();
        let state = Arc::new(Mutex::new(PublicValues::<u32, u64, u64, u32>::default().reset()));
        let deferred = Arc::new(Mutex::new(ExecutionRecord::new(program.clone())));
        let record_worker_channels = Arc::new(WorkerQueue::new(record_worker_channels));

        // Setup the runtime.
        let mut runtime =
            Box::new(Executor::with_context(program.clone(), self.opts.clone(), context));
        runtime.write_vecs(&stdin.buffer);
        for proof in stdin.proofs.iter() {
            let (proof, vk) = proof.clone();
            runtime.write_proof(proof, vk);
        }

        // Generate checkpoints until the execution is done.
        let mut index = 0;
        let mut done = false;
        while !done {
            // Send and receive ownership of `runtime: Box<Executor<'_>>`.
            // The `.unwrap()` propagates panics from `generate_checkpoint`.
            let checkpoint_result;
            (runtime, checkpoint_result) = tokio::task::spawn_blocking(move || {
                let res = generate_checkpoint(&mut runtime);
                (runtime, res)
            })
            .await
            .map_err(MachineExecutorError::ExecutorPanicked)?;

            match checkpoint_result {
                Ok((checkpoint_file, report, is_done)) => {
                    // Update the finished flag.
                    done = is_done;
                    // Create a new record generation task.
                    let record_task = RecordTask {
                        index,
                        checkpoint_file,
                        done,
                        program: program.clone(),
                        report,
                        record_gen_sync: record_gen_sync.clone(),
                        state: state.clone(),
                        deferred: deferred.clone(),
                        record_tx: record_tx.clone(),
                        span: tracing::debug_span!("execute record"),
                    };

                    // Send the checkpoint to the record generation worker.
                    let record_worker = record_worker_channels
                        .clone()
                        .pop()
                        .await
                        .expect("failed to pop record worker from channel");

                    // Send the task to the worker.
                    record_worker
                        .send(record_task)
                        .map_err(|_| MachineExecutorError::ExecutorClosed)?;

                    // Increment the index.
                    index += 1;
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        // Execution is done, send the output to the sender.
        let public_value_stream = runtime.state.public_values_stream;
        let cycles = runtime.state.global_clk;

        Ok(ExecutionOutput { public_value_stream, cycles })
    }
}

#[derive(Error, Debug)]
pub enum MachineExecutorError {
    #[error("Failed to execute program: {0}")]
    ExecutionError(ExecutionError),
    #[error("IO error: {0}")]
    IoError(io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(bincode::Error),
    #[error("Executor is already closed")]
    ExecutorClosed,
    #[error("Task failed: {0:?}")]
    ExecutorPanicked(#[from] tokio::task::JoinError),
    #[error("Failed to send record to prover channel")]
    ProverChannelClosed,
}

/// The output of the machine executor.
pub struct ExecutionOutput {
    pub public_value_stream: Vec<u8>,
    pub cycles: u64,
}

struct RecordTask {
    index: usize,
    checkpoint_file: StdFile,
    done: bool,
    program: Arc<Program>,
    report: ExecutionReport,
    record_gen_sync: AsyncTurn,
    state: Arc<Mutex<PublicValues<u32, u64, u64, u32>>>,
    deferred: Arc<Mutex<ExecutionRecord>>,
    record_tx: mpsc::UnboundedSender<(ExecutionRecord, Option<MemoryPermit>)>,
    #[allow(unused)]
    span: Span,
}

/// Trace a checkpoint.
fn trace_checkpoint(
    program: Arc<Program>,
    file: &StdFile,
    opts: SP1CoreOpts,
) -> (Box<ExecutionRecord>, ExecutionReport) {
    let noop = NoOpSubproofVerifier;

    let mut reader = std::io::BufReader::new(file);
    let state: ExecutionState =
        bincode::deserialize_from(&mut reader).expect("failed to deserialize state");
    let mut runtime = Executor::recover(program, state, opts);

    // We already passed the deferred proof verifier when creating checkpoints, so the proofs were
    // already verified. So here we use a noop verifier to not print any warnings.
    runtime.subproof_verifier = Some(Arc::new(noop));

    // Execute from the checkpoint.
    let (mut record, done) = runtime.execute_record(true).unwrap();
    let pv = record.public_values;

    // Handle the case where the COMMIT happens across the last two shards.
    if !done
        && (pv.committed_value_digest.iter().any(|v| *v != 0)
            || pv.deferred_proofs_digest.iter().any(|v| *v != 0))
    {
        // We turn off the `print_report` flag to avoid modifying the report.
        runtime.print_report = false;
        let (_, next_pv, _) = runtime.execute_state(true).unwrap();
        record.public_values.committed_value_digest = next_pv.committed_value_digest;
        record.public_values.deferred_proofs_digest = next_pv.deferred_proofs_digest;
    }

    (record, runtime.report)
}

fn generate_checkpoint(
    runtime: &mut Executor,
) -> Result<(StdFile, ExecutionReport, bool), MachineExecutorError> {
    // Ensure the report is counted.
    runtime.print_report = true;

    // Execute the runtime until we reach a checkpoint.
    let (checkpoint, _, done) =
        runtime.execute_state(false).map_err(MachineExecutorError::ExecutionError)?;

    let report = std::mem::take(&mut runtime.report);

    // Save the checkpoint to a temp file.
    let mut checkpoint_file = tempfile::tempfile().map_err(MachineExecutorError::IoError)?;
    checkpoint.save(&mut checkpoint_file).map_err(MachineExecutorError::IoError)?;

    Ok((checkpoint_file, report, done))
}

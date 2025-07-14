use std::{
    fs::File,
    io::{self, Seek, SeekFrom},
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use futures::stream::{AbortHandle, AbortRegistration};
use slop_algebra::PrimeField32;
use slop_futures::{handle::TaskHandle, queue::WorkerQueue};
use sp1_core_executor::{
    subproof::NoOpSubproofVerifier, ExecutionError, ExecutionRecord, ExecutionReport,
    ExecutionState, Executor, Program, SP1Context, SP1CoreOpts,
};
use sp1_stark::{air::PublicValues, Machine, MachineRecord};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tracing::Span;

use crate::{io::SP1Stdin, riscv::RiscvAir, utils::concurrency::TurnBasedSync};

pub struct MachineExecutor<F: PrimeField32> {
    task_tx: mpsc::UnboundedSender<ExecuteTask>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: PrimeField32> MachineExecutor<F> {
    pub fn execute(
        &self,
        program: Arc<Program>,
        stdin: SP1Stdin,
        context: SP1Context<'static>,
        record_tx: mpsc::Sender<ExecutionRecord>,
    ) -> TaskHandle<ExecutionOutput, MachineExecutorError> {
        // Create a span for this task.
        let span = tracing::Span::current();
        // Create a channel for the output.
        let (output_tx, output_rx) = oneshot::channel();
        // Create an abortable task.
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let task =
            ExecuteTask { program, stdin, context, output_tx, record_tx, abort_registration, span };
        self.task_tx.send(task).unwrap();
        TaskHandle::new(output_rx, abort_handle)
    }
}

pub struct MachineExecutorBuilder<F: PrimeField32> {
    machine: Machine<F, RiscvAir<F>>,
    num_record_workers: usize,
    opts: SP1CoreOpts,
}

#[derive(Error, Debug)]
pub enum MachineExecutorError {
    #[error("failed to execute program: {0}")]
    ExecutionError(ExecutionError),
    #[error("io error: {0}")]
    IoError(io::Error),
    #[error("serialization error: {0}")]
    SerializationError(bincode::Error),
    // The task was aborted.
    #[error("task was aborted")]
    TaskAborted,
    #[error("executor is already closed")]
    ExecutorClosed,
    // Task failed.
    #[error("task failed: {0}")]
    ExecutorPanicked(#[from] oneshot::error::RecvError),
}

#[allow(dead_code)]
struct ExecuteTask {
    program: Arc<Program>,
    stdin: SP1Stdin,
    context: SP1Context<'static>,
    record_tx: mpsc::Sender<ExecutionRecord>,
    output_tx: oneshot::Sender<Result<ExecutionOutput, MachineExecutorError>>,
    abort_registration: AbortRegistration,
    span: Span,
}

pub struct ExecutionOutput {
    pub public_value_stream: Vec<u8>,
    pub cycles: u64,
}

#[allow(dead_code)]
struct RecordTask {
    index: usize,
    checkpoint_file: File,
    done: bool,
    program: Arc<Program>,
    record_gen_sync: Arc<TurnBasedSync>,
    state: Arc<Mutex<PublicValues<u32, u64, u64, u32>>>,
    deferred: Arc<Mutex<ExecutionRecord>>,
    record_tx: mpsc::Sender<ExecutionRecord>,
    abort_handle: AbortHandle,
    span: Span,
}

pub fn trace_checkpoint(
    program: Arc<Program>,
    file: &File,
    opts: SP1CoreOpts,
) -> (Vec<ExecutionRecord>, ExecutionReport) {
    let noop = NoOpSubproofVerifier;

    let mut reader = std::io::BufReader::new(file);
    let state: ExecutionState =
        bincode::deserialize_from(&mut reader).expect("failed to deserialize state");
    let mut runtime = Executor::recover(program, state, opts);

    // We already passed the deferred proof verifier when creating checkpoints, so the proofs were
    // already verified. So here we use a noop verifier to not print any warnings.
    runtime.subproof_verifier = Some(Arc::new(noop));

    // Execute from the checkpoint.
    let (records, done) = runtime.execute_record(true).unwrap();

    let mut records = records.into_iter().map(|r| *r).collect::<Vec<_>>();
    let pv = records.last().unwrap().public_values;

    // Handle the case where the COMMIT happens across the last two shards.
    if !done
        && (pv.committed_value_digest.iter().any(|v| *v != 0)
            || pv.deferred_proofs_digest.iter().any(|v| *v != 0))
    {
        // We turn off the `print_report` flag to avoid modifying the report.
        runtime.print_report = false;
        let (_, next_pv, _) = runtime.execute_state(true).unwrap();
        for record in records.iter_mut() {
            record.public_values.committed_value_digest = next_pv.committed_value_digest;
            record.public_values.deferred_proofs_digest = next_pv.deferred_proofs_digest;
        }
    }

    (records, runtime.report)
}

impl<F: PrimeField32> MachineExecutorBuilder<F> {
    pub fn new(opts: SP1CoreOpts, num_record_workers: usize) -> Self {
        let machine = RiscvAir::machine();

        Self { machine, num_record_workers, opts }
    }

    pub fn num_workers(&mut self, num_record_workers: usize) -> &mut Self {
        self.num_record_workers = num_record_workers;
        self
    }

    pub fn build(&mut self) -> MachineExecutor<F> {
        let (task_tx, mut task_rx) = mpsc::unbounded_channel::<ExecuteTask>();

        // Spawn the record generation tasks and initialize the channels for them.
        let mut record_worker_channels = Vec::with_capacity(self.num_record_workers);
        for _ in 0..self.num_record_workers {
            let (tx, mut rx) = mpsc::unbounded_channel::<RecordTask>();
            record_worker_channels.push(tx);
            let machine = self.machine.clone();
            let opts = self.opts.clone();
            tokio::task::spawn_blocking(move || {
                while let Some(task) = rx.blocking_recv() {
                    let RecordTask {
                        index,
                        mut checkpoint_file,
                        done,
                        program,
                        record_gen_sync,
                        state,
                        deferred,
                        record_tx,
                        abort_handle,
                        // TODO: Use the span.
                        span: _,
                    } = task;
                    if abort_handle.is_aborted() {
                        continue;
                    }

                    let (mut records, _) =
                        tracing::debug_span!("trace checkpoint").in_scope(|| {
                            trace_checkpoint(program.clone(), &checkpoint_file, opts.clone())
                        });

                    checkpoint_file
                        .seek(SeekFrom::Start(0))
                        .expect("failed to seek to start of tempfile");

                    // Wait for our turn to update the state.
                    record_gen_sync.wait_for_turn(index);

                    // Update the public values & prover state for the shards which contain
                    // "cpu events".
                    let mut state = state.lock().unwrap();
                    for record in records.iter_mut() {
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
                    }

                    // Defer events that are too expensive to include in every shard.
                    let mut deferred = deferred.lock().unwrap();
                    for record in records.iter_mut() {
                        deferred.append(&mut record.defer(&opts.retained_events_presets));
                    }

                    let can_pack_global_memory = done
                        && records.len() == 1
                        && records.last().unwrap().estimated_trace_area
                            < opts.split_opts.combine_memory_threshold.0
                        && deferred.global_memory_initialize_events.len()
                            < opts.split_opts.combine_memory_threshold.1
                        && deferred.global_memory_finalize_events.len()
                            < opts.split_opts.combine_memory_threshold.1;

                    let last_record =
                        if can_pack_global_memory { records.last_mut() } else { None };

                    // See if any deferred shards are ready to be committed to.
                    let mut deferred = deferred.split(done, last_record, opts.split_opts);
                    tracing::debug!("deferred {} records", deferred.len());

                    // Update the public values & prover state for the shards which do not
                    // contain "cpu events" before committing to them.
                    state.execution_shard = state.next_execution_shard;
                    for record in deferred.iter_mut() {
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
                    records.append(&mut deferred);

                    // Generate the dependencies.
                    machine.generate_dependencies(&mut records, None);

                    // Let another worker update the state.
                    record_gen_sync.advance_turn();

                    // Send the records to the output channel.
                    for record in records {
                        record_tx.blocking_send(record).unwrap();
                    }
                }
            });
        }

        // Spawn the checkpoint generation task.
        let opts = self.opts.clone();
        tokio::task::spawn(async move {
            let record_worker_channels = Arc::new(WorkerQueue::new(record_worker_channels));
            'task_loop: while let Some(task) = task_rx.recv().await {
                let ExecuteTask {
                    program,
                    stdin,
                    context,
                    record_tx,
                    output_tx,
                    abort_registration,
                    // TODO: Use the span.
                    span: _,
                } = task;

                if abort_registration.handle().is_aborted() {
                    // If the task was aborted, send an error to the output channel and continue, if
                    //  the channel is closed, just drop the message.
                    output_tx.send(Err(MachineExecutorError::TaskAborted)).ok();
                    continue;
                }

                // Initialize the record generation state.
                let record_gen_sync = Arc::new(TurnBasedSync::new());
                let state =
                    Arc::new(Mutex::new(PublicValues::<u32, u64, u64, u32>::default().reset()));
                let deferred = Arc::new(Mutex::new(ExecutionRecord::new(program.clone())));

                // Check if the task was aborted again.
                if abort_registration.handle().is_aborted() {
                    output_tx.send(Err(MachineExecutorError::TaskAborted)).ok();
                    continue;
                }

                // Setup the runtime.
                let mut runtime =
                    Box::new(Executor::with_context(program.clone(), opts.clone(), context));
                runtime.write_vecs(&stdin.buffer);
                for proof in stdin.proofs.iter() {
                    let (proof, vk) = proof.clone();
                    runtime.write_proof(proof, vk);
                }

                // Generate checkpoints until the execution is done.
                let mut index = 0;
                let abort_handle = abort_registration.handle();
                let mut done = false;
                while !done && !abort_handle.is_aborted() {
                    // Send and receive ownership of `runtime: Box<Executor<'_>>`.
                    // The `.unwrap()` propagates panics from `generate_checkpoint`.
                    let checkpoint_result;
                    (runtime, checkpoint_result) = tokio::task::spawn_blocking(move || {
                        let res = generate_checkpoint(&mut runtime);
                        (runtime, res)
                    })
                    .await
                    .unwrap();
                    match checkpoint_result {
                        Ok((checkpoint_file, is_done)) => {
                            // Update the finished flag.
                            done = is_done;
                            // Create a new record generation task.
                            let record_task = RecordTask {
                                index,
                                checkpoint_file,
                                done,
                                program: program.clone(),
                                record_gen_sync: record_gen_sync.clone(),
                                state: state.clone(),
                                deferred: deferred.clone(),
                                record_tx: record_tx.clone(),
                                abort_handle: abort_registration.handle(),
                                span: tracing::debug_span!("execute record"),
                            };
                            // Send the checkpoint to the record generation worker.
                            let record_worker = record_worker_channels.clone().pop().await.unwrap();
                            // Send the task to the worker.
                            record_worker.send(record_task).unwrap();
                            // Increment the index.
                            index += 1;
                        }
                        Err(e) => {
                            output_tx.send(Err(e)).ok();
                            continue 'task_loop;
                        }
                    }
                }
                // Execution is done, send the output to the sender.
                let public_value_stream = runtime.state.public_values_stream;
                let cycles = runtime.state.global_clk;
                output_tx.send(Ok(ExecutionOutput { public_value_stream, cycles })).ok();
            }
        });

        MachineExecutor { task_tx, _marker: PhantomData }
    }
}

pub fn generate_checkpoint(runtime: &mut Executor) -> Result<(File, bool), MachineExecutorError> {
    // Execute the runtime until we reach a checkpoint.
    let (checkpoint, _, done) =
        runtime.execute_state(false).map_err(MachineExecutorError::ExecutionError)?;

    // Save the checkpoint to a temp file.
    let mut checkpoint_file = tempfile::tempfile().map_err(MachineExecutorError::IoError)?;
    checkpoint.save(&mut checkpoint_file).map_err(MachineExecutorError::IoError)?;

    Ok((checkpoint_file, done))
}

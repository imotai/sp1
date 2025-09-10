use std::{
    io,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use futures::future::try_join_all;
use hashbrown::HashSet;
use slop_algebra::PrimeField32;
use slop_futures::queue::WorkerQueue;
use sp1_core_executor::{
    events::MemoryRecord, ExecutionError, ExecutionRecord, Program, SP1Context, SP1CoreOpts,
    SplicingVM, SplitOpts, TracingVM,
};
use sp1_hypercube::{
    air::PublicValues,
    prover::{MemoryPermit, MemoryPermitting},
    Machine, MachineRecord,
};
use thiserror::Error;
use tokio::sync::mpsc::{self, UnboundedSender};
use tracing::Instrument;

use sp1_core_executor::{CycleResult, MinimalExecutor, SplicedMinimalTrace, TraceChunkRaw};

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

        Self {
            num_record_workers,
            opts,
            machine,
            memory: MemoryPermitting::new(record_buffer_size),
            _marker: PhantomData,
        }
    }

    /// Get a reference to the core options.
    pub fn opts(&self) -> &SP1CoreOpts {
        &self.opts
    }

    pub async fn execute(
        &self,
        program: Arc<Program>,
        stdin: SP1Stdin,
        _context: SP1Context<'static>,
        record_tx: mpsc::UnboundedSender<(ExecutionRecord, Option<MemoryPermit>)>,
    ) -> Result<ExecutionOutput, MachineExecutorError> {
        let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<TraceChunkRaw>();
        let (last_record_tx, mut last_record_rx) = tokio::sync::mpsc::channel::<ExecutionRecord>(1);
        let record_gen_sync = AsyncTurn::new();
        let state = Arc::new(Mutex::new(PublicValues::<u32, u64, u64, u32>::default().reset()));
        let deferred = Arc::new(Mutex::new(ExecutionRecord::new(program.clone())));
        let mut record_worker_channels = Vec::with_capacity(self.num_record_workers);

        // todo: use page protection
        let split_opts = SplitOpts::new(&self.opts, program.instructions.len(), false);

        tracing::debug!("starting {} record worker channels", self.num_record_workers);
        let mut handles = Vec::with_capacity(self.num_record_workers);
        for i in 0..self.num_record_workers {
            let (tx, mut rx) = mpsc::unbounded_channel::<RecordTask>();
            record_worker_channels.push(tx);
            let machine = self.machine.clone();
            let opts = self.opts.clone();
            let program = program.clone();
            let record_gen_sync = record_gen_sync.clone();
            let record_tx = record_tx.clone();
            let state = state.clone();
            let deferred: Arc<Mutex<ExecutionRecord>> = deferred.clone();
            let last_record_tx: mpsc::Sender<ExecutionRecord> = last_record_tx.clone();
            let permitting = self.memory.clone();

            handles.push(tokio::task::spawn(
                async move {
                    while let Some(task) = rx.recv().await {
                        let RecordTask { index, chunk } = task;
                        eprintln!("tracing chunk at idx: {}, with worker: {}", index, i);

                        tracing::debug!("tracing chunk at idx: {}", index);

                        // Assume a record is 4Gb for now.
                        let permit = permitting.acquire(4 * 1024 * 1024 * 1024).await.unwrap();

                        let program = program.clone();
                        let opts = opts.clone();
                        let (done, mut record) = tokio::task::spawn_blocking({
                            let program = program.clone();
                            let opts = opts.clone();

                            move || {
                                let _debug_span =
                                    tracing::debug_span!("tracing chunk blocking task").entered();
                                trace_chunk(program, opts, chunk)
                            }
                        })
                        .await
                        .expect("error: trace chunk task panicked")
                        .expect("todo: handle error");

                        // Wait for our turn to update the state.
                        let _turn_guard = record_gen_sync.wait_for_turn(index).await;

                        let deferred_records = if done {
                            tracing::debug!("last record at idx: {}", index);

                            // If this is the last record, we have special handling for the memory events.
                            last_record_tx.send(record).await.unwrap();
                            return;
                        } else {
                            tracing::debug!("defferring record at idx: {}", index);

                            let mut state = state.lock().unwrap();
                            let mut deferred = deferred.lock().unwrap();

                            defer::<F>(
                                &mut state,
                                &mut record,
                                &mut deferred,
                                &split_opts,
                                opts,
                                done,
                            )
                        };

                        start_prove(
                            machine.clone(),
                            record_tx.clone(),
                            Some(permit),
                            record,
                            deferred_records,
                        )
                        .await;
                    }
                }
                .instrument(tracing::debug_span!("tracing worker")),
            ));
        }

        let record_worker_channels = Arc::new(WorkerQueue::new(record_worker_channels));

        let minimal_executor_handle = tokio::task::spawn_blocking({
            let program = program.clone();

            move || {
                let _debug_span = tracing::debug_span!("minimal executor task").entered();
                let mut minimal_executor = MinimalExecutor::new(program.clone(), true, false, None);

                for buf in stdin.buffer {
                    minimal_executor.with_input(&buf);
                }

                while let Some(chunk) = minimal_executor.execute_chunk() {
                    tracing::debug!("program is done?: {}", minimal_executor.is_done());

                    chunk_tx.send(chunk).unwrap();
                }

                minimal_executor
            }
        });

        let (touched_addresses, final_registers) = tokio::task::spawn({
            let program: Arc<Program> = program.clone();

            async move {
                let touched_addresses =
                    Arc::new(Mutex::new(HashSet::from_iter(program.memory_image.keys().copied())));
                let mut final_registers = [MemoryRecord::default(); 32];
                let mut idx = 0;

                while let Some(chunk) = chunk_rx.recv().await {
                    send_full_trace(record_worker_channels.clone(), chunk.clone(), idx).await;
                    idx += 1;

                    (idx, final_registers) = tokio::task::spawn_blocking({
                        let program = program.clone();
                        let touched_addresses = touched_addresses.clone();
                        let record_worker_channels = record_worker_channels.clone();

                        move || {
                            generate_chunks(
                                program,
                                chunk,
                                record_worker_channels,
                                touched_addresses,
                                idx,
                            )
                        }
                    })
                    .await
                    .expect("error: generate trace chunks task panicked")
                    .expect("todo: handle error");
                }

                (touched_addresses, final_registers)
            }
            .instrument(tracing::debug_span!("splitting task"))
        })
        .await
        .unwrap();

        // Wait for the minimal executor to finish.
        let minimal_executor = minimal_executor_handle.await.unwrap();

        // Wait for the record workers to finish.
        try_join_all(handles).await.unwrap();

        // Wait for the last record to be traced.
        let mut last_record = last_record_rx.recv().await.unwrap();
        tracing::info!(
            "last_record.public_values.committed_value_digest: {:?}",
            last_record.public_values.committed_value_digest
        );
        let deferred_records = {
            // Take the lock on the touched addresses.
            let touched_addresses = std::mem::take(&mut *touched_addresses.lock().unwrap());
            // Insert the global memory events into the last record.
            minimal_executor.emit_globals(&mut last_record, final_registers, touched_addresses);

            let mut deferred = deferred.lock().unwrap();
            let mut state = state.lock().unwrap();
            tracing::debug_span!("postprocessing task").in_scope(|| {
                defer::<F>(
                    &mut state,
                    &mut last_record,
                    &mut deferred,
                    &split_opts,
                    self.opts.clone(),
                    true,
                )
            })
        };

        start_prove(self.machine.clone(), record_tx, None, last_record, deferred_records).await;

        Ok(ExecutionOutput {
            cycles: minimal_executor.global_clk(),
            public_value_stream: minimal_executor.into_public_values_stream(),
        })
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
    chunk: SplicedMinimalTrace<TraceChunkRaw>,
}

/// Generate the chunks (corresponding to shards) and send them to the record workers.
fn generate_chunks(
    program: Arc<Program>,
    chunk: TraceChunkRaw,
    record_worker_channels: Arc<WorkerQueue<UnboundedSender<RecordTask>>>,
    touched_addresses: Arc<Mutex<HashSet<u64>>>,
    mut idx: usize,
) -> Result<(usize, [MemoryRecord; 32]), ExecutionError> {
    let mut touched_addresses = touched_addresses.lock().unwrap();
    let mut vm = SplicingVM::new(&chunk, program.clone(), &mut touched_addresses);

    loop {
        tracing::debug!("starting new shard idx: {} at clk: {}", idx, vm.core.clk());

        match vm.execute().expect("todo: handle result") {
            CycleResult::ShardBoundry => {
                // Note: Chunk implentations should always be cheap to clone.
                if let Some(spliced) = vm.splice(chunk.clone()) {
                    tracing::debug!("generated chunk for shard {}", idx);
                    tracing::debug!("shard ended at clk: {}", vm.core.clk());
                    tracing::debug!("shard ended at pc: {}", vm.core.pc());
                    tracing::debug!("shard ended at global clk: {}", vm.core.global_clk());
                    tracing::debug!("shard ended with {} mem reads left ", vm.core.mem_reads.len());

                    // Send the spliced trace to a available worker.
                    send_spliced_trace_blocking(record_worker_channels.clone(), spliced, idx);

                    // Bump the shard index.
                    idx += 1;
                } else {
                    tracing::debug!("shard idx ran out of trace and was cut early: {}", idx);
                    // Since we didnt get a `done` status, we dont really need the final registers.
                    return Ok((idx, [MemoryRecord::default(); 32]));
                }
            }
            CycleResult::Done(true) => {
                return Ok((idx, *vm.core.registers()));
            }
            CycleResult::Done(false) | CycleResult::TraceEnd => {
                // Note: Trace ends get mapped to shard boundaries.
                unreachable!("The executor should never return an imcomplete program without a shard boundary");
            }
        }
    }
}

/// Trace a single [`SplicedMinimalTrace`] (corresponding to a shard) and return the execution record.
fn trace_chunk(
    program: Arc<Program>,
    _opts: SP1CoreOpts,
    chunk: SplicedMinimalTrace<TraceChunkRaw>,
) -> Result<(bool, ExecutionRecord), ExecutionError> {
    let mut vm = TracingVM::new(&chunk, program);
    let status = vm.execute()?;
    tracing::debug!("chunk ended at clk: {}", vm.core.clk());
    tracing::debug!("chunk ended at pc: {}", vm.core.pc());

    let mut record = std::mem::take(&mut vm.record);
    let pv = record.public_values;

    // Handle the case where `COMMIT` or `COMMIT_DEFERRED_PROOFS` happens across last two shards.
    //
    // todo: does this actually work in the new regieme? what if the shard is stopped due to the clk limit?
    // if so, does that mean this could be wrong? its unclear!
    if status.is_shard_boundry() && (pv.commit_syscall == 1 || pv.commit_deferred_syscall == 1) {
        tracing::debug!("commit syscall or commit deferred proofs across last two shards");

        loop {
            // Execute until we get a done status.
            if vm.execute()?.is_done() {
                let pv = vm.record.public_values;

                // Update the record.
                record.public_values.commit_syscall = 1;
                record.public_values.commit_deferred_syscall = 1;
                record.public_values.committed_value_digest = pv.committed_value_digest;
                record.public_values.deferred_proofs_digest = pv.deferred_proofs_digest;

                break;
            }
        }
    }

    Ok((status.is_done(), record))
}

#[tracing::instrument(name = "defer", skip_all)]
fn defer<F: PrimeField32>(
    state: &mut PublicValues<u32, u64, u64, u32>,
    record: &mut ExecutionRecord,
    deferred: &mut ExecutionRecord,
    split_opts: &SplitOpts,
    opts: SP1CoreOpts,
    done: bool,
) -> Vec<ExecutionRecord> {
    state.is_execution_shard = 1;
    state.pc_start = record.public_values.pc_start;
    state.next_pc = record.public_values.next_pc;
    state.initial_timestamp = record.public_values.initial_timestamp;
    state.last_timestamp = record.public_values.last_timestamp;
    state.is_first_shard = (record.public_values.initial_timestamp == 1) as u32;

    let initial_timestamp_high = (state.initial_timestamp >> 24) as u32;
    let initial_timestamp_low = (state.initial_timestamp & 0xFFFFFF) as u32;
    let last_timestamp_high = (state.last_timestamp >> 24) as u32;
    let last_timestamp_low = (state.last_timestamp & 0xFFFFFF) as u32;

    state.initial_timestamp_inv = if state.initial_timestamp == 1 {
        0
    } else {
        F::from_canonical_u32(initial_timestamp_high + initial_timestamp_low - 1)
            .inverse()
            .as_canonical_u32()
    };

    state.last_timestamp_inv = F::from_canonical_u32(last_timestamp_high + last_timestamp_low - 1)
        .inverse()
        .as_canonical_u32();

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
        state.committed_value_digest = record.public_values.committed_value_digest;
    }
    if state.deferred_proofs_digest == [0u32; 8] {
        state.deferred_proofs_digest = record.public_values.deferred_proofs_digest;
    }
    if state.commit_syscall == 0 {
        state.commit_syscall = record.public_values.commit_syscall;
    }
    if state.commit_deferred_syscall == 0 {
        state.commit_deferred_syscall = record.public_values.commit_deferred_syscall;
    }
    if state.exit_code == 0 {
        state.exit_code = record.public_values.exit_code;
    }
    record.public_values = *state;
    state.prev_exit_code = state.exit_code;
    state.prev_commit_syscall = state.commit_syscall;
    state.prev_commit_deferred_syscall = state.commit_deferred_syscall;
    state.prev_committed_value_digest = state.committed_value_digest;
    state.prev_deferred_proofs_digest = state.deferred_proofs_digest;
    state.initial_timestamp = state.last_timestamp;

    // Defer events that are too expensive to include in every shard.
    deferred.append(&mut record.defer(&opts.retained_events_presets));

    let can_pack_global_memory = done
        && record.estimated_trace_area <= split_opts.pack_trace_threshold
        && deferred.global_memory_initialize_events.len() <= split_opts.combine_memory_threshold
        && deferred.global_memory_finalize_events.len() <= split_opts.combine_memory_threshold
        && deferred.global_page_prot_initialize_events.len()
            <= split_opts.combine_page_prot_threshold
        && deferred.global_page_prot_finalize_events.len()
            <= split_opts.combine_page_prot_threshold;

    // See if any deferred shards are ready to be committed to.
    let mut deferred_records =
        deferred.split(done, can_pack_global_memory.then_some(record), split_opts);
    tracing::debug!("split deffered into {} records", deferred_records.len());

    // Update the public values & prover state for the shards which do not
    // contain "cpu events" before committing to them.
    for record in deferred_records.iter_mut() {
        state.previous_init_addr = record.public_values.previous_init_addr;
        state.last_init_addr = record.public_values.last_init_addr;
        state.previous_finalize_addr = record.public_values.previous_finalize_addr;
        state.last_finalize_addr = record.public_values.last_finalize_addr;

        state.pc_start = state.next_pc;
        state.prev_exit_code = state.exit_code;
        state.prev_commit_syscall = state.commit_syscall;

        state.prev_commit_deferred_syscall = state.commit_deferred_syscall;
        state.prev_committed_value_digest = state.committed_value_digest;
        state.prev_deferred_proofs_digest = state.deferred_proofs_digest;

        state.last_timestamp = state.initial_timestamp;
        state.is_timestamp_high_eq = 1;
        state.is_timestamp_low_eq = 1;

        state.is_first_shard = 0;
        state.is_execution_shard = 0;

        let initial_timestamp_high = (state.initial_timestamp >> 24) as u32;
        let initial_timestamp_low = (state.initial_timestamp & 0xFFFFFF) as u32;
        let last_timestamp_high = (state.last_timestamp >> 24) as u32;
        let last_timestamp_low = (state.last_timestamp & 0xFFFFFF) as u32;

        state.is_first_shard = (record.public_values.initial_timestamp == 1) as u32;
        state.initial_timestamp_inv =
            F::from_canonical_u32(initial_timestamp_high + initial_timestamp_low - 1)
                .inverse()
                .as_canonical_u32();
        state.last_timestamp_inv =
            F::from_canonical_u32(last_timestamp_high + last_timestamp_low - 1)
                .inverse()
                .as_canonical_u32();
        record.public_values = *state;
    }

    deferred_records
}

/// Generate the dependencies and send the records to the prover channel.
#[tracing::instrument(name = "start_prove", skip_all)]
async fn start_prove<F: PrimeField32>(
    machine: Machine<F, RiscvAir<F>>,
    record_tx: mpsc::UnboundedSender<(ExecutionRecord, Option<MemoryPermit>)>,
    permit: Option<MemoryPermit>,
    mut record: ExecutionRecord,
    mut deferred_records: Vec<ExecutionRecord>,
) {
    tracing::debug!("num deferred records: {:#?}", deferred_records.len());

    // Generate the dependencies.
    tokio::task::spawn_blocking({
        let machine = machine.clone();
        let record_tx = record_tx.clone();

        move || {
            let record_iter = std::iter::once(&mut record);
            machine.generate_dependencies(record_iter, None);
            machine.generate_dependencies(deferred_records.iter_mut(), None);

            // Send the records to the output channel.
            record_tx.send((record, permit)).unwrap();

            // If there are deferred records, send them to the output channel.
            for record in deferred_records {
                record_tx.send((record, None)).unwrap();
            }
        }
    })
    .await
    .expect("failed to send records");
}

/// The first thing accepted by a record worker is always the full trace.
async fn send_full_trace(
    record_worker_channels: Arc<WorkerQueue<UnboundedSender<RecordTask>>>,
    chunk: TraceChunkRaw,
    idx: usize,
) {
    let worker = record_worker_channels.clone().pop().await.unwrap();
    let full_trace = SplicedMinimalTrace::new_full_trace(chunk.clone());
    worker.send(RecordTask { index: idx, chunk: full_trace }).unwrap();
}

/// Send the splice trace to a available record worker.
fn send_spliced_trace_blocking(
    record_worker_channels: Arc<WorkerQueue<UnboundedSender<RecordTask>>>,
    chunk: SplicedMinimalTrace<TraceChunkRaw>,
    idx: usize,
) {
    loop {
        match record_worker_channels.clone().try_pop() {
            Ok(worker) => {
                worker.send(RecordTask { index: idx, chunk }).unwrap();
                break;
            }
            // todo: patch slop to return what kind of error so we can break correctly if the channel is closed.
            Err(_) => {
                std::hint::spin_loop();
            }
        }
    }
}

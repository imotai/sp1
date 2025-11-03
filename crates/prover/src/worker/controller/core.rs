use std::sync::{Arc, Mutex, OnceLock};

use futures::{prelude::*, stream::FuturesUnordered};
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use opentelemetry::Context;
use serde::{Deserialize, Serialize};
use slop_futures::pipeline::Pipeline;
use sp1_core_executor::{
    chunked_memory_init_events,
    events::{MemoryInitializeFinalizeEvent, MemoryRecord},
    syscalls::SyscallCode,
    CoreVM, ExecutionError, ExecutionRecord, MinimalExecutor, Program, SP1CoreOpts,
    SplicedMinimalTrace, SplitOpts, TraceChunkRaw,
};
use sp1_core_machine::{executor::ExecutionOutput, io::SP1Stdin};
use sp1_hypercube::air::{ShardRange, PROOF_NONCE_NUM_WORDS, PV_DIGEST_NUM_WORDS};
use sp1_jit::MinimalTrace;
use sp1_prover_types::{
    await_scoped_vec, network_base_types::ProofMode, Artifact, ArtifactClient, ArtifactType,
    TaskType,
};
use tokio::{sync::mpsc, task::JoinSet};

use crate::{
    worker::{
        ProofId, ProveShardTaskRequest, RequesterId, SplicingEngine, SplicingTask, TaskError,
        TaskId, WorkerClient,
    },
    SP1VerifyingKey,
};

/// String used as key for add_ref to ensure precompile artifacts are not cleaned up before they
/// are fully split into multiple shards.
const CONTROLLER_PRECOMPILE_ARTIFACT_REF: &str = "_controller";

#[derive(Debug)]
pub struct ProofData {
    pub task_id: TaskId,
    pub range: ShardRange,
    pub proof: Artifact,
}

pub struct SendSpliceTask {
    pub chunk: SplicedMinimalTrace<TraceChunkRaw>,
    pub range: ShardRange,
}

#[derive(Serialize, Deserialize)]
pub enum TraceData {
    /// A core record to be proven.
    Core(Vec<u8>),
    // Precompile data. Several `PrecompileArtifactSlice`s, and the type of precompile.
    Precompile(Vec<PrecompileArtifactSlice>, SyscallCode),
    /// Memory data.
    Memory(Box<GlobalMemoryShard>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMemoryShard {
    pub final_state: FinalVmState,
    pub initialize_events: Vec<MemoryInitializeFinalizeEvent>,
    pub finalize_events: Vec<MemoryInitializeFinalizeEvent>,
    pub previous_init_addr: u64,
    pub previous_finalize_addr: u64,
    pub previous_init_page_idx: u64,
    pub previous_finalize_page_idx: u64,
    pub last_init_addr: u64,
    pub last_finalize_addr: u64,
    pub last_init_page_idx: u64,
    pub last_finalize_page_idx: u64,
}

pub struct ProveShardInput {
    pub elf: Vec<u8>,
    pub common_input: CommonProverInput,
    pub record: TraceData,
    pub opts: SP1CoreOpts,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CommonProverInput {
    pub vk: SP1VerifyingKey,
    pub mode: ProofMode,
    pub deferred_digest: [u32; 8],
    pub num_deferred_proofs: usize,
}

pub struct SP1CoreExecutor<A, W> {
    splicing_engine: Arc<SplicingEngine<A, W>>,
    elf: Artifact,
    stdin: Arc<SP1Stdin>,
    common_input: Artifact,
    opts: SP1CoreOpts,
    num_deferred_proofs: usize,
    proof_id: ProofId,
    parent_id: Option<TaskId>,
    parent_context: Option<Context>,
    requester_id: RequesterId,
    sender: mpsc::UnboundedSender<ProofData>,
    artifact_client: A,
    worker_client: W,
}

impl<A, W> SP1CoreExecutor<A, W> {
    pub fn new(
        splicing_engine: Arc<SplicingEngine<A, W>>,
        elf: Artifact,
        stdin: Arc<SP1Stdin>,
        common_input: Artifact,
        opts: SP1CoreOpts,
        num_deferred_proofs: usize,
        proof_id: ProofId,
        parent_id: Option<TaskId>,
        parent_context: Option<Context>,
        requester_id: RequesterId,
        sender: mpsc::UnboundedSender<ProofData>,
        artifact_client: A,
        worker_client: W,
    ) -> Self {
        Self {
            splicing_engine,
            elf,
            stdin,
            common_input,
            opts,
            num_deferred_proofs,
            proof_id,
            parent_id,
            parent_context,
            requester_id,
            sender,
            artifact_client,
            worker_client,
        }
    }
}

impl<A, W> SP1CoreExecutor<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    pub async fn execute(self) -> Result<ExecutionOutput, TaskError> {
        let elf_bytes = self.artifact_client.download_program(&self.elf).await?;
        let stdin = self.stdin.clone();
        let opts = self.opts.clone();

        // Get the program from the elf. TODO: handle errors.
        let program = Arc::new(Program::from(&elf_bytes).map_err(|e| {
            TaskError::Execution(ExecutionError::Other(format!(
                "failed to dissassemble program: {}",
                e
            )))
        })?);

        // Initialize the touched addresses map.
        let all_touched_addresses = TouchedAddresses::new();
        // Initialize the final vm state.
        let final_vm_state = FinalVmStateLock::new();

        // Start the minimal executor.
        let parent = tracing::Span::current();
        // TODO: think about cancellation handling
        let minimal_executor_handle = tokio::task::spawn_blocking({
            let program = program.clone();
            let splicing_engine = self.splicing_engine.clone();
            let elf = self.elf.clone();
            let common_input_artifact = self.common_input.clone();
            let proof_id = self.proof_id.clone();
            let parent_id = self.parent_id.clone();
            let parent_context = self.parent_context.clone();
            let requester_id = self.requester_id.clone();
            let all_touched_addresses = all_touched_addresses.clone();
            let sender = self.sender.clone();
            let final_vm_state = final_vm_state.clone();
            let opts = opts.clone();
            move || {
                let _guard = parent.enter();

                let _minimal_exec_span = tracing::debug_span!("minimal executor task").entered();
                let max_trace_size = opts.minimal_trace_chunk_threshold;
                let mut minimal_executor =
                    MinimalExecutor::tracing(program.clone(), max_trace_size);

                // Write input to the minimal executor.
                for buf in stdin.buffer.iter() {
                    minimal_executor.with_input(buf);
                }

                let splicing_handles = FuturesUnordered::new();
                tracing::info!("Starting minimal executor");
                let now = std::time::Instant::now();
                while let Some(chunk) = minimal_executor.execute_chunk() {
                    tracing::trace!("program is done?: {}", minimal_executor.is_done());
                    tracing::trace!(
                        "mem reads chunk size bytes {}",
                        chunk.num_mem_reads() * std::mem::size_of::<sp1_jit::MemValue>() as u64
                    );

                    // Create a splicing task
                    let task = SplicingTask {
                        program: program.clone(),
                        chunk,
                        elf_artifact: elf.clone(),
                        common_input_artifact: common_input_artifact.clone(),
                        num_deferred_proofs: self.num_deferred_proofs,
                        all_touched_addresses: all_touched_addresses.clone(),
                        final_vm_state: final_vm_state.clone(),
                        prove_shard_tx: sender.clone(),
                        proof_id: proof_id.clone(),
                        parent_id: parent_id.clone(),
                        parent_context: parent_context.clone(),
                        requester_id: requester_id.clone(),
                        opts: opts.clone(),
                    };

                    let splicing_handle = splicing_engine
                        .blocking_submit(task)
                        .expect("failed to submit splicing task");

                    splicing_handles.push(splicing_handle);
                }
                let elapsed = now.elapsed().as_secs_f64();
                tracing::info!(
                    "executor finished. elapsed: {}s, mhz: {}",
                    elapsed,
                    minimal_executor.global_clk() as f64 / (elapsed * 1e6)
                );
                (minimal_executor, splicing_handles)
            }
        });

        let (minimal_executor, splicing_handles) =
            minimal_executor_handle.await.expect("executor task panicked");

        // Wait for splicing tasks to finish, catch any potential errors
        splicing_handles
            .map(|nested| {
                let result = nested.map_err(|e| anyhow::anyhow!("splicing task panicked: {}", e));
                let result =
                    result.and_then(|r| r.map_err(|e| anyhow::anyhow!("execution error: {}", e)));
                result
            })
            .try_collect::<Vec<()>>()
            .await?;
        tracing::trace!("splicing tasks finished");

        let touched_addresses = all_touched_addresses.take();
        let final_state = *final_vm_state.get().expect("final vm state not set");

        let cycles = minimal_executor.global_clk();
        let public_value_stream = minimal_executor.public_values_stream().clone();

        let output = ExecutionOutput { cycles, public_value_stream };

        self.emit_global_memory_shards(
            minimal_executor,
            program,
            final_state,
            touched_addresses,
            opts,
        )
        .await?;

        Ok(output)
    }

    async fn emit_global_memory_shards(
        &self,
        minimal_executor: MinimalExecutor,
        program: Arc<Program>,
        final_state: FinalVmState,
        touched_addresses: HashSet<u64>,
        opts: SP1CoreOpts,
    ) -> anyhow::Result<()> {
        // Get the split opts
        let split_opts = SplitOpts::new(&opts, program.instructions.len(), false);

        let (shard_data_tx, mut shard_data_rx) = mpsc::channel(1);

        let mut join_set = JoinSet::new();

        // Spawn the task that creates the memory shards
        join_set.spawn_blocking({
            let threshold = split_opts.memory;
            move || {
                let (global_memory_initialize_events, global_memory_finalize_events) =
                    Self::global_memory_events(
                        minimal_executor,
                        program,
                        final_state.registers,
                        touched_addresses,
                    );

                for chunks in global_memory_initialize_events
                    .chunks(threshold)
                    .into_iter()
                    .zip_longest(global_memory_finalize_events.chunks(threshold).into_iter())
                {
                    match chunks {
                        itertools::EitherOrBoth::Left(initialize_events) => {
                            let initialize_events =
                                initialize_events.into_iter().collect::<Vec<_>>();
                            shard_data_tx
                                .blocking_send((initialize_events, vec![]))
                                .expect("failed to send initialize events");
                        }
                        itertools::EitherOrBoth::Right(finalize_events) => {
                            let finalize_events = finalize_events.into_iter().collect::<Vec<_>>();
                            shard_data_tx
                                .blocking_send((vec![], finalize_events))
                                .expect("failed to send finalize events");
                        }
                        itertools::EitherOrBoth::Both(initialize_events, finalize_events) => {
                            let initialize_events =
                                initialize_events.into_iter().collect::<Vec<_>>();
                            let finalize_events = finalize_events.into_iter().collect::<Vec<_>>();
                            shard_data_tx
                                .blocking_send((initialize_events, finalize_events))
                                .expect("failed to send events");
                        }
                    }
                }
            }
        });

        // Spawn the task that creates the proving tasks and submits them to the worker
        let num_deferred_proofs = self.num_deferred_proofs;
        join_set.spawn({
            let artifact_client = self.artifact_client.clone();
            let worker_client = self.worker_client.clone();
            let elf_artifact = self.elf.clone();
            let common_input_artifact = self.common_input.clone();
            let proof_id = self.proof_id.clone();
            let parent_id = self.parent_id.clone();
            let parent_context = self.parent_context.clone();
            let requester_id = self.requester_id.clone();
            let prove_shard_tx = self.sender.clone();

            async move {
                let mut counter = 0;
                let mut previous_init_addr = 0;
                let mut previous_finalize_addr = 0;
                let mut previous_init_page_idx = 0;
                let mut previous_finalize_page_idx = 0;
                while let Some((initialize_events, finalize_events)) = shard_data_rx.recv().await {
                    tracing::trace!("Got global memory shard number {counter}");
                    let last_init_addr = initialize_events
                        .last()
                        .map(|event| event.addr)
                        .unwrap_or(previous_init_addr);
                    let last_finalize_addr = finalize_events
                        .last()
                        .map(|event| event.addr)
                        .unwrap_or(previous_finalize_addr);
                    tracing::info!("last_init_addr: {last_init_addr}, last_finalize_addr: {last_finalize_addr}");
                    let last_init_page_idx = previous_init_page_idx;
                    let last_finalize_page_idx = previous_finalize_page_idx;

                    // Calculate the range of the shard.
                    let range = ShardRange {
                        timestamp_range: (final_state.timestamp, final_state.timestamp),
                        initialized_address_range: (previous_init_addr, last_init_addr),
                        finalized_address_range: (previous_finalize_addr, last_finalize_addr),
                        initialized_page_index_range: (previous_init_page_idx, last_init_page_idx),
                        finalized_page_index_range: (
                            previous_finalize_page_idx,
                            last_finalize_page_idx,
                        ),
                        deferred_proof_range: (
                            num_deferred_proofs as u64,
                            num_deferred_proofs as u64,
                        ),
                    };
                    let mem_global_shard = GlobalMemoryShard {
                        final_state,
                        initialize_events,
                        finalize_events,
                        previous_init_addr,
                        previous_finalize_addr,
                        previous_init_page_idx,
                        previous_finalize_page_idx,
                        last_init_addr,
                        last_finalize_addr,
                        last_init_page_idx,
                        last_finalize_page_idx,
                    };

                    let data = TraceData::Memory(Box::new(mem_global_shard));

                    // Upload the data
                    let data_artifact = artifact_client
                        .create_artifact()
                        .expect("failed to create record artifact");
                    artifact_client
                        .upload(&data_artifact, data)
                        .await
                        .expect("failed to upload record");

                    let deferred_marker_artifact =
                        Artifact::from("global memory dummy artifact".to_string());

                    // Allocate an artifact for the proof
                    let proof_artifact =
                        artifact_client.create_artifact().expect("failed to create proof artifact");

                    let dummy_output_artifact =
                        Artifact::from("dummy global memory deferred output artifact".to_string());

                    let request = ProveShardTaskRequest {
                        proof_id: proof_id.clone(),
                        elf: elf_artifact.clone(),
                        common_input: common_input_artifact.clone(),
                        record: data_artifact,
                        output: proof_artifact.clone(),
                        deferred_marker_task: deferred_marker_artifact,
                        deferred_output: dummy_output_artifact,
                        parent_id: parent_id.clone(),
                        parent_context: parent_context.clone(),
                        requester_id: requester_id.clone(),
                    };
                    let task = request.into_raw().expect("failed to convert to raw task request");

                    // Send the task to the worker.
                    let task_id = worker_client
                        .submit_task(TaskType::ProveShard, task)
                        .await
                        .expect("failed to send task");

                    // Send the task data
                    let proof_data = ProofData { task_id, range, proof: proof_artifact };
                    prove_shard_tx.send(proof_data).expect("failed to send task id");
                    tracing::debug!("Submitted memory global shard {counter}");
                    counter += 1;
                    previous_init_addr = last_init_addr;
                    previous_finalize_addr = last_finalize_addr;
                    previous_init_page_idx = last_init_page_idx;
                    previous_finalize_page_idx = last_finalize_page_idx;
                }
            }
        });

        // Wait for the tasks to finish
        join_set.join_all().await;

        Ok(())
    }

    fn global_memory_events(
        minimal_executor: MinimalExecutor,
        program: Arc<Program>,
        final_registers: [MemoryRecord; 32],
        mut touched_addresses: HashSet<u64>,
    ) -> (
        impl Iterator<Item = MemoryInitializeFinalizeEvent>,
        impl Iterator<Item = MemoryInitializeFinalizeEvent>,
    ) {
        // Add all the finalize addresses to the touched addresses.
        touched_addresses.extend(program.memory_image.keys().copied());

        let global_memory_initialize_events = final_registers
            .into_iter()
            .enumerate()
            .filter(|(_, e)| e.timestamp != 0)
            .map(|(i, _)| MemoryInitializeFinalizeEvent::initialize(i as u64, 0));

        let global_memory_finalize_events =
            final_registers.into_iter().enumerate().filter(|(_, e)| e.timestamp != 0).map(
                |(i, entry)| {
                    MemoryInitializeFinalizeEvent::finalize(i as u64, entry.value, entry.timestamp)
                },
            );

        let hint_init_events: Vec<MemoryInitializeFinalizeEvent> = minimal_executor
            .hints()
            .iter()
            .flat_map(|(addr, value)| chunked_memory_init_events(*addr, value))
            .collect::<Vec<_>>();
        let hint_addrs = hint_init_events.iter().map(|event| event.addr).collect::<HashSet<_>>();

        // Initialize the all the hints written during execution.
        let global_memory_initialize_events =
            global_memory_initialize_events.chain(hint_init_events);

        // Initialize the memory addresses that were touched during execution.
        // We don't initialize the memory addresses that were in the program image, since they were
        // initialized in the MemoryProgram chip.
        let hint_addresses = hint_addrs.clone();
        let program_memory_image = minimal_executor.program().memory_image.clone();
        let memory_init_events = touched_addresses
            .clone()
            .into_iter()
            .filter(move |addr| !program_memory_image.contains_key(addr))
            .filter(move |addr| !hint_addresses.contains(addr))
            .map(|addr| MemoryInitializeFinalizeEvent::initialize(addr, 0));
        let global_memory_initialize_events =
            global_memory_initialize_events.chain(memory_init_events);

        // Ensure all the hinted addresses are initialized.
        touched_addresses.extend(hint_addrs);

        // Finalize the memory addresses that were touched during execution.
        let global_memory_finalize_events =
            global_memory_finalize_events.chain(touched_addresses.into_iter().map(move |addr| {
                let entry = minimal_executor.get_memory_value(addr);
                MemoryInitializeFinalizeEvent::finalize(addr, entry.value, entry.clk)
            }));

        (
            global_memory_initialize_events.sorted_by_key(|event| event.addr),
            global_memory_finalize_events.sorted_by_key(|event| event.addr),
        )
    }
}

#[derive(Debug, Clone)]
pub struct TouchedAddresses {
    inner: Arc<Mutex<HashSet<u64>>>,
}

impl TouchedAddresses {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashSet::new())) }
    }

    pub fn extend(&self, addresses: impl IntoIterator<Item = u64>) {
        self.inner.lock().unwrap().extend(addresses);
    }

    pub fn take(self) -> HashSet<u64> {
        std::mem::take(&mut *self.inner.lock().unwrap())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FinalVmState {
    pub registers: [MemoryRecord; 32],
    pub timestamp: u64,
    pub pc: u64,
    pub exit_code: u32,
    pub public_value_digest: [u32; PV_DIGEST_NUM_WORDS],
    pub proof_nonce: [u32; PROOF_NONCE_NUM_WORDS],
}

impl FinalVmState {
    pub fn new<'a, 'b>(vm: &'a CoreVM<'b>) -> Self {
        let registers = *vm.registers();
        let timestamp = vm.clk();
        let pc = vm.pc();
        let exit_code = vm.exit_code();
        let public_value_digest = vm.public_value_digest;
        let proof_nonce = vm.proof_nonce;

        Self { registers, timestamp, pc, exit_code, public_value_digest, proof_nonce }
    }
}

#[derive(Debug, Clone)]
pub struct FinalVmStateLock {
    inner: Arc<OnceLock<FinalVmState>>,
}

impl FinalVmStateLock {
    pub fn new() -> Self {
        Self { inner: Arc::new(OnceLock::new()) }
    }

    pub fn set(&self, state: FinalVmState) {
        self.inner.set(state).expect("final vm state already set");
    }

    pub fn get(&self) -> Option<&FinalVmState> {
        self.inner.get()
    }
}

/// An artifact of precompile events, and the range of indices to index into.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecompileArtifactSlice {
    pub artifact: Artifact,
    pub start_idx: usize,
    pub end_idx: usize,
}

/// A lightweight container for the precompile events in a shard.
///
/// Rather than actually holding all of the events, the events are represented as `Artifact`s with
/// start and end indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferredEvents(pub HashMap<SyscallCode, Vec<PrecompileArtifactSlice>>);

impl DeferredEvents {
    /// Defer all events in an ExecutionRecord by uploading each precompile in chunks.
    pub async fn defer_record<A: ArtifactClient>(
        record: ExecutionRecord,
        client: &A,
        split_opts: SplitOpts,
    ) -> Result<DeferredEvents, TaskError> {
        let mut deferred: HashMap<SyscallCode, Vec<PrecompileArtifactSlice>> = HashMap::new();
        let mut futures = Vec::new();
        for (code, events) in record.precompile_events.events.iter() {
            let threshold = split_opts.syscall_threshold[*code];
            futures.extend(
                events
                    .chunks(threshold)
                    .map(|chunk| {
                        let client = client.clone();
                        let artifact = client.create_artifact().unwrap();
                        async move {
                            client.upload(&artifact, chunk).await.unwrap();

                            (*code, artifact.clone(), chunk.len())
                        }
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let res =
            await_scoped_vec(futures).await.map_err(|e| TaskError::Fatal(anyhow::anyhow!(e)))?;
        for (code, artifact, count) in res {
            deferred.entry(code).or_default().push(PrecompileArtifactSlice {
                artifact,
                start_idx: 0,
                end_idx: count,
            });
        }
        Ok(DeferredEvents(deferred))
    }

    /// Create an empty DeferredEvents.
    pub fn empty() -> Self {
        Self(HashMap::new())
    }

    /// Append the events from another DeferredEvents to self. Analogous to
    /// `ExecutionRecord::append`.
    pub async fn append(&mut self, other: DeferredEvents, client: &impl ArtifactClient) {
        for (code, events) in other.0 {
            // Add task references for artifacts so they are not cleaned up before they are fully split.
            for PrecompileArtifactSlice { artifact, .. } in &events {
                if let Err(e) = client.add_ref(artifact, CONTROLLER_PRECOMPILE_ARTIFACT_REF).await {
                    tracing::error!("Failed to add ref to artifact {:?}: {:?}", artifact, e);
                }
            }
            self.0.entry(code).or_default().extend(events);
        }
    }

    /// Split the DeferredEvents into multiple TraceData. Similar to `ExecutionRecord::split`.
    pub async fn split(
        &mut self,
        last: bool,
        opts: SplitOpts,
        client: &impl ArtifactClient,
    ) -> Vec<TraceData> {
        let mut shards = Vec::new();
        let keys = self.0.keys().cloned().collect::<Vec<_>>();
        for code in keys {
            let threshold = opts.syscall_threshold[code];
            // self.0[code] contains uploaded artifacts with start and end indices. start is
            // initially 0. Create shards of precompiles from self.0[code] up to
            // threshold, then update new [start, end) indices for future splits. If
            // last is true, don't leave any remainder.
            loop {
                let mut count = 0;
                // Loop through until we've found enough precompiles, and remove from self.0[code].
                // `index` will be set such that artifacts [0, index) will be made into a shard.
                let mut index = 0;
                for (i, artifact_slice) in self.0[&code].iter().enumerate() {
                    let PrecompileArtifactSlice { start_idx, end_idx, .. } = artifact_slice;
                    count += end_idx - start_idx;
                    // Break if we've found enough or it's the last Artifact and `last` is true.
                    if count >= threshold || (last && i == self.0[&code].len() - 1) {
                        index = i + 1;
                        break;
                    }
                }
                // If not enough was found, break.
                if index == 0 {
                    break;
                }
                // Otherwise remove the artifacts and handle remainder of last artifact if there is
                // any.
                let mut artifacts =
                    self.0.get_mut(&code).unwrap().drain(..index).collect::<Vec<_>>();
                // For each artifact, add refs for the range needed in prove_shard, and then remove
                // the controller ref if it's been fully split.
                for (i, slice) in artifacts.iter().enumerate() {
                    let PrecompileArtifactSlice { artifact, start_idx, end_idx } = slice;
                    if let Err(e) =
                        client.add_ref(artifact, &format!("{:?}_{:?}", start_idx, end_idx)).await
                    {
                        tracing::error!("Failed to add ref to artifact {}: {:?}", artifact, e);
                    }
                    // If there's a remainder, don't remove the controller ref yet.
                    if i == artifacts.len() - 1 && count > threshold {
                        break;
                    }
                    if let Err(e) = client
                        .remove_ref(
                            artifact,
                            ArtifactType::UnspecifiedArtifactType,
                            CONTROLLER_PRECOMPILE_ARTIFACT_REF,
                        )
                        .await
                    {
                        tracing::error!("Failed to remove ref to artifact {}: {:?}", artifact, e);
                    }
                }
                // If there's extra in the last artifact, truncate it and leave it in the front of
                // self.0[code].
                if count > threshold {
                    let mut new_range = artifacts.last().cloned().unwrap();
                    new_range.start_idx = new_range.end_idx - (count - threshold);
                    artifacts[index - 1].end_idx = new_range.start_idx;
                    self.0.get_mut(&code).unwrap().insert(0, new_range);
                }
                shards.push(TraceData::Precompile(artifacts, code));
            }
        }
        shards
    }
}

pub struct DeferredMessage {
    pub task_id: TaskId,
    pub record: Artifact,
}

pub struct SpawnProveOutput {
    pub deferred_message: Option<DeferredMessage>,
    pub proof_data: ProofData,
}

use futures::{
    prelude::*,
    stream::{FuturesOrdered, FuturesUnordered},
};
use slop_futures::queue::WorkerQueue;
use sp1_recursion_circuit::{
    machine::{SP1CompressWitnessValues, SP1DeferredWitnessValues, SP1RecursionWitnessValues},
    InnerSC,
};
use std::{
    collections::{BTreeMap, VecDeque},
    env,
    ops::Range,
    sync::Arc,
};

use slop_algebra::AbstractField;
use slop_baby_bear::BabyBear;
use sp1_core_executor::{
    subproof::SubproofVerifier, ExecutionError, ExecutionRecord, ExecutionReport, Executor,
    Program, SP1Context, SP1CoreOpts, SP1ReduceProof,
};
use sp1_core_machine::{
    executor::{MachineExecutor, MachineExecutorBuilder},
    io::SP1Stdin,
};
use sp1_primitives::io::SP1PublicValues;
use sp1_recursion_executor::ExecutionRecord as RecursionRecord;
use sp1_stark::{
    prover::{MachineProverError, MachineProvingKey},
    BabyBearPoseidon2, MachineVerifierError, MachineVerifyingKey, ShardProof, Word,
};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::{
    components::SP1ProverComponents, error::SP1ProverError, recursion::SP1RecursionProver, CoreSC,
    HashableKey, SP1CircuitWitness, SP1CoreProof, SP1CoreProofData, SP1Prover, SP1VerifyingKey,
};

#[derive(Debug, Clone, Copy)]
pub struct LocalProverOpts {
    pub core_opts: SP1CoreOpts,
    pub records_capacity_buffer: usize,
    pub num_record_workers: usize,
    pub num_recursion_executors: usize,
}

impl Default for LocalProverOpts {
    fn default() -> Self {
        let core_opts = SP1CoreOpts::default();

        const DEFAULT_RECORDS_CAPACITY_BUFFER: usize = 1;
        let records_capacity_buffer = env::var("SP1_PROVER_RECORDS_CAPACITY_BUFFER")
            .unwrap_or_else(|_| DEFAULT_RECORDS_CAPACITY_BUFFER.to_string())
            .parse::<usize>()
            .unwrap_or(DEFAULT_RECORDS_CAPACITY_BUFFER);

        const DEFAULT_NUM_RECORD_WORKERS: usize = 8;
        let num_record_workers = env::var("SP1_PROVER_NUM_RECORD_WORKERS")
            .unwrap_or_else(|_| DEFAULT_NUM_RECORD_WORKERS.to_string())
            .parse::<usize>()
            .unwrap_or(DEFAULT_NUM_RECORD_WORKERS);

        const DEFAULT_NUM_RECURSION_EXECUTORS: usize = 16;
        let num_recursion_executors = env::var("SP1_PROVER_NUM_RECURSION_EXECUTORS")
            .unwrap_or_else(|_| DEFAULT_NUM_RECURSION_EXECUTORS.to_string())
            .parse::<usize>()
            .unwrap_or(DEFAULT_NUM_RECURSION_EXECUTORS);

        Self { core_opts, records_capacity_buffer, num_record_workers, num_recursion_executors }
    }
}

pub struct LocalProver<C: SP1ProverComponents> {
    prover: SP1Prover<C>,
    executor: MachineExecutor<BabyBear>,
    records_task_capacity: usize,
    reduce_batch_size: usize,
    recursion_batch_size: usize,
    num_recursion_executors: usize,
}

impl<C: SP1ProverComponents> LocalProver<C> {
    pub fn new(prover: SP1Prover<C>, opts: LocalProverOpts) -> Self {
        let records_task_capacity =
            prover.core().num_prover_workers() * opts.records_capacity_buffer;
        let executor = MachineExecutorBuilder::new(opts.core_opts, opts.num_record_workers).build();

        let reduce_batch_size = prover.recursion().max_reduce_arity();
        let recursion_batch_size = prover.recursion().recursion_batch_size();
        Self {
            prover,
            executor,
            records_task_capacity,
            reduce_batch_size,
            recursion_batch_size,
            num_recursion_executors: opts.num_recursion_executors,
        }
    }

    pub fn execute(
        self: Arc<Self>,
        elf: &[u8],
        stdin: &SP1Stdin,
        mut context: SP1Context,
    ) -> Result<(SP1PublicValues, ExecutionReport), ExecutionError> {
        context.subproof_verifier = Some(self);
        let opts = SP1CoreOpts::default();
        let mut runtime = Executor::with_context_and_elf(opts, context, elf);

        runtime.write_vecs(&stdin.buffer);
        for (proof, vkey) in stdin.proofs.iter() {
            runtime.write_proof(proof.clone(), vkey.clone());
        }
        runtime.run_fast()?;
        Ok((SP1PublicValues::from(&runtime.state.public_values_stream), runtime.report))
    }

    /// Get a reference to the underlying [SP1Prover]
    #[inline]
    #[must_use]
    pub fn prover(&self) -> &SP1Prover<C> {
        &self.prover
    }

    /// Get a reference to the underlying [MachineExecutor]
    #[inline]
    #[must_use]
    pub fn executor(&self) -> &MachineExecutor<BabyBear> {
        &self.executor
    }

    /// Generate shard proofs which split up and prove the valid execution of a RISC-V program with
    /// the core prover. Uses the provided context.
    pub async fn prove_core(
        self: Arc<Self>,
        pk: Arc<MachineProvingKey<C::CoreComponents>>,
        program: Program,
        stdin: SP1Stdin,
        mut context: SP1Context<'static>,
    ) -> Result<SP1CoreProof, SP1ProverError> {
        context.subproof_verifier = Some(Arc::new(self.clone()));

        let (records_tx, mut records_rx) =
            mpsc::channel::<ExecutionRecord>(self.records_task_capacity);

        let prover = self.clone();

        let program = Arc::new(program);

        // let (proofs_tx, mut proofs_rx) = mpsc::channel(self.prover_task_capacity);
        let shard_proofs = tokio::spawn(async move {
            let mut shard_proofs = Vec::new();
            let mut tasks = FuturesOrdered::new();
            loop {
                tokio::select! {
                    Some(record) = records_rx.recv() => {
                        let span = tracing::debug_span!("prove core shard").entered();
                        let handle = prover
                            .prover
                            .core()
                            .prove_shard(pk.clone(), record);
                        span.exit();
                        tasks.push_back(handle);
                    }
                    Some(result) = tasks.next() => {
                        let proof = result.map_err(SP1ProverError::CoreProverError)?;
                        shard_proofs.push(proof);
                    }
                    else => {
                        break;
                    }
                }
            }
            Result::<_, SP1ProverError>::Ok(shard_proofs)
        });

        // Run the machine executor.
        let prover = self.clone();
        let inputs = stdin.clone();
        let output = tokio::spawn(async move {
            prover.executor.execute(program, inputs, context, records_tx).await
        });

        // Wait for the executor to finish.
        let output = output.await.unwrap().map_err(SP1ProverError::CoreExecutorError)?;
        let pv_stream = output.public_value_stream;
        let cycles = output.cycles;
        let public_values = SP1PublicValues::from(&pv_stream);
        let shard_proofs = shard_proofs.await.unwrap()?;

        // Check for high cycle count.
        Self::check_for_high_cycles(cycles);

        Ok(SP1CoreProof { proof: SP1CoreProofData(shard_proofs), stdin, public_values, cycles })
    }

    fn check_for_high_cycles(cycles: u64) {
        if cycles > 100_000_000 {
            tracing::warn!(
                    "High cycle count detected ({}M cycles). For better performance, consider using the Succinct Prover Network: https://docs.succinct.xyz/generating-proofs/prover-network",
                    cycles / 1_000_000
                );
        }
    }

    /// Generate shard proofs which split up and prove the valid execution of a RISC-V program with
    /// the core prover. Uses the provided context.
    pub async fn compress(
        self: Arc<Self>,
        vk: &SP1VerifyingKey,
        proof: SP1CoreProof,
        deferred_proofs: Vec<SP1ReduceProof<InnerSC>>,
    ) -> Result<SP1ReduceProof<InnerSC>, SP1ProverError> {
        // Initialize the recursion tree channels.
        let (recursion_tree_tx, mut recursion_tree_rx) =
            mpsc::unbounded_channel::<RecursionProof>();

        // Spawn the executor workers
        let (prove_task_tx, mut prove_task_rx) = mpsc::unbounded_channel::<ProveTask<C>>();

        let mut recursion_executors = Vec::new();
        for _ in 0..self.num_recursion_executors {
            let (executor_tx, mut executor_rx) = mpsc::unbounded_channel();
            recursion_executors.push(executor_tx);
            let prover = self.clone();
            let prove_task_tx = prove_task_tx.clone();
            tokio::task::spawn_blocking(move || {
                while let Some(task) = executor_rx.blocking_recv() {
                    let ExecuteTask { input, range } = task;
                    let keys = prover.prover().recursion().keys(&input);
                    let record = prover.prover().recursion().execute(&input).unwrap();
                    let prove_task = ProveTask { keys, range, record };
                    prove_task_tx.send(prove_task).unwrap();
                }
            });
        }
        drop(prove_task_tx);
        let recursion_executors = Arc::new(WorkerQueue::new(recursion_executors));

        // Get the first layer inputs
        let inputs = self.get_first_layer_inputs(
            vk,
            &proof.proof.0,
            &deferred_proofs,
            self.recursion_batch_size,
        );

        let full_range = 0..inputs.len();

        // Spawn the recursion tasks for the core shards.
        let executors = recursion_executors.clone();
        tokio::spawn(async move {
            for (i, input) in inputs.into_iter().enumerate() {
                // Get an executor for the input
                let executor = executors.clone().pop().await.unwrap();
                let range = i..i + 1;
                executor.send(ExecuteTask { input, range }).unwrap();
            }
        });

        // Spawn the prover controller task
        let prover = self.clone();
        let tree_tx = recursion_tree_tx.clone();
        tokio::spawn(async move {
            let mut setup_and_prove_tasks = FuturesUnordered::new();
            let mut prove_tasks = FuturesUnordered::new();
            loop {
                tokio::select! {
                    Some(task) = prove_task_rx.recv() => {
                        let ProveTask { keys, range, record } = task;
                        if let Some((pk, vk)) = keys {
                            let span = tracing::debug_span!("prove compress shard").entered();
                            let handle = prover.prover().recursion().prove_shard(pk, record)
                            .map_ok(move |proof|{
                                let proof = SP1ReduceProof { vk, proof };
                                RecursionProof { shard_range: range, proof }
                            });
                            prove_tasks.push(handle);
                            span.exit();
                        }
                        else {
                        let span = tracing::debug_span!("prove compress shard").entered();
                        let handle = prover.prover().recursion().setup_and_prove_shard(record.program.clone(), None, record)
                            .map_ok(move |(vk, proof)|  {
                            let proof = SP1ReduceProof { vk, proof };

                            RecursionProof { shard_range: range, proof }
                          });
                          span.exit();
                          setup_and_prove_tasks.push(handle);
                        }
                    }
                    Some(result) = setup_and_prove_tasks.next() => {
                        let proof = result.unwrap();
                        tree_tx.send(proof).unwrap();
                    }
                    Some(result) = prove_tasks.next() => {
                        let proof = result.unwrap();
                        tree_tx.send(proof).unwrap();
                    }
                    else => {
                        break;
                    }
                }
            }
        });

        // Reduce the proofs in the tree.
        let mut reduce_batch_size = self.reduce_batch_size;
        let mut full_range = full_range;
        while reduce_batch_size > 1 {
            let mut recursion_tree = RecursionTree::new(reduce_batch_size);
            let proofs = recursion_tree
                .reduce_proofs(&full_range, &mut recursion_tree_rx, recursion_executors.clone())
                .await
                .unwrap();
            if reduce_batch_size == 2 {
                let (cache_total_calls, cache_hits, cache_hit_rate) =
                    self.prover.recursion().recursion_program_cache_stats();
                tracing::debug!(
                    "Recursion program cache stats: total calls: {}, hits: {}, hit rate: {}",
                    cache_total_calls,
                    cache_hits,
                    cache_hit_rate
                );
                return Ok(proofs[0].clone());
            }
            full_range = 0..proofs.len();
            reduce_batch_size /= 2;
            // Split the proof into tasks and send them
            for (i, proof) in proofs.into_iter().enumerate() {
                let proof = RecursionProof { shard_range: i..i + 1, proof };
                recursion_tree_tx.send(proof).unwrap();
            }
        }
        drop(recursion_tree_tx);

        Err(SP1ProverError::RecursionProverError(MachineProverError::ProverClosed))
    }

    /// Generate the inputs for the first layer of recursive proofs.
    #[allow(clippy::type_complexity)]
    pub fn get_first_layer_inputs<'a>(
        &'a self,
        vk: &'a SP1VerifyingKey,
        shard_proofs: &[ShardProof<InnerSC>],
        deferred_proofs: &[SP1ReduceProof<InnerSC>],
        batch_size: usize,
    ) -> Vec<SP1CircuitWitness> {
        let (deferred_inputs, deferred_digest) =
            self.get_deferred_inputs(&vk.vk, deferred_proofs, batch_size);

        let is_complete = shard_proofs.len() == 1 && deferred_proofs.is_empty();
        let core_inputs = self.get_recursion_core_inputs(
            vk,
            shard_proofs,
            batch_size,
            is_complete,
            deferred_digest,
        );

        let mut inputs = Vec::new();
        inputs.extend(deferred_inputs.into_iter().map(SP1CircuitWitness::Deferred));
        inputs.extend(core_inputs.into_iter().map(SP1CircuitWitness::Core));
        inputs
    }

    #[inline]
    pub fn get_deferred_inputs<'a>(
        &'a self,
        vk: &'a MachineVerifyingKey<CoreSC>,
        deferred_proofs: &[SP1ReduceProof<InnerSC>],
        batch_size: usize,
    ) -> (Vec<SP1DeferredWitnessValues<InnerSC>>, [BabyBear; 8]) {
        self.get_deferred_inputs_with_initial_digest(
            vk,
            deferred_proofs,
            [BabyBear::zero(); 8],
            batch_size,
        )
    }

    pub fn get_deferred_inputs_with_initial_digest<'a>(
        &'a self,
        vk: &'a MachineVerifyingKey<CoreSC>,
        deferred_proofs: &[SP1ReduceProof<InnerSC>],
        initial_deferred_digest: [BabyBear; 8],
        batch_size: usize,
    ) -> (Vec<SP1DeferredWitnessValues<InnerSC>>, [BabyBear; 8]) {
        // Prepare the inputs for the deferred proofs recursive verification.
        let mut deferred_digest = initial_deferred_digest;
        let mut deferred_inputs = Vec::new();

        for batch in deferred_proofs.chunks(batch_size) {
            let vks_and_proofs =
                batch.iter().cloned().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();

            let input = SP1CompressWitnessValues { vks_and_proofs, is_complete: true };
            // let input = self.make_merkle_proofs(input);
            // let SP1CompressWitnessValues { compress_val } = input;

            deferred_inputs.push(SP1DeferredWitnessValues {
                vks_and_proofs: input.vks_and_proofs,
                // vk_merkle_data: merkle_val,
                start_reconstruct_deferred_digest: deferred_digest,
                is_complete: false,
                sp1_vk_digest: vk.hash_babybear(),
                end_pc: vk.pc_start,
                end_shard: BabyBear::one(),
                end_execution_shard: BabyBear::one(),
                init_addr_word: Word([BabyBear::zero(); 2]),
                finalize_addr_word: Word([BabyBear::zero(); 2]),
                committed_value_digest: [[BabyBear::zero(); 4]; 8],
                deferred_proofs_digest: [BabyBear::zero(); 8],
            });

            deferred_digest = SP1RecursionProver::<C::RecursionComponents>::hash_deferred_proofs(
                deferred_digest,
                batch,
            );
        }
        (deferred_inputs, deferred_digest)
    }

    pub fn get_recursion_core_inputs(
        &self,
        vk: &SP1VerifyingKey,
        shard_proofs: &[ShardProof<CoreSC>],
        batch_size: usize,
        is_complete: bool,
        deferred_digest: [BabyBear; 8],
    ) -> Vec<SP1RecursionWitnessValues<CoreSC>> {
        let mut core_inputs = Vec::new();

        // Prepare the inputs for the recursion programs.
        for (batch_idx, batch) in shard_proofs.chunks(batch_size).enumerate() {
            let proofs = batch.to_vec();

            core_inputs.push(SP1RecursionWitnessValues {
                vk: vk.vk.clone(),
                shard_proofs: proofs.clone(),
                is_complete,
                is_first_shard: batch_idx == 0,
                // vk_root: self.recursion_vk_root,
                reconstruct_deferred_digest: deferred_digest,
            });
        }
        core_inputs
    }
}

impl<C: SP1ProverComponents> SubproofVerifier for LocalProver<C> {
    fn verify_deferred_proof(
        &self,
        proof: &SP1ReduceProof<BabyBearPoseidon2>,
        vk: &MachineVerifyingKey<BabyBearPoseidon2>,
        vk_hash: [u32; 8],
        committed_value_digest: [u32; 8],
    ) -> Result<(), MachineVerifierError<BabyBearPoseidon2>> {
        self.prover.verify_deferred_proof(proof, vk, vk_hash, committed_value_digest)
    }
}

pub struct RecursionTree {
    map: BTreeMap<usize, RangeProofs>,
    batch_size: usize,
}

#[derive(Clone, Debug)]
struct RangeProofs {
    shard_range: Range<usize>,
    proofs: VecDeque<SP1ReduceProof<InnerSC>>,
}

impl RangeProofs {
    pub fn new(shard_range: Range<usize>, proofs: VecDeque<SP1ReduceProof<InnerSC>>) -> Self {
        Self { shard_range, proofs }
    }

    pub fn push_right(&mut self, proof: RecursionProof) {
        assert_eq!(proof.shard_range.end, self.shard_range.start);
        self.shard_range = proof.shard_range.start..self.shard_range.end;
        self.proofs.push_front(proof.proof);
    }

    pub fn push_left(&mut self, proof: RecursionProof) {
        assert_eq!(proof.shard_range.start, self.shard_range.end);
        self.shard_range = self.shard_range.start..proof.shard_range.end;
        self.proofs.push_back(proof.proof);
    }

    pub fn split_off(&mut self, at: usize) -> Option<Self> {
        if at >= self.proofs.len() {
            return None;
        }
        let proofs = self.proofs.split_off(at);
        let end_point = std::cmp::min(self.shard_range.end, self.shard_range.start + at);
        let split_range = end_point..self.shard_range.end;
        self.shard_range = self.shard_range.start..end_point;
        Some(Self { shard_range: split_range, proofs })
    }

    pub fn push_both(&mut self, middle: RecursionProof, right: Self) {
        assert_eq!(middle.shard_range.start, self.shard_range.end);
        assert_eq!(right.shard_range.start, middle.shard_range.end);
        // Push the middle to the queue.
        self.proofs.push_back(middle.proof);
        // Append the right proofs to the queue.
        for proof in right.proofs {
            self.proofs.push_back(proof);
        }
        // Update the shard range.
        self.shard_range = self.shard_range.start..right.shard_range.end;
    }

    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    pub fn is_complete(&self, full_range: &Range<usize>) -> bool {
        &self.shard_range == full_range
    }

    pub fn into_witness(
        self,
        full_range: &Range<usize>,
    ) -> (Range<usize>, SP1CompressWitnessValues<InnerSC>) {
        let is_complete = self.is_complete(full_range);
        let vks_and_proofs =
            self.proofs.into_iter().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();
        (self.shard_range, SP1CompressWitnessValues { vks_and_proofs, is_complete })
    }
}

impl RecursionTree {
    /// Create a new recursion tree.
    fn new(batch_size: usize) -> Self {
        Self { map: BTreeMap::new(), batch_size }
    }

    /// Insert a new range of proofs into the tree.
    fn insert(&mut self, proofs: RangeProofs) {
        self.map.insert(proofs.shard_range.start, proofs);
    }

    /// Get the sibling of a proof.
    fn sibling(&mut self, proof: &RecursionProof) -> Option<Sibling> {
        // Check for a left sibling
        if let Some(previous) = self.map.range(0..proof.shard_range.start).next_back() {
            let (start, proofs) = previous;
            let start = *start;
            let proofs = proofs.clone();

            if proofs.shard_range.end == proof.shard_range.start {
                let left = self.map.remove(&start).unwrap();
                // Check for a right sibling.
                if let Some(right) = self.map.remove(&proof.shard_range.end) {
                    return Some(Sibling::Both(left, right));
                } else {
                    return Some(Sibling::Left(left));
                }
            }
        }
        // If there is no left sibling, check for a right sibling.
        if let Some(right) = self.map.remove(&proof.shard_range.end) {
            return Some(Sibling::Right(right));
        }

        // No sibling found.
        None
    }

    async fn reduce_proofs(
        &mut self,
        full_range: &Range<usize>,
        proofs_rx: &mut UnboundedReceiver<RecursionProof>,
        recursion_executors: Arc<WorkerQueue<UnboundedSender<ExecuteTask>>>,
    ) -> Result<Vec<SP1ReduceProof<InnerSC>>, SP1ProverError> {
        // Populate the recursion proofs into the tree until we reach the reduce batch size.
        while let Some(proof) = proofs_rx.recv().await {
            if proof.is_complete(full_range) {
                return Ok(vec![proof.proof]);
            }

            // Check if there is a neighboring range.
            if let Some(sibling) = self.sibling(&proof) {
                let mut proofs = match sibling {
                    Sibling::Left(mut proofs) => {
                        proofs.push_left(proof);
                        proofs
                    }
                    Sibling::Right(mut proofs) => {
                        proofs.push_right(proof);
                        proofs
                    }
                    Sibling::Both(mut proofs, right) => {
                        proofs.push_both(proof, right);
                        proofs
                    }
                };

                // Check for proofs to split and put back the reminder.
                let split = proofs.split_off(self.batch_size);
                if let Some(split) = split {
                    self.insert(split);
                }

                if proofs.is_complete(full_range) {
                    return Ok(proofs.proofs.into_iter().collect());
                }

                if proofs.len() > self.batch_size {
                    tracing::error!("Proofs are larger than the batch size: {:?}", proofs.len());
                    panic!("Proofs are larger than the batch size: {:?}", proofs.len());
                }

                if proofs.len() == self.batch_size {
                    // Compress all the proofs into a single proof.
                    let (range, input) = proofs.into_witness(full_range);
                    let input = SP1CircuitWitness::Compress(input);
                    // Wait for an executor to be available.
                    let executor = recursion_executors.clone().pop().await.unwrap();
                    executor.send(ExecuteTask { input, range }).ok();
                } else {
                    self.insert(proofs);
                }
            } else {
                // If there is no neighboring range, add the proof to the tree.
                let RecursionProof { shard_range, proof } = proof;
                let mut queue = VecDeque::with_capacity(self.batch_size);
                queue.push_back(proof);
                let proofs = RangeProofs::new(shard_range, queue);
                self.insert(proofs);
            }
        }
        Err(SP1ProverError::RecursionProverError(MachineProverError::ProverClosed))
    }
}

#[derive(Debug, Clone)]
struct RecursionProof {
    shard_range: Range<usize>,
    proof: SP1ReduceProof<InnerSC>,
}

impl RecursionProof {
    fn is_complete(&self, full_range: &Range<usize>) -> bool {
        &self.shard_range == full_range
    }
}

enum Sibling {
    Left(RangeProofs),
    Right(RangeProofs),
    Both(RangeProofs, RangeProofs),
}

struct ExecuteTask {
    range: Range<usize>,
    input: SP1CircuitWitness,
}

#[allow(clippy::type_complexity)]
struct ProveTask<C: SP1ProverComponents> {
    keys: Option<(Arc<MachineProvingKey<C::RecursionComponents>>, MachineVerifyingKey<InnerSC>)>,
    range: Range<usize>,
    record: RecursionRecord<BabyBear>,
}

#[cfg(test)]
pub mod tests {
    use tracing::Instrument;

    use slop_algebra::PrimeField32;

    use crate::components::CpuSP1ProverComponents;
    use crate::SP1ProverBuilder;

    use super::*;

    use anyhow::Result;

    #[cfg(test)]
    use serial_test::serial;
    #[cfg(test)]
    use sp1_core_machine::utils::setup_logger;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Test {
        Core,
        Compress,
    }

    pub async fn test_e2e_prover<C: SP1ProverComponents>(
        prover: Arc<LocalProver<C>>,
        elf: &[u8],
        stdin: SP1Stdin,
        test_kind: Test,
    ) -> Result<()> {
        let (pk, program, vk) = prover
            .prover()
            .core()
            .setup(elf)
            .instrument(tracing::debug_span!("setup").or_current())
            .await;

        let pk = unsafe { pk.into_inner() };

        let core_proof = prover
            .clone()
            .prove_core(pk, program, stdin, SP1Context::default())
            .instrument(tracing::info_span!("prove core"))
            .await
            .unwrap();

        let cycles = core_proof.cycles as usize;
        let num_shards = core_proof.proof.0.len();
        tracing::info!("Cycles: {}, number of shards: {}", cycles, num_shards);

        // Verify the proof
        let core_proof_data = SP1CoreProofData(core_proof.proof.0.clone());
        prover.prover().verify(&core_proof_data, &vk).unwrap();

        if let Test::Core = test_kind {
            return Ok(());
        }

        // Make the compress proof.
        let compress_proof = prover
            .clone()
            .compress(&vk, core_proof, vec![])
            .instrument(tracing::info_span!("compress"))
            .await
            .unwrap();

        // Verify the compress proof
        prover.prover().verify_compressed(&compress_proof, &vk).unwrap();

        Ok(())
    }

    /// Tests an end-to-end workflow of proving a program across the entire proof generation
    /// pipeline.
    ///
    #[tokio::test]
    #[serial]
    async fn test_e2e() -> Result<()> {
        let elf = test_artifacts::SSZ_WITHDRAWALS_ELF;
        setup_logger();

        let sp1_prover = SP1ProverBuilder::<CpuSP1ProverComponents>::cpu().build().await;
        let opts = LocalProverOpts::default();
        let prover = Arc::new(LocalProver::new(sp1_prover, opts));

        test_e2e_prover::<CpuSP1ProverComponents>(prover, elf, SP1Stdin::default(), Test::Compress)
            .await
    }

    #[tokio::test]
    #[serial]
    async fn test_deferred_compress() -> Result<()> {
        setup_logger();

        let sp1_prover = SP1ProverBuilder::<CpuSP1ProverComponents>::cpu().build().await;
        let opts = LocalProverOpts::default();
        let prover = Arc::new(LocalProver::new(sp1_prover, opts));

        // Test program which proves the Keccak-256 hash of various inputs.
        let keccak_elf = test_artifacts::KECCAK256_ELF;

        // Test program which verifies proofs of a vkey and a list of committed inputs.
        let verify_elf = test_artifacts::VERIFY_PROOF_ELF;

        tracing::info!("setup keccak elf");
        let (keccak_pk, keccak_program, keccak_vk) = prover.prover().core().setup(keccak_elf).await;

        let keccak_pk = unsafe { keccak_pk.into_inner() };

        tracing::info!("setup verify elf");
        let (verify_pk, verify_program, verify_vk) = prover.prover().core().setup(verify_elf).await;

        let verify_pk = unsafe { verify_pk.into_inner() };

        tracing::info!("prove subproof 1");
        let mut stdin = SP1Stdin::new();
        stdin.write(&1usize);
        stdin.write(&vec![0u8, 0, 0]);
        let deferred_proof_1 = prover
            .clone()
            .prove_core(keccak_pk.clone(), keccak_program.clone(), stdin, Default::default())
            .await?;
        let pv_1 = deferred_proof_1.public_values.as_slice().to_vec().clone();

        // Generate a second proof of keccak of various inputs.
        tracing::info!("prove subproof 2");
        let mut stdin = SP1Stdin::new();
        stdin.write(&3usize);
        stdin.write(&vec![0u8, 1, 2]);
        stdin.write(&vec![2, 3, 4]);
        stdin.write(&vec![5, 6, 7]);
        let deferred_proof_2 =
            prover.clone().prove_core(keccak_pk, keccak_program, stdin, Default::default()).await?;
        let pv_2 = deferred_proof_2.public_values.as_slice().to_vec().clone();

        // Generate recursive proof of first subproof.
        tracing::info!("compress subproof 1");
        let deferred_reduce_1 =
            prover.clone().compress(&keccak_vk, deferred_proof_1, vec![]).await?;
        prover.prover().verify_compressed(&deferred_reduce_1, &keccak_vk)?;

        // Generate recursive proof of second subproof.
        tracing::info!("compress subproof 2");
        let deferred_reduce_2 =
            prover.clone().compress(&keccak_vk, deferred_proof_2, vec![]).await?;
        prover.prover().verify_compressed(&deferred_reduce_2, &keccak_vk)?;

        // Run verify program with keccak vkey, subproofs, and their committed values.
        let mut stdin = SP1Stdin::new();
        let vkey_digest = keccak_vk.hash_babybear();
        let vkey_digest: [u32; 8] = vkey_digest
            .iter()
            .map(|n| n.as_canonical_u32())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        stdin.write(&vkey_digest);
        stdin.write(&vec![pv_1.clone(), pv_2.clone(), pv_2.clone()]);
        stdin.write_proof(deferred_reduce_1.clone(), keccak_vk.vk.clone());
        stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());
        stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());

        tracing::info!("proving verify program (core)");
        let verify_proof =
            prover.clone().prove_core(verify_pk, verify_program, stdin, Default::default()).await?;

        // Generate recursive proof of verify program
        tracing::info!("compress verify program");
        let verify_reduce = prover
            .clone()
            .compress(
                &verify_vk,
                verify_proof,
                vec![deferred_reduce_1, deferred_reduce_2.clone(), deferred_reduce_2],
            )
            .await?;

        tracing::info!("verify verify program");
        prover.prover().verify_compressed(&verify_reduce, &verify_vk)?;

        Ok(())
    }
}

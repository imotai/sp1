use crate::SP1NormalizeInputShape;
use futures::{
    prelude::*,
    stream::{FuturesOrdered, FuturesUnordered},
};
use slop_algebra::{AbstractField, PrimeField, PrimeField32};
use slop_bn254::Bn254Fr;
use slop_futures::queue::WorkerQueue;
use sp1_core_executor::{
    subproof::SubproofVerifier, ExecutionError, ExecutionRecord, ExecutionReport, Executor,
    Program, SP1Context, SP1CoreOpts, SP1RecursionProof,
};
use sp1_core_machine::{executor::MachineExecutor, io::SP1Stdin};
use sp1_hypercube::{
    prover::{MachineProvingKey, MemoryPermit},
    MachineVerifierConfigError, MachineVerifyingKey, ShardProof,
};
use sp1_primitives::{io::SP1PublicValues, SP1Field, SP1GlobalContext, SP1OuterGlobalContext};
use sp1_recursion_circuit::{
    machine::{SP1DeferredWitnessValues, SP1NormalizeWitnessValues, SP1ShapedWitnessValues},
    utils::{
        koalabear_bytes_to_bn254, koalabears_proof_nonce_to_bn254, koalabears_to_bn254,
        words_to_bytes,
    },
    witness::{OuterWitness, Witnessable},
    InnerSC,
};
use sp1_recursion_executor::{ExecutionRecord as RecursionRecord, RecursionPublicValues};
use sp1_recursion_gnark_ffi::{
    Groth16Bn254Proof, Groth16Bn254Prover, PlonkBn254Proof, PlonkBn254Prover,
};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, VecDeque},
    env,
    ops::Range,
    path::Path,
    sync::Arc,
};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tracing::Instrument;

use crate::{
    components::SP1ProverComponents, error::SP1ProverError, recursion::SP1RecursionProver, CoreSC,
    HashableKey, OuterSC, SP1CircuitWitness, SP1CoreProof, SP1CoreProofData, SP1Prover,
    SP1VerifyingKey,
};

#[derive(Debug, Clone)]
pub struct LocalProverOpts {
    pub core_opts: SP1CoreOpts,
    pub records_buffer_size: u64,
    pub num_record_workers: usize,
    pub num_recursion_executors: usize,
}

impl Default for LocalProverOpts {
    fn default() -> Self {
        let core_opts = SP1CoreOpts::default();

        let sysinfo = sysinfo::System::new_all();
        let total_memory = sysinfo.total_memory();
        let used_memory = sysinfo.used_memory();
        let free_memory = total_memory - used_memory;

        tracing::info!("Free memory at prover init: {} bytes", free_memory);

        // Allow half the available memory for tracegen by default.
        let records_buffer_size = free_memory / 2;

        // Reserve ~12Gb of memory for records by default.
        let records_buffer_size = env::var("SP1_PROVER_RECORDS_CAPACITY_BUFFER")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(records_buffer_size);

        const DEFAULT_NUM_RECORD_WORKERS: usize = 2;
        let num_record_workers = env::var("SP1_PROVER_NUM_RECORD_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_RECORD_WORKERS);

        const DEFAULT_NUM_RECURSION_EXECUTORS: usize = 4;
        let num_recursion_executors = env::var("SP1_PROVER_NUM_RECURSION_EXECUTORS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_RECURSION_EXECUTORS);

        Self { core_opts, records_buffer_size, num_record_workers, num_recursion_executors }
    }
}

pub struct LocalProver<C: SP1ProverComponents> {
    prover: SP1Prover<C>,
    executor: MachineExecutor<SP1Field>,
    compose_batch_size: usize,
    normalize_batch_size: usize,
    num_recursion_executors: usize,
}

impl<C: SP1ProverComponents> LocalProver<C> {
    pub fn new(prover: SP1Prover<C>, opts: LocalProverOpts) -> Self {
        let executor =
            MachineExecutor::new(opts.records_buffer_size, opts.num_record_workers, opts.core_opts);

        let compose_batch_size = prover.recursion().max_compose_arity();
        let normalize_batch_size = prover.recursion().normalize_batch_size();
        Self {
            prover,
            executor,
            compose_batch_size,
            normalize_batch_size,
            num_recursion_executors: opts.num_recursion_executors,
        }
    }

    pub fn execute(
        self: Arc<Self>,
        elf: &[u8],
        stdin: &SP1Stdin,
        mut context: SP1Context,
    ) -> Result<(SP1PublicValues, [u8; 32], ExecutionReport), ExecutionError> {
        context.subproof_verifier = Some(self.clone());
        let opts = self.executor.opts().clone();
        let program = Arc::new(Program::from(elf).unwrap());

        let mut runtime = Executor::with_context(program, opts, context);
        runtime.maybe_setup_profiler(elf);

        runtime.write_vecs(&stdin.buffer);
        for (proof, vkey) in stdin.proofs.iter() {
            runtime.write_proof(proof.clone(), vkey.clone());
        }
        runtime.run_fast()?;

        let mut committed_value_digest = [0u8; 32];

        runtime.record.public_values.committed_value_digest.iter().enumerate().for_each(
            |(i, word)| {
                let bytes = word.to_le_bytes();
                committed_value_digest[i * 4..(i + 1) * 4].copy_from_slice(&bytes[0..4]);
            },
        );

        Ok((
            SP1PublicValues::from(&runtime.state.public_values_stream),
            committed_value_digest,
            runtime.report,
        ))
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
    pub fn executor(&self) -> &MachineExecutor<SP1Field> {
        &self.executor
    }

    /// Generate shard proofs which split up and prove the valid execution of a RISC-V program with
    /// the core prover. Uses the provided context.
    pub async fn prove_core(
        self: Arc<Self>,
        pk: Arc<MachineProvingKey<SP1GlobalContext, C::CoreComponents>>,
        program: Arc<Program>,
        stdin: SP1Stdin,
        mut context: SP1Context<'static>,
    ) -> Result<SP1CoreProof, SP1ProverError> {
        let dummy_vk = pk.as_ref().vk.clone();

        context.subproof_verifier = Some(Arc::new(self.clone()));

        let (records_tx, mut records_rx) =
            mpsc::unbounded_channel::<(ExecutionRecord, Option<MemoryPermit>)>();

        let prover = self.clone();

        let shard_proofs = tokio::spawn(async move {
            let mut shape_count = 0;
            let mut shard_proofs = Vec::new();
            let mut prove_shard_task = FuturesOrdered::new();
            loop {
                tokio::select! {
                    // Accquire a permit and start the exeuction.
                    Some((record, permit)) = records_rx.recv() => {
                        let shape = prover.prover.core().core_shape_from_record(&record).unwrap();

                        let proof = async {
                            let proof = prover
                                .prover
                                .core()
                                .prove_shard(pk.clone(), record)
                                .await;

                            drop(permit);

                            proof
                        };

                        prove_shard_task.push_back(proof);

                        if shape_count < 3 {
                            let prover = prover.clone();
                            let dummy_vk = dummy_vk.clone();
                            tokio::spawn(async move {
                                let normalize_shape = SP1NormalizeInputShape {
                                    proof_shapes: vec![shape],
                                    max_log_row_count: prover.prover.recursion_prover.core_verifier.max_log_row_count(),
                                    log_blowup: prover.prover.recursion_prover.core_verifier.fri_config().log_blowup,
                                    log_stacking_height: prover.prover.recursion_prover.core_verifier.log_stacking_height() as usize,
                                };
                                let witness = normalize_shape.dummy_input(SP1VerifyingKey { vk: dummy_vk } );
                                prover.prover.recursion_prover.normalize_program(&witness);
                            });
                            shape_count += 1;
                        }
                    }
                    Some(proof) = prove_shard_task.next() => {
                        shard_proofs.push(proof);
                    }
                    else => {
                        break;
                    }
                }
            }
            Result::<_, SP1ProverError>::Ok(shard_proofs)
        }.in_current_span());

        // Run the machine executor with the generated nonce.
        let prover = self.clone();
        let inputs = stdin.clone();

        // Wait for the executor to finish.
        let output = prover
            .executor
            .execute(program, inputs, context, records_tx)
            .await
            .map_err(SP1ProverError::CoreExecutorError)?;

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
        deferred_proofs: Vec<SP1RecursionProof<SP1GlobalContext, InnerSC>>,
    ) -> Result<SP1RecursionProof<SP1GlobalContext, InnerSC>, SP1ProverError> {
        // Initialize the recursion tree channels.
        let (compress_tree_tx, mut compress_tree_rx) = mpsc::unbounded_channel::<RecursionProof>();

        // Spawn the executor workers
        let (prove_task_tx, mut prove_task_rx) = mpsc::unbounded_channel::<ProveTask<C>>();

        let mut recursion_executors = Vec::new();
        for _ in 0..self.num_recursion_executors {
            let (executor_tx, mut executor_rx) = mpsc::unbounded_channel();
            recursion_executors.push(executor_tx);
            let prover = self.clone();
            let prove_task_tx = prove_task_tx.clone();
            let parent = tracing::Span::current();
            tokio::task::spawn_blocking(move || {
                let _guard = parent.enter();
                while let Some(task) = executor_rx.blocking_recv() {
                    let ExecuteTask { input, range } = task;
                    let keys = prover.prover().recursion().keys(&input);
                    let span = tracing::debug_span!("execute recursion program").entered();
                    let record = prover.prover().recursion().execute(input).unwrap();
                    span.exit();
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
            self.normalize_batch_size,
        );

        let full_range = 0..inputs.len();

        // Spawn the recursion tasks for the core shards.
        let executors = recursion_executors.clone();
        tokio::spawn(
            async move {
                for (i, input) in inputs.into_iter().enumerate() {
                    // Get an executor for the input
                    let executor = executors.clone().pop().await.unwrap();
                    let range = i..i + 1;
                    executor.send(ExecuteTask { input, range }).unwrap();
                }
            }
            .in_current_span(),
        );

        // Spawn the prover controller task
        let prover = self.clone();
        let tree_tx = compress_tree_tx.clone();
        tokio::spawn(async move {
            let mut setup_and_prove_tasks = FuturesUnordered::new();
            let mut prove_tasks = FuturesUnordered::new();
            loop {
                tokio::select! {
                    Some(task) = prove_task_rx.recv() => {
                        let ProveTask { keys, range, record } = task;
                        if let Some((pk, vk)) = keys {
                            let span = tracing::debug_span!("prove compress shard").entered();
                            let handle = async {
                                let proof = prover.prover().recursion().prove_shard(pk, record).await;
                                let proof = SP1RecursionProof { vk, proof };
                                RecursionProof { shard_range: range, proof }
                            };

                            prove_tasks.push(handle);
                            span.exit();
                        }
                        else {
                            let span = tracing::debug_span!("prove compress shard").entered();
                            let handle = async {
                                let (vk, proof) = prover.prover().recursion().setup_and_prove_shard(record.program.clone(), None, record).await;
                                let proof = SP1RecursionProof { vk, proof };
                                RecursionProof { shard_range: range, proof }
                            };

                            setup_and_prove_tasks.push(handle);
                            span.exit();
                        }
                    }
                    Some(proof) = setup_and_prove_tasks.next() => {
                        tree_tx.send(proof).unwrap();
                    }
                    Some(proof) = prove_tasks.next() => {
                        tree_tx.send(proof).unwrap();
                    }
                    else => {
                        break;
                    }
                }
            }
        }.in_current_span());

        // Reduce the proofs in the tree.
        let mut reduce_batch_size = self.compose_batch_size;
        let mut full_range = full_range;
        while reduce_batch_size > 1 {
            let mut compress_tree = CompressTree::new(reduce_batch_size);
            let proofs = compress_tree
                .reduce_proofs(&full_range, &mut compress_tree_rx, recursion_executors.clone())
                .await
                .unwrap();
            if reduce_batch_size == 2 {
                let (cache_total_calls, cache_hits, cache_hit_rate) =
                    self.prover.recursion().normalize_program_cache_stats();
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
                compress_tree_tx.send(proof).unwrap();
            }
        }
        drop(compress_tree_tx);

        unreachable!("todo explain this")
    }

    #[tracing::instrument(name = "prove shrink", skip_all)]
    pub async fn shrink(
        &self,
        compressed_proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
    ) -> Result<SP1RecursionProof<SP1GlobalContext, InnerSC>, SP1ProverError> {
        // Make the compress proof.
        let SP1RecursionProof { vk: compressed_vk, proof: compressed_proof } = compressed_proof;
        let input = SP1ShapedWitnessValues {
            vks_and_proofs: vec![(compressed_vk.clone(), compressed_proof)],
            is_complete: true,
        };

        let input = self.prover.recursion().make_merkle_proofs(input);
        let witness = SP1CircuitWitness::Shrink(input);

        // Run key initialization and witness execution in parallel
        let key_task = async {
            let (_, vk) = self.prover.recursion().get_shrink_keys_async().await;
            Ok::<_, SP1ProverError>(vk)
        };

        let execute_task = async {
            self.prover
                .recursion()
                .execute(witness)
                .map_err(|e| SP1ProverError::Other(format!("Runtime panicked: {e}")))
        };

        let (vk, record) = tokio::try_join!(key_task, execute_task)?;

        let proof = self.prover.recursion().prove_shrink(record).await;

        Ok(SP1RecursionProof { vk, proof })
    }

    #[tracing::instrument(name = "prove wrap", skip_all)]
    pub async fn wrap(
        &self,
        shrunk_proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
    ) -> Result<SP1RecursionProof<SP1OuterGlobalContext, OuterSC>, SP1ProverError> {
        let SP1RecursionProof { vk: compressed_vk, proof: compressed_proof } = shrunk_proof;
        let input = SP1ShapedWitnessValues {
            vks_and_proofs: vec![(compressed_vk.clone(), compressed_proof)],
            is_complete: true,
        };
        let input = self.prover.recursion().make_merkle_proofs(input);
        let witness = SP1CircuitWitness::Wrap(input);
        // Run key initialization and witness execution in parallel
        let key_task = async {
            let (_, vk) = self.prover.recursion().get_wrap_keys_async().await;
            Ok::<_, SP1ProverError>(vk)
        };

        let execute_task = async {
            self.prover
                .recursion()
                .execute(witness)
                .map_err(|e| SP1ProverError::Other(format!("Runtime panicked: {e}")))
        };

        let (vk, record) = tokio::try_join!(key_task, execute_task)?;

        let proof = self.prover.recursion().prove_wrap(record).await;

        Ok(SP1RecursionProof { vk, proof })
    }

    #[tracing::instrument(name = "prove wrap plonk bn254", skip_all)]
    pub async fn wrap_plonk_bn254(
        &self,
        wrap_proof: SP1RecursionProof<SP1OuterGlobalContext, OuterSC>,
        build_dir: &Path,
    ) -> PlonkBn254Proof {
        let SP1RecursionProof { vk: wrap_vk, proof: wrap_proof } = wrap_proof;
        let input = SP1ShapedWitnessValues {
            vks_and_proofs: vec![(wrap_vk.clone(), wrap_proof.clone())],
            is_complete: true,
        };

        let pv: &RecursionPublicValues<SP1Field> = wrap_proof.public_values.as_slice().borrow();

        let vkey_hash = koalabears_to_bn254(&pv.sp1_vk_digest);
        let committed_values_digest_bytes: [SP1Field; 32] =
            words_to_bytes(&pv.committed_value_digest).try_into().unwrap();
        let committed_values_digest = koalabear_bytes_to_bn254(&committed_values_digest_bytes);
        let exit_code = Bn254Fr::from_canonical_u32(pv.exit_code.as_canonical_u32());
        let vk_root = koalabears_to_bn254(&pv.vk_root);
        let proof_nonce = koalabears_proof_nonce_to_bn254(&pv.proof_nonce);
        let mut witness = OuterWitness::default();
        input.write(&mut witness);
        witness.write_committed_values_digest(committed_values_digest);
        witness.write_vkey_hash(vkey_hash);
        witness.write_exit_code(exit_code);
        witness.write_vk_root(vk_root);
        witness.write_proof_nonce(proof_nonce);
        let prover = PlonkBn254Prover::new();
        let proof = prover.prove(witness, build_dir.to_path_buf());

        // Verify the proof.
        prover
            .verify(
                &proof,
                &vkey_hash.as_canonical_biguint(),
                &committed_values_digest.as_canonical_biguint(),
                &exit_code.as_canonical_biguint(),
                &vk_root.as_canonical_biguint(),
                &proof_nonce.as_canonical_biguint(),
                build_dir,
            )
            .expect("Failed to verify proof");

        proof
    }

    #[tracing::instrument(name = "prove wrap plonk bn254", skip_all)]
    pub async fn wrap_groth16_bn254(
        &self,
        wrap_proof: SP1RecursionProof<SP1OuterGlobalContext, OuterSC>,
        build_dir: &Path,
    ) -> Groth16Bn254Proof {
        let SP1RecursionProof { vk: wrap_vk, proof: wrap_proof } = wrap_proof;
        let input = SP1ShapedWitnessValues {
            vks_and_proofs: vec![(wrap_vk.clone(), wrap_proof.clone())],
            is_complete: true,
        };

        let pv: &RecursionPublicValues<SP1Field> = wrap_proof.public_values.as_slice().borrow();

        let vkey_hash = koalabears_to_bn254(&pv.sp1_vk_digest);
        let committed_values_digest_bytes: [SP1Field; 32] =
            words_to_bytes(&pv.committed_value_digest).try_into().unwrap();
        let committed_values_digest = koalabear_bytes_to_bn254(&committed_values_digest_bytes);
        let exit_code = Bn254Fr::from_canonical_u32(pv.exit_code.as_canonical_u32());
        let proof_nonce = koalabears_proof_nonce_to_bn254(&pv.proof_nonce);
        let vk_root = koalabears_to_bn254(&pv.vk_root);
        let mut witness = OuterWitness::default();
        input.write(&mut witness);
        witness.write_committed_values_digest(committed_values_digest);
        witness.write_vkey_hash(vkey_hash);
        witness.write_exit_code(exit_code);
        witness.write_vk_root(vk_root);
        witness.write_proof_nonce(proof_nonce);
        let prover = Groth16Bn254Prover::new();
        let proof = prover.prove(witness, build_dir.to_path_buf());

        // Verify the proof.
        prover
            .verify(
                &proof,
                &vkey_hash.as_canonical_biguint(),
                &committed_values_digest.as_canonical_biguint(),
                &exit_code.as_canonical_biguint(),
                &vk_root.as_canonical_biguint(),
                &proof_nonce.as_canonical_biguint(),
                build_dir,
            )
            .expect("Failed to verify wrap proof");

        proof
    }

    /// Generate the inputs for the first layer of recursive proofs.
    #[allow(clippy::type_complexity)]
    pub fn get_first_layer_inputs<'a>(
        &'a self,
        vk: &'a SP1VerifyingKey,
        shard_proofs: &[ShardProof<SP1GlobalContext, InnerSC>],
        deferred_proofs: &[SP1RecursionProof<SP1GlobalContext, InnerSC>],
        batch_size: usize,
    ) -> Vec<SP1CircuitWitness> {
        // We arbitrarily grab the page prot and nonce values from the first shard because it should
        // be the same for all shards.
        let pv: &RecursionPublicValues<SP1Field> =
            shard_proofs[0].public_values.as_slice().borrow();
        let proof_nonce = pv.proof_nonce;

        let (deferred_inputs, deferred_digest) =
            self.get_deferred_inputs(&vk.vk, deferred_proofs, batch_size, proof_nonce);

        let is_complete = shard_proofs.len() == 1 && deferred_proofs.is_empty();
        let core_inputs = self.get_normalize_witnesses(
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
        vk: &'a MachineVerifyingKey<SP1GlobalContext, CoreSC>,
        deferred_proofs: &[SP1RecursionProof<SP1GlobalContext, InnerSC>],
        batch_size: usize,
        proof_nonce: [SP1Field; 4],
    ) -> (Vec<SP1DeferredWitnessValues<SP1GlobalContext, InnerSC>>, [SP1Field; 8]) {
        self.get_deferred_inputs_with_initial_digest(
            vk,
            deferred_proofs,
            [SP1Field::zero(); 8],
            batch_size,
            proof_nonce,
        )
    }

    pub fn get_deferred_inputs_with_initial_digest<'a>(
        &'a self,
        vk: &'a MachineVerifyingKey<SP1GlobalContext, CoreSC>,
        deferred_proofs: &[SP1RecursionProof<SP1GlobalContext, InnerSC>],
        initial_deferred_digest: [SP1Field; 8],
        batch_size: usize,
        proof_nonce: [SP1Field; 4],
    ) -> (Vec<SP1DeferredWitnessValues<SP1GlobalContext, InnerSC>>, [SP1Field; 8]) {
        // Prepare the inputs for the deferred proofs recursive verification.
        let mut deferred_digest = initial_deferred_digest;
        let mut deferred_inputs = Vec::new();

        for batch in deferred_proofs.chunks(batch_size) {
            let vks_and_proofs =
                batch.iter().cloned().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();

            let input = SP1ShapedWitnessValues { vks_and_proofs, is_complete: true };
            let input = self.prover.recursion().make_merkle_proofs(input);

            deferred_inputs.push(SP1DeferredWitnessValues {
                vks_and_proofs: input.compress_val.vks_and_proofs,
                vk_merkle_data: input.merkle_val,
                start_reconstruct_deferred_digest: deferred_digest,
                sp1_vk_digest: vk.hash_koalabear(),
                end_pc: vk.pc_start,
                proof_nonce,
            });

            deferred_digest = SP1RecursionProver::<C>::hash_deferred_proofs(deferred_digest, batch);
        }
        (deferred_inputs, deferred_digest)
    }

    pub fn get_normalize_witnesses(
        &self,
        vk: &SP1VerifyingKey,
        shard_proofs: &[ShardProof<SP1GlobalContext, CoreSC>],
        batch_size: usize,
        is_complete: bool,
        deferred_digest: [SP1Field; 8],
    ) -> Vec<SP1NormalizeWitnessValues<SP1GlobalContext, CoreSC>> {
        let mut core_inputs = Vec::new();

        // Prepare the inputs for the recursion programs.
        for batch in shard_proofs.chunks(batch_size) {
            let proofs = batch.to_vec();

            core_inputs.push(SP1NormalizeWitnessValues {
                vk: vk.vk.clone(),
                shard_proofs: proofs.clone(),
                is_complete,
                vk_root: self.prover.recursion().recursion_vk_root,
                reconstruct_deferred_digest: deferred_digest,
            });
        }
        core_inputs
    }
}

impl<C: SP1ProverComponents> SubproofVerifier for LocalProver<C> {
    fn verify_deferred_proof(
        &self,
        proof: &SP1RecursionProof<SP1GlobalContext, InnerSC>,
        vk: &MachineVerifyingKey<SP1GlobalContext, CoreSC>,
        vk_hash: [u64; 4],
        committed_value_digest: [u64; 4],
    ) -> Result<(), MachineVerifierConfigError<SP1GlobalContext, CoreSC>> {
        self.prover.verify_deferred_proof(proof, vk, vk_hash, committed_value_digest)
    }
}

pub struct CompressTree {
    map: BTreeMap<usize, RangeProofs>,
    batch_size: usize,
}

#[derive(Clone, Debug)]
struct RangeProofs {
    shard_range: Range<usize>,
    proofs: VecDeque<SP1RecursionProof<SP1GlobalContext, InnerSC>>,
}

impl RangeProofs {
    pub fn new(
        shard_range: Range<usize>,
        proofs: VecDeque<SP1RecursionProof<SP1GlobalContext, InnerSC>>,
    ) -> Self {
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
    ) -> (Range<usize>, SP1ShapedWitnessValues<SP1GlobalContext, InnerSC>) {
        let is_complete = self.is_complete(full_range);
        let vks_and_proofs =
            self.proofs.into_iter().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();
        (self.shard_range, SP1ShapedWitnessValues { vks_and_proofs, is_complete })
    }
}

impl CompressTree {
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
    ) -> Result<Vec<SP1RecursionProof<SP1GlobalContext, InnerSC>>, SP1ProverError> {
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

                if proofs.len() > self.batch_size {
                    tracing::error!("Proofs are larger than the batch size: {:?}", proofs.len());
                    panic!("Proofs are larger than the batch size: {:?}", proofs.len());
                }

                if proofs.len() == self.batch_size || proofs.is_complete(full_range) {
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

        unreachable!("todo explain this")
    }
}

#[derive(Debug, Clone)]
struct RecursionProof {
    shard_range: Range<usize>,
    proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
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
    keys: Option<(
        Arc<MachineProvingKey<SP1GlobalContext, C::RecursionComponents>>,
        MachineVerifyingKey<SP1GlobalContext, InnerSC>,
    )>,
    range: Range<usize>,
    record: RecursionRecord<SP1Field>,
}

#[cfg(all(test, feature = "unsound"))]
pub mod tests {
    use sp1_core_executor::RetainedEventsPreset;
    use tracing::Instrument;

    use slop_algebra::PrimeField32;

    use crate::{
        build::{try_build_groth16_bn254_artifacts_dev, try_build_plonk_bn254_artifacts_dev},
        components::CpuSP1ProverComponents,
        SP1ProverBuilder,
    };

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
        Shrink,
        Wrap,
        OnChain,
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

        let public_values = core_proof.public_values.clone();

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

        if let Test::Compress = test_kind {
            return Ok(());
        }

        let shrink_proof = prover.clone().shrink(compress_proof).await.unwrap();
        prover.prover().verify_shrink(&shrink_proof, &vk).unwrap();

        if let Test::Shrink = test_kind {
            return Ok(());
        }

        let wrap_proof = prover.clone().wrap(shrink_proof).await.unwrap();
        prover.prover().verify_wrap_bn254(&wrap_proof, &vk).unwrap();

        if let Test::Wrap = test_kind {
            return Ok(());
        }

        let artifacts_dir = try_build_plonk_bn254_artifacts_dev(&wrap_proof.vk, &wrap_proof.proof);
        let plonk_bn254_proof = prover.wrap_plonk_bn254(wrap_proof.clone(), &artifacts_dir).await;

        prover
            .prover()
            .verify_plonk_bn254(&plonk_bn254_proof, &vk, &public_values, &artifacts_dir)
            .unwrap();

        let artifacts_dir =
            try_build_groth16_bn254_artifacts_dev(&wrap_proof.vk, &wrap_proof.proof);
        let groth16_bn254_proof =
            prover.wrap_groth16_bn254(wrap_proof.clone(), &artifacts_dir).await;

        prover
            .prover()
            .verify_groth16_bn254(&groth16_bn254_proof, &vk, &public_values, &artifacts_dir)
            .unwrap();

        Ok(())
    }

    /// Tests an end-to-end workflow of proving a program across the entire proof generation
    /// pipeline.
    #[tokio::test]
    #[serial]
    async fn test_e2e() -> Result<()> {
        let elf = test_artifacts::FIBONACCI_ELF;
        setup_logger();

        let sp1_prover = SP1ProverBuilder::<CpuSP1ProverComponents>::new()
            .without_vk_verification()
            .build()
            .await;
        let opts = LocalProverOpts {
            core_opts: SP1CoreOpts {
                retained_events_presets: [RetainedEventsPreset::Sha256].into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let prover = Arc::new(LocalProver::new(sp1_prover, opts));

        test_e2e_prover::<CpuSP1ProverComponents>(prover, &elf, SP1Stdin::default(), Test::OnChain)
            .await
    }

    #[tokio::test]
    #[serial]
    async fn test_deferred_compress() -> Result<()> {
        setup_logger();

        let sp1_prover = SP1ProverBuilder::<CpuSP1ProverComponents>::new()
            .without_vk_verification()
            .build()
            .await;
        let opts = LocalProverOpts::default();
        let prover = Arc::new(LocalProver::new(sp1_prover, opts));

        // Test program which proves the Keccak-256 hash of various inputs.
        let keccak_elf = test_artifacts::KECCAK256_ELF;

        // Test program which verifies proofs of a vkey and a list of committed inputs.
        let verify_elf = test_artifacts::VERIFY_PROOF_ELF;

        tracing::info!("setup keccak elf");
        let (keccak_pk, keccak_program, keccak_vk) =
            prover.prover().core().setup(&keccak_elf).await;

        let keccak_pk = unsafe { keccak_pk.into_inner() };

        tracing::info!("setup verify elf");
        let (verify_pk, verify_program, verify_vk) =
            prover.prover().core().setup(&verify_elf).await;

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
        let vkey_digest = keccak_vk.hash_koalabear();
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

        prover.prover().verify(&verify_proof.proof, &verify_vk)?;

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

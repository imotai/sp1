use crate::{
    build::{try_build_groth16_bn254_artifacts_dev, try_build_plonk_bn254_artifacts_dev},
    components::RecursionProver,
    components::WrapProver as InnerWrapProver,
    recursion::{
        compose_program_from_input, recursive_verifier, shrink_program_from_input,
        wrap_program_from_input, RecursionConfig,
    },
    shapes::SP1RecursionProofShape,
    worker::{
        CommonProverInput, ProverMetrics, RangeProofs, RawTaskRequest, TaskContext, TaskError,
        TaskMetadata,
    },
    CompressAir, CoreSC, HashableKey, InnerSC, OuterSC, SP1CircuitWitness, SP1ProverComponents,
};
use slop_algebra::PrimeField32;
use slop_algebra::{AbstractField, PrimeField};
use slop_bn254::Bn254Fr;
use slop_challenger::IopCtx;
use slop_futures::pipeline::{
    AsyncEngine, AsyncWorker, BlockingEngine, BlockingWorker, Chain, Pipeline, SubmitError,
    SubmitHandle,
};
use sp1_hypercube::{
    air::SP1CorePublicValues,
    inner_perm,
    prover::{AirProver, MachineProverComponents, MachineProvingKey, ProverSemaphore},
    MachineProof, MachineVerifier, MachineVerifyingKey, SP1RecursionProof, ShardProof, DIGEST_SIZE,
};
use sp1_primitives::{SP1ExtensionField, SP1Field, SP1GlobalContext, SP1OuterGlobalContext};
use sp1_prover_types::{Artifact, ArtifactClient, ArtifactId};
use sp1_recursion_circuit::{
    basefold::merkle_tree::MerkleTree,
    machine::{
        SP1CompressWithVKeyWitnessValues, SP1MerkleProofWitnessValues, SP1NormalizeWitnessValues,
        SP1ShapedWitnessValues,
    },
    utils::{
        koalabear_bytes_to_bn254, koalabears_proof_nonce_to_bn254, koalabears_to_bn254,
        words_to_bytes,
    },
    witness::{OuterWitness, Witnessable},
    WrapConfig,
};
use sp1_recursion_compiler::config::InnerConfig;
use sp1_recursion_executor::{
    shape::RecursionShape, Block, ExecutionRecord, Executor, RecursionProgram,
    RecursionPublicValues,
};
use sp1_recursion_gnark_ffi::{Groth16Bn254Prover, PlonkBn254Prover};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
};
use tokio::sync::oneshot;
use tracing::Instrument;

/// Configuration for the core prover.
#[derive(Debug, Clone)]
pub struct SP1RecursionProverConfig {
    /// The number of prepare reduce workers.
    pub num_prepare_reduce_workers: usize,
    /// The buffer size for the prepare reduce.
    pub prepare_reduce_buffer_size: usize,
    /// The number of recursion executor workers.
    pub num_recursion_executor_workers: usize,
    /// The buffer size for the recursion executor.
    pub recursion_executor_buffer_size: usize,
    /// The number of recursion prover workers.
    pub num_recursion_prover_workers: usize,
    /// The buffer size for the recursion prover.
    pub recursion_prover_buffer_size: usize,
    /// The maximum compose arity.
    pub max_compose_arity: usize,
    /// Whether to verify the recursion vks
    pub vk_verification: bool,
    /// Whether or not to verify the proof result at the end.
    pub verify_intermediates: bool,
}

pub struct ReduceTaskRequest {
    pub range_proofs: RangeProofs,
    pub is_complete: bool,
    pub output: Artifact,
    pub context: TaskContext,
}

impl ReduceTaskRequest {
    pub fn from_raw(request: RawTaskRequest) -> Result<Self, TaskError> {
        let RawTaskRequest { inputs, mut outputs, context } = request;
        let is_complete = inputs[0].id().parse::<bool>().map_err(|e| TaskError::Fatal(e.into()))?;
        let range_proofs = RangeProofs::from_artifacts(&inputs[1..])?;
        let output =
            outputs.pop().ok_or(TaskError::Fatal(anyhow::anyhow!("No output artifact")))?;
        Ok(ReduceTaskRequest { range_proofs, is_complete, output, context })
    }

    pub fn into_raw(self) -> Result<RawTaskRequest, TaskError> {
        let ReduceTaskRequest { range_proofs, is_complete, output, context } = self;
        let is_complete_artifact = Artifact::from(is_complete.to_string());
        let mut inputs = Vec::with_capacity(2 * range_proofs.len() + 2);
        inputs.push(is_complete_artifact);
        inputs.extend(range_proofs.as_artifacts());
        let raw_task_request = RawTaskRequest { inputs, outputs: vec![output], context };
        Ok(raw_task_request)
    }
}

pub struct PrepareReduceTaskWorker<A, C: SP1ProverComponents> {
    prover_data: Arc<RecursionProverData<C>>,
    artifact_client: A,
}

impl<A: ArtifactClient, C: SP1ProverComponents>
    AsyncWorker<ReduceTaskRequest, Result<RecursionTask, TaskError>>
    for PrepareReduceTaskWorker<A, C>
{
    #[tracing::instrument(level = "trace", name = "prepare_reduce_task", skip(self, input))]
    async fn call(&self, input: ReduceTaskRequest) -> Result<RecursionTask, TaskError> {
        let ReduceTaskRequest { range_proofs, is_complete, output, .. } = input;

        let program = self.prover_data.compose_programs.get(&range_proofs.len()).cloned().ok_or(
            TaskError::Fatal(anyhow::anyhow!(
                "Compress program not found for arity {}",
                range_proofs.len()
            )),
        )?;

        let witness = range_proofs.download_witness(is_complete, &self.artifact_client).await?;

        let metrics = ProverMetrics::new();
        Ok(RecursionTask { program, witness, output, metrics })
    }
}

pub struct RecursionTask {
    program: Arc<RecursionProgram<SP1Field>>,
    witness: SP1CircuitWitness,
    output: Artifact,
    metrics: ProverMetrics,
}

pub struct RecursionExecutorWorker<C: SP1ProverComponents> {
    compress_verifier: MachineVerifier<SP1GlobalContext, InnerSC, CompressAir<SP1Field>>,
    prover_data: Arc<RecursionProverData<C>>,
}

impl<C: SP1ProverComponents>
    BlockingWorker<Result<RecursionTask, TaskError>, Result<ProveRecursionTask<C>, TaskError>>
    for RecursionExecutorWorker<C>
{
    fn call(
        &self,
        input: Result<RecursionTask, TaskError>,
    ) -> Result<ProveRecursionTask<C>, TaskError> {
        let RecursionTask { program, witness, output, metrics } = input?;

        // Execute the runtime.
        let runtime_span = tracing::debug_span!("execute runtime").entered();
        let mut runtime =
            Executor::<SP1Field, SP1ExtensionField, _>::new(program.clone(), inner_perm());
        runtime.witness_stream = self.prover_data.witness_stream(&witness)?;
        runtime.run().map_err(|e| TaskError::Fatal(e.into()))?;
        let mut record = runtime.record;
        runtime_span.exit();

        // Generate the dependencies.
        tracing::debug_span!("generate dependencies").in_scope(|| {
            self.compress_verifier
                .machine()
                .generate_dependencies(std::iter::once(&mut record), None)
        });

        let keys = tracing::debug_span!("get keys").in_scope(|| match witness {
            SP1CircuitWitness::Core(_) => anyhow::Ok(RecursionKeys::Program(program)),
            SP1CircuitWitness::Compress(input) => {
                let arity = input.vks_and_proofs.len();
                let (pk, vk) = self.prover_data.compose_keys.get(&arity).cloned().ok_or(
                    TaskError::Fatal(anyhow::anyhow!("Compose key not found for arity {}", arity)),
                )?;
                anyhow::Ok(RecursionKeys::Exists(pk, vk))
            }
            _ => unimplemented!(),
        })?;

        Ok(ProveRecursionTask { record, keys, output, metrics })
    }
}

enum RecursionKeys<C: SP1ProverComponents> {
    Exists(
        Arc<MachineProvingKey<SP1GlobalContext, C::RecursionComponents>>,
        MachineVerifyingKey<SP1GlobalContext, RecursionConfig<C>>,
    ),
    Program(Arc<RecursionProgram<SP1Field>>),
}

pub struct ProveRecursionTask<C: SP1ProverComponents> {
    record: ExecutionRecord<SP1Field>,
    keys: RecursionKeys<C>,
    output: Artifact,
    metrics: ProverMetrics,
}

pub struct RecursionProverWorker<A, C: SP1ProverComponents> {
    recursion_prover: Arc<RecursionProver<C>>,
    permits: ProverSemaphore,
    artifact_client: A,
    verify_intermediates: bool,
}

impl<A: ArtifactClient, C: SP1ProverComponents> RecursionProverWorker<A, C> {
    async fn prove_shard(
        &self,
        keys: RecursionKeys<C>,
        record: ExecutionRecord<SP1Field>,
        metrics: ProverMetrics,
    ) -> Result<SP1RecursionProof<SP1GlobalContext, InnerSC>, TaskError> {
        let proof = match keys {
            RecursionKeys::Exists(pk, vk) => {
                let mut challenger = SP1GlobalContext::default_challenger();
                vk.observe_into(&mut challenger);
                let (proof, permit) = self
                    .recursion_prover
                    .prove_shard_with_pk(pk.clone(), record, self.permits.clone(), &mut challenger)
                    .await;
                let duration = permit.release();
                metrics.increment_permit_time(duration);

                if self.verify_intermediates {
                    C::compress_verifier()
                        .verify(&vk, &MachineProof::from(vec![proof.clone()]))
                        .map_err(|e| {
                            TaskError::Retryable(anyhow::anyhow!("compress verify failed: {}", e))
                        })?;
                }
                SP1RecursionProof { vk, proof }
            }
            RecursionKeys::Program(program) => {
                let mut challenger = SP1GlobalContext::default_challenger();
                let (vk, proof, permit) = self
                    .recursion_prover
                    .setup_and_prove_shard(
                        program,
                        record,
                        None,
                        self.permits.clone(),
                        &mut challenger,
                    )
                    .await;
                let duration = permit.release();
                metrics.increment_permit_time(duration);
                if self.verify_intermediates {
                    C::compress_verifier()
                        .verify(&vk, &MachineProof::from(vec![proof.clone()]))
                        .map_err(|e| {
                            TaskError::Retryable(anyhow::anyhow!("compress verify failed: {}", e))
                        })?;
                }
                SP1RecursionProof { vk, proof }
            }
        };

        Ok(proof)
    }
}

impl<A: ArtifactClient, C: SP1ProverComponents>
    AsyncWorker<Result<ProveRecursionTask<C>, TaskError>, Result<TaskMetadata, TaskError>>
    for RecursionProverWorker<A, C>
{
    async fn call(
        &self,
        input: Result<ProveRecursionTask<C>, TaskError>,
    ) -> Result<TaskMetadata, TaskError> {
        // Get the input or return an error
        let ProveRecursionTask { record, keys, output, metrics, .. } = input?;
        // Prove the shard
        let proof = self.prove_shard(keys, record, metrics.clone()).await?;
        // Upload the proof

        self.artifact_client.upload(&output, proof.clone()).await?;
        let metadata = metrics.to_metadata();

        Ok(metadata)
    }
}

type ExecutorEngine<C> = Arc<
    BlockingEngine<
        Result<RecursionTask, TaskError>,
        Result<ProveRecursionTask<C>, TaskError>,
        RecursionExecutorWorker<C>,
    >,
>;

type RecursionProverEngine<A, C> = Arc<
    AsyncEngine<
        Result<ProveRecursionTask<C>, TaskError>,
        Result<TaskMetadata, TaskError>,
        RecursionProverWorker<A, C>,
    >,
>;

type PrepareReduceEngine<A, C> = Arc<
    AsyncEngine<ReduceTaskRequest, Result<RecursionTask, TaskError>, PrepareReduceTaskWorker<A, C>>,
>;

type RecursionProvePipeline<A, C> = Chain<ExecutorEngine<C>, RecursionProverEngine<A, C>>;

type ReducePipeline<A, C> = Chain<PrepareReduceEngine<A, C>, Arc<RecursionProvePipeline<A, C>>>;

pub type RecursionProveSubmitHandle<A, C> = SubmitHandle<RecursionProvePipeline<A, C>>;

pub type ReduceSubmitHandle<A, C> = SubmitHandle<ReducePipeline<A, C>>;

pub struct SP1RecursionProver<A, C: SP1ProverComponents> {
    reduce_pipeline: Arc<ReducePipeline<A, C>>,
    pub shrink_prover: Arc<ShrinkProver<C>>,
    pub wrap_prover: Arc<WrapProver<C>>,
    prover_data: Arc<RecursionProverData<C>>,
    artifact_client: A,
}

impl<A: Clone, C: SP1ProverComponents> Clone for SP1RecursionProver<A, C> {
    fn clone(&self) -> Self {
        Self {
            reduce_pipeline: self.reduce_pipeline.clone(),
            shrink_prover: self.shrink_prover.clone(),
            wrap_prover: self.wrap_prover.clone(),
            prover_data: self.prover_data.clone(),
            artifact_client: self.artifact_client.clone(),
        }
    }
}

impl<A: ArtifactClient, C: SP1ProverComponents> SP1RecursionProver<A, C> {
    pub async fn new(
        config: SP1RecursionProverConfig,
        artifact_client: A,
        (air_prover, air_prover_permits): (Arc<RecursionProver<C>>, ProverSemaphore),
        (shrink_prover, shrink_prover_permits): (Arc<RecursionProver<C>>, ProverSemaphore),
        (wrap_prover, wrap_prover_permits): (Arc<InnerWrapProver<C>>, ProverSemaphore),
    ) -> Self {
        tokio::task::spawn_blocking(move || {
            // Get the reduce shape.
            let reduce_shape =
                SP1RecursionProofShape::compress_proof_shape_from_arity(config.max_compose_arity)
                    .expect("arity not supported");

            // Make the reduce programs and keys.
            let mut compose_programs = BTreeMap::new();
            let mut compose_keys = BTreeMap::new();

            let file = std::fs::File::open("./src/vk_map.bin").ok();

            let allowed_vk_map: BTreeMap<[SP1Field; DIGEST_SIZE], usize> = if config.vk_verification
            {
                file.and_then(|file| bincode::deserialize_from(file).ok()).unwrap_or_else(|| {
                    (0..1 << 18)
                        .map(|i| ([SP1Field::from_canonical_u32(i as u32); DIGEST_SIZE], i))
                        .collect()
                })
            } else {
                // Dummy merkle tree when vk_verification is false.
                (0..1 << 18)
                    .map(|i| ([SP1Field::from_canonical_u32(i as u32); DIGEST_SIZE], i))
                    .collect()
            };

            let compress_verifier = C::compress_verifier();
            let recursive_compress_verifier =
                recursive_verifier::<SP1GlobalContext, _, InnerSC, InnerConfig>(
                    compress_verifier.shard_verifier(),
                );
            let (vk_root, recursion_vk_tree) =
                MerkleTree::<SP1GlobalContext>::commit(allowed_vk_map.keys().copied().collect());
            for arity in 1..=config.max_compose_arity {
                let dummy_input =
                    dummy_compose_input::<C>(&reduce_shape, arity, recursion_vk_tree.height);
                let mut program = compose_program_from_input(
                    &recursive_compress_verifier,
                    config.vk_verification,
                    &dummy_input,
                );
                program.shape = Some(reduce_shape.shape.clone());
                let program = Arc::new(program);

                // Make the reduce keys.
                let (tx, rx) = oneshot::channel();
                tokio::task::spawn({
                    let program = program.clone();
                    let air_prover = air_prover.clone();
                    async move {
                        let permits = ProverSemaphore::new(1);
                        let (pk, vk) = air_prover.setup(program, permits).await;
                        tx.send((pk, vk)).ok();
                    }
                });
                let (pk, vk) = rx.blocking_recv().unwrap();
                let pk = unsafe { pk.into_inner() };
                compose_keys.insert(arity, (pk, vk));
                compose_programs.insert(arity, program);
            }

            let prover_data = Arc::new(RecursionProverData {
                vk_root,
                allowed_vk_map,
                recursion_vk_tree,
                reduce_shape,
                compose_programs,
                compose_keys,
                vk_verification: config.vk_verification,
            });

            let compress_verifier = C::compress_verifier();

            // Initialize the prepare reduce engine.
            let prepare_reduce_workers = (0..config.num_prepare_reduce_workers)
                .map(|_| PrepareReduceTaskWorker {
                    prover_data: prover_data.clone(),
                    artifact_client: artifact_client.clone(),
                })
                .collect();
            let prepare_reduce_engine = Arc::new(AsyncEngine::new(
                prepare_reduce_workers,
                config.prepare_reduce_buffer_size,
            ));

            // Initialize the executor engine.
            let executor_workers = (0..config.num_recursion_executor_workers)
                .map(|_| RecursionExecutorWorker {
                    compress_verifier: compress_verifier.clone(),
                    prover_data: prover_data.clone(),
                })
                .collect();

            let executor_engine = Arc::new(BlockingEngine::new(
                executor_workers,
                config.recursion_executor_buffer_size,
            ));

            // Initialize the prove engine.
            let prove_workers = (0..config.num_recursion_prover_workers)
                .map(|_| RecursionProverWorker {
                    recursion_prover: air_prover.clone(),
                    permits: air_prover_permits.clone(),
                    artifact_client: artifact_client.clone(),
                    verify_intermediates: config.verify_intermediates,
                })
                .collect();
            let prove_engine =
                Arc::new(AsyncEngine::new(prove_workers, config.recursion_prover_buffer_size));

            // Make the recursion pipeline.
            let recursion_pipeline = Arc::new(Chain::new(executor_engine, prove_engine));

            // Make the reduce pipeline.
            let reduce_pipeline = Arc::new(Chain::new(prepare_reduce_engine, recursion_pipeline));

            let shrink_prover = Arc::new(ShrinkProver::new(
                shrink_prover,
                shrink_prover_permits,
                prover_data.clone(),
                config.clone(),
            ));

            let wrap_prover = Arc::new(WrapProver::new(
                wrap_prover,
                wrap_prover_permits,
                prover_data.clone(),
                config.clone(),
                shrink_prover.proving_key.clone(),
            ));

            Self { reduce_pipeline, shrink_prover, wrap_prover, prover_data, artifact_client }
        })
        .await
        .unwrap()
    }

    pub fn recursion_prover_pipeline(&self) -> &Arc<RecursionProvePipeline<A, C>> {
        self.reduce_pipeline.second()
    }

    pub async fn submit_prove_shard(
        &self,
        program: Arc<RecursionProgram<SP1Field>>,
        witness: SP1CircuitWitness,
        output: Artifact,
        metrics: ProverMetrics,
    ) -> Result<RecursionProveSubmitHandle<A, C>, SubmitError> {
        self.recursion_prover_pipeline()
            .submit(Ok(RecursionTask { program, witness, output, metrics }))
            .await
    }

    pub async fn submit_recursion_reduce(
        &self,
        request: RawTaskRequest,
    ) -> Result<ReduceSubmitHandle<A, C>, TaskError> {
        let input = ReduceTaskRequest::from_raw(request)?;
        let handle = self.reduce_pipeline.submit(input).await?;
        Ok(handle)
    }

    pub async fn run_shrink_wrap(&self, request: RawTaskRequest) -> Result<(), TaskError> {
        let RawTaskRequest { inputs, outputs, .. } = request;
        let [compress_proof_artifact] = inputs.try_into().unwrap();
        let [wrap_proof_artifact] = outputs.try_into().unwrap();

        let compress_proof = self
            .artifact_client
            .download(&compress_proof_artifact)
            .instrument(tracing::debug_span!("download compress proof"))
            .await?;

        let shrink_proof = self
            .shrink_prover
            .prove(compress_proof)
            .instrument(tracing::info_span!("prove shrink"))
            .await?;

        tracing::debug_span!("verify shrink proof")
            .in_scope(|| self.shrink_prover.verify(&shrink_proof))?;

        let wrap_proof = self
            .wrap_prover
            .prove(shrink_proof)
            .instrument(tracing::info_span!("prove wrap"))
            .await?;

        tracing::debug_span!("verify wrap proof")
            .in_scope(|| self.wrap_prover.verify(&wrap_proof))?;

        self.artifact_client
            .upload(&wrap_proof_artifact, wrap_proof)
            .instrument(tracing::debug_span!("upload wrap proof"))
            .await?;

        Ok(())
    }

    pub async fn run_groth16(&self, request: RawTaskRequest) -> Result<(), TaskError> {
        let RawTaskRequest { inputs, outputs, .. } = request;
        let [wrap_proof_artifact] = inputs.try_into().unwrap();
        let [groth16_proof_artifact] = outputs.try_into().unwrap();

        let wrap_proof: SP1RecursionProof<SP1OuterGlobalContext, OuterSC> = self
            .artifact_client
            .download(&wrap_proof_artifact)
            .instrument(tracing::debug_span!("download wrap proof"))
            .await?;

        let groth16_proof = tokio::task::spawn_blocking(|| {
            // TODO: Change this after v6.0.0 binary release
            let build_dir =
                try_build_groth16_bn254_artifacts_dev(&wrap_proof.vk, &wrap_proof.proof);
            let SP1RecursionProof { vk, proof } = wrap_proof;
            let input = SP1ShapedWitnessValues {
                vks_and_proofs: vec![(vk, proof.clone())],
                is_complete: true,
            };
            let pv: &RecursionPublicValues<SP1Field> = proof.public_values.as_slice().borrow();
            let vkey_hash = koalabears_to_bn254(&pv.sp1_vk_digest);
            let committed_values_digest_bytes: [SP1Field; 32] =
                words_to_bytes(&pv.committed_value_digest).try_into().unwrap();
            let committed_values_digest = koalabear_bytes_to_bn254(&committed_values_digest_bytes);
            let exit_code = Bn254Fr::from_canonical_u32(pv.exit_code.as_canonical_u32());
            let proof_nonce = koalabears_proof_nonce_to_bn254(&pv.proof_nonce);
            let vk_root = koalabears_to_bn254(&pv.vk_root);
            let witness = {
                let mut witness = OuterWitness::default();
                input.write(&mut witness);
                witness.write_committed_values_digest(committed_values_digest);
                witness.write_vkey_hash(vkey_hash);
                witness.write_exit_code(exit_code);
                witness.write_vk_root(vk_root);
                witness.write_proof_nonce(proof_nonce);
                witness
            };
            let prover = Groth16Bn254Prover::new();
            let proof = prover.prove(witness, build_dir.to_path_buf());
            prover
                .verify(
                    &proof,
                    &vkey_hash.as_canonical_biguint(),
                    &committed_values_digest.as_canonical_biguint(),
                    &exit_code.as_canonical_biguint(),
                    &vk_root.as_canonical_biguint(),
                    &proof_nonce.as_canonical_biguint(),
                    &build_dir,
                )
                .expect("Failed to verify wrap proof");
            proof
        })
        .instrument(tracing::info_span!("prove groth16"))
        .await
        .unwrap();

        self.artifact_client
            .upload(&groth16_proof_artifact, groth16_proof)
            .instrument(tracing::debug_span!("upload groth16 proof"))
            .await?;
        Ok(())
    }

    pub async fn run_plonk(&self, request: RawTaskRequest) -> Result<(), TaskError> {
        let RawTaskRequest { inputs, outputs, .. } = request;
        let [wrap_proof_artifact] = inputs.try_into().unwrap();
        let [plonk_proof_artifact] = outputs.try_into().unwrap();
        let wrap_proof: SP1RecursionProof<SP1OuterGlobalContext, OuterSC> = self
            .artifact_client
            .download(&wrap_proof_artifact)
            .instrument(tracing::debug_span!("download wrap proof"))
            .await?;

        let plonk_proof = tokio::task::spawn_blocking(|| {
            // TODO: Change this after v6.0.0 binary release
            let build_dir = try_build_plonk_bn254_artifacts_dev(&wrap_proof.vk, &wrap_proof.proof);
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
            let witness = {
                let mut witness = OuterWitness::default();
                input.write(&mut witness);
                witness.write_committed_values_digest(committed_values_digest);
                witness.write_vkey_hash(vkey_hash);
                witness.write_exit_code(exit_code);
                witness.write_vk_root(vk_root);
                witness.write_proof_nonce(proof_nonce);
                witness
            };
            let prover = PlonkBn254Prover::new();
            let proof = prover.prove(witness, build_dir.to_path_buf());
            prover
                .verify(
                    &proof,
                    &vkey_hash.as_canonical_biguint(),
                    &committed_values_digest.as_canonical_biguint(),
                    &exit_code.as_canonical_biguint(),
                    &vk_root.as_canonical_biguint(),
                    &proof_nonce.as_canonical_biguint(),
                    &build_dir,
                )
                .expect("Failed to verify proof");
            proof
        })
        .instrument(tracing::info_span!("prove plonk"))
        .await
        .unwrap();

        self.artifact_client
            .upload(&plonk_proof_artifact, plonk_proof)
            .instrument(tracing::debug_span!("upload plonk proof"))
            .await?;
        Ok(())
    }

    #[inline]
    #[must_use]
    pub fn recursion_vk_root(&self) -> [SP1Field; DIGEST_SIZE] {
        self.prover_data.vk_root
    }

    #[must_use]
    pub fn recursion_vk_map(&self) -> &BTreeMap<[SP1Field; DIGEST_SIZE], usize> {
        &self.prover_data.allowed_vk_map
    }

    #[must_use]
    pub fn vk_verification(&self) -> bool {
        self.prover_data.vk_verification
    }

    #[must_use]
    pub fn get_normalize_witness(
        &self,
        common_input: &CommonProverInput,
        proof: &ShardProof<SP1GlobalContext, CoreSC>,
        is_complete: bool,
    ) -> SP1NormalizeWitnessValues<SP1GlobalContext, CoreSC> {
        let pv: &SP1CorePublicValues<SP1Field> = proof.public_values.as_slice().borrow();
        let reconstruct_deferred_digest = pv.deferred_proofs_digest;
        SP1NormalizeWitnessValues {
            vk: common_input.vk.vk.clone(),
            shard_proofs: vec![proof.clone()],
            is_complete,
            vk_root: self.recursion_vk_root(),
            reconstruct_deferred_digest,
            num_deferred_proofs: SP1Field::from_canonical_usize(common_input.num_deferred_proofs),
        }
    }

    pub fn reduce_shape(&self) -> &SP1RecursionProofShape {
        &self.prover_data.reduce_shape
    }
}

type CompressKeys<C> = (
    Arc<MachineProvingKey<SP1GlobalContext, <C as SP1ProverComponents>::RecursionComponents>>,
    MachineVerifyingKey<SP1GlobalContext, RecursionConfig<C>>,
);

pub struct RecursionProverData<C: SP1ProverComponents> {
    vk_root: [SP1Field; DIGEST_SIZE],
    allowed_vk_map: BTreeMap<[SP1Field; DIGEST_SIZE], usize>,
    recursion_vk_tree: MerkleTree<SP1GlobalContext>,
    reduce_shape: SP1RecursionProofShape,
    compose_programs: BTreeMap<usize, Arc<RecursionProgram<SP1Field>>>,
    compose_keys: BTreeMap<usize, CompressKeys<C>>,
    vk_verification: bool,
}

impl<C: SP1ProverComponents> RecursionProverData<C> {
    pub fn make_merkle_proofs(
        &self,
        input: SP1ShapedWitnessValues<SP1GlobalContext, CoreSC>,
    ) -> Result<SP1CompressWithVKeyWitnessValues<CoreSC>, TaskError> {
        let num_vks = self.allowed_vk_map.len();
        let (vk_indices, vk_digest_values): (Vec<_>, Vec<_>) = if self.vk_verification {
            input
                .vks_and_proofs
                .iter()
                .map(|(vk, _)| {
                    let vk_digest = vk.hash_koalabear();
                    let index = self.allowed_vk_map.get(&vk_digest).expect("vk not allowed");
                    (index, vk_digest)
                })
                .unzip()
        } else {
            input
                .vks_and_proofs
                .iter()
                .map(|(vk, _)| {
                    let vk_digest = vk.hash_koalabear();
                    let index = (vk_digest[0].as_canonical_u32() as usize) % num_vks;
                    (index, [SP1Field::from_canonical_usize(index); 8])
                })
                .unzip()
        };

        let proofs = vk_indices
            .iter()
            .map(|index| {
                let (value, proof) = MerkleTree::open(&self.recursion_vk_tree, *index);

                MerkleTree::verify(proof.clone(), value, self.vk_root).expect("invalid proof");
                proof
            })
            .collect();

        let merkle_val = SP1MerkleProofWitnessValues {
            root: self.vk_root,
            values: vk_digest_values,
            vk_merkle_proofs: proofs,
        };

        Ok(SP1CompressWithVKeyWitnessValues { compress_val: input, merkle_val })
    }

    pub fn witness_stream(
        &self,
        witness: &SP1CircuitWitness,
    ) -> Result<VecDeque<Block<SP1Field>>, TaskError> {
        let mut witness_stream = Vec::new();
        match witness {
            SP1CircuitWitness::Core(input) => {
                Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
            }
            SP1CircuitWitness::Deferred(input) => {
                Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
            }
            SP1CircuitWitness::Compress(input) => {
                let input_with_merkle = self.make_merkle_proofs(input.clone())?;
                Witnessable::<InnerConfig>::write(&input_with_merkle, &mut witness_stream);
            }
            SP1CircuitWitness::Shrink(input) => {
                Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
            }
            SP1CircuitWitness::Wrap(input) => {
                Witnessable::<WrapConfig>::write(&input, &mut witness_stream);
            }
        }
        Ok(witness_stream.into())
    }
}

fn dummy_compose_input<C: SP1ProverComponents>(
    shape: &SP1RecursionProofShape,
    arity: usize,
    height: usize,
) -> SP1CompressWithVKeyWitnessValues<InnerSC> {
    let verifier = C::compress_verifier();
    shape.dummy_input(
        arity,
        height,
        verifier.shard_verifier().machine().chips().iter().cloned().collect::<BTreeSet<_>>(),
        verifier.max_log_row_count(),
        *verifier.fri_config(),
        verifier.log_stacking_height() as usize,
    )
}

pub struct ShrinkProver<C: SP1ProverComponents> {
    prover: Arc<RecursionProver<C>>,
    permits: ProverSemaphore,
    program: Arc<RecursionProgram<SP1Field>>,
    pub proving_key: Arc<MachineProvingKey<SP1GlobalContext, C::RecursionComponents>>,
    pub verifying_key: MachineVerifyingKey<SP1GlobalContext, RecursionConfig<C>>,
    prover_data: Arc<RecursionProverData<C>>,
}

impl<C: SP1ProverComponents> ShrinkProver<C> {
    fn new(
        prover: Arc<RecursionProver<C>>,
        permits: ProverSemaphore,
        prover_data: Arc<RecursionProverData<C>>,
        config: SP1RecursionProverConfig,
    ) -> Self {
        let verifier = C::compress_verifier();
        let input = prover_data.reduce_shape.dummy_input(
            1,
            prover_data.recursion_vk_tree.height,
            verifier.shard_verifier().machine().chips().iter().cloned().collect::<BTreeSet<_>>(),
            verifier.max_log_row_count(),
            *verifier.fri_config(),
            verifier.log_stacking_height() as usize,
        );
        let program = Arc::new(shrink_program_from_input(
            &recursive_verifier(verifier.shard_verifier()),
            config.vk_verification,
            &input,
        ));
        let (pk, vk) = {
            let (prover, program, permits) = (prover.clone(), program.clone(), permits.clone());
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn(async move {
                tx.send(prover.setup(program.clone(), permits.clone()).await).ok()
            });
            rx.blocking_recv().unwrap()
        };
        Self {
            prover,
            permits,
            program,
            proving_key: unsafe { pk.into_inner() },
            verifying_key: vk,
            prover_data,
        }
    }

    async fn prove(
        &self,
        compressed_proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
    ) -> Result<SP1RecursionProof<SP1GlobalContext, InnerSC>, TaskError> {
        let execution_record = {
            let mut runtime =
                Executor::<SP1Field, SP1ExtensionField, _>::new(self.program.clone(), inner_perm());
            runtime.witness_stream = self.prover_data.witness_stream(&{
                let SP1RecursionProof { vk, proof } = compressed_proof;
                let input =
                    SP1ShapedWitnessValues { vks_and_proofs: vec![(vk, proof)], is_complete: true };
                SP1CircuitWitness::Shrink(self.prover_data.make_merkle_proofs(input)?)
            })?;
            runtime.run().map_err(|e| TaskError::Fatal(e.into()))?;
            runtime.record
        };

        let mut challenger = SP1GlobalContext::default_challenger();
        self.verifying_key.observe_into(&mut challenger);

        let (proof, _permit) = self
            .prover
            .prove_shard_with_pk(
                self.proving_key.clone(),
                execution_record,
                self.permits.clone(),
                &mut challenger,
            )
            .await;
        Ok(SP1RecursionProof { vk: self.verifying_key.clone(), proof })
    }

    fn verify(
        &self,
        compressed_proof: &SP1RecursionProof<SP1GlobalContext, InnerSC>,
    ) -> Result<(), TaskError> {
        let SP1RecursionProof { vk, proof } = compressed_proof;
        let mut challenger = SP1GlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        C::shrink_verifier()
            .verify_shard(vk, proof, &mut challenger)
            .map_err(|e| TaskError::Fatal(e.into()))?;
        Ok(())
    }
}

pub struct WrapProver<C: SP1ProverComponents> {
    prover: Arc<InnerWrapProver<C>>,
    permits: ProverSemaphore,
    program: Arc<RecursionProgram<SP1Field>>,
    pub proving_key: Arc<MachineProvingKey<SP1OuterGlobalContext, C::WrapComponents>>,
    pub verifying_key: MachineVerifyingKey<SP1OuterGlobalContext, crate::recursion::WrapConfig<C>>,
    prover_data: Arc<RecursionProverData<C>>,
}

impl<C: SP1ProverComponents> WrapProver<C> {
    fn new(
        prover: Arc<InnerWrapProver<C>>,
        permits: ProverSemaphore,
        prover_data: Arc<RecursionProverData<C>>,
        config: SP1RecursionProverConfig,
        shrink_proving_key: Arc<MachineProvingKey<SP1GlobalContext, C::RecursionComponents>>,
    ) -> Self {
        let verifier = C::shrink_verifier();
        let heights = {
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn(async move {
                let heights = <C::RecursionComponents as MachineProverComponents<
                    SP1GlobalContext,
                >>::preprocessed_table_heights(shrink_proving_key)
                .await;
                tx.send(heights).ok();
            });
            rx.blocking_recv().unwrap()
        };
        let shrink_proof_shape = SP1RecursionProofShape { shape: RecursionShape::new(heights) };
        let wrap_input = shrink_proof_shape.dummy_input(
            1,
            prover_data.recursion_vk_tree.height,
            verifier.shard_verifier().machine().chips().iter().cloned().collect::<BTreeSet<_>>(),
            verifier.max_log_row_count(),
            *verifier.fri_config(),
            verifier.log_stacking_height() as usize,
        );

        let program = Arc::new(wrap_program_from_input(
            &recursive_verifier(verifier.shard_verifier()),
            config.vk_verification,
            &wrap_input,
        ));
        let (proving_key, verifying_key) = {
            let (prover, program, permits) = (prover.clone(), program.clone(), permits.clone());
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn(async move {
                tx.send(prover.setup(program.clone(), permits).await).ok();
            });
            rx.blocking_recv().unwrap()
        };

        Self {
            prover,
            permits,
            program,
            proving_key: unsafe { proving_key.into_inner() },
            verifying_key,
            prover_data,
        }
    }

    pub async fn prove(
        &self,
        shrunk_proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
    ) -> Result<SP1RecursionProof<SP1OuterGlobalContext, OuterSC>, TaskError> {
        let execution_record = {
            let mut runtime =
                Executor::<SP1Field, SP1ExtensionField, _>::new(self.program.clone(), inner_perm());
            runtime.witness_stream = self.prover_data.witness_stream(&{
                let SP1RecursionProof { vk, proof } = shrunk_proof;
                let input =
                    SP1ShapedWitnessValues { vks_and_proofs: vec![(vk, proof)], is_complete: true };
                SP1CircuitWitness::Wrap(self.prover_data.make_merkle_proofs(input)?)
            })?;
            runtime.run().map_err(|e| TaskError::Fatal(e.into()))?;
            runtime.record
        };

        let mut challenger = SP1OuterGlobalContext::default_challenger();
        self.verifying_key.observe_into(&mut challenger);

        let (proof, _permit) = self
            .prover
            .prove_shard_with_pk(
                self.proving_key.clone(),
                execution_record,
                self.permits.clone(),
                &mut challenger,
            )
            .await;
        Ok(SP1RecursionProof { vk: self.verifying_key.clone(), proof })
    }

    fn verify(
        &self,
        wrapped_proof: &SP1RecursionProof<SP1OuterGlobalContext, OuterSC>,
    ) -> Result<(), TaskError> {
        let SP1RecursionProof { vk, proof } = wrapped_proof;
        let mut challenger = SP1OuterGlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        C::wrap_verifier()
            .verify_shard(vk, proof, &mut challenger)
            .map_err(|e| TaskError::Fatal(e.into()))?;
        Ok(())
    }
}

use slop_algebra::AbstractField;
use slop_challenger::IopCtx;
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
};

use crate::{
    components::RecursionProver,
    recursion::{compose_program_from_input, recursive_verifier, RecursionConfig},
    shapes::SP1RecursionProofShape,
    worker::{
        CommonProverInput, RangeProofs, RawTaskRequest, TaskContext, TaskError, TaskMetadata,
    },
    CompressAir, CoreSC, HashableKey, InnerSC, SP1CircuitWitness, SP1ProverComponents,
};
use slop_algebra::PrimeField32;
use slop_futures::pipeline::{
    AsyncEngine, AsyncWorker, BlockingEngine, BlockingWorker, Chain, Pipeline, SubmitError,
    SubmitHandle,
};
use sp1_hypercube::{
    air::SP1CorePublicValues,
    inner_perm,
    prover::{AirProver, MachineProvingKey, ProverSemaphore},
    MachineProof, MachineVerifier, MachineVerifyingKey, SP1RecursionProof, ShardProof, DIGEST_SIZE,
};
use sp1_primitives::{SP1ExtensionField, SP1Field, SP1GlobalContext};
use sp1_prover_types::{Artifact, ArtifactClient, ArtifactId};
use sp1_recursion_circuit::{
    basefold::merkle_tree::MerkleTree,
    machine::{
        SP1CompressWithVKeyWitnessValues, SP1MerkleProofWitnessValues, SP1NormalizeWitnessValues,
        SP1ShapedWitnessValues,
    },
    witness::Witnessable,
    WrapConfig,
};
use sp1_recursion_compiler::config::InnerConfig;
use sp1_recursion_executor::{Block, ExecutionRecord, Executor, RecursionProgram};
use tokio::sync::oneshot;

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

        Ok(RecursionTask { program, witness, is_complete, output })
    }
}

pub struct RecursionTask {
    program: Arc<RecursionProgram<SP1Field>>,
    witness: SP1CircuitWitness,
    is_complete: bool,
    output: Artifact,
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
        let RecursionTask { program, witness, is_complete, output } = input?;

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

        Ok(ProveRecursionTask { record, keys, output, is_complete })
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
    is_complete: bool,
}

pub struct RecursionProverWorker<A, C: SP1ProverComponents> {
    recursion_prover: Arc<RecursionProver<C>>,
    permits: ProverSemaphore,
    artifact_client: A,
}

impl<A: ArtifactClient, C: SP1ProverComponents> RecursionProverWorker<A, C> {
    async fn prove_shard(
        &self,
        keys: RecursionKeys<C>,
        record: ExecutionRecord<SP1Field>,
    ) -> Result<(SP1RecursionProof<SP1GlobalContext, InnerSC>, TaskMetadata), TaskError> {
        let proof = match keys {
            RecursionKeys::Exists(pk, vk) => {
                let mut challenger = SP1GlobalContext::default_challenger();
                vk.observe_into(&mut challenger);
                let (proof, _) = self
                    .recursion_prover
                    .prove_shard_with_pk(pk.clone(), record, self.permits.clone(), &mut challenger)
                    .await;
                C::compress_verifier()
                    .verify(&vk, &MachineProof::from(vec![proof.clone()]))
                    .map_err(|e| {
                        TaskError::Retryable(anyhow::anyhow!("compress verify failed: {}", e))
                    })?;
                SP1RecursionProof { vk, proof }
            }
            RecursionKeys::Program(program) => {
                let mut challenger = SP1GlobalContext::default_challenger();
                let (vk, proof, _) = self
                    .recursion_prover
                    .setup_and_prove_shard(
                        program,
                        record,
                        None,
                        self.permits.clone(),
                        &mut challenger,
                    )
                    .await;
                C::compress_verifier()
                    .verify(&vk, &MachineProof::from(vec![proof.clone()]))
                    .map_err(|e| {
                        TaskError::Retryable(anyhow::anyhow!("compress verify failed: {}", e))
                    })?;
                SP1RecursionProof { vk, proof }
            }
        };

        // TODO: Add the busy time here.
        Ok((proof, TaskMetadata::default()))
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
        let ProveRecursionTask { record, keys, output, is_complete } = input?;
        // Prove the shard
        let (proof, metadata) = self.prove_shard(keys, record).await?;
        // Upload the proof
        if is_complete {
            self.artifact_client.upload_proof(&output, proof.clone()).await?;
        } else {
            self.artifact_client.upload(&output, proof.clone()).await?;
        }

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
    prover_data: Arc<RecursionProverData<C>>,
}

impl<A, C: SP1ProverComponents> Clone for SP1RecursionProver<A, C> {
    fn clone(&self) -> Self {
        Self {
            reduce_pipeline: self.reduce_pipeline.clone(),
            prover_data: self.prover_data.clone(),
        }
    }
}

impl<A: ArtifactClient, C: SP1ProverComponents> SP1RecursionProver<A, C> {
    pub async fn new(
        config: SP1RecursionProverConfig,
        artifact_client: A,
        air_prover: Arc<RecursionProver<C>>,
        permits: ProverSemaphore,
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

            let prover_data = RecursionProverData {
                vk_root,
                allowed_vk_map,
                recursion_vk_tree,
                reduce_shape,
                compose_programs,
                compose_keys,
                vk_verification: config.vk_verification,
            };
            let prover_data = Arc::new(prover_data);

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
                    permits: permits.clone(),
                    artifact_client: artifact_client.clone(),
                })
                .collect();
            let prove_engine =
                Arc::new(AsyncEngine::new(prove_workers, config.recursion_prover_buffer_size));

            // Make the recursion pipeline.
            let recursion_pipeline = Arc::new(Chain::new(executor_engine, prove_engine));

            // Make the reduce pipeline.
            let reduce_pipeline = Arc::new(Chain::new(prepare_reduce_engine, recursion_pipeline));

            Self { reduce_pipeline, prover_data }
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
        is_complete: bool,
    ) -> Result<RecursionProveSubmitHandle<A, C>, SubmitError> {
        self.recursion_prover_pipeline()
            .submit(Ok(RecursionTask { program, witness, is_complete, output }))
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

struct RecursionProverData<C: SP1ProverComponents> {
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
    let chips =
        verifier.shard_verifier().machine().chips().iter().cloned().collect::<BTreeSet<_>>();

    let max_log_row_count = verifier.max_log_row_count();
    let log_stacking_height = verifier.log_stacking_height() as usize;

    shape.dummy_input(
        arity,
        height,
        chips,
        max_log_row_count,
        *verifier.fri_config(),
        log_stacking_height,
    )
}

use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Arc,
};

use slop_algebra::AbstractField;
use slop_baby_bear::BabyBear;
use slop_futures::handle::TaskHandle;
use slop_jagged::JaggedConfig;
use sp1_core_executor::SP1ReduceProof;
use sp1_core_machine::riscv::RiscvAir;
use sp1_primitives::{consts::WORD_SIZE, hash_deferred_proof};
use sp1_recursion_circuit::{
    basefold::{
        stacked::RecursiveStackedPcsVerifier, tcs::RecursiveMerkleTreeTcs,
        RecursiveBasefoldConfigImpl, RecursiveBasefoldVerifier,
    },
    jagged::{
        RecursiveJaggedConfigImpl, RecursiveJaggedEvalSumcheckConfig, RecursiveJaggedPcsVerifier,
    },
    machine::{
        InnerVal, PublicValuesOutputDigest, SP1CompressVerifier, SP1CompressWitnessValues,
        SP1DeferredVerifier, SP1DeferredWitnessValues, SP1RecursionWitnessValues,
        SP1RecursiveVerifier, JC,
    },
    shard::RecursiveShardVerifier,
    witness::Witnessable,
};
use sp1_recursion_compiler::{
    circuit::AsmCompiler,
    config::InnerConfig,
    ir::{Builder, DslIrProgram},
};
use sp1_recursion_executor::{
    ExecutionRecord, RecursionProgram, RecursionPublicValues, DIGEST_SIZE,
};
use sp1_stark::{
    air::{POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    prover::{MachineProver, MachineProverComponents, MachineProverError, MachineProvingKey},
    BabyBearPoseidon2, Machine, MachineVerifier, MachineVerifyingKey, ShardProof, ShardVerifier,
    Word,
};

use crate::{
    shapes::{SP1RecursionCache, SP1RecursionShape, SP1ReduceShape},
    utils::words_to_bytes,
    CompressAir, CoreSC, InnerSC,
};

#[allow(clippy::type_complexity)]
pub struct SP1RecursionProver<C: RecursionProverComponents> {
    prover: MachineProver<C>,
    core_verifier: MachineVerifier<CoreSC, RiscvAir<BabyBear>>,
    recursion_program_cache: SP1RecursionCache,
    reduce_shape: SP1ReduceShape,
    reduce_program: Option<Arc<RecursionProgram<BabyBear>>>,
    deferred_program: Option<Arc<RecursionProgram<BabyBear>>>,
    deferred_keys: Option<(Arc<MachineProvingKey<C>>, MachineVerifyingKey<C::Config>)>,
    reduce_keys: Option<(Arc<MachineProvingKey<C>>, MachineVerifyingKey<C::Config>)>,
    recursive_core_verifier:
        RecursiveShardVerifier<RiscvAir<BabyBear>, CoreSC, InnerConfig, JC<InnerConfig, CoreSC>>,
    recursive_compress_verifier: RecursiveShardVerifier<
        CompressAir<InnerVal>,
        InnerSC,
        InnerConfig,
        JC<InnerConfig, InnerSC>,
    >,
}

const RECURSION_LOG_BLOWUP: usize = 1;
const RECURSION_LOG_STACKING_HEIGHT: u32 = 20;
const RECURSION_MAX_LOG_ROW_COUNT: usize = 20;

pub trait RecursionProverComponents:
    MachineProverComponents<
    F = <InnerSC as JaggedConfig>::F,
    Config = InnerSC,
    Record = ExecutionRecord<<InnerSC as JaggedConfig>::F>,
    Program = RecursionProgram<<InnerSC as JaggedConfig>::F>,
    Challenger = <InnerSC as JaggedConfig>::Challenger,
    Air = CompressAir<<InnerSC as JaggedConfig>::F>,
>
{
    fn verifier() -> MachineVerifier<InnerSC, CompressAir<InnerVal>> {
        let compress_log_blowup = RECURSION_LOG_BLOWUP;
        let compress_log_stacking_height = RECURSION_LOG_STACKING_HEIGHT;
        let compress_max_log_row_count = RECURSION_MAX_LOG_ROW_COUNT;

        let machine = CompressAir::<BabyBear>::machine_wide_with_all_chips();
        let recursion_shard_verifier = ShardVerifier::from_basefold_parameters(
            compress_log_blowup,
            compress_log_stacking_height,
            compress_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(recursion_shard_verifier)
    }
}

impl<C> RecursionProverComponents for C where
    C: MachineProverComponents<
        F = <InnerSC as JaggedConfig>::F,
        Config = InnerSC,
        Record = ExecutionRecord<<InnerSC as JaggedConfig>::F>,
        Program = RecursionProgram<<InnerSC as JaggedConfig>::F>,
        Challenger = <InnerSC as JaggedConfig>::Challenger,
        Air = CompressAir<<InnerSC as JaggedConfig>::F>,
    >
{
}

impl<C: RecursionProverComponents> SP1RecursionProver<C> {
    pub async fn new(
        core_verifier: ShardVerifier<CoreSC, RiscvAir<BabyBear>>,
        prover: MachineProver<C>,
        recursion_programs_cache_size: usize,
        recursion_programs: BTreeMap<SP1RecursionShape, Arc<RecursionProgram<BabyBear>>>,
    ) -> Self {
        let recursive_core_verifier = {
            let recursive_verifier = RecursiveBasefoldVerifier {
                fri_config: core_verifier.pcs_verifier.stacked_pcs_verifier.pcs_verifier.fri_config,
                tcs: RecursiveMerkleTreeTcs::<InnerConfig, CoreSC>(PhantomData),
            };
            let core_log_stacking_height = core_verifier.log_stacking_height();
            let recursive_verifier =
                RecursiveStackedPcsVerifier::new(recursive_verifier, core_log_stacking_height);

            let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
                CoreSC,
                InnerConfig,
                RecursiveJaggedConfigImpl<
                    InnerConfig,
                    CoreSC,
                    RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<InnerConfig, CoreSC>>,
                >,
            > {
                stacked_pcs_verifier: recursive_verifier,
                max_log_row_count: core_verifier.max_log_row_count(),
                jagged_evaluator: RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(
                    PhantomData,
                ),
            };

            RecursiveShardVerifier {
                machine: core_verifier.machine().clone(),
                pcs_verifier: recursive_jagged_verifier,
                _phantom: std::marker::PhantomData,
            }
        };

        let recursive_verifier = prover.verifier().shard_verifier();

        let recursive_compress_verifier = {
            let compress_log_stacking_height = recursive_verifier.log_stacking_height();
            let compress_max_log_row_count = recursive_verifier.max_log_row_count();
            let compress_machine = recursive_verifier.machine().clone();
            let recursive_verifier = RecursiveBasefoldVerifier {
                fri_config: recursive_verifier
                    .pcs_verifier
                    .stacked_pcs_verifier
                    .pcs_verifier
                    .fri_config,
                tcs: RecursiveMerkleTreeTcs::<InnerConfig, InnerSC>(PhantomData),
            };
            let recursive_verifier =
                RecursiveStackedPcsVerifier::new(recursive_verifier, compress_log_stacking_height);

            let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
                InnerSC,
                InnerConfig,
                RecursiveJaggedConfigImpl<
                    InnerConfig,
                    InnerSC,
                    RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<InnerConfig, InnerSC>>,
                >,
            > {
                stacked_pcs_verifier: recursive_verifier,
                max_log_row_count: compress_max_log_row_count,
                jagged_evaluator: RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(
                    PhantomData,
                ),
            };

            RecursiveShardVerifier {
                machine: compress_machine,
                pcs_verifier: recursive_jagged_verifier,
                _phantom: std::marker::PhantomData,
            }
        };

        // Instantiate the cache.
        let recursion_program_cache = SP1RecursionCache::new(recursion_programs_cache_size);
        for (shape, program) in recursion_programs {
            recursion_program_cache.push(shape, program);
        }

        let reduce_shape = SP1ReduceShape::default();

        // Make the reduce program and proving key.
        let dummy_input = dummy_reduce_input(&prover, &reduce_shape, 2);
        let mut program =
            compress_program_from_input(&recursive_compress_verifier, &dummy_input, true);
        program.shape = Some(reduce_shape.shape.clone());
        let program = Arc::new(program);

        // Make the reduce keys.
        let (pk, vk) = prover.setup(program.clone(), None).await.unwrap();
        let pk = unsafe { pk.into_inner() };
        let reduce_keys = Some((pk, vk));
        let reduce_program = Some(program);

        //  Make the deferred program and proving key.
        let program = dummy_deferred_input(&prover, &reduce_shape);
        let mut program = deferred_program_from_input(&recursive_compress_verifier, &program, true);
        program.shape = Some(reduce_shape.shape.clone());
        let program = Arc::new(program);

        // Make the deferred keys.
        let (pk, vk) = prover.setup(program.clone(), None).await.unwrap();
        let pk = unsafe { pk.into_inner() };
        let deferred_keys = Some((pk, vk));
        let deferred_program = Some(program);

        Self {
            prover,
            core_verifier: MachineVerifier::new(core_verifier),
            recursive_core_verifier,
            recursive_compress_verifier,
            recursion_program_cache,
            reduce_shape,
            reduce_keys,
            reduce_program,
            deferred_program,
            deferred_keys,
        }
    }

    pub fn verifier(&self) -> &MachineVerifier<C::Config, CompressAir<C::F>> {
        self.prover.verifier()
    }

    pub fn machine(&self) -> &Machine<C::F, CompressAir<C::F>> {
        self.prover.machine()
    }

    #[inline]
    #[must_use]
    pub fn prove_shard(
        &self,
        pk: Arc<MachineProvingKey<C>>,
        record: ExecutionRecord<BabyBear>,
    ) -> TaskHandle<ShardProof<InnerSC>, MachineProverError> {
        self.prover.prove_shard(pk, record)
    }

    #[inline]
    #[must_use]
    pub fn setup_and_prove_shard(
        &self,
        program: Arc<RecursionProgram<BabyBear>>,
        vk: Option<MachineVerifyingKey<InnerSC>>,
        record: ExecutionRecord<BabyBear>,
    ) -> TaskHandle<(MachineVerifyingKey<InnerSC>, ShardProof<InnerSC>), MachineProverError> {
        self.prover.setup_and_prove_shard(program, vk, record)
    }

    pub fn recursion_program(
        &self,
        input: &SP1RecursionWitnessValues<CoreSC>,
        compute_event_counts: bool,
    ) -> Arc<RecursionProgram<BabyBear>> {
        let proof_shapes = input
            .shard_proofs
            .iter()
            .map(|proof| self.core_verifier.shape_from_proof(proof))
            .collect::<Vec<_>>();
        let shape = SP1RecursionShape {
            proof_shapes,
            max_log_row_count: self.core_verifier.max_log_row_count(),
            log_blowup: self.core_verifier.fri_config().log_blowup,
            log_stacking_height: self.core_verifier.log_stacking_height() as usize,
        };
        if let Some(program) = self.recursion_program_cache.get(&shape) {
            return program.clone();
        }
        let mut program = recursion_program_from_input(
            &self.recursive_core_verifier,
            input,
            compute_event_counts,
        );
        program.shape = Some(self.reduce_shape.shape.clone());
        let program = Arc::new(program);
        self.recursion_program_cache.push(shape, program.clone());
        program
    }

    pub fn compress_program(
        &self,
        input: &SP1CompressWitnessValues<InnerSC>,
        compute_event_counts: bool,
    ) -> Arc<RecursionProgram<BabyBear>> {
        if let Some(program) = self.reduce_program.as_ref() {
            return program.clone();
        }
        let mut program = compress_program_from_input(
            &self.recursive_compress_verifier,
            input,
            compute_event_counts,
        );
        program.shape = Some(self.reduce_shape.shape.clone());
        Arc::new(program)
    }

    pub fn recursion_program_cache_stats(&self) -> (usize, usize, f64) {
        self.recursion_program_cache.stats()
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn deferred_keys(
        &self,
    ) -> Option<(Arc<MachineProvingKey<C>>, MachineVerifyingKey<C::Config>)> {
        self.deferred_keys.clone()
    }

    pub fn deferred_program(
        &self,
        input: &SP1DeferredWitnessValues<InnerSC>,
        compute_event_counts: bool,
    ) -> Arc<RecursionProgram<BabyBear>> {
        if let Some(program) = self.deferred_program.as_ref() {
            return program.clone();
        }
        let mut program = deferred_program_from_input(
            &self.recursive_compress_verifier,
            input,
            compute_event_counts,
        );
        program.shape = Some(self.reduce_shape.shape.clone());
        Arc::new(program)
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn reduce_keys(
        &self,
        arity: usize,
    ) -> Option<(Arc<MachineProvingKey<C>>, MachineVerifyingKey<C::Config>)> {
        debug_assert_eq!(arity, 2);
        self.reduce_keys.clone()
    }

    pub fn hash_deferred_proofs(
        prev_digest: [<CoreSC as JaggedConfig>::F; DIGEST_SIZE],
        deferred_proofs: &[SP1ReduceProof<InnerSC>],
    ) -> [<CoreSC as JaggedConfig>::F; 8] {
        let mut digest = prev_digest;
        for proof in deferred_proofs.iter() {
            let pv: &RecursionPublicValues<<CoreSC as JaggedConfig>::F> =
                proof.proof.public_values.as_slice().borrow();
            let committed_values_digest = words_to_bytes(&pv.committed_value_digest);
            digest = hash_deferred_proof(
                &digest,
                &pv.sp1_vk_digest,
                &committed_values_digest.try_into().unwrap(),
            );
        }
        digest
    }
}

fn recursion_program_from_input(
    recursion_verifier: &RecursiveShardVerifier<
        RiscvAir<BabyBear>,
        CoreSC,
        InnerConfig,
        JC<InnerConfig, CoreSC>,
    >,
    input: &SP1RecursionWitnessValues<CoreSC>,
    compute_event_counts: bool,
) -> RecursionProgram<BabyBear> {
    // Get the operations.
    let builder_span = tracing::debug_span!("build recursion program").entered();
    let mut builder = Builder::<InnerConfig>::default();
    let input_variable = input.read(&mut builder);
    SP1RecursiveVerifier::verify(&mut builder, recursion_verifier, input_variable);
    let block = builder.into_root_block();
    // SAFETY: The circuit is well-formed. It does not use synchronization primitives
    // (or possibly other means) to violate the invariants.
    let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };
    builder_span.exit();

    // Compile the program.
    let compiler_span = tracing::debug_span!("compile recursion program").entered();
    let mut compiler = AsmCompiler::<InnerConfig>::default();
    let program = compiler.compile(dsl_program, compute_event_counts);
    // if let Some(inn_recursion_shape_config) = &self.compress_shape_config {
    //     inn_recursion_shape_config.fix_shape(&mut program);
    // }
    compiler_span.exit();
    program
}

fn deferred_program_from_input(
    recursion_verifier: &RecursiveShardVerifier<
        CompressAir<InnerVal>,
        InnerSC,
        InnerConfig,
        JC<InnerConfig, InnerSC>,
    >,
    // vk_verification: bool,
    //TODO: Add VK verification back in
    input: &SP1DeferredWitnessValues<InnerSC>,
    compute_event_counts: bool,
) -> RecursionProgram<BabyBear> {
    // Get the operations.
    let operations_span = tracing::debug_span!("get operations for the deferred program").entered();
    let mut builder = Builder::<InnerConfig>::default();
    let input_read_span = tracing::debug_span!("Read input values").entered();
    let input = input.read(&mut builder);
    input_read_span.exit();
    let verify_span = tracing::debug_span!("Verify deferred program").entered();

    // Verify the proof.
    SP1DeferredVerifier::verify(&mut builder, recursion_verifier, input);
    verify_span.exit();
    let block = builder.into_root_block();
    operations_span.exit();
    // SAFETY: The circuit is well-formed. It does not use synchronization primitives
    // (or possibly other means) to violate the invariants.
    let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

    let compiler_span = tracing::debug_span!("compile deferred program").entered();
    let mut compiler = AsmCompiler::<InnerConfig>::default();
    let program = compiler.compile(dsl_program, compute_event_counts);
    compiler_span.exit();
    program
}

fn compress_program_from_input(
    // config: Option<&RecursionShapeConfig<BabyBear, CompressAir<BabyBear>>>,
    recursion_verifier: &RecursiveShardVerifier<
        CompressAir<InnerVal>,
        InnerSC,
        InnerConfig,
        JC<InnerConfig, InnerSC>,
    >,
    // vk_verification: bool,
    //TODO: Add VK verification back in
    input: &SP1CompressWitnessValues<InnerSC>,
    compute_event_counts: bool,
) -> RecursionProgram<BabyBear> {
    let builder_span = tracing::debug_span!("build compress program").entered();
    let mut builder = Builder::<InnerConfig>::default();
    // read the input.
    let input = input.read(&mut builder);

    // Verify the proof.
    SP1CompressVerifier::verify(
        &mut builder,
        recursion_verifier,
        input,
        // vk_verification,
        PublicValuesOutputDigest::Reduce,
    );
    let block = builder.into_root_block();
    builder_span.exit();
    // SAFETY: The circuit is well-formed. It does not use synchronization primitives
    // (or possibly other means) to violate the invariants.
    let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

    // Compile the program.
    let compiler_span = tracing::debug_span!("compile compress program").entered();
    let mut compiler = AsmCompiler::<InnerConfig>::default();
    let program = compiler.compile(dsl_program, compute_event_counts);
    compiler_span.exit();
    program
}

// fn dummy_reduce_program(arity: yusi)

fn dummy_reduce_input<C: RecursionProverComponents>(
    prover: &MachineProver<C>,
    shape: &SP1ReduceShape,
    arity: usize,
) -> SP1CompressWitnessValues<InnerSC> {
    let chips = prover
        .verifier()
        .shard_verifier()
        .machine()
        .chips()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    let max_log_row_count = prover.verifier().max_log_row_count();
    let log_blowup = prover.verifier().fri_config().log_blowup();
    let log_stacking_height = prover.verifier().log_stacking_height() as usize;

    shape.dummy_input(arity, chips, max_log_row_count, log_blowup, log_stacking_height)
}

fn dummy_deferred_input<C: RecursionProverComponents>(
    prover: &MachineProver<C>,
    shape: &SP1ReduceShape,
) -> SP1DeferredWitnessValues<InnerSC> {
    let chips = prover
        .verifier()
        .shard_verifier()
        .machine()
        .chips()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    let max_log_row_count = prover.verifier().max_log_row_count();
    let log_blowup = prover.verifier().fri_config().log_blowup();
    let log_stacking_height = prover.verifier().log_stacking_height() as usize;

    let compress_input =
        shape.dummy_input(1, chips, max_log_row_count, log_blowup, log_stacking_height);

    let SP1CompressWitnessValues { vks_and_proofs, .. } = compress_input;

    SP1DeferredWitnessValues {
        vks_and_proofs,
        // pub vk_merkle_data: SP1MerkleProofWitnessValues<SC>,
        start_reconstruct_deferred_digest: [BabyBear::zero(); POSEIDON_NUM_WORDS],
        sp1_vk_digest: [BabyBear::zero(); DIGEST_SIZE],
        committed_value_digest: [[BabyBear::zero(); 4]; PV_DIGEST_NUM_WORDS],
        deferred_proofs_digest: [BabyBear::zero(); POSEIDON_NUM_WORDS],
        end_pc: BabyBear::zero(),
        end_shard: BabyBear::zero(),
        end_execution_shard: BabyBear::zero(),
        init_addr_word: Word([BabyBear::zero(); WORD_SIZE]),
        finalize_addr_word: Word([BabyBear::zero(); WORD_SIZE]),
        is_complete: false,
    }
}

use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Arc,
};

use slop_algebra::{extension::BinomialExtensionField, AbstractField};
use slop_baby_bear::BabyBear;
use slop_futures::handle::TaskHandle;
use slop_jagged::JaggedConfig;
use slop_merkle_tree::my_bb_16_perm;
use sp1_core_executor::SP1ReduceProof;
use sp1_core_machine::riscv::RiscvAir;
use sp1_primitives::{consts::WORD_SIZE, hash_deferred_proof};
use sp1_recursion_circuit::{
    basefold::{
        stacked::RecursiveStackedPcsVerifier, tcs::RecursiveMerkleTreeTcs,
        RecursiveBasefoldConfigImpl, RecursiveBasefoldVerifier,
    },
    jagged::{
        RecursiveJaggedConfig, RecursiveJaggedEvalSumcheckConfig, RecursiveJaggedPcsVerifier,
    },
    machine::{
        InnerVal, PublicValuesOutputDigest, SP1CompressRootVerifier, SP1CompressVerifier,
        SP1CompressWitnessValues, SP1DeferredVerifier, SP1DeferredWitnessValues,
        SP1RecursionWitnessValues, SP1RecursiveVerifier, JC,
    },
    shard::RecursiveShardVerifier,
    witness::Witnessable,
    BabyBearFriConfigVariable, CircuitConfig, WrapConfig as CircuitWrapConfig,
};
use sp1_recursion_compiler::{
    circuit::AsmCompiler,
    config::InnerConfig,
    ir::{Builder, DslIrProgram},
};
use sp1_recursion_executor::{
    ExecutionRecord, RecursionProgram, RecursionPublicValues, Runtime, DIGEST_SIZE,
};
use sp1_stark::{
    air::{MachineAir, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    prover::{MachineProver, MachineProverComponents, MachineProverError, MachineProvingKey},
    Machine, MachineVerifier, MachineVerifyingKey, ShardProof, ShardVerifier, Word,
};

use crate::{
    components::SP1ProverComponents,
    shapes::{SP1RecursionCache, SP1RecursionShape, SP1ReduceShape},
    utils::words_to_bytes,
    CompressAir, CoreSC, InnerSC, OuterSC, SP1CircuitWitness, SP1RecursionProverError, WrapAir,
};

pub mod components;
pub use components::*;

type RecursionConfig<C> =
    <<C as SP1ProverComponents>::RecursionComponents as MachineProverComponents>::Config;

type RecursionF<C> =
    <<C as SP1ProverComponents>::RecursionComponents as MachineProverComponents>::F;

type WrapConfig<C> =
    <<C as SP1ProverComponents>::WrapComponents as MachineProverComponents>::Config;

#[allow(clippy::type_complexity)]
pub struct SP1RecursionProver<C: SP1ProverComponents> {
    prover: MachineProver<C::RecursionComponents>,
    shrink_prover: MachineProver<C::RecursionComponents>,
    wrap_prover: MachineProver<C::WrapComponents>,
    pub(crate) core_verifier: MachineVerifier<CoreSC, RiscvAir<BabyBear>>,
    pub(crate) recursion_program_cache: SP1RecursionCache,
    reduce_shape: SP1ReduceShape,
    reduce_programs: BTreeMap<usize, Arc<RecursionProgram<BabyBear>>>,
    reduce_keys: BTreeMap<
        usize,
        (Arc<MachineProvingKey<C::RecursionComponents>>, MachineVerifyingKey<RecursionConfig<C>>),
    >,
    deferred_program: Option<Arc<RecursionProgram<BabyBear>>>,
    deferred_keys: Option<(
        Arc<MachineProvingKey<C::RecursionComponents>>,
        MachineVerifyingKey<RecursionConfig<C>>,
    )>,
    shrink_program: Arc<RecursionProgram<BabyBear>>,
    shrink_keys:
        (Arc<MachineProvingKey<C::RecursionComponents>>, MachineVerifyingKey<RecursionConfig<C>>),
    wrap_program: Arc<RecursionProgram<BabyBear>>,
    wrap_keys: (Arc<MachineProvingKey<C::WrapComponents>>, MachineVerifyingKey<WrapConfig<C>>),
    recursive_core_verifier:
        RecursiveShardVerifier<RiscvAir<BabyBear>, CoreSC, InnerConfig, JC<InnerConfig, CoreSC>>,
    recursive_compress_verifier: RecursiveShardVerifier<
        CompressAir<InnerVal>,
        InnerSC,
        InnerConfig,
        JC<InnerConfig, InnerSC>,
    >,
    max_reduce_arity: usize,
    recursion_batch_size: usize,
}

impl<C: SP1ProverComponents> SP1RecursionProver<C> {
    pub async fn new(
        core_verifier: ShardVerifier<CoreSC, RiscvAir<BabyBear>>,
        prover: MachineProver<C::RecursionComponents>,
        shrink_prover: MachineProver<C::RecursionComponents>,
        wrap_prover: MachineProver<C::WrapComponents>,
        recursion_programs_cache_size: usize,
        recursion_programs: BTreeMap<SP1RecursionShape, Arc<RecursionProgram<BabyBear>>>,
        max_reduce_arity: usize,
    ) -> Self {
        let recursive_core_verifier =
            recursive_verifier::<_, CoreSC, InnerConfig, _>(&core_verifier);

        let recursive_compress_verifier =
            recursive_verifier::<_, InnerSC, InnerConfig, _>(prover.verifier().shard_verifier());

        let recursive_shrink_verifier = recursive_verifier::<_, InnerSC, InnerConfig, _>(
            shrink_prover.verifier().shard_verifier(),
        );

        // Instantiate the cache.
        let recursion_program_cache = SP1RecursionCache::new(recursion_programs_cache_size);
        for (shape, program) in recursion_programs {
            recursion_program_cache.push(shape, program);
        }

        // Get the reduce shape.
        let reduce_shape =
            SP1ReduceShape::reduce_shape_from_arity(max_reduce_arity).expect("arity not supported");

        // Make the reduce programs and keys.
        let mut reduce_programs = BTreeMap::new();
        let mut reduce_keys = BTreeMap::new();

        for arity in 1..=max_reduce_arity {
            let dummy_input = dummy_reduce_input(&prover, &reduce_shape, arity);
            let mut program =
                compress_program_from_input(&recursive_compress_verifier, &dummy_input);
            program.shape = Some(reduce_shape.shape.clone());
            let program = Arc::new(program);

            // Make the reduce keys.
            let (pk, vk) = prover.setup(program.clone(), None).await.unwrap();
            let pk = unsafe { pk.into_inner() };
            reduce_keys.insert(arity, (pk, vk));
            reduce_programs.insert(arity, program);
        }

        let shrink_input = dummy_reduce_input(&prover, &reduce_shape, 1);
        let shrink_program = shrink_program_from_input(&recursive_compress_verifier, &shrink_input);
        let shrink_program = Arc::new(shrink_program);

        let wrap_shape =
            SP1ReduceShape::shrink_shape_from_arity(max_reduce_arity).expect("arity not supported");

        let wrap_input = dummy_reduce_input(&shrink_prover, &wrap_shape, 1);
        let wrap_program = wrap_program_from_input(&recursive_shrink_verifier, &wrap_input);
        let wrap_program = Arc::new(wrap_program);

        //  Make the deferred program and proving key.
        let program = dummy_deferred_input(&prover, &reduce_shape);
        let mut program = deferred_program_from_input(&recursive_compress_verifier, &program);
        program.shape = Some(reduce_shape.shape.clone());
        let program = Arc::new(program);

        let (shrink_pk, shrink_vk) =
            shrink_prover.setup(shrink_program.clone(), None).await.unwrap();
        let shrink_keys = (unsafe { shrink_pk.into_inner() }, shrink_vk);

        let (wrap_pk, wrap_vk) = wrap_prover.setup(wrap_program.clone(), None).await.unwrap();
        let wrap_keys = (unsafe { wrap_pk.into_inner() }, wrap_vk);

        // Make the deferred keys.
        let (pk, vk) = prover.setup(program.clone(), None).await.unwrap();
        let pk = unsafe { pk.into_inner() };
        let deferred_keys = Some((pk, vk));
        let deferred_program = Some(program);

        Self {
            prover,
            shrink_prover,
            wrap_prover,
            core_verifier: MachineVerifier::new(core_verifier),
            recursive_core_verifier,
            recursive_compress_verifier,
            recursion_program_cache,
            reduce_shape,
            reduce_keys,
            reduce_programs,
            deferred_program,
            deferred_keys,
            shrink_program,
            shrink_keys,
            wrap_program,
            wrap_keys,
            max_reduce_arity,
            recursion_batch_size: 1,
        }
    }

    pub fn verifier(&self) -> &MachineVerifier<RecursionConfig<C>, CompressAir<RecursionF<C>>> {
        self.prover.verifier()
    }

    pub fn shrink_verifier(
        &self,
    ) -> &MachineVerifier<RecursionConfig<C>, CompressAir<RecursionF<C>>> {
        self.shrink_prover.verifier()
    }

    pub fn wrap_verifier(&self) -> &MachineVerifier<WrapConfig<C>, WrapAir<RecursionF<C>>> {
        self.wrap_prover.verifier()
    }

    pub fn machine(&self) -> &Machine<RecursionF<C>, CompressAir<RecursionF<C>>> {
        self.prover.machine()
    }

    /// Get the maximum reduce arity supported by the prover.
    pub fn max_reduce_arity(&self) -> usize {
        self.max_reduce_arity
    }

    #[inline]
    #[must_use]
    pub fn prove_shard(
        &self,
        pk: Arc<MachineProvingKey<C::RecursionComponents>>,
        record: ExecutionRecord<BabyBear>,
    ) -> TaskHandle<ShardProof<InnerSC>, MachineProverError> {
        self.prover.prove_shard(pk, record)
    }

    #[inline]
    #[must_use]
    pub fn prove_shrink(
        &self,
        record: ExecutionRecord<BabyBear>,
    ) -> TaskHandle<ShardProof<InnerSC>, MachineProverError> {
        self.shrink_prover.prove_shard(self.shrink_keys.0.clone(), record)
    }

    #[inline]
    #[must_use]
    pub fn prove_wrap(
        &self,
        record: ExecutionRecord<BabyBear>,
    ) -> TaskHandle<ShardProof<OuterSC>, MachineProverError> {
        self.wrap_prover.prove_shard(self.wrap_keys.0.clone(), record)
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

    #[inline]
    pub fn recursion_batch_size(&self) -> usize {
        self.recursion_batch_size
    }

    pub fn recursion_program(
        &self,
        input: &SP1RecursionWitnessValues<CoreSC>,
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
        let mut program = recursion_program_from_input(&self.recursive_core_verifier, input);
        program.shape = Some(self.reduce_shape.shape.clone());
        let program = Arc::new(program);
        self.recursion_program_cache.push(shape, program.clone());
        program
    }

    pub fn compress_program(
        &self,
        input: &SP1CompressWitnessValues<InnerSC>,
    ) -> Arc<RecursionProgram<BabyBear>> {
        let arity = input.vks_and_proofs.len();
        self.reduce_programs[&arity].clone()
    }

    pub fn recursion_program_cache_stats(&self) -> (usize, usize, f64) {
        self.recursion_program_cache.stats()
    }

    #[inline]
    #[must_use]
    pub(crate) fn compress_program_from_input(
        &self,
        input: &SP1CompressWitnessValues<InnerSC>,
    ) -> RecursionProgram<BabyBear> {
        compress_program_from_input(&self.recursive_compress_verifier, input)
    }

    pub fn dummy_reduce_input(&self, arity: usize) -> SP1CompressWitnessValues<InnerSC> {
        self.dummy_reduce_input_with_shape(arity, &self.reduce_shape)
    }

    pub(crate) fn dummy_reduce_input_with_shape(
        &self,
        arity: usize,
        shape: &SP1ReduceShape,
    ) -> SP1CompressWitnessValues<InnerSC> {
        dummy_reduce_input(&self.prover, shape, arity)
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn keys(
        &self,
        input: &SP1CircuitWitness,
    ) -> Option<(
        Arc<MachineProvingKey<C::RecursionComponents>>,
        MachineVerifyingKey<RecursionConfig<C>>,
    )> {
        match input {
            SP1CircuitWitness::Core(_) => None,
            SP1CircuitWitness::Deferred(_) => self.deferred_keys(),
            SP1CircuitWitness::Compress(input) => self.reduce_keys(input.vks_and_proofs.len()),
            SP1CircuitWitness::Shrink(_) => Some(self.shrink_keys.clone()),
            SP1CircuitWitness::Wrap(_) => None,
        }
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn wrap_keys(
        &self,
    ) -> (Arc<MachineProvingKey<C::WrapComponents>>, MachineVerifyingKey<WrapConfig<C>>) {
        self.wrap_keys.clone()
    }

    pub fn execute(
        &self,
        input: &SP1CircuitWitness,
    ) -> Result<ExecutionRecord<BabyBear>, SP1RecursionProverError> {
        let (program, witness_stream) = tracing::debug_span!("get program and witness stream")
            .in_scope(|| {
                match &input {
                    SP1CircuitWitness::Core(input) => {
                        let mut witness_stream = Vec::new();
                        Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
                        (self.recursion_program(input), witness_stream)
                    }
                    SP1CircuitWitness::Deferred(input) => {
                        let mut witness_stream = Vec::new();
                        Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
                        (self.deferred_program(), witness_stream)
                    }
                    SP1CircuitWitness::Compress(input) => {
                        let mut witness_stream = Vec::new();

                        // TODO: Add Merkle tree logic back in.
                        // let input_with_merkle = self.make_merkle_proofs(input);

                        Witnessable::<InnerConfig>::write(&input, &mut witness_stream);

                        (self.compress_program(input), witness_stream)
                    }
                    SP1CircuitWitness::Shrink(input) => {
                        let mut witness_stream = Vec::new();
                        Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
                        (self.shrink_program.clone(), witness_stream)
                    }
                    SP1CircuitWitness::Wrap(input) => {
                        let mut witness_stream = Vec::new();
                        Witnessable::<CircuitWrapConfig>::write(&input, &mut witness_stream);

                        (self.wrap_program.clone(), witness_stream)
                    }
                }
            });

        // Execute the runtime.
        let runtime_span = tracing::debug_span!("execute runtime").entered();
        let mut runtime =
            Runtime::<<InnerSC as JaggedConfig>::F, <InnerSC as JaggedConfig>::EF, _>::new(
                program.clone(),
                my_bb_16_perm(),
            );
        runtime.witness_stream = witness_stream.into();
        runtime.run().map_err(|e| SP1RecursionProverError::RuntimeError(e.to_string()))?;
        let record = runtime.record;
        runtime_span.exit();

        // Generate the dependencies.
        let mut records = vec![record];
        tracing::debug_span!("generate dependencies")
            .in_scope(|| self.machine().generate_dependencies(&mut records, None));
        let record = records.pop().unwrap();
        Ok(record)
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn deferred_keys(
        &self,
    ) -> Option<(
        Arc<MachineProvingKey<C::RecursionComponents>>,
        MachineVerifyingKey<RecursionConfig<C>>,
    )> {
        self.deferred_keys.clone()
    }

    pub fn deferred_program(&self) -> Arc<RecursionProgram<BabyBear>> {
        self.deferred_program.clone().unwrap()
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn reduce_keys(
        &self,
        arity: usize,
    ) -> Option<(
        Arc<MachineProvingKey<C::RecursionComponents>>,
        MachineVerifyingKey<RecursionConfig<C>>,
    )> {
        self.reduce_keys.get(&arity).cloned()
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

/// The "core" or "lift" program.
fn recursion_program_from_input(
    recursion_verifier: &RecursiveShardVerifier<
        RiscvAir<BabyBear>,
        CoreSC,
        InnerConfig,
        JC<InnerConfig, CoreSC>,
    >,
    input: &SP1RecursionWitnessValues<CoreSC>,
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
    let program = compiler.compile(dsl_program);
    compiler_span.exit();
    program
}

/// The deferred program.
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
    let program = compiler.compile(dsl_program);
    compiler_span.exit();
    program
}

/// The "compress" or "join" program.
fn compress_program_from_input(
    recursion_verifier: &RecursiveShardVerifier<
        CompressAir<InnerVal>,
        InnerSC,
        InnerConfig,
        JC<InnerConfig, InnerSC>,
    >,
    // vk_verification: bool,
    //TODO: Add VK verification back in
    input: &SP1CompressWitnessValues<InnerSC>,
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
    let program = compiler.compile(dsl_program);
    compiler_span.exit();
    program
}

/// The "shrink" program, which only verifies the single root shard.
fn shrink_program_from_input(
    recursion_verifier: &RecursiveShardVerifier<
        CompressAir<InnerVal>,
        InnerSC,
        InnerConfig,
        JC<InnerConfig, InnerSC>,
    >,
    input: &SP1CompressWitnessValues<InnerSC>,
    // todo add vk verification back in
) -> RecursionProgram<BabyBear> {
    let builder_span = tracing::debug_span!("build shrink program").entered();
    let mut builder = Builder::<InnerConfig>::default();
    // read the input.
    let input = input.read(&mut builder);

    // Verify the root proof.
    // todo add
    SP1CompressRootVerifier::verify(&mut builder, recursion_verifier, input);

    let block = builder.into_root_block();
    builder_span.exit();
    // SAFETY: The circuit is well-formed. It does not use synchronization primitives
    // (or possibly other means) to violate the invariants.
    let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

    // Compile the program.
    let compiler_span = tracing::debug_span!("compile shrink program").entered();
    let mut compiler = AsmCompiler::<InnerConfig>::default();
    let program = compiler.compile(dsl_program);
    compiler_span.exit();

    program
}

/// The "wrap" program, which only verifies the single root shard.
fn wrap_program_from_input(
    recursion_verifier: &RecursiveShardVerifier<
        CompressAir<InnerVal>,
        InnerSC,
        InnerConfig,
        JC<InnerConfig, InnerSC>,
    >,
    input: &SP1CompressWitnessValues<InnerSC>,
    // todo add vk verification back in
) -> RecursionProgram<BabyBear> {
    let builder_span = tracing::debug_span!("build wrap program").entered();
    let mut builder = Builder::<InnerConfig>::default();
    // read the input.
    let input = input.read(&mut builder);

    // Verify the root proof.
    // todo add vkey verification back in
    SP1CompressRootVerifier::verify(&mut builder, recursion_verifier, input);

    let block = builder.into_root_block();
    builder_span.exit();
    // SAFETY: The circuit is well-formed. It does not use synchronization primitives
    // (or possibly other means) to violate the invariants.
    let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

    // Compile the program.
    let compiler_span = tracing::debug_span!("compile wrap program").entered();
    let mut compiler = AsmCompiler::<InnerConfig>::default();
    let program = compiler.compile(dsl_program);
    compiler_span.exit();

    program
}

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

pub(crate) fn recursive_verifier<A, SC, C, JC>(
    shard_verifier: &ShardVerifier<SC, A>,
) -> RecursiveShardVerifier<A, SC, C, JC>
where
    A: MachineAir<C::F>,
    SC: BabyBearFriConfigVariable<C> + JaggedConfig<F = C::F, EF = C::EF>,
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
        JaggedEvaluator = RecursiveJaggedEvalSumcheckConfig<SC>,
    >,
{
    let log_stacking_height = shard_verifier.log_stacking_height();
    let max_log_row_count = shard_verifier.max_log_row_count();
    let machine = shard_verifier.machine().clone();
    let pcs_verifier = RecursiveBasefoldVerifier {
        fri_config: shard_verifier.pcs_verifier.stacked_pcs_verifier.pcs_verifier.fri_config,
        tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
    };
    let recursive_verifier = RecursiveStackedPcsVerifier::new(pcs_verifier, log_stacking_height);

    let recursive_jagged_verifier = RecursiveJaggedPcsVerifier {
        stacked_pcs_verifier: recursive_verifier,
        max_log_row_count,
        jagged_evaluator: RecursiveJaggedEvalSumcheckConfig::<SC>(PhantomData),
    };

    RecursiveShardVerifier {
        machine,
        pcs_verifier: recursive_jagged_verifier,
        _phantom: std::marker::PhantomData,
    }
}

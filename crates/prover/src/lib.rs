//! An end-to-end-prover implementation for the SP1 RISC-V zkVM.
//!
//! Separates the proof generation process into multiple stages:
//!
//! 1. Generate shard proofs which split up and prove the valid execution of a RISC-V program.
//! 2. Compress shard proofs into a single shard proof.
//! 3. Wrap the shard proof into a SNARK-friendly field.
//! 4. Wrap the last shard proof, proven over the SNARK-friendly field, into a PLONK proof.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::new_without_default)]
#![allow(clippy::collapsible_else_if)]

// pub mod build;
pub mod components;
// pub mod gas; // TODO reimplement gas
mod core;
pub mod error;
pub mod local;
mod recursion;
pub mod shapes;
mod types;
pub mod utils;
pub mod verify;

use core::SP1CoreProver;
pub use recursion::SP1RecursionProver;
use shapes::SP1RecursionShape;
use sp1_core_executor::Program;
use std::{collections::BTreeMap, sync::Arc};

use slop_baby_bear::BabyBear;

use sp1_recursion_executor::RecursionProgram;
use sp1_stark::prover::{CpuShardProver, MachineProverBuilder, ProverSemaphore, ShardProver};

use sp1_stark::{prover::MachineProvingKey, BabyBearPoseidon2};

pub use types::*;

use components::{CpuSP1ProverComponents, SP1ProverComponents};

/// The global version for all components of SP1.
///
/// This string should be updated whenever any step in verifying an SP1 proof changes, including
/// core, recursion, and plonk-bn254. This string is used to download SP1 artifacts and the gnark
/// docker image.
pub const SP1_CIRCUIT_VERSION: &str = include_str!("../SP1_VERSION");

/// The configuration for the core prover.
pub type CoreSC = BabyBearPoseidon2;
pub const CORE_LOG_BLOWUP: usize = 1;

/// The configuration for the inner prover.
pub type InnerSC = BabyBearPoseidon2;
pub const COMPRESS_LOG_BLOWUP: usize = 1;

// The configuration for the outer prover.
// pub type OuterSC = BabyBearPoseidon2Outer;

// pub type DeviceProvingKey<C> = <<C as SP1ProverComponents>::CoreProver as MachineProver<
//     BabyBearPoseidon2,
//     RiscvAir<BabyBear>,
// >>::DeviceProvingKey;
use sp1_recursion_machine::RecursionAir;

const COMPRESS_DEGREE: usize = 3;
const SHRINK_DEGREE: usize = 3;
const WRAP_DEGREE: usize = 9;

// const CORE_CACHE_SIZE: usize = 5;
pub const REDUCE_BATCH_SIZE: usize = 2;

pub type CompressAir<F> = RecursionAir<F, COMPRESS_DEGREE>;
pub type ShrinkAir<F> = RecursionAir<F, SHRINK_DEGREE>;
pub type WrapAir<F> = RecursionAir<F, WRAP_DEGREE>;

pub struct SP1Prover<C: SP1ProverComponents> {
    core_prover: SP1CoreProver<C::CoreComponents>,
    recursion_prover: SP1RecursionProver<C::RecursionComponents>,
}

pub struct SP1ProverBuilder<C: SP1ProverComponents> {
    core_prover_builder: MachineProverBuilder<C::CoreComponents>,
    recursion_prover_builder: MachineProverBuilder<C::RecursionComponents>,
    recursion_programs_cache_size: usize,
    max_reduce_arity: usize,
    join_pk: BTreeMap<SP1RecursionShape, Arc<MachineProvingKey<C::RecursionComponents>>>,
    recursion_programs: BTreeMap<SP1RecursionShape, Arc<RecursionProgram<BabyBear>>>,
}

impl<C: SP1ProverComponents> SP1ProverBuilder<C> {
    pub fn new_multi_permits(
        base_core_provers: Vec<Arc<ShardProver<C::CoreComponents>>>,
        core_prover_permits: Vec<ProverSemaphore>,
        nums_core_workers: Vec<usize>,
        base_recursion_provers: Vec<Arc<ShardProver<C::RecursionComponents>>>,
        recursion_prover_permits: Vec<ProverSemaphore>,
        nums_recursion_workers: Vec<usize>,
        recursion_programs_cache_size: usize,
        max_reduce_arity: usize,
    ) -> Self {
        let core_verifier = C::core_verifier();
        let core_prover_builder = MachineProverBuilder::new(
            core_verifier.shard_verifier().clone(),
            core_prover_permits,
            base_core_provers,
        );

        let recursion_verifier = C::recursion_verifier();
        let recursion_prover_builder = MachineProverBuilder::new(
            recursion_verifier.shard_verifier().clone(),
            recursion_prover_permits,
            base_recursion_provers,
        );

        let mut builder = Self {
            core_prover_builder,
            recursion_prover_builder,
            recursion_programs_cache_size,
            join_pk: BTreeMap::new(),
            recursion_programs: BTreeMap::new(),
            max_reduce_arity,
        };

        let _ = builder.num_core_workers_per_kind(nums_core_workers);
        let _ = builder.num_recursion_workers_per_kind(nums_recursion_workers);

        builder
    }

    pub fn new_single_permit(
        core_prover: ShardProver<C::CoreComponents>,
        core_prover_permit: ProverSemaphore,
        num_core_workers: usize,
        recursion_prover: ShardProver<C::RecursionComponents>,
        recursion_prover_permit: ProverSemaphore,
        num_recursion_workers: usize,
        recursion_programs_cache_size: usize,
        max_reduce_arity: usize,
    ) -> Self {
        Self::new_multi_permits(
            vec![Arc::new(core_prover)],
            vec![core_prover_permit],
            vec![num_core_workers],
            vec![Arc::new(recursion_prover)],
            vec![recursion_prover_permit],
            vec![num_recursion_workers],
            recursion_programs_cache_size,
            max_reduce_arity,
        )
    }

    pub fn max_reduce_arity(&mut self, max_reduce_arity: usize) -> &mut Self {
        self.max_reduce_arity = max_reduce_arity;
        self
    }

    /// Set the number of workers for a given base kind.
    pub fn num_core_workers_for_base_kind(
        &mut self,
        base_kind: usize,
        num_workers: usize,
    ) -> &mut Self {
        self.core_prover_builder.num_workers_for_base_kind(base_kind, num_workers);
        self
    }

    /// Set the number of workers for each base kind.
    pub fn num_core_workers_per_kind(&mut self, num_workers_per_kind: Vec<usize>) -> &mut Self {
        self.core_prover_builder.num_workers_per_kind(num_workers_per_kind);
        self
    }

    /// Set the number of workers for all base kinds.
    pub fn num_core_workers(&mut self, num_workers: usize) -> &mut Self {
        self.core_prover_builder.num_workers(num_workers);
        self
    }

    pub fn num_recursion_workers_for_base_kind(
        &mut self,
        base_kind: usize,
        num_workers: usize,
    ) -> &mut Self {
        self.recursion_prover_builder.num_workers_for_base_kind(base_kind, num_workers);
        self
    }

    pub fn num_recursion_workers_per_kind(
        &mut self,
        num_workers_per_kind: Vec<usize>,
    ) -> &mut Self {
        self.recursion_prover_builder.num_workers_per_kind(num_workers_per_kind);
        self
    }

    pub fn num_recursion_workers(&mut self, num_recursion_workers: usize) -> &mut Self {
        let _ = self.recursion_prover_builder.num_workers(num_recursion_workers);
        self
    }

    pub fn recursion_cache_size(&mut self, compress_programs_cache_size: usize) -> &mut Self {
        self.recursion_programs_cache_size = compress_programs_cache_size;
        self
    }

    pub fn with_join_pk(
        &mut self,
        join_pk: BTreeMap<SP1RecursionShape, Arc<MachineProvingKey<C::RecursionComponents>>>,
    ) -> &mut Self {
        self.join_pk = join_pk;
        self
    }

    pub fn insert_join_pk(
        &mut self,
        shape: SP1RecursionShape,
        pk: Arc<MachineProvingKey<C::RecursionComponents>>,
    ) -> &mut Self {
        self.join_pk.insert(shape, pk);
        self
    }

    pub fn with_recursion_programs(
        &mut self,
        recursion_programs: BTreeMap<SP1RecursionShape, Arc<RecursionProgram<BabyBear>>>,
    ) -> &mut Self {
        self.recursion_programs = recursion_programs;
        self
    }

    pub fn insert_recursion_program(
        &mut self,
        shape: SP1RecursionShape,
        program: Arc<RecursionProgram<BabyBear>>,
    ) -> &mut Self {
        self.recursion_programs.insert(shape, program);
        self
    }

    pub async fn build(&mut self) -> SP1Prover<C> {
        let core_prover = self.core_prover_builder.build();
        let core_verifier = core_prover.verifier().shard_verifier().clone();
        let core_prover = SP1CoreProver::new(core_prover);
        let recursion_prover = self.recursion_prover_builder.build();
        let _join_pk = std::mem::take(&mut self.join_pk);
        let recursion_programs = std::mem::take(&mut self.recursion_programs);
        let recursion_prover = SP1RecursionProver::new(
            core_verifier,
            recursion_prover,
            self.recursion_programs_cache_size,
            recursion_programs,
            self.max_reduce_arity,
        )
        .await;
        SP1Prover { core_prover, recursion_prover }
    }
}

impl<C: SP1ProverComponents> SP1Prover<C> {
    // TODO: hide behind builder pattern
    pub fn new(
        core_prover: SP1CoreProver<C::CoreComponents>,
        compress_prover: SP1RecursionProver<C::RecursionComponents>,
    ) -> Self {
        // TODO: make as part of the input.
        Self { core_prover, recursion_prover: compress_prover }
    }

    pub fn core(&self) -> &SP1CoreProver<C::CoreComponents> {
        &self.core_prover
    }

    pub fn recursion(&self) -> &SP1RecursionProver<C::RecursionComponents> {
        &self.recursion_prover
    }

    // /// Generate shard proofs which split up and prove the valid execution of a RISC-V program
    // with /// the core prover. Uses the provided context.
    // pub async fn prove_core(
    //     self: Arc<Self>,
    //     pk: Arc<MachineProvingKey<C::CoreComponents>>,
    //     program: Program,
    //     stdin: SP1Stdin,
    //     mut context: SP1Context<'static>,
    // ) -> Result<SP1CoreProof, SP1ProverError>
    // where
    //     C::CoreComponents: MachineProverComponents,
    // {
    //     context.subproof_verifier = Some(Arc::new(ArcProver(self.clone())));

    //     let (records_tx, mut records_rx) =
    //         mpsc::channel::<ExecutionRecord>(self.records_channel_capacity);

    //     let core_prover = self.clone();
    //     let shard_proof_handles = tokio::spawn(async move {
    //         let mut handles = Vec::new();
    //         while let Some(record) = records_rx.recv().await {
    //             let handle = core_prover.prove_core_shard(pk.clone(), record);
    //             handles.push(handle);
    //         }
    //         handles
    //     });

    //     // Run the machine executor.
    //     let output = self
    //         .core_prover
    //         .execute(Arc::new(program), stdin.clone(), context, records_tx)
    //         .await
    //         .unwrap();

    //     // Collect the shard proofs and wait for the to finish.
    //     let shard_proof_shandles = shard_proof_handles.await.unwrap();
    //     let mut shard_proofs = Vec::with_capacity(shard_proof_shandles.len());
    //     for handle in shard_proof_shandles {
    //         let proof = handle.await.map_err(SP1ProverError::CoreProverError)?;
    //         shard_proofs.push(proof);
    //     }

    //     let pv_stream = output.public_value_stream;
    //     let cycles = output.cycles;

    //     let public_values = SP1PublicValues::from(&pv_stream);
    //     Self::check_for_high_cycles(cycles);
    //     Ok(SP1CoreProof { proof: SP1CoreProofData(shard_proofs), stdin, public_values, cycles })
    // }

    /// Get the program from an elf.
    pub fn get_program(&self, elf: &[u8]) -> eyre::Result<Program> {
        let program = Program::from(elf)?;
        Ok(program)
    }

    // #[inline]
    // pub fn compress_program(
    //     &self,
    //     // TODO Add VKey back in
    //     input: &SP1CompressWitnessValues<InnerSC>,
    // ) -> Arc<RecursionProgram<BabyBear>> {
    //     self.recursion_prover.compress_program(input)
    //     // self.join_programs_map.get(&input.shape()).cloned().unwrap_or_else(|| {
    //     // tracing::warn!("join program not found in map, recomputing join program.");
    //     // Get the operations.
    //     // Arc::new(compress_program_from_input::<C>(&self.recursive_compress_verifier, input))
    //     // })
    // }

    // /// Generate the inputs for the first layer of recursive proofs.
    // #[allow(clippy::type_complexity)]
    // pub fn get_first_layer_inputs<'a>(
    //     &'a self,
    //     vk: &'a SP1VerifyingKey,
    //     shard_proofs: &[ShardProof<InnerSC>],
    //     deferred_proofs: &[SP1ReduceProof<InnerSC>],
    //     batch_size: usize,
    // ) -> Vec<SP1CircuitWitness> {
    //     let (deferred_inputs, deferred_digest) =
    //         self.get_recursion_deferred_inputs(vk, deferred_proofs, batch_size);

    //     assert!(deferred_proofs.is_empty());

    //     let is_complete = shard_proofs.len() == 1 && deferred_proofs.is_empty();
    //     let core_inputs = self.get_recursion_core_inputs(
    //         vk,
    //         shard_proofs,
    //         batch_size,
    //         is_complete,
    //         deferred_digest,
    //     );

    //     let mut inputs = Vec::new();
    //     inputs.extend(deferred_inputs.into_iter().map(SP1CircuitWitness::Deferred));
    //     inputs.extend(core_inputs.into_iter().map(SP1CircuitWitness::Core));
    //     inputs
    // }

    // pub fn leaf_compress_witness(
    //     &self,
    //     vk: SP1VerifyingKey,
    //     shard_proofs: Vec<ShardProof<CoreSC>>,
    //     is_complete: bool,
    //     is_first_shard: bool,
    //     deferred_digest: [<CoreSC as JaggedConfig>::F; 8],
    // ) -> SP1RecursionWitnessValues<CoreSC> {
    //     SP1RecursionWitnessValues {
    //         vk,
    //         shard_proofs,
    //         is_complete,
    //         is_first_shard,
    //         // vk_root: self.recursion_vk_root,
    //         reconstruct_deferred_digest: deferred_digest,
    //     }
    // }

    // pub fn deferred_program(
    //     &self,
    //     input: &SP1DeferredWitnessValues<InnerSC>,
    // ) -> Arc<RecursionProgram<BabyBear>> {
    //     // Compile the program.

    //     // Get the operations.
    //     let operations_span =
    //         tracing::debug_span!("get operations for the deferred program").entered();
    //     let mut builder = Builder::<InnerConfig>::default();
    //     let input_read_span = tracing::debug_span!("Read input values").entered();
    //     let input = input.read(&mut builder);
    //     input_read_span.exit();
    //     let verify_span = tracing::debug_span!("Verify deferred program").entered();

    //     // TODO: trait-safe way to do this?
    //     let mut challenger =
    //         <BabyBearPoseidon2 as BabyBearFriConfigVariable<InnerConfig>>::challenger_variable(
    //             &mut builder,
    //         );

    //     // Verify the proof.
    //     SP1DeferredVerifier::verify(
    //         &mut builder,
    //         &self.recursive_compress_verifier,
    //         input,
    //         &mut challenger, // self.vk_verification,
    //     );
    //     verify_span.exit();
    //     let block = builder.into_root_block();
    //     operations_span.exit();
    //     // SAFETY: The circuit is well-formed. It does not use synchronization primitives
    //     // (or possibly other means) to violate the invariants.
    //     let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

    //     let compiler_span = tracing::debug_span!("compile deferred program").entered();
    //     let mut compiler = AsmCompiler::<InnerConfig>::default();
    //     let program = compiler.compile(dsl_program);
    //     // if let Some(recursion_shape_config) = &self.compress_shape_config {
    //     //     recursion_shape_config.fix_shape(&mut program);
    //     // }
    //     let program = Arc::new(program);
    //     compiler_span.exit();
    //     program
    // }

    // / Accumulate deferred proofs into a single digest.
}

// /// Wrap a reduce proof into a STARK proven over a SNARK-friendly field.
// #[instrument(name = "shrink", level = "info", skip_all)]
// pub fn shrink(
//     &self,
//     reduced_proof: SP1ReduceProof<InnerSC>,
//     opts: SP1ProverOpts,
// ) -> Result<SP1ReduceProof<InnerSC>, SP1RecursionProverError> {
//     // Make the compress proof.
//     let SP1ReduceProof { vk: compressed_vk, proof: compressed_proof } = reduced_proof;
//     let input = SP1CompressWitnessValues {
//         vks_and_proofs: vec![(compressed_vk.clone(), compressed_proof)],
//         is_complete: true,
//     };

//     let input_with_merkle = self.make_merkle_proofs(input);

//     let program =
//         self.shrink_program(ShrinkAir::<BabyBear>::shrink_shape(), &input_with_merkle);

//     // Run the compress program.
//     let mut runtime = RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _>::new(
//         program.clone(),
//         self.shrink_prover.config().perm.clone(),
//     );

//     let mut witness_stream = Vec::new();
//     Witnessable::<InnerConfig>::write(&input_with_merkle, &mut witness_stream);

//     runtime.witness_stream = witness_stream.into();

//     runtime.run().map_err(|e| SP1RecursionProverError::RuntimeError(e.to_string()))?;

//     runtime.print_stats();
//     tracing::debug!("Shrink program executed successfully");

//     let (shrink_pk, shrink_vk) =
//         tracing::debug_span!("setup shrink").in_scope(|| self.shrink_prover.setup(&program));

//     // Prove the compress program.
//     let mut compress_challenger = self.shrink_prover.config().challenger();
//     let mut compress_proof = self
//         .shrink_prover
//         .prove(&shrink_pk, vec![runtime.record], &mut compress_challenger, opts.recursion_opts)
//         .unwrap();

//     Ok(SP1ReduceProof { vk: shrink_vk, proof: compress_proof.shard_proofs.pop().unwrap() })
// }

// /// Wrap a reduce proof into a STARK proven over a SNARK-friendly field.
// #[instrument(name = "wrap_bn254", level = "info", skip_all)]
// pub fn wrap_bn254(
//     &self,
//     compressed_proof: SP1ReduceProof<InnerSC>,
//     opts: SP1ProverOpts,
// ) -> Result<SP1ReduceProof<OuterSC>, SP1RecursionProverError> {
//     let SP1ReduceProof { vk: compressed_vk, proof: compressed_proof } = compressed_proof;
//     let input = SP1CompressWitnessValues {
//         vks_and_proofs: vec![(compressed_vk, compressed_proof)],
//         is_complete: true,
//     };
//     let input_with_vk = self.make_merkle_proofs(input);

//     let program = self.wrap_program();

//     // Run the compress program.
//     let mut runtime = RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _>::new(
//         program.clone(),
//         self.shrink_prover.config().perm.clone(),
//     );

//     let mut witness_stream = Vec::new();
//     Witnessable::<InnerConfig>::write(&input_with_vk, &mut witness_stream);

//     runtime.witness_stream = witness_stream.into();

//     runtime.run().map_err(|e| SP1RecursionProverError::RuntimeError(e.to_string()))?;

//     runtime.print_stats();
//     tracing::debug!("wrap program executed successfully");

//     // Setup the wrap program.
//     let (wrap_pk, wrap_vk) =
//         tracing::debug_span!("setup wrap").in_scope(|| self.wrap_prover.setup(&program));

//     if self.wrap_vk.set(wrap_vk.clone()).is_ok() {
//         tracing::debug!("wrap verifier key set");
//     }

//     // Prove the wrap program.
//     let mut wrap_challenger = self.wrap_prover.config().challenger();
//     let time = std::time::Instant::now();
//     let mut wrap_proof = self
//         .wrap_prover
//         .prove(&wrap_pk, vec![runtime.record], &mut wrap_challenger, opts.recursion_opts)
//         .unwrap();
//     let elapsed = time.elapsed();
//     tracing::debug!("wrap proving time: {:?}", elapsed);
//     let mut wrap_challenger = self.wrap_prover.config().challenger();
//     self.wrap_prover.machine().verify(&wrap_vk, &wrap_proof, &mut wrap_challenger).unwrap();
//     tracing::info!("wrapping successful");

//     Ok(SP1ReduceProof { vk: wrap_vk, proof: wrap_proof.shard_proofs.pop().unwrap() })
// }

// /// Wrap the STARK proven over a SNARK-friendly field into a PLONK proof.
// #[instrument(name = "wrap_plonk_bn254", level = "info", skip_all)]
// pub fn wrap_plonk_bn254(
//     &self,
//     proof: SP1ReduceProof<OuterSC>,
//     build_dir: &Path,
// ) -> PlonkBn254Proof {
//     let input = SP1CompressWitnessValues {
//         vks_and_proofs: vec![(proof.vk.clone(), proof.proof.clone())],
//         is_complete: true,
//     };
//     let vkey_hash = sp1_vkey_digest_bn254(&proof);
//     let committed_values_digest = sp1_committed_values_digest_bn254(&proof);

//     let mut witness = Witness::default();
//     input.write(&mut witness);
//     witness.write_committed_values_digest(committed_values_digest);
//     witness.write_vkey_hash(vkey_hash);

//     let prover = PlonkBn254Prover::new();
//     let proof = prover.prove(witness, build_dir.to_path_buf());

//     // Verify the proof.
//     prover.verify(
//         &proof,
//         &vkey_hash.as_canonical_biguint(),
//         &committed_values_digest.as_canonical_biguint(),
//         build_dir,
//     );

//     proof
// }

// /// Wrap the STARK proven over a SNARK-friendly field into a Groth16 proof.
// #[instrument(name = "wrap_groth16_bn254", level = "info", skip_all)]
// pub fn wrap_groth16_bn254(
//     &self,
//     proof: SP1ReduceProof<OuterSC>,
//     build_dir: &Path,
// ) -> Groth16Bn254Proof {
//     let input = SP1CompressWitnessValues {
//         vks_and_proofs: vec![(proof.vk.clone(), proof.proof.clone())],
//         is_complete: true,
//     };
//     let vkey_hash = sp1_vkey_digest_bn254(&proof);
//     let committed_values_digest = sp1_committed_values_digest_bn254(&proof);

//     let mut witness = Witness::default();
//     input.write(&mut witness);
//     witness.write_committed_values_digest(committed_values_digest);
//     witness.write_vkey_hash(vkey_hash);

//     let prover = Groth16Bn254Prover::new();
//     let proof = prover.prove(witness, build_dir.to_path_buf());

//     // Verify the proof.
//     prover.verify(
//         &proof,
//         &vkey_hash.as_canonical_biguint(),
//         &committed_values_digest.as_canonical_biguint(),
//         build_dir,
//     );

//     proof
// }

// pub fn recursion_program(
//     &self,
//     input: &SP1RecursionWitnessValues<CoreSC>,
// ) -> Arc<RecursionProgram<BabyBear>> {
//     // Check if the program is in the cache.
//     let mut cache = self.lift_programs_lru.lock().unwrap_or_else(|e| e.into_inner());
//     let shape = input.shape();
//     let program = cache.get(&shape).cloned();
//     drop(cache);
//     match program {
//         Some(program) => program,
//         None => {
//             let misses = self.lift_cache_misses.fetch_add(1, Ordering::Relaxed);
//             tracing::debug!("core cache miss, misses: {}", misses);
//             // Get the operations.
//             let builder_span = tracing::debug_span!("build recursion program").entered();
//             let mut builder = Builder::<InnerConfig>::default();

//             let input =
//                 tracing::debug_span!("read input").in_scope(|| input.read(&mut builder));
//             tracing::debug_span!("verify").in_scope(|| {
//                 SP1RecursiveVerifier::verify(&mut builder, self.core_prover.machine(), input)
//             });
//             let block =
//                 tracing::debug_span!("build block").in_scope(|| builder.into_root_block());
//             builder_span.exit();
//             // SAFETY: The circuit is well-formed. It does not use synchronization primitives
//             // (or possibly other means) to violate the invariants.
//             let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

//             // Compile the program.
//             let compiler_span = tracing::debug_span!("compile recursion program").entered();
//             let mut compiler = AsmCompiler::<InnerConfig>::default();
//             let mut program = compiler.compile(dsl_program);
//             if let Some(inn_recursion_shape_config) = &self.compress_shape_config {
//                 inn_recursion_shape_config.fix_shape(&mut program);
//             }
//             let program = Arc::new(program);
//             compiler_span.exit();

//             // Insert the program into the cache.
//             let mut cache = self.lift_programs_lru.lock().unwrap_or_else(|e| e.into_inner());
//             cache.put(shape, program.clone());
//             drop(cache);
//             program
//         }
//     }
// }

// pub fn compress_program(
//     &self,
//     input: &SP1CompressWithVKeyWitnessValues<InnerSC>,
// ) -> Arc<RecursionProgram<BabyBear>> {
//     self.join_programs_map.get(&input.shape()).cloned().unwrap_or_else(|| {
//         tracing::warn!("join program not found in map, recomputing join program.");
//         // Get the operations.
//         Arc::new(compress_program_from_input::<C>(
//             self.compress_shape_config.as_ref(),
//             &self.compress_prover,
//             self.vk_verification,
//             input,
//         ))
//     })
// }

// pub fn shrink_program(
//     &self,
//     shrink_shape: RecursionShape,
//     input: &SP1CompressWithVKeyWitnessValues<InnerSC>,
// ) -> Arc<RecursionProgram<BabyBear>> {
//     // Get the operations.
//     let builder_span = tracing::debug_span!("build shrink program").entered();
//     let mut builder = Builder::<InnerConfig>::default();
//     let input = input.read(&mut builder);
//     // Verify the proof.
//     SP1CompressRootVerifierWithVKey::verify(
//         &mut builder,
//         self.compress_prover.machine(),
//         input,
//         self.vk_verification,
//         PublicValuesOutputDigest::Reduce,
//     );
//     let block = builder.into_root_block();
//     builder_span.exit();
//     // SAFETY: The circuit is well-formed. It does not use synchronization primitives
//     // (or possibly other means) to violate the invariants.
//     let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

//     // Compile the program.
//     let compiler_span = tracing::debug_span!("compile shrink program").entered();
//     let mut compiler = AsmCompiler::<InnerConfig>::default();
//     let mut program = compiler.compile(dsl_program);

//     *program.shape_mut() = Some(shrink_shape);
//     let program = Arc::new(program);
//     compiler_span.exit();
//     program
// }

// pub fn wrap_program(&self) -> Arc<RecursionProgram<BabyBear>> {
//     self.wrap_program
//         .get_or_init(|| {
//             // Get the operations.
//             let builder_span = tracing::debug_span!("build compress program").entered();
//             let mut builder = Builder::<WrapConfig>::default();

//             let shrink_shape: OrderedShape = ShrinkAir::<BabyBear>::shrink_shape().into();
//             let input_shape = SP1CompressShape::from(vec![shrink_shape]);
//             let shape = SP1CompressWithVkeyShape {
//                 compress_shape: input_shape,
//                 merkle_tree_height: self.recursion_vk_tree.height,
//             };
//             let dummy_input =
//                 SP1CompressWithVKeyWitnessValues::dummy(self.shrink_prover.machine(), &shape);

//             let input = dummy_input.read(&mut builder);

//             // Attest that the merkle tree root is correct.
//             let root = input.merkle_var.root;
//             for (val, expected) in root.iter().zip(self.recursion_vk_root.iter()) {
//                 builder.assert_felt_eq(*val, *expected);
//             }
//             // Verify the proof.
//             SP1CompressRootVerifierWithVKey::verify(
//                 &mut builder,
//                 self.shrink_prover.machine(),
//                 input,
//                 self.vk_verification,
//                 PublicValuesOutputDigest::Root,
//             );

//             let block = builder.into_root_block();
//             builder_span.exit();
//             // SAFETY: The circuit is well-formed. It does not use synchronization primitives
//             // (or possibly other means) to violate the invariants.
//             let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

//             // Compile the program.
//             let compiler_span = tracing::debug_span!("compile compress program").entered();
//             let mut compiler = AsmCompiler::<WrapConfig>::default();
//             let program = Arc::new(compiler.compile(dsl_program));
//             compiler_span.exit();
//             program
//         })
//         .clone()
// }

//     pub fn make_merkle_proofs(
//         &self,
//         input: SP1CompressWitnessValues<CoreSC>,
//     ) -> SP1CompressWithVKeyWitnessValues<CoreSC> {
//         let num_vks = self.recursion_vk_map.len();
//         let (vk_indices, vk_digest_values): (Vec<_>, Vec<_>) = if self.vk_verification {
//             input
//                 .vks_and_proofs
//                 .iter()
//                 .map(|(vk, _)| {
//                     let vk_digest = vk.hash_babybear();
//                     let index = self.recursion_vk_map.get(&vk_digest).expect("vk not allowed");
//                     (index, vk_digest)
//                 })
//                 .unzip()
//         } else {
//             input
//                 .vks_and_proofs
//                 .iter()
//                 .map(|(vk, _)| {
//                     let vk_digest = vk.hash_babybear();
//                     let index = (vk_digest[0].as_canonical_u32() as usize) % num_vks;
//                     (index, [BabyBear::from_canonical_usize(index); 8])
//                 })
//                 .unzip()
//         };

//         let proofs = vk_indices
//             .iter()
//             .map(|index| {
//                 let (_, proof) = MerkleTree::open(&self.recursion_vk_tree, *index);
//                 proof
//             })
//             .collect();

//         let merkle_val = SP1MerkleProofWitnessValues {
//             root: self.recursion_vk_root,
//             values: vk_digest_values,
//             vk_merkle_proofs: proofs,
//         };

//         SP1CompressWithVKeyWitnessValues { compress_val: input, merkle_val }
//     }

// }

impl SP1ProverBuilder<CpuSP1ProverComponents> {
    pub fn cpu() -> Self {
        let cpu_ram_gb = sysinfo::System::new_all().total_memory() / (1024 * 1024 * 1024);
        let num_workers = match cpu_ram_gb {
            0..33 => 1,
            33..49 => 2,
            49..65 => 3,
            65..81 => 4,
            81.. => 4,
        };

        let prover_permits = ProverSemaphore::new(num_workers);

        let core_verifier = CpuSP1ProverComponents::core_verifier();
        let cpu_shard_prover = CpuShardProver::new(core_verifier.shard_verifier().clone());

        let recursion_verifier = CpuSP1ProverComponents::recursion_verifier();
        let recursion_shard_prover =
            CpuShardProver::new(recursion_verifier.shard_verifier().clone());

        let num_core_workers = num_workers;
        let num_recursion_workers = num_workers;
        let recursion_programs_cache_size = 5;
        let max_reduce_arity = 2;

        SP1ProverBuilder::new_single_permit(
            cpu_shard_prover,
            prover_permits.clone(),
            num_core_workers,
            recursion_shard_prover,
            prover_permits,
            num_recursion_workers,
            recursion_programs_cache_size,
            max_reduce_arity,
        )
    }
}

// #[cfg(test)]
// pub mod tests {
//     #![allow(clippy::print_stdout)]

//     use sp1_recursion_circuit::basefold::RecursiveBasefoldConfigImpl;
//     use sp1_recursion_circuit::jagged::RecursiveJaggedConfigImpl;

//     type SC = BabyBearPoseidon2;
//     type C = InnerConfig;

//     use crate::components::CpuSP1ProverComponents;

//     use super::*;
//     use sp1_recursion_circuit::jagged::RecursiveJaggedPcsVerifier;
//     use std::marker::PhantomData;

//     // use crate::build::try_build_plonk_bn254_artifacts_dev;
//     use anyhow::Result;
//     // use build::{build_constraints_and_witness, try_build_groth16_bn254_artifacts_dev};
//     use sp1_recursion_circuit::jagged::RecursiveJaggedEvalSumcheckConfig;

//     use sp1_recursion_circuit::basefold::stacked::RecursiveStackedPcsVerifier;
//     use sp1_recursion_circuit::basefold::tcs::RecursiveMerkleTreeTcs;
//     use sp1_recursion_circuit::basefold::RecursiveBasefoldVerifier;

//     #[cfg(test)]
//     use serial_test::serial;
//     #[cfg(test)]
//     use sp1_core_machine::utils::setup_logger;
//     use sp1_stark::prover::CpuProver;

//     #[derive(Debug, Clone, Copy, PartialEq, Eq)]
//     pub enum Test {
//         Core,
//         Compress,
//     }

//     pub async fn test_e2e_prover<C: SP1ProverComponents>(
//         prover: Arc<SP1Prover<C>>,
//         core_verifier: &ShardVerifier<CoreSC, RiscvAir<BabyBear>>,
//         elf: &[u8],
//         stdin: SP1Stdin,
//         opts: SP1ProverOpts,
//         test_kind: Test,
//     ) -> Result<()> {
//         run_e2e_prover_with_options(prover, elf, stdin, opts, test_kind, true,
// core_verifier).await     }

//     pub async fn run_e2e_prover_with_options<C: SP1ProverComponents>(
//         prover: Arc<SP1Prover<C>>,
//         elf: &[u8],
//         stdin: SP1Stdin,
//         opts: SP1ProverOpts,
//         test_kind: Test,
//         _verify: bool,
//         _core_verifier: &ShardVerifier<CoreSC, RiscvAir<BabyBear>>,
//     ) -> Result<()> {
//         tracing::info!("initializing prover");
//         let context = SP1Context::default();

//         tracing::info!("setup elf");
//         let (pk, program, vk) = prover.setup(elf).await;
//         let pk = Arc::new(pk);

//         tracing::info!("prove core");
//         let new_prover = prover.clone();
//         let core_proof = new_prover.prove_core(pk, program, &stdin, opts, context).await?;
//         // let public_values = core_proof.public_values.clone();

//         if test_kind == Test::Core {
//             return Ok(());
//         }

//         tracing::info!("compress");
//         let prover = prover.clone();
//         let compress_span = tracing::debug_span!("compress").entered();
//         let _compressed_proof = prover.compress(&vk, core_proof, vec![], opts).await?;
//         compress_span.exit();

//         if test_kind == Test::Compress {
//             return Ok(());
//         }

//         Ok(())
//     }

//     /// Tests an end-to-end workflow of proving a program across the entire proof generation
//     /// pipeline.
//     ///
//     /// Add `FRI_QUERIES`=1 to your environment for faster execution. Should only take a few
// minutes     /// on a Mac M2. Note: This test always re-builds the plonk bn254 artifacts, so
// setting SP1_DEV     /// is not needed.
//     #[tokio::test]
//     #[serial]
//     async fn test_e2e() -> Result<()> {
//         let elf = test_artifacts::SSZ_WITHDRAWALS_ELF;
//         setup_logger();
//         let opts = SP1ProverOpts::auto();
//         // TODO(mattstam): We should Test::Plonk here, but this uses the existing
//         // docker image which has a different API than the current. So we need to wait until the
//         // next release (v1.2.0+), and then switch it back.
//         let log_blowup = 1;
//         let log_stacking_height = 21;
//         let max_log_row_count = 21;
//         let machine = RiscvAir::machine();
//         let core_verifier = ShardVerifier::from_basefold_parameters(
//             log_blowup,
//             log_stacking_height,
//             max_log_row_count,
//             machine.clone(),
//         );

//         let recursive_verifier = RecursiveBasefoldVerifier {
//             fri_config: core_verifier.pcs_verifier.stacked_pcs_verifier.pcs_verifier.fri_config,
//             tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
//         };
//         let recursive_verifier =
//             RecursiveStackedPcsVerifier::new(recursive_verifier, log_stacking_height);

//         let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
//             SC,
//             C,
//             RecursiveJaggedConfigImpl<
//                 C,
//                 SC,
//                 RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
//             >,
//         > { stacked_pcs_verifier: recursive_verifier, max_log_row_count, jagged_evaluator:
//         > RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(PhantomData),
//         };

//         let core_stark_verifier = RecursiveShardVerifier {
//             machine,
//             pcs_verifier: recursive_jagged_verifier,
//             _phantom: std::marker::PhantomData,
//         };

//         let machine = RecursionAir::<BabyBear, 3>::machine_wide_with_all_chips();
//         let verifier_2 = ShardVerifier::from_basefold_parameters(
//             log_blowup,
//             log_stacking_height,
//             max_log_row_count,
//             machine.clone(),
//         );

//         let recursive_verifier = RecursiveBasefoldVerifier {
//             fri_config: core_verifier.pcs_verifier.stacked_pcs_verifier.pcs_verifier.fri_config,
//             tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
//         };
//         let recursive_verifier =
//             RecursiveStackedPcsVerifier::new(recursive_verifier, log_stacking_height);

//         let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
//             SC,
//             C,
//             RecursiveJaggedConfigImpl<
//                 C,
//                 SC,
//                 RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
//             >,
//         > { stacked_pcs_verifier: recursive_verifier, max_log_row_count, jagged_evaluator:
//         > RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(PhantomData),
//         };

//         let recursion_stark_verifier = RecursiveShardVerifier {
//             machine,
//             pcs_verifier: recursive_jagged_verifier,
//             _phantom: std::marker::PhantomData,
//         };

//         let core_prover = Arc::new(CpuProver::new(core_verifier.clone()));

//         let compress_prover = Arc::new(CpuProver::new(verifier_2.clone()));
//         let prover = SP1Prover {
//             compress_prover,
//             core_prover,
//             core_verifier: core_verifier.clone(),
//             recursive_core_verifier: core_stark_verifier,
//             recursive_compress_verifier: recursion_stark_verifier,
//             compress_verifier: verifier_2,
//         };

//         let prover = Arc::new(prover);

//         test_e2e_prover::<CpuSP1ProverComponents>(
//             prover,
//             &core_verifier,
//             elf,
//             SP1Stdin::default(),
//             opts,
//             Test::Compress,
//         )
//         .await
//     }
// }

// //     pub fn bench_e2e_prover<C: SP1ProverComponents>(
// //         prover: &SP1Prover<C>,
// //         elf: &[u8],
// //         stdin: SP1Stdin,
// //         opts: SP1ProverOpts,
// //         test_kind: Test,
// //     ) -> Result<()> {
// //         run_e2e_prover_with_options(prover, elf, stdin, opts, test_kind, false)
// //     }

// //     pub fn test_e2e_with_deferred_proofs_prover<C: SP1ProverComponents>(
// //         opts: SP1ProverOpts,
// //     ) -> Result<()> {
// //         // Test program which proves the Keccak-256 hash of various inputs.
// //         let keccak_elf = test_artifacts::KECCAK256_ELF;

// //         // Test program which verifies proofs of a vkey and a list of committed inputs.
// //         let verify_elf = test_artifacts::VERIFY_PROOF_ELF;

// //         tracing::info!("initializing prover");
// //         let prover = SP1Prover::<C>::new();

// //         tracing::info!("setup keccak elf");
// //         let (_, keccak_pk_d, keccak_program, keccak_vk) = prover.setup(keccak_elf);

// //         tracing::info!("setup verify elf");
// //         let (_, verify_pk_d, verify_program, verify_vk) = prover.setup(verify_elf);

// //         tracing::info!("prove subproof 1");
// //         let mut stdin = SP1Stdin::new();
// //         stdin.write(&1usize);
// //         stdin.write(&vec![0u8, 0, 0]);
// //         let deferred_proof_1 = prover.prove_core(
// //             &keccak_pk_d,
// //             keccak_program.clone(),
// //             &stdin,
// //             opts,
// //             Default::default(),
// //         )?;
// //         let pv_1 = deferred_proof_1.public_values.as_slice().to_vec().clone();

// //         // Generate a second proof of keccak of various inputs.
// //         tracing::info!("prove subproof 2");
// //         let mut stdin = SP1Stdin::new();
// //         stdin.write(&3usize);
// //         stdin.write(&vec![0u8, 1, 2]);
// //         stdin.write(&vec![2, 3, 4]);
// //         stdin.write(&vec![5, 6, 7]);
// //         let deferred_proof_2 =
// //             prover.prove_core(&keccak_pk_d, keccak_program, &stdin, opts,
// Default::default())?; //         let pv_2 =
// deferred_proof_2.public_values.as_slice().to_vec().clone();

// //         // Generate recursive proof of first subproof.
// //         tracing::info!("compress subproof 1");
// //         let deferred_reduce_1 = prover.compress(&keccak_vk, deferred_proof_1, vec![], opts)?;
// //         prover.verify_compressed(&deferred_reduce_1, &keccak_vk)?;

// //         // Generate recursive proof of second subproof.
// //         tracing::info!("compress subproof 2");
// //         let deferred_reduce_2 = prover.compress(&keccak_vk, deferred_proof_2, vec![], opts)?;
// //         prover.verify_compressed(&deferred_reduce_2, &keccak_vk)?;

// //         // Run verify program with keccak vkey, subproofs, and their committed values.
// //         let mut stdin = SP1Stdin::new();
// //         let vkey_digest = keccak_vk.hash_babybear();
// //         let vkey_digest: [u32; 8] = vkey_digest
// //             .iter()
// //             .map(|n| n.as_canonical_u32())
// //             .collect::<Vec<_>>()
// //             .try_into()
// //             .unwrap();
// //         stdin.write(&vkey_digest);
// //         stdin.write(&vec![pv_1.clone(), pv_2.clone(), pv_2.clone()]);
// //         stdin.write_proof(deferred_reduce_1.clone(), keccak_vk.vk.clone());
// //         stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());
// //         stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());

// //         tracing::info!("proving verify program (core)");
// //         let verify_proof =
// //             prover.prove_core(&verify_pk_d, verify_program, &stdin, opts,
// Default::default())?; //         // let public_values = verify_proof.public_values.clone();

// //         // Generate recursive proof of verify program
// //         tracing::info!("compress verify program");
// //         let verify_reduce = prover.compress(
// //             &verify_vk,
// //             verify_proof,
// //             vec![deferred_reduce_1, deferred_reduce_2.clone(), deferred_reduce_2],
// //             opts,
// //         )?;
// //         let reduce_pv: &RecursionPublicValues<_> =
// //             verify_reduce.proof.public_values.as_slice().borrow();
// //         println!("deferred_hash: {:?}", reduce_pv.deferred_proofs_digest);
// //         println!("complete: {:?}", reduce_pv.is_complete);

// //         tracing::info!("verify verify program");
// //         prover.verify_compressed(&verify_reduce, &verify_vk)?;

// //         let shrink_proof = prover.shrink(verify_reduce, opts)?;

// //         tracing::info!("verify shrink");
// //         prover.verify_shrink(&shrink_proof, &verify_vk)?;

// //         tracing::info!("wrap bn254");
// //         let wrapped_bn254_proof = prover.wrap_bn254(shrink_proof, opts)?;

// //         tracing::info!("verify wrap bn254");
// //         println!("verify wrap bn254 {:#?}", wrapped_bn254_proof.vk.commit);
// //         prover.verify_wrap_bn254(&wrapped_bn254_proof, &verify_vk).unwrap();

// //         Ok(())
// //     }

// //     /// Tests an end-to-end workflow of proving a program across the entire proof generation
// //     /// pipeline in addition to verifying deferred proofs.
// //     #[test]
// //     #[serial]
// //     fn test_e2e_with_deferred_proofs() -> Result<()> {
// //         setup_logger();
// //         test_e2e_with_deferred_proofs_prover::<CpuProverComponents>(SP1ProverOpts::auto())
// //     }
// // }

use std::{
    collections::BTreeSet,
    num::NonZero,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use lru::LruCache;
use serde::{Deserialize, Serialize};
use slop_air::BaseAir;
use slop_algebra::AbstractField;
use slop_baby_bear::BabyBear;
use sp1_core_machine::riscv::RiscvAir;
use sp1_recursion_circuit::{
    dummy::{dummy_shard_proof, dummy_vk},
    machine::{
        SP1CompressWithVKeyWitnessValues, SP1CompressWitnessValues, SP1MerkleProofWitnessValues,
        SP1RecursionWitnessValues,
    },
};
use sp1_recursion_executor::{shape::RecursionShape, RecursionProgram, DIGEST_SIZE};
use sp1_recursion_machine::chips::{
    alu_base::BaseAluChip,
    alu_ext::ExtAluChip,
    mem::{MemoryConstChip, MemoryVarChip},
    poseidon2_wide::Poseidon2WideChip,
    prefix_sum_checks::PrefixSumChecksChip,
    public_values::PublicValuesChip,
    select::SelectChip,
};
use sp1_stark::{
    air::MachineAir,
    prover::{CoreProofShape, DefaultTraceGenerator, ProverSemaphore, TraceGenerator},
    Chip, Machine,
};

use crate::{
    components::SP1ProverComponents, recursion::RECURSION_MAX_LOG_ROW_COUNT, CompressAir, CoreSC,
    InnerSC, SP1RecursionProver, SP1VerifyingKey, ShrinkAir,
};

/// The shape of the recursion proof.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SP1RecursionShape {
    pub proof_shapes: Vec<CoreProofShape<BabyBear, RiscvAir<BabyBear>>>,
    pub max_log_row_count: usize,
    pub log_blowup: usize,
    pub log_stacking_height: usize,
}

impl SP1RecursionShape {
    pub fn dummy_input(&self, vk: SP1VerifyingKey) -> SP1RecursionWitnessValues<CoreSC> {
        let shard_proofs = self
            .proof_shapes
            .iter()
            .map(|core_shape| {
                dummy_shard_proof(
                    core_shape.shard_chips.clone(),
                    self.max_log_row_count,
                    self.log_blowup,
                    self.log_stacking_height,
                    &[core_shape.preprocessed_multiple, core_shape.main_multiple],
                )
            })
            .collect::<Vec<_>>();

        SP1RecursionWitnessValues {
            vk: vk.vk,
            shard_proofs,
            is_complete: false,
            is_first_shard: false,
            vk_root: [BabyBear::zero(); DIGEST_SIZE],
            reconstruct_deferred_digest: [BabyBear::zero(); 8],
        }
    }
}

pub struct SP1RecursionCache {
    lru: Arc<Mutex<LruCache<SP1RecursionShape, Arc<RecursionProgram<BabyBear>>>>>,
    total_calls: AtomicUsize,
    hits: AtomicUsize,
}

impl SP1RecursionCache {
    pub fn new(size: usize) -> Self {
        let size = NonZero::new(size).expect("size must be non-zero");
        let lru = LruCache::new(size);
        let lru = Arc::new(Mutex::new(lru));
        Self { lru, total_calls: AtomicUsize::new(0), hits: AtomicUsize::new(0) }
    }

    pub fn get(&self, shape: &SP1RecursionShape) -> Option<Arc<RecursionProgram<BabyBear>>> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(program) = self.lru.lock().unwrap().get(shape).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(program)
        } else {
            None
        }
    }

    pub fn push(&self, shape: SP1RecursionShape, program: Arc<RecursionProgram<BabyBear>>) {
        self.lru.lock().unwrap().push(shape, program);
    }

    pub fn stats(&self) -> (usize, usize, f64) {
        (
            self.total_calls.load(Ordering::Relaxed),
            self.hits.load(Ordering::Relaxed),
            self.hits.load(Ordering::Relaxed) as f64
                / self.total_calls.load(Ordering::Relaxed) as f64,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SP1ReduceShape {
    pub shape: RecursionShape<BabyBear>,
}

impl Default for SP1ReduceShape {
    fn default() -> Self {
        Self::reduce_shape_from_arity(2).unwrap()
    }
}

impl SP1ReduceShape {
    pub fn reduce_shape_from_arity(arity: usize) -> Option<Self> {
        let shape = match arity {
            1 | 2 => [
                (CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 200640),
                (CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 254080),
                (CompressAir::<BabyBear>::BaseAlu(BaseAluChip), 242976),
                (CompressAir::<BabyBear>::ExtAlu(ExtAluChip), 348544),
                (CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 58784),
                (CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 249984),
                (CompressAir::<BabyBear>::Select(SelectChip), 403488),
                (CompressAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
            ]
            .into_iter()
            .collect(),
            3 => [
                (CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 301280),
                (CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 385184),
                (CompressAir::<BabyBear>::BaseAlu(BaseAluChip), 364416),
                (CompressAir::<BabyBear>::ExtAlu(ExtAluChip), 544224),
                (CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 89120),
                (CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 249984),
                (CompressAir::<BabyBear>::Select(SelectChip), 605248),
                (CompressAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
            ]
            .into_iter()
            .collect(),
            4 => [
                (CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 402016),
                (CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 529280),
                (CompressAir::<BabyBear>::BaseAlu(BaseAluChip), 485824),
                (CompressAir::<BabyBear>::ExtAlu(ExtAluChip), 751232),
                (CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 120064),
                (CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 249984),
                (CompressAir::<BabyBear>::Select(SelectChip), 806976),
                (CompressAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
            ]
            .into_iter()
            .collect(),
            _ => return None,
        };
        Some(Self { shape })
    }

    pub fn shrink_shape_from_arity(arity: usize) -> Option<Self> {
        let shape = match arity {
            4 => [
                (ShrinkAir::<BabyBear>::BaseAlu(BaseAluChip), 121568),
                (ShrinkAir::<BabyBear>::ExtAlu(ExtAluChip), 187808),
                (ShrinkAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 100736),
                (ShrinkAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 129472),
                (ShrinkAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 30048),
                (ShrinkAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 26112),
                (ShrinkAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
                (ShrinkAir::<BabyBear>::Select(SelectChip), 201760),
            ],
            _ => return None,
        };

        Some(Self { shape: shape.into_iter().collect() })
    }

    pub fn dummy_input(
        &self,
        arity: usize,
        height: usize,
        chips: BTreeSet<Chip<BabyBear, CompressAir<BabyBear>>>,
        max_log_row_count: usize,
        log_blowup: usize,
        log_stacking_height: usize,
    ) -> SP1CompressWithVKeyWitnessValues<InnerSC> {
        let preprocessed_chip_information = self.shape.preprocessed_chip_information(&chips);
        let dummy_vk = dummy_vk(preprocessed_chip_information);

        let preprocessed_multiple = chips
            .iter()
            .map(|chip| self.shape.height(chip).unwrap() * chip.preprocessed_width())
            .sum::<usize>()
            .div_ceil(1 << log_stacking_height);

        let main_multiple = chips
            .iter()
            .map(|chip| self.shape.height(chip).unwrap() * chip.width())
            .sum::<usize>()
            .div_ceil(1 << log_stacking_height);

        let dummy_proof = dummy_shard_proof(
            chips,
            max_log_row_count,
            log_blowup,
            log_stacking_height,
            &[preprocessed_multiple, main_multiple],
        );

        let vks_and_proofs =
            (0..arity).map(|_| (dummy_vk.clone(), dummy_proof.clone())).collect::<Vec<_>>();

        SP1CompressWithVKeyWitnessValues {
            compress_val: SP1CompressWitnessValues { vks_and_proofs, is_complete: false },
            merkle_val: SP1MerkleProofWitnessValues::dummy(arity, height),
        }
    }

    pub async fn check_compatibility(
        &self,
        program: Arc<RecursionProgram<BabyBear>>,
        machine: Machine<BabyBear, CompressAir<BabyBear>>,
    ) -> bool {
        // Generate the preprocessed traces to get the heights.
        let trace_generator = DefaultTraceGenerator::new(machine);
        let setup_permits = ProverSemaphore::new(1);
        let preprocessed_traces = trace_generator
            .generate_preprocessed_traces(program, RECURSION_MAX_LOG_ROW_COUNT, setup_permits)
            .await;

        let mut is_compatible = true;
        for (chip, trace) in preprocessed_traces.preprocessed_traces.into_iter() {
            let real_height = trace.num_real_entries();
            let expected_height = self.shape.height_of_name(&chip).unwrap();
            if real_height > expected_height {
                tracing::warn!(
                    "program is incompatible with shape: {} > {} for chip {}",
                    real_height,
                    expected_height,
                    chip
                );
                is_compatible = false;
            }
        }
        is_compatible
    }

    #[allow(dead_code)]
    async fn max_arity<C: SP1ProverComponents>(&self, prover: &SP1RecursionProver<C>) -> usize {
        let mut arity = 0;
        for possible_arity in 1.. {
            let input = prover.dummy_reduce_input_with_shape(possible_arity, self);
            let program = prover.compress_program_from_input(&input);
            let program = Arc::new(program);
            let is_compatible = self.check_compatibility(program, prover.machine().clone()).await;
            if !is_compatible {
                break;
            }
            arity = possible_arity;
        }
        arity
    }
}

// use std::{
//     // collections::{BTreeMap, BTreeSet, HashSet},
//     // fs::File,
//     hash::{DefaultHasher, Hash, Hasher},
//     // panic::{catch_unwind, AssertUnwindSafe},
//     // path::PathBuf,
//     // sync::{Arc, Mutex},
// };

// // use eyre::Result;
// use serde::{Deserialize, Serialize};
// // use slop_algebra::AbstractField;
// // use slop_baby_bear::BabyBear;
// // use sp1_core_machine::shape::CoreShapeConfig;
// use sp1_recursion_circuit::{SP1CompressWithVkeyShape, SP1DeferredShape, SP1RecursionShape};
// // use sp1_recursion_executor::RecursionProgram;
// // use sp1_recursion_circuit::machine::{
// //     SP1CompressWithVKeyWitnessValues, SP1CompressWithVkeyShape, SP1DeferredShape,
// //     SP1DeferredWitnessValues, SP1RecursionShape, SP1RecursionWitnessValues,
// // };
// // use sp1_recursion_core::{
// //     //     shape::{RecursionShape, RecursionShapeConfig},
// //     RecursionProgram,
// // };
// use sp1_stark::shape::OrderedShape;
// use thiserror::Error;

// use crate::{components::SP1ProverComponents, CompressAir, HashableKey, SP1Prover, ShrinkAir};

// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
// pub enum SP1ProofShape {
//     Recursion(OrderedShape),
//     Compress(Vec<OrderedShape>),
//     Deferred(OrderedShape),
//     Shrink(OrderedShape),
// }

// #[derive(Debug, Clone, Hash)]
// pub enum SP1CompressProgramShape {
//     Recursion(SP1RecursionShape),
//     Compress(SP1CompressWithVkeyShape),
//     Deferred(SP1DeferredShape),
//     Shrink(SP1CompressWithVkeyShape),
// }

// impl SP1CompressProgramShape {
//     pub fn hash_u64(&self) -> u64 {
//         let mut hasher = DefaultHasher::new();
//         Hash::hash(&self, &mut hasher);
//         hasher.finish()
//     }
// }

// #[derive(Debug, Error)]
// pub enum VkBuildError {
//     #[error("IO error: {0}")]
//     IO(#[from] std::io::Error),
//     #[error("Serialization error: {0}")]
//     Bincode(#[from] bincode::Error),
// }

// pub fn check_shapes<C: SP1ProverComponents>(
//     reduce_batch_size: usize,
//     no_precompiles: bool,
//     num_compiler_workers: usize,
//     prover: &mut SP1Prover<C>,
// ) -> bool {
//     let (shape_tx, shape_rx) =
//         std::sync::mpsc::sync_channel::<SP1CompressProgramShape>(num_compiler_workers);
//     let (panic_tx, panic_rx) = std::sync::mpsc::channel();
//     let core_shape_config = prover.core_shape_config.as_ref().expect("core shape config not
// found");     let recursion_shape_config =
//         prover.compress_shape_config.as_ref().expect("recursion shape config not found");

//     let shape_rx = Mutex::new(shape_rx);

//     let all_maximal_shapes = SP1ProofShape::generate_maximal_shapes(
//         core_shape_config,
//         recursion_shape_config,
//         reduce_batch_size,
//         no_precompiles,
//     )
//     .collect::<BTreeSet<SP1ProofShape>>();
//     let num_shapes = all_maximal_shapes.len();
//     tracing::info!("number of shapes: {}", num_shapes);

//     // The Merkle tree height.
//     let height = num_shapes.next_power_of_two().ilog2() as usize;

//     // Empty the join program map so that we recompute the join program.
//     prover.join_programs_map.clear();

//     let compress_ok = std::thread::scope(|s| {
//         // Initialize compiler workers.
//         for _ in 0..num_compiler_workers {
//             let shape_rx = &shape_rx;
//             let prover = &prover;
//             let panic_tx = panic_tx.clone();
//             s.spawn(move || {
//                 while let Ok(shape) = shape_rx.lock().unwrap().recv() {
//                     tracing::info!("shape is {:?}", shape);
//                     let program = catch_unwind(AssertUnwindSafe(|| {
//                         // Try to build the recursion program from the given shape.
//                         prover.program_from_shape(shape.clone(), None)
//                     }));
//                     match program {
//                         Ok(_) => {}
//                         Err(e) => {
//                             tracing::warn!(
//                                 "Program generation failed for shape {:?}, with error: {:?}",
//                                 shape,
//                                 e
//                             );
//                             panic_tx.send(true).unwrap();
//                         }
//                     }
//                 }
//             });
//         }

//         // Generate shapes and send them to the compiler workers.
//         all_maximal_shapes.into_iter().for_each(|program_shape| {
//             shape_tx
//                 .send(SP1CompressProgramShape::from_proof_shape(program_shape, height))
//                 .unwrap();
//         });

//         drop(shape_tx);
//         drop(panic_tx);

//         // If the panic receiver has no panics, then the shape is correct.
//         panic_rx.iter().next().is_none()
//     });

//     compress_ok
// }

// pub fn build_vk_map<C: SP1ProverComponents + 'static>(
//     reduce_batch_size: usize,
//     dummy: bool,
//     num_compiler_workers: usize,
//     num_setup_workers: usize,
//     indices: Option<Vec<usize>>,
// ) -> (BTreeSet<[BabyBear; DIGEST_SIZE]>, Vec<usize>, usize) {
//     // Setup the prover.
//     let mut prover = SP1Prover::<C>::new();
//     prover.vk_verification = !dummy;
//     if !dummy {
//         prover.join_programs_map.clear();
//     }
//     let prover = Arc::new(prover);

//     // Get the shape configs.
//     let core_shape_config = prover.core_shape_config.as_ref().expect("core shape config not
// found");     let recursion_shape_config =
//         prover.compress_shape_config.as_ref().expect("recursion shape config not found");

//     let (vk_set, panic_indices, height) = if dummy {
//         tracing::warn!("building a dummy vk map");
//         let dummy_set = SP1ProofShape::dummy_vk_map(
//             core_shape_config,
//             recursion_shape_config,
//             reduce_batch_size,
//         )
//         .into_keys()
//         .collect::<BTreeSet<_>>();
//         let height = dummy_set.len().next_power_of_two().ilog2() as usize;
//         (dummy_set, vec![], height)
//     } else {
//         tracing::info!("building vk map");

//         // Setup the channels.
//         let (vk_tx, vk_rx) = std::sync::mpsc::channel();
//         let (shape_tx, shape_rx) =
//             std::sync::mpsc::sync_channel::<(usize,
// SP1CompressProgramShape)>(num_compiler_workers);         let (program_tx, program_rx) =
// std::sync::mpsc::sync_channel(num_setup_workers);         let (panic_tx, panic_rx) =
// std::sync::mpsc::channel();

//         // Setup the mutexes.
//         let shape_rx = Mutex::new(shape_rx);
//         let program_rx = Mutex::new(program_rx);

//         // Generate all the possible shape inputs we encounter in recursion. This may span lift,
//         // join, deferred, shrink, etc.
//         let indices_set = indices.map(|indices| indices.into_iter().collect::<HashSet<_>>());
//         let mut all_shapes = BTreeSet::new();
//         let start = std::time::Instant::now();
//         for shape in
//             SP1ProofShape::generate(core_shape_config, recursion_shape_config, reduce_batch_size)
//         {
//             all_shapes.insert(shape);
//         }

//         let num_shapes = all_shapes.len();
//         tracing::info!("number of shapes: {} in {:?}", num_shapes, start.elapsed());

//         let height = num_shapes.next_power_of_two().ilog2() as usize;
//         let chunk_size = indices_set.as_ref().map(|indices| indices.len()).unwrap_or(num_shapes);

//         std::thread::scope(|s| {
//             // Initialize compiler workers.
//             for _ in 0..num_compiler_workers {
//                 let program_tx = program_tx.clone();
//                 let shape_rx = &shape_rx;
//                 let prover = prover.clone();
//                 let panic_tx = panic_tx.clone();
//                 s.spawn(move || {
//                     while let Ok((i, shape)) = shape_rx.lock().unwrap().recv() {
//                         eprintln!("shape: {:?}", shape);
//                         let is_shrink = matches!(shape, SP1CompressProgramShape::Shrink(_));
//                         let prover = prover.clone();
//                         let shape_clone = shape.clone();
//                         // Spawn on another thread to handle panics.
//                         let program_thread = std::thread::spawn(move || {
//                             prover.program_from_shape(shape_clone, None)
//                         });
//                         match program_thread.join() {
//                             Ok(program) => program_tx.send((i, program, is_shrink)).unwrap(),
//                             Err(e) => {
//                                 tracing::warn!(
//                                     "Program generation failed for shape {} {:?}, with error:
// {:?}",                                     i,
//                                     shape,
//                                     e
//                                 );
//                                 panic_tx.send(i).unwrap();
//                             }
//                         }
//                     }
//                 });
//             }

//             // Initialize setup workers.
//             for _ in 0..num_setup_workers {
//                 let vk_tx = vk_tx.clone();
//                 let program_rx = &program_rx;
//                 let prover = &prover;
//                 let panic_tx = panic_tx.clone();
//                 s.spawn(move || {
//                     let mut done = 0;
//                     while let Ok((i, program, is_shrink)) = program_rx.lock().unwrap().recv() {
//                         let prover = prover.clone();
//                         let vk_thread = std::thread::spawn(move || {
//                             if is_shrink {
//                                 prover.shrink_prover.setup(&program).1
//                             } else {
//                                 prover.compress_prover.setup(&program).1
//                             }
//                         });
//                         let vk = tracing::debug_span!("setup for program {}", i)
//                             .in_scope(|| vk_thread.join());
//                         done += 1;

//                         if let Err(e) = vk {
//                             tracing::error!("failed to setup program {}: {:?}", i, e);
//                             panic_tx.send(i).unwrap();
//                             continue;
//                         }
//                         let vk = vk.unwrap();

//                         let vk_digest = vk.hash_babybear();
//                         tracing::info!(
//                             "program {} = {:?}, {}% done",
//                             i,
//                             vk_digest,
//                             done * 100 / chunk_size
//                         );
//                         vk_tx.send(vk_digest).unwrap();
//                     }
//                 });
//             }

//             // Generate shapes and send them to the compiler workers.
//             let subset_shapes = all_shapes
//                 .into_iter()
//                 .enumerate()
//                 .filter(|(i, _)| indices_set.as_ref().map(|set| set.contains(i)).unwrap_or(true))
//                 .collect::<Vec<_>>();

//             subset_shapes
//                 .clone()
//                 .into_iter()
//                 .map(|(i, shape)| (i, SP1CompressProgramShape::from_proof_shape(shape, height)))
//                 .for_each(|(i, program_shape)| {
//                     shape_tx.send((i, program_shape)).unwrap();
//                 });

//             drop(shape_tx);
//             drop(program_tx);
//             drop(vk_tx);
//             drop(panic_tx);

//             let vk_set = vk_rx.iter().collect::<BTreeSet<_>>();

//             let panic_indices = panic_rx.iter().collect::<Vec<_>>();
//             for (i, shape) in subset_shapes {
//                 if panic_indices.contains(&i) {
//                     tracing::info!("panic shape {}: {:?}", i, shape);
//                 }
//             }

//             (vk_set, panic_indices, height)
//         })
//     };
//     tracing::info!("compress vks generated, number of keys: {}", vk_set.len());
//     (vk_set, panic_indices, height)
// }

// pub fn build_vk_map_to_file<C: SP1ProverComponents + 'static>(
//     build_dir: PathBuf,
//     reduce_batch_size: usize,
//     dummy: bool,
//     num_compiler_workers: usize,
//     num_setup_workers: usize,
//     range_start: Option<usize>,
//     range_end: Option<usize>,
// ) -> Result<(), VkBuildError> {
//     // Create the build directory if it doesn't exist.
//     std::fs::create_dir_all(&build_dir)?;

//     // Build the vk map.
//     let (vk_set, _, _) = build_vk_map::<C>(
//         reduce_batch_size,
//         dummy,
//         num_compiler_workers,
//         num_setup_workers,
//         range_start.and_then(|start| range_end.map(|end| (start..end).collect())),
//     );

//     // Serialize the vk into an ordering.
//     let vk_map = vk_set.into_iter().enumerate().map(|(i, vk)| (vk, i)).collect::<BTreeMap<_,
// _>>();

//     // Create the file to store the vk map.
//     let mut file = if dummy {
//         File::create(build_dir.join("dummy_vk_map.bin"))?
//     } else {
//         File::create(build_dir.join("vk_map.bin"))?
//     };

//     Ok(bincode::serialize_into(&mut file, &vk_map)?)
// }

// impl SP1ProofShape {
//     pub fn generate<'a>(
//         core_shape_config: &'a CoreShapeConfig<BabyBear>,
//         recursion_shape_config: &'a RecursionShapeConfig<BabyBear, CompressAir<BabyBear>>,
//         reduce_batch_size: usize,
//     ) -> impl Iterator<Item = Self> + 'a {
//         core_shape_config
//             .all_shapes()
//             .map(Self::Recursion)
//             .chain((1..=reduce_batch_size).flat_map(|batch_size| {
//                 recursion_shape_config.get_all_shape_combinations(batch_size).map(Self::Compress)
//             }))
//             .chain(
//                 recursion_shape_config
//                     .get_all_shape_combinations(1)
//                     .map(|mut x| Self::Deferred(x.pop().unwrap())),
//             )
//             .chain(
//                 recursion_shape_config
//                     .get_all_shape_combinations(1)
//                     .map(|mut x| Self::Shrink(x.pop().unwrap())),
//             )
//     }

//     pub fn generate_compress_shapes(
//         recursion_shape_config: &'_ RecursionShapeConfig<BabyBear, CompressAir<BabyBear>>,
//         reduce_batch_size: usize,
//     ) -> impl Iterator<Item = Vec<OrderedShape>> + '_ {
//         recursion_shape_config.get_all_shape_combinations(reduce_batch_size)
//     }

//     pub fn generate_maximal_shapes<'a>(
//         core_shape_config: &'a CoreShapeConfig<BabyBear>,
//         recursion_shape_config: &'a RecursionShapeConfig<BabyBear, CompressAir<BabyBear>>,
//         reduce_batch_size: usize,
//         no_precompiles: bool,
//     ) -> impl Iterator<Item = Self> + 'a {
//         let core_shape_iter = if no_precompiles {
//             core_shape_config.maximal_core_shapes(21).into_iter()
//         } else {
//             core_shape_config.maximal_core_plus_precompile_shapes(21).into_iter()
//         };
//         core_shape_iter
//             .map(|core_shape| {
//                 Self::Recursion(OrderedShape {
//                     inner: core_shape.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
//                 })
//             })
//             .chain((1..=reduce_batch_size).flat_map(|batch_size| {
//                 recursion_shape_config.get_all_shape_combinations(batch_size).map(Self::Compress)
//             }))
//             .chain(
//                 recursion_shape_config
//                     .get_all_shape_combinations(1)
//                     .map(|mut x| Self::Deferred(x.pop().unwrap())),
//             )
//             .chain(
//                 recursion_shape_config
//                     .get_all_shape_combinations(1)
//                     .map(|mut x| Self::Shrink(x.pop().unwrap())),
//             )
//     }

//     pub fn dummy_vk_map<'a>(
//         core_shape_config: &'a CoreShapeConfig<BabyBear>,
//         recursion_shape_config: &'a RecursionShapeConfig<BabyBear, CompressAir<BabyBear>>,
//         reduce_batch_size: usize,
//     ) -> BTreeMap<[BabyBear; DIGEST_SIZE], usize> {
//         Self::generate(core_shape_config, recursion_shape_config, reduce_batch_size)
//             .enumerate()
//             .map(|(i, _)| ([BabyBear::from_canonical_usize(i); DIGEST_SIZE], i))
//             .collect()
//     }
// }

// impl SP1CompressProgramShape {
//     pub fn from_proof_shape(shape: SP1ProofShape, height: usize) -> Self {
//         match shape {
//             SP1ProofShape::Recursion(proof_shape) => Self::Recursion(proof_shape.into()),
//             SP1ProofShape::Deferred(proof_shape) => {
//                 Self::Deferred(SP1DeferredShape::new(vec![proof_shape].into(), height))
//             }
//             SP1ProofShape::Compress(proof_shapes) => Self::Compress(SP1CompressWithVkeyShape {
//                 compress_shape: proof_shapes.into(),
//                 merkle_tree_height: height,
//             }),
//             SP1ProofShape::Shrink(proof_shape) => Self::Shrink(SP1CompressWithVkeyShape {
//                 compress_shape: vec![proof_shape].into(),
//                 merkle_tree_height: height,
//             }),
//         }
//     }
// }

// impl<C: SP1ProverComponents> SP1Prover<C> {
//     pub fn program_from_shape(
//         &self,
//         shape: SP1CompressProgramShape,
//         shrink_shape: Option<SP1RecursionShape>,
//     ) -> Arc<RecursionProgram<BabyBear>> {
//         match shape {
//             SP1CompressProgramShape::Recursion(shape) => {
//                 let input = SP1RecursionWitnessValues::dummy(self.core_prover.machine(), &shape);
//                 self.recursion_program(&input)
//             }
//             SP1CompressProgramShape::Deferred(shape) => {
//                 let input = SP1DeferredWitnessValues::dummy(self.compress_prover.machine(),
// &shape);                 self.deferred_program(&input)
//             }
//             SP1CompressProgramShape::Compress(shape) => {
//                 let input =
//                     SP1CompressWithVKeyWitnessValues::dummy(self.compress_prover.machine(),
// &shape);                 self.compress_program(&input)
//             }
//             SP1CompressProgramShape::Shrink(shape) => {
//                 let input =
//                     SP1CompressWithVKeyWitnessValues::dummy(self.compress_prover.machine(),
// &shape);                 self.shrink_program(
//                     shrink_shape.unwrap_or_else(ShrinkAir::<BabyBear>::shrink_shape),
//                     &input,
//                 )
//             }
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     #![allow(clippy::print_stdout)]

//     use super::*;

//     #[test]
//     #[ignore]
//     fn test_generate_all_shapes() {
//         let core_shape_config = CoreShapeConfig::default();
//         let recursion_shape_config = RecursionShapeConfig::default();
//         let reduce_batch_size = 2;
//         let all_shapes =
//             SP1ProofShape::generate(&core_shape_config, &recursion_shape_config,
// reduce_batch_size)                 .collect::<BTreeSet<_>>();

//         println!("Number of compress shapes: {}", all_shapes.len());
//     }
// }

#[cfg(test)]
mod tests {
    use sp1_core_machine::utils::setup_logger;

    use crate::SP1ProverBuilder;

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_max_arity() {
        setup_logger();
        let prover = SP1ProverBuilder::cpu().build().await;
        // arity 3:
        // let shape = [
        //     (CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 154816),
        //     (CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 393408),
        //     (CompressAir::<BabyBear>::BaseAlu(BaseAluChip), 91232),
        //     (CompressAir::<BabyBear>::ExtAlu(ExtAluChip), 148256),
        //     (CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 89824),
        //     (CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 249984),
        //     (CompressAir::<BabyBear>::Select(SelectChip), 604800),
        //     (CompressAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
        // ]
        // .into_iter()
        // .collect();

        let shape = [
            (CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 402016),
            (CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 529280),
            (CompressAir::<BabyBear>::BaseAlu(BaseAluChip), 485824),
            (CompressAir::<BabyBear>::ExtAlu(ExtAluChip), 751232),
            (CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 120064),
            (CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 249984),
            (CompressAir::<BabyBear>::Select(SelectChip), 806976),
            (CompressAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
        ]
        .into_iter()
        .collect();

        let reduce_shape = SP1ReduceShape { shape };

        let arity = reduce_shape.max_arity(prover.recursion()).await;
        tracing::info!("arity: {}", arity);
    }
}

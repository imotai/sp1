use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    fs::File,
    num::NonZero,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
// use tokio::sync::Mutex;

use hashbrown::HashSet;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use slop_air::BaseAir;
use slop_algebra::AbstractField;
use slop_baby_bear::BabyBear;
use slop_jagged::JaggedConfig;
use sp1_core_executor::{ELEMENT_THRESHOLD, MAX_PROGRAM_SIZE};
use sp1_core_machine::{
    bytes::columns::NUM_BYTE_PREPROCESSED_COLS, program::NUM_PROGRAM_PREPROCESSED_COLS,
    range::columns::NUM_RANGE_PREPROCESSED_COLS, riscv::RiscvAir,
};
use sp1_recursion_circuit::{
    dummy::{dummy_shard_proof, dummy_vk},
    machine::{
        SP1CompressWithVKeyWitnessValues, SP1MerkleProofWitnessValues, SP1NormalizeWitnessValues,
        SP1ShapedWitnessValues,
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
    log2_ceil_usize,
    prover::{CoreProofShape, DefaultTraceGenerator, ProverSemaphore, TraceGenerator},
    Chip, ChipDimensions, Machine, MachineShape,
};
use thiserror::Error;
use tokio::task::JoinSet;

use crate::{
    components::SP1ProverComponents,
    core::{CORE_LOG_STACKING_HEIGHT, CORE_MAX_LOG_ROW_COUNT},
    recursion::{
        deferred_program_from_input, dummy_compose_input, dummy_deferred_input,
        shrink_program_from_input, RECURSION_MAX_LOG_ROW_COUNT,
    },
    types::HashableKey,
    CompressAir, CoreSC, InnerSC, SP1Prover, SP1RecursionProver, SP1VerifyingKey, CORE_LOG_BLOWUP,
};

pub const DEFAULT_ARITY: usize = 4;

/// The shape of the "normalize" program, which proves the correct execution for the verifier of a
/// single core shard proof.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SP1NormalizeInputShape {
    pub proof_shapes: Vec<CoreProofShape<BabyBear, RiscvAir<BabyBear>>>,
    pub max_log_row_count: usize,
    pub log_blowup: usize,
    pub log_stacking_height: usize,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
pub enum SP1RecursionProgramShape {
    // The program that verifies a core shard proof.
    Normalize(CoreProofShape<BabyBear, RiscvAir<BabyBear>>),
    // Compose(arity) is the program that verifies a batch of Normalize proofs of size arity.
    Compose(usize),
    // The deferred proof program.
    Deferred,
    // The shrink program that verifies the the root of the recursion tree.
    Shrink,
}

#[derive(Debug, Error)]
pub enum VkBuildError {
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),
}

impl SP1NormalizeInputShape {
    pub fn dummy_input(&self, vk: SP1VerifyingKey) -> SP1NormalizeWitnessValues<CoreSC> {
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
                    &[core_shape.preprocessed_padding_cols, core_shape.main_padding_cols],
                )
            })
            .collect::<Vec<_>>();

        SP1NormalizeWitnessValues {
            vk: vk.vk,
            shard_proofs,
            is_complete: false,
            is_first_shard: false,
            vk_root: [BabyBear::zero(); DIGEST_SIZE],
            reconstruct_deferred_digest: [BabyBear::zero(); 8],
        }
    }
}

pub struct SP1NormalizeCache {
    lru: Arc<Mutex<LruCache<SP1NormalizeInputShape, Arc<RecursionProgram<BabyBear>>>>>,
    total_calls: AtomicUsize,
    hits: AtomicUsize,
}

impl SP1NormalizeCache {
    pub fn new(size: usize) -> Self {
        let size = NonZero::new(size).expect("size must be non-zero");
        let lru = LruCache::new(size);
        let lru = Arc::new(Mutex::new(lru));
        Self { lru, total_calls: AtomicUsize::new(0), hits: AtomicUsize::new(0) }
    }

    pub fn get(&self, shape: &SP1NormalizeInputShape) -> Option<Arc<RecursionProgram<BabyBear>>> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(program) = self.lru.lock().unwrap().get(shape).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(program)
        } else {
            None
        }
    }

    pub fn push(&self, shape: SP1NormalizeInputShape, program: Arc<RecursionProgram<BabyBear>>) {
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
pub struct SP1RecursionProofShape {
    pub shape: RecursionShape<BabyBear>,
}

impl Default for SP1RecursionProofShape {
    fn default() -> Self {
        Self::compress_proof_shape_from_arity(DEFAULT_ARITY).unwrap()
    }
}

impl SP1RecursionProofShape {
    pub fn compress_proof_shape_from_arity(arity: usize) -> Option<Self> {
        let shape = match arity {
            DEFAULT_ARITY => [
                (CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 347_904),
                (CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 442_400),
                (CompressAir::<BabyBear>::BaseAlu(BaseAluChip), 399_968),
                (CompressAir::<BabyBear>::ExtAlu(ExtAluChip), 762_752),
                (CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 101_696),
                (CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 241_500),
                (CompressAir::<BabyBear>::Select(SelectChip), 677_984),
                (CompressAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
            ]
            .into_iter()
            .collect(),
            _ => return None,
        };
        Some(Self { shape })
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

        let preprocessed_padding_cols = ((preprocessed_multiple * (1 << log_stacking_height))
            - chips
                .iter()
                .map(|chip| self.shape.height(chip).unwrap() * chip.preprocessed_width())
                .sum::<usize>())
        .div_ceil(1 << max_log_row_count);

        let main_padding_cols = ((main_multiple * (1 << log_stacking_height))
            - chips
                .iter()
                .map(|chip| self.shape.height(chip).unwrap() * chip.width())
                .sum::<usize>())
        .div_ceil(1 << max_log_row_count);

        let dummy_proof = dummy_shard_proof(
            chips,
            max_log_row_count,
            log_blowup,
            log_stacking_height,
            &[preprocessed_multiple, main_multiple],
            &[preprocessed_padding_cols, main_padding_cols],
        );

        let vks_and_proofs =
            (0..arity).map(|_| (dummy_vk.clone(), dummy_proof.clone())).collect::<Vec<_>>();

        SP1CompressWithVKeyWitnessValues {
            compress_val: SP1ShapedWitnessValues { vks_and_proofs, is_complete: false },
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
            let program = prover.compose_program_from_input(&input);
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

pub async fn build_vk_map<C: SP1ProverComponents + 'static>(
    dummy: bool,
    num_compiler_workers: usize,
    num_setup_workers: usize,
    indices: Option<Vec<usize>>,
    max_arity: usize,
    prover: Arc<SP1Prover<C>>,
) -> (BTreeSet<[BabyBear; DIGEST_SIZE]>, Vec<usize>) {
    if dummy {
        let dummy_set = dummy_vk_map(&prover).into_keys().collect::<BTreeSet<_>>();
        return (dummy_set, vec![]);
    }

    // Setup the channels.
    let (vk_tx, mut vk_rx) =
        tokio::sync::mpsc::unbounded_channel::<(usize, [BabyBear; DIGEST_SIZE])>();
    let (shape_tx, shape_rx) =
        tokio::sync::mpsc::channel::<(usize, SP1RecursionProgramShape)>(num_compiler_workers);
    let (program_tx, program_rx) = tokio::sync::mpsc::channel(num_setup_workers);
    let (panic_tx, mut panic_rx) = tokio::sync::mpsc::unbounded_channel();

    // Setup the mutexes.
    let shape_rx = Arc::new(tokio::sync::Mutex::new(shape_rx));
    let program_rx = Arc::new(tokio::sync::Mutex::new(program_rx));

    // Generate all the possible shape inputs we encounter in recursion. This may span lift,
    // join, deferred, shrink, etc.
    let all_shapes = create_all_input_shapes(prover.core().machine().shape(), max_arity);

    let num_shapes = all_shapes.len();
    let height = log2_ceil_usize(indices.as_ref().map(Vec::len).unwrap_or(num_shapes));

    let indices_set = indices.map(|indices| indices.into_iter().collect::<HashSet<_>>());

    let vk_map_size = indices_set.as_ref().map(|indices| indices.len()).unwrap_or(num_shapes);

    let mut set = JoinSet::new();

    // Initialize compiler workers.
    for _ in 0..num_compiler_workers {
        let program_tx = program_tx.clone();
        let shape_rx = shape_rx.clone();
        let prover = prover.clone();
        let panic_tx = panic_tx.clone();
        set.spawn(async move {
            while let Some((i, shape)) = shape_rx.lock().await.recv().await {
                // eprintln!("shape: {:?}", shape);
                // let is_shrink = matches!(shape, SP1CompressProgramShape::Shrink(_));
                let prover = prover.clone();
                // Spawn on another thread to handle panics.
                let program_thread = tokio::spawn(async move {
                    let prover = prover.clone();

                    let prover = prover.clone();
                    match shape {
                        SP1RecursionProgramShape::Normalize(shape_clone) => {
                            let normalize_shape = SP1NormalizeInputShape {
                                proof_shapes: vec![shape_clone],
                                max_log_row_count: CORE_MAX_LOG_ROW_COUNT,
                                log_blowup: CORE_LOG_BLOWUP,
                                log_stacking_height: CORE_LOG_STACKING_HEIGHT as usize,
                            };
                            let dummy_vk = dummy_vk(
                                vec![
                                    (
                                        "Byte".to_string(),
                                        ChipDimensions {
                                            height: <CoreSC as JaggedConfig>::F::zero(),
                                            num_polynomials: <CoreSC as JaggedConfig>::F::zero(),
                                        },
                                    ),
                                    (
                                        "Program".to_string(),
                                        ChipDimensions {
                                            height: <CoreSC as JaggedConfig>::F::zero(),
                                            num_polynomials: <CoreSC as JaggedConfig>::F::zero(),
                                        },
                                    ),
                                    (
                                        "Range".to_string(),
                                        ChipDimensions {
                                            height: <CoreSC as JaggedConfig>::F::zero(),
                                            num_polynomials: <CoreSC as JaggedConfig>::F::zero(),
                                        },
                                    ),
                                ]
                                .into_iter()
                                .collect(),
                            );
                            let witness =
                                normalize_shape.dummy_input(SP1VerifyingKey { vk: dummy_vk });
                            (prover.recursion().normalize_program(&witness), false)
                        }
                        SP1RecursionProgramShape::Compose(arity) => {
                            let dummy_input = dummy_compose_input(
                                &prover.recursion().prover,
                                &SP1RecursionProofShape::compress_proof_shape_from_arity(max_arity)
                                    .expect("max arity not supported"),
                                arity,
                                height,
                            );
                            (
                                Arc::new(
                                    prover.recursion().compose_program_from_input(&dummy_input),
                                ),
                                false,
                            )
                        }
                        SP1RecursionProgramShape::Deferred => {
                            let dummy_input = dummy_deferred_input(
                                &prover.recursion().prover,
                                &SP1RecursionProofShape::compress_proof_shape_from_arity(max_arity)
                                    .expect("max arity not supported"),
                                height,
                            );
                            (
                                Arc::new(deferred_program_from_input(
                                    &prover.recursion().recursive_compress_verifier,
                                    true,
                                    &dummy_input,
                                )),
                                false,
                            )
                        }
                        SP1RecursionProgramShape::Shrink => {
                            let dummy_input = dummy_compose_input(
                                &prover.recursion().prover,
                                &SP1RecursionProofShape::compress_proof_shape_from_arity(max_arity)
                                    .expect("max arity not supported"),
                                1,
                                height,
                            );
                            let program = shrink_program_from_input(
                                &prover.recursion().recursive_compress_verifier,
                                true,
                                &dummy_input,
                            );

                            (Arc::new(program), true)
                        }
                    }
                });
                match program_thread.await {
                    Ok((program, is_shrink)) => {
                        program_tx.send((i, program, is_shrink)).await.unwrap()
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Program generation failed for shape {}, with error: {:?}",
                            i,
                            e
                        );
                        panic_tx.send(i).unwrap();
                    }
                }
            }
        });
    }

    // Initialize setup workers.
    for _ in 0..num_setup_workers {
        let vk_tx = vk_tx.clone();
        let program_rx = program_rx.clone();
        let prover = prover.clone();
        set.spawn(async move {
            let mut done = 0;
            while let Some((i, program, is_shrink)) = program_rx.lock().await.recv().await {
                let prover = prover.clone();
                let vk_thread = tokio::spawn(async move {
                    if is_shrink {
                        prover.recursion().shrink_prover.setup(program, None).await
                    } else {
                        prover.recursion().prover.setup(program, None).await
                    }
                });
                let vk = vk_thread.await.unwrap();
                done += 1;

                let vk_digest = vk.1.hash_babybear();

                tracing::info!(
                    "program {} = {:?}, {}% done",
                    i,
                    vk_digest,
                    done * 100 / vk_map_size
                );
                vk_tx.send((i, vk_digest)).unwrap();
            }
        });
    }

    // Generate shapes and send them to the compiler workers.
    let subset_shapes = all_shapes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| indices_set.as_ref().map(|set| set.contains(i)).unwrap_or(true))
        .collect::<Vec<_>>();

    for (i, shape) in subset_shapes.iter() {
        shape_tx.send((*i, shape.clone())).await.unwrap();
    }

    drop(shape_tx);
    drop(program_tx);
    drop(vk_tx);
    drop(panic_tx);

    set.join_all().await;

    let mut vk_map = BTreeMap::new();
    while let Some((i, vk)) = vk_rx.recv().await {
        vk_map.insert(i, vk);
    }

    // Build vk_set in the same order as shapes were sent
    let vk_set: BTreeSet<[BabyBear; DIGEST_SIZE]> = vk_map.into_values().collect();

    let mut panic_indices = vec![];
    while let Some(i) = panic_rx.recv().await {
        panic_indices.push(i);
    }
    for (i, shape) in subset_shapes {
        if panic_indices.contains(&i) {
            tracing::info!("panic shape {}: {:?}", i, shape);
        }
    }

    (vk_set, panic_indices)
}

pub async fn build_vk_map_to_file<C: SP1ProverComponents + 'static>(
    build_dir: PathBuf,
    max_arity: usize,
    dummy: bool,
    num_compiler_workers: usize,
    num_setup_workers: usize,
    indices: Option<Vec<usize>>,
    prover: Arc<SP1Prover<C>>,
) -> Result<(), VkBuildError> {
    // Create the build directory if it doesn't exist.
    std::fs::create_dir_all(&build_dir)?;

    // Build the vk map.
    let (vk_set, _) = build_vk_map::<C>(
        dummy,
        num_compiler_workers,
        num_setup_workers,
        indices,
        max_arity,
        prover.clone(),
    )
    .await;

    let vk_map = vk_set.into_iter().enumerate().map(|(i, vk)| (vk, i)).collect::<BTreeMap<_, _>>();

    // Create the file to store the vk map.
    let mut file = if dummy {
        File::create(build_dir.join("dummy_vk_map.bin"))?
    } else {
        File::create(build_dir.join("vk_map.bin"))?
    };

    Ok(bincode::serialize_into(&mut file, &vk_map)?)
}

fn max_main_multiple_for_preprocessed_multiple(preprocessed_multiple: usize) -> usize {
    (ELEMENT_THRESHOLD - preprocessed_multiple as u64 * (1 << CORE_LOG_STACKING_HEIGHT))
        .div_ceil(1 << CORE_LOG_STACKING_HEIGHT as u64) as usize
}

fn create_all_input_shapes(
    core_shape: &MachineShape<BabyBear, RiscvAir<BabyBear>>,
    max_arity: usize,
) -> Vec<SP1RecursionProgramShape> {
    let (max_preprocessed_multiple, _, capacity) = normalize_program_parameter_space();
    let max_num_padding_cols =
        ((1 << CORE_LOG_STACKING_HEIGHT) as usize).div_ceil(1 << CORE_MAX_LOG_ROW_COUNT);

    let mut result: Vec<SP1RecursionProgramShape> = Vec::with_capacity(capacity);
    for preprocessed_multiple in 1..=max_preprocessed_multiple {
        for main_multiple in 1..=max_main_multiple_for_preprocessed_multiple(preprocessed_multiple)
        {
            for main_padding_cols in 1..=max_num_padding_cols {
                for preprocessed_padding_cols in 1..=max_num_padding_cols {
                    for cluster in &core_shape.chip_clusters {
                        result.push(SP1RecursionProgramShape::Normalize(CoreProofShape {
                            shard_chips: cluster.clone(),
                            preprocessed_multiple,
                            main_multiple,
                            preprocessed_padding_cols,
                            main_padding_cols,
                        }));
                    }
                }
            }
        }
    }

    // Add the compose shapes for each arity.
    for arity in 1..=max_arity {
        result.push(SP1RecursionProgramShape::Compose(arity));
    }

    // Add the deferred shape.
    result.push(SP1RecursionProgramShape::Deferred);
    // Add the shrink shape.
    result.push(SP1RecursionProgramShape::Shrink);
    result
}

pub fn normalize_program_parameter_space() -> (usize, usize, usize) {
    let max_preprocessed_multiple = (MAX_PROGRAM_SIZE * NUM_PROGRAM_PREPROCESSED_COLS
        + (1 << 17) * NUM_RANGE_PREPROCESSED_COLS
        + (1 << 16) * NUM_BYTE_PREPROCESSED_COLS)
        .div_ceil(1 << CORE_LOG_STACKING_HEIGHT);
    let max_main_multiple = (ELEMENT_THRESHOLD).div_ceil(1 << CORE_LOG_STACKING_HEIGHT) as usize;

    let num_shapes = (0..=max_preprocessed_multiple)
        .map(max_main_multiple_for_preprocessed_multiple)
        .sum::<usize>();

    (max_preprocessed_multiple, max_main_multiple, num_shapes)
}

pub fn dummy_vk_map<C: SP1ProverComponents>(
    prover: &SP1Prover<C>,
) -> BTreeMap<[BabyBear; DIGEST_SIZE], usize> {
    create_all_input_shapes(prover.core().machine().shape(), DEFAULT_ARITY)
        .iter()
        .enumerate()
        .map(|(i, _)| ([BabyBear::from_canonical_usize(i); DIGEST_SIZE], i))
        .collect()
}

#[cfg(all(test, feature = "unsound"))]
mod tests {

    use crate::{
        core::{CORE_LOG_STACKING_HEIGHT, CORE_MAX_LOG_ROW_COUNT},
        local::LocalProver,
        recursion::normalize_program_from_input,
        CORE_LOG_BLOWUP,
    };
    use sp1_core_executor::{SP1Context, ELEMENT_THRESHOLD, MAX_PROGRAM_SIZE};
    use sp1_core_machine::{
        bytes::columns::NUM_BYTE_PREPROCESSED_COLS, io::SP1Stdin,
        program::NUM_PROGRAM_PREPROCESSED_COLS, range::columns::NUM_RANGE_PREPROCESSED_COLS,
        utils::setup_logger,
    };

    use sp1_recursion_executor::RecursionAirEventCount;
    use sp1_recursion_machine::chips::poseidon2_helper::{
        convert::ConvertChip, linear::Poseidon2LinearLayerChip, sbox::Poseidon2SBoxChip,
    };

    use crate::SP1ProverBuilder;

    use super::*;

    #[tokio::test]
    #[ignore = "should be invoked specifically"]
    async fn test_max_arity() {
        setup_logger();
        let prover = SP1ProverBuilder::new().build().await;

        let shape = [
            (CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()), 347904),
            (CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::default()), 442400),
            (CompressAir::<BabyBear>::BaseAlu(BaseAluChip), 399968),
            (CompressAir::<BabyBear>::ExtAlu(ExtAluChip), 762732),
            (CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip), 101664),
            (CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip), 241500),
            (CompressAir::<BabyBear>::Select(SelectChip), 677984),
            (CompressAir::<BabyBear>::PublicValues(PublicValuesChip), 16),
        ]
        .into_iter()
        .collect();

        let reduce_shape = SP1RecursionProofShape { shape };

        let arity = reduce_shape.max_arity(prover.recursion()).await;
        tracing::info!("arity: {}", arity);
    }

    fn max_count(a: RecursionAirEventCount, b: RecursionAirEventCount) -> RecursionAirEventCount {
        use std::cmp::max;
        RecursionAirEventCount {
            mem_const_events: max(a.mem_const_events, b.mem_const_events),
            mem_var_events: max(a.mem_var_events, b.mem_var_events),
            base_alu_events: max(a.base_alu_events, b.base_alu_events),
            ext_alu_events: max(a.ext_alu_events, b.ext_alu_events),
            ext_felt_conversion_events: max(
                a.ext_felt_conversion_events,
                b.ext_felt_conversion_events,
            ),
            poseidon2_wide_events: max(a.poseidon2_wide_events, b.poseidon2_wide_events),
            poseidon2_linear_layer_events: max(
                a.poseidon2_linear_layer_events,
                b.poseidon2_linear_layer_events,
            ),
            poseidon2_sbox_events: max(a.poseidon2_sbox_events, b.poseidon2_sbox_events),
            select_events: max(a.select_events, b.select_events),
            prefix_sum_checks_events: max(a.prefix_sum_checks_events, b.prefix_sum_checks_events),
            commit_pv_hash_events: max(a.commit_pv_hash_events, b.commit_pv_hash_events),
        }
    }

    fn create_test_shape(
        cluster: &BTreeSet<Chip<BabyBear, RiscvAir<BabyBear>>>,
    ) -> SP1NormalizeInputShape {
        let preprocessed_multiple = (MAX_PROGRAM_SIZE * NUM_PROGRAM_PREPROCESSED_COLS
            + (1 << 17) * NUM_RANGE_PREPROCESSED_COLS
            + (1 << 16) * NUM_BYTE_PREPROCESSED_COLS)
            .div_ceil(1 << CORE_LOG_STACKING_HEIGHT);
        let main_multiple = (ELEMENT_THRESHOLD).div_ceil(1 << CORE_LOG_STACKING_HEIGHT) as usize;
        let num_padding_cols =
            ((1 << CORE_LOG_STACKING_HEIGHT) as usize).div_ceil(1 << CORE_MAX_LOG_ROW_COUNT);
        SP1NormalizeInputShape {
            proof_shapes: vec![CoreProofShape {
                shard_chips: cluster.clone(),
                preprocessed_multiple,
                main_multiple,
                preprocessed_padding_cols: num_padding_cols,
                main_padding_cols: num_padding_cols,
            }],
            max_log_row_count: CORE_MAX_LOG_ROW_COUNT,
            log_stacking_height: CORE_LOG_STACKING_HEIGHT as usize,
            log_blowup: CORE_LOG_BLOWUP,
        }
    }

    fn build_recursion_count_from_shape(
        shape: &RecursionShape<BabyBear>,
    ) -> RecursionAirEventCount {
        RecursionAirEventCount {
            mem_const_events: shape
                .height(&CompressAir::<BabyBear>::MemoryConst(MemoryConstChip::default()))
                .unwrap(),
            mem_var_events: shape
                .height(
                    &CompressAir::<BabyBear>::MemoryVar(MemoryVarChip::<BabyBear, 2>::default()),
                )
                .unwrap()
                * 2,
            base_alu_events: shape.height(&CompressAir::<BabyBear>::BaseAlu(BaseAluChip)).unwrap(),
            ext_alu_events: shape.height(&CompressAir::<BabyBear>::ExtAlu(ExtAluChip)).unwrap(),
            ext_felt_conversion_events: shape
                .height(&CompressAir::<BabyBear>::ExtFeltConvert(ConvertChip))
                .unwrap_or(0),
            poseidon2_wide_events: shape
                .height(&CompressAir::<BabyBear>::Poseidon2Wide(Poseidon2WideChip))
                .unwrap_or(0),
            poseidon2_linear_layer_events: shape
                .height(&CompressAir::<BabyBear>::Poseidon2LinearLayer(Poseidon2LinearLayerChip))
                .unwrap_or(0),
            poseidon2_sbox_events: shape
                .height(&CompressAir::<BabyBear>::Poseidon2SBox(Poseidon2SBoxChip))
                .unwrap_or(0),
            select_events: shape.height(&CompressAir::<BabyBear>::Select(SelectChip)).unwrap(),
            prefix_sum_checks_events: shape
                .height(&CompressAir::<BabyBear>::PrefixSumChecks(PrefixSumChecksChip))
                .unwrap_or(0),
            commit_pv_hash_events: shape
                .height(&CompressAir::<BabyBear>::PublicValues(PublicValuesChip))
                .unwrap(),
        }
    }

    #[tokio::test]
    async fn test_core_shape_fit() {
        setup_logger();
        let elf = test_artifacts::FIBONACCI_ELF;
        let prover = SP1ProverBuilder::new().build().await;
        let (_, _, vk) = prover.core().setup(&elf).await;

        let machine = RiscvAir::<BabyBear>::machine();
        let chip_clusters = &machine.shape().chip_clusters;
        let mut max_cluster_count = RecursionAirEventCount::default();

        for cluster in chip_clusters {
            let shape = create_test_shape(cluster);
            let program = normalize_program_from_input(
                &prover.recursion().recursive_core_verifier,
                &shape.dummy_input(vk.clone()),
            );
            max_cluster_count = max_count(max_cluster_count, program.event_counts);
        }

        tracing::info!("max_cluster_count: {:?}", max_cluster_count);

        let reduce_shape =
            SP1RecursionProofShape::compress_proof_shape_from_arity(DEFAULT_ARITY).unwrap();
        let arity = reduce_shape.max_arity(prover.recursion()).await;
        assert!(arity >= DEFAULT_ARITY);

        let arity_4_count = build_recursion_count_from_shape(&reduce_shape.shape);
        let combined_count = max_count(max_cluster_count, arity_4_count);

        assert_eq!(combined_count, arity_4_count);
    }

    #[tokio::test]
    async fn test_build_vk_map() {
        setup_logger();
        let prover = SP1ProverBuilder::new().build().await;

        let elf = test_artifacts::FIBONACCI_ELF;
        let (pk, program, vk) = prover.core().setup(&elf).await;

        let local_prover = Arc::new(LocalProver::new(prover, Default::default()));

        let pk = unsafe { pk.into_inner() };

        // Make a proof to get proof shapes to populate the vk map with.
        let proof = local_prover
            .clone()
            .prove_core(pk, program, SP1Stdin::default(), SP1Context::default())
            .await
            .expect("Failed to prove");

        // Create all circuit shapes.
        let shapes =
            create_all_input_shapes(local_prover.prover().core().machine().shape(), DEFAULT_ARITY);

        // Determine the indices in `shapes` of the shapes appear in the proof.
        let mut shape_indices = vec![];

        for proof in &proof.proof.0 {
            let shape = SP1RecursionProgramShape::Normalize(
                local_prover.prover().core().verifier().shape_from_proof(proof),
            );

            shape_indices.push(shapes.iter().position(|s| s == &shape).unwrap());
        }

        // Build the vk map that includes all of the proof shapes in the proof.
        let prover = Arc::new(SP1ProverBuilder::new().build().await);

        let shape_indices =
            shape_indices.into_iter().chain(shapes.len() - 12..shapes.len()).collect::<Vec<_>>();

        let shape_indices_len = shape_indices.len();

        build_vk_map_to_file(
            "../../../".into(),
            DEFAULT_ARITY,
            false,
            1,
            1,
            Some(shape_indices),
            prover.clone(),
        )
        .await
        .unwrap();

        tracing::info!("Built vk map with {} shapes", shape_indices_len);

        // Build a new prover that performs the vk verification check using the built vk map.
        let prover =
            SP1ProverBuilder::new().with_vk_map_path("../../../vk_map.bin".into()).build().await;

        tracing::info!("Rebuilt prover with vk map.");

        let local_prover = Arc::new(LocalProver::new(prover, Default::default()));

        local_prover.prover().verify(&proof.proof, &vk).expect("Failed to verify proof");

        tracing::info!("Core proof verified successfully.");

        let compress_proof = local_prover.clone().compress(&vk, proof, vec![]).await.unwrap();

        local_prover
            .prover()
            .verify_compressed(&compress_proof, &vk)
            .expect("Failed to verify compress proof");

        tracing::info!("Compress proof verified successfully.");

        let shrink_proof = local_prover.clone().shrink(compress_proof).await.unwrap();

        local_prover
            .prover()
            .verify_shrink(&shrink_proof, &vk)
            .expect("Failed to verify shrink proof");

        std::fs::remove_file("../../../vk_map.bin").unwrap();
    }
}

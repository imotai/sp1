use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    future::Future,
    iter::once,
    sync::Arc,
};

use derive_where::derive_where;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_air::Air;
use slop_algebra::{AbstractField, ExtensionField, Field};
use slop_alloc::{Backend, Buffer, CanCopyFrom, CanCopyFromRef, CpuBackend};
use slop_challenger::{CanObserve, FieldChallenger};
use slop_commit::Rounds;
use slop_jagged::{JaggedBackend, JaggedProver, JaggedProverComponents, JaggedProverData};
use slop_matrix::dense::RowMajorMatrixView;
use slop_multilinear::{
    Evaluations, HostEvaluationBackend, MleEval, Point, PointBackend, VirtualGeq,
};
use slop_sumcheck::{reduce_sumcheck_to_evaluation, PartialSumcheckProof};
use slop_tensor::Tensor;
use tracing::Instrument;

use crate::{
    air::{MachineAir, MachineProgram},
    prover::{ProverPermit, ProverSemaphore, ZeroCheckPoly, ZerocheckAir},
    septic_digest::SepticDigest,
    AirOpenedValues, Chip, ChipDimensions, ChipEvaluation, ChipOpenedValues, ChipStatistics,
    ConstraintSumcheckFolder, LogUpEvaluations, LogUpGkrProver, Machine, MachineConfig,
    MachineRecord, MachineVerifyingKey, ShardOpenedValues, ShardProof,
};

use super::{TraceGenerator, Traces, ZercocheckBackend, ZerocheckProverData};

/// A prover for an AIR.
pub trait AirProver<C: MachineConfig, Air: MachineAir<C::F>>:
    'static + Send + Sync + Sized
{
    /// The proving key type.
    type PreprocessedData: 'static + Send + Sync;

    /// Get the machine.
    fn machine(&self) -> &Machine<C::F, Air>;

    /// Setup from a verifying key.
    fn setup_from_vk(
        &self,
        program: Arc<Air::Program>,
        vk: Option<MachineVerifyingKey<C>>,
        prover_permits: ProverSemaphore,
    ) -> impl Future<Output = (PreprocessedData<ProvingKey<C, Air, Self>>, MachineVerifyingKey<C>)> + Send;

    /// Setup and prove a shard.
    fn setup_and_prove_shard(
        &self,
        program: Arc<Air::Program>,
        record: Air::Record,
        vk: Option<MachineVerifyingKey<C>>,
        prover_permits: ProverSemaphore,
        challenger: &mut C::Challenger,
    ) -> impl Future<Output = (MachineVerifyingKey<C>, ShardProof<C>, ProverPermit)> + Send;

    /// Prove a shard with a given proving key.
    fn prove_shard_with_pk(
        &self,
        pk: Arc<ProvingKey<C, Air, Self>>,
        record: Air::Record,
        prover_permits: ProverSemaphore,
        challenger: &mut C::Challenger,
    ) -> impl Future<Output = (ShardProof<C>, ProverPermit)> + Send;

    /// Get all the chips in the machine.
    fn all_chips(&self) -> &[Chip<C::F, Air>] {
        self.machine().chips()
    }

    /// Setup from a program.
    ///
    /// The setup phase produces a pair '(pk, vk)' of proving and verifying keys. The proving key
    /// consists of information used by the prover that only depends on the program itself and not
    /// a specific execution.
    fn setup(
        &self,
        program: Arc<Air::Program>,
        setup_permits: ProverSemaphore,
    ) -> impl Future<Output = (PreprocessedData<ProvingKey<C, Air, Self>>, MachineVerifyingKey<C>)> + Send
    {
        self.setup_from_vk(program, None, setup_permits)
    }
}

/// A proving key for an AIR prover.
pub struct ProvingKey<C: MachineConfig, Air: MachineAir<C::F>, Prover: AirProver<C, Air>> {
    /// The verifying key.
    pub vk: MachineVerifyingKey<C>,
    /// The preprocessed data.
    pub preprocessed_data: Prover::PreprocessedData,
}

/// The components of the machine prover.
///
/// This trait is used specify a configuration of a hypercube prover.
pub trait ShardProverComponents: 'static + Send + Sync + Sized {
    /// The base field.
    ///
    /// This is the field on which the traces committed to are defined over.
    type F: Field;
    /// The field of random elements.
    ///
    /// This is an extension field of the base field which is of cryptographically secure size. The
    /// random evaluation points of the protocol are drawn from `EF`.
    type EF: ExtensionField<Self::F>;
    /// The program type.
    type Program: MachineProgram<Self::F> + Send + Sync + 'static;
    /// The record type.
    type Record: MachineRecord;
    /// The Air for which this prover.
    type Air: ZerocheckAir<Self::F, Self::EF, Program = Self::Program, Record = Self::Record>;
    /// The backend used by the prover.
    type B: JaggedBackend<Self::F, Self::EF>
        + ZercocheckBackend<Self::F, Self::EF, Self::ZerocheckProverData>
        + PointBackend<Self::EF>
        + HostEvaluationBackend<Self::F, Self::EF>
        + HostEvaluationBackend<Self::F, Self::F>
        + HostEvaluationBackend<Self::EF, Self::EF>
        + CanCopyFrom<Buffer<Self::EF>, CpuBackend, Output = Buffer<Self::EF, Self::B>>;

    /// The commitment representing a batch of traces sent to the verifier.
    type Commitment: 'static + Clone + Send + Sync + Serialize + DeserializeOwned;

    /// The challenger type that creates the random challenges via Fiat-Shamir.
    ///
    /// The challenger is observing all the messages sent throughout the protocol and uses this
    /// to create the verifier messages of the IOP.
    type Challenger: FieldChallenger<Self::F>
        + CanObserve<Self::Commitment>
        + Send
        + Sync
        + 'static
        + Clone;

    /// The machine configuration for which this prover can make proofs for.
    type Config: MachineConfig<
        F = Self::F,
        EF = Self::EF,
        Commitment = Self::Commitment,
        Challenger = Self::Challenger,
    >;

    /// The trace generator.
    type TraceGenerator: TraceGenerator<Self::F, Self::Air, Self::B>;

    /// The zerocheck prover data.
    ///
    /// The zerocheck prover data contains the information needed to make a zerocheck prover given
    /// an AIR. The zerocheck prover implements the zerocheck IOP and reduces the claim that
    /// constraints vanish into an evaluation claim at a random point for the traces, considered
    /// as multilinear polynomials.
    type ZerocheckProverData: ZerocheckProverData<Self::F, Self::EF, Self::B, Air = Self::Air>;

    /// The necessary pieces to form a GKR proof for the `LogUp` permutation argument.
    type GkrProver: LogUpGkrProver<
        F = Self::F,
        EF = Self::EF,
        A = Self::Air,
        B = Self::B,
        Challenger = Self::Challenger,
    >;

    /// The components of the jagged PCS prover.
    type PcsProverComponents: JaggedProverComponents<
            F = Self::F,
            EF = Self::EF,
            A = Self::B,
            Commitment = Self::Commitment,
            Challenger = Self::Challenger,
            Config = Self::Config,
        > + Send
        + Sync
        + 'static;
}

/// A collection of main traces with a permit.
pub struct ShardData<C: ShardProverComponents> {
    /// The proving key.
    pub pk: Arc<ProvingKey<C::Config, C::Air, ShardProver<C>>>,
    /// Main trace data
    pub main_trace_data: MainTraceData<C::F, C::Air, C::B>,
}

/// The main traces for a program, with a permit.
pub struct MainTraceData<F: Field, A: MachineAir<F>, B: Backend> {
    /// The traces.
    pub traces: Traces<F, B>,
    /// The public values.
    pub public_values: Vec<F>,
    /// The shape cluster corresponding to the traces.
    pub shard_chips: BTreeSet<Chip<F, A>>,
    /// A permit for a prover resource.
    pub permit: ProverPermit,
}

/// The total trace data for a shard.
pub struct TraceData<F: Field, A: MachineAir<F>, B: Backend> {
    /// The preprocessed traces.
    pub preprocessed_traces: Traces<F, B>,
    /// The main traces.
    pub main_trace_data: MainTraceData<F, A, B>,
}

/// The preprocessed traces for a program.
pub struct PreprocessedTraceData<F: Field, B: Backend> {
    /// The preprocessed traces.
    pub preprocessed_traces: Traces<F, B>,
    /// A permit for a prover resource.
    pub permit: ProverPermit,
}

/// The preprocessed data for a program.
pub struct PreprocessedData<T> {
    /// The proving key.
    pk: Arc<T>,
    /// A permit for a prover resource.
    pub permit: ProverPermit,
}

impl<T> PreprocessedData<T> {
    /// Unsafely take the inner proving key.
    ///
    /// # Safety
    /// This is unsafe because the permit is dropped.
    #[must_use]
    #[inline]
    pub unsafe fn into_inner(self) -> Arc<T> {
        self.pk
    }
}

/// A prover for the hypercube STARK, given a configuration.
pub struct ShardProver<C: ShardProverComponents> {
    /// The trace generator.
    pub trace_generator: C::TraceGenerator,
    /// The logup GKR prover.
    pub logup_gkr_prover: C::GkrProver,
    /// A prover for the zerocheck IOP.
    pub zerocheck_prover_data: C::ZerocheckProverData,
    /// A prover for the PCS.
    pub pcs_prover: JaggedProver<C::PcsProverComponents>,
}

impl<C: ShardProverComponents> AirProver<C::Config, C::Air> for ShardProver<C> {
    type PreprocessedData = ShardProverData<C>;

    fn machine(&self) -> &Machine<C::F, C::Air> {
        self.trace_generator.machine()
    }

    /// Setup a shard, using a verifying key if provided.
    async fn setup_from_vk(
        &self,
        program: Arc<C::Program>,
        vk: Option<MachineVerifyingKey<C::Config>>,
        prover_permits: ProverSemaphore,
    ) -> (PreprocessedData<ProvingKey<C::Config, C::Air, Self>>, MachineVerifyingKey<C::Config>)
    {
        if let Some(vk) = vk {
            let initial_global_cumulative_sum = vk.initial_global_cumulative_sum;
            self.setup_with_initial_global_cumulative_sum(
                program,
                initial_global_cumulative_sum,
                prover_permits,
            )
            .await
        } else {
            let program_sent = program.clone();
            let initial_global_cumulative_sum =
                tokio::task::spawn_blocking(move || program_sent.initial_global_cumulative_sum())
                    .await
                    .unwrap();
            self.setup_with_initial_global_cumulative_sum(
                program,
                initial_global_cumulative_sum,
                prover_permits,
            )
            .await
        }
    }

    /// Setup and prove a shard.
    async fn setup_and_prove_shard(
        &self,
        program: Arc<C::Program>,
        record: C::Record,
        vk: Option<MachineVerifyingKey<C::Config>>,
        prover_permits: ProverSemaphore,
        challenger: &mut C::Challenger,
    ) -> (MachineVerifyingKey<C::Config>, ShardProof<C::Config>, ProverPermit) {
        // Get the initial global cumulative sum and pc start.
        let pc_start_rel = program.pc_start_rel();
        let initial_global_cumulative_sum = if let Some(vk) = vk {
            vk.initial_global_cumulative_sum
        } else {
            let program = program.clone();
            tokio::task::spawn_blocking(move || program.initial_global_cumulative_sum())
                .instrument(tracing::debug_span!("initial_global_cumulative_sum"))
                .await
                .unwrap()
        };

        // Generate trace.
        let trace_data = self
            .trace_generator
            .generate_traces(program, record, self.max_log_row_count(), prover_permits)
            .instrument(tracing::debug_span!("generate main traces"))
            .await;

        let TraceData { preprocessed_traces, main_trace_data } = trace_data;

        let (pk, vk) = self
            .setup_from_preprocessed_data_and_traces(
                pc_start_rel,
                initial_global_cumulative_sum,
                preprocessed_traces,
            )
            .instrument(tracing::debug_span!("setup_from_preprocessed_data_and_traces"))
            .await;

        let pk = ProvingKey { vk: vk.clone(), preprocessed_data: pk };

        let pk = Arc::new(pk);

        // Observe the preprocessed information.
        vk.observe_into(challenger);

        let shard_data = ShardData { pk, main_trace_data };

        let (shard_proof, permit) = self
            .prove_shard_with_data(shard_data, challenger)
            .instrument(tracing::debug_span!("prove shard with data"))
            .await;

        (vk, shard_proof, permit)
    }

    /// Prove a shard with a given proving key.
    async fn prove_shard_with_pk(
        &self,
        pk: Arc<ProvingKey<C::Config, C::Air, Self>>,
        record: C::Record,
        prover_permits: ProverSemaphore,
        challenger: &mut C::Challenger,
    ) -> (ShardProof<C::Config>, ProverPermit) {
        // Generate the traces.
        let main_trace_data = self
            .trace_generator
            .generate_main_traces(record, self.max_log_row_count(), prover_permits)
            .instrument(tracing::debug_span!("generate main traces"))
            .await;

        let shard_data = ShardData { pk, main_trace_data };

        self.prove_shard_with_data(shard_data, challenger)
            .instrument(tracing::debug_span!("prove shard with data"))
            .await
    }
}

impl<C: ShardProverComponents> ShardProver<C> {
    /// Get all the chips in the machine.
    pub fn all_chips(&self) -> &[Chip<C::F, C::Air>] {
        self.trace_generator.machine().chips()
    }

    /// Get the machine.
    pub fn machine(&self) -> &Machine<C::F, C::Air> {
        self.trace_generator.machine()
    }

    /// Get the number of public values in the machine.
    pub fn num_pv_elts(&self) -> usize {
        self.trace_generator.machine().num_pv_elts()
    }

    /// Get the maximum log row count.
    #[inline]
    pub const fn max_log_row_count(&self) -> usize {
        self.pcs_prover.max_log_row_count
    }

    /// Setup from preprocessed data and traces.
    pub async fn setup_from_preprocessed_data_and_traces(
        &self,
        pc_start_rel: C::F,
        initial_global_cumulative_sum: SepticDigest<C::F>,
        preprocessed_traces: Traces<C::F, C::B>,
    ) -> (ShardProverData<C>, MachineVerifyingKey<C::Config>) {
        // Commit to the preprocessed traces, if there are any.
        let (preprocessed_commit, preprocessed_data) = if !preprocessed_traces.is_empty() {
            let message = preprocessed_traces.values().cloned().collect::<Vec<_>>();
            let (commit, data) = self.pcs_prover.commit_multilinears(message).await.unwrap();

            (Some(commit), Some(data))
        } else {
            (None, None)
        };

        let preprocessed_chip_information = preprocessed_traces
            .iter()
            .map(|(name, trace)| {
                (
                    name.to_owned(),
                    ChipDimensions {
                        height: trace.num_real_entries(),
                        num_polynomials: trace.num_polynomials(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        let vk = MachineVerifyingKey {
            pc_start_rel,
            initial_global_cumulative_sum,
            preprocessed_commit,
            preprocessed_chip_information,
        };

        let pk = ShardProverData { preprocessed_traces, preprocessed_data };

        (pk, vk)
    }

    /// Setup from a program with a specific initial global cumulative sum.
    pub async fn setup_with_initial_global_cumulative_sum(
        &self,
        program: Arc<C::Program>,
        initial_global_cumulative_sum: SepticDigest<C::F>,
        setup_permits: ProverSemaphore,
    ) -> (PreprocessedData<ProvingKey<C::Config, C::Air, Self>>, MachineVerifyingKey<C::Config>)
    {
        let pc_start_rel = program.pc_start_rel();
        let preprocessed_data = self
            .trace_generator
            .generate_preprocessed_traces(program, self.max_log_row_count(), setup_permits)
            .await;

        let PreprocessedTraceData { preprocessed_traces, permit } = preprocessed_data;

        let (pk, vk) = self
            .setup_from_preprocessed_data_and_traces(
                pc_start_rel,
                initial_global_cumulative_sum,
                preprocessed_traces,
            )
            .await;

        let pk = ProvingKey { vk: vk.clone(), preprocessed_data: pk };

        let pk = Arc::new(pk);

        (PreprocessedData { pk, permit }, vk)
    }

    async fn commit_traces(
        &self,
        traces: &Traces<C::F, C::B>,
    ) -> (C::Commitment, JaggedProverData<C::PcsProverComponents>) {
        let message = traces.values().cloned().collect::<Vec<_>>();
        self.pcs_prover.commit_multilinears(message).await.unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    async fn zerocheck(
        &self,
        chips: &BTreeSet<Chip<C::F, C::Air>>,
        preprocessed_traces: Traces<C::F, C::B>,
        traces: Traces<C::F, C::B>,
        batching_challenge: C::EF,
        gkr_opening_batch_randomness: C::EF,
        logup_evaluations: &LogUpEvaluations<C::EF>,
        public_values: Vec<C::F>,
        challenger: &mut C::Challenger,
    ) -> (ShardOpenedValues<C::F, C::EF>, PartialSumcheckProof<C::EF>) {
        let max_num_constraints =
            itertools::max(chips.iter().map(|chip| chip.num_constraints)).unwrap();
        let powers_of_challenge =
            batching_challenge.powers().take(max_num_constraints).collect::<Vec<_>>();
        let airs =
            chips.iter().map(|chip| (chip.air.clone(), chip.num_constraints)).collect::<Vec<_>>();

        let public_values = Arc::new(public_values);

        let mut zerocheck_polys = Vec::new();
        let mut chip_sumcheck_claims = Vec::new();

        let LogUpEvaluations { point: gkr_point, chip_openings } = logup_evaluations;

        let mut chip_heights = BTreeMap::new();
        for ((air, num_constraints), chip) in airs.iter().cloned().zip_eq(chips.iter()) {
            let ChipEvaluation {
                main_trace_evaluations: main_opening,
                preprocessed_trace_evaluations: prep_opening,
            } = chip_openings.get(&chip.name()).unwrap();

            let main_trace = traces.get(&air.name()).unwrap().clone();
            let num_real_entries = main_trace.num_real_entries();

            let threshold_point =
                Point::from_usize(num_real_entries, self.pcs_prover.max_log_row_count + 1);
            chip_heights.insert(air.name(), threshold_point);
            let name = air.name();
            let num_variables = main_trace.num_variables();
            assert_eq!(num_variables, self.pcs_prover.max_log_row_count as u32);

            let preprocessed_width = air.preprocessed_width();
            let dummy_preprocessed_trace = vec![C::F::zero(); preprocessed_width];
            let dummy_main_trace = vec![C::F::zero(); main_trace.num_polynomials()];

            // Calculate powers of alpha for constraint evaluation:
            // 1. Generate sequence [α⁰, α¹, ..., α^(n-1)] where n = num_constraints.
            // 2. Reverse to [α^(n-1), ..., α¹, α⁰] to align with Horner's method in the verifier.
            let mut chip_powers_of_alpha = powers_of_challenge[0..num_constraints].to_vec();
            chip_powers_of_alpha.reverse();

            let mut folder = ConstraintSumcheckFolder {
                preprocessed: RowMajorMatrixView::new_row(&dummy_preprocessed_trace),
                main: RowMajorMatrixView::new_row(&dummy_main_trace),
                accumulator: C::EF::zero(),
                public_values: &public_values,
                constraint_index: 0,
                powers_of_alpha: &chip_powers_of_alpha,
            };

            air.eval(&mut folder);
            let padded_row_adjustment = folder.accumulator;

            // TODO: This could be computed once for the maximally wide chip and stored for later
            // use, but since it's a computation that's done once per chip, we have chosen not to
            // perform this optimization for now.
            let gkr_opening_batch_randomness_powers = gkr_opening_batch_randomness
                .powers()
                .skip(1)
                .take(
                    main_opening.num_polynomials()
                        + prep_opening.as_ref().map_or(0, MleEval::num_polynomials),
                )
                .collect::<Vec<_>>();
            let gkr_powers = Arc::new(gkr_opening_batch_randomness_powers);

            let alpha_powers = Arc::new(chip_powers_of_alpha);
            let air_data = self
                .zerocheck_prover_data
                .round_prover(air, public_values.clone(), alpha_powers, gkr_powers.clone())
                .await;
            let preprocessed_trace = preprocessed_traces.get(&name).cloned();

            let chip_sumcheck_claim = main_opening
                .evaluations()
                .as_slice()
                .iter()
                .chain(
                    prep_opening
                        .as_ref()
                        .map_or_else(Vec::new, |mle| mle.evaluations().as_slice().to_vec())
                        .iter(),
                )
                .zip(gkr_powers.iter())
                .map(|(opening, power)| *opening * *power)
                .sum::<C::EF>();

            let initial_geq_value =
                if main_trace.num_real_entries() > 0 { C::EF::zero() } else { C::EF::one() };

            let virtual_geq = VirtualGeq::new(
                main_trace.num_real_entries() as u32,
                C::F::one(),
                C::F::zero(),
                self.pcs_prover.max_log_row_count as u32,
            );

            let zerocheck_poly = ZeroCheckPoly::new(
                air_data,
                gkr_point.clone(),
                preprocessed_trace,
                main_trace,
                C::EF::one(),
                initial_geq_value,
                padded_row_adjustment,
                virtual_geq,
            );
            zerocheck_polys.push(zerocheck_poly);
            chip_sumcheck_claims.push(chip_sumcheck_claim);
        }

        // Same lambda for the RLC of the zerocheck polynomials.
        let lambda = challenger.sample_ext_element::<C::EF>();

        // Compute the sumcheck proof for the zerocheck polynomials.
        let (partial_sumcheck_proof, component_poly_evals) = reduce_sumcheck_to_evaluation(
            zerocheck_polys,
            challenger,
            chip_sumcheck_claims,
            1,
            lambda,
        )
        .await;

        let mut point_extended = partial_sumcheck_proof.point_and_eval.0.clone();
        point_extended.add_dimension(C::EF::zero());

        // Compute the chip openings from the component poly evaluations.

        debug_assert_eq!(component_poly_evals.len(), airs.len());
        let shard_open_values = airs
            .into_iter()
            .zip_eq(component_poly_evals)
            .map(|((air, _), evals)| {
                let (preprocessed_evals, main_evals) = evals.split_at(air.preprocessed_width());

                // Observe the openings
                for eval in preprocessed_evals.iter() {
                    challenger.observe_ext_element(*eval);
                }
                for eval in main_evals.iter() {
                    challenger.observe_ext_element(*eval);
                }

                let preprocessed =
                    AirOpenedValues { local: preprocessed_evals.to_vec(), next: vec![] };

                let main = AirOpenedValues { local: main_evals.to_vec(), next: vec![] };

                ChipOpenedValues {
                    preprocessed,
                    main,
                    local_cumulative_sum: C::EF::zero(),
                    degree: chip_heights[&air.name()].clone(),
                }
            })
            .collect::<Vec<_>>();

        let shard_open_values = ShardOpenedValues { chips: shard_open_values };

        (shard_open_values, partial_sumcheck_proof)
    }

    /// Generate a proof for a given execution record.
    async fn prove_shard_with_data(
        &self,
        data: ShardData<C>,
        challenger: &mut C::Challenger,
    ) -> (ShardProof<C::Config>, ProverPermit) {
        let ShardData { pk, main_trace_data } = data;
        let MainTraceData { traces, public_values, shard_chips, permit } = main_trace_data;

        // Log the shard data.

        let mut total_number_of_cells = 0;
        tracing::info!("Proving shard");
        for (chip, trace) in shard_chips.iter().zip_eq(traces.values()) {
            let height = trace.num_real_entries();
            let stats = ChipStatistics::new(chip, height);
            tracing::info!("{}", stats);
            total_number_of_cells += stats.total_number_of_cells();
        }
        tracing::info!(
            "Total number of cells: {}, number of variables: {}",
            total_number_of_cells,
            total_number_of_cells.next_power_of_two().ilog2()
        );

        // Observe the public values.
        challenger.observe_slice(&public_values[0..self.num_pv_elts()]);

        // Commit to the traces.
        let (main_commit, main_data) =
            self.commit_traces(&traces).instrument(tracing::debug_span!("commit traces")).await;
        // Observe the commitments.
        challenger.observe(main_commit.clone());

        for chips in shard_chips.iter() {
            let num_real_entries = traces.get(&chips.air.name()).unwrap().num_real_entries();
            challenger.observe(C::F::from_canonical_usize(num_real_entries));
        }

        // Sample the logup challenges.
        let alpha = challenger.sample_ext_element::<C::EF>();
        let beta = challenger.sample_ext_element::<C::EF>();
        let _pv_challenge = challenger.sample_ext_element::<C::EF>();

        let logup_gkr_proof = self
            .logup_gkr_prover
            .prove_logup_gkr(
                &shard_chips,
                pk.preprocessed_data.preprocessed_traces.clone(),
                traces.clone(),
                alpha,
                beta,
                challenger,
            )
            .instrument(tracing::debug_span!("logup gkr proof"))
            .await;
        // Get the challenge for batching constraints.
        let batching_challenge = challenger.sample_ext_element::<C::EF>();
        // Get the challenge for batching the evaluations from the GKR proof.
        let gkr_opening_batch_challenge = challenger.sample_ext_element::<C::EF>();

        // Generate the zerocheck proof.
        let (shard_open_values, zerocheck_partial_sumcheck_proof) = self
            .zerocheck(
                &shard_chips,
                pk.preprocessed_data.preprocessed_traces.clone(),
                traces,
                batching_challenge,
                gkr_opening_batch_challenge,
                &logup_gkr_proof.logup_evaluations,
                public_values.clone(),
                challenger,
            )
            .instrument(tracing::debug_span!("zerocheck"))
            .await;

        // Get the evaluation point for the trace polynomials.
        let evaluation_point = zerocheck_partial_sumcheck_proof.point_and_eval.0.clone();
        let mut preprocessed_evaluation_claims: Option<Evaluations<C::EF, C::B>> = None;
        let mut main_evaluation_claims = Evaluations::new(vec![]);

        let alloc = self.trace_generator.allocator();

        for open_values in shard_open_values.chips.iter() {
            let prep_local = &open_values.preprocessed.local;
            let main_local = &open_values.main.local;
            if !prep_local.is_empty() {
                let preprocessed_evals =
                    alloc.copy_to(&MleEval::from(prep_local.clone())).await.unwrap();
                if let Some(preprocessed_claims) = preprocessed_evaluation_claims.as_mut() {
                    preprocessed_claims.push(preprocessed_evals);
                } else {
                    let evals = Evaluations::new(vec![preprocessed_evals]);
                    preprocessed_evaluation_claims = Some(evals);
                }
            }
            let main_evals = alloc.copy_to(&MleEval::from(main_local.clone())).await.unwrap();
            main_evaluation_claims.push(main_evals);
        }

        let round_evaluation_claims = preprocessed_evaluation_claims
            .into_iter()
            .chain(once(main_evaluation_claims))
            .collect::<Rounds<_>>();

        let round_prover_data = pk
            .preprocessed_data
            .preprocessed_data
            .clone()
            .into_iter()
            .chain(once(main_data))
            .collect::<Rounds<_>>();

        // Generate the evaluation proof.
        let evaluation_proof = self
            .pcs_prover
            .prove_trusted_evaluations(
                evaluation_point,
                round_evaluation_claims,
                round_prover_data,
                challenger,
            )
            .instrument(tracing::debug_span!("prove evaluation claims"))
            .await
            .unwrap();

        let shard_chips = shard_chips.iter().map(MachineAir::name).collect::<BTreeSet<_>>();

        let proof = ShardProof {
            main_commitment: main_commit,
            opened_values: shard_open_values,
            logup_gkr_proof,
            evaluation_proof,
            zerocheck_proof: zerocheck_partial_sumcheck_proof,
            public_values,
            shard_chips,
        };

        (proof, permit)
    }
}

/// The shape of the core proof. This and prover setup parameters should entirely determine the
/// verifier circuit.
#[derive_where(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug)]
pub struct CoreProofShape<F: Field, A: MachineAir<F>> {
    /// The chips included in the record.
    pub shard_chips: BTreeSet<Chip<F, A>>,

    /// The multiple of `log_stacking_height` that the preprocessed area adds up to.
    pub preprocessed_multiple: usize,

    /// The multiple of `log_stacking_height` that the main area adds up to.
    pub main_multiple: usize,
}

/// A proving key for a STARK.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Tensor<C::F, C::B>: Serialize, JaggedProverData<C::PcsProverComponents>: Serialize, C::F: Serialize, C::B: Serialize, "
))]
#[serde(bound(
    deserialize = "Tensor<C::F, C::B>: Deserialize<'de>, JaggedProverData<C::PcsProverComponents>: Deserialize<'de>, C::F: Deserialize<'de>, C::B: Deserialize<'de>, "
))]
pub struct ShardProverData<C: ShardProverComponents> {
    /// The preprocessed traces.
    pub preprocessed_traces: Traces<C::F, C::B>,
    /// The pcs data for the preprocessed traces.
    pub preprocessed_data: Option<JaggedProverData<C::PcsProverComponents>>,
}

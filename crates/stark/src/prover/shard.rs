use std::{collections::BTreeMap, fmt::Debug, iter::once, sync::Arc};

use itertools::Itertools;
use p3_uni_stark::get_symbolic_constraints;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_air::Air;
use slop_algebra::{AbstractField, ExtensionField, Field};
use slop_alloc::{Backend, CanCopyFromRef};
use slop_challenger::{CanObserve, FieldChallenger};
use slop_commit::Rounds;
use slop_jagged::{JaggedBackend, JaggedProver, JaggedProverComponents, JaggedProverData};
use slop_matrix::dense::RowMajorMatrixView;
use slop_multilinear::{
    Evaluations, HostEvaluationBackend, MleEval, MultilinearPcsChallenger, PointBackend,
};
use slop_sumcheck::{reduce_sumcheck_to_evaluation, PartialSumcheckProof};
use slop_tensor::Tensor;
use tokio::sync::mpsc::Sender;
use tracing::Instrument;

use crate::{
    air::{MachineAir, MachineProgram},
    prover::{ZeroCheckPoly, ZerocheckAir},
    septic_digest::SepticDigest,
    AirOpenedValues, Chip, ChipDimensions, ChipOpenedValues, ConstraintSumcheckFolder, Machine,
    MachineConfig, MachineRecord, MachineVerifyingKey, ShardOpenedValues, ShardProof,
    PROOF_MAX_NUM_PVS,
};

use super::{TraceGenerator, Traces, ZercocheckBackend, ZerocheckProverData};

/// The components of the machine prover.
///
/// This trait is used specify a configuration of a hypercube prover.
pub trait MachineProverComponents: 'static + Send + Sync + Sized + Debug {
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
        + HostEvaluationBackend<Self::EF, Self::EF>;

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
    /// constraints vanish into an evaluation claim at a random point for the traces, consideres
    /// as multilinear polynomials.
    type ZerocheckProverData: ZerocheckProverData<Self::F, Self::EF, Self::B, Air = Self::Air>;

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

/// A collection of traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize, Tensor<F, B>: Serialize,"))]
#[serde(bound(deserialize = "F: Deserialize<'de>, Tensor<F, B>: Deserialize<'de>, "))]
pub struct ShardData<F, B: Backend> {
    /// The traces.
    pub traces: Traces<F, B>,
    /// The public values.
    pub public_values: Vec<F>,
}

/// A prover for the hypercube STARK, given a configuration.
pub struct ShardProver<C: MachineProverComponents> {
    /// A prover for the PCS.
    pub pcs_prover: JaggedProver<C::PcsProverComponents>,
    /// A prover for the zerocheck IOP.
    pub zerocheck_prover_data: C::ZerocheckProverData,
    /// The trace generator.
    pub trace_generator: C::TraceGenerator,
}

impl<C: MachineProverComponents> ShardProver<C> {
    /// Get all the chips in the machine.
    pub fn chips(&self) -> &[Chip<C::F, C::Air>] {
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

    /// Setup from a program.
    ///
    /// The setup phase produces a pair '(pk, vk)' of proving and verifying keys. The proving key
    /// consists of information used by the prover that only depends on the program itself and not
    /// a specific execution.
    pub async fn setup(
        &self,
        program: Arc<C::Program>,
    ) -> (MachineProvingKey<C>, MachineVerifyingKey<C::Config>) {
        let program_sent = program.clone();
        let initial_global_cumulative_sum =
            tokio::task::spawn_blocking(move || program_sent.initial_global_cumulative_sum())
                .await
                .unwrap();
        self.setup_with_initial_global_cumulative_sum(program, initial_global_cumulative_sum).await
    }

    /// Setup from a program with a specific initial global cumulative sum.
    pub async fn setup_with_initial_global_cumulative_sum(
        &self,
        program: Arc<C::Program>,
        initial_global_cumulative_sum: SepticDigest<C::F>,
    ) -> (MachineProvingKey<C>, MachineVerifyingKey<C::Config>) {
        let pc_start = program.pc_start();
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        self.trace_generator
            .generate_preprocessed_traces(program, self.max_log_row_count(), &tx)
            .await;

        let preprocessed_traces = rx.recv().await.unwrap();

        let constraints_map = self
            .chips()
            .iter()
            .map(|chip| {
                // Count the number of constraints.
                let num_main_constraints = get_symbolic_constraints(
                    chip.air.as_ref(),
                    chip.preprocessed_width(),
                    PROOF_MAX_NUM_PVS,
                )
                .len();
                (chip.name(), num_main_constraints)
            })
            .collect::<BTreeMap<_, _>>();

        // Commit to the preprocessed traces, if there are any.
        let (preprocessed_commit, preprocessed_data) = if preprocessed_traces.len() > 0 {
            let message = self
                .chips()
                .iter()
                .filter_map(|air| preprocessed_traces.get(&air.name()))
                .cloned()
                .collect::<Vec<_>>();
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

        let pk = MachineProvingKey {
            pc_start,
            initial_global_cumulative_sum,
            preprocessed_commit: preprocessed_commit.clone(),
            preprocessed_traces,
            preprocessed_data,
            constraints_map,
        };

        let vk = MachineVerifyingKey {
            pc_start,
            initial_global_cumulative_sum,
            preprocessed_commit,
            preprocessed_chip_information,
        };

        (pk, vk)
    }

    async fn commit_traces(
        &self,
        traces: &Traces<C::F, C::B>,
    ) -> (C::Commitment, JaggedProverData<C::PcsProverComponents>) {
        let message = self
            .chips()
            .iter()
            .map(|air| traces.get(&air.name()).unwrap().clone())
            .collect::<Vec<_>>();
        self.pcs_prover.commit_multilinears(message).await.unwrap()
    }

    async fn zerocheck(
        &self,
        preprocessed_traces: Traces<C::F, C::B>,
        traces: Traces<C::F, C::B>,
        constraints_map: &BTreeMap<String, usize>,
        batching_challenge: C::EF,
        public_values: Vec<C::F>,
        challenger: &mut C::Challenger,
    ) -> (ShardOpenedValues<C::F, C::EF>, PartialSumcheckProof<C::EF>) {
        // Sample the random point to make the zerocheck claims.
        let zeta = challenger.sample_point::<C::EF>(self.max_log_row_count() as u32);
        let max_num_constraints = itertools::max(constraints_map.values()).unwrap();
        let powers_of_challenge =
            batching_challenge.powers().take(*max_num_constraints).collect::<Vec<_>>();
        let airs = self.chips().iter().map(|chip| chip.air.clone()).collect::<Vec<_>>();

        let public_values = Arc::new(public_values);

        let mut zerocheck_polys = Vec::new();

        let mut log_degrees = BTreeMap::new();
        for air in self.chips().iter() {
            let main_trace = traces.get(&air.name()).unwrap().clone();
            let log_degree = main_trace.num_real_entries().next_power_of_two().checked_ilog2();
            log_degrees.insert(air.name(), log_degree);
            let name = air.name();
            let num_variables = main_trace.num_variables();
            assert_eq!(num_variables, self.pcs_prover.max_log_row_count as u32);

            let preprocessed_width = air.preprocessed_width();
            let dummy_preprocessed_trace = vec![C::F::zero(); preprocessed_width];
            let dummy_main_trace = vec![C::F::zero(); main_trace.num_polynomials()];

            let chip_num_constraints = constraints_map[&name];

            // Calculate powers of alpha for constraint evaluation:
            // 1. Generate sequence [α⁰, α¹, ..., α^(n-1)] where n = chip_num_constraints.
            // 2. Reverse to [α^(n-1), ..., α¹, α⁰] to align with Horner's method in the verifier.
            let mut chip_powers_of_alpha = powers_of_challenge[0..chip_num_constraints].to_vec();
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

            let alpha_powers = Arc::new(chip_powers_of_alpha);
            let air_data = self.zerocheck_prover_data.round_prover(
                air.air.clone(),
                public_values.clone(),
                alpha_powers,
            );
            let preprocessed_trace = preprocessed_traces.get(&name).cloned();

            let initial_geq_value =
                if main_trace.num_real_entries() > 0 { C::EF::zero() } else { C::EF::one() };
            let zerocheck_poly = ZeroCheckPoly::new(
                air_data,
                zeta.clone(),
                preprocessed_trace,
                main_trace,
                C::EF::one(),
                initial_geq_value,
                padded_row_adjustment,
            );
            zerocheck_polys.push(zerocheck_poly);
        }

        // Same lambda for the RLC of the zerocheck polynomials.
        let lambda = challenger.sample_ext_element::<C::EF>();

        let num_zerocheck_polys = zerocheck_polys.len();
        // Compute the sumcheck proof for the zerocheck polynomials.
        let (partial_sumcheck_proof, component_poly_evals) = reduce_sumcheck_to_evaluation(
            zerocheck_polys,
            challenger,
            vec![C::EF::zero(); num_zerocheck_polys],
            1,
            lambda,
        )
        .instrument(tracing::debug_span!("zerocheck sumcheck proof"))
        .await;

        // Compute the chip openings from the component poly evaluations.

        assert_eq!(component_poly_evals.len(), airs.len());
        let shard_open_values = airs
            .into_iter()
            .zip_eq(component_poly_evals)
            .map(|(air, evals)| {
                let (preprocessed_evals, main_evals) = evals.split_at(air.preprocessed_width());

                let preprocessed =
                    AirOpenedValues { local: preprocessed_evals.to_vec(), next: vec![] };

                let main = AirOpenedValues { local: main_evals.to_vec(), next: vec![] };

                ChipOpenedValues {
                    preprocessed,
                    main,
                    global_cumulative_sum: SepticDigest::zero(),
                    local_cumulative_sum: C::EF::zero(),
                    log_degree: log_degrees[&air.name()],
                }
            })
            .collect::<Vec<_>>();

        let shard_open_values = ShardOpenedValues { chips: shard_open_values };

        (shard_open_values, partial_sumcheck_proof)
    }

    /// Generate the shard data
    pub async fn generate_traces(&self, record: C::Record, tx: &Sender<ShardData<C::F, C::B>>) {
        // Generate the traces.
        self.trace_generator.generate_main_traces(record, self.max_log_row_count(), tx).await;
    }

    /// Generate a proof for a given execution record.
    pub async fn prove_shard(
        &self,
        pk: &MachineProvingKey<C>,
        data: ShardData<C::F, C::B>,
        challenger: &mut C::Challenger,
    ) -> ShardProof<C::Config> {
        let ShardData { traces, public_values } = data;
        // Observe the public values.
        challenger.observe_slice(&public_values[0..self.num_pv_elts()]);

        // Commit to the traces.
        let (main_commit, main_data) =
            self.commit_traces(&traces).instrument(tracing::debug_span!("commit traces")).await;
        // Observe the commitments.
        challenger.observe(main_commit.clone());

        // Get the challenge for batching constraints.
        let batching_challenge = challenger.sample_ext_element::<C::EF>();

        // Generate the zerocheck proof.
        let (shard_open_values, zerocheck_partial_sumcheck_proof) = self
            .zerocheck(
                pk.preprocessed_traces.clone(),
                traces,
                &pk.constraints_map,
                batching_challenge,
                // TODO: remove the need for this clone
                public_values.clone(),
                challenger,
            )
            .instrument(tracing::debug_span!("zerocheck"))
            .await;

        // Get the evaluation point for the trace polynomials.
        let evaluation_point = zerocheck_partial_sumcheck_proof.point_and_eval.0.clone();
        // Get the evaluation claims of the trace polynomials.
        let mut preprocessed_evaluation_claims: Option<Evaluations<C::EF, C::B>> = None;
        let mut main_evaluation_claims = Evaluations::new(vec![]);

        let alloc = self.trace_generator.allocator();

        for (air, open_values) in self.chips().iter().zip_eq(shard_open_values.chips.iter()) {
            tracing::info!("air: {:?}", air.name());
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

        let round_prover_data =
            pk.preprocessed_data.clone().into_iter().chain(once(main_data)).collect::<Rounds<_>>();

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

        ShardProof {
            main_commitment: main_commit,
            opened_values: shard_open_values,
            evaluation_proof,
            zerocheck_proof: zerocheck_partial_sumcheck_proof,
            public_values,
        }
    }
}

/// A proving key for a STARK.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Tensor<C::F, C::B>: Serialize, JaggedProverData<C::PcsProverComponents>: Serialize"
))]
#[serde(bound(
    deserialize = "Tensor<C::F, C::B>: Deserialize<'de>, JaggedProverData<C::PcsProverComponents>: Deserialize<'de>"
))]
pub struct MachineProvingKey<C: MachineProverComponents> {
    /// The start pc of the program.
    pub pc_start: C::F,
    /// The starting global digest of the program, after incorporating the initial memory.
    pub initial_global_cumulative_sum: SepticDigest<C::F>,
    /// The commitment to the preprocessed traces.
    pub preprocessed_commit: Option<C::Commitment>,
    /// The preprocessed traces.
    pub preprocessed_traces: Traces<C::F, C::B>,
    /// The pcs data for the preprocessed traces.
    pub preprocessed_data: Option<JaggedProverData<C::PcsProverComponents>>,
    /// The number of total constraints for each chip.
    pub constraints_map: BTreeMap<String, usize>,
}

impl<C: MachineProverComponents> MachineProvingKey<C> {
    /// Observes the values of the proving key into the challenger.
    pub fn observe_into(&self, challenger: &mut C::Challenger) {
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        // Observe the padding.
        challenger.observe(C::F::zero());
    }
}

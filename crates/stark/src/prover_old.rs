use crate::block_on;
use crate::septic_digest::SepticDigest;
use crate::{AirOpenedValues, ChipOpenedValues, ShardOpenedValues, ZeroCheckPoly};
use core::fmt::Display;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::SymbolicAirBuilder;
use serde::{de::DeserializeOwned, Serialize};
use slop_commit::{Message, Rounds};
use slop_multilinear::{Evaluations, Mle, MleEval, Point};
use slop_sumcheck::reduce_sumcheck_to_evaluation;
use slop_tensor::Tensor;
use std::sync::Arc;
use std::{cmp::Reverse, error::Error, time::Instant};

use super::{
    Com, OpeningProof, StarkGenericConfig, StarkMachine, StarkProvingKey, Val,
    VerifierConstraintFolder,
};
use crate::{
    air::MachineAir, lookup::InteractionBuilder, opts::SP1CoreOpts, record::MachineRecord,
    zerocheck::ZerocheckCpuProver, Challenger, ConstraintSumcheckFolder, DebugConstraintBuilder,
    MachineChip, MachineProof, PcsProverData, ShardMainData, ShardProof, StarkVerifyingKey,
};

/// An algorithmic & hardware independent prover implementation for any [`MachineAir`].
pub trait MachineProver<
    SC: StarkGenericConfig,
    A: MachineAir<SC::Val> + Air<SymbolicAirBuilder<SC::Val>>,
>: 'static + Send + Sync
{
    /// The type used to store the polynomial commitment schemes data.
    type DeviceProverData;

    /// The type used to store the proving key.
    type DeviceProvingKey: MachineProvingKey<SC>;

    /// The type used for error handling.
    type Error: Error + Send + Sync;

    /// Create a new prover from a given machine.
    fn new(machine: StarkMachine<SC, A>) -> Self;

    /// A reference to the machine that this prover is using.
    fn machine(&self) -> &StarkMachine<SC, A>;

    /// Setup the preprocessed data into a proving and verifying key.
    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, StarkVerifyingKey<SC>);

    /// Setup the proving key given a verifying key. This is similar to `setup` but faster since
    /// some computed information is already in the verifying key.
    fn pk_from_vk(
        &self,
        program: &A::Program,
        vk: &StarkVerifyingKey<SC>,
    ) -> Self::DeviceProvingKey;

    /// Copy the proving key from the host to the device.
    fn pk_to_device(&self, pk: &StarkProvingKey<SC>) -> Self::DeviceProvingKey;

    /// Copy the proving key from the device to the host.
    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> StarkProvingKey<SC>;

    /// Generate the main traces.
    fn generate_traces(&self, record: &A::Record) -> Vec<(String, RowMajorMatrix<Val<SC>>)> {
        let shard_chips = self.shard_chips(record).collect::<Vec<_>>();

        // For each chip, generate the trace.
        let parent_span = tracing::debug_span!("generate traces for shard");
        parent_span.in_scope(|| {
            shard_chips
                .par_iter()
                .map(|chip| {
                    let chip_name = chip.name();
                    let begin = Instant::now();
                    let trace = chip.generate_trace(record, &mut A::Record::default());
                    tracing::debug!(
                        parent: &parent_span,
                        "generated trace for chip {} in {:?}",
                        chip_name,
                        begin.elapsed()
                    );
                    (chip_name, trace)
                })
                .collect::<Vec<_>>()
        })
    }

    /// Commit to the main traces.
    fn commit(
        &self,
        record: &A::Record,
        traces: Vec<(String, RowMajorMatrix<Val<SC>>)>,
    ) -> ShardMainData<SC>;

    /// Observe the main commitment and public values and update the challenger.
    fn observe(
        &self,
        challenger: &mut SC::Challenger,
        commitment: Com<SC>,
        public_values: &[SC::Val],
    ) {
        // Observe the commitment.
        challenger.observe(commitment);

        // Observe the public values.
        challenger.observe_slice(public_values);
    }

    /// Compute the openings of the traces.
    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        data: ShardMainData<SC>,
        challenger: &mut SC::Challenger,
    ) -> Result<ShardProof<SC>, Self::Error>;

    /// Generate a proof for the given records.
    fn prove(
        &self,
        pk: &Self::DeviceProvingKey,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
            + Air<SymbolicAirBuilder<Val<SC>>>;

    /// The stark config for the machine.
    fn config(&self) -> &SC {
        self.machine().config()
    }

    /// The number of public values elements.
    fn num_pv_elts(&self) -> usize {
        self.machine().num_pv_elts()
    }

    /// The chips that will be necessary to prove this record.
    fn shard_chips<'a, 'b>(
        &'a self,
        record: &'b A::Record,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
        SC: 'b,
    {
        self.machine().shard_chips(record)
    }

    /// Debug the constraints for the given inputs.
    fn debug_constraints(
        &self,
        pk: &StarkProvingKey<SC>,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
    ) where
        SC::Val: PrimeField32,
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        self.machine().debug_constraints(pk, records, challenger);
    }
}

/// A proving key for any [`MachineAir`] that is agnostic to hardware.
pub trait MachineProvingKey<SC: StarkGenericConfig>: Send + Sync {
    /// The main commitment.
    fn preprocessed_commit(&self) -> Option<Com<SC>>;

    /// The start pc.
    fn pc_start(&self) -> Val<SC>;

    /// The initial global cumulative sum.
    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>>;

    /// Observe itself in the challenger.
    fn observe_into(&self, challenger: &mut Challenger<SC>);
}

/// A prover implementation based on x86 and ARM CPUs.
pub struct CpuProver<SC: StarkGenericConfig, A> {
    machine: StarkMachine<SC, A>,
}

/// An error that occurs during the execution of the [`CpuProver`].
#[derive(Debug, Clone, Copy)]
pub struct CpuProverError;

impl<SC, A> MachineProver<SC, A> for CpuProver<SC, A>
where
    SC: 'static + StarkGenericConfig + Send + Sync,
    A: MachineAir<SC::Val>
        + for<'a> Air<ConstraintSumcheckFolder<'a, SC::Val, SC::Val, SC::Challenge>>
        + for<'a> Air<ConstraintSumcheckFolder<'a, SC::Val, SC::Challenge, SC::Challenge>>
        + Air<InteractionBuilder<Val<SC>>>
        + for<'a> Air<VerifierConstraintFolder<'a, SC>>
        + Air<SymbolicAirBuilder<Val<SC>>>,
    A::Record: MachineRecord<Config = SP1CoreOpts>,
    SC::Val: PrimeField32,
    Com<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync + Serialize + DeserializeOwned,
    OpeningProof<SC>: Send + Sync,
    SC::Challenger: Clone,
{
    type DeviceProverData = PcsProverData<SC>;
    type DeviceProvingKey = StarkProvingKey<SC>;
    type Error = CpuProverError;

    fn new(machine: StarkMachine<SC, A>) -> Self {
        Self { machine }
    }

    fn machine(&self) -> &StarkMachine<SC, A> {
        &self.machine
    }

    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, StarkVerifyingKey<SC>) {
        let (pk, vk) = self.machine().setup(program);
        (pk, vk)
    }

    fn pk_from_vk(
        &self,
        program: &A::Program,
        vk: &StarkVerifyingKey<SC>,
    ) -> Self::DeviceProvingKey {
        self.machine().setup_core(program, vk.initial_global_cumulative_sum).0
    }

    fn pk_to_device(&self, pk: &StarkProvingKey<SC>) -> Self::DeviceProvingKey {
        pk.clone()
    }

    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> StarkProvingKey<SC> {
        pk.clone()
    }

    fn commit(
        &self,
        record: &A::Record,
        mut named_traces: Vec<(String, RowMajorMatrix<Val<SC>>)>,
    ) -> ShardMainData<SC> {
        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_traces.sort_by_key(|(name, trace)| (Reverse(trace.height()), name.clone()));

        let pcs = self.config().prover_pcs();

        let traces = named_traces
            .iter()
            .map(|(_, trace)| Mle::new(Tensor::from(trace.to_owned())))
            .collect::<Vec<_>>();

        let (main_commit, main_data) =
            block_on(pcs.commit_multilinears(Message::from(traces))).ok().unwrap();

        // Get the chip ordering.
        let chip_ordering =
            named_traces.iter().enumerate().map(|(i, (name, _))| (name.to_owned(), i)).collect();

        let traces = named_traces.into_iter().map(|(_, trace)| trace.into()).collect::<Vec<_>>();

        ShardMainData {
            traces,
            main_commit,
            main_data,
            chip_ordering,
            public_values: record.public_values(),
        }
    }

    /// Prove the program for the given shard and given a commitment to the main data.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::redundant_closure_for_method_calls)]
    #[allow(clippy::map_unwrap_or)]
    fn open(
        &self,
        pk: &StarkProvingKey<SC>,
        data: ShardMainData<SC>,
        challenger: &mut <SC as StarkGenericConfig>::Challenger,
    ) -> Result<ShardProof<SC>, Self::Error> {
        let chips = self.machine().shard_chips_ordered(&data.chip_ordering).collect::<Vec<_>>();
        let traces = data.traces;

        let log_degrees =
            traces.iter().map(|trace| trace.num_variables() as usize).collect::<Vec<_>>();

        let has_preprocess = pk.preprocessed_commit.is_some();
        assert!(has_preprocess == pk.preprocessed_data.is_some());

        if has_preprocess {
            challenger.observe(pk.preprocessed_commit.as_ref().unwrap().clone());
        }
        challenger.observe(data.main_commit.clone());
        // Finalize commit.
        let pcs = self.config().prover_pcs();

        let (prover_data, commitments) = if has_preprocess {
            (
                Rounds { rounds: vec![pk.preprocessed_data.clone().unwrap(), data.main_data] },
                Rounds {
                    rounds: vec![
                        pk.preprocessed_commit.as_ref().unwrap().clone(),
                        data.main_commit.clone(),
                    ],
                },
            )
        } else {
            (Rounds { rounds: vec![data.main_data] }, Rounds { rounds: vec![data.main_commit] })
        };

        challenger.observe_slice(&data.public_values[0..self.num_pv_elts()]);

        let prep_traces = tracing::debug_span!("generate permutation traces").in_scope(|| {
            chips
                .par_iter()
                .map(|chip| pk.chip_ordering.get(&chip.name()).map(|&index| &pk.traces[index]))
                .collect::<Vec<_>>()
        });

        // Compute some statistics.
        for i in 0..chips.len() {
            let trace_width = traces[i].num_polynomials();
            let trace_height = 2usize.pow(traces[i].num_variables());
            let prep_width = prep_traces[i].map_or(0, |x| x.num_polynomials());
            tracing::debug!(
                "{:<15} | Main Cols = {:<5} | Pre Cols = {:<5}  | Rows = {:<5} | Cells = {:<10}",
                chips[i].name(),
                trace_width,
                prep_width,
                trace_height,
                trace_width * trace_height,
            );
        }

        // Generate the zerocheck sumcheck proof for each chip.
        let alpha: SC::Challenge = challenger.sample_ext_element::<SC::Challenge>();

        // Get the max num variables for all chips to determine the random point dimensions.
        let zeta = Point::<SC::Challenge>::new(
            (0..pcs.max_log_row_count).map(|_| challenger.sample()).collect::<Vec<_>>().into(),
        );

        let parent_span = tracing::debug_span!("compute zerocheck sumcheck proofs");

        let max_num_constraints = pk.constraints_map.values().max().unwrap();
        let powers_of_alpha = alpha.powers().take(*max_num_constraints).collect::<Vec<_>>();

        let public_values = Arc::new(data.public_values.clone());
        // First collect the zerocheck polynomials.
        let (zerocheck_polys, num_preprocessed_columns): (Vec<_>, Vec<_>) = chips
            .iter()
            .zip_eq(traces)
            .enumerate()
            .map(|(i, (chip, main_trace))| {
                let preprocessed_trace =
                    pk.chip_ordering.get(&chip.name()).map(|&index| pk.traces[index].clone());

                let log_height = main_trace.num_variables();

                let num_preprocessed_cols =
                    preprocessed_trace.as_ref().map_or(0, |pp_trace| pp_trace.num_polynomials());

                let num_padded_vars = pcs.max_log_row_count - log_height as usize;
                tracing::debug!(
                    "zerocheck poly: chip {} has num_padded_vars: {}",
                    chip.name(),
                    num_padded_vars
                );

                let dummy_preprocessed_trace = vec![SC::Val::zero(); chip.preprocessed_width()];
                let dummy_main_trace = vec![SC::Val::zero(); chip.width()];

                let chip_num_constraints = pk.constraints_map.get(&chips[i].name()).unwrap();

                // Calculate powers of alpha for constraint evaluation:
                // 1. Generate sequence [α⁰, α¹, ..., α^(n-1)] where n = chip_num_constraints.
                // 2. Reverse to [α^(n-1), ..., α¹, α⁰] to align with Horner's method in the verifier.
                let mut chip_powers_of_alpha = powers_of_alpha[0..*chip_num_constraints].to_vec();
                chip_powers_of_alpha.reverse();

                let mut folder = ConstraintSumcheckFolder {
                    preprocessed: RowMajorMatrixView::new_row(&dummy_preprocessed_trace),
                    main: RowMajorMatrixView::new_row(&dummy_main_trace),
                    accumulator: SC::Challenge::zero(),
                    public_values: &data.public_values,
                    constraint_index: 0,
                    powers_of_alpha: &chip_powers_of_alpha,
                };
                chip.air.eval(&mut folder);
                let padded_row_adjustment = folder.accumulator;

                let main_trace = Arc::new(main_trace);
                let alpha_powers = Arc::new(chip_powers_of_alpha);
                let air_data =
                    ZerocheckCpuProver::new(chip.air.clone(), public_values.clone(), alpha_powers);
                (
                    ZeroCheckPoly::new(
                        air_data,
                        zeta.clone(),
                        preprocessed_trace,
                        main_trace,
                        SC::Challenge::one(),
                        SC::Challenge::zero(),
                        num_padded_vars,
                        padded_row_adjustment,
                    ),
                    num_preprocessed_cols,
                )
            })
            .unzip();

        // Same lambda for the RLC of the zerocheck polynomials.
        let lambda = challenger.sample_ext_element::<SC::Challenge>();

        let num_zerocheck_polys = zerocheck_polys.len();

        // Compute the sumcheck proof for the zerocheck polynomials.
        let (sc_proof, component_poly_evals) = parent_span.in_scope(|| {
            block_on(reduce_sumcheck_to_evaluation::<SC::Val, SC::Challenge, SC::Challenger>(
                zerocheck_polys,
                challenger,
                vec![SC::Challenge::zero(); num_zerocheck_polys],
                1,
                lambda,
            ))
        });

        // Split the component polynomial evaluations into preprocessed and main openings
        let (preprocessed_openings, main_openings): (Vec<_>, Evaluations<_>) = component_poly_evals
            .iter()
            .zip(num_preprocessed_columns.iter())
            .map(|(evals, num_cols)| {
                let (preprocessed_evals, main_evals) = evals.split_at(*num_cols);

                (preprocessed_evals, MleEval::from(main_evals.to_vec()))
            })
            .unzip();

        // // Verify the main openings.
        // main_openings.iter().zip(traces.iter()).for_each(|(main_opening, trace)| {
        //     assert!(Mle::eval_matrix_at_point(trace, &sc_proof.point_and_eval.0) == *main_opening);
        // });

        // Filter out preprocessing openings that are empty (e.g. the table doesn't have any
        // preprocessing columns).
        let filtered_preprocessed_openings = preprocessed_openings
            .clone()
            .into_iter()
            .filter(|x| !x.is_empty())
            .map(|x| x.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        let openings = if has_preprocess {
            Rounds { rounds: vec![filtered_preprocessed_openings, main_openings.clone()] }
        } else {
            Rounds { rounds: vec![main_openings.clone()] }
        };

        // Generate the opening proof.
        let opening_proof = block_on(pcs.prove_trusted_evaluations(
            sc_proof.point_and_eval.0.clone(),
            openings,
            prover_data,
            challenger,
        ))
        .ok()
        .unwrap();

        // Collect the opened values for each chip.
        let preprocessed_opened_values = preprocessed_openings
            .into_iter()
            .zip(chips.iter())
            .map(|(op, _)| AirOpenedValues { local: op.to_vec(), next: vec![] })
            .collect::<Vec<_>>();

        let main_opened_values = main_openings
            .into_iter()
            .zip(chips.iter())
            .map(|(op, _)| AirOpenedValues { local: op.to_vec(), next: vec![] })
            .collect::<Vec<_>>();

        assert!(preprocessed_opened_values.len() == chips.len());
        assert!(main_opened_values.len() == chips.len());

        let opened_values = main_opened_values
            .into_iter()
            .zip_eq(preprocessed_opened_values)
            .zip_eq(log_degrees)
            .map(|((main, preprocessed), log_degree)| ChipOpenedValues {
                preprocessed,
                main,
                global_cumulative_sum: SepticDigest::<Val<SC>>::zero(),
                local_cumulative_sum: SC::Challenge::zero(),
                log_degree,
            })
            .collect::<Vec<_>>();

        Ok(ShardProof::<SC> {
            commitments: commitments.rounds.into_iter().collect(),
            zerocheck_proof: sc_proof,
            opening_proof,
            opened_values: ShardOpenedValues { chips: opened_values },
            chip_ordering: data.chip_ordering,
            public_values: data.public_values,
        })
    }

    /// Prove the execution record is valid.
    ///
    /// Given a proving key `pk` and a matching execution record `record`, this function generates
    /// a STARK proof that the execution record is valid.
    #[allow(clippy::needless_for_each)]
    fn prove(
        &self,
        pk: &StarkProvingKey<SC>,
        mut records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        // Generate dependencies.
        self.machine().generate_dependencies(&mut records, &opts, None);

        // Observe the preprocessed commitment.
        pk.observe_into(challenger);

        let shard_proofs = tracing::info_span!("prove_shards").in_scope(|| {
            records
                .into_par_iter()
                .map(|record| {
                    let named_traces = self.generate_traces(&record);
                    let shard_data = self.commit(&record, named_traces);
                    self.open(pk, shard_data, &mut challenger.clone())
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        Ok(MachineProof { shard_proofs })
    }
}

impl<SC> MachineProvingKey<SC> for StarkProvingKey<SC>
where
    SC: 'static + StarkGenericConfig + Send + Sync,
    PcsProverData<SC>: Send + Sync + Serialize + DeserializeOwned,
    Com<SC>: Send + Sync,
{
    fn preprocessed_commit(&self) -> Option<Com<SC>> {
        self.preprocessed_commit.clone()
    }

    fn pc_start(&self) -> Val<SC> {
        self.pc_start
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>> {
        self.initial_global_cumulative_sum
    }

    fn observe_into(&self, challenger: &mut Challenger<SC>) {
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        let zero = Val::<SC>::zero();
        challenger.observe(zero);
    }
}

impl Display for CpuProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DefaultProverError")
    }
}

impl Error for CpuProverError {}

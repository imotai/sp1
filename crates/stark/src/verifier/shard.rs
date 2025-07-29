use derive_where::derive_where;
use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    ops::Deref,
};

use itertools::Itertools;
use slop_air::{Air, BaseAir};
use slop_algebra::{AbstractField, PrimeField32};
use slop_basefold::DefaultBasefoldConfig;
use slop_challenger::{CanObserve, FieldChallenger};
use slop_commit::Rounds;
use slop_jagged::{
    JaggedBasefoldConfig, JaggedConfig, JaggedEvalConfig, JaggedPcsVerifier,
    JaggedPcsVerifierError, MachineJaggedPcsVerifier,
};
use slop_matrix::dense::RowMajorMatrixView;
use slop_multilinear::{full_geq, Evaluations, Mle, MleEval, MultilinearPcsVerifier};
use slop_sumcheck::{partially_verify_sumcheck_proof, SumcheckError};
use thiserror::Error;

use crate::{
    air::MachineAir, prover::CoreProofShape, Chip, ChipOpenedValues, LogUpEvaluations,
    LogUpGkrVerifier, LogupGkrVerificationError, Machine, VerifierConstraintFolder,
    VerifierPublicValuesConstraintFolder,
};

use super::{MachineConfig, MachineVerifyingKey, ShardOpenedValues, ShardProof};
use crate::record::MachineRecord;

/// A verifier for shard proofs.
#[derive_where(Clone)]
pub struct ShardVerifier<C: MachineConfig, A: MachineAir<C::F>> {
    /// The jagged pcs verifier.
    pub pcs_verifier: JaggedPcsVerifier<C>,
    /// The machine.
    pub machine: Machine<C::F, A>,
}

/// An error that occurs during the verification of a shard proof.
#[derive(Debug, Error)]
pub enum ShardVerifierError<EF, PcsError> {
    /// The pcs opening proof is invalid.
    #[error("invalid pcs opening proof: {0}")]
    InvalidopeningArgument(#[from] JaggedPcsVerifierError<EF, PcsError>),
    /// The constraints check failed.
    #[error("constraints check failed: {0}")]
    ConstraintsCheckFailed(SumcheckError),
    /// The cumulative sums error.
    #[error("cumulative sums error: {0}")]
    CumulativeSumsError(&'static str),
    /// The preprocessed chip id mismatch.
    #[error("preprocessed chip id mismatch: {0}")]
    PreprocessedChipIdMismatch(String, String),
    /// The error to report when the preprocessed chip height in the verifying key does not match
    /// the chip opening height.
    #[error("preprocessed chip height mismatch: {0}")]
    PreprocessedChipHeightMismatch(String),
    /// The chip opening length mismatch.
    #[error("chip opening length mismatch")]
    ChipOpeningLengthMismatch,
    /// The cpu chip is missing.
    #[error("missing cpu chip")]
    MissingCpuChip,
    /// The shape of the openings does not match the expected shape.
    #[error("opening shape mismatch: {0}")]
    OpeningShapeMismatch(#[from] OpeningShapeError),
    /// The GKR verification failed.
    #[error("GKR verification failed: {0}")]
    GkrVerificationFailed(LogupGkrVerificationError<EF>),
    /// The public values verification failed.
    #[error("public values verification failed")]
    InvalidPublicValues,
    /// The proof has entries with invalid shape.
    #[error("invalid shape of proof")]
    InvalidShape,
    /// The provided chip opened values has incorrect order.
    #[error("invalid chip opening order")]
    InvalidChipOrder(String, String),
    /// The height of the chip is not sent over correctly as bitwise decomposition.
    #[error("invalid height bit decomposition")]
    InvalidHeightBitDecomposition,
    /// The height is larger than `1 << max_log_row_count`.
    #[error("height is larger than maximum possible value")]
    HeightTooLarge,
    /// Invalid number of added columns.
    #[error("invalid number of added columns")]
    InvalidAddedColumns,
    /// Invalid prefix sum.
    #[error("invalid prefix sum")]
    InvalidPrefixSum,
}

/// Derive the error type from the jagged config.
pub type ShardVerifierConfigError<C> = ShardVerifierError<
    <C as JaggedConfig>::EF,
    <<C as JaggedConfig>::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError,
>;

/// An error that occurs when the shape of the openings does not match the expected shape.
#[derive(Debug, Error)]
pub enum OpeningShapeError {
    /// The width of the preprocessed trace does not match the expected width.
    #[error("preprocessed width mismatch: {0} != {1}")]
    PreprocessedWidthMismatch(usize, usize),
    /// The width of the main trace does not match the expected width.
    #[error("main width mismatch: {0} != {1}")]
    MainWidthMismatch(usize, usize),
}

impl<C: MachineConfig, A: MachineAir<C::F>> ShardVerifier<C, A> {
    /// Get a shard verifier from a jagged pcs verifier.
    pub fn new(pcs_verifier: JaggedPcsVerifier<C>, machine: Machine<C::F, A>) -> Self {
        Self { pcs_verifier, machine }
    }

    /// Get the maximum log row count.
    #[must_use]
    #[inline]
    pub fn max_log_row_count(&self) -> usize {
        self.pcs_verifier.max_log_row_count
    }

    /// Get the machine.
    #[must_use]
    #[inline]
    pub fn machine(&self) -> &Machine<C::F, A> {
        &self.machine
    }

    /// Get the log stacking height.
    #[must_use]
    #[inline]
    pub fn log_stacking_height(&self) -> u32 {
        self.pcs_verifier.stacked_pcs_verifier.log_stacking_height
    }

    /// Get a new challenger.
    #[must_use]
    #[inline]
    pub fn challenger(&self) -> C::Challenger {
        self.pcs_verifier.challenger()
    }

    /// Get the shape of a shard proof.
    pub fn shape_from_proof(&self, proof: &ShardProof<C>) -> CoreProofShape<C::F, A> {
        let shard_chips = self
            .machine()
            .chips()
            .iter()
            .filter(|air| proof.shard_chips.contains(&air.name()))
            .cloned()
            .collect::<BTreeSet<_>>();
        debug_assert_eq!(shard_chips.len(), proof.shard_chips.len());

        let preprocessed_multiple =
            proof.evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[0].round_evaluations
                [0]
            .num_polynomials();
        let main_multiple = proof.evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[1]
            .round_evaluations[0]
            .num_polynomials();

        let main_padding_cols = proof.evaluation_proof.added_columns[1];
        let preprocessed_padding_cols = proof.evaluation_proof.added_columns[0];

        CoreProofShape {
            shard_chips,
            preprocessed_multiple,
            main_multiple,
            preprocessed_padding_cols,
            main_padding_cols,
        }
    }

    /// Compute the padded row adjustment for a chip.
    pub fn compute_padded_row_adjustment(
        chip: &Chip<C::F, A>,
        alpha: C::EF,
        public_values: &[C::F],
    ) -> C::EF
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
    {
        let dummy_preprocessed_trace = vec![C::EF::zero(); chip.preprocessed_width()];
        let dummy_main_trace = vec![C::EF::zero(); chip.width()];

        let mut folder = VerifierConstraintFolder::<C> {
            preprocessed: RowMajorMatrixView::new_row(&dummy_preprocessed_trace),
            main: RowMajorMatrixView::new_row(&dummy_main_trace),
            alpha,
            accumulator: C::EF::zero(),
            public_values,
            _marker: PhantomData,
        };

        chip.eval(&mut folder);

        folder.accumulator
    }

    /// Evaluates the constraints for a chip and opening.
    pub fn eval_constraints(
        chip: &Chip<C::F, A>,
        opening: &ChipOpenedValues<C::F, C::EF>,
        alpha: C::EF,
        public_values: &[C::F],
    ) -> C::EF
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
    {
        let mut folder = VerifierConstraintFolder::<C> {
            preprocessed: RowMajorMatrixView::new_row(&opening.preprocessed.local),
            main: RowMajorMatrixView::new_row(&opening.main.local),
            alpha,
            accumulator: C::EF::zero(),
            public_values,
            _marker: PhantomData,
        };

        chip.eval(&mut folder);

        folder.accumulator
    }

    fn verify_opening_shape(
        chip: &Chip<C::F, A>,
        opening: &ChipOpenedValues<C::F, C::EF>,
    ) -> Result<(), OpeningShapeError> {
        // Verify that the preprocessed width matches the expected value for the chip.
        if opening.preprocessed.local.len() != chip.preprocessed_width() {
            return Err(OpeningShapeError::PreprocessedWidthMismatch(
                chip.preprocessed_width(),
                opening.preprocessed.local.len(),
            ));
        }

        // Verify that the main width matches the expected value for the chip.
        if opening.main.local.len() != chip.width() {
            return Err(OpeningShapeError::MainWidthMismatch(
                chip.width(),
                opening.main.local.len(),
            ));
        }

        Ok(())
    }
}

impl<C: MachineConfig, A: MachineAir<C::F>> ShardVerifier<C, A>
where
    C::F: PrimeField32,
{
    /// Verify the zerocheck proof.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_zerocheck(
        &self,
        shard_chips: &BTreeSet<Chip<C::F, A>>,
        opened_values: &ShardOpenedValues<C::F, C::EF>,
        gkr_evaluations: &LogUpEvaluations<C::EF>,
        proof: &ShardProof<C>,
        public_values: &[C::F],
        challenger: &mut C::Challenger,
    ) -> Result<
        (),
        ShardVerifierError<C::EF, <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError>,
    >
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
    {
        let max_log_row_count = self.pcs_verifier.max_log_row_count;

        // Get the random challenge to merge the constraints.
        let alpha = challenger.sample_ext_element::<C::EF>();

        let gkr_batch_open_challenge = challenger.sample_ext_element::<C::EF>();

        // Get the random lambda to RLC the zerocheck polynomials.
        let lambda = challenger.sample_ext_element::<C::EF>();

        if gkr_evaluations.point.dimension() != max_log_row_count
            || proof.zerocheck_proof.point_and_eval.0.dimension() != max_log_row_count
        {
            return Err(ShardVerifierError::InvalidShape);
        }

        // Get the value of eq(zeta, sumcheck's reduced point).
        let zerocheck_eq_val = Mle::full_lagrange_eval(
            &gkr_evaluations.point,
            &proof.zerocheck_proof.point_and_eval.0,
        );
        let zerocheck_eq_vals = vec![zerocheck_eq_val; shard_chips.len()];

        // To verify the constraints, we need to check that the RLC'ed reduced eval in the zerocheck
        // proof is correct.
        let mut rlc_eval = C::EF::zero();
        for ((chip, (_, openings)), zerocheck_eq_val) in
            shard_chips.iter().zip_eq(opened_values.chips.iter()).zip_eq(zerocheck_eq_vals)
        {
            // Verify the shape of the opening arguments matches the expected values.
            Self::verify_opening_shape(chip, openings)?;

            let mut point_extended = proof.zerocheck_proof.point_and_eval.0.clone();
            point_extended.add_dimension(C::EF::zero());
            for &x in openings.degree.iter() {
                if x * (x - C::F::one()) != C::F::zero() {
                    return Err(ShardVerifierError::InvalidHeightBitDecomposition);
                }
            }
            for &x in openings.degree.iter().skip(1) {
                if x * *openings.degree.first().unwrap() != C::F::zero() {
                    return Err(ShardVerifierError::HeightTooLarge);
                }
            }

            let geq_val = full_geq(&openings.degree, &point_extended);

            let padded_row_adjustment =
                Self::compute_padded_row_adjustment(chip, alpha, public_values);

            let constraint_eval = Self::eval_constraints(chip, openings, alpha, public_values)
                - padded_row_adjustment * geq_val;

            let openings_batch = openings
                .main
                .local
                .iter()
                .chain(openings.preprocessed.local.iter())
                .copied()
                .zip(gkr_batch_open_challenge.powers().skip(1))
                .map(|(opening, power)| opening * power)
                .sum::<C::EF>();

            // Horner's method.
            rlc_eval = rlc_eval * lambda + zerocheck_eq_val * (constraint_eval + openings_batch);
        }

        if proof.zerocheck_proof.point_and_eval.1 != rlc_eval {
            return Err(ShardVerifierError::<
                _,
                <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError,
            >::ConstraintsCheckFailed(SumcheckError::InconsistencyWithEval));
        }

        let zerocheck_sum_modifications_from_gkr = gkr_evaluations
            .chip_openings
            .values()
            .map(|chip_evaluation| {
                chip_evaluation
                    .main_trace_evaluations
                    .deref()
                    .iter()
                    .copied()
                    .chain(
                        chip_evaluation
                            .preprocessed_trace_evaluations
                            .as_ref()
                            .iter()
                            .flat_map(|&evals| evals.deref().iter().copied()),
                    )
                    .zip(gkr_batch_open_challenge.powers().skip(1))
                    .map(|(opening, power)| opening * power)
                    .sum::<C::EF>()
            })
            .collect::<Vec<_>>();

        let zerocheck_sum_modification = zerocheck_sum_modifications_from_gkr
            .iter()
            .fold(C::EF::zero(), |acc, modification| lambda * acc + *modification);

        // Verify that the rlc claim matches the random linear combination of evaluation claims from
        // gkr.
        if proof.zerocheck_proof.claimed_sum != zerocheck_sum_modification {
            return Err(ShardVerifierError::<
                _,
                <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError,
            >::ConstraintsCheckFailed(
                SumcheckError::InconsistencyWithClaimedSum
            ));
        }

        // Verify the zerocheck proof.
        partially_verify_sumcheck_proof(&proof.zerocheck_proof, challenger, max_log_row_count, 4)
            .map_err(|e| {
            ShardVerifierError::<
                _,
                <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError,
            >::ConstraintsCheckFailed(e)
        })?;

        // Observe the openings
        for (_, opening) in opened_values.chips.iter() {
            for eval in opening.preprocessed.local.iter() {
                challenger.observe_ext_element(*eval);
            }
            for eval in opening.main.local.iter() {
                challenger.observe_ext_element(*eval);
            }
        }

        Ok(())
    }

    /// Verify the public values satisfy the required constraints, and return the cumulative sum.
    pub fn verify_public_values(
        &self,
        challenge: C::EF,
        alpha: C::EF,
        beta: C::EF,
        public_values: &[C::F],
    ) -> Result<C::EF, ShardVerifierConfigError<C>> {
        let mut folder = VerifierPublicValuesConstraintFolder::<C> {
            perm_challenges: &[alpha, beta],
            alpha: challenge,
            accumulator: C::EF::zero(),
            local_interaction_digest: C::EF::zero(),
            public_values,
            _marker: PhantomData,
        };
        A::Record::eval_public_values(&mut folder);
        if folder.accumulator == C::EF::zero() {
            Ok(folder.local_interaction_digest)
        } else {
            Err(ShardVerifierError::<
                _,
                <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError,
            >::InvalidPublicValues)
        }
    }

    /// Verify a shard proof.
    #[allow(clippy::too_many_lines)]
    pub fn verify_shard(
        &self,
        vk: &MachineVerifyingKey<C>,
        proof: &ShardProof<C>,
        challenger: &mut C::Challenger,
    ) -> Result<
        (),
        ShardVerifierError<C::EF, <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError>,
    >
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
    {
        let ShardProof {
            shard_chips,
            main_commitment,
            opened_values,
            evaluation_proof,
            zerocheck_proof,
            public_values,
            logup_gkr_proof,
        } = proof;

        let max_log_row_count = self.pcs_verifier.max_log_row_count;

        if public_values.len() < self.machine.num_pv_elts() {
            return Err(ShardVerifierError::InvalidPublicValues);
        }

        // Observe the public values.
        challenger.observe_slice(&public_values[0..self.machine.num_pv_elts()]);
        // Observe the main commitment.
        challenger.observe(main_commitment.clone());

        let mut heights: BTreeMap<String, C::F> = BTreeMap::new();
        for (name, chip_values) in opened_values.chips.iter() {
            if chip_values.degree.len() != max_log_row_count + 1 || chip_values.degree.len() >= 30 {
                return Err(ShardVerifierError::InvalidShape);
            }
            let acc = chip_values.degree.iter().fold(C::F::zero(), |acc, &x| x + C::F::two() * acc);
            heights.insert(name.clone(), acc);
            challenger.observe(acc);
        }

        for (chip, dimensions) in vk.preprocessed_chip_information.iter() {
            if let Some(height) = heights.get(chip) {
                if *height != dimensions.height {
                    return Err(ShardVerifierError::PreprocessedChipHeightMismatch(chip.clone()));
                }
            } else {
                return Err(ShardVerifierError::PreprocessedChipHeightMismatch(chip.clone()));
            }
        }

        let alpha = challenger.sample_ext_element::<C::EF>();
        let beta = challenger.sample_ext_element::<C::EF>();
        let pv_challenge = challenger.sample_ext_element::<C::EF>();

        let cumulative_sum =
            -self.verify_public_values(pv_challenge, alpha, beta, public_values)?;

        let shard_chips = self
            .machine
            .chips()
            .iter()
            .filter(|chip| shard_chips.contains(&chip.name()))
            .cloned()
            .collect::<BTreeSet<_>>();

        let degrees = opened_values
            .chips
            .iter()
            .map(|(name, x)| (name.clone(), x.degree.clone()))
            .collect::<BTreeMap<_, _>>();

        if shard_chips.len() != opened_values.chips.len() || shard_chips.len() != degrees.len() {
            return Err(ShardVerifierError::InvalidShape);
        }

        for (shard_chip, (chip_name, _)) in shard_chips.iter().zip_eq(opened_values.chips.iter()) {
            if shard_chip.name() != *chip_name {
                return Err(ShardVerifierError::InvalidChipOrder(
                    shard_chip.name(),
                    chip_name.to_string(),
                ));
            }
        }

        // Verify the logup GKR proof.
        LogUpGkrVerifier::<_, _, A>::verify_logup_gkr(
            &shard_chips,
            &degrees,
            alpha,
            beta,
            cumulative_sum,
            max_log_row_count,
            logup_gkr_proof,
            challenger,
        )
        .map_err(ShardVerifierError::GkrVerificationFailed)?;

        // Verify the zerocheck proof.
        self.verify_zerocheck(
            &shard_chips,
            opened_values,
            &logup_gkr_proof.logup_evaluations,
            proof,
            public_values,
            challenger,
        )?;

        // Verify the opening proof.
        // `preprocessed_openings_for_proof` is `Vec` of preprocessed `AirOpenedValues` of chips.
        // `main_openings_for_proof` is `Vec` of main `AirOpenedValues` of chips.
        let (preprocessed_openings_for_proof, main_openings_for_proof): (Vec<_>, Vec<_>) = proof
            .opened_values
            .chips
            .values()
            .map(|opening| (opening.preprocessed.clone(), opening.main.clone()))
            .unzip();

        // `preprocessed_openings` is the `Vec` of preprocessed openings of all chips.
        let preprocessed_openings = preprocessed_openings_for_proof
            .iter()
            .map(|x| x.local.iter().as_slice())
            .collect::<Vec<_>>();

        // `unfiltered_preprocessed_column_count` is the `Vec` of all preprocessed opening lengths.
        let unfiltered_preprocessed_column_count = preprocessed_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .collect::<Vec<_>>();

        // `main_openings` is the `Evaluations` derived by collecting all the main openings.
        let main_openings = main_openings_for_proof
            .iter()
            .map(|x| x.local.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        // `filtered_preprocessed_openings` is the `Evaluations` derived by collecting all the
        // non-empty preprocessed openings.
        let filtered_preprocessed_openings = preprocessed_openings
            .into_iter()
            .filter(|x| !x.is_empty())
            .map(|x| x.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        // `preprocessed_column_count` is the `Vec` of all non-zero preprocessed opening lengths.
        let preprocessed_column_count = filtered_preprocessed_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .collect::<Vec<_>>();

        // `main_column_count` is the `Vec` of all (non-zero) main opening lengths.
        let main_column_count =
            main_openings.iter().map(|table_openings| table_openings.len()).collect::<Vec<_>>();

        let max_added_column =
            (1usize << self.log_stacking_height()).div_ceil(1 << max_log_row_count);
        let added_columns = &evaluation_proof.added_columns;
        if added_columns.len() != 2 {
            return Err(ShardVerifierError::InvalidShape);
        }
        for &added_column in added_columns {
            if added_column == 0 || added_column > max_added_column {
                return Err(ShardVerifierError::InvalidAddedColumns);
            }
        }

        let column_counts = heights
            .values()
            .zip_eq(&unfiltered_preprocessed_column_count)
            .flat_map(|(&h, &c)| std::iter::repeat_n(h, c))
            // Add the preprocessed padding columns (except the last one)
            .chain(std::iter::repeat_n(
                C::F::from_canonical_u32(1 << max_log_row_count),
                added_columns[0] - 1,
            ))
            .chain(
                heights
                    .values()
                    .zip_eq(&main_column_count)
                    .flat_map(|(&h, &c)| std::iter::repeat_n(h, c)),
            )
            // Add the main padding columns (except the last one)
            .chain(std::iter::repeat_n(
                C::F::from_canonical_u32(1 << max_log_row_count),
                added_columns[1] - 1,
            ))
            .collect::<Vec<C::F>>();
        let preprocessed_column_count_total =
            unfiltered_preprocessed_column_count.iter().sum::<usize>();
        let main_column_count_total = main_column_count.iter().sum::<usize>();

        let col_prefix_sum = evaluation_proof
            .params
            .col_prefix_sums
            .iter()
            .map(|x| {
                if x.dimension() >= 31 {
                    return Err(ShardVerifierError::InvalidPrefixSum);
                }
                x.iter().try_fold(C::F::zero(), |acc, &y| {
                    if y != C::F::zero() && y != C::F::one() {
                        return Err(ShardVerifierError::InvalidPrefixSum);
                    }
                    Ok(y + C::F::two() * acc)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if evaluation_proof.params.col_prefix_sums.len()
            != 1 + preprocessed_column_count_total
                + main_column_count_total
                + added_columns[0]
                + added_columns[1]
        {
            return Err(ShardVerifierError::InvalidPrefixSum);
        }

        // Need to do special checks for the final columns of each commit, which are padding
        // columns of unknown height.
        let skip_indices =
            [preprocessed_column_count_total + added_columns[0] - 1, col_prefix_sum.len() - 2];

        let mut param_index = 0;
        for (i, (x, y)) in col_prefix_sum.iter().zip(col_prefix_sum.iter().skip(1)).enumerate() {
            if !skip_indices.contains(&i) {
                if *y != *x + column_counts[param_index] {
                    return Err(ShardVerifierError::InvalidPrefixSum);
                }
                param_index += 1;
            }
        }

        if col_prefix_sum[0] != C::F::zero() {
            return Err(ShardVerifierError::InvalidPrefixSum);
        }

        if proof.evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds.len() != 2 {
            return Err(ShardVerifierError::InvalidShape);
        }

        let preprocessed_round_num_poly =
            (proof.evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[0]
                .iter()
                .map(slop_multilinear::MleEval::num_polynomials)
                .sum::<usize>()) as u32;

        let main_round_num_poly =
            (proof.evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[1]
                .iter()
                .map(slop_multilinear::MleEval::num_polynomials)
                .sum::<usize>()) as u32;

        // Check that the prefix sum value at the first skip index is the correct multiple of the
        // stacking height.
        if col_prefix_sum[skip_indices[0] + 1]
            != C::F::from_canonical_u32(1 << self.log_stacking_height())
                * C::F::from_canonical_u32(preprocessed_round_num_poly)
            || preprocessed_round_num_poly.saturating_mul(1 << self.log_stacking_height())
                >= (1 << 30)
            || (col_prefix_sum[skip_indices[0] + 1] - col_prefix_sum[skip_indices[0]])
                .as_canonical_u32()
                > (1u32 << max_log_row_count)
        {
            return Err(ShardVerifierError::InvalidPrefixSum);
        }

        // Check that the prefix sum value at the second skip index is the correct multiple of the
        // stacking height (total padded trace area committed to).
        if col_prefix_sum[skip_indices[1] + 1]
            != C::F::from_canonical_u32(1 << self.log_stacking_height())
                * C::F::from_canonical_u32(preprocessed_round_num_poly + main_round_num_poly)
            || (preprocessed_round_num_poly.saturating_add(main_round_num_poly))
                .saturating_mul(1 << self.log_stacking_height())
                >= (1 << 30)
            || (col_prefix_sum[skip_indices[1] + 1] - col_prefix_sum[skip_indices[1]])
                .as_canonical_u32()
                > (1u32 << max_log_row_count)
        {
            return Err(ShardVerifierError::InvalidPrefixSum);
        }

        for skip_index in skip_indices {
            let padded_column_height = col_prefix_sum[skip_index + 1] - col_prefix_sum[skip_index];
            if padded_column_height.as_canonical_u32() > (1u32 << max_log_row_count) {
                return Err(ShardVerifierError::InvalidPrefixSum);
            }
        }

        let (commitments, column_counts, openings) = (
            vec![vk.preprocessed_commit.clone(), main_commitment.clone()],
            vec![preprocessed_column_count, main_column_count],
            Rounds { rounds: vec![filtered_preprocessed_openings, main_openings] },
        );

        let machine_jagged_verifier =
            MachineJaggedPcsVerifier::new(&self.pcs_verifier, column_counts);

        machine_jagged_verifier
            .verify_trusted_evaluations(
                &commitments,
                zerocheck_proof.point_and_eval.0.clone(),
                openings.as_slice(),
                evaluation_proof,
                challenger,
            )
            .map_err(ShardVerifierError::InvalidopeningArgument)?;

        Ok(())
    }
}

impl<BC, EC, A> ShardVerifier<JaggedBasefoldConfig<BC, EC>, A>
where
    A: MachineAir<BC::F>,
    BC: DefaultBasefoldConfig,
    BC::F: PrimeField32,
    BC::Commitment: std::fmt::Debug,
    EC: JaggedEvalConfig<BC::F, BC::EF, BC::Challenger> + std::fmt::Debug + Default,
{
    /// Create a shard verifier from basefold parameters.
    #[must_use]
    pub fn from_basefold_parameters(
        log_blowup: usize,
        log_stacking_height: u32,
        max_log_row_count: usize,
        machine: Machine<BC::F, A>,
    ) -> Self {
        let pcs_verifier =
            JaggedPcsVerifier::new(log_blowup, log_stacking_height, max_log_row_count);
        Self { pcs_verifier, machine }
    }
}

use std::marker::PhantomData;

use derive_where::derive_where;
use itertools::Itertools;
use slop_air::{Air, BaseAir};
use slop_algebra::AbstractField;
use slop_basefold::DefaultBasefoldConfig;
use slop_challenger::{CanObserve, FieldChallenger, Synchronizable};
use slop_commit::Rounds;
use slop_jagged::{
    JaggedBasefoldConfig, JaggedEvalConfig, JaggedPcsVerifier, JaggedPcsVerifierError,
    MachineJaggedPcsVerifier,
};
use slop_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
use slop_multilinear::{full_geq, Evaluations, Mle, MleEval, Point};
use slop_sumcheck::{partially_verify_sumcheck_proof, SumcheckError};
use thiserror::Error;

use crate::{
    air::MachineAir, septic_digest::SepticDigest, verify_permutation_gkr_proof, Chip,
    ChipOpenedValues, GkrVerificationResult, LogupGkrVerificationError, Machine,
    VerifierConstraintFolder,
};

use super::{MachineConfig, MachineVerifyingKey, ShardOpenedValues, ShardProof};

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
pub enum ShardVerifierError<C: MachineConfig> {
    /// The pcs opening proof is invalid.
    #[error("invalid pcs opening proof: {0}")]
    InvalidopeningArgument(JaggedPcsVerifierError<C>),
    /// The constraints check failed.
    #[error("constraints check failed: {0}")]
    ConstraintsCheckFailed(SumcheckError),
    /// The cumulative sums error.
    #[error("cumulative sums error: {0}")]
    CumulativeSumsError(&'static str),
    /// The preprocessed chip id mismatch.
    #[error("preprocessed chip id mismatch: {0}")]
    PreprocessedChipIdMismatch(String, String),
    /// The chip opening length mismatch.
    #[error("chip opening length mismatch")]
    ChipOpeningLengthMismatch,
    /// The cpu chip is missing.
    #[error("missing cpu chip")]
    MissingCpuChip,
    /// The shape of the openings does not match the expected shape.
    #[error("opening shape mismatch: {0}")]
    OpeningShapeMismatch(#[from] OpeningShapeError),
    /// The GKR Proof Fails
    #[error("GKR proof Failed: {0}")]
    GkrProofFailed(LogupGkrVerificationError),
}

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

impl<C: MachineConfig, A: MachineAir<C::F>> ShardVerifier<C, A>
where
    A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
{
    /// Get a shard verifier from a jagged pcs verifier.
    pub fn new(pcs_verifier: JaggedPcsVerifier<C>, machine: Machine<C::F, A>) -> Self {
        Self { pcs_verifier, machine }
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

        let default_challenge = C::EF::default();
        let default_septic_digest = SepticDigest::<C::F>::default();

        let mut folder = VerifierConstraintFolder::<C> {
            preprocessed: VerticalPair::new(
                RowMajorMatrixView::new_row(&dummy_preprocessed_trace),
                RowMajorMatrixView::new_row(&dummy_preprocessed_trace),
            ),
            main: VerticalPair::new(
                RowMajorMatrixView::new_row(&dummy_main_trace),
                RowMajorMatrixView::new_row(&dummy_main_trace),
            ),
            perm: VerticalPair::new(
                RowMajorMatrixView::new_row(&[]),
                RowMajorMatrixView::new_row(&[]),
            ),
            perm_challenges: &[],
            local_cumulative_sum: &default_challenge,
            global_cumulative_sum: &default_septic_digest,
            is_first_row: default_challenge,
            is_last_row: default_challenge,
            is_transition: default_challenge,
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
        let default_challenge = C::EF::default();
        let default_septic_digest = SepticDigest::<C::F>::default();

        let mut folder = VerifierConstraintFolder::<C> {
            preprocessed: VerticalPair::new(
                RowMajorMatrixView::new_row(&opening.preprocessed.local),
                RowMajorMatrixView::new_row(&opening.preprocessed.local),
            ),
            main: VerticalPair::new(
                RowMajorMatrixView::new_row(&opening.main.local),
                RowMajorMatrixView::new_row(&opening.main.local),
            ),
            perm: VerticalPair::new(
                RowMajorMatrixView::new_row(&[]),
                RowMajorMatrixView::new_row(&[]),
            ),
            perm_challenges: &[],
            local_cumulative_sum: &default_challenge,
            global_cumulative_sum: &default_septic_digest,
            is_first_row: default_challenge,
            is_last_row: default_challenge,
            is_transition: default_challenge,
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

    /// Verify the zerocheck proof.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_zerocheck(
        &self,
        challenger: &mut C::Challenger,
        opened_values: &ShardOpenedValues<C::F, C::EF>,
        gkr_points: &[Point<C::EF>],
        proof: &ShardProof<C>,
        gkr_column_openings: &[(MleEval<C::EF>, Option<MleEval<C::EF>>)],
        public_values: &[C::F],
    ) -> Result<(), ShardVerifierError<C>> {
        // Get the random challenge to merge the constraints.
        let alpha = challenger.sample_ext_element::<C::EF>();

        let gkr_batch_open_challenge = challenger.sample_ext_element::<C::EF>();

        // Get the random lambda to RLC the zerocheck polynomials.
        let lambda = challenger.sample_ext_element::<C::EF>();

        // Get the value of eq(zeta, sumcheck's reduced point).
        let zerocheck_eq_vals: Vec<C::EF> = gkr_points
            .iter()
            .map(|zeta| Mle::full_lagrange_eval(zeta, &proof.zerocheck_proof.point_and_eval.0))
            .collect();

        // To verify the constraints, we need to check that the RLC'ed reduced eval in the zerocheck
        // proof is correct.
        let mut rlc_eval = C::EF::zero();
        let max_log_row_count = self.pcs_verifier.max_log_row_count;
        for ((chip, openings), zerocheck_eq_val) in
            self.machine.chips().iter().zip_eq(opened_values.chips.iter()).zip_eq(zerocheck_eq_vals)
        {
            // Verify the shape of the opening arguments matches the expected values.
            Self::verify_opening_shape(chip, openings)?;

            let dimension = proof.zerocheck_proof.point_and_eval.0.dimension();

            assert_eq!(dimension, max_log_row_count);

            let mut point_extended = proof.zerocheck_proof.point_and_eval.0.clone();
            point_extended.add_dimension(C::EF::zero());
            openings.degree.iter().for_each(|x| {
                assert_eq!(*x * (*x - C::EF::one()), C::EF::zero());
            });
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
                .zip(gkr_batch_open_challenge.powers())
                .map(|(opening, power)| opening * power)
                .sum::<C::EF>();

            // Horner's method.
            rlc_eval = rlc_eval * lambda + zerocheck_eq_val * (constraint_eval + openings_batch);
        }

        if proof.zerocheck_proof.point_and_eval.1 != rlc_eval {
            return Err(ShardVerifierError::ConstraintsCheckFailed(
                SumcheckError::InconsistencyWithEval,
            ));
        }

        let zerocheck_sum_modifications_from_gkr = gkr_column_openings
            .iter()
            .map(|(main_openings, preprocessed_openings)| {
                main_openings
                    .to_vec()
                    .iter()
                    .chain(
                        preprocessed_openings
                            .as_ref()
                            .map(MleEval::to_vec)
                            .unwrap_or_default()
                            .iter(),
                    )
                    .zip(gkr_batch_open_challenge.powers())
                    .map(|(opening, power)| *opening * power)
                    .sum::<C::EF>()
            })
            .collect::<Vec<_>>();

        let zerocheck_sum_modification = zerocheck_sum_modifications_from_gkr
            .iter()
            .fold(C::EF::zero(), |acc, modification| lambda * acc + *modification);

        // Verify that the rlc claim matches the random linear combination of evaluation claims from
        // gkr.
        if proof.zerocheck_proof.claimed_sum != zerocheck_sum_modification {
            return Err(ShardVerifierError::ConstraintsCheckFailed(
                SumcheckError::InconsistencyWithClaimedSum,
            ));
        }

        // Verify the zerocheck proof.
        partially_verify_sumcheck_proof(&proof.zerocheck_proof, challenger)
            .map_err(|e| ShardVerifierError::ConstraintsCheckFailed(e))?;

        Ok(())
    }

    /// Verify a shard proof.
    #[allow(clippy::too_many_lines)]
    pub fn verify_shard(
        &self,
        vk: &MachineVerifyingKey<C>,
        proof: &ShardProof<C>,
        challenger: &mut C::Challenger,
    ) -> Result<(), ShardVerifierError<C>>
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
        C::Challenger: Synchronizable,
    {
        let ShardProof {
            main_commitment,
            opened_values,
            gkr_proofs,
            evaluation_proof,
            zerocheck_proof,
            public_values,
            ..
        } = proof;
        // Observe the public values.
        challenger.observe_slice(&public_values[0..self.machine.num_pv_elts()]);
        // Observe the main commitment.
        challenger.observe(main_commitment.clone());

        let alpha = challenger.sample_ext_element::<C::EF>();
        let beta = challenger.sample_ext_element::<C::EF>();

        let mut cumulative_sum = C::EF::zero();

        let mut challengers = Vec::new();
        let mut gkr_points = Vec::new();

        for ((chip, gkr_proof), openings) in
            self.machine.chips().iter().zip_eq(gkr_proofs.iter()).zip_eq(opened_values.chips.iter())
        {
            let mut challenger_clone = challenger.clone();
            let GkrVerificationResult { eval_point: point, challenger: new_challenger } =
                verify_permutation_gkr_proof::<C>(
                    gkr_proof,
                    &mut challenger_clone,
                    chip.sends(),
                    chip.receives(),
                    (alpha, beta),
                    &openings.degree,
                )
                .map_err(ShardVerifierError::<C>::GkrProofFailed)?;

            cumulative_sum += gkr_proof
                .numerator_claims
                .iter()
                .copied()
                .zip(gkr_proof.denom_claims.iter().copied())
                .map(|(num, den)| num / den)
                .sum::<C::EF>();

            challengers.push(new_challenger);
            gkr_points.push(point);
        }

        if cumulative_sum != C::EF::zero() {
            return Err(ShardVerifierError::CumulativeSumsError(
                "local cumulative sum is not zero",
            ));
        }

        let mut synchronized_challenger = C::Challenger::synchronize_challengers(challengers);

        let challenger = &mut synchronized_challenger;

        // Verify the zerocheck proof.
        self.verify_zerocheck(
            challenger,
            opened_values,
            &gkr_points,
            proof,
            &gkr_proofs
                .iter()
                .map(|gkr_proof| gkr_proof.column_openings.clone())
                .collect::<Vec<_>>(),
            public_values,
        )?;

        // Verify the opening proof.
        let (preprocessed_openings_for_proof, main_openings_for_proof): (Vec<_>, Vec<_>) = proof
            .opened_values
            .chips
            .iter()
            .map(|opening| (opening.preprocessed.clone(), opening.main.clone()))
            .unzip();

        let preprocessed_openings = preprocessed_openings_for_proof
            .iter()
            .map(|x| x.local.iter().as_slice())
            .collect::<Vec<_>>();

        let main_openings = main_openings_for_proof
            .iter()
            .map(|x| x.local.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        let filtered_preprocessed_openings = preprocessed_openings
            .into_iter()
            .filter(|x| !x.is_empty())
            .map(|x| x.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        let preprocessed_column_count = filtered_preprocessed_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .collect::<Vec<_>>();

        let main_column_count =
            main_openings.iter().map(|table_openings| table_openings.len()).collect::<Vec<_>>();

        let only_has_main_commitment = vk.preprocessed_commit.is_none();

        let (commitments, column_counts, openings) = if only_has_main_commitment {
            (
                vec![main_commitment.clone()],
                vec![main_column_count],
                Rounds { rounds: vec![main_openings] },
            )
        } else {
            (
                vec![vk.preprocessed_commit.clone().unwrap(), main_commitment.clone()],
                vec![preprocessed_column_count, main_column_count],
                Rounds { rounds: vec![filtered_preprocessed_openings, main_openings] },
            )
        };
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
    EC: JaggedEvalConfig<BC::EF, BC::Challenger> + std::fmt::Debug + Default,
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

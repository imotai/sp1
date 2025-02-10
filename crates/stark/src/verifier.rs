use core::fmt::Display;
use std::{
    fmt::{Debug, Formatter},
    marker::PhantomData,
};

use itertools::Itertools;
use num_traits::cast::ToPrimitive;
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{AbstractField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::SymbolicAirBuilder;
use slop_jagged::MachineJaggedPcs;
use slop_multilinear::{full_geq, Mle, Point};
use slop_sumcheck::{partially_verify_sumcheck_proof, SumcheckError};

use super::{
    types::{ChipOpenedValues, ShardProof},
    OpeningError, StarkGenericConfig, StarkVerifyingKey, Val,
};
use crate::{air::MachineAir, septic_digest::SepticDigest, MachineChip, VerifierConstraintFolder};

/// A verifier for a collection of air chips.
pub struct Verifier<SC, A>(PhantomData<SC>, PhantomData<A>);

impl<SC: StarkGenericConfig, A: MachineAir<Val<SC>> + Air<SymbolicAirBuilder<Val<SC>>>>
    Verifier<SC, A>
{
    /// Verify a proof for a collection of air chips.
    #[allow(clippy::too_many_lines)]
    pub fn verify_shard(
        config: &SC,
        _vk: &StarkVerifyingKey<SC>,
        chips: &[&MachineChip<SC, A>],
        challenger: &mut SC::Challenger,
        proof: &ShardProof<SC>,
        num_pv_elts: usize,
    ) -> Result<(), VerificationError<SC>>
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        let pcs = config.verifier_pcs();

        let ShardProof { commitments, opened_values, zerocheck_proof, public_values, .. } = proof;

        if chips.len() != opened_values.chips.len() {
            return Err(VerificationError::ChipOpeningLengthMismatch);
        }

        // Assert that the byte multiplicities don't overflow.
        let mut max_byte_lookup_mult = 0u64;
        chips.iter().zip(opened_values.chips.iter()).for_each(|(chip, val)| {
            max_byte_lookup_mult = max_byte_lookup_mult
                .checked_add(
                    (chip.num_sent_byte_lookups() as u64)
                        .checked_mul(1u64.checked_shl(val.log_degree as u32).unwrap())
                        .unwrap(),
                )
                .unwrap();
        });

        assert!(
            max_byte_lookup_mult <= SC::Val::order().to_u64().unwrap(),
            "Byte multiplicities overflow"
        );

        challenger.observe_slice(commitments);
        challenger.observe_slice(&public_values[0..num_pv_elts]);

        // Get the random challenge to merge the constraints.
        let alpha = challenger.sample_ext_element::<SC::Challenge>();

        // Get the random point to evaluate the batched sumcheck polynomial.  It must evaluate to zero
        // for the constraint check to pass.
        let zeta = Point::<SC::Challenge>::new(
            (0..pcs.max_log_row_count).map(|_| challenger.sample()).collect::<Vec<_>>().into(),
        );

        // Get the random lambda to RLC the zerocheck polynomials.
        let lambda = challenger.sample_ext_element::<SC::Challenge>();

        // Get the value of eq(zeta, sumcheck's reduced point).
        let zerocheck_eq_val =
            Mle::full_lagrange_eval(&zeta, &proof.zerocheck_proof.point_and_eval.0);

        // First check that the RLC'ed reduced eval in the zerocheck proof is correct.
        let mut rlc_eval = SC::Challenge::zero();
        for (chip, openings) in chips.iter().zip_eq(opened_values.chips.iter()) {
            // Verify the shape of the opening arguments matches the expected values.
            Self::verify_opening_shape(chip, openings)
                .map_err(|e| VerificationError::OpeningShapeError(chip.name(), e))?;

            let dimension = proof.zerocheck_proof.point_and_eval.0.dimension();

            assert!(dimension >= openings.log_degree);

            let geq_val = if openings.log_degree < pcs.max_log_row_count {
                // Create the threshold point.  This should be the big-endian bit representation of
                // 2^openings.log_degree.
                let mut threshold_point_vals = vec![SC::Challenge::zero(); dimension];
                threshold_point_vals[pcs.max_log_row_count - openings.log_degree - 1] =
                    SC::Challenge::one();
                let threshold_point = Point::new(threshold_point_vals.into());
                full_geq(&threshold_point, &proof.zerocheck_proof.point_and_eval.0)
            } else {
                // If the log_degree is equal to the max_log_row_count, then there were no padded variables
                // used, so there is no need to adjust the constraint eval with the padded row adjustment.
                assert!(openings.log_degree == pcs.max_log_row_count);
                SC::Challenge::zero()
            };

            let padded_row_adjustment =
                Self::compute_padded_row_adjustment(chip, alpha, public_values);

            let constraint_eval = Self::eval_constraints(chip, openings, alpha, public_values)
                - padded_row_adjustment * geq_val;

            // Horner's method.
            rlc_eval = rlc_eval * lambda + zerocheck_eq_val * constraint_eval;
        }

        if proof.zerocheck_proof.point_and_eval.1 != rlc_eval {
            return Err(VerificationError::ConstraintsCheckFailed(
                SumcheckError::InconsistencyWithEval,
            ));
        }

        // Verify that the rlc claim is zero.
        if proof.zerocheck_proof.claimed_sum != SC::Challenge::zero() {
            return Err(VerificationError::ConstraintsCheckFailed(
                SumcheckError::InconsistencyWithClaimedSum,
            ));
        }

        // Verify the zerocheck proof.
        partially_verify_sumcheck_proof(&proof.zerocheck_proof, challenger)
            .map_err(|e| VerificationError::ConstraintsCheckFailed(e))?;

        // Verify the opening proof.
        let opening_proof = &proof.opening_proof;
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

        let main_openings =
            main_openings_for_proof.iter().map(|x| x.local.iter().as_slice()).collect::<Vec<_>>();

        let filtered_preprocessed_openings =
            preprocessed_openings.into_iter().filter(|x| !x.is_empty()).collect::<Vec<_>>();

        let preprocessed_column_count = filtered_preprocessed_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .collect::<Vec<_>>();

        let main_column_count =
            main_openings.iter().map(|table_openings| table_openings.len()).collect::<Vec<_>>();

        let only_has_main_commitment = commitments.len() == 1;

        let (column_counts, openings) = if only_has_main_commitment {
            (vec![main_column_count], vec![main_openings.as_slice()])
        } else {
            (
                vec![preprocessed_column_count, main_column_count],
                vec![filtered_preprocessed_openings.as_slice(), main_openings.as_slice()],
            )
        };

        let machine_pcs = MachineJaggedPcs::new(pcs, column_counts);

        machine_pcs
            .verify_trusted_evaluations(
                zerocheck_proof.point_and_eval.0.clone(),
                openings.as_slice(),
                commitments,
                opening_proof,
                challenger,
            )
            .map_err(|e| VerificationError::InvalidopeningArgument(e))?;

        Ok(())
    }

    fn verify_opening_shape(
        chip: &MachineChip<SC, A>,
        opening: &ChipOpenedValues<Val<SC>, SC::Challenge>,
    ) -> Result<(), OpeningShapeError> {
        // Verify that the preprocessed width matches the expected value for the chip.
        if opening.preprocessed.local.len() != chip.preprocessed_width() {
            return Err(OpeningShapeError::PreprocessedWidthMismatch(
                chip.preprocessed_width(),
                opening.preprocessed.local.len(),
            ));
        }
        // if opening.preprocessed.next.len() != chip.preprocessed_width() {
        //     return Err(OpeningShapeError::PreprocessedWidthMismatch(
        //         chip.preprocessed_width(),
        //         opening.preprocessed.next.len(),
        //     ));
        // }

        // Verify that the main width matches the expected value for the chip.
        if opening.main.local.len() != chip.width() {
            return Err(OpeningShapeError::MainWidthMismatch(
                chip.width(),
                opening.main.local.len(),
            ));
        }
        // if opening.main.next.len() != chip.width() {
        //     return Err(OpeningShapeError::MainWidthMismatch(
        //         chip.width(),
        //         opening.main.next.len(),
        //     ));
        // }

        Ok(())
    }

    /// Compute the padded row adjustment for a chip.
    pub fn compute_padded_row_adjustment(
        chip: &MachineChip<SC, A>,
        alpha: SC::Challenge,
        public_values: &[Val<SC>],
    ) -> SC::Challenge
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        let dummy_preprocessed_trace = vec![SC::Challenge::zero(); chip.preprocessed_width()];
        let dummy_main_trace = vec![SC::Challenge::zero(); chip.width()];

        let default_challenge = SC::Challenge::default();
        let default_septic_digest = SepticDigest::<SC::Val>::default();

        let mut folder = VerifierConstraintFolder::<SC> {
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
            accumulator: SC::Challenge::zero(),
            public_values,
            _marker: PhantomData,
        };

        chip.eval(&mut folder);

        folder.accumulator
    }

    /// Evaluates the constraints for a chip and opening.
    pub fn eval_constraints(
        chip: &MachineChip<SC, A>,
        opening: &ChipOpenedValues<Val<SC>, SC::Challenge>,
        alpha: SC::Challenge,
        public_values: &[Val<SC>],
    ) -> SC::Challenge
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        let default_challenge = SC::Challenge::default();
        let default_septic_digest = SepticDigest::<SC::Val>::default();

        let mut folder = VerifierConstraintFolder::<SC> {
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
            accumulator: SC::Challenge::zero(),
            public_values,
            _marker: PhantomData,
        };

        chip.eval(&mut folder);

        folder.accumulator
    }
}

/// An error that occurs when the shape of the openings does not match the expected shape.
pub enum OpeningShapeError {
    /// The width of the preprocessed trace does not match the expected width.
    PreprocessedWidthMismatch(usize, usize),
    /// The width of the main trace does not match the expected width.
    MainWidthMismatch(usize, usize),
    /// The width of the permutation trace does not match the expected width.
    PermutationWidthMismatch(usize, usize),
    /// The width of the quotient trace does not match the expected width.
    QuotientWidthMismatch(usize, usize),
    /// The chunk size of the quotient trace does not match the expected chunk size.
    QuotientChunkSizeMismatch(usize, usize),
}

/// An error that occurs during the verification.
pub enum VerificationError<SC: StarkGenericConfig> {
    /// Opening proof is invalid.
    InvalidopeningArgument(OpeningError<SC>),
    /// The constraints check failed.
    ConstraintsCheckFailed(SumcheckError),
    /// The shape of the opening arguments is invalid.
    OpeningShapeError(String, OpeningShapeError),
    /// The cpu chip is missing.
    MissingCpuChip,
    /// The length of the chip opening does not match the expected length.
    ChipOpeningLengthMismatch,
    /// The preprocessed chip id does not match the claimed opening id.
    PreprocessedChipIdMismatch(String, String),
    /// Cumulative sums error
    CumulativeSumsError(&'static str),
}

impl Debug for OpeningShapeError {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            OpeningShapeError::PreprocessedWidthMismatch(expected, actual) => {
                write!(f, "Preprocessed width mismatch: expected {}, got {}", expected, actual)
            }
            OpeningShapeError::MainWidthMismatch(expected, actual) => {
                write!(f, "Main width mismatch: expected {}, got {}", expected, actual)
            }
            OpeningShapeError::PermutationWidthMismatch(expected, actual) => {
                write!(f, "Permutation width mismatch: expected {}, got {}", expected, actual)
            }
            OpeningShapeError::QuotientWidthMismatch(expected, actual) => {
                write!(f, "Quotient width mismatch: expected {}, got {}", expected, actual)
            }
            OpeningShapeError::QuotientChunkSizeMismatch(expected, actual) => {
                write!(f, "Quotient chunk size mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl Display for OpeningShapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<SC: StarkGenericConfig> Debug for VerificationError<SC> {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            VerificationError::InvalidopeningArgument(e) => {
                write!(f, "Invalid opening argument: {:?}", e)
            }
            VerificationError::ConstraintsCheckFailed(chip) => {
                write!(f, "Constraints check failed on chip {}", chip)
            }
            VerificationError::OpeningShapeError(chip, e) => {
                write!(f, "Invalid opening shape for chip {}: {:?}", chip, e)
            }
            VerificationError::MissingCpuChip => {
                write!(f, "Missing CPU chip")
            }
            VerificationError::ChipOpeningLengthMismatch => {
                write!(f, "Chip opening length mismatch")
            }
            VerificationError::PreprocessedChipIdMismatch(expected, actual) => {
                write!(f, "Preprocessed chip id mismatch: expected {}, got {}", expected, actual)
            }
            VerificationError::CumulativeSumsError(s) => write!(f, "cumulative sums error: {}", s),
        }
    }
}

impl<SC: StarkGenericConfig> Display for VerificationError<SC> {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            VerificationError::InvalidopeningArgument(_) => {
                write!(f, "Invalid opening argument")
            }
            VerificationError::ConstraintsCheckFailed(chip) => {
                write!(f, "Constraints check failed on chip {}", chip)
            }
            VerificationError::OpeningShapeError(chip, e) => {
                write!(f, "Invalid opening shape for chip {}: {}", chip, e)
            }
            VerificationError::MissingCpuChip => {
                write!(f, "Missing CPU chip in shard")
            }
            VerificationError::ChipOpeningLengthMismatch => {
                write!(f, "Chip opening length mismatch")
            }
            VerificationError::CumulativeSumsError(s) => write!(f, "cumulative sums error: {}", s),
            VerificationError::PreprocessedChipIdMismatch(expected, actual) => {
                write!(f, "Preprocessed chip id mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl<SC: StarkGenericConfig> std::error::Error for VerificationError<SC> {}

use derive_where::derive_where;
use itertools::Itertools;
use slop_algebra::AbstractField;
use slop_basefold::DefaultBasefoldConfig;
use slop_challenger::{CanObserve, CanSample, FieldChallenger};
use slop_commit::Rounds;
use slop_jagged::{
    JaggedBasefoldConfig, JaggedPcsVerifier, JaggedPcsVerifierError, MachineJaggedPcsVerifier,
};
use slop_multilinear::{full_geq, Evaluations, Mle, MleEval, MultilinearPcsChallenger, Point};
use slop_sumcheck::{partially_verify_sumcheck_proof, SumcheckError};
use thiserror::Error;

use crate::{air::MachineAir, Machine};

use super::{MachineConfig, MachineVerifyingKey, ShardProof};

#[derive_where(Clone)]
pub struct ShardVerifier<C: MachineConfig, A: MachineAir<C::F>> {
    pub pcs_verifier: JaggedPcsVerifier<C>,
    pub machine: Machine<C::F, A>,
}

#[derive(Debug, Error)]
pub enum ShardVerifierError<C: MachineConfig> {
    #[error("invalid pcs opening proof: {0}")]
    InvalidopeningArgument(JaggedPcsVerifierError<C>),
    #[error("constraints check failed: {0}")]
    ConstraintsCheckFailed(SumcheckError),
    #[error("cumulative sums error: {0}")]
    CumulativeSumsError(&'static str),
    #[error("preprocessed chip id mismatch: {0}")]
    PreprocessedChipIdMismatch(String, String),
    #[error("chip opening length mismatch")]
    ChipOpeningLengthMismatch,
    #[error("missing cpu chip")]
    MissingCpuChip,
}

// /// An error that occurs during the verification.
// pub enum VerificationError<SC: StarkGenericConfig> {
//     /// Opening proof is invalid.
//     InvalidopeningArgument(OpeningError<SC>),
//     /// The constraints check failed.
//     ConstraintsCheckFailed(SumcheckError),
//     /// The shape of the opening arguments is invalid.
//     OpeningShapeError(String, OpeningShapeError),
//     /// The cpu chip is missing.
//     MissingCpuChip,
//     /// The length of the chip opening does not match the expected length.
//     ChipOpeningLengthMismatch,
//     /// The preprocessed chip id does not match the claimed opening id.
//     PreprocessedChipIdMismatch(String, String),
//     /// Cumulative sums error
//     CumulativeSumsError(&'static str),
// }

impl<C: MachineConfig, A: MachineAir<C::F>> ShardVerifier<C, A> {
    /// Get a shard verifier from a jagged pcs verifier.
    pub fn new(pcs_verifier: JaggedPcsVerifier<C>, machine: Machine<C::F, A>) -> Self {
        Self { pcs_verifier, machine }
    }

    /// Verify a shard proof.
    pub fn verify_shard(
        &self,
        vk: &MachineVerifyingKey<C>,
        proof: &ShardProof<C>,
        challenger: &mut C::Challenger,
    ) -> Result<(), ShardVerifierError<C>> {
        let ShardProof {
            main_commitment,
            opened_values,
            evaluation_proof,
            zerocheck_proof,
            public_values,
        } = proof;
        // Observe the public values.
        challenger.observe_slice(&public_values[0..self.machine.num_pv_elts()]);
        // Observe the main commitment.
        challenger.observe(main_commitment.clone());

        // Get the random challenge to merge the constraints.
        let alpha = challenger.sample_ext_element::<C::EF>();

        // Get the random point to evaluate the batched sumcheck polynomial.  It must evaluate to zero
        // for the constraint check to pass.
        let zeta = challenger.sample_point::<C::EF>(self.pcs_verifier.max_log_row_count as u32);

        // Get the random lambda to RLC the zerocheck polynomials.
        let lambda = challenger.sample_ext_element::<C::EF>();

        // Get the value of eq(zeta, sumcheck's reduced point).
        let zerocheck_eq_val =
            Mle::full_lagrange_eval(&zeta, &proof.zerocheck_proof.point_and_eval.0);

        // // To verify the constraints, we need to check that the RLC'ed reduced eval in the zerocheck
        // // proof is correct.
        // let mut rlc_eval = C::EF::zero();
        // let max_log_row_count = self.pcs_verifier.max_log_row_count;
        // for (chip, openings) in self.machine.chips().iter().zip_eq(opened_values.chips.iter()) {
        //     // Verify the shape of the opening arguments matches the expected values.
        //     Self::verify_opening_shape(chip, openings)
        //         .map_err(|e| ShardVerifierError::OpeningShapeError(chip.name(), e))?;

        //     let dimension = proof.zerocheck_proof.point_and_eval.0.dimension();

        //     assert!(dimension >= openings.log_degree);

        //     let geq_val = if openings.log_degree < max_log_row_count {
        //         // Create the threshold point.  This should be the big-endian bit representation of
        //         // 2^openings.log_degree.
        //         let mut threshold_point_vals = vec![C::EF::zero(); dimension];
        //         threshold_point_vals[max_log_row_count - openings.log_degree - 1] = C::EF::one();
        //         let threshold_point = Point::new(threshold_point_vals.into());
        //         full_geq(&threshold_point, &proof.zerocheck_proof.point_and_eval.0)
        //     } else {
        //         // If the log_degree is equal to the max_log_row_count, then there were no padded variables
        //         // used, so there is no need to adjust the constraint eval with the padded row adjustment.
        //         assert!(openings.log_degree == max_log_row_count);
        //         C::EF::zero()
        //     };

        //     let padded_row_adjustment =
        //         Self::compute_padded_row_adjustment(chip, alpha, public_values);

        //     let constraint_eval = Self::eval_constraints(chip, openings, alpha, public_values)
        //         - padded_row_adjustment * geq_val;

        //     // Horner's method.
        //     rlc_eval = rlc_eval * lambda + zerocheck_eq_val * constraint_eval;
        // }

        // if proof.zerocheck_proof.point_and_eval.1 != rlc_eval {
        //     return Err(ShardVerifierError::ConstraintsCheckFailed(
        //         SumcheckError::InconsistencyWithEval,
        //     ));
        // }

        // // Verify that the rlc claim is zero.
        // if proof.zerocheck_proof.claimed_sum != C::EF::zero() {
        //     return Err(ShardVerifierError::ConstraintsCheckFailed(
        //         SumcheckError::InconsistencyWithClaimedSum,
        //     ));
        // }

        // Verify the zerocheck proof.
        partially_verify_sumcheck_proof(&proof.zerocheck_proof, challenger)
            .map_err(|e| ShardVerifierError::ConstraintsCheckFailed(e))?;

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

        let verify_pcs: bool =
            std::env::var("VERIFY_PCS").unwrap_or("true".to_string()).parse().unwrap();
        if verify_pcs {
            machine_jagged_verifier
                .verify_trusted_evaluations(
                    &commitments,
                    zerocheck_proof.point_and_eval.0.clone(),
                    openings.as_slice(),
                    &proof.evaluation_proof,
                    challenger,
                )
                .map_err(ShardVerifierError::InvalidopeningArgument)?;
        }

        Ok(())
    }
}

impl<BC, A> ShardVerifier<JaggedBasefoldConfig<BC>, A>
where
    A: MachineAir<BC::F>,
    BC: DefaultBasefoldConfig,
{
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

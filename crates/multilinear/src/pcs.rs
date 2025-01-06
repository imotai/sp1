use p3_challenger::FieldChallenger;
use p3_matrix::dense::RowMajorMatrix;
use slop_algebra::{ExtensionField, Field};

use crate::Point;

/// A trait representing a multilinear polynomial commitment scheme.
pub trait MultilinearPcsVerifier {
    type F: Field;
    type EF: ExtensionField<Self::F>;
    type Proof;
    type Commitment;
    type Error;
    type Challenger: FieldChallenger<Self::F>;

    fn verify_trusted_evaluations(
        &self,
        point: Point<Self::EF>,
        evaluation_claims: &[&[Self::EF]],
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error>;

    fn verify_untrusted_evaluations(
        &self,
        point: Point<Self::EF>,
        evaluation_claims: &[&[Self::EF]],
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        evaluation_claims.iter().for_each(|eval_set| {
            eval_set.iter().for_each(|eval| challenger.observe_ext_element(*eval))
        });
        self.verify_trusted_evaluations(point, evaluation_claims, commitment, proof, challenger)
    }
}

pub trait MultilinearPcsProver {
    type PCS: MultilinearPcsVerifier;
    type OpeningProof;
    type MultilinearProverData;
    type MultilinearCommitment;

    fn commit_multilinears(
        &self,
        data: Vec<RowMajorMatrix<<Self::PCS as MultilinearPcsVerifier>::F>>,
    ) -> (Self::MultilinearCommitment, Self::MultilinearProverData);

    fn prove_trusted_evaluations(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsVerifier>::EF>,
        expected_evals: &[&[<Self::PCS as MultilinearPcsVerifier>::EF]],
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsVerifier>::Challenger,
    ) -> Self::OpeningProof;

    fn prove_untrusted_evaluations(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsVerifier>::EF>,
        expected_evals: &[&[<Self::PCS as MultilinearPcsVerifier>::EF]],
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsVerifier>::Challenger,
    ) -> Self::OpeningProof {
        expected_evals.iter().for_each(|eval_set| {
            eval_set.iter().for_each(|eval| challenger.observe_ext_element(*eval))
        });
        self.prove_trusted_evaluations(eval_point, expected_evals, prover_data, challenger)
    }
}

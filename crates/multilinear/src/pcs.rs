use p3_matrix::dense::RowMajorMatrix;
use spl_algebra::{ExtensionField, Field};

use crate::Point;

/// A trait representing a multilinear polynomial commitment scheme.
pub trait MultilinearPcsVerifier<Challenger> {
    type F: Field;
    type EF: ExtensionField<Self::F>;
    type Proof;
    type Commitment;
    type Error;

    fn verify_evaluations(
        &self,
        point: Point<Self::EF>,
        evaluation_claims: &[Self::EF],
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}

pub trait MultilinearPcsProver<Challenger> {
    type PCS: MultilinearPcsVerifier<Challenger>;
    type OpeningProof;
    type MultilinearProverData;
    type MultilinearCommitment;

    fn commit_multilinears(
        &self,
        data: Vec<RowMajorMatrix<<Self::PCS as MultilinearPcsVerifier<Challenger>>::F>>,
    ) -> (Self::MultilinearCommitment, Self::MultilinearProverData);

    fn prove_evaluations(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsVerifier<Challenger>>::EF>,
        expected_evals: &[<Self::PCS as MultilinearPcsVerifier<Challenger>>::EF],
        prover_data: Self::MultilinearProverData,
        challenger: &mut Challenger,
    ) -> Self::OpeningProof;
}

use std::fmt::Debug;

use serde::{de::DeserializeOwned, Serialize};
use slop_algebra::{ExtensionField, Field};
use slop_challenger::FieldChallenger;
use slop_matrix::dense::RowMajorMatrix;

use crate::Point;

/// A trait representing a multilinear polynomial commitment scheme.
pub trait MultilinearPcsBatchVerifier {
    type F: Field;
    type EF: ExtensionField<Self::F>;
    type Proof: Serialize + DeserializeOwned + Clone;
    type Commitment: Clone + Serialize + DeserializeOwned;
    type Error: Debug;
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

/// A trait representing a multilinear polynomial commitment scheme.
pub trait MultilinearPcsVerifier {
    type F: Field;
    type EF: ExtensionField<Self::F>;
    type Proof;
    type Commitment: Clone + Serialize + DeserializeOwned;
    type Error: Debug;
    type Challenger: FieldChallenger<Self::F>;

    fn verify_trusted_evaluation(
        &self,
        point: Point<Self::EF>,
        evaluation_claim: Self::EF,
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error>;

    fn verify_untrusted_evaluation(
        &self,
        point: Point<Self::EF>,
        evaluation_claim: Self::EF,
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        challenger.observe_ext_element(evaluation_claim);
        self.verify_trusted_evaluation(point, evaluation_claim, commitment, proof, challenger)
    }
}

pub trait MultilinearPcsBatchProver {
    type PCS: MultilinearPcsBatchVerifier;
    type MultilinearProverData: Clone + Serialize + DeserializeOwned;

    fn commit_multilinears(
        &self,
        data: Vec<RowMajorMatrix<<Self::PCS as MultilinearPcsBatchVerifier>::F>>,
    ) -> (<Self::PCS as MultilinearPcsBatchVerifier>::Commitment, Self::MultilinearProverData);

    fn prove_trusted_evaluations(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsBatchVerifier>::EF>,
        expected_evals: &[&[<Self::PCS as MultilinearPcsBatchVerifier>::EF]],
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsBatchVerifier>::Challenger,
    ) -> <Self::PCS as MultilinearPcsBatchVerifier>::Proof;

    fn prove_untrusted_evaluations(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsBatchVerifier>::EF>,
        expected_evals: &[&[<Self::PCS as MultilinearPcsBatchVerifier>::EF]],
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsBatchVerifier>::Challenger,
    ) -> <Self::PCS as MultilinearPcsBatchVerifier>::Proof {
        expected_evals.iter().for_each(|eval_set| {
            eval_set.iter().for_each(|eval| challenger.observe_ext_element(*eval))
        });
        self.prove_trusted_evaluations(eval_point, expected_evals, prover_data, challenger)
    }
}

pub trait MultilinearPcsProver {
    type PCS: MultilinearPcsVerifier;
    type MultilinearProverData: Clone + Serialize + DeserializeOwned;

    fn commit_multilinear(
        &self,
        data: Vec<Vec<<Self::PCS as MultilinearPcsVerifier>::F>>,
    ) -> (<Self::PCS as MultilinearPcsVerifier>::Commitment, Self::MultilinearProverData);

    fn prove_trusted_evaluation(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsVerifier>::EF>,
        expected_eval: <Self::PCS as MultilinearPcsVerifier>::EF,
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsVerifier>::Challenger,
    ) -> <Self::PCS as MultilinearPcsVerifier>::Proof;

    fn prove_untrusted_evaluation(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsVerifier>::EF>,
        expected_eval: <Self::PCS as MultilinearPcsVerifier>::EF,
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsVerifier>::Challenger,
    ) -> <Self::PCS as MultilinearPcsVerifier>::Proof {
        challenger.observe_ext_element(expected_eval);
        self.prove_trusted_evaluation(eval_point, expected_eval, prover_data, challenger)
    }
}

/// A trait for prover data where the prover has keeps the matrices that were committed to.
pub trait MainTraceProverData<T> {
    type BaseProverData;
    fn split_off_main_traces(self) -> (Self::BaseProverData, Vec<T>);

    fn reconstitute(base_data: Self::BaseProverData, main_traces: Vec<T>) -> Self;
}

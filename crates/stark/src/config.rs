#![allow(missing_docs)]

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{ExtensionField, Field, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use serde::{de::DeserializeOwned, Serialize};
use slop_jagged::{JaggedPcs, JaggedPcsError};
use slop_multilinear::{
    MainTraceProverData, MultilinearPcsBatchProver, MultilinearPcsBatchVerifier, StackedPcsError,
    StackedPcsProver, StackedPcsVerifier,
};
use slop_sumcheck::SumcheckError;

pub type Val<SC> = <<SC as StarkGenericConfig>::MLPCSVerifier as MultilinearPcsBatchVerifier>::F;

pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

pub type Com<SC> =
    <<SC as StarkGenericConfig>::MLPCSVerifier as MultilinearPcsBatchVerifier>::Commitment;

pub type OpeningProof<SC> =
    <<SC as StarkGenericConfig>::MLPCSVerifier as MultilinearPcsBatchVerifier>::Proof;

// TODO:  This is too hard coded.
pub type OpeningError<SC> = JaggedPcsError<
    StackedPcsError<
        <<SC as StarkGenericConfig>::MLPCSVerifier as MultilinearPcsBatchVerifier>::Error,
    >,
    SumcheckError,
>;

pub type PcsProverData<SC> =
    <<SC as StarkGenericConfig>::MLPCSProver as MultilinearPcsBatchProver>::MultilinearProverData;

pub type Challenge<SC> = <SC as StarkGenericConfig>::Challenge;
pub type Challenger<SC> = <SC as StarkGenericConfig>::Challenger;

pub trait StarkGenericConfig: 'static + Send + Sync + Serialize + DeserializeOwned + Clone {
    type Val: PrimeField;

    type MLPCSProverData: Clone
        + Serialize
        + DeserializeOwned
        + MainTraceProverData<RowMajorMatrix<Self::Val>>;

    type MLPCSProver: MultilinearPcsBatchProver<
        PCS = Self::MLPCSVerifier,
        MultilinearProverData = Self::MLPCSProverData,
    >;

    type MLPCSVerifier: MultilinearPcsBatchVerifier<
        F = Self::Val,
        EF = Self::Challenge,
        Challenger = Self::Challenger,
    >;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Val>;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanObserve<<Self::MLPCSVerifier as MultilinearPcsBatchVerifier>::Commitment>
        + CanSample<Self::Challenge>
        + CanSample<Val<Self>>
        + Serialize
        + DeserializeOwned;

    /// Get the PCS used by this configuration.
    fn prover_pcs(&self) -> &JaggedPcs<StackedPcsProver<Self::MLPCSProver>>;

    /// Get the PCS used by this configuration.
    fn verifier_pcs(&self) -> &JaggedPcs<StackedPcsVerifier<Self::MLPCSVerifier>>;

    /// Initialize a new challenger.
    fn challenger(&self) -> Self::Challenger;
}

pub trait ZeroCommitment<SC: StarkGenericConfig> {
    fn zero_commitment(&self) -> Com<SC>;
}

// pub struct UniConfig<SC>(pub SC);

// impl<SC: StarkGenericConfig> p3_uni_stark::StarkGenericConfig for UniConfig<SC> {
//     type Pcs = SC::Pcs;

//     type Challenge = SC::Challenge;

//     type Challenger = SC::Challenger;

//     /// Get the PCS used by this configuration.
//     fn prover_pcs(&self) -> &Self::MLPCSProver;

//     /// Get the PCS used by this configuration.
//     fn verifier_pcs(&self) -> &Self::MLPCSVerifier;
// }

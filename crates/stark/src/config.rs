#![allow(missing_docs)]

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{ExtensionField, Field, PrimeField};
use serde::{de::DeserializeOwned, Serialize};
use slop_alloc::CpuBackend;
use slop_jagged::{
    JaggedConfig, JaggedPcsProof, JaggedPcsVerifier, JaggedPcsVerifierError, JaggedProver,
    JaggedProverComponents, JaggedProverData,
};

pub type Val<SC> = <SC as StarkGenericConfig>::Val;

pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

pub type Com<SC> = <SC as StarkGenericConfig>::Commitment;

pub type OpeningProof<SC> = JaggedPcsProof<<SC as StarkGenericConfig>::JaggedConfig>;

// TODO:  This is too hard coded.
pub type OpeningError<SC> = JaggedPcsVerifierError<<SC as StarkGenericConfig>::JaggedConfig>;

pub type PcsProverData<SC> = JaggedProverData<<SC as StarkGenericConfig>::CpuProverComponents>;

pub type Challenge<SC> = <SC as StarkGenericConfig>::Challenge;
pub type Challenger<SC> = <SC as StarkGenericConfig>::Challenger;

pub trait StarkGenericConfig: 'static + Send + Sync + Serialize + DeserializeOwned + Clone {
    type Val: PrimeField;

    type Commitment: 'static + Send + Sync + Serialize + DeserializeOwned + Clone;

    type JaggedConfig: JaggedConfig<
        F = Self::Val,
        EF = Self::Challenge,
        Challenger = Self::Challenger,
        Commitment = Self::Commitment,
    >;

    type CpuProverComponents: JaggedProverComponents<
        F = Self::Val,
        EF = Self::Challenge,
        Config = Self::JaggedConfig,
        Challenger = Self::Challenger,
        Commitment = Self::Commitment,
        A = CpuBackend,
    >;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Val>;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanSample<Self::Challenge>
        + CanObserve<Self::Commitment>
        + CanSample<Val<Self>>
        + Serialize
        + DeserializeOwned;

    /// Get the PCS used by this configuration.
    fn prover_pcs(&self) -> &JaggedProver<Self::CpuProverComponents>;

    /// Get the PCS used by this configuration.
    fn pcs_verifier(&self) -> &JaggedPcsVerifier<Self::JaggedConfig>;

    /// Initialize a new challenger.
    fn challenger(&self) -> Self::Challenger;
}

pub trait ZeroCommitment<SC: StarkGenericConfig> {
    fn zero_commitment(&self) -> Com<SC>;
}

use serde::{de::DeserializeOwned, Serialize};
use slop_algebra::{ExtensionField, Field};
use slop_challenger::FieldChallenger;
use slop_multilinear::MultilinearPcsVerifier;

use std::fmt::Debug;

pub trait JaggedConfig:
    'static + Clone + Debug + Send + Clone + Serialize + DeserializeOwned
{
    type F: Field;
    type EF: ExtensionField<Self::F>;

    type Commitment: 'static + Clone + Send + Sync + Serialize + DeserializeOwned;

    /// The challenger type that creates the random challenges via Fiat-Shamir.
    ///
    /// The challenger is observing all the messages sent throughout the protocol and uses this
    /// to create the verifier messages of the IOP.
    type Challenger: FieldChallenger<Self::F>;

    type BatchPcsProof: 'static + Clone + Send + Sync + Serialize + DeserializeOwned;

    type BatchPcsVerifier: MultilinearPcsVerifier<
        F = Self::F,
        EF = Self::EF,
        Challenger = Self::Challenger,
        Proof = Self::BatchPcsProof,
        Commitment = Self::Commitment,
    >;
}

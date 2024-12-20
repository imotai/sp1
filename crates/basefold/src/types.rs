use spl_multilinear::Mle;
use std::fmt::Debug;
use thiserror::Error;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_fri::{FriConfig, QueryProof};

use crate::FriError;
use spl_algebra::{ExtensionField, Field, TwoAdicField};

/// The data necessary to construct a BaseFold opening proof.
pub struct BaseFoldProof<K: Field, M: Mmcs<K>, Witness> {
    /// The univariate polynomials that are used in the sumcheck part of the BaseFold protocol.
    pub univariate_messages: Vec<[K; 2]>,

    /// The FRI parts of the proof.
    /// The commitments to the folded polynomials produced in the commit phase.
    pub commitments: Vec<M::Commitment>,

    /// The query openings and the FRI query proofs for the FRI query phase.
    pub query_phase_proofs: Vec<(K, QueryProof<K, M>)>,

    /// The prover performs FRI until we reach a polynomial of degree 0, and return the constant value of
    /// this polynomial.
    pub final_poly: K,

    /// Proof-of-work witness.
    pub pow_witness: Witness,
}

#[derive(Debug)]
pub struct BaseFoldPcs<K: Field, EK: ExtensionField<K>, InnerMmcs: Mmcs<K>, Challenger>
where
    Challenger: GrindingChallenger
        + FieldChallenger<K>
        + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>,
{
    pub(crate) fri_config: FriConfig<ExtensionMmcs<K, EK, InnerMmcs>>,
    pub(crate) inner_mmcs: InnerMmcs,
    _phantom: std::marker::PhantomData<(K, EK, Challenger)>,
}

impl<K: Field, EK: ExtensionField<K>, InnerMmcs: Mmcs<K>, Challenger>
    BaseFoldPcs<K, EK, InnerMmcs, Challenger>
where
    Challenger: GrindingChallenger
        + FieldChallenger<K>
        + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>,
{
    pub fn new(
        fri_config: FriConfig<ExtensionMmcs<K, EK, InnerMmcs>>,
        inner_mmcs: InnerMmcs,
    ) -> Self {
        Self { inner_mmcs, fri_config, _phantom: std::marker::PhantomData }
    }
}

#[derive(Debug, Error)]
pub enum BaseFoldError<MmcsError> {
    #[error("Fri Error")]
    Fri(#[from] FriError<MmcsError>),

    #[error("Sumcheck Error")]
    Sumcheck,

    #[error("Proof of work error")]
    Pow,

    #[error("Incorrect shape")]
    IncorrectShape,
}

pub struct BaseFoldProver<K: TwoAdicField, EK: ExtensionField<K>, InnerMmcs: Mmcs<K>, Challenger>
where
    Challenger: GrindingChallenger
        + FieldChallenger<K>
        + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>,
    <ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment: Debug,
{
    pub(crate) pcs: BaseFoldPcs<K, EK, InnerMmcs, Challenger>,
}

pub struct BaseFoldProverData<K, D> {
    pub(crate) vals: Mle<K>,
    pub(crate) data: D,
}

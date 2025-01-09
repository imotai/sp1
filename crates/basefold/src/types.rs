use std::fmt::Debug;
use thiserror::Error;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_fri::{FriConfig, QueryProof};

use crate::FriError;
use slop_algebra::{ExtensionField, Field};

pub type QueryResultWithProof<K, EK, M> = (EK, QueryProof<EK, ExtensionMmcs<K, EK, M>>);

/// The data necessary to construct a BaseFold opening proof.
pub struct BaseFoldProof<K: Field, EK: ExtensionField<K>, M: Mmcs<K>, Witness> {
    /// The univariate polynomials that are used in the sumcheck part of the BaseFold protocol.
    pub univariate_messages: Vec<[EK; 2]>,

    /// The FRI parts of the proof.
    /// The commitments to the folded polynomials produced in the commit phase.
    pub commitments: Vec<<ExtensionMmcs<K, EK, M> as Mmcs<EK>>::Commitment>,

    /// The query openings and the FRI query proofs for the FRI query phase.
    /// TODO: The EK component is probably redundant; remove it.
    pub query_phase_proofs: Vec<QueryResultWithProof<K, EK, M>>,

    /// The prover performs FRI until we reach a polynomial of degree 0, and return the constant value of
    /// this polynomial.
    pub final_poly: EK,

    /// Proof-of-work witness.
    pub pow_witness: Witness,

    /// The query openings of the individual polynmomials.
    pub query_openings: Vec<(Vec<Vec<K>>, M::Proof)>,
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

    pub fn fri_config(&self) -> &FriConfig<ExtensionMmcs<K, EK, InnerMmcs>> {
        &self.fri_config
    }

    pub fn inner_mmcs(&self) -> &InnerMmcs {
        &self.inner_mmcs
    }
}

#[derive(Debug, Error)]
pub enum BaseFoldError<MmcsError> {
    #[error("Fri Error")]
    Fri(#[from] FriError<MmcsError>),

    #[error("Mmcs Error")]
    Mmcs(#[from] MmcsError),

    #[error("Batching Error")]
    Batching,

    #[error("Sumcheck Error")]
    Sumcheck,

    #[error("Proof of work error")]
    Pow,

    #[error("Incorrect shape")]
    IncorrectShape,
}

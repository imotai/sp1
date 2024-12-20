use spl_multilinear::Mle;
use std::fmt::Debug;
use thiserror::Error;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_fri::{FriConfig, QueryProof};

use crate::FriError;
use spl_algebra::{Field, TwoAdicField};

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
pub struct BaseFoldPcs<K: Field, M: Mmcs<K>, Challenger>
where
    Challenger: GrindingChallenger + FieldChallenger<K> + CanObserve<M::Commitment>,
{
    pub(crate) fri_config: FriConfig<M>,
    _phantom: std::marker::PhantomData<(K, M, Challenger)>,
}

impl<K: Field, M: Mmcs<K>, Challenger> BaseFoldPcs<K, M, Challenger>
where
    Challenger: GrindingChallenger + FieldChallenger<K> + CanObserve<M::Commitment>,
{
    pub fn new(fri_config: FriConfig<M>) -> Self {
        Self { fri_config, _phantom: std::marker::PhantomData }
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

pub struct BaseFoldProver<K: TwoAdicField, M: Mmcs<K>, Challenger>
where
    Challenger: GrindingChallenger + FieldChallenger<K> + CanObserve<M::Commitment>,
    M::Commitment: Debug,
{
    pub(crate) pcs: BaseFoldPcs<K, M, Challenger>,
}

pub struct BaseFoldProverData<K, D> {
    pub(crate) vals: Mle<K>,
    pub(crate) data: D,
}

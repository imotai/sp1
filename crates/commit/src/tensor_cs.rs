use std::{borrow::Borrow, error::Error};

use slop_alloc::Backend;
use slop_tensor::Tensor;

/// An opening of a tensor commitment scheme.
pub struct TensorCsOpening<C: TensorCs> {
    /// The claimed values of the opening.
    pub values: Vec<Tensor<C::Data>>,
    /// The proof of the opening.
    pub proof: <C as TensorCs>::Proof,
}

/// Tensor commitment scheme.
///
/// A tensor commitment scheme is essentially a batch vector commitment scheme, where the latter
/// allows you to commit to a list of elements of type [Self::Data] and later provide a verifier
/// oracle access to a specific element at a specific index. In a Tensor commitment scheme, the
/// verifier oracle access is to a specific slice of the input tensor t[[.., i, ...]]. The prover
/// is free to choose the dimension along which the commitment is made.
///
/// As tensors are stored contiguously in memory, it is not always desirable to have all committed
/// data in a single tensor. Hence, a tensor commitment scheme assumes the prover commits as above
/// to a list of tensors of the same shape at a given order.
pub trait TensorCs: Sized {
    type Data: Clone + Send + Sync;
    type Commitment: Clone + Send + Sync;
    type Proof;
    type VerifierError: Error;

    /// Verify a batch of openings.
    ///
    /// The claimed valued tensors are assumed to be of shape [indices.len(), ..]. For each index,
    /// the collection of claimed values indexed at [index,...] is the data of the corresponding
    /// committed tensors at the given index.
    fn verify_tensor_openings(
        &self,
        commit: &Self::Commitment,
        indices: &[usize],
        opening: &TensorCsOpening<Self>,
    ) -> Result<(), Self::VerifierError>;
}

impl<C: TensorCs> TensorCsOpening<C> {
    #[inline]
    pub const fn new(values: Vec<Tensor<C::Data>>, proof: <C as TensorCs>::Proof) -> Self {
        Self { values, proof }
    }
}

pub trait TensorCsProver<A: Backend> {
    type Cs: TensorCs;
    type ProverData;
    type ProverError: Error;

    /// Commit to a batch of tensors of the same shape.
    ///
    /// The prover is free to choose which dimension index is supported.
    #[allow(clippy::type_complexity)]
    fn commit_tensors(
        &self,
        tensors: &[impl Borrow<Tensor<<Self::Cs as TensorCs>::Data, A>>],
    ) -> Result<(<Self::Cs as TensorCs>::Commitment, Self::ProverData), Self::ProverError>;

    /// Prove openings at a list of indices.
    fn prove_openings_at_indices(
        &self,
        data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<<Self::Cs as TensorCs>::Proof, Self::ProverError>;
}

pub trait ComputeTcsOpenings<A: Backend>: TensorCsProver<A> {
    fn compute_openings_at_indices(
        &self,
        tensors: &[impl Borrow<Tensor<<Self::Cs as TensorCs>::Data, A>>],
        indices: &[usize],
    ) -> Vec<Tensor<<Self::Cs as TensorCs>::Data, A>>;
}

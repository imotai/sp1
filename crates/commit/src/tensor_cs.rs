use std::{error::Error, fmt::Debug, future::Future};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_alloc::Backend;
use slop_futures::OwnedBorrow;
use slop_tensor::Tensor;

use crate::Message;

/// An opening of a tensor commitment scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TensorCsOpening<C: TensorCs> {
    /// The claimed values of the opening.
    pub values: Tensor<C::Data>,
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
pub trait TensorCs: 'static + Clone + Send + Sync {
    type Data: Clone + Send + Sync + Serialize + DeserializeOwned;
    type Commitment: 'static + Clone + Send + Sync + Serialize + DeserializeOwned;
    type Proof: Debug + Clone + Send + Sync + Serialize + DeserializeOwned;
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
        expected_path_len: usize,
    ) -> Result<(), Self::VerifierError>;
}

impl<C: TensorCs> TensorCsOpening<C> {
    #[inline]
    pub const fn new(values: Tensor<C::Data>, proof: <C as TensorCs>::Proof) -> Self {
        Self { values, proof }
    }
}

pub trait TensorCsProver<A: Backend>: 'static + Send + Sync {
    type Cs: TensorCs;
    type ProverData: 'static + Send + Sync + Debug + Clone;
    type ProverError: Error;

    /// Commit to a batch of tensors of the same shape.
    ///
    /// The prover is free to choose which dimension index is supported.
    #[allow(clippy::type_complexity)]
    fn commit_tensors<T>(
        &self,
        tensors: Message<T>,
    ) -> impl Future<
        Output = Result<(<Self::Cs as TensorCs>::Commitment, Self::ProverData), Self::ProverError>,
    > + Send
    where
        T: OwnedBorrow<Tensor<<Self::Cs as TensorCs>::Data, A>>;

    /// Prove openings at a list of indices.
    fn prove_openings_at_indices(
        &self,
        data: Self::ProverData,
        indices: &[usize],
    ) -> impl Future<Output = Result<<Self::Cs as TensorCs>::Proof, Self::ProverError>> + Send;
}

pub trait ComputeTcsOpenings<A: Backend>: TensorCsProver<A> {
    fn compute_openings_at_indices<T>(
        &self,
        tensors: Message<T>,
        indices: &[usize],
    ) -> impl Future<Output = Tensor<<Self::Cs as TensorCs>::Data>> + Send
    where
        T: OwnedBorrow<Tensor<<Self::Cs as TensorCs>::Data, A>>;
}

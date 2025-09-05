use std::fmt::Debug;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use slop_challenger::IopCtx;
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use slop_tensor::Tensor;
use thiserror::Error;

/// An opening of a tensor commitment scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct MerkleTreeOpening<GC: IopCtx> {
    /// The claimed values of the opening.
    pub values: Tensor<GC::F>,
    /// The proof of the opening.
    pub proof: MerkleTreeTcsProof<GC::Digest>,
}

/// An interfacr defining a Merkle tree.
pub trait MerkleTreeConfig<GC: IopCtx>: 'static + Clone + Send + Sync {
    type Hasher: CryptographicHasher<GC::F, GC::Digest> + Send + Sync + Clone;
    type Compressor: PseudoCompressionFunction<GC::Digest, 2> + Send + Sync + Clone;
}

pub trait DefaultMerkleTreeConfig<GC: IopCtx>: MerkleTreeConfig<GC> {
    fn default_hasher_and_compressor() -> (Self::Hasher, Self::Compressor);
}

/// A merkle tree Tensor commitment scheme.
///
/// A tensor commitment scheme based on merkleizing the committed tensors at a given dimension,
/// which the prover is free to choose.
#[derive(Debug, Clone, Copy)]
pub struct MerkleTreeTcs<GC: IopCtx, M: MerkleTreeConfig<GC>> {
    pub hasher: M::Hasher,
    pub compressor: M::Compressor,
}

#[derive(Debug, Clone, Copy, Error)]
pub enum MerkleTreeTcsError {
    #[error("root mismatch")]
    RootMismatch,
    #[error("proof has incorrect shape")]
    IncorrectShape,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTreeTcsProof<T> {
    pub paths: Tensor<T>,
}

impl<GC: IopCtx, M: DefaultMerkleTreeConfig<GC>> Default for MerkleTreeTcs<GC, M> {
    #[inline]
    fn default() -> Self {
        let (hasher, compressor) = M::default_hasher_and_compressor();
        Self { hasher, compressor }
    }
}

impl<GC: IopCtx, M: MerkleTreeConfig<GC>> MerkleTreeTcs<GC, M> {
    pub fn verify_tensor_openings(
        &self,
        commit: &GC::Digest,
        indices: &[usize],
        opening: &MerkleTreeOpening<GC>,
        expected_path_len: usize,
    ) -> Result<(), MerkleTreeTcsError> {
        if opening.proof.paths.dimensions.sizes().len() != 2
            || opening.values.dimensions.sizes().len() != 2
        {
            return Err(MerkleTreeTcsError::IncorrectShape);
        }
        if indices.len() != opening.proof.paths.dimensions.sizes()[0] {
            return Err(MerkleTreeTcsError::IncorrectShape);
        }
        if indices.len() != opening.values.dimensions.sizes()[0] {
            return Err(MerkleTreeTcsError::IncorrectShape);
        }
        if indices.is_empty() {
            return Ok(());
        }
        let expected_value_len = opening.values.get(0).unwrap().as_slice().len();
        for (i, (index, path)) in indices.iter().zip_eq(opening.proof.paths.split()).enumerate() {
            // Collect the lead slices of the claimed values.
            let claimed_values_slices = opening.values.get(i).unwrap().as_slice();
            if claimed_values_slices.len() != expected_value_len {
                return Err(MerkleTreeTcsError::IncorrectShape);
            }

            let path = path.as_slice();

            // Iterate the path and compute the root.
            let digest = self.hasher.hash_iter_slices(vec![claimed_values_slices]);

            let mut root = digest;
            let mut index = *index;

            if path.len() != expected_path_len {
                return Err(MerkleTreeTcsError::IncorrectShape);
            }

            for sibling in path.iter().cloned() {
                let (left, right) = if index & 1 == 0 { (root, sibling) } else { (sibling, root) };
                root = self.compressor.compress([left, right]);
                index >>= 1;
            }

            if root != *commit {
                return Err(MerkleTreeTcsError::RootMismatch);
            }

            if index != 0 {
                return Err(MerkleTreeTcsError::IncorrectShape);
            }
        }

        Ok(())
    }
}

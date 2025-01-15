use std::marker::PhantomData;

use csl_device::{
    cuda::task::TaskScope, mem::DeviceData, tensor::TensorView, Buffer, CudaSend, DeviceScope,
    GlobalAllocator, HostTensorView, Tensor,
};
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use thiserror::Error;

use crate::MerkleTree;

pub trait Tcs {
    type Data: DeviceData;

    type Commitment;
    type Proof;
    type VerifierError;

    fn verify_openings(
        &self,
        commit: &Self::Commitment,
        claimed_values: HostTensorView<Self::Data>,
        indices: &[usize],
        proof: &Self::Proof,
    ) -> Result<(), Self::VerifierError>;
}

pub trait TensorCSPDeviceProver<C: Tcs> {
    type ProverData;
    type ProverError;
    type ProverScope: DeviceScope;

    fn commit_tensor(
        &self,
        tensor: TensorView<C::Data, Self::ProverScope>,
        scope: &Self::ProverScope,
    ) -> Result<(C::Commitment, Self::ProverData), Self::ProverError>;

    fn prove_openings(
        &self,
        data: &Self::ProverData,
        indices: &[usize],
        scope: &Self::ProverScope,
    ) -> Result<C::Proof, Self::ProverError>;
}

#[derive(Debug, Clone, Copy)]
pub struct MerkleTreeTcs<T, D, H, C> {
    hasher: H,
    compressor: C,
    _marker: PhantomData<(T, D)>,
}

#[derive(Debug, Clone, Copy, Error)]
pub enum MerkleTreeTcsError {
    #[error("proof length mismatch got {0} paths but {1} claimed values")]
    ProofLengthMismatch(usize, usize),
    #[error("root mismatch for path {0}")]
    RootMismatch(usize),
}

#[derive(Debug, Clone, CudaSend)]
pub struct MerkleTreeTcsProof<D: DeviceData, S: DeviceScope> {
    pub paths: Tensor<D, S>,
}

impl<T, Digest, Hasher, Compressor> MerkleTreeTcs<T, Digest, Hasher, Compressor>
where
    T: DeviceData,
    Digest: DeviceData + Eq,
    Hasher: CryptographicHasher<T, Digest>,
    Compressor: PseudoCompressionFunction<Digest, 2>,
{
    pub fn new(hasher: Hasher, compressor: Compressor) -> Self {
        Self { hasher, compressor, _marker: PhantomData }
    }
}

impl<T, Digest, Hasher, Compressor> Tcs for MerkleTreeTcs<T, Digest, Hasher, Compressor>
where
    T: DeviceData,
    Digest: DeviceData + Eq,
    Hasher: CryptographicHasher<T, Digest>,
    Compressor: PseudoCompressionFunction<Digest, 2>,
{
    type Data = T;
    type Commitment = Digest;
    type Proof = MerkleTreeTcsProof<Digest, GlobalAllocator>;
    type VerifierError = MerkleTreeTcsError;

    fn verify_openings(
        &self,
        commit: &Self::Commitment,
        claimed_values: HostTensorView<Self::Data>,
        indices: &[usize],
        proof: &Self::Proof,
    ) -> Result<(), Self::VerifierError> {
        let paths_dim = proof.paths.sizes();
        if paths_dim[0] != claimed_values.sizes()[0] {
            return Err(Self::VerifierError::ProofLengthMismatch(
                paths_dim[0],
                claimed_values.sizes()[0],
            ));
        }

        // Iterate over all paths and claimed values and verify the proof
        let height = proof.paths.sizes()[1];
        let index_shift = (1 << height) - 1;
        for (i, ((index, path), claimed_value)) in
            indices.iter().zip(proof.paths.split()).zip(claimed_values.split()).enumerate()
        {
            assert_eq!(path.sizes().len(), 1);
            assert_eq!(claimed_value.sizes().len(), 1);

            let digest = self.hasher.hash_slice(claimed_value.as_slice());

            let mut root = digest;
            let mut index = index_shift + *index;
            for sibling in path.as_slice().iter().copied() {
                let (left, right) =
                    if (index - 1) & 1 == 0 { (root, sibling) } else { (sibling, root) };

                root = self.compressor.compress([left, right]);
                index = (index - 1) >> 1;
            }

            if root != *commit {
                return Err(Self::VerifierError::RootMismatch(i));
            }
        }

        Ok(())
    }
}

pub struct HostMerkleSimpleTcsProver<T: DeviceData, Digest, Hasher, Compressor> {
    tcs: MerkleTreeTcs<T, Digest, Hasher, Compressor>,
}

impl<T: DeviceData, Digest, Hasher, Compressor>
    HostMerkleSimpleTcsProver<T, Digest, Hasher, Compressor>
where
    T: DeviceData,
    Digest: DeviceData + Eq,
    Hasher: CryptographicHasher<T, Digest>,
    Compressor: PseudoCompressionFunction<Digest, 2>,
{
    pub fn new(tcs: MerkleTreeTcs<T, Digest, Hasher, Compressor>) -> Self {
        Self { tcs }
    }
}

impl<T: DeviceData, Digest, Hasher, Compressor>
    TensorCSPDeviceProver<MerkleTreeTcs<T, Digest, Hasher, Compressor>>
    for HostMerkleSimpleTcsProver<T, Digest, Hasher, Compressor>
where
    T: DeviceData,
    Digest: DeviceData + Eq,
    Hasher: CryptographicHasher<T, Digest>,
    Compressor: PseudoCompressionFunction<Digest, 2>,
{
    type ProverData = MerkleTree<Digest, GlobalAllocator>;
    type ProverError = ();
    type ProverScope = GlobalAllocator;

    fn commit_tensor(
        &self,
        tensor: TensorView<T, GlobalAllocator>,
        _scope: &Self::ProverScope,
    ) -> Result<(Digest, Self::ProverData), Self::ProverError> {
        let height = tensor.sizes()[0].ilog2();
        let mut digest_layers = Vec::new();
        let mut digests = vec![];
        for values in tensor.split() {
            let digest = self.tcs.hasher.hash_slice(values.as_slice());
            digests.push(digest);
        }
        digest_layers.push(digests);

        for k in (0..height).rev() {
            let mut new_layer = vec![];
            let prev_layer = digest_layers.last().unwrap();

            for i in 0..(1 << k) {
                let i = i as usize;
                let left = prev_layer[2 * i];
                let right = prev_layer[2 * i + 1];
                let digest = self.tcs.compressor.compress([left, right]);
                new_layer.push(digest);
            }
            digest_layers.push(new_layer);
        }
        let all_digests = digest_layers.into_iter().rev().flatten().collect::<Vec<_>>();
        let root = all_digests[0];
        let tree = unsafe { MerkleTree::new_from_digests(Buffer::from_vec(all_digests)) };
        Ok((root, tree))
    }

    fn prove_openings(
        &self,
        data: &Self::ProverData,
        indices: &[usize],
        _scope: &Self::ProverScope,
    ) -> Result<<MerkleTreeTcs<T, Digest, Hasher, Compressor> as Tcs>::Proof, Self::ProverError>
    {
        let mut proof = MerkleTreeTcsProof::<Digest, GlobalAllocator> {
            paths: Tensor::with_sizes([indices.len(), data.height()]),
        };
        unsafe {
            proof.paths.assume_init();
        }
        for (i, idx) in indices.iter().enumerate() {
            let mut idx = (1 << data.height()) - 1 + idx;
            let mut path = vec![];
            for _ in (0..data.height).rev() {
                let sibling_idx = ((idx - 1) ^ 1) + 1;
                let parent_idx = (idx - 1) >> 1;
                let digest = *data.digests[sibling_idx];
                path.push(digest);
                idx = parent_idx;
            }
            proof.paths.index_mut(i).as_mut_slice().copy_from_slice(&path);
        }

        Ok(proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use csl_baby_bear::config::posedion2_config;
    use csl_device::HostTensor;
    use rand::Rng;
    use slop_baby_bear::BabyBear;

    #[test]
    fn test_merkle_tree_tcs() {
        let perm = posedion2_config::default_perm();
        let hasher = posedion2_config::Hash::new(perm.clone());
        let compressor = posedion2_config::Compress::new(perm);
        let tcs = MerkleTreeTcs::<
            BabyBear,
            [BabyBear; 8],
            posedion2_config::Hash,
            posedion2_config::Compress,
        > {
            hasher,
            compressor,
            _marker: PhantomData,
        };

        let prover = HostMerkleSimpleTcsProver::new(tcs.clone());
        let mut rng = rand::thread_rng();

        let batch_size = 10;
        let num_values = 1 << 16;

        let values =
            (0..(batch_size * num_values)).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();
        let tensor =
            HostTensor::<BabyBear>::from(values).reshape([num_values, batch_size]).unwrap();
        let (commit, data) = prover.commit_tensor(tensor.as_view(), &GlobalAllocator).unwrap();
        for i in 0..num_values {
            let proof = prover.prove_openings(&data, &[i], &GlobalAllocator).unwrap();
            tcs.verify_openings(
                &commit,
                tensor.index(i).reshape([1, batch_size]).unwrap(),
                &[i],
                &proof,
            )
            .unwrap();
        }

        // Prove a slice of openings
        let proof = prover.prove_openings(&data, &[1, 2, 3], &GlobalAllocator).unwrap();
        tcs.verify_openings(
            &commit,
            tensor.index(1..4).reshape([3, batch_size]).unwrap(),
            &[1, 2, 3],
            &proof,
        )
        .unwrap();
    }
}

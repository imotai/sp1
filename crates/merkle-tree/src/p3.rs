use std::{
    cmp::Reverse, convert::Infallible, marker::PhantomData, mem::ManuallyDrop, ops::Deref,
    sync::Arc,
};

use derive_where::derive_where;
use itertools::Itertools;
use p3_merkle_tree::{compress_and_inject, first_digest_layer};
use serde::{Deserialize, Serialize};
use slop_algebra::{Field, PackedField, PackedValue};
use slop_alloc::CpuBackend;
use slop_baby_bear::BabyBear;
use slop_commit::{ComputeTcsOpenings, Message, TensorCsProver};
use slop_futures::OwnedBorrow;
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use slop_tensor::Tensor;

use crate::{
    DefaultMerkleTreeConfig, MerkleTreeConfig, MerkleTreeTcs, MerkleTreeTcsProof,
    Poseidon2BabyBearConfig,
};

#[derive_where(Default; M : DefaultMerkleTreeConfig)]
pub struct FieldMerkleTreeProver<P, PW, M: MerkleTreeConfig, const DIGEST_ELEMS: usize> {
    tcs: Arc<MerkleTreeTcs<M>>,
    _phantom: PhantomData<(P, PW)>,
}

impl<P, PW, M: MerkleTreeConfig, const DIGEST_ELEMS: usize> std::fmt::Debug
    for FieldMerkleTreeProver<P, PW, M, DIGEST_ELEMS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FieldMerkleTreeProver")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[W; DIGEST_ELEMS]: Serialize",
    deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct FieldMerkleTreeDigests<W, const DIGEST_ELEMS: usize> {
    pub digest_layers: Arc<Vec<Vec<[W; DIGEST_ELEMS]>>>,
}

impl<P, PW, M: MerkleTreeConfig, const DIGEST_ELEMS: usize>
    FieldMerkleTreeProver<P, PW, M, DIGEST_ELEMS>
{
    #[inline]
    pub fn new(tcs: MerkleTreeTcs<M>) -> Self {
        Self { tcs: Arc::new(tcs), _phantom: PhantomData }
    }
}

pub type Poseidon2BabyBear16Prover = FieldMerkleTreeProver<
    <BabyBear as Field>::Packing,
    <BabyBear as Field>::Packing,
    Poseidon2BabyBearConfig,
    8,
>;

impl<P, PW, M, const DIGEST_ELEMS: usize> TensorCsProver<CpuBackend>
    for FieldMerkleTreeProver<P, PW, M, DIGEST_ELEMS>
where
    P: PackedField,
    PW: PackedValue,
    M::Digest: Into<[PW::Value; DIGEST_ELEMS]> + From<[PW::Value; DIGEST_ELEMS]>,
    M: MerkleTreeConfig<Data = P::Scalar>,
    M::Hasher: CryptographicHasher<P::Scalar, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P::Scalar, M::Digest>,
    M::Hasher: CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    M::Hasher: Sync,
    M::Compressor: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<M::Digest, 2>,
    M::Compressor: PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
    M::Compressor: Sync,
    PW::Value: Eq + std::fmt::Debug,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type Cs = MerkleTreeTcs<M>;
    type ProverError = Infallible;
    type ProverData = FieldMerkleTreeDigests<PW::Value, DIGEST_ELEMS>;

    async fn commit_tensors<T>(
        &self,
        tensors: Message<T>,
    ) -> Result<
        (<Self::Cs as slop_commit::TensorCs>::Commitment, Self::ProverData),
        Self::ProverError,
    >
    where
        T: OwnedBorrow<Tensor<P::Scalar, CpuBackend>>,
    {
        let tcs = self.tcs.clone();
        let (tx, rx) = tokio::sync::oneshot::channel::<
            Result<([PW::Value; DIGEST_ELEMS], Self::ProverData), Self::ProverError>,
        >();
        slop_futures::rayon::spawn(move || {
            let leaves_owned = tensors
                .iter()
                .map(|t| {
                    let t = t.borrow();
                    let ptr = t.as_ptr() as *mut P::Value;
                    let height = t.sizes()[0];
                    let width = t.sizes()[1];
                    let vec = unsafe { Vec::from_raw_parts(ptr, height * width, height * width) };
                    let matrix = RowMajorMatrix::new(vec, width);
                    ManuallyDrop::new(matrix)
                })
                .collect::<Vec<_>>();
            let leaves = leaves_owned.iter().map(|m| m.deref()).collect::<Vec<_>>();
            assert!(!tensors.is_empty(), "No matrices given?");

            assert_eq!(P::WIDTH, PW::WIDTH, "Packing widths must match");

            // check height property
            assert!(
                leaves
                    .iter()
                    .map(|m| m.height())
                    .sorted()
                    .tuple_windows()
                    .all(|(curr, next)| curr == next
                        || curr.next_power_of_two() != next.next_power_of_two()),
                "matrix heights that round up to the same power of two must be equal"
            );

            let mut leaves_largest_first =
                leaves.iter().sorted_by_key(|l| Reverse(l.height())).peekable();

            let max_height = leaves_largest_first.peek().unwrap().height();
            let tallest_matrices = leaves_largest_first
                .peeking_take_while(|m| m.height() == max_height)
                .copied()
                .collect_vec();

            let mut digest_layers = vec![first_digest_layer::<
                P,
                PW,
                M::Hasher,
                RowMajorMatrix<P::Scalar>,
                DIGEST_ELEMS,
            >(&tcs.hasher, tallest_matrices)];
            loop {
                let prev_layer = digest_layers.last().unwrap().as_slice();
                if prev_layer.len() == 1 {
                    break;
                }
                let next_layer_len = prev_layer.len() / 2;

                // The matrices that get injected at this layer.
                let matrices_to_inject = leaves_largest_first
                    .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                    .copied()
                    .collect_vec();

                let next_digests = compress_and_inject::<
                    P,
                    PW,
                    M::Hasher,
                    M::Compressor,
                    RowMajorMatrix<P::Scalar>,
                    DIGEST_ELEMS,
                >(
                    prev_layer, matrices_to_inject, &tcs.hasher, &tcs.compressor
                );
                digest_layers.push(next_digests);
            }

            let digests = FieldMerkleTreeDigests { digest_layers: Arc::new(digest_layers) };
            let root = digests.digest_layers.last().unwrap()[0];
            tx.send(Ok((root, digests))).ok().unwrap();
        });
        let (a, b) = rx.await.unwrap()?;
        Ok((a.into(), b))
    }

    async fn prove_openings_at_indices(
        &self,
        data: Self::ProverData,
        indices: &[usize],
    ) -> Result<<Self::Cs as slop_commit::TensorCs>::Proof, Self::ProverError> {
        let height = data.digest_layers.len() - 1;
        let path_storage = indices
            .iter()
            .flat_map(|idx| {
                data.digest_layers
                    .iter()
                    .take(height)
                    .enumerate()
                    .map(move |(i, layer)| layer[(idx >> i) ^ 1].into())
            })
            .collect::<Vec<_>>();
        let paths = Tensor::from(path_storage).reshape([indices.len(), height]);
        let proof = MerkleTreeTcsProof { paths };
        Ok(proof)
    }
}

impl<P, PW, M, const DIGEST_ELEMS: usize> ComputeTcsOpenings<CpuBackend>
    for FieldMerkleTreeProver<P, PW, M, DIGEST_ELEMS>
where
    P: PackedField,
    PW: PackedValue,
    M: MerkleTreeConfig<
        Data = P::Scalar,
        Digest: Into<[PW::Value; DIGEST_ELEMS]> + From<[PW::Value; DIGEST_ELEMS]>,
    >,
    M::Hasher: CryptographicHasher<P::Scalar, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P::Scalar, M::Digest>,
    M::Hasher: CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    M::Hasher: Sync,
    M::Compressor: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<M::Digest, 2>,
    M::Compressor: PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
    M::Compressor: Sync,
    PW::Value: Eq + std::fmt::Debug,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    async fn compute_openings_at_indices<T>(
        &self,
        tensors: Message<T>,
        indices: &[usize],
    ) -> Tensor<<Self::Cs as slop_commit::TensorCs>::Data, CpuBackend>
    where
        T: OwnedBorrow<Tensor<<Self::Cs as slop_commit::TensorCs>::Data, CpuBackend>>,
    {
        let tensors = tensors.to_vec();
        let indices = indices.to_vec();
        let total_width = tensors.iter().map(|tensor| tensor.borrow().sizes()[1]).sum::<usize>();
        slop_futures::rayon::spawn(move || {
            let mut openings = Vec::new(); // Vec::with_capacity(tensors.len());
            for tensor in tensors.iter() {
                let tensor = tensor.borrow();
                let width = tensor.sizes()[1];
                let openings_for_tensor = indices
                    .iter()
                    .flat_map(|idx| tensor.get(*idx).unwrap().as_slice())
                    .cloned()
                    .collect::<Vec<_>>();
                let openings_for_tensor =
                    RowMajorMatrix::new(openings_for_tensor, width).transpose().values;
                openings.extend(openings_for_tensor);
            }
            Tensor::from(openings).reshape([total_width, indices.len()]).transpose()
        })
        .await
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use slop_commit::{TensorCs, TensorCsOpening};

    use super::*;

    #[tokio::test]
    async fn test_merkle_proof() {
        let mut rng = thread_rng();

        let height = 1000;
        let width = 25;
        let num_tensors = 10;

        let num_indices = 5;

        let tensors = (0..num_tensors)
            .map(|_| Tensor::<BabyBear>::rand(&mut rng, [height, width]))
            .collect::<Message<_>>();

        let prover = Poseidon2BabyBear16Prover::default();
        let (root, data) = prover.commit_tensors(tensors.clone()).await.unwrap();

        let indices = (0..num_indices).map(|_| rng.gen_range(0..height)).collect_vec();
        let proof = prover.prove_openings_at_indices(data, &indices).await.unwrap();
        let openings = prover.compute_openings_at_indices(tensors, &indices).await;

        let opening = TensorCsOpening { values: openings, proof };
        let tcs = MerkleTreeTcs::<Poseidon2BabyBearConfig>::default();
        tcs.verify_tensor_openings(&root, &indices, &opening).unwrap();
    }
}

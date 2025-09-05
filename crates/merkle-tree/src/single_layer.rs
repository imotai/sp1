use std::{marker::PhantomData, os::raw::c_void};

use csl_cuda::{
    args,
    sys::{
        merkle_tree::{
            compress_merkle_tree_bn254_kernel, compress_merkle_tree_koala_bear_16_kernel,
            compute_openings_merkle_tree_bn254_kernel,
            compute_openings_merkle_tree_koala_bear_16_kernel,
            compute_paths_merkle_tree_bn254_kernel, compute_paths_merkle_tree_koala_bear_16_kernel,
            leaf_hash_merkle_tree_bn254_kernel, leaf_hash_merkle_tree_koala_bear_16_kernel,
        },
        runtime::{Dim3, KernelPtr},
    },
    CudaError, TaskScope,
};
use slop_algebra::extension::BinomialExtensionField;
use slop_algebra::{AbstractField, Field};
use slop_alloc::CpuBackend;
use slop_alloc::{mem::CopyError, Buffer, HasBackend, IntoHost};
use slop_bn254::{Bn254Fr, BNGC};
use slop_challenger::IopCtx;
use slop_commit::Message;
use slop_futures::OwnedBorrow;
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex, Poseidon2KoalaBearConfig};
use slop_merkle_tree::{
    bn254_poseidon2_rc3, MerkleTreeConfig, MerkleTreeTcsProof, Poseidon2Bn254Config,
};
use slop_merkle_tree::{ComputeTcsOpenings, TensorCsProver};
use slop_tensor::Tensor;
use thiserror::Error;

use crate::tree::MerkleTree;
use crate::MerkleTreeHasher;

/// # Safety
///
/// The implementor must make sure that the kernel signatures are the same as the ones expected
/// by [`MerkleTreeSingleLayerProver`].
pub unsafe trait MerkleTreeSingleLayerKernels<GC: IopCtx, M: MerkleTreeConfig<GC>>:
    'static + Send + Sync
{
    fn leaf_hash_kernel() -> KernelPtr;

    fn compress_layer_kernel() -> KernelPtr;

    fn compute_paths_kernel() -> KernelPtr;

    fn compute_openings_kernel() -> KernelPtr;
}

pub trait Hasher<F: Field, const WIDTH: usize>: 'static + Send + Sync {
    fn hasher() -> MerkleTreeHasher<F, CpuBackend, WIDTH>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MerkleTreeSingleLayerProver<GC, W, K, H, M, const WIDTH: usize>(
    PhantomData<(GC, W, K, H, M)>,
);

#[derive(Debug, Clone, Copy, Error)]
pub enum SingleLayerMerkleTreeProverError {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),
    #[error("copy error: {0}")]
    Copy(#[from] CopyError),
}

impl<GC: IopCtx, W, K, H, M, const WIDTH: usize> TensorCsProver<GC, TaskScope>
    for MerkleTreeSingleLayerProver<GC, W, K, H, M, WIDTH>
where
    W: Field,
    M: MerkleTreeConfig<GC>,
    K: MerkleTreeSingleLayerKernels<GC, M>,
    H: Hasher<W, WIDTH>,
{
    type MerkleConfig = M;
    type ProverError = SingleLayerMerkleTreeProverError;
    type ProverData = MerkleTree<GC::Digest, TaskScope>;

    async fn commit_tensors<T>(
        &self,
        tensors: Message<T>,
    ) -> Result<(GC::Digest, Self::ProverData), Self::ProverError>
    where
        T: OwnedBorrow<Tensor<GC::F, TaskScope>>,
    {
        // assert_eq!(tensors.len(), 1, "Only one tensor is supported");
        let scope = tensors[0].borrow().backend();
        let hasher = H::hasher();
        let hasher_device = scope.to_device(&hasher).await.unwrap();
        let height = tensors[0].borrow().sizes()[1].ilog2() as usize;
        let (tensor_ptrs_host, widths_host): (Vec<_>, Vec<usize>) = tensors
            .iter()
            .map(|t| {
                let tensor = t.borrow();
                assert_eq!(tensor.sizes().len(), 2, "Tensor must be 2D");
                assert_eq!(tensor.sizes()[1], 1 << height, "Height must be a power of two");
                (tensor.as_ptr(), tensor.sizes()[0])
            })
            .unzip();
        let num_inputs = tensors.len();
        let mut tensor_ptrs = Buffer::with_capacity_in(tensor_ptrs_host.len(), scope.clone());
        tensor_ptrs.extend_from_host_slice(&tensor_ptrs_host)?;
        let mut widths = Buffer::with_capacity_in(widths_host.len(), scope.clone());
        widths.extend_from_host_slice(&widths_host)?;
        let height = tensors[0].borrow().sizes()[1].ilog2() as usize;
        assert_eq!(1 << height, tensors[0].borrow().sizes()[1], "Height must be a power of two");
        let mut tree = MerkleTree::<GC::Digest, _>::uninit(height, scope.clone());
        unsafe {
            tree.assume_init();
            // Compute the leaf hashes.
            let block_dim = 256;
            let grid_dim = (1usize << height).div_ceil(block_dim);
            let args = args!(
                hasher_device.as_raw(),
                tensor_ptrs.as_ptr(),
                tree.digests.as_mut_ptr(),
                widths.as_ptr(),
                num_inputs,
                height
            );
            scope.launch_kernel(K::leaf_hash_kernel(), grid_dim, block_dim, &args, 0)?;
        }

        // Iterate over the layers and compute the compressions.
        for k in (0..height).rev() {
            let block_dim: Dim3 = (128u32, 4, 1).into();
            let grid_dim: Dim3 = ((1u32 << k).div_ceil(block_dim.x), 1, num_inputs as u32).into();
            let args = args!(hasher_device.as_raw(), tree.digests.as_mut_ptr(), k);
            unsafe {
                scope.launch_kernel(K::compress_layer_kernel(), grid_dim, block_dim, &args, 0)?;
            }
        }

        let root = tree.digests[0].copy_into_host(scope);

        Ok((root, tree))
    }

    async fn prove_openings_at_indices(
        &self,
        data: Self::ProverData,
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<GC::Digest>, Self::ProverError> {
        let paths = {
            let scope = data.backend();
            let mut paths =
                Tensor::<GC::Digest, _>::with_sizes_in([indices.len(), data.height], scope.clone());
            let mut indices_buffer =
                Buffer::<usize, _>::with_capacity_in(indices.len(), scope.clone());
            indices_buffer.extend_from_host_slice(indices)?;
            let indices = indices_buffer;
            unsafe {
                paths.assume_init();

                let block_dim = 256;
                let grid_dim = indices.len().div_ceil(block_dim);
                let args = [
                    &(paths.as_mut_ptr()) as *const _ as *mut c_void,
                    &(indices.as_ptr()) as *const _ as *mut c_void,
                    &indices.len() as *const usize as _,
                    &(data.digests.as_ptr()) as *const _ as *mut c_void,
                    (&data.height) as *const usize as _,
                ];
                scope.launch_kernel(K::compute_paths_kernel(), grid_dim, block_dim, &args, 0)?;
            }
            paths
        };
        let paths = paths.into_host().await.unwrap();

        Ok(MerkleTreeTcsProof { paths })
    }
}

impl<GC: IopCtx, W, M, K, H, const WIDTH: usize> ComputeTcsOpenings<GC, TaskScope>
    for MerkleTreeSingleLayerProver<GC, W, K, H, M, WIDTH>
where
    W: Field,
    M: MerkleTreeConfig<GC>,
    K: MerkleTreeSingleLayerKernels<GC, M>,
    H: Hasher<W, WIDTH>,
{
    async fn compute_openings_at_indices<T>(
        &self,
        tensors: Message<T>,
        indices: &[usize],
    ) -> Tensor<GC::F>
    where
        T: OwnedBorrow<Tensor<GC::F, TaskScope>>,
    {
        // let mut openings
        let openings = {
            let num_opening_values = tensors.iter().map(|t| t.borrow().sizes()[0]).sum::<usize>();
            let height = tensors[0].borrow().sizes()[1].ilog2() as usize;
            let (tensor_ptrs_host, widths_host): (Vec<_>, Vec<usize>) = tensors
                .iter()
                .map(|t| {
                    let tensor = t.borrow();
                    assert_eq!(tensor.sizes().len(), 2, "Tensor must be 2D");
                    assert_eq!(tensor.sizes()[1], 1 << height, "Height must be a power of two");
                    (tensor.as_ptr(), tensor.sizes()[0])
                })
                .unzip();
            let scope = tensors[0].borrow().backend();
            let num_inputs = tensors.len();
            let mut tensor_ptrs = Buffer::with_capacity_in(tensor_ptrs_host.len(), scope.clone());
            tensor_ptrs.extend_from_host_slice(&tensor_ptrs_host).unwrap();
            let mut widths = Buffer::with_capacity_in(widths_host.len(), scope.clone());
            widths.extend_from_host_slice(&widths_host).unwrap();
            let tensor_height = tensors[0].borrow().sizes()[1];

            // Allocate tensors for the openings.
            let mut openings = Tensor::<GC::F, _>::with_sizes_in(
                [indices.len(), num_opening_values],
                scope.clone(),
            );
            let mut indices_buffer =
                Buffer::<usize, _>::with_capacity_in(indices.len(), scope.clone());
            indices_buffer.extend_from_host_slice(indices).unwrap();
            let indices = indices_buffer;

            let offsets = widths_host
                .iter()
                .scan(0, |offset, &width| {
                    let old_offset = *offset;
                    *offset += width;
                    Some(old_offset)
                })
                .collect::<Vec<_>>();
            // let total = offsets.pop().unwrap();
            // assert_eq!(total, num_opening_values);
            let mut offsets_buffer =
                Buffer::<usize, _>::with_capacity_in(offsets.len(), scope.clone());
            offsets_buffer.extend_from_host_slice(&offsets).unwrap();
            let offsets = offsets_buffer;
            unsafe {
                openings.assume_init();

                let block_dim = 256;
                let grid_dim = indices.len().div_ceil(block_dim);
                let args = args!(
                    tensor_ptrs.as_ptr(),
                    openings.as_mut_ptr(),
                    indices.as_ptr(),
                    indices.len(),
                    num_inputs,
                    widths.as_ptr(),
                    offsets.as_ptr(),
                    tensor_height,
                    num_opening_values
                );
                scope
                    .launch_kernel(K::compute_openings_kernel(), grid_dim, block_dim, &args, 0)
                    .unwrap();
            }
            openings
        };
        openings.into_host().await.unwrap()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2KoalaBear16Kernels;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2Bn254Kernels;

pub type Poseidon2KoalaBear16CudaProver = MerkleTreeSingleLayerProver<
    KoalaBearDegree4Duplex,
    KoalaBear,
    Poseidon2KoalaBear16Kernels,
    Poseidon2KoalaBear16Hasher,
    Poseidon2KoalaBearConfig,
    16,
>;

pub type Poseidon2Bn254CudaProver = MerkleTreeSingleLayerProver<
    BNGC<KoalaBear, BinomialExtensionField<KoalaBear, 4>>,
    Bn254Fr,
    Poseidon2Bn254Kernels,
    Poseidon2Bn254Hasher,
    Poseidon2Bn254Config<KoalaBear>,
    3,
>;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2KoalaBear16Hasher;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2Bn254Hasher;

impl Hasher<KoalaBear, 16> for Poseidon2KoalaBear16Hasher {
    fn hasher() -> MerkleTreeHasher<KoalaBear, CpuBackend, 16> {
        MerkleTreeHasher::default()
    }
}

impl Hasher<Bn254Fr, 3> for Poseidon2Bn254Hasher {
    fn hasher() -> MerkleTreeHasher<Bn254Fr, CpuBackend, 3> {
        let (internal_round_constants, external_round_constants, diffusion_matrix_m1) =
            poseidon2_bn254_3_constants();
        MerkleTreeHasher::new(
            internal_round_constants.into(),
            external_round_constants.into(),
            diffusion_matrix_m1.into(),
            Bn254Fr::one(),
        )
    }
}

pub fn poseidon2_bn254_3_constants() -> (Vec<Bn254Fr>, Vec<[Bn254Fr; 3]>, Vec<Bn254Fr>) {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;
    let mut round_constants = bn254_poseidon2_rc3();
    let internal_start = ROUNDS_F / 2;
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let internal_round_constants =
        round_constants.drain(internal_start..internal_end).map(|vec| vec[0]).collect::<Vec<_>>();
    let external_round_constants = round_constants;
    let diffusion_matrix_m1 = [Bn254Fr::one(), Bn254Fr::one(), Bn254Fr::two()].to_vec();
    (internal_round_constants, external_round_constants, diffusion_matrix_m1)
}

unsafe impl MerkleTreeSingleLayerKernels<KoalaBearDegree4Duplex, Poseidon2KoalaBearConfig>
    for Poseidon2KoalaBear16Kernels
{
    #[inline]
    fn leaf_hash_kernel() -> KernelPtr {
        unsafe { leaf_hash_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn compress_layer_kernel() -> KernelPtr {
        unsafe { compress_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn compute_paths_kernel() -> KernelPtr {
        unsafe { compute_paths_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn compute_openings_kernel() -> KernelPtr {
        unsafe { compute_openings_merkle_tree_koala_bear_16_kernel() }
    }
}

unsafe impl
    MerkleTreeSingleLayerKernels<
        BNGC<KoalaBear, BinomialExtensionField<KoalaBear, 4>>,
        Poseidon2Bn254Config<KoalaBear>,
    > for Poseidon2Bn254Kernels
{
    #[inline]
    fn leaf_hash_kernel() -> KernelPtr {
        unsafe { leaf_hash_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn compress_layer_kernel() -> KernelPtr {
        unsafe { compress_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn compute_paths_kernel() -> KernelPtr {
        unsafe { compute_paths_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn compute_openings_kernel() -> KernelPtr {
        unsafe { compute_openings_merkle_tree_bn254_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use slop_bn254::Bn254Fr;
    use slop_koala_bear::{KoalaBear, Poseidon2KoalaBearConfig};
    use slop_merkle_tree::{
        FieldMerkleTreeProver, MerkleTreeOpening, MerkleTreeTcs, Poseidon2KoalaBear16Prover,
    };

    use super::*;

    pub type Poseidon2Bn254Prover = FieldMerkleTreeProver<
        KoalaBear,
        Bn254Fr,
        BNGC<KoalaBear, BinomialExtensionField<KoalaBear, 4>>,
        Poseidon2Bn254Config<KoalaBear>,
        1,
    >;

    #[tokio::test]
    async fn test_merkle_proof_koala_bear() {
        let mut rng = thread_rng();

        let merkle_height = 12;
        let height = 1 << merkle_height;

        let num_indices = 50;
        let widths = [10, 20, 30, 5, 140];

        let host_tensors = widths
            .into_iter()
            .map(|width| Tensor::<KoalaBear>::rand(&mut rng, [height, width]))
            .collect::<Message<_>>();
        let indices = (0..num_indices).map(|_| rng.gen_range(0..height)).collect::<Vec<_>>();

        let host_prover = Poseidon2KoalaBear16Prover::default();
        let (host_root, _) = host_prover.commit_tensors(host_tensors.clone()).await.unwrap();
        csl_cuda::spawn(move |t| async move {
            let mut tensors = vec![];
            for host_tensor in host_tensors {
                let host_tensor = Tensor::<KoalaBear>::clone(&host_tensor);
                let tensor = t.into_device(host_tensor).await.unwrap().transpose();
                tensors.push(tensor);
            }
            let tensors = Message::<Tensor<KoalaBear, TaskScope>>::from(tensors);
            let prover = Poseidon2KoalaBear16CudaProver::default();
            let (root, data) = prover.commit_tensors(tensors.clone()).await.unwrap();

            let proof = prover.prove_openings_at_indices(data, &indices).await.unwrap();
            let openings = prover.compute_openings_at_indices(tensors, &indices).await;

            assert_eq!(host_root, root);

            let tcs = MerkleTreeTcs::<_, Poseidon2KoalaBearConfig>::default();
            let opening = MerkleTreeOpening { values: openings, proof };
            tcs.verify_tensor_openings(&root, &indices, &opening, merkle_height).unwrap();
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_merkle_proof_bn254() {
        let mut rng = thread_rng();

        let merkle_height = 12;
        let height = 1 << merkle_height;

        let num_indices = 50;
        let widths = [10, 20, 30, 5, 140];

        let host_tensors = widths
            .into_iter()
            .map(|width| Tensor::<KoalaBear>::rand(&mut rng, [height, width]))
            .collect::<Message<_>>();
        let indices = (0..num_indices).map(|_| rng.gen_range(0..height)).collect::<Vec<_>>();

        let host_prover = Poseidon2Bn254Prover::default();
        let (host_root, _) = host_prover.commit_tensors(host_tensors.clone()).await.unwrap();
        csl_cuda::spawn(move |t| async move {
            let mut tensors = vec![];
            for host_tensor in host_tensors {
                let host_tensor = Tensor::<KoalaBear>::clone(&host_tensor);
                let tensor = t.into_device(host_tensor).await.unwrap().transpose();
                tensors.push(tensor);
            }
            let tensors = Message::<Tensor<KoalaBear, TaskScope>>::from(tensors);
            let prover = Poseidon2Bn254CudaProver::default();
            let (root, data) = prover.commit_tensors(tensors.clone()).await.unwrap();

            let proof = prover.prove_openings_at_indices(data, &indices).await.unwrap();
            let openings = prover.compute_openings_at_indices(tensors, &indices).await;

            assert_eq!(host_root, root);

            let tcs = MerkleTreeTcs::<_, Poseidon2Bn254Config<KoalaBear>>::default();
            let opening =
                MerkleTreeOpening::<BNGC<KoalaBear, BinomialExtensionField<KoalaBear, 4>>> {
                    values: openings,
                    proof,
                };
            tcs.verify_tensor_openings(&root, &indices, &opening, merkle_height).unwrap();
        })
        .await
        .unwrap();
    }
}

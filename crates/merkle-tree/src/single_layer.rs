use std::{marker::PhantomData, os::raw::c_void};

use csl_device::{
    cuda::{CudaError, DeviceTransposeKernel, TaskScope},
    mem::DeviceData,
    tensor::TensorView,
    DeviceBuffer, DeviceTensor, GlobalAllocator, KernelPtr,
};
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};

use crate::{MerkleTree, MerkleTreeTcs, MerkleTreeTcsProof, TensorCSPDeviceProver};

/// # Safety
///
/// The implementor must make sure that the kernel signatures are the same as the ones expected
/// by [`MerkleTreeSingleLayerProver`].
pub unsafe trait MerkeTreeSingleLayerKernels {
    type Data: DeviceData;
    type Digest: DeviceData + Eq + DeviceTransposeKernel;
    type Hasher: CryptographicHasher<Self::Data, Self::Digest>;
    type Compressor: PseudoCompressionFunction<Self::Digest, 2>;

    fn leaf_hash_kernel() -> KernelPtr;

    fn compress_layer_kernel() -> KernelPtr;

    fn compute_paths_kernel() -> KernelPtr;

    fn compute_openings_kernel() -> KernelPtr;
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MerkleTreeSingleLayerProver<K>(PhantomData<K>);

impl<K: MerkeTreeSingleLayerKernels> MerkleTreeSingleLayerProver<K> {
    pub fn compute_openings(
        &self,
        tensor: TensorView<K::Data, TaskScope>,
        indices: &[usize],
        scope: &TaskScope,
    ) -> Result<DeviceTensor<K::Data>, CudaError> {
        let mut openings = scope.tensor::<K::Data>([indices.len(), tensor.sizes()[0]]);
        let indices_device =
            DeviceBuffer::from_host_slice_blocking(indices, scope.clone()).unwrap();
        unsafe {
            openings.assume_init();

            let block_dim = 256;
            let grid_dim = indices.len().div_ceil(block_dim);
            let args = [
                &tensor.as_ptr() as *const _ as *mut c_void,
                &mut (openings.as_mut_ptr()) as *mut _ as *mut c_void,
                &(indices_device.as_ptr()) as *const _ as *mut c_void,
                &indices.len() as *const usize as _,
                &tensor.sizes()[0] as *const usize as _,
                &tensor.sizes()[1] as *const usize as _,
            ];
            scope.launch_kernel(K::compute_openings_kernel(), grid_dim, block_dim, &args, 0)?;
        }

        Ok(openings)
    }
}

impl<K: MerkeTreeSingleLayerKernels>
    TensorCSPDeviceProver<MerkleTreeTcs<K::Data, K::Digest, K::Hasher, K::Compressor>>
    for MerkleTreeSingleLayerProver<K>
{
    type ProverData = MerkleTree<K::Digest, TaskScope>;
    type ProverError = CudaError;
    type ProverScope = TaskScope;

    fn commit_tensor(
        &self,
        tensor: TensorView<K::Data, TaskScope>,
        scope: &Self::ProverScope,
    ) -> Result<(K::Digest, Self::ProverData), Self::ProverError> {
        let height = tensor.sizes()[1].ilog2() as usize;
        let mut tree = MerkleTree::<K::Digest, _>::uninit(height, scope.clone());
        unsafe {
            tree.assume_init();
            // Compute the leaf hashes.
            let block_dim = 256;
            let grid_dim = tensor.sizes()[1].div_ceil(block_dim);
            let args = [
                &(tensor.as_ptr()) as *const _ as *mut c_void,
                &mut (tree.digests.as_mut_ptr()) as *mut _ as *mut c_void,
                &tensor.sizes()[0] as *const usize as _,
                &height as *const usize as _,
            ];
            scope.launch_kernel(K::leaf_hash_kernel(), grid_dim, block_dim, &args, 0)?;
        }

        // Iterate over the layers and compute the compressions.
        for k in (0..height).rev() {
            let block_dim = 256;
            let grid_dim = ((1 << k) as usize).div_ceil(block_dim);
            let args = [
                &mut (tree.digests.as_mut_ptr()) as *mut _ as *mut c_void,
                &k as *const usize as _,
            ];
            unsafe {
                scope.launch_kernel(K::compress_layer_kernel(), grid_dim, block_dim, &args, 0)?;
            }
        }

        let root = tree.digests[0].to_host_blocking(scope).unwrap();

        Ok((root, tree))
    }

    fn prove_openings(
        &self,
        data: &Self::ProverData,
        indices: &[usize],
        scope: &Self::ProverScope,
    ) -> Result<MerkleTreeTcsProof<K::Digest, GlobalAllocator>, Self::ProverError> {
        let mut paths = scope.tensor::<K::Digest>([indices.len(), data.height]);
        let indices_device =
            DeviceBuffer::from_host_slice_blocking(indices, scope.clone()).unwrap();
        unsafe {
            paths.assume_init();

            let block_dim = 256;
            let grid_dim = indices.len().div_ceil(block_dim);
            let args = [
                &(paths.as_mut_ptr()) as *const _ as *mut c_void,
                &(indices_device.as_ptr()) as *const _ as *mut c_void,
                &indices.len() as *const usize as _,
                &(data.digests.as_ptr()) as *const _ as *mut c_void,
                (&data.height) as *const usize as _,
            ];
            scope.launch_kernel(K::compute_paths_kernel(), grid_dim, block_dim, &args, 0)?;
        }

        let host_paths = paths.into_host_blocking().unwrap();

        Ok(MerkleTreeTcsProof { paths: host_paths })
    }
}

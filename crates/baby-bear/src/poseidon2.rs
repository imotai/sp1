use csl_device::KernelPtr;
use csl_merkle_tree::{MerkeTreeSingleLayerKernels, MerkleTreeSingleLayerProver};
use csl_sys::poseidon2::{
    compress_poseidon_2_baby_bear_16_kernel, compute_openings_poseidon_2_baby_bear_16_kernel,
    compute_paths_poseidon_2_baby_bear_16_kernel, leaf_hash_poseidon_2_baby_bear_16_kernel,
};
use slop_baby_bear::BabyBear;

use crate::config::posedion2_config;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2BabyBear16Kernels;

pub type Poseidon2TensorCsProver = MerkleTreeSingleLayerProver<Poseidon2BabyBear16Kernels>;

unsafe impl MerkeTreeSingleLayerKernels for Poseidon2BabyBear16Kernels {
    type Data = BabyBear;
    type Digest = [BabyBear; 8];
    type Hasher = posedion2_config::Hash;
    type Compressor = posedion2_config::Compress;
    #[inline]
    fn leaf_hash_kernel() -> KernelPtr {
        unsafe { leaf_hash_poseidon_2_baby_bear_16_kernel() }
    }

    #[inline]
    fn compress_layer_kernel() -> KernelPtr {
        unsafe { compress_poseidon_2_baby_bear_16_kernel() }
    }

    #[inline]
    fn compute_paths_kernel() -> KernelPtr {
        unsafe { compute_paths_poseidon_2_baby_bear_16_kernel() }
    }

    #[inline]
    fn compute_openings_kernel() -> KernelPtr {
        unsafe { compute_openings_poseidon_2_baby_bear_16_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use csl_device::{DeviceTensor, HostTensor};
    use csl_merkle_tree::{MerkleTreeTcs, Tcs, TensorCSPDeviceProver};
    use rand::Rng;

    use super::*;

    #[tokio::test]
    async fn test_poseidon2_baby_bear_16_merkle_tree() {
        let mut rng = rand::thread_rng();

        let batch_size = 1 << 7;

        let perm = posedion2_config::default_perm();
        let hasher = posedion2_config::Hash::new(perm.clone());
        let compressor = posedion2_config::Compress::new(perm);
        let tcs = MerkleTreeTcs::<BabyBear, [BabyBear; 8], _, _>::new(hasher, compressor);

        for num_values in [1 << 16, 1 << 17, 1 << 21, 1 << 22] {
            let values =
                (0..(batch_size * num_values)).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();
            let tensor =
                HostTensor::<BabyBear>::from(values).reshape([batch_size, num_values]).unwrap();

            let indices = (0..100).map(|_| rng.gen_range(0..num_values)).collect::<Vec<_>>();

            let prover = Poseidon2TensorCsProver::default();

            let indices_ref = &indices;
            let (root, proof, openings) = csl_device::cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let tensor = DeviceTensor::from_host(tensor, t.clone()).await.unwrap();
                    t.synchronize().await;
                    let time = std::time::Instant::now();
                    let (root, tree) = prover.commit_tensor(tensor.as_view(), &t).unwrap();
                    println!("Commit time: {:?}", time.elapsed());

                    let openings =
                        prover.compute_openings(tensor.as_view(), indices_ref, &t).unwrap();

                    let time = std::time::Instant::now();
                    let proof = prover.prove_openings(&tree, indices_ref, &t).unwrap();
                    println!("Opening time: {:?}", time.elapsed());

                    let openings = openings.into_host().await.unwrap();

                    (root, proof, openings)
                })
                .await
                .await
                .unwrap();

            tcs.verify_openings(&root, openings.as_view(), &indices, &proof).unwrap();
        }
    }
}

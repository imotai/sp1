use std::future::Future;

use csl_challenger::DuplexChallenger;
use csl_cuda::sys::challenger::grind_baby_bear;
use csl_cuda::{args, sys::runtime::KernelPtr, TaskScope};
use slop_algebra::PrimeField64;
use slop_alloc::{Buffer, CpuBackend, IntoHost};
use slop_baby_bear::BabyBear;
use slop_challenger::GrindingChallenger;
use slop_symmetric::CryptographicPermutation;

/// # Safety
/// TODO
pub unsafe trait GrindingChallengerKernel<F> {
    fn grind_kernel() -> KernelPtr;
}

/// A [`GrindingChallenger`] that can also grind on device.
///
/// Useful for finding a proof-of-work witness on machines with not that many cores.
pub trait DeviceGrindingChallenger: GrindingChallenger {
    type OnDeviceChallenger;
    /// Grinds on device.
    fn grind_device(&mut self, bits: usize) -> impl Future<Output = Self::Witness> + Send;

    fn into_device(
        self,
        backend: TaskScope,
    ) -> impl Future<Output = Self::OnDeviceChallenger> + Send + Sync;
}

impl<F, P, const WIDTH: usize, const RATE: usize> DeviceGrindingChallenger
    for slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64 + Send + Sync,
    P: CryptographicPermutation<[F; WIDTH]> + Send + Sync,
    TaskScope: GrindingChallengerKernel<F>,
{
    type OnDeviceChallenger = DuplexChallenger<F, TaskScope>;
    async fn grind_device(&mut self, bits: usize) -> Self::Witness {
        let cpu_challenger: DuplexChallenger<F, _> = self.clone().into();
        let handle = csl_cuda::spawn(move |t| async move {
            let mut result: Buffer<F, TaskScope> = Buffer::with_capacity_in(1, t.clone());
            let mut found_flag: Buffer<bool, TaskScope> = Buffer::with_capacity_in(1, t.clone());
            let mut gpu_challenger = t.into_device(cpu_challenger).await.unwrap();
            let block_dim: usize = 512;
            let grid_dim: usize = 1;
            let n = F::ORDER_U64;
            unsafe {
                result.assume_init();
                found_flag.assume_init();
                let args = args!(
                    gpu_challenger.as_mut_raw(),
                    result.as_mut_ptr(),
                    bits,
                    n,
                    found_flag.as_mut_ptr()
                );
                t.launch_kernel(TaskScope::grind_kernel(), (grid_dim, 1, 1), block_dim, &args, 0)
                    .unwrap();
            }
            result
        });
        let result = handle.await.unwrap();
        let cpu_result = result.into_host().await.unwrap();
        let result = cpu_result.first().unwrap();
        // Check the witness. This is necessary, because it changes the internal state of the
        // challenger, and the CPU version of the challenger does this as well. It's also necessary
        // for the security of the protocol.
        assert!(self.check_witness(bits, *result));
        *result
    }

    async fn into_device(self, backend: TaskScope) -> Self::OnDeviceChallenger {
        backend.into_device(DuplexChallenger::<F, CpuBackend>::from(self)).await.unwrap()
    }
}

unsafe impl GrindingChallengerKernel<BabyBear> for TaskScope {
    fn grind_kernel() -> KernelPtr {
        unsafe { grind_baby_bear() }
    }
}

#[cfg(test)]
mod tests {
    use crate::grinding_challenger::DeviceGrindingChallenger;
    use slop_algebra::AbstractField;
    use slop_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use slop_challenger::{CanObserve, CanSample, GrindingChallenger};
    use slop_merkle_tree::my_bb_16_perm;
    use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

    pub type Perm =
        Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;

    #[tokio::test]
    async fn test_grinding() {
        for bits in 1..20 {
            let default_perm = my_bb_16_perm();
            let mut challenger =
                slop_challenger::DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm);

            // Observe 7 elements to make the input buffer almost full and trigger duplexing on
            challenger.observe(BabyBear::from_canonical_u32(0));
            challenger.observe(BabyBear::from_canonical_u32(1));
            challenger.observe(BabyBear::from_canonical_u32(2));
            challenger.observe(BabyBear::from_canonical_u32(3));
            challenger.observe(BabyBear::from_canonical_u32(4));
            challenger.observe(BabyBear::from_canonical_u32(5));
            challenger.observe(BabyBear::from_canonical_u32(6));
            challenger.observe(BabyBear::from_canonical_u32(7));

            // Make another challenger that also samples before grinding (this empties the input buffer).
            let mut challenger_2 = challenger.clone();
            let _: BabyBear = challenger.sample();

            let mut original_challenger = challenger.clone();
            let result = challenger.grind_device(bits).await;

            assert!(original_challenger.check_witness(bits, result));

            let mut original_challenger_2 = challenger_2.clone();
            let result_2 = challenger_2.grind_device(bits).await;

            assert!(original_challenger_2.check_witness(bits, result_2));

            // Checks to make sure the pow witness was properly observed in `grind_on_device`.
            assert!(original_challenger_2.sponge_state == challenger_2.sponge_state);
            assert!(original_challenger_2.input_buffer == challenger_2.input_buffer);
            assert!(original_challenger_2.output_buffer == challenger_2.output_buffer);
        }
    }
}

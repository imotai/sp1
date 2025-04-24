use csl_sys::{
    mle::{mle_fold_baby_bear_base_base, mle_fold_baby_bear_ext_ext},
    runtime::KernelPtr,
};
use slop_algebra::{extension::BinomialExtensionField, Field};
use slop_alloc::Backend;
use slop_baby_bear::BabyBear;
use slop_multilinear::{MleBaseBackend, MleFoldBackend};
use slop_tensor::Tensor;

use crate::{args, TaskScope};

/// # Safety
///
/// todo
pub unsafe trait FoldKernel<F: Field>: Backend {
    fn fold_kernel() -> KernelPtr;
}

impl<F: Field> MleFoldBackend<F> for TaskScope
where
    TaskScope: FoldKernel<F> + MleBaseBackend<F>,
{
    async fn fold_mle(guts: &Tensor<F, Self>, beta: F) -> slop_tensor::Tensor<F, Self> {
        let num_polynomials = Self::num_polynomials(guts);
        let num_non_zero_entries = Self::num_non_zero_entries(guts);
        let folded_num_non_zero_entries = num_non_zero_entries / 2;
        let mut folded_guts =
            Self::uninit_mle(guts.backend(), num_polynomials, folded_num_non_zero_entries);

        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 16;
        let block_dim = BLOCK_SIZE;
        let grid_size_x = folded_num_non_zero_entries.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size_y = num_polynomials;
        let grid_dim = (grid_size_x, grid_size_y, 1);
        let args = args!(
            guts.as_ptr(),
            folded_guts.as_mut_ptr(),
            beta,
            folded_num_non_zero_entries,
            num_polynomials
        );
        unsafe {
            folded_guts.assume_init();
            guts.backend()
                .launch_kernel(Self::fold_kernel(), grid_dim, block_dim, &args, 0)
                .unwrap();
        }
        folded_guts
    }
}

unsafe impl FoldKernel<BabyBear> for TaskScope {
    fn fold_kernel() -> KernelPtr {
        unsafe { mle_fold_baby_bear_base_base() }
    }
}

unsafe impl FoldKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn fold_kernel() -> KernelPtr {
        unsafe { mle_fold_baby_bear_ext_ext() }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use slop_alloc::IntoHost;
    use slop_multilinear::Mle;

    use super::*;

    #[tokio::test]
    async fn test_fold_mle() {
        let num_variables = 11;

        type EF = BinomialExtensionField<BabyBear, 4>;

        let mut rng = rand::thread_rng();

        let mle = Mle::<EF>::rand(&mut rng, 1, num_variables);
        let beta = rng.gen::<EF>();

        let folded_mle_host = mle.fold(beta).await;

        let folded_mle_cuda = crate::run_in_place(|t| async move {
            let mle = t.into_device(mle).await.unwrap();
            let folded_mle_cuda = mle.fold(beta).await;
            folded_mle_cuda.into_host().await.unwrap()
        })
        .await
        .await
        .unwrap();

        for (val, exp) in
            folded_mle_host.guts().as_slice().iter().zip(folded_mle_cuda.guts().as_slice())
        {
            assert_eq!(val, exp);
        }
    }
}

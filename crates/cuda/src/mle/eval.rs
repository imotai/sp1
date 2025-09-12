use csl_sys::{
    mle::{
        partial_geq_koala_bear, partial_lagrange_koala_bear, partial_lagrange_koala_bear_extension,
    },
    runtime::KernelPtr,
};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field};
use slop_alloc::IntoHost;
use slop_koala_bear::KoalaBear;
use slop_multilinear::{
    MleBaseBackend, MleEvaluationBackend, MleFixedAtZeroBackend, PartialLagrangeBackend, Point,
    ZeroEvalBackend,
};
use slop_tensor::{DotBackend, Tensor};

use crate::{args, TaskScope};

/// # Safety
///
pub unsafe trait PartialLagrangeKernel<F: Field> {
    fn partial_lagrange_kernel() -> KernelPtr;
}

/// # Safety
///
pub unsafe trait PartialGeqKernel<F: Field> {
    fn partial_geq_kernel() -> KernelPtr;
}

impl<F: Field> MleBaseBackend<F> for TaskScope {
    #[inline]
    fn uninit_mle(&self, num_polynomials: usize, num_non_zero_entries: usize) -> Tensor<F, Self> {
        Tensor::with_sizes_in([num_polynomials, num_non_zero_entries], self.clone())
    }

    #[inline]
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize {
        guts.sizes()[0]
    }

    #[inline]
    fn num_variables(guts: &Tensor<F, Self>) -> u32 {
        guts.sizes()[1].next_power_of_two().ilog2()
    }

    #[inline]
    fn num_non_zero_entries(guts: &Tensor<F, Self>) -> usize {
        guts.sizes()[1]
    }
}

impl<F: Field> PartialLagrangeBackend<F> for TaskScope
where
    TaskScope: PartialLagrangeKernel<F>,
{
    async fn partial_lagrange(point: &Point<F, Self>) -> Tensor<F, Self> {
        let dimension = point.dimension();
        let num_elements = 1 << dimension;
        let mut eq = point.backend().uninit_mle(1, num_elements);
        unsafe {
            eq.assume_init();
            let block_dim = 256;
            let grid_dim = ((1 << dimension) as u32).div_ceil(block_dim);
            let args = args!(eq.as_mut_ptr(), point.as_ptr(), dimension);
            point
                .backend()
                .launch_kernel(
                    <Self as PartialLagrangeKernel<F>>::partial_lagrange_kernel(),
                    grid_dim,
                    block_dim,
                    &args,
                    0,
                )
                .unwrap();
        }
        eq
    }
}

impl<F: Field> ZeroEvalBackend<F> for TaskScope {
    fn zero_evaluations(&self, num_polynomials: usize) -> Tensor<F, Self> {
        Tensor::zeros_in([num_polynomials], self.clone())
    }
}

impl<F: Field, EF: ExtensionField<F>> MleEvaluationBackend<F, EF> for TaskScope
where
    TaskScope: PartialLagrangeKernel<EF> + DotBackend<F, EF>,
{
    async fn eval_mle_at_point(mle: &Tensor<F, Self>, point: &Point<EF, Self>) -> Tensor<EF, Self> {
        let eq = Self::partial_lagrange(point).await;
        mle.dot(&eq, 1).await
    }

    async fn eval_mle_at_eq(mle: &Tensor<F, Self>, eq: &Tensor<EF, Self>) -> Tensor<EF, Self> {
        mle.dot(eq, 1).await
    }
}

unsafe impl PartialLagrangeKernel<KoalaBear> for TaskScope {
    fn partial_lagrange_kernel() -> KernelPtr {
        unsafe { partial_lagrange_koala_bear() }
    }
}

unsafe impl PartialLagrangeKernel<BinomialExtensionField<KoalaBear, 4>> for TaskScope {
    fn partial_lagrange_kernel() -> KernelPtr {
        unsafe { partial_lagrange_koala_bear_extension() }
    }
}

unsafe impl PartialGeqKernel<KoalaBear> for TaskScope {
    fn partial_geq_kernel() -> KernelPtr {
        unsafe { partial_geq_koala_bear() }
    }
}

pub async fn partial_geq<F: Field>(
    threshold: usize,
    num_variables: usize,
    backend: &TaskScope,
) -> Tensor<F, TaskScope>
where
    TaskScope: PartialGeqKernel<F>,
{
    let mut eq = backend.uninit_mle(1, 1 << num_variables);
    unsafe {
        eq.assume_init();
        let block_dim = 256;
        let grid_dim = ((1 << num_variables) as u32).div_ceil(block_dim);
        let args = args!(eq.as_mut_ptr(), threshold, num_variables);
        backend
            .launch_kernel(
                <TaskScope as PartialGeqKernel<F>>::partial_geq_kernel(),
                grid_dim,
                block_dim,
                &args,
                0,
            )
            .unwrap();
    }
    eq
}

impl<F: Field, EF: ExtensionField<F>> MleFixedAtZeroBackend<F, EF> for TaskScope
where
    TaskScope: MleEvaluationBackend<F, EF>,
{
    async fn fixed_at_zero(mle: &Tensor<F, Self>, point: &Point<EF>) -> Tensor<EF> {
        // For now, we are just adding a zero to the point and evaluating the mle at that point.
        // TODO: use a kernel to access only even elements
        let mut point = point.clone();
        point.add_dimension_back(EF::zero());
        let point_with_zero = mle.backend().into_device(point).await.unwrap();
        let evals = TaskScope::eval_mle_at_point(mle, &point_with_zero).await;
        evals.into_host().await.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use slop_algebra::extension::BinomialExtensionField;
    use slop_alloc::IntoHost;
    use slop_koala_bear::KoalaBear;
    use slop_multilinear::{full_geq, Mle, Point};

    use crate::partial_geq;

    #[tokio::test]
    async fn test_mle_eval() {
        let mut rng = rand::thread_rng();

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;

        let mle = Mle::<F>::rand(&mut rng, 100, 16);
        let point = Point::<EF>::rand(&mut rng, 16);

        let d_mle = mle.clone();

        let point_ref = point.clone();
        let evals = crate::run_in_place(|t| async move {
            let d_point = point_ref.copy_into(&t);
            let d_mle = t.into_device(d_mle).await.unwrap();
            let eval = d_mle.eval_at(&d_point).await;
            eval.into_evaluations().into_host().await.unwrap()
        })
        .await
        .await
        .unwrap()
        .into_buffer()
        .into_vec();

        let host_evals = mle.eval_at(&point).await.to_vec();
        assert_eq!(evals, host_evals);
    }

    #[tokio::test]
    async fn test_partial_geq() {
        let mut rng = rand::thread_rng();

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;

        let num_variables = 14;
        let point = Point::<EF>::rand(&mut rng, num_variables);

        for threshold in 0..1 << num_variables {
            let point_ref = &point;
            let eval = crate::run_in_place(|t| async move {
                let partial_geq = partial_geq::<F>(threshold, num_variables as usize, &t).await;
                let d_point = point_ref.copy_into(&t);
                let eval = Mle::new(partial_geq).eval_at(&d_point).await;
                eval.into_evaluations().into_host().await.unwrap()
            })
            .await
            .await
            .unwrap();

            let copied_eval = eval.into_host().await.unwrap();

            let host_evals =
                full_geq(&Point::<EF>::from_usize(threshold, num_variables as usize), &point);
            let eval = copied_eval.as_slice().to_vec();
            assert_eq!(eval.len(), 1);
            assert_eq!(eval[0], host_evals);
        }
    }
}

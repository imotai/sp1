use csl_device::{
    args,
    cuda::TaskScope,
    sys::mle::{partial_lagrange_baby_bear, partial_lagrange_baby_bear_extension},
    tensor::DotBackend,
    GlobalAllocator, KernelPtr, Tensor,
};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field};
use slop_baby_bear::BabyBear;

use crate::Mle;

use super::{MleBaseBackend, MleEvaluationBackend, PartialLargangeBackend, PointBackend};

/// # Safety
///
pub unsafe trait PartialLagrangeKernel<F: Field> {
    fn partial_lagrange_kernel() -> KernelPtr;
}

impl<F: Field> PointBackend<F> for TaskScope {
    #[inline]
    fn dimension(values: &Tensor<F, Self>) -> usize {
        values.sizes()[0]
    }
}

impl<F: Field> MleBaseBackend<F> for TaskScope {
    #[inline]
    fn uninit_mle(&self, num_polynomials: usize, num_variables: usize) -> Tensor<F, Self> {
        Tensor::with_sizes_in([num_polynomials, 1 << num_variables], self.clone())
    }

    #[inline]
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize {
        guts.sizes()[0]
    }

    #[inline]
    fn num_variables(guts: &Tensor<F, Self>) -> u32 {
        guts.sizes()[1].ilog2()
    }
}

impl<F: Field> PartialLargangeBackend<F> for TaskScope
where
    TaskScope: PartialLagrangeKernel<F>,
{
    fn partial_lagrange(point: &Tensor<F, Self>) -> Tensor<F, Self> {
        let dimension = Self::dimension(point);
        let mut eq = point.scope().uninit_mle(1, dimension);
        unsafe {
            eq.assume_init();
            let block_dim = 256;
            let grid_dim = ((1 << dimension) as u32).div_ceil(block_dim);
            let args = args!(eq.as_mut_ptr(), point.as_ptr(), dimension);
            point
                .scope()
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

impl<F: Field, EF: ExtensionField<F>> MleEvaluationBackend<F, EF> for TaskScope
where
    TaskScope: PartialLagrangeKernel<EF> + DotBackend<F, EF>,
{
    fn eval_mle_at_point(mle: &Tensor<F, Self>, point: &Tensor<EF, Self>) -> Tensor<EF, Self> {
        let eq = Self::partial_lagrange(point);
        let eq_view = eq.as_view().flatten();
        mle.dot(eq_view, 1)
    }
}

unsafe impl PartialLagrangeKernel<BabyBear> for TaskScope {
    fn partial_lagrange_kernel() -> KernelPtr {
        unsafe { partial_lagrange_baby_bear() }
    }
}

unsafe impl PartialLagrangeKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn partial_lagrange_kernel() -> KernelPtr {
        unsafe { partial_lagrange_baby_bear_extension() }
    }
}

impl<F: Field> Mle<F, TaskScope> {
    #[inline]
    pub async fn into_host(self) -> Mle<F, GlobalAllocator> {
        let new_sizes = [1 << self.num_variables(), self.num_polynomials()];
        Mle::new(self.into_guts().into_host().await.unwrap().reshape(new_sizes).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use csl_device::{DeviceTensor, HostTensor};
    use itertools::Itertools;
    use rand::Rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_multilinear::Point as SlopPoint;

    use crate::Point;

    use super::*;

    #[tokio::test]
    async fn test_partial_lagrange_eval_baby_bear() {
        let mut rng = rand::thread_rng();

        for dimension in [4, 6, 10, 21, 22] {
            let point_values =
                HostTensor::from((0..dimension).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>());

            let point_values_sent = point_values.clone();
            let eq = csl_device::cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let point_values_device =
                        DeviceTensor::from_host(point_values_sent.clone(), t.clone())
                            .await
                            .unwrap();
                    let z = Point::new(point_values_device);
                    t.synchronize_blocking().unwrap();
                    let time = std::time::Instant::now();
                    let eq = Mle::partial_lagrange(&z);
                    t.synchronize_blocking().unwrap();
                    println!("Device time: {:?}", time.elapsed());
                    eq.into_host().await
                })
                .await
                .await
                .unwrap();

            assert_eq!(eq.num_variables(), dimension);

            let point = SlopPoint::new(point_values.into_buffer().into_vec());
            let time = std::time::Instant::now();
            let eq_expected = slop_multilinear::partial_lagrange_eval(&point);
            println!("Host time: {:?}", time.elapsed());

            let eq_from_device = eq.into_guts().into_buffer().into_vec();

            for (val, expected) in eq_from_device.iter().zip_eq(eq_expected.iter()) {
                assert_eq!(val, expected);
            }
        }
    }

    #[tokio::test]
    async fn test_partial_lagrange_eval_baby_bear_extension() {
        let mut rng = rand::thread_rng();

        type EF = BinomialExtensionField<BabyBear, 4>;

        for dimension in [4, 6, 8, 10, 20, 21] {
            println!("Dimension: {}", dimension);
            let point_values =
                HostTensor::from((0..dimension).map(|_| rng.gen::<EF>()).collect::<Vec<_>>());

            let point_values_sent = point_values.clone();
            let eq = csl_device::cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let point_values_device =
                        DeviceTensor::from_host(point_values_sent.clone(), t.clone())
                            .await
                            .unwrap();
                    let z = Point::new(point_values_device);
                    t.synchronize_blocking().unwrap();
                    let time = std::time::Instant::now();
                    let eq = Mle::partial_lagrange(&z);
                    t.synchronize_blocking().unwrap();
                    println!("Device time: {:?}", time.elapsed());
                    eq.into_host().await
                })
                .await
                .await
                .unwrap();

            assert_eq!(eq.num_variables(), dimension);

            let point = SlopPoint::new(point_values.into_buffer().into_vec());
            let time = std::time::Instant::now();
            let eq_expected = slop_multilinear::partial_lagrange_eval(&point);
            println!("Host time: {:?}", time.elapsed());

            let eq_from_device = eq.into_guts().into_buffer().into_vec();

            for (val, expected) in eq_from_device.iter().zip_eq(eq_expected.iter()) {
                assert_eq!(val, expected);
            }
        }
    }

    #[tokio::test]
    async fn test_eval_mle_at_point_baby_bear_extension() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let num_polynomials = 128;

        for dimension in [4, 6, 8, 10, 16] {
            let values_base = HostTensor::from(
                (0..num_polynomials * (1 << dimension)).map(|_| rng.gen::<F>()).collect::<Vec<_>>(),
            )
            .reshape([num_polynomials, 1 << dimension])
            .unwrap();

            let point_values =
                HostTensor::from((0..dimension).map(|_| rng.gen::<EF>()).collect::<Vec<_>>());

            let point_values_sent = point_values.clone();
            let values_base_sent = values_base.clone();
            let evals = csl_device::cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let point_values_device =
                        DeviceTensor::from_host(point_values_sent.clone(), t.clone())
                            .await
                            .unwrap();
                    let z = Point::new(point_values_device);
                    let base_values_device =
                        DeviceTensor::from_host(values_base_sent.clone(), t.clone()).await.unwrap();
                    let mle = Mle::new(base_values_device);
                    t.synchronize_blocking().unwrap();
                    let time = std::time::Instant::now();
                    let values = mle.eval_at(&z);
                    t.synchronize_blocking().unwrap();
                    println!("Device time for dimension {}: {:?}", dimension, time.elapsed());
                    values.into_evalutions().into_host().await.unwrap()
                })
                .await
                .await
                .unwrap();

            let point = SlopPoint::new(point_values.into_buffer().into_vec());

            for i in 0..num_polynomials {
                let host_mle = slop_multilinear::Mle::new(values_base.index(i).as_slice().to_vec());
                let host_eval = host_mle.eval_at_point(&point);
                assert_eq!(*evals[[i]], host_eval);
            }
        }
    }
}

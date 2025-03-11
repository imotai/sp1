use std::future::Future;

use csl_cuda::{gkr_circuit_transition, TaskScope};
use csl_jagged::Poseidon2BabyBearJaggedCudaProverComponents;
use slop_jagged::JaggedProverComponents;
use sp1_stark::{GkrBackend, GkrProver};

#[derive(Clone, Debug)]
pub struct Poseidon2BabyBearGkrCudaProverComponents(
    pub Poseidon2BabyBearJaggedCudaProverComponents,
);

impl GkrProver for Poseidon2BabyBearGkrCudaProverComponents
where
    TaskScope: GkrBackend<
        <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::F,
        <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::EF,
    >,
{
    type F = <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::F;

    type EF = <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::EF;

    type Challenger =
        <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::Challenger;

    type B = TaskScope;

    async fn generate_gkr_input_mles(
        preprocessed: Option<&slop_multilinear::PaddedMle<Self::F, Self::B>>,
        main: &slop_multilinear::PaddedMle<Self::F, Self::B>,
        sends: &[sp1_stark::Interaction<Self::F>],
        receives: &[sp1_stark::Interaction<Self::F>],
        alpha: Self::EF,
        betas: &slop_algebra::Powers<Self::EF>,
        log_max_row_height: usize,
    ) -> (
        slop_multilinear::PaddedMle<Self::F, Self::B>,
        slop_multilinear::PaddedMle<Self::EF, Self::B>,
    ) {
        csl_cuda::generate_gkr_input_mles(
            preprocessed,
            main,
            sends,
            receives,
            alpha,
            betas,
            log_max_row_height,
        )
        .await
    }

    fn first_circuit_layer_execution(
        numerator_input_mle: &slop_multilinear::PaddedMle<Self::F, Self::B>,
        denom_input_mle: &slop_multilinear::PaddedMle<Self::EF, Self::B>,
    ) -> impl Future<
        Output = (
            slop_multilinear::PaddedMle<Self::EF, Self::B>,
            slop_multilinear::PaddedMle<Self::EF, Self::B>,
        ),
    > + Send {
        gkr_circuit_transition(numerator_input_mle, denom_input_mle)
    }

    fn circuit_layer_execution(
        current_numerator_mle: &slop_multilinear::PaddedMle<Self::EF, Self::B>,
        current_denom_mle: &slop_multilinear::PaddedMle<Self::EF, Self::B>,
    ) -> impl Future<
        Output = (
            slop_multilinear::PaddedMle<Self::EF, Self::B>,
            slop_multilinear::PaddedMle<Self::EF, Self::B>,
        ),
    > + Send {
        gkr_circuit_transition(current_numerator_mle, current_denom_mle)
    }
}

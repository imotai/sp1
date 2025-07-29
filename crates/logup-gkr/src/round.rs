use std::iter::once;

use csl_cuda::{
    args,
    sys::{
        logup_gkr::{
            logup_gkr_fix_last_row_last_circuit_layer_kernel_circuit_layer_baby_bear_extension,
            logup_gkr_fix_last_variable_circuit_layer_kernel_baby_bear_extension,
            logup_gkr_fix_last_variable_first_layer_kernel_baby_bear,
            logup_gkr_fix_last_variable_interactions_layer_kernel_baby_bear_extension,
            logup_gkr_sum_as_poly_first_layer_kernel_baby_bear,
            logup_gkr_sum_as_poly_layer_kernel_circuit_layer_baby_bear_extension,
            logup_gkr_sum_as_poly_layer_kernel_interactions_layer_baby_bear_extension,
        },
        runtime::KernelPtr,
    },
    PartialLagrangeKernel, TaskScope, ToDevice,
};
use slop_algebra::{
    extension::BinomialExtensionField, interpolate_univariate_polynomial, AbstractField,
    ExtensionField, Field, UnivariatePolynomial,
};
use slop_alloc::{Buffer, ToHost};
use slop_baby_bear::BabyBear;
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, MleEvaluationBackend, MleFixLastVariableBackend, Point, PointBackend};
use slop_sumcheck::{
    reduce_sumcheck_to_evaluation, ComponentPoly, SumcheckPoly, SumcheckPolyBase,
    SumcheckPolyFirstRound,
};
use slop_tensor::{ReduceSumBackend, Tensor};
use sp1_stark::{LogUpGkrRoundProver, LogupGkrRoundProof};

use crate::{FirstGkrLayer, GkrCircuitLayer, GkrLayer};

const STRIDE_FACTOR: usize = 1 << 17;

#[derive(Clone, Debug)]
pub struct LogupGkrCudaRoundProver<F, EF, Challenger> {
    _marker: std::marker::PhantomData<(F, EF, Challenger)>,
}

impl<F: Field, EF: ExtensionField<F>, Challenger> Default
    for LogupGkrCudaRoundProver<F, EF, Challenger>
{
    fn default() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}

/// # Safety
///
/// TODO
pub unsafe trait MaterializedLayerKernels<EF: AbstractField>:
    MleFixLastVariableBackend<EF, EF> + ReduceSumBackend<EF>
{
    fn partial_sum_as_poly_circuit_layer() -> KernelPtr;
    fn partial_sum_as_poly_interactions_layer() -> KernelPtr;
    fn logup_gkr_fix_last_variable_circuit_layer() -> KernelPtr;
    fn logup_gkr_fix_last_variable_last_circuit_layer() -> KernelPtr;
    fn logup_gkr_fix_last_variable_interactions_layer() -> KernelPtr;
}

/// # Safety
///
/// TODO
pub unsafe trait FirstLayerKernels<F: Field, EF: ExtensionField<F>> {
    fn logup_gkr_fix_last_variable_first_layer() -> KernelPtr;
    fn logup_gkr_sum_as_poly_first_layer() -> KernelPtr;
}

impl<F, EF, Challenger> LogupGkrCudaRoundProver<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    TaskScope:
        MleEvaluationBackend<EF, EF> + PartialLagrangeKernel<EF> + MaterializedLayerKernels<EF>,
{
    async fn prove_materialized_round(
        &self,
        layer: GkrLayer<EF>,
        eval_point: &Point<EF>,
        numerator_eval: EF,
        denominator_eval: EF,
        challenger: &mut Challenger,
    ) -> LogupGkrRoundProof<EF> {
        let lambda = challenger.sample_ext_element::<EF>();
        let (interaction_point, row_point) =
            eval_point.split_at(layer.num_interaction_variables as usize);
        let backend = layer.layer.backend().clone();
        let interaction_point = interaction_point.to_device_in(&backend).await.unwrap();
        let row_point = row_point.to_device_in(&backend).await.unwrap();
        let eq_interaction = Mle::partial_lagrange(&interaction_point).await;
        let eq_row = Mle::partial_lagrange(&row_point).await;
        let sumcheck_poly = LogupRoundPolynomial {
            layer: PolynomialLayer::CircuitLayer(layer),
            eq_row,
            eq_interaction,
            lambda,
            eq_adjustment: EF::one(),
            padding_adjustment: EF::one(),
            point: eval_point.clone(),
        };
        let claim = numerator_eval * lambda + denominator_eval;
        // Produce the suncheck proof.
        let (sumcheck_proof, mut openings) =
            reduce_sumcheck_to_evaluation(vec![sumcheck_poly], challenger, vec![claim], 1, lambda)
                .await;
        let openings = openings.pop().unwrap();
        let [numerator_0, numerator_1, denominator_0, denominator_1] = openings.try_into().unwrap();

        LogupGkrRoundProof {
            numerator_0,
            numerator_1,
            denominator_0,
            denominator_1,
            sumcheck_proof,
        }
    }

    async fn prove_first_round(
        &self,
        layer: FirstGkrLayer<F, EF>,
        eval_point: &Point<EF>,
        numerator_eval: EF,
        denominator_eval: EF,
        challenger: &mut Challenger,
    ) -> LogupGkrRoundProof<EF>
    where
        TaskScope: FirstLayerKernels<F, EF>,
    {
        let lambda = challenger.sample_ext_element::<EF>();
        let (interaction_point, row_point) =
            eval_point.split_at(layer.num_interaction_variables as usize);
        let backend = layer.layer.numerator.backend();
        let interaction_point = interaction_point.to_device_in(backend).await.unwrap();
        let row_point = row_point.to_device_in(backend).await.unwrap();
        let eq_interaction = Mle::partial_lagrange(&interaction_point).await;
        let eq_row = Mle::partial_lagrange(&row_point).await;

        let sumcheck_poly = FirstLayerPolynomial {
            layer,
            eq_row,
            eq_interaction,
            lambda,
            point: eval_point.clone(),
        };
        let claim = numerator_eval * lambda + denominator_eval;
        // Produce the suncheck proof.
        let (sumcheck_proof, mut openings) =
            reduce_sumcheck_to_evaluation(vec![sumcheck_poly], challenger, vec![claim], 1, lambda)
                .await;
        let openings = openings.pop().unwrap();
        let [numerator_0, numerator_1, denominator_0, denominator_1] = openings.try_into().unwrap();

        LogupGkrRoundProof {
            numerator_0,
            numerator_1,
            denominator_0,
            denominator_1,
            sumcheck_proof,
        }
    }
}

impl<F, EF, Challenger> LogUpGkrRoundProver<F, EF, Challenger, TaskScope>
    for LogupGkrCudaRoundProver<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + 'static + Send + Sync,
    TaskScope: MleEvaluationBackend<EF, EF>
        + PartialLagrangeKernel<EF>
        + MaterializedLayerKernels<EF>
        + FirstLayerKernels<F, EF>,
{
    type CircuitLayer = GkrCircuitLayer<F, EF>;

    async fn prove_round(
        &self,
        circuit: Self::CircuitLayer,
        eval_point: &Point<EF>,
        numerator_eval: EF,
        denominator_eval: EF,
        challenger: &mut Challenger,
    ) -> LogupGkrRoundProof<EF> {
        match circuit {
            GkrCircuitLayer::Materialized(layer) => {
                self.prove_materialized_round(
                    layer,
                    eval_point,
                    numerator_eval,
                    denominator_eval,
                    challenger,
                )
                .await
            }
            GkrCircuitLayer::FirstLayer(layer) => {
                self.prove_first_round(
                    layer,
                    eval_point,
                    numerator_eval,
                    denominator_eval,
                    challenger,
                )
                .await
            }
            GkrCircuitLayer::FirstLayerVirtual(_) => unreachable!(),
        }
    }
}

pub enum PolynomialLayer<EF> {
    CircuitLayer(GkrLayer<EF>),
    InteractionsLayer(Tensor<EF, TaskScope>),
}

pub struct LogupRoundPolynomial<EF> {
    // The values of the numerator and denominator polynomials
    layer: PolynomialLayer<EF>,
    /// The partial lagrange evaluation for the row variables
    eq_row: Mle<EF, TaskScope>,
    /// The partial lagrange evaluation for the interaction variables
    eq_interaction: Mle<EF, TaskScope>,
    /// The correcton term for the eq polynomial.
    eq_adjustment: EF,
    /// The correction term for padding
    padding_adjustment: EF,
    /// The batching factor for the numerator and denominator claims.
    lambda: EF,
    /// The random point for the current GKR round.
    point: Point<EF>,
}

impl<EF: Field> SumcheckPolyBase for LogupRoundPolynomial<EF> {
    fn num_variables(&self) -> u32 {
        self.eq_row.num_variables() + self.eq_interaction.num_variables()
    }
}

impl<EF: Field> ComponentPoly<EF> for LogupRoundPolynomial<EF>
where
    TaskScope: MleEvaluationBackend<EF, EF> + PointBackend<EF>,
{
    async fn get_component_poly_evals(&self) -> Vec<EF> {
        match &self.layer {
            PolynomialLayer::InteractionsLayer(guts) => {
                debug_assert_eq!(guts.sizes(), [4, 1]);
                guts.as_buffer().to_host().await.unwrap().to_vec()
            }
            PolynomialLayer::CircuitLayer(_) => unreachable!(),
        }
    }
}

impl<EF: Field> SumcheckPoly<EF> for LogupRoundPolynomial<EF>
where
    TaskScope: MleEvaluationBackend<EF, EF> + PointBackend<EF> + MaterializedLayerKernels<EF>,
{
    async fn fix_last_variable(mut self, alpha: EF) -> Self {
        // Remove the last coordinate from the point
        let last_coordinate = self.point.remove_last_coordinate();
        let padding_adjustment = self.padding_adjustment
            * (last_coordinate * alpha + (EF::one() - last_coordinate) * (EF::one() - alpha));

        match &self.layer {
            PolynomialLayer::InteractionsLayer(guts) => {
                let height = guts.sizes()[1];
                let output_height = height.div_ceil(2);
                let backend = guts.backend();

                let mut output = Tensor::with_sizes_in([4, output_height], backend.clone());

                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
                let grid_size = (grid_size_x, 1, 1);

                unsafe {
                    let args =
                        args!(guts.as_ptr(), output.as_mut_ptr(), alpha, height, output_height);
                    output.assume_init();
                    backend
                        .launch_kernel(
                            TaskScope::logup_gkr_fix_last_variable_interactions_layer(),
                            grid_size,
                            BLOCK_SIZE,
                            &args,
                            0,
                        )
                        .unwrap();
                }

                let layer = PolynomialLayer::InteractionsLayer(output);

                let eq_interaction = self.eq_interaction.fix_last_variable(alpha).await;

                LogupRoundPolynomial {
                    layer,
                    eq_row: self.eq_row,
                    eq_interaction,
                    lambda: self.lambda,
                    point: self.point,
                    eq_adjustment: self.eq_adjustment,
                    padding_adjustment,
                }
            }
            PolynomialLayer::CircuitLayer(circuit) => {
                let backend = circuit.layer.backend();
                let height = circuit.layer.sizes()[2];
                // If this is the last layer, we need to fix the last variable and create an
                // interaction layer.
                if circuit.num_row_variables == 1 {
                    let mut output: Tensor<EF, TaskScope> =
                        Tensor::with_sizes_in([4, height], backend.clone());

                    const BLOCK_SIZE: usize = 256;
                    let stride = height.div_ceil(STRIDE_FACTOR);
                    let grid_size_x = height.div_ceil(BLOCK_SIZE * stride);
                    let grid_size = (grid_size_x, 1, 1);
                    unsafe {
                        let args =
                            args!(circuit.layer.as_ptr(), alpha, output.as_mut_ptr(), height);
                        output.assume_init();
                        backend
                            .launch_kernel(
                                TaskScope::logup_gkr_fix_last_variable_last_circuit_layer(),
                                grid_size,
                                BLOCK_SIZE,
                                &args,
                                0,
                            )
                            .unwrap();
                    }

                    let eq_row = self.eq_row.fix_last_variable(alpha).await;

                    return LogupRoundPolynomial {
                        layer: PolynomialLayer::InteractionsLayer(output),
                        eq_row,
                        eq_interaction: self.eq_interaction,
                        lambda: self.lambda,
                        point: self.point,
                        eq_adjustment: padding_adjustment,
                        padding_adjustment: EF::one(),
                    };
                }

                let output_interaction_row_counts = circuit
                    .interaction_row_counts
                    .iter()
                    .map(|count| count.div_ceil(2))
                    .collect::<Vec<_>>();
                // The output indices is just the prefix sum of the interaction row counts.
                let output_interaction_start_indices = once(0)
                    .chain(output_interaction_row_counts.iter().scan(0u32, |acc, x| {
                        *acc += x;
                        Some(*acc)
                    }))
                    .collect::<Buffer<_>>();
                let output_height =
                    output_interaction_start_indices.last().copied().unwrap() as usize;
                let output_interaction_start_indices =
                    output_interaction_start_indices.to_device_in(backend).await.unwrap();

                // Create a new layer
                let mut output_layer: Tensor<EF, TaskScope> =
                    Tensor::with_sizes_in([4, 2, output_height], backend.clone());
                let mut output_interaction_data: Buffer<u32, TaskScope> =
                    Buffer::with_capacity_in(output_height, backend.clone());

                // populate the new layer
                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
                let grid_size = (grid_size_x, 1, 1);
                let block_dim = BLOCK_SIZE;
                unsafe {
                    output_layer.assume_init();
                    output_interaction_data.assume_init();
                    let args = args!(
                        circuit.layer.as_ptr(),
                        circuit.interaction_data.as_ptr(),
                        circuit.interaction_start_indices.as_ptr(),
                        alpha,
                        output_layer.as_mut_ptr(),
                        output_interaction_data.as_mut_ptr(),
                        output_interaction_start_indices.as_ptr(),
                        height,
                        output_height
                    );
                    backend
                        .launch_kernel(
                            TaskScope::logup_gkr_fix_last_variable_circuit_layer(),
                            grid_size,
                            block_dim,
                            &args,
                            0,
                        )
                        .unwrap();
                }
                // Fix the eq_row variables
                let eq_row = self.eq_row.fix_last_variable(alpha).await;

                let output_layer = GkrLayer {
                    layer: output_layer,
                    interaction_data: output_interaction_data,
                    interaction_start_indices: output_interaction_start_indices,
                    interaction_row_counts: output_interaction_row_counts,
                    num_row_variables: circuit.num_row_variables - 1,
                    num_interaction_variables: circuit.num_interaction_variables,
                };

                LogupRoundPolynomial {
                    layer: PolynomialLayer::CircuitLayer(output_layer),
                    eq_row,
                    eq_interaction: self.eq_interaction,
                    lambda: self.lambda,
                    point: self.point,
                    eq_adjustment: self.eq_adjustment,
                    padding_adjustment,
                }
            }
        }
    }

    async fn sum_as_poly_in_last_variable(&self, claim: Option<EF>) -> UnivariatePolynomial<EF> {
        let claim = claim.expect("Claim is required for the sumcheck polynomial");

        let (mut eval_zero, mut eval_half, eq_sum) = match &self.layer {
            PolynomialLayer::CircuitLayer(circuit) => {
                // println!("number of row variables: {}", circuit.num_row_variables);
                let height = circuit.layer.sizes()[2];
                let scope = circuit.layer.backend();

                const BLOCK_SIZE: usize = 256;
                let stride = height.div_ceil(STRIDE_FACTOR);
                let grid_dim = height.div_ceil(BLOCK_SIZE).div_ceil(stride);
                let mut output =
                    Tensor::<EF, TaskScope>::with_sizes_in([3, grid_dim], scope.clone());
                let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
                let shared_mem = num_tiles * std::mem::size_of::<EF>();
                unsafe {
                    output.assume_init();
                    let args = args!(
                        output.as_mut_ptr(),
                        circuit.layer.as_ptr(),
                        circuit.interaction_data.as_ptr(),
                        circuit.interaction_start_indices.as_ptr(),
                        self.eq_row.guts().as_ptr(),
                        self.eq_interaction.guts().as_ptr(),
                        self.lambda,
                        height
                    );
                    scope
                        .launch_kernel(
                            TaskScope::partial_sum_as_poly_circuit_layer(),
                            grid_dim,
                            BLOCK_SIZE,
                            &args,
                            shared_mem,
                        )
                        .unwrap();
                }
                let evals = output.sum(1).await.into_buffer().to_host().await.unwrap();

                let eval_zero: EF = *evals[0];
                let eval_half: EF = *evals[1];
                let eq_sum = *evals[2];
                (eval_zero, eval_half, eq_sum)
            }
            PolynomialLayer::InteractionsLayer(guts) => {
                let height = guts.sizes()[1];
                let output_height = height.div_ceil(2);

                let scope = guts.backend();

                const BLOCK_SIZE: usize = 256;
                let stride = height.div_ceil(STRIDE_FACTOR);
                let grid_dim = output_height.div_ceil(BLOCK_SIZE).div_ceil(stride);
                let mut univariate_evals =
                    Tensor::<EF, TaskScope>::with_sizes_in([3, grid_dim], scope.clone());

                let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
                let shared_mem = num_tiles * std::mem::size_of::<EF>();

                unsafe {
                    univariate_evals.assume_init();
                    let args = args!(
                        univariate_evals.as_mut_ptr(),
                        guts.as_ptr(),
                        self.eq_interaction.guts().as_ptr(),
                        self.lambda,
                        height,
                        output_height
                    );
                    scope
                        .launch_kernel(
                            TaskScope::partial_sum_as_poly_interactions_layer(),
                            grid_dim,
                            BLOCK_SIZE,
                            &args,
                            shared_mem,
                        )
                        .unwrap();
                }
                let evals = univariate_evals.sum(1).await.into_buffer().to_host().await.unwrap();
                let eval_zero: EF = *evals[0];
                let eval_half: EF = *evals[1];
                let eq_sum = *evals[2];
                (eval_zero, eval_half, eq_sum)
            }
        };

        // Correct the evaluations by the sum of the eq polynomial, which accounts for the
        // contribution of padded row for the denominator expression
        // `\Sum_i eq * denominator_0 * denominator_1`.
        let eq_correction_term = self.padding_adjustment - eq_sum;
        // The evaluation at zero just gets the eq correction term.
        eval_zero += eq_correction_term * (EF::one() - *self.point.last().unwrap());
        // The evaluation at 1/2 gets the eq correction term times 4, since the denominators
        // have a 1/2 in them for the rest of the evaluations (so we multiply by 2 twice).
        eval_half += eq_correction_term * EF::from_canonical_u16(4);

        // Since the sumcheck polynomial is homogeneous of degree 3, we need to divide by
        // 8 = 2^3 to account for the evaluations at 1/2 to be double their true value.
        let eval_half = eval_half * EF::from_canonical_u16(8).inverse();

        let eval_zero = eval_zero * self.eq_adjustment;
        let eval_half = eval_half * self.eq_adjustment;

        // Get the root of the eq polynomial which gives an evaluation of zero.
        let point_last = self.point.last().unwrap();
        let b_const = (EF::one() - *point_last) / (EF::one() - point_last.double());

        let eval_one = claim - eval_zero;

        interpolate_univariate_polynomial(
            &[
                EF::from_canonical_u16(0),
                EF::from_canonical_u16(1),
                EF::from_canonical_u16(2).inverse(),
                b_const,
            ],
            &[eval_zero, eval_one, eval_half, EF::zero()],
        )
    }
}

impl<EF: Field> SumcheckPolyFirstRound<EF> for LogupRoundPolynomial<EF>
where
    TaskScope: MleEvaluationBackend<EF, EF> + PointBackend<EF> + MaterializedLayerKernels<EF>,
{
    type NextRoundPoly = LogupRoundPolynomial<EF>;

    async fn fix_t_variables(self, alpha: EF, t: usize) -> Self::NextRoundPoly {
        debug_assert_eq!(t, 1);
        self.fix_last_variable(alpha).await
    }

    async fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        debug_assert_eq!(t, 1);
        self.sum_as_poly_in_last_variable(claim).await
    }
}

pub struct FirstLayerPolynomial<F, EF> {
    layer: FirstGkrLayer<F, EF>,
    eq_row: Mle<EF, TaskScope>,
    eq_interaction: Mle<EF, TaskScope>,
    lambda: EF,
    point: Point<EF>,
}

impl<F, EF> SumcheckPolyBase for FirstLayerPolynomial<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn num_variables(&self) -> u32 {
        self.eq_row.num_variables() + self.eq_interaction.num_variables()
    }
}

impl<F, EF> SumcheckPolyFirstRound<EF> for FirstLayerPolynomial<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    TaskScope: MleEvaluationBackend<EF, EF>
        + PointBackend<EF>
        + MaterializedLayerKernels<EF>
        + FirstLayerKernels<F, EF>,
{
    type NextRoundPoly = LogupRoundPolynomial<EF>;

    async fn fix_t_variables(mut self, alpha: EF, t: usize) -> Self::NextRoundPoly {
        debug_assert_eq!(t, 1);
        // Remove the last coordinate from the point
        let last_coordinate = self.point.remove_last_coordinate();
        let padding_adjustment =
            last_coordinate * alpha + (EF::one() - last_coordinate) * (EF::one() - alpha);

        let backend = self.layer.layer.numerator.backend();
        let height = self.layer.layer.numerator.sizes()[2];
        // If this is not the last layer, we need to fix the last variable and create a
        // new circuit layer.
        let output_interaction_row_counts = self
            .layer
            .interaction_row_counts
            .iter()
            .map(|count| count.div_ceil(2))
            .collect::<Vec<_>>();
        // The output indices is just the prefix sum of the interaction row counts.
        let output_interaction_start_indices = once(0)
            .chain(output_interaction_row_counts.iter().scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            }))
            .collect::<Buffer<_>>();
        let output_height = output_interaction_start_indices.last().copied().unwrap() as usize;
        let output_interaction_start_indices =
            output_interaction_start_indices.to_device_in(backend).await.unwrap();

        // Create a new layer
        let mut output_layer: Tensor<EF, TaskScope> =
            Tensor::with_sizes_in([4, 2, output_height], backend.clone());
        let mut output_interaction_data: Buffer<u32, TaskScope> =
            Buffer::with_capacity_in(output_height, backend.clone());

        // populate the new layer
        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 32;
        let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size = (grid_size_x, 1, 1);
        let block_dim = BLOCK_SIZE;
        unsafe {
            output_layer.assume_init();
            output_interaction_data.assume_init();
            let args = args!(
                self.layer.layer.numerator.as_ptr(),
                self.layer.layer.denominator.as_ptr(),
                self.layer.interaction_data.as_ptr(),
                self.layer.interaction_start_indices.as_ptr(),
                alpha,
                output_layer.as_mut_ptr(),
                output_interaction_data.as_mut_ptr(),
                output_interaction_start_indices.as_ptr(),
                height,
                output_height
            );
            backend
                .launch_kernel(
                    TaskScope::logup_gkr_fix_last_variable_first_layer(),
                    grid_size,
                    block_dim,
                    &args,
                    0,
                )
                .unwrap();
        }
        // Fix the eq_row variables
        let eq_row = self.eq_row.fix_last_variable(alpha).await;

        let output_layer = GkrLayer {
            layer: output_layer,
            interaction_data: output_interaction_data,
            interaction_start_indices: output_interaction_start_indices,
            interaction_row_counts: output_interaction_row_counts,
            num_row_variables: self.layer.num_row_variables - 1,
            num_interaction_variables: self.layer.num_interaction_variables,
        };

        LogupRoundPolynomial {
            layer: PolynomialLayer::CircuitLayer(output_layer),
            eq_row,
            eq_interaction: self.eq_interaction,
            lambda: self.lambda,
            point: self.point,
            eq_adjustment: EF::one(),
            padding_adjustment,
        }
    }

    async fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        debug_assert_eq!(t, 1);
        let claim = claim.unwrap();

        let circuit = &self.layer.layer;

        let height = circuit.numerator.sizes()[2];

        let scope = circuit.numerator.backend();

        const BLOCK_SIZE: usize = 256;

        let stride = height.div_ceil(STRIDE_FACTOR);
        let grid_dim = height.div_ceil(BLOCK_SIZE).div_ceil(stride);
        let mut output = Tensor::<EF, TaskScope>::with_sizes_in([3, grid_dim], scope.clone());

        let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
        let shared_mem = num_tiles * std::mem::size_of::<EF>();

        unsafe {
            output.assume_init();
            let args = args!(
                output.as_mut_ptr(),
                circuit.numerator.as_ptr(),
                circuit.denominator.as_ptr(),
                self.layer.interaction_data.as_ptr(),
                self.layer.interaction_start_indices.as_ptr(),
                self.eq_row.guts().as_ptr(),
                self.eq_interaction.guts().as_ptr(),
                self.lambda,
                height
            );
            scope
                .launch_kernel(
                    TaskScope::logup_gkr_sum_as_poly_first_layer(),
                    grid_dim,
                    BLOCK_SIZE,
                    &args,
                    shared_mem,
                )
                .unwrap();
        }
        let evals = output.sum(1).await.into_buffer().to_host().await.unwrap();

        let mut eval_zero: EF = *evals[0];
        let mut eval_half: EF = *evals[1];
        let eq_sum = *evals[2];

        // Correct the evaluations by the sum of the eq polynomial, which accounts for the
        // contribution of padded row for the denominator expression
        // `\Sum_i eq * denominator_0 * denominator_1`.
        let eq_correction_term = EF::one() - eq_sum;
        // The evaluation at zero just gets the eq correction term.
        eval_zero += eq_correction_term * (EF::one() - *self.point.last().unwrap());
        // The evaluation at 1/2 gets the eq correction term times 4, since the denominators
        // have a 1/2 in them for the rest of the evaluations (so we multiply by 2 twice).
        eval_half += eq_correction_term * EF::from_canonical_u16(4);

        // Since the sumcheck polynomial is homogeneous of degree 3, we need to divide by
        // 8 = 2^3 to account for the evaluations at 1/2 to be double their true value.
        let eval_half = eval_half * EF::from_canonical_u16(8).inverse();

        // Get the root of the eq polynomial which gives an evaluation of zero.
        let point_last = self.point.last().unwrap();
        let b_const = (EF::one() - *point_last) / (EF::one() - point_last.double());

        let eval_one = claim - eval_zero;

        interpolate_univariate_polynomial(
            &[
                EF::from_canonical_u16(0),
                EF::from_canonical_u16(1),
                EF::from_canonical_u16(2).inverse(),
                b_const,
            ],
            &[eval_zero, eval_one, eval_half, EF::zero()],
        )
    }
}

unsafe impl MaterializedLayerKernels<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn logup_gkr_fix_last_variable_circuit_layer() -> KernelPtr {
        unsafe { logup_gkr_fix_last_variable_circuit_layer_kernel_baby_bear_extension() }
    }

    fn partial_sum_as_poly_circuit_layer() -> KernelPtr {
        unsafe { logup_gkr_sum_as_poly_layer_kernel_circuit_layer_baby_bear_extension() }
    }

    fn partial_sum_as_poly_interactions_layer() -> KernelPtr {
        unsafe { logup_gkr_sum_as_poly_layer_kernel_interactions_layer_baby_bear_extension() }
    }

    fn logup_gkr_fix_last_variable_last_circuit_layer() -> KernelPtr {
        unsafe {
            logup_gkr_fix_last_row_last_circuit_layer_kernel_circuit_layer_baby_bear_extension()
        }
    }

    fn logup_gkr_fix_last_variable_interactions_layer() -> KernelPtr {
        unsafe { logup_gkr_fix_last_variable_interactions_layer_kernel_baby_bear_extension() }
    }
}

unsafe impl FirstLayerKernels<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn logup_gkr_fix_last_variable_first_layer() -> KernelPtr {
        unsafe { logup_gkr_fix_last_variable_first_layer_kernel_baby_bear() }
    }

    fn logup_gkr_sum_as_poly_first_layer() -> KernelPtr {
        unsafe { logup_gkr_sum_as_poly_first_layer_kernel_baby_bear() }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rayon::prelude::*;
    use slop_alloc::{Backend, CpuBackend, IntoHost};
    use slop_basefold::{BasefoldConfig, BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_sumcheck::partially_verify_sumcheck_proof;
    use std::iter::once;

    use crate::{FirstLayerData, LogUpGkrCudaTraceGenerator};

    use super::*;

    use csl_cuda::IntoDevice;
    use rand::{thread_rng, Rng};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    struct GkrTestData<F, B: Backend> {
        numerator_0: Mle<F, B>,
        numerator_1: Mle<F, B>,
        denominator_0: Mle<F, B>,
        denominator_1: Mle<F, B>,
    }

    async fn random_first_layer<R: Rng>(
        rng: &mut R,
        interaction_row_counts: Vec<u32>,
        num_row_variables: Option<u32>,
    ) -> FirstGkrLayer<F, EF, CpuBackend> {
        let max_row_variables =
            interaction_row_counts.iter().max().copied().unwrap().next_power_of_two().ilog2() + 1;

        let num_row_variables = if let Some(num_vars) = num_row_variables {
            assert!(num_vars >= max_row_variables);
            num_vars
        } else {
            max_row_variables
        };

        let num_interaction_variables = interaction_row_counts.len().next_power_of_two().ilog2();

        let interaction_start_indices = once(0)
            .chain(interaction_row_counts.iter().scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            }))
            .collect::<Buffer<_>>();
        let height = interaction_start_indices.last().copied().unwrap() as usize;
        let interaction_data = interaction_row_counts
            .iter()
            .enumerate()
            .flat_map(|(i, c)| {
                let dimension = c.next_power_of_two().ilog2() + 1;
                let data = i as u32 + (dimension << 24);
                vec![data; *c as usize]
            })
            .collect::<Buffer<_>>();

        let numerator = Tensor::<F>::rand(rng, [2, 2, height]);
        let denominator = Tensor::<EF>::rand(rng, [2, 2, height]);
        let layer_data = FirstLayerData { numerator, denominator };

        FirstGkrLayer {
            layer: layer_data,
            interaction_data,
            interaction_start_indices,
            interaction_row_counts,
            num_interaction_variables,
            num_row_variables,
        }
    }

    async fn random_layer<R: Rng>(
        rng: &mut R,
        interaction_row_counts: Vec<u32>,
        num_row_variables: Option<u32>,
    ) -> GkrLayer<EF, CpuBackend> {
        let max_row_variables =
            interaction_row_counts.iter().max().copied().unwrap().next_power_of_two().ilog2() + 1;

        let num_row_variables = if let Some(num_vars) = num_row_variables {
            assert!(num_vars >= max_row_variables);
            num_vars
        } else {
            max_row_variables
        };

        let num_interaction_variables = interaction_row_counts.len().next_power_of_two().ilog2();

        let interaction_start_indices = once(0)
            .chain(interaction_row_counts.iter().scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            }))
            .collect::<Buffer<_>>();
        let height = interaction_start_indices.last().copied().unwrap() as usize;
        let interaction_data = interaction_row_counts
            .iter()
            .enumerate()
            .flat_map(|(i, c)| {
                let dimension = c.next_power_of_two().ilog2() + 1;
                let data = i as u32 + (dimension << 24);
                vec![data; *c as usize]
            })
            .collect::<Buffer<_>>();

        let layer_data = Tensor::<EF>::rand(rng, [4, 2, height]);

        GkrLayer {
            layer: layer_data,
            interaction_data,
            interaction_start_indices,
            interaction_row_counts,
            num_interaction_variables,
            num_row_variables,
        }
    }

    async fn generate_test_data<R: Rng>(
        rng: &mut R,
        interaction_row_counts: Vec<u32>,
        num_row_variables: Option<u32>,
    ) -> (GkrLayer<EF, CpuBackend>, GkrTestData<EF, CpuBackend>) {
        let layer = random_layer(rng, interaction_row_counts, num_row_variables).await;
        let test_data = get_polys_from_layer(&layer);
        (layer, test_data)
    }

    fn get_polys_from_layer(layer: &GkrLayer<EF, CpuBackend>) -> GkrTestData<EF, CpuBackend> {
        let GkrLayer {
            layer: layer_data,
            interaction_row_counts,
            num_interaction_variables,
            num_row_variables,
            ..
        } = layer;

        let full_padded_height = 1usize << num_row_variables;
        let get_mle = |f_0: &[EF], f_1: &[EF], padding: EF| {
            let mut values = f_0.iter().interleave(f_1.iter()).copied().collect::<Vec<_>>();
            // Make the padded polynomials by padding each row dimension with the correct amount of
            // padding values.
            let num_padding_polynomials =
                (1 << num_interaction_variables) - interaction_row_counts.len();
            let padding_polynomials =
                (0..num_padding_polynomials).map(|_| vec![padding; full_padded_height]);
            let guts = interaction_row_counts
                .iter()
                .rev()
                .map(|h| {
                    let h = (*h as usize) << 1;
                    let split_point = values.len() - h;
                    let mut poly_real_values = values.split_off(split_point);
                    poly_real_values.resize(full_padded_height, padding);
                    poly_real_values
                })
                .collect::<Vec<_>>();
            let mut guts = guts;
            guts.reverse();
            let guts = guts.into_iter().chain(padding_polynomials).flatten().collect::<Vec<_>>();
            Mle::from(guts)
        };

        // Extract numerator_0, numerator_1, denominator_0, denominator_1 from the layer_data
        let numerator_0_0 = layer_data.get(0).unwrap().get(0).unwrap().as_slice();
        let numerator_1_0 = layer_data.get(0).unwrap().get(1).unwrap().as_slice();
        let numerator_0 = get_mle(numerator_0_0, numerator_1_0, EF::zero());

        let numerator_0_1 = layer_data.get(1).unwrap().get(0).unwrap().as_slice();
        let numerator_1_1 = layer_data.get(1).unwrap().get(1).unwrap().as_slice();
        let numerator_1 = get_mle(numerator_0_1, numerator_1_1, EF::zero());

        let denominator_0_0 = layer_data.get(2).unwrap().get(0).unwrap().as_slice();
        let denominator_1_0 = layer_data.get(2).unwrap().get(1).unwrap().as_slice();
        let denominator_0 = get_mle(denominator_0_0, denominator_1_0, EF::one());

        let denominator_0_1 = layer_data.get(3).unwrap().get(0).unwrap().as_slice();
        let denominator_1_1 = layer_data.get(3).unwrap().get(1).unwrap().as_slice();
        let denominator_1 = get_mle(denominator_0_1, denominator_1_1, EF::one());

        GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 }
    }

    #[tokio::test]
    async fn test_logup_round_polynomial_fix_last_variable() {
        let mut rng = thread_rng();

        let interaction_row_counts: Vec<u32> =
            vec![(1 << 8) + 2, (1 << 4), (1 << 10), 1 << 10, 1 << 8, 1 << 10, (1 << 8) + 2];
        let (layer, test_data) =
            generate_test_data(&mut rng, interaction_row_counts, Some(12)).await;
        let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

        let GkrLayer {
            layer: layer_data,
            interaction_data,
            interaction_start_indices,
            interaction_row_counts,
            num_interaction_variables,
            num_row_variables,
        } = layer;

        let poly_point = Point::<EF>::rand(&mut rng, num_row_variables + num_interaction_variables);
        let (interaction_point, row_point) =
            poly_point.split_at(num_interaction_variables as usize);

        let random_point =
            Point::<EF>::rand(&mut rng, num_row_variables + num_interaction_variables);

        let lambda = rng.gen::<EF>();

        csl_cuda::spawn(move |t| async move {
            let layer_data = layer_data.into_device_in(&t).await.unwrap();
            let interaction_data = interaction_data.into_device_in(&t).await.unwrap();
            let interaction_start_indices =
                interaction_start_indices.into_device_in(&t).await.unwrap();

            let row_point = row_point.to_device_in(&t).await.unwrap();
            let interaction_point = interaction_point.to_device_in(&t).await.unwrap();

            let eq_row = Mle::partial_lagrange(&row_point).await;
            let eq_interaction = Mle::partial_lagrange(&interaction_point).await;

            let layer = GkrLayer {
                layer: layer_data,
                interaction_data,
                interaction_start_indices,
                interaction_row_counts,
                num_interaction_variables,
                num_row_variables,
            };

            let mut polynomial = LogupRoundPolynomial {
                layer: PolynomialLayer::CircuitLayer(layer),
                eq_row,
                eq_interaction,
                lambda,
                eq_adjustment: EF::one(),
                padding_adjustment: EF::one(),
                point: poly_point,
            };

            // Get the exepcted evaluations
            let numerator_0_eval = numerator_0.eval_at(&random_point).await[0];
            let numerator_1_eval = numerator_1.eval_at(&random_point).await[0];
            let denominator_0_eval = denominator_0.eval_at(&random_point).await[0];
            let denominator_1_eval = denominator_1.eval_at(&random_point).await[0];

            for alpha in random_point.iter().rev() {
                polynomial = polynomial.fix_last_variable(*alpha).await;
            }

            // Get the values from the sumcheck polynomial
            let [n_0, n_1, d_0, d_1] =
                polynomial.get_component_poly_evals().await.try_into().unwrap();

            assert_eq!(numerator_0_eval, n_0);
            assert_eq!(numerator_1_eval, n_1);
            assert_eq!(denominator_0_eval, d_0);
            assert_eq!(denominator_1_eval, d_1);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_logup_round_sumcheck_polynomial() {
        let mut rng = thread_rng();

        type Config = Poseidon2BabyBear16BasefoldConfig;
        let verifier = BasefoldVerifier::<Config>::new(1);
        let get_challenger = move || verifier.clone().challenger();

        let interaction_row_counts: Vec<u32> = vec![
            1 << 10,
            (1 << 8) + 2,
            (1 << 10) + 2,
            1 << 8,
            1 << 6,
            1 << 10,
            1 << 8,
            (1 << 6) + 2,
        ];
        let (layer, test_data) =
            generate_test_data(&mut rng, interaction_row_counts, Some(15)).await;
        let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

        let GkrLayer {
            layer: layer_data,
            interaction_data,
            interaction_start_indices,
            interaction_row_counts,
            num_interaction_variables,
            num_row_variables,
        } = layer;

        println!("num_row_variables: {num_row_variables}");
        println!("num_interaction_variables: {num_interaction_variables}");
        let poly_point = Point::<EF>::rand(&mut rng, num_row_variables + num_interaction_variables);
        let (interaction_point, row_point) =
            poly_point.split_at(num_interaction_variables as usize);

        let lambda = rng.gen::<EF>();

        csl_cuda::spawn(move |t| async move {
            let layer_data = layer_data.into_device_in(&t).await.unwrap();
            let interaction_data = interaction_data.into_device_in(&t).await.unwrap();
            let interaction_start_indices =
                interaction_start_indices.into_device_in(&t).await.unwrap();

            let row_point = row_point.to_device_in(&t).await.unwrap();
            let interaction_point = interaction_point.to_device_in(&t).await.unwrap();

            let eq_row = Mle::partial_lagrange(&row_point).await;
            let eq_interaction = Mle::partial_lagrange(&interaction_point).await;

            let layer = GkrLayer {
                layer: layer_data,
                interaction_data,
                interaction_start_indices,
                interaction_row_counts,
                num_interaction_variables,
                num_row_variables,
            };

            let polynomial = LogupRoundPolynomial {
                layer: PolynomialLayer::CircuitLayer(layer),
                eq_row,
                eq_interaction,
                lambda,
                eq_adjustment: EF::one(),
                padding_adjustment: EF::one(),
                point: poly_point.clone(),
            };

            let host_eq = Mle::partial_lagrange(&poly_point).await;
            let claim = slop_futures::rayon::spawn(move || {
                numerator_0
                    .guts()
                    .as_slice()
                    .par_iter()
                    .zip_eq(numerator_1.guts().as_slice().par_iter())
                    .zip_eq(denominator_0.guts().as_slice().par_iter())
                    .zip_eq(denominator_1.guts().as_slice().par_iter())
                    .zip_eq(host_eq.guts().as_slice().par_iter())
                    .map(|((((n_0, n_1), d_0), d_1), eq)| {
                        let numerator_eval = *n_0 * *d_1 + *n_1 * *d_0;
                        let denominator_eval = *d_0 * *d_1;
                        *eq * (numerator_eval * lambda + denominator_eval)
                    })
                    .sum::<EF>()
            })
            .await
            .unwrap();

            let mut challenger = get_challenger();
            let (proof, evals) = reduce_sumcheck_to_evaluation(
                vec![polynomial],
                &mut challenger,
                vec![claim],
                1,
                EF::one(),
            )
            .await;

            let mut challenger = get_challenger();
            partially_verify_sumcheck_proof(
                &proof,
                &mut challenger,
                (num_row_variables + num_interaction_variables) as usize,
                3,
            )
            .unwrap();

            let (point, expected_final_eval) = proof.point_and_eval;

            // Assert that the point has the expected dimension.
            assert_eq!(point.dimension() as u32, num_row_variables + num_interaction_variables);

            // Calculate the expected evaluations at the point.
            let [evals] = evals.try_into().unwrap();
            assert_eq!(evals.len(), 4);
            let [n_0, n_1, d_0, d_1] = evals.try_into().unwrap();

            let eq_eval = Mle::full_lagrange_eval(&poly_point, &point);

            let expected_numerator_eval = n_0 * d_1 + n_1 * d_0;
            let expected_denominator_eval = d_0 * d_1;
            let eval = expected_numerator_eval * lambda + expected_denominator_eval;
            let final_eval = eq_eval * eval;

            // Assert that the final eval is correct.
            assert_eq!(final_eval, expected_final_eval);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_logup_gkr_circuit_transition() {
        let mut rng = thread_rng();

        type TraceGenerator = LogUpGkrCudaTraceGenerator<BabyBear, EF, ()>;
        let trace_generator = TraceGenerator::default();

        let interaction_row_counts: Vec<u32> =
            vec![(1 << 10) + 32, (1 << 10) - 2, 1 << 6, 1 << 8, (1 << 10) + 2];
        let (layer, test_data) = generate_test_data(&mut rng, interaction_row_counts, None).await;
        let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

        let GkrLayer {
            layer: layer_data,
            interaction_data,
            interaction_start_indices,
            interaction_row_counts,
            num_interaction_variables,
            num_row_variables,
        } = layer;

        csl_cuda::spawn(move |t| async move {
            let layer_data = layer_data.into_device_in(&t).await.unwrap();
            let interaction_data = interaction_data.into_device_in(&t).await.unwrap();
            let interaction_start_indices =
                interaction_start_indices.into_device_in(&t).await.unwrap();

            let layer = GkrLayer {
                layer: layer_data,
                interaction_data,
                interaction_start_indices,
                interaction_row_counts,
                num_interaction_variables,
                num_row_variables,
            };

            // Test a single transition.
            let next_layer = trace_generator.layer_transition(&layer).await;

            let GkrLayer {
                layer: next_layer_data,
                interaction_data,
                interaction_start_indices,
                interaction_row_counts,
                num_interaction_variables,
                num_row_variables,
            } = next_layer;

            let next_layer_data = next_layer_data.into_host().await.unwrap();
            let interaction_data = interaction_data.into_host().await.unwrap();
            let interaction_start_indices = interaction_start_indices.into_host().await.unwrap();

            let next_layer_host = GkrLayer {
                layer: next_layer_data,
                interaction_data,
                interaction_start_indices,
                interaction_row_counts,
                num_interaction_variables,
                num_row_variables,
            };

            let next_layer_data = get_polys_from_layer(&next_layer_host);

            let next_numerator_0 = next_layer_data.numerator_0;
            let next_numerator_1 = next_layer_data.numerator_1;
            let next_denominator_0 = next_layer_data.denominator_0;
            let next_denominator_1 = next_layer_data.denominator_1;

            let next_n_values = next_numerator_0
                .guts()
                .as_slice()
                .iter()
                .interleave(next_numerator_1.guts().as_slice())
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(next_n_values.len(), numerator_0.guts().as_slice().len());
            let next_d_values = next_denominator_0
                .guts()
                .as_slice()
                .iter()
                .interleave(next_denominator_1.guts().as_slice())
                .copied()
                .collect::<Vec<_>>();

            for (i, (((((next_n, next_d), n_0), n_1), d_0), d_1)) in next_n_values
                .iter()
                .zip_eq(next_d_values)
                .zip_eq(numerator_0.guts().as_slice())
                .zip_eq(numerator_1.guts().as_slice())
                .zip_eq(denominator_0.guts().as_slice())
                .zip_eq(denominator_1.guts().as_slice())
                .enumerate()
            {
                assert_eq!(next_d, *d_0 * *d_1, "failed at index {i}");
                assert_eq!(*next_n, *n_0 * *d_1 + *n_1 * *d_0, "failed at index {i}");
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_logup_gkr_round_prover() {
        let mut rng = thread_rng();

        type Config = Poseidon2BabyBear16BasefoldConfig;
        type Challenger = <Config as BasefoldConfig>::Challenger;
        let verifier = BasefoldVerifier::<Config>::new(1);
        let get_challenger = move || verifier.clone().challenger();
        type TraceGenerator = LogUpGkrCudaTraceGenerator<BabyBear, EF, ()>;
        let trace_generator = TraceGenerator::default();

        let prover = LogupGkrCudaRoundProver::<BabyBear, EF, Challenger>::default();

        let interaction_row_counts: Vec<u32> =
            [vec![(1 << 0) + 14; 50], vec![(1 << 16) - 12; 2], vec![(1 << 10) + 2; 10]].concat();
        let layer = random_first_layer(&mut rng, interaction_row_counts, Some(21)).await;
        println!("generated test data");

        let FirstGkrLayer {
            layer: layer_data,
            interaction_data,
            interaction_start_indices,
            interaction_row_counts,
            num_interaction_variables,
            num_row_variables,
        } = layer;

        let first_eval_point = Point::<EF>::rand(&mut rng, num_interaction_variables + 1);

        csl_cuda::spawn(move |t| async move {
            let numerator = layer_data.numerator.into_device_in(&t).await.unwrap();
            let denominator = layer_data.denominator.into_device_in(&t).await.unwrap();
            let layer_data = FirstLayerData { numerator, denominator };
            let interaction_data = interaction_data.into_device_in(&t).await.unwrap();
            let interaction_start_indices =
                interaction_start_indices.into_device_in(&t).await.unwrap();

            let layer = FirstGkrLayer {
                layer: layer_data,
                interaction_data,
                interaction_start_indices,
                interaction_row_counts,
                num_interaction_variables,
                num_row_variables,
            };
            let layer = GkrCircuitLayer::FirstLayer(layer);

            t.synchronize().await.unwrap();
            let time = tokio::time::Instant::now();
            let mut layers = vec![layer];
            for _ in 0..num_row_variables - 1 {
                let layer = trace_generator.gkr_transition(layers.last().unwrap()).await;
                layers.push(layer);
            }
            t.synchronize().await.unwrap();
            println!("trace generation time: {:?}", time.elapsed());

            let time = tokio::time::Instant::now();
            layers.reverse();
            let first_layer =
                if let GkrCircuitLayer::Materialized(first_layer) = layers.first().unwrap() {
                    first_layer
                } else {
                    panic!("first layer not correct");
                };
            assert_eq!(first_layer.num_row_variables, 1);

            let output = trace_generator.extract_outputs(first_layer, num_interaction_variables);
            println!("time to extract values: {:?}", time.elapsed());

            // assert_eq!(first_eval_point.dimension(), numerator.num_variables() as usize);
            let first_point_device = first_eval_point.to_device_in(&t).await.unwrap();
            let first_numerator_eval =
                output.numerator.eval_at(&first_point_device).await.to_host().await.unwrap()[0];
            let first_denominator_eval =
                output.denominator.eval_at(&first_point_device).await.to_host().await.unwrap()[0];

            let mut challenger = get_challenger();
            t.synchronize().await.unwrap();
            let time = tokio::time::Instant::now();
            let mut round_proofs = Vec::new();
            // Follow the GKR protocol layer by layer.
            let mut numerator_eval = first_numerator_eval;
            let mut denominator_eval = first_denominator_eval;
            let mut eval_point = first_eval_point.clone();
            for layer in layers {
                let round_proof = prover
                    .prove_round(
                        layer,
                        &eval_point,
                        numerator_eval,
                        denominator_eval,
                        &mut challenger,
                    )
                    .await;
                // Observe the prover message.
                challenger.observe_ext_element(round_proof.numerator_0);
                challenger.observe_ext_element(round_proof.numerator_1);
                challenger.observe_ext_element(round_proof.denominator_0);
                challenger.observe_ext_element(round_proof.denominator_1);
                // Get the evaluation point for the claims.
                eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();
                // Sample the last coordinate.
                let last_coordinate = challenger.sample_ext_element::<EF>();
                // Compute the evaluation of the numerator and denominator at the last coordinate.
                numerator_eval = round_proof.numerator_0
                    + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
                denominator_eval = round_proof.denominator_0
                    + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
                eval_point.add_dimension_back(last_coordinate);
                // Add the round proof to the total
                round_proofs.push(round_proof);
            }
            t.synchronize().await.unwrap();
            println!("proof generation time: {:?}", time.elapsed());

            // Follow the GKR protocol layer by layer.
            let mut challenger = get_challenger();
            let mut numerator_eval = first_numerator_eval;
            let mut denominator_eval = first_denominator_eval;
            let mut eval_point = first_eval_point;
            let num_proofs = round_proofs.len();
            println!("Num rounds: {num_proofs}");
            for (i, round_proof) in round_proofs.iter().enumerate() {
                // Get the batching challenge for combining the claims.
                let lambda = challenger.sample_ext_element::<EF>();
                // Check that the claimed sum is consitent with the previous round values.
                let expected_claim = numerator_eval * lambda + denominator_eval;
                assert_eq!(round_proof.sumcheck_proof.claimed_sum, expected_claim);
                // Verify the sumcheck proof.
                partially_verify_sumcheck_proof(
                    &round_proof.sumcheck_proof,
                    &mut challenger,
                    i + num_interaction_variables as usize + 1,
                    3,
                )
                .unwrap();
                // Verify that the evaluation claim is consistent with the prover messages.
                let (point, final_eval) = round_proof.sumcheck_proof.point_and_eval.clone();
                let eq_eval = Mle::full_lagrange_eval(&point, &eval_point);
                let numerator_sumcheck_eval = round_proof.numerator_0 * round_proof.denominator_1
                    + round_proof.numerator_1 * round_proof.denominator_0;
                let denominator_sumcheck_eval =
                    round_proof.denominator_0 * round_proof.denominator_1;
                let expected_final_eval =
                    eq_eval * (numerator_sumcheck_eval * lambda + denominator_sumcheck_eval);

                assert_eq!(final_eval, expected_final_eval, "Failure in round {i}");

                // Observe the prover message.
                challenger.observe_ext_element(round_proof.numerator_0);
                challenger.observe_ext_element(round_proof.numerator_1);
                challenger.observe_ext_element(round_proof.denominator_0);
                challenger.observe_ext_element(round_proof.denominator_1);

                // Get the evaluation point for the claims.
                eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();

                // Sample the last coordinate and add to the point.
                let last_coordinate = challenger.sample_ext_element::<EF>();
                eval_point.add_dimension_back(last_coordinate);
                // Update the evaluation of the numerator and denominator at the last coordinate.
                numerator_eval = round_proof.numerator_0
                    + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
                denominator_eval = round_proof.denominator_0
                    + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
            }
        })
        .await
        .unwrap();
    }
}

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Arc,
};

use slop_algebra::{ExtensionField, Field};
use slop_alloc::CpuBackend;
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, PaddedMle};
use slop_sumcheck::reduce_sumcheck_to_evaluation;

use crate::{
    air::{InteractionScope, MachineAir},
    debug_interactions_with_all_chips,
    prover::Traces,
    Chip, InteractionKind, LogupRoundPolynomial, PolynomialLayer,
};

use super::{
    LogUpGkrCircuit, LogUpGkrOutput, LogUpGkrProverComponents, LogUpGkrRoundProver,
    LogUpGkrTraceGenerator,
};

/// A trace generator for the GKR circuit.
pub struct LogupGkrCpuTraceGenerator<F, EF, A>(PhantomData<(F, EF, A)>);

impl<F, EF, A> Default for LogupGkrCpuTraceGenerator<F, EF, A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// A trace generator for the GKR circuit.
pub struct LogupGkrCpuCircuit<F: Field, EF> {
    layers: Vec<GkrCircuitLayer<F, EF>>,
}

/// A layer of the GKR circuit.
pub enum GkrCircuitLayer<F: Field, EF> {
    /// An intermediate layer of the GKR circuit.
    Layer(LogUpGkrCpuLayer<EF, EF>),
    /// The first layer of the GKR circuit.
    FirstLayer(LogUpGkrCpuLayer<F, EF>),
}

/// A layer of the GKR circuit.
pub struct LogUpGkrCpuLayer<F, EF> {
    /// The numerators of the layer (`PaddedMle<F>` per table with dimensions `num_row_variables` x
    /// `num_interaction_variables`)
    pub numerator_0: Vec<PaddedMle<F>>,
    /// The denominators of the layer (`PaddedMle<EF>` per table with dimensions
    /// `num_row_variables` x `num_interaction_variables`)
    pub denominator_0: Vec<PaddedMle<EF>>,
    /// The numerators of the layer (`PaddedMle<F>` per table with dimensions `num_row_variables` x
    /// `num_interaction_variables`)
    pub numerator_1: Vec<PaddedMle<F>>,
    /// The denominators of the layer (`PaddedMle<EF>` per table with dimensions
    /// `num_row_variables` x `num_interaction_variables`)
    pub denominator_1: Vec<PaddedMle<EF>>,
    /// The number of row variables (log height of each mle)
    pub num_row_variables: usize,
    /// The number of interaction variables (log width of each mle)
    pub num_interaction_variables: usize,
}

/// An interaction layer of the GKR circuit (`num_row_variables` == 1).
pub struct InteractionLayer<F, EF> {
    /// The numerators of the layer (`PaddedMle<F>` per table with dimensions
    /// `num_interaction_variables` x 1)
    pub numerator_0: Arc<Mle<F>>,
    /// The denominators of the layer (`PaddedMle<EF>` per table with dimensions
    /// `num_interaction_variables` x 1)
    pub denominator_0: Arc<Mle<EF>>,
    /// The numerators of the layer (`PaddedMle<F>` per table with dimensions
    /// `num_interaction_variables` x 1)
    pub numerator_1: Arc<Mle<F>>,
    /// The denominators of the layer (`PaddedMle<EF>` per table with dimensions
    /// `num_interaction_variables` x 1)
    pub denominator_1: Arc<Mle<EF>>,
}

impl<F: Field, EF: ExtensionField<F>, A: MachineAir<F>> LogUpGkrTraceGenerator<F, EF, A, CpuBackend>
    for LogupGkrCpuTraceGenerator<F, EF, A>
{
    type Circuit = LogupGkrCpuCircuit<F, EF>;

    #[allow(unused_variables)]
    async fn generate_gkr_circuit(
        &self,
        chips: &BTreeSet<Chip<F, A>>,
        preprocessed_traces: Traces<F, CpuBackend>,
        traces: Traces<F, CpuBackend>,
        alpha: EF,
        beta: EF,
    ) -> (LogUpGkrOutput<EF>, Self::Circuit) {
        let interactions = chips
            .iter()
            .map(|chip| {
                let interactions = chip
                    .sends()
                    .iter()
                    .map(|int| (int, true))
                    .chain(chip.receives().iter().map(|int| (int, false)))
                    .collect::<Vec<_>>();
                (chip.name(), interactions)
            })
            .collect::<BTreeMap<_, _>>();

        tracing::debug_span!("debug local interactions").in_scope(|| {
            debug_interactions_with_all_chips::<F, A>(
                &chips.iter().cloned().collect::<Vec<_>>(),
                &preprocessed_traces,
                &traces,
                InteractionKind::all_kinds(),
                InteractionScope::Local,
            )
        });

        let first_layer = self
            .generate_first_layer(&interactions, &traces, &preprocessed_traces, alpha, beta)
            .await;
        let num_row_variables = first_layer.num_row_variables;
        // println!("num_row_variables: {:?}", num_row_variables);
        let num_interaction_variables = first_layer.num_interaction_variables;
        let mut layers = Vec::new();
        layers.push(GkrCircuitLayer::FirstLayer(first_layer));

        for _ in 0..num_row_variables - 1 {
            let next_layer = match layers.last().unwrap() {
                GkrCircuitLayer::Layer(layer) => self.layer_transition(layer),
                GkrCircuitLayer::FirstLayer(layer) => self.layer_transition(layer),
            };
            layers.push(GkrCircuitLayer::Layer(next_layer));
        }

        let last_layer = layers.last().unwrap();
        let last_layer = match last_layer {
            GkrCircuitLayer::Layer(layer) => layer,
            GkrCircuitLayer::FirstLayer(layer) => unreachable!(),
        };
        assert_eq!(last_layer.num_row_variables, 1);
        let output = self.extract_outputs(last_layer).await;

        let circuit_generator = Some(Self::default());
        let circuit = LogupGkrCpuCircuit { layers };

        (output, circuit)
    }
}

impl<F: Field, EF: ExtensionField<F>> Iterator for LogupGkrCpuCircuit<F, EF> {
    type Item = GkrCircuitLayer<F, EF>;

    fn next(&mut self) -> Option<Self::Item> {
        self.layers.pop()
    }
}

/// Basic information about the GKR circuit.
impl<F: Field, EF: ExtensionField<F>> LogUpGkrCircuit for LogupGkrCpuCircuit<F, EF> {
    type CircuitLayer = GkrCircuitLayer<F, EF>;

    async fn next(&mut self) -> Option<Self::CircuitLayer> {
        self.layers.pop()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// A prover for the GKR circuit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogupGkrCpuRoundProver;

impl<F: Field, EF: ExtensionField<F>, Challenger> LogUpGkrRoundProver<F, EF, Challenger, CpuBackend>
    for LogupGkrCpuRoundProver
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + 'static + Send + Sync,
{
    type CircuitLayer = GkrCircuitLayer<F, EF>;

    #[allow(unused_variables)]
    async fn prove_round(
        &self,
        circuit: Self::CircuitLayer,
        eval_point: &slop_multilinear::Point<EF>,
        numerator_eval: EF,
        denominator_eval: EF,
        challenger: &mut Challenger,
    ) -> super::LogupGkrRoundProof<EF> {
        let lambda = challenger.sample_ext_element::<EF>();

        let (numerator_0, denominator_0, numerator_1, denominator_1, sumcheck_proof) = match circuit
        {
            GkrCircuitLayer::Layer(layer) => {
                let (interaction_point, row_point) =
                    eval_point.split_at(layer.num_interaction_variables);
                let eq_interaction = Mle::partial_lagrange(&interaction_point).await;
                let eq_row = Mle::partial_lagrange(&row_point).await;
                let sumcheck_poly = LogupRoundPolynomial {
                    layer: PolynomialLayer::CircuitLayer(layer),
                    eq_row: Arc::new(eq_row),
                    eq_interaction: Arc::new(eq_interaction),
                    lambda,
                    eq_adjustment: EF::one(),
                    padding_adjustment: EF::one(),
                    point: eval_point.clone(),
                };
                let claim = numerator_eval * lambda + denominator_eval;

                let (sumcheck_proof, mut openings) = reduce_sumcheck_to_evaluation(
                    vec![sumcheck_poly],
                    challenger,
                    vec![claim],
                    1,
                    lambda,
                )
                .await;

                let openings = openings.pop().unwrap();
                let [numerator_0, denominator_0, numerator_1, denominator_1] =
                    openings.try_into().unwrap();
                (numerator_0, denominator_0, numerator_1, denominator_1, sumcheck_proof)
            }
            GkrCircuitLayer::FirstLayer(layer) => {
                let (interaction_point, row_point) =
                    eval_point.split_at(layer.num_interaction_variables);
                let eq_interaction = Mle::partial_lagrange(&interaction_point).await;
                let eq_row = Mle::partial_lagrange(&row_point).await;
                let sumcheck_poly = LogupRoundPolynomial {
                    layer: PolynomialLayer::CircuitLayer(layer),
                    eq_row: Arc::new(eq_row),
                    eq_interaction: Arc::new(eq_interaction),
                    lambda,
                    eq_adjustment: EF::one(),
                    padding_adjustment: EF::one(),
                    point: eval_point.clone(),
                };
                let claim = numerator_eval * lambda + denominator_eval;
                let (sumcheck_proof, mut openings) = reduce_sumcheck_to_evaluation(
                    vec![sumcheck_poly],
                    challenger,
                    vec![claim],
                    1,
                    lambda,
                )
                .await;
                let openings = openings.pop().unwrap();
                let [numerator_0, denominator_0, numerator_1, denominator_1] =
                    openings.try_into().unwrap();
                (numerator_0, denominator_0, numerator_1, denominator_1, sumcheck_proof)
            }
        };

        super::LogupGkrRoundProof {
            numerator_0,
            numerator_1,
            denominator_0,
            denominator_1,
            sumcheck_proof,
        }
    }
}

/// The components of the GKR prover for the CPU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogupGkrCpuProverComponents<F, EF, A, Challenger>(PhantomData<(F, EF, A, Challenger)>);

impl<F, EF, A, Challenger> LogUpGkrProverComponents
    for LogupGkrCpuProverComponents<F, EF, A, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    A: MachineAir<F>,
    Challenger: FieldChallenger<F> + 'static + Send + Sync,
{
    type F = F;
    type EF = EF;
    type A = A;
    type B = CpuBackend;
    type Challenger = Challenger;
    type CircuitLayer = GkrCircuitLayer<F, EF>;
    type Circuit = LogupGkrCpuCircuit<F, EF>;
    type TraceGenerator = LogupGkrCpuTraceGenerator<F, EF, A>;
    type RoundProver = LogupGkrCpuRoundProver;
}

use std::marker::PhantomData;

use csl_cuda::{PartialLagrangeKernel, TaskScope};
use slop_algebra::{ExtensionField, Field};
use slop_challenger::FieldChallenger;
use slop_multilinear::MleEvaluationBackend;
use slop_tensor::{AddAssignBackend, TransposeBackend};
use sp1_stark::{air::MachineAir, GkrProverImpl, LogUpGkrProverComponents};

use crate::{
    CircuitTransitionKernel, ExtractOutputKernel, FirstLayerKernels, FirstLayerTransitionKernel,
    GkrCircuitLayer, LogUpCudaCircuit, LogUpGkrCudaTraceGenerator, LogupGkrCudaRoundProver,
    MaterializedLayerKernels, PopulateLastCircuitLayerKernel,
};

pub trait GkrCudaBackend<F: Field, EF: ExtensionField<F>>:
    MleEvaluationBackend<EF, EF>
    + PartialLagrangeKernel<EF>
    + MaterializedLayerKernels<EF>
    + FirstLayerKernels<F, EF>
    + CircuitTransitionKernel<EF>
    + FirstLayerTransitionKernel<F, EF>
    + AddAssignBackend<EF>
    + MleEvaluationBackend<F, F>
    + MleEvaluationBackend<F, EF>
    + PopulateLastCircuitLayerKernel<F, EF>
    + TransposeBackend<EF>
    + ExtractOutputKernel<EF>
{
}

impl<F: Field, EF: ExtensionField<F>> GkrCudaBackend<F, EF> for TaskScope where
    TaskScope: MleEvaluationBackend<EF, EF>
        + PartialLagrangeKernel<EF>
        + MaterializedLayerKernels<EF>
        + AddAssignBackend<EF>
        + MleEvaluationBackend<F, F>
        + FirstLayerKernels<F, EF>
        + CircuitTransitionKernel<EF>
        + FirstLayerTransitionKernel<F, EF>
        + MleEvaluationBackend<F, EF>
        + PopulateLastCircuitLayerKernel<F, EF>
        + TransposeBackend<EF>
        + ExtractOutputKernel<EF>
{
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogupGkrCudaProverComponents<F, EF, A, Challenger>(PhantomData<(F, EF, A, Challenger)>);

impl<F, EF, A, Challenger> LogupGkrCudaProverComponents<F, EF, A, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    A: MachineAir<F>,
    Challenger: FieldChallenger<F> + 'static + Send + Sync,
    TaskScope: GkrCudaBackend<F, EF>,
{
    pub fn default_prover() -> GkrProverImpl<Self> {
        let trace_generator = LogUpGkrCudaTraceGenerator::default();
        let round_prover = LogupGkrCudaRoundProver::default();
        GkrProverImpl::new(trace_generator, round_prover)
    }
}

impl<F, EF, A, Challenger> LogUpGkrProverComponents
    for LogupGkrCudaProverComponents<F, EF, A, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    A: MachineAir<F>,
    Challenger: FieldChallenger<F> + 'static + Send + Sync,
    TaskScope: GkrCudaBackend<F, EF>,
{
    type F = F;
    type EF = EF;
    type A = A;
    type B = TaskScope;
    type Challenger = Challenger;
    type CircuitLayer = GkrCircuitLayer<F, EF>;
    type Circuit = LogUpCudaCircuit<F, EF, A>;
    type TraceGenerator = LogUpGkrCudaTraceGenerator<F, EF, A>;
    type RoundProver = LogupGkrCudaRoundProver<F, EF, Challenger>;
}

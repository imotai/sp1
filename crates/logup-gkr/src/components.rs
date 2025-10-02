use std::marker::PhantomData;

use csl_cuda::{transpose::DeviceTransposeKernel, PartialLagrangeKernel, TaskScope};
use slop_algebra::{ExtensionField, Field};
use slop_challenger::IopCtx;
use slop_multilinear::MleEvaluationBackend;
use slop_tensor::{AddAssignBackend, TransposeBackend};
use sp1_hypercube::{air::MachineAir, GkrProverImpl, LogUpGkrProverComponents};

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
pub struct LogupGkrCudaProverComponents<GC, A>(PhantomData<(GC, A)>);

impl<GC, A> LogupGkrCudaProverComponents<GC, A>
where
    GC: IopCtx,
    TaskScope: GkrCudaBackend<GC::F, GC::EF> + DeviceTransposeKernel<GC::F>,
    A: MachineAir<GC::F>,
{
    pub fn default_prover() -> GkrProverImpl<GC, Self> {
        let trace_generator = LogUpGkrCudaTraceGenerator::default();
        let round_prover = LogupGkrCudaRoundProver::default();
        GkrProverImpl::new(trace_generator, round_prover)
    }
}

impl<GC, A> LogUpGkrProverComponents<GC> for LogupGkrCudaProverComponents<GC, A>
where
    GC: IopCtx,
    A: MachineAir<GC::F>,
    TaskScope: GkrCudaBackend<GC::F, GC::EF> + DeviceTransposeKernel<GC::F>,
{
    type A = A;
    type B = TaskScope;
    type CircuitLayer = GkrCircuitLayer<GC::F, GC::EF>;
    type Circuit = LogUpCudaCircuit<GC::F, GC::EF, A>;
    type TraceGenerator = LogUpGkrCudaTraceGenerator<GC::F, GC::EF, A>;
    type RoundProver = LogupGkrCudaRoundProver<GC::F, GC::EF, GC::Challenger>;
}

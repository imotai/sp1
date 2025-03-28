use std::{collections::BTreeSet, marker::PhantomData};

use slop_algebra::{ExtensionField, Field};
use slop_alloc::CpuBackend;
use slop_challenger::FieldChallenger;

use crate::{air::MachineAir, prover::Traces, Chip};

use super::{
    LogUpGkrCircuit, LogUpGkrOutput, LogUpGkrProverComponents, LogUpGkrRoundProver,
    LogUpGkrTraceGenerator,
};

/// A trace generator for the GKR circuit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogupGkrCpuTraceGenerator;

/// A trace generator for the GKR circuit.
pub struct LogupGkrCpuCircuit(usize);

/// A layer of the GKR circuit.
pub struct LogUpGkrCpuLayer;

impl<F: Field, EF: ExtensionField<F>, A: MachineAir<F>> LogUpGkrTraceGenerator<F, EF, A, CpuBackend>
    for LogupGkrCpuTraceGenerator
{
    type Circuit = LogupGkrCpuCircuit;

    #[allow(unused_variables)]
    async fn generate_gkr_circuit(
        &self,
        chips: &BTreeSet<Chip<F, A>>,
        preprocessed_traces: Traces<F, CpuBackend>,
        traces: Traces<F, CpuBackend>,
        alpha: EF,
        beta: EF,
    ) -> (LogUpGkrOutput<EF>, Self::Circuit) {
        todo!()
    }
}

impl Iterator for LogupGkrCpuCircuit {
    type Item = LogUpGkrCpuLayer;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/// Basic information about the GKR circuit.
impl LogUpGkrCircuit for LogupGkrCpuCircuit {
    type CircuitLayer = LogUpGkrCpuLayer;

    async fn next(&mut self) -> Option<Self::CircuitLayer> {
        todo!()
    }

    fn num_layers(&self) -> usize {
        self.0
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
    type CircuitLayer = LogUpGkrCpuLayer;

    #[allow(unused_variables)]
    async fn prove_round(
        &self,
        circuit: Self::CircuitLayer,
        eval_point: &slop_multilinear::Point<EF>,
        numerator_eval: EF,
        denominator_eval: EF,
        challenger: &mut Challenger,
    ) -> super::LogupGkrRoundProof<EF> {
        todo!()
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
    type CircuitLayer = LogUpGkrCpuLayer;
    type Circuit = LogupGkrCpuCircuit;
    type TraceGenerator = LogupGkrCpuTraceGenerator;
    type RoundProver = LogupGkrCpuRoundProver;
}

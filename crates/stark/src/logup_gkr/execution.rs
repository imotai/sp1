use std::{collections::BTreeSet, future::Future};

use slop_algebra::{ExtensionField, Field};
use slop_alloc::Backend;

use crate::{air::MachineAir, prover::Traces, Chip};

use super::LogUpGkrOutput;

/// TODO
pub trait LogUpGkrTraceGenerator<F: Field, EF: ExtensionField<F>, A: MachineAir<F>, B: Backend>:
    'static + Send + Sync
{
    /// The Gkr Circuit type.
    ///
    /// The circuit contains all the information required for the prover to generate proofs for each
    /// circuit layer.
    type Circuit: LogUpGkrCircuit;

    /// Generate the GKR circuit for the given chips, preprocessed traces, main traces, and the
    /// permutation challenges `alpha` and `beta`.
    ///
    /// `alpha` is the challenge used for the Reed-Solomon fingerprint of the messages and `beta` is
    /// the challenge point for the log-derivative expression.
    fn generate_gkr_circuit(
        &self,
        chips: &BTreeSet<Chip<F, A>>,
        preprocessed_traces: Traces<F, B>,
        traces: Traces<F, B>,
        alpha: EF,
        beta: EF,
    ) -> impl Future<Output = (LogUpGkrOutput<EF, B>, Self::Circuit)> + Send;
}

/// Basic information about the GKR circuit.
pub trait LogUpGkrCircuit {
    /// The layer type of the GKR circuit.
    type CircuitLayer;

    /// The number of layers in the GKR circuit.
    fn num_layers(&self) -> usize;

    /// Get the next layer of the GKR circuit.
    fn next(&mut self) -> impl Future<Output = Option<Self::CircuitLayer>> + Send;
}

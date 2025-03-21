use crate::{BabyBearFriConfigVariable, CircuitConfig};
use derive_where::derive_where;
use slop_baby_bear::BabyBear;
use slop_sumcheck::PartialSumcheckProof;
use sp1_recursion_compiler::{
    ir::{Config, Felt},
    prelude::Ext,
};
use sp1_stark::{air::MachineAir, Machine, ShardOpenedValues};

/// A verifier for shard proofs.
#[derive_where(Clone)]
pub struct StarkVerifier<C: Config, SC, A: MachineAir<C::F>> {
    /// The machine.
    pub machine: Machine<C::F, A>,
    /// TODO: The jagged pcs verifier.
    // pub pcs_verifier: JaggedPcsVerifier<C>,
    _phantom: std::marker::PhantomData<(C, SC, A)>,
}

#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct ShardProofVariable<C: CircuitConfig<F = BabyBear>, SC: BabyBearFriConfigVariable<C>> {
    /// The commitments to main traces.
    pub main_commitment: SC::DigestVariable,
    /// The values of the traces at the final random point.
    pub opened_values: ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    /// TODO: The evaluation proof.
    // pub evaluation_proof: JaggedPcsProof<SC>,
    /// The zerocheck IOP proof.
    pub zerocheck_proof: PartialSumcheckProof<Ext<C::F, C::EF>>,
    /// The public values
    pub public_values: Vec<Felt<C::F>>,
    // TODO: The `LogUp+GKR` IOP proofs.
    // pub gkr_proofs: Vec<LogupGkrProof<Ext<C::F, C::EF>>>,
}

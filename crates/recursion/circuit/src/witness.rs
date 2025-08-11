use std::collections::BTreeMap;

use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldVerifier},
    hash::FieldHasherVariable,
    jagged::RecursiveJaggedConfig,
    shard::{MachineVerifyingKeyVariable, ShardProofVariable},
    AsRecursive, CircuitConfig, SP1FieldConfigVariable,
};
use slop_algebra::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};
use slop_bn254::Bn254Fr;
use slop_commit::Rounds;
use slop_jagged::{JaggedConfig, JaggedEvalConfig};
use sp1_hypercube::{
    septic_curve::SepticCurve, septic_digest::SepticDigest, septic_extension::SepticExtension,
    AirOpenedValues, ChipDimensions, ChipOpenedValues, MachineConfig, MachineVerifyingKey,
    ShardOpenedValues, ShardProof,
};
use sp1_primitives::SP1Field;
pub use sp1_recursion_compiler::ir::Witness as OuterWitness;
use sp1_recursion_compiler::{
    config::OuterConfig,
    ir::{Builder, Config, Ext, Felt, Var},
};
use sp1_recursion_executor::Block;

pub trait WitnessWriter<C: CircuitConfig>: Sized {
    fn write_bit(&mut self, value: bool);

    fn write_var(&mut self, value: C::N);

    fn write_felt(&mut self, value: C::F);

    fn write_ext(&mut self, value: C::EF);
}

impl WitnessWriter<OuterConfig> for OuterWitness<OuterConfig> {
    fn write_bit(&mut self, value: bool) {
        self.vars.push(Bn254Fr::from_bool(value));
    }

    fn write_var(&mut self, value: Bn254Fr) {
        self.vars.push(value);
    }

    fn write_felt(&mut self, value: SP1Field) {
        self.felts.push(value);
    }

    fn write_ext(&mut self, value: BinomialExtensionField<SP1Field, 4>) {
        self.exts.push(value);
    }
}

pub type WitnessBlock<C> = Block<<C as Config>::F>;

impl<C: CircuitConfig<F = SP1Field, Bit = Felt<SP1Field>>> WitnessWriter<C>
    for Vec<WitnessBlock<C>>
{
    fn write_bit(&mut self, value: bool) {
        self.push(Block::from(C::F::from_bool(value)))
    }

    fn write_var(&mut self, _value: <C>::N) {
        unimplemented!("Cannot write Var<N> in this configuration")
    }

    fn write_felt(&mut self, value: <C>::F) {
        self.push(Block::from(value))
    }

    fn write_ext(&mut self, value: <C>::EF) {
        self.push(Block::from(value.as_base_slice()))
    }
}

pub trait Witnessable<C: CircuitConfig> {
    type WitnessVariable;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable;

    fn write(&self, witness: &mut impl WitnessWriter<C>);
}

impl<C: CircuitConfig> Witnessable<C> for () {
    type WitnessVariable = ();

    fn read(&self, _builder: &mut Builder<C>) -> Self::WitnessVariable {}

    fn write(&self, _witness: &mut impl WitnessWriter<C>) {}
}

impl<C: CircuitConfig> Witnessable<C> for bool {
    type WitnessVariable = C::Bit;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_bit(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_bit(*self);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for &T {
    type WitnessVariable = T::WitnessVariable;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        (*self).read(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        (*self).write(witness)
    }
}

impl<C: CircuitConfig, T: Witnessable<C>, U: Witnessable<C>> Witnessable<C> for (T, U) {
    type WitnessVariable = (T::WitnessVariable, U::WitnessVariable);

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        (self.0.read(builder), self.1.read(builder))
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.0.write(witness);
        self.1.write(witness);
    }
}

impl<C: CircuitConfig<F = SP1Field>> Witnessable<C> for SP1Field {
    type WitnessVariable = Felt<SP1Field>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_felt(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_felt(*self);
    }
}

impl<C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>> Witnessable<C>
    for BinomialExtensionField<SP1Field, 4>
{
    type WitnessVariable = Ext<SP1Field, BinomialExtensionField<SP1Field, 4>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_ext(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_ext(*self);
    }
}

impl<C: CircuitConfig<N = Bn254Fr>> Witnessable<C> for Bn254Fr {
    type WitnessVariable = Var<Bn254Fr>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        builder.witness_var()
    }
    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_var(*self)
    }
}

impl<C: CircuitConfig, T: Witnessable<C>, const N: usize> Witnessable<C> for [T; N] {
    type WitnessVariable = [T::WitnessVariable; N];

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect::<Vec<_>>().try_into().unwrap_or_else(
            |x: Vec<_>| {
                // Cannot just `.unwrap()` without requiring Debug bounds.
                panic!("could not coerce vec of len {} into array of len {N}", x.len())
            },
        )
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Vec<T> {
    type WitnessVariable = Vec<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect()
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig, K: Clone + Ord, V: Witnessable<C>> Witnessable<C> for BTreeMap<K, V> {
    type WitnessVariable = BTreeMap<K, V::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|(k, v)| (k.clone(), v.read(builder))).collect()
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for v in self.values() {
            v.write(witness);
        }
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Rounds<T> {
    type WitnessVariable = Rounds<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect()
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig<F = SP1Field>> Witnessable<C> for SepticDigest<C::F> {
    type WitnessVariable = SepticDigest<Felt<C::F>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let x = self.0.x.0.read(builder);
        let y = self.0.y.0.read(builder);
        SepticDigest(SepticCurve { x: SepticExtension(x), y: SepticExtension(y) })
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.0.x.0.write(witness);
        self.0.y.0.write(witness);
    }
}

impl<C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>> Witnessable<C>
    for ShardOpenedValues<C::F, C::EF>
{
    type WitnessVariable = ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let chips = self.chips.read(builder);
        Self::WitnessVariable { chips }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.chips.write(witness);
    }
}

impl<C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>> Witnessable<C>
    for ChipOpenedValues<C::F, C::EF>
{
    type WitnessVariable = ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let preprocessed = self.preprocessed.read(builder);
        let main = self.main.read(builder);
        let local_cumulative_sum = self.local_cumulative_sum.read(builder);
        let degree = self.degree.read(builder);
        Self::WitnessVariable { preprocessed, main, local_cumulative_sum, degree }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.preprocessed.write(witness);
        self.main.write(witness);
        self.local_cumulative_sum.write(witness);
        self.degree.write(witness);
    }
}

impl<C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>> Witnessable<C>
    for AirOpenedValues<C::EF>
{
    type WitnessVariable = AirOpenedValues<Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let local = self.local.read(builder);
        Self::WitnessVariable { local }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.local.write(witness);
    }
}

impl<C, SC, RecursiveStackedPcsProof, RecursiveJaggedEvalProof> Witnessable<C> for ShardProof<SC>
where
    C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>,
    SC: SP1FieldConfigVariable<C>
        + MachineConfig
        + JaggedConfig<
            F = C::F,
            EF = C::EF,
            BatchPcsProof: Witnessable<C, WitnessVariable = RecursiveStackedPcsProof>,
        > + AsRecursive<C>,
    <<SC as JaggedConfig>::JaggedEvaluator as JaggedEvalConfig<
        C::F,
        C::EF,
        <SC as JaggedConfig>::Challenger,
    >>::JaggedEvalProof: Witnessable<C, WitnessVariable = RecursiveJaggedEvalProof>,
    SC::Recursive: RecursiveJaggedConfig<
        F = C::F,
        EF = C::EF,
        Circuit = C,
        BatchPcsProof = RecursiveStackedPcsProof,
        JaggedEvalProof = RecursiveJaggedEvalProof,
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
    C::EF: Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
    SC::Commitment:
        Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
{
    type WitnessVariable = ShardProofVariable<C, SC, SC::Recursive>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let public_values = self.public_values.read(builder);
        let main_commitment = self.main_commitment.read(builder);
        let logup_gkr_proof = self.logup_gkr_proof.read(builder);
        let zerocheck_proof = self.zerocheck_proof.read(builder);
        let opened_values = self.opened_values.read(builder);
        let evaluation_proof = self.evaluation_proof.read(builder);
        Self::WitnessVariable {
            main_commitment,
            zerocheck_proof,
            opened_values,
            public_values,
            logup_gkr_proof,
            evaluation_proof,
            shard_chips: self.shard_chips.clone(),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.public_values.write(witness);
        self.main_commitment.write(witness);
        self.logup_gkr_proof.write(witness);
        self.zerocheck_proof.write(witness);
        self.opened_values.write(witness);
        self.evaluation_proof.write(witness);
    }
}

impl<C, MC> Witnessable<C> for MachineVerifyingKey<MC>
where
    C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>,
    MC: MachineConfig + SP1FieldConfigVariable<C> + MachineConfig + JaggedConfig,
    MC::Commitment:
        Witnessable<C, WitnessVariable = <MC as FieldHasherVariable<C>>::DigestVariable>,
{
    type WitnessVariable = MachineVerifyingKeyVariable<C, MC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let pc_start = self.pc_start.read(builder);
        let initial_global_cumulative_sum = self.initial_global_cumulative_sum.read(builder);
        let preprocessed_commit = self.preprocessed_commit.read(builder);
        let preprocessed_chip_information = self
            .preprocessed_chip_information
            .iter()
            .map(|(name, dimensions)| {
                (
                    name.clone(),
                    ChipDimensions {
                        height: dimensions.height.read(builder),
                        num_polynomials: dimensions.num_polynomials.read(builder),
                    },
                )
            })
            .collect();
        Self::WitnessVariable {
            pc_start,
            initial_global_cumulative_sum,
            preprocessed_commit,
            preprocessed_chip_information,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.pc_start.write(witness);
        self.initial_global_cumulative_sum.write(witness);
        self.preprocessed_commit.write(witness);
        self.preprocessed_chip_information.values().for_each(|dims| {
            dims.height.write(witness);
            dims.num_polynomials.write(witness)
        });
    }
}

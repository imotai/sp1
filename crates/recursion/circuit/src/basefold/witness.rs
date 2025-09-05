use crate::{
    basefold::tcs::RecursiveTensorCsOpening,
    hash::FieldHasherVariable,
    witness::{WitnessWriter, Witnessable},
    AsRecursive, CircuitConfig,
};
use slop_alloc::Buffer;
use slop_basefold::{BasefoldConfig, BasefoldProof};
use slop_challenger::{GrindingChallenger, IopCtx};
use slop_merkle_tree::{MerkleTreeOpening, MerkleTreeTcsProof};
use slop_multilinear::{Evaluations, Mle, MleEval};
use slop_stacked::StackedPcsProof;
use slop_tensor::Tensor;
use sp1_primitives::{SP1ExtensionField, SP1Field};
use sp1_recursion_compiler::ir::{Builder, Felt};

use super::{stacked::RecursiveStackedPcsProof, RecursiveBasefoldConfig, RecursiveBasefoldProof};

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Tensor<T> {
    type WitnessVariable = Tensor<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        Tensor {
            storage: Buffer::from(
                self.as_slice().iter().map(|x| x.read(builder)).collect::<Vec<_>>(),
            ),
            dimensions: self.dimensions.clone(),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.as_slice() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Mle<T> {
    type WitnessVariable = Mle<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let guts = self.guts().read(builder);
        Mle::new(guts)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.guts().write(witness);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for MleEval<T> {
    type WitnessVariable = MleEval<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let evaluations = self.evaluations().read(builder);
        MleEval::new(evaluations)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.evaluations().write(witness);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Evaluations<T> {
    type WitnessVariable = Evaluations<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let round_evaluations = self.round_evaluations.read(builder);
        Evaluations { round_evaluations }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.round_evaluations.write(witness);
    }
}

impl<GC: IopCtx<F = SP1Field>, C: CircuitConfig> Witnessable<C> for MerkleTreeOpening<GC>
where
    GC::Digest: Witnessable<C>,
{
    type WitnessVariable =
        RecursiveTensorCsOpening<<GC::Digest as Witnessable<C>>::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let values: Tensor<Felt<SP1Field>> = self.values.read(builder);
        let proof = self.proof.paths.read(builder);
        RecursiveTensorCsOpening::<<GC::Digest as Witnessable<C>>::WitnessVariable> {
            values,
            proof,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.values.write(witness);
        self.proof.write(witness);
    }
}

impl<C, T> Witnessable<C> for MerkleTreeTcsProof<T>
where
    C: CircuitConfig,
    T: Witnessable<C>,
{
    type WitnessVariable = MerkleTreeTcsProof<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let paths = self.paths.read(builder);
        MerkleTreeTcsProof { paths }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.paths.write(witness);
    }
}

impl<C, BC, GC> Witnessable<C> for BasefoldProof<GC, BC>
where
    C: CircuitConfig,
    GC: IopCtx<F = SP1Field, EF = SP1ExtensionField>,
    BC: BasefoldConfig<GC> + AsRecursive<C>,
    <GC::Challenger as GrindingChallenger>::Witness:
        Witnessable<C, WitnessVariable = Felt<SP1Field>>,
    BC::Recursive: RecursiveBasefoldConfig<F = SP1Field, EF = SP1ExtensionField, Circuit = C>,
    GC::Digest: Witnessable<
        C,
        WitnessVariable = <<BC::Recursive as RecursiveBasefoldConfig>::M as FieldHasherVariable<
            C,
        >>::DigestVariable,
    >,
    MerkleTreeOpening<GC>: Witnessable<
        C,
        WitnessVariable = RecursiveTensorCsOpening<<GC::Digest as Witnessable<C>>::WitnessVariable>,
    >,
{
    type WitnessVariable = RecursiveBasefoldProof<BC::Recursive>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let univariate_messages = self.univariate_messages.read(builder);
        let fri_commitments = self.fri_commitments.read(builder);
        let component_polynomials_query_openings =
            self.component_polynomials_query_openings.read(builder);
        let query_phase_openings = self.query_phase_openings.read(builder);
        let final_poly = self.final_poly.read(builder);
        let pow_witness = self.pow_witness.read(builder);
        RecursiveBasefoldProof::<BC::Recursive> {
            univariate_messages,
            fri_commitments,
            component_polynomials_query_openings,
            query_phase_openings,
            final_poly,
            pow_witness,
        }
    }
    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.univariate_messages.write(witness);
        self.fri_commitments.write(witness);
        self.component_polynomials_query_openings.write(witness);
        self.query_phase_openings.write(witness);
        self.final_poly.write(witness);
        self.pow_witness.write(witness);
    }
}

impl<C, PcsProof, RecursivePcsProof> Witnessable<C> for StackedPcsProof<PcsProof, SP1ExtensionField>
where
    C: CircuitConfig,
    PcsProof: Witnessable<C, WitnessVariable = RecursivePcsProof>,
{
    type WitnessVariable = RecursiveStackedPcsProof<RecursivePcsProof, SP1Field, SP1ExtensionField>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let batch_evaluations = self.batch_evaluations.read(builder);
        let pcs_proof = self.pcs_proof.read(builder);
        RecursiveStackedPcsProof::<RecursivePcsProof, SP1Field, SP1ExtensionField> {
            pcs_proof,
            batch_evaluations,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.batch_evaluations.write(witness);
        self.pcs_proof.write(witness);
    }
}

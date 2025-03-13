use crate::basefold::tcs::{RecursiveTcs, RecursiveTensorCsOpening};
use crate::{
    witness::{WitnessWriter, Witnessable},
    AsRecursive, CircuitConfig,
};
use slop_alloc::Buffer;
use slop_commit::{TensorCs, TensorCsOpening};
use slop_merkle_tree::MerkleTreeTcsProof;
use slop_tensor::Tensor;
use sp1_recursion_compiler::ir::{Builder, Felt};

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

impl<C, TC> Witnessable<C> for TensorCsOpening<TC>
where
    C: CircuitConfig,
    TC: TensorCs<Data = C::F> + AsRecursive<C>,
    <TC as TensorCs>::Proof: Witnessable<C>,
    TC::Recursive: RecursiveTcs<
        Data = Felt<C::F>,
        Proof = <<TC as TensorCs>::Proof as Witnessable<C>>::WitnessVariable,
    >,
    C::F: Witnessable<C, WitnessVariable = Felt<C::F>>,
{
    type WitnessVariable = RecursiveTensorCsOpening<TC::Recursive>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let values: Vec<Tensor<Felt<C::F>>> = self.values.iter().map(|v| v.read(builder)).collect();
        let proof = self.proof.read(builder);
        RecursiveTensorCsOpening::<TC::Recursive> { values, proof }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for value in &self.values {
            value.write(witness);
        }
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

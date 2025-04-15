use std::marker::PhantomData;

use itertools::Itertools;
use slop_merkle_tree::{MerkleTreeTcs, MerkleTreeTcsProof, Poseidon2BabyBearConfig};
use slop_tensor::Tensor;
use sp1_recursion_compiler::ir::{Builder, Felt};
use sp1_stark::BabyBearPoseidon2;

use crate::{basefold::merkle_tree::verify, hash::FieldHasherVariable, AsRecursive, CircuitConfig};

pub trait RecursiveTcs: Sized {
    type Data;
    type Commitment;
    type Proof;
    type Circuit: CircuitConfig<Bit = Self::Bit>;
    type Bit;

    fn verify_tensor_openings(
        builder: &mut Builder<Self::Circuit>,
        commit: &Self::Commitment,
        indices: &[Vec<Self::Bit>],
        opening: &RecursiveTensorCsOpening<Self>,
    );
}

/// An opening of a tensor commitment scheme.
pub struct RecursiveTensorCsOpening<C: RecursiveTcs> {
    /// The claimed values of the opening.
    pub values: Tensor<C::Data>,
    /// The proof of the opening.
    pub proof: <C as RecursiveTcs>::Proof,
}

#[derive(Debug, Copy, PartialEq, Eq)]
pub struct RecursiveMerkleTreeTcs<C, M>(pub PhantomData<(C, M)>);

impl<C, M> Clone for RecursiveMerkleTreeTcs<C, M> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<C: CircuitConfig> AsRecursive<C> for MerkleTreeTcs<Poseidon2BabyBearConfig> {
    type Recursive = RecursiveMerkleTreeTcs<C, BabyBearPoseidon2>;
}

impl<C, M> RecursiveTcs for RecursiveMerkleTreeTcs<C, M>
where
    C: CircuitConfig,
    M: FieldHasherVariable<C>,
{
    type Data = Felt<C::F>;
    type Commitment = M::DigestVariable;
    type Proof = MerkleTreeTcsProof<M::DigestVariable>;
    type Circuit = C;
    type Bit = C::Bit;

    fn verify_tensor_openings(
        builder: &mut Builder<Self::Circuit>,
        commit: &Self::Commitment,
        indices: &[Vec<Self::Bit>],
        opening: &RecursiveTensorCsOpening<Self>,
    ) {
        for (i, (index, path)) in indices.iter().zip_eq(opening.proof.paths.split()).enumerate() {
            // Collect the lead slices of the claimed values.
            let claimed_values_slices = opening.values.get(i).unwrap().as_slice().to_vec();

            let path = path.as_slice().to_vec();
            let digest = M::hash(builder, &claimed_values_slices);

            verify::<C, M>(builder, path, index.to_vec(), digest, *commit);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use slop_commit::Message;
    use sp1_recursion_compiler::circuit::AsmConfig;
    use std::sync::Arc;

    use slop_algebra::extension::BinomialExtensionField;
    use slop_baby_bear::{BabyBear, DiffusionMatrixBabyBear};

    use crate::witness::Witnessable;

    use super::*;
    use itertools::Itertools;
    use slop_commit::{ComputeTcsOpenings, TensorCsOpening, TensorCsProver};
    use slop_merkle_tree::{my_bb_16_perm, Poseidon2BabyBear16Prover};
    use slop_tensor::Tensor;
    use sp1_recursion_compiler::circuit::{AsmBuilder, AsmCompiler};
    use sp1_recursion_executor::Runtime;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[tokio::test]
    async fn test_merkle_proof() {
        let mut rng = thread_rng();

        let height = rng.gen_range(500..2000);
        let width = rng.gen_range(15..30);
        let num_tensors = rng.gen_range(5..15);

        let num_indices = rng.gen_range(2..10);

        let tensors = (0..num_tensors)
            .map(|_| Tensor::<BabyBear>::rand(&mut rng, [height, width]))
            .collect::<Message<_>>();

        let prover = Poseidon2BabyBear16Prover::default();
        let (root, data) = prover.commit_tensors(tensors.clone()).await.unwrap();

        let indices = (0..num_indices).map(|_| rng.gen_range(0..height)).collect_vec();
        let proof = prover.prove_openings_at_indices(data, &indices).await.unwrap();
        let openings = prover.compute_openings_at_indices(tensors, &indices).await;
        let opening: TensorCsOpening<MerkleTreeTcs<Poseidon2BabyBearConfig>> =
            TensorCsOpening { values: openings, proof };

        let bit_len = height.next_power_of_two().ilog2();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut witness_stream = Vec::new();

        let mut index_bits = Vec::new();
        for index in indices {
            let bits = (0..bit_len).map(|i| ((index >> i) & 1 == 1)).collect_vec();
            Witnessable::<AsmConfig<F, EF>>::write(&bits, &mut witness_stream);
            let bits = bits.read(&mut builder);
            index_bits.push(bits);
        }

        Witnessable::<AsmConfig<F, EF>>::write(&root, &mut witness_stream);
        let root = root.read(&mut builder);
        Witnessable::<AsmConfig<F, EF>>::write(&opening, &mut witness_stream);
        let opening = opening.read(&mut builder);

        RecursiveMerkleTreeTcs::<AsmConfig<F, EF>, BabyBearPoseidon2>::verify_tensor_openings(
            &mut builder,
            &root,
            &index_bits,
            &opening,
        );

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime =
            Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program.clone(), my_bb_16_perm());
        runtime.witness_stream = witness_stream.into();
        runtime.run().unwrap();
    }

    #[tokio::test]
    async fn test_invalid_merkle_proof() {
        let mut rng = thread_rng();

        let height = rng.gen_range(500..2000);
        let width = rng.gen_range(15..30);
        let num_tensors = rng.gen_range(5..15);

        let num_indices = rng.gen_range(2..10);

        let tensors = (0..num_tensors)
            .map(|_| Tensor::<BabyBear>::rand(&mut rng, [height, width]))
            .collect::<Message<_>>();

        let prover = Poseidon2BabyBear16Prover::default();
        let (root, data) = prover.commit_tensors(tensors.clone()).await.unwrap();

        let indices = (0..num_indices).map(|_| rng.gen_range(0..height)).collect_vec();
        let proof = prover.prove_openings_at_indices(data, &indices).await.unwrap();
        let openings = prover.compute_openings_at_indices(tensors, &indices).await;
        let opening: TensorCsOpening<MerkleTreeTcs<Poseidon2BabyBearConfig>> =
            TensorCsOpening { values: openings, proof };

        let bit_len = height.next_power_of_two().ilog2();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut witness_stream = Vec::new();

        let mut index_bits = Vec::new();
        for index in indices {
            let bits = (0..bit_len)
                .map(|i| if i == 0 { (index >> i) & 1 == 0 } else { (index >> i) & 1 == 1 })
                .collect_vec();
            Witnessable::<AsmConfig<F, EF>>::write(&bits, &mut witness_stream);
            let bits = bits.read(&mut builder);
            index_bits.push(bits);
        }

        Witnessable::<AsmConfig<F, EF>>::write(&root, &mut witness_stream);
        let root = root.read(&mut builder);
        Witnessable::<AsmConfig<F, EF>>::write(&opening, &mut witness_stream);
        let opening = opening.read(&mut builder);

        RecursiveMerkleTreeTcs::<AsmConfig<F, EF>, BabyBearPoseidon2>::verify_tensor_openings(
            &mut builder,
            &root,
            &index_bits,
            &opening,
        );

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime =
            Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program.clone(), my_bb_16_perm());
        runtime.witness_stream = witness_stream.into();
        runtime.run().expect_err("merkle proof should not verify");
    }
}

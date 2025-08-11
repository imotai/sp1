use crate::{
    machine::{InnerVal, SP1ShapedWitnessValues},
    shard::RecursiveShardVerifier,
};
use std::marker::PhantomData;

use super::{PublicValuesOutputDigest, SP1CompressVerifier, SP1ShapedWitnessVariable};
use crate::{
    basefold::{
        merkle_tree::{verify, MerkleProof},
        RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier,
    },
    challenger::DuplexChallengerVariable,
    hash::FieldHasher,
    jagged::RecursiveJaggedConfig,
    zerocheck::RecursiveVerifierConstraintFolder,
    CircuitConfig, FieldHasherVariable, SP1FieldConfigVariable, EF,
};
use serde::{Deserialize, Serialize};
use slop_air::Air;
use slop_algebra::{extension::BinomialExtensionField, AbstractField};
use slop_jagged::JaggedConfig;
use sp1_hypercube::{air::MachineAir, MachineConfig, SP1CoreJaggedConfig};
use sp1_primitives::SP1Field;
use sp1_recursion_compiler::ir::{Builder, Felt};
use sp1_recursion_executor::DIGEST_SIZE;

/// A program to verify a batch of recursive proofs and aggregate their public values.
#[derive(Debug, Clone, Copy)]
pub struct SP1MerkleProofVerifier<C, SC> {
    _phantom: PhantomData<(C, SC)>,
}

#[derive(Clone)]
pub struct MerkleProofVariable<C: CircuitConfig, HV: FieldHasherVariable<C>> {
    pub index: Vec<C::Bit>,
    pub path: Vec<HV::DigestVariable>,
}

/// Witness layout for the compress stage verifier.
pub struct SP1MerkleProofWitnessVariable<
    C: CircuitConfig<F = SP1Field>,
    SC: FieldHasherVariable<C> + SP1FieldConfigVariable<C>,
> {
    /// The shard proofs to verify.
    pub vk_merkle_proofs: Vec<MerkleProofVariable<C, SC>>,
    /// Hinted values to enable dummy digests.
    pub values: Vec<SC::DigestVariable>,
    /// The root of the merkle tree.
    pub root: SC::DigestVariable,
}

/// An input layout for the reduce verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "SC::Digest: Serialize"))]
#[serde(bound(deserialize = "SC::Digest: Deserialize<'de>"))]
pub struct SP1MerkleProofWitnessValues<SC: FieldHasher<SP1Field>> {
    pub vk_merkle_proofs: Vec<MerkleProof<SP1Field, SC>>,
    pub values: Vec<SC::Digest>,
    pub root: SC::Digest,
}

impl<C, SC> SP1MerkleProofVerifier<C, SC>
where
    SC: SP1FieldConfigVariable<C>,
    C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>,
{
    /// Verify (via Merkle tree) that the vkey digests of a proof belong to a specified set
    /// (encoded the Merkle tree proofs in input).
    pub fn verify(
        builder: &mut Builder<C>,
        digests: Vec<SC::DigestVariable>,
        input: SP1MerkleProofWitnessVariable<C, SC>,
        value_assertions: bool,
    ) {
        let SP1MerkleProofWitnessVariable { vk_merkle_proofs, values, root } = input;
        for ((proof, value), expected_value) in
            vk_merkle_proofs.into_iter().zip(values).zip(digests)
        {
            verify::<C, SC>(builder, proof.path, proof.index, value, root);
            if value_assertions {
                SC::assert_digest_eq(builder, expected_value, value);
            } else {
                SC::assert_digest_eq(builder, value, value);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SP1CompressWithVKeyVerifier<C, SC, A, JC> {
    _phantom: PhantomData<(C, SC, A, JC)>,
}

/// Witness layout for the verifier of the proof shape phase of the compress stage.
pub struct SP1CompressWithVKeyWitnessVariable<
    C: CircuitConfig<F = SP1Field, EF = EF>,
    SC: SP1FieldConfigVariable<C> + Send + Sync,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    pub compress_var: SP1ShapedWitnessVariable<C, SC, JC>,
    pub merkle_var: SP1MerkleProofWitnessVariable<C, SC>,
}

/// An input layout for the verifier of the proof shape phase of the compress stage.
pub struct SP1CompressWithVKeyWitnessValues<SC: MachineConfig + FieldHasher<SP1Field>> {
    pub compress_val: SP1ShapedWitnessValues<SC>,
    pub merkle_val: SP1MerkleProofWitnessValues<SC>,
}

impl<C, SC, A, JC> SP1CompressWithVKeyVerifier<C, SC, A, JC>
where
    SC: SP1FieldConfigVariable<
        C,
        FriChallengerVariable = DuplexChallengerVariable<C>,
        DigestVariable = [Felt<SP1Field>; DIGEST_SIZE],
    >,
    C: CircuitConfig<F = SP1Field, EF = <SC as JaggedConfig>::EF, Bit = Felt<SP1Field>>,
    A: MachineAir<InnerVal> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
    JC: RecursiveJaggedConfig<
        F = SP1Field,
        EF = C::EF,
        Circuit = C,
        Commitment = SC::DigestVariable,
        Challenger = SC::FriChallengerVariable,
        BatchPcsProof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>,
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
{
    /// Verify the proof shape phase of the compress stage.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<A, SC, C, JC>,
        input: SP1CompressWithVKeyWitnessVariable<C, SC, JC>,
        value_assertions: bool,
        kind: PublicValuesOutputDigest,
    ) {
        let values = input
            .compress_var
            .vks_and_proofs
            .iter()
            .map(|(vk, _)| vk.hash(builder))
            .collect::<Vec<_>>();
        let vk_root = input.merkle_var.root.map(|x| builder.eval(x));
        SP1MerkleProofVerifier::verify(builder, values, input.merkle_var, value_assertions);
        SP1CompressVerifier::verify(builder, machine, input.compress_var, vk_root, kind);
    }
}

impl SP1MerkleProofWitnessValues<SP1CoreJaggedConfig> {
    pub fn dummy(num_proofs: usize, height: usize) -> Self {
        let dummy_digest = [SP1Field::zero(); DIGEST_SIZE];
        let vk_merkle_proofs =
            vec![MerkleProof { index: 0, path: vec![dummy_digest; height] }; num_proofs];
        let values = vec![dummy_digest; num_proofs];

        Self { vk_merkle_proofs, values, root: dummy_digest }
    }
}

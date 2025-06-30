use std::{borrow::Borrow, path::Path, str::FromStr};

use crate::{
    utils::{is_recursion_public_values_valid, is_root_public_values_valid},
    HashableKey, OuterSC,
};
use anyhow::Result;
use num_bigint::BigUint;
use slop_algebra::{AbstractField, PrimeField};
use slop_baby_bear::BabyBear;
use sp1_core_executor::{subproof::SubproofVerifier, SP1ReduceProof};
use sp1_primitives::{
    consts::WORD_SIZE,
    io::{blake3_hash, SP1PublicValues},
};
use sp1_recursion_circuit::machine::RootPublicValues;
use sp1_recursion_executor::RecursionPublicValues;
use sp1_recursion_gnark_ffi::{
    Groth16Bn254Proof, Groth16Bn254Prover, PlonkBn254Proof, PlonkBn254Prover,
};
use sp1_stark::{
    air::{PublicValues, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    BabyBearPoseidon2, Bn254JaggedConfig, MachineVerifierConfigError, MachineVerifierError, Word,
};
// // use sp1_recursion_circuit::machine::RootPublicValues;
// // use sp1_recursion_core::{air::RecursionPublicValues, stark::BabyBearPoseidon2Outer};
// // use sp1_recursion_gnark_ffi::{
// //     Groth16Bn254Proof, Groth16Bn254Prover, PlonkBn254Proof, PlonkBn254Prover,
// // };
// use sp1_stark::{
//     air::{PublicValues, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
//     MachineProof, MachineVerifierError, Word,
// };
use thiserror::Error;

// use crate::{
//     components::SP1ProverComponents, CoreSC, SP1CoreProofData, SP1Prover, SP1VerifyingKey,
// };

#[derive(Error, Debug)]
pub enum PlonkVerificationError {
    #[error(
        "the verifying key does not match the inner plonk bn254 proof's committed verifying key"
    )]
    InvalidVerificationKey,
    #[error(
        "the public values in the sp1 proof do not match the public values in the inner plonk
bn254 proof"
    )]
    InvalidPublicValues,
}

#[derive(Error, Debug)]
pub enum Groth16VerificationError {
    #[error(
        "the verifying key does not match the inner groth16 bn254 proof's committed verifying
key"
    )]
    InvalidVerificationKey,
    #[error(
        "the public values in the sp1 proof do not match the public values in the inner groth16
bn254 proof"
    )]
    InvalidPublicValues,
}

impl<C: SP1ProverComponents> SP1Prover<C> {
    /// Verify a core proof by verifying the shards, verifying lookup bus, verifying that the
    /// shards are contiguous and complete.
    pub fn verify(
        &self,
        proof: &SP1CoreProofData,
        vk: &SP1VerifyingKey,
    ) -> Result<(), MachineVerifierConfigError<CoreSC>> {
        let SP1VerifyingKey { vk } = vk;

        if proof.0.is_empty() {
            return Err(MachineVerifierError::EmptyProof);
        }

        // First shard has a "CPU" constraint.
        //
        // Assert that the first shard has a "CPU".
        let first_shard = proof.0.first().unwrap();
        let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
            first_shard.public_values.as_slice().borrow();
        if public_values.execution_shard != BabyBear::one()
            || public_values.next_execution_shard != BabyBear::two()
        {
            return Err(MachineVerifierError::InvalidPublicValues(
                "first shard does not contain CPU",
            ));
        }

        // Shard constraints.
        //
        // Initialization:
        // - Shard should start at one.
        //
        // Transition:
        // - Shard should increment by one for each shard.
        let mut current_shard = BabyBear::zero();
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
                shard_proof.public_values.as_slice().borrow();
            current_shard += BabyBear::one();
            if public_values.shard != current_shard {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "shard index should be the previous shard index + 1 and start at 1",
                ));
            }
        }

        // Execution shard and timestamp constraints.
        let mut prev_next_execution_shard = BabyBear::one();
        let mut prev_timestamp =
            [BabyBear::zero(), BabyBear::zero(), BabyBear::zero(), BabyBear::one()];
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
                shard_proof.public_values.as_slice().borrow();
            if public_values.execution_shard != prev_next_execution_shard {
                return Err(MachineVerifierError::InvalidPublicValues("invalid execution shard"));
            }
            if public_values.initial_timestamp != prev_timestamp {
                return Err(MachineVerifierError::InvalidPublicValues("invalid initial timestamp"));
            }
            if public_values.execution_shard != public_values.next_execution_shard
                && public_values.initial_timestamp == public_values.last_timestamp
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "timestamp should change on execution shard",
                ));
            }
            if public_values.execution_shard == public_values.next_execution_shard
                && public_values.initial_timestamp != public_values.last_timestamp
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "timestamp should not change on non-execution shard",
                ));
            }
            prev_timestamp = public_values.last_timestamp;
            prev_next_execution_shard = public_values.next_execution_shard;
        }

        // Program counter constraints.
        //
        // Initialization:
        // - `start_pc` should start as `vk.start_pc`.
        //
        // Transition:
        // - `next_pc` of the previous shard should equal `start_pc`.
        // - If it's not a shard with "CPU", then `start_pc` equals `next_pc`.
        // - If it's a shard with "CPU", then `start_pc` should never equal zero.
        //
        // Finalization:
        // - `next_pc` should equal zero.
        let mut prev_next_pc = BabyBear::zero();
        for (i, shard_proof) in proof.0.iter().enumerate() {
            let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
                shard_proof.public_values.as_slice().borrow();
            if i == 0 && public_values.start_pc != vk.pc_start {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "start_pc != vk.start_pc: program counter should start at vk.start_pc",
                ));
            } else if i != 0 && public_values.start_pc != prev_next_pc {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "start_pc != next_pc_prev: start_pc should equal next_pc_prev for all shards",
                ));
            } else if i == proof.0.len() - 1 && public_values.next_pc != BabyBear::zero() {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "next_pc != 0: execution should have halted",
                ));
            } else if public_values.execution_shard == public_values.next_execution_shard
                && public_values.start_pc != public_values.next_pc
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "start_pc != next_pc: start_pc should equal next_pc for non-cpu shards",
                ));
            }
            prev_next_pc = public_values.next_pc;
        }

        // Exit code constraints.
        //
        // - In every shard, the exit code should be zero.
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
                shard_proof.public_values.as_slice().borrow();
            if public_values.shard == BabyBear::one()
                && public_values.prev_exit_code != BabyBear::zero()
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "prev_exit_code != 0: previous exit code should be zero for first shard",
                ));
            }
            if public_values.execution_shard == public_values.next_execution_shard
                && public_values.prev_exit_code != public_values.exit_code
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "prev_exit_code != exit_code: exit code should be same in non-cpu shards",
                ));
            }
            if public_values.prev_exit_code != BabyBear::zero()
                && public_values.prev_exit_code != public_values.exit_code
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "prev_exit_code != exit_code: exit code should change at most once",
                ));
            }
        }

        // Memory initialization & finalization constraints.
        //
        // Initialization:
        // - `previous_init_addr_bits` should be zero.
        // - `previous_finalize_addr_bits` should be zero.
        //
        // Transition:
        // - For all shards, `previous_init_addr_bits` should equal `last_init_addr_bits` of the
        //   previous shard.
        // - For all shards, `previous_finalize_addr_bits` should equal `last_finalize_addr_bits` of
        //   the previous shard.
        // - For shards without "MemoryInit", `previous_init_addr_bits` should equal
        //   `last_init_addr_bits`.
        // - For shards without "MemoryFinalize", `previous_finalize_addr_bits` should equal
        //   `last_finalize_addr_bits`.
        let mut last_init_addr_word_prev = Word([BabyBear::zero(); WORD_SIZE]);
        let mut last_finalize_addr_word_prev = Word([BabyBear::zero(); WORD_SIZE]);
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
                shard_proof.public_values.as_slice().borrow();
            if public_values.previous_init_addr_word != last_init_addr_word_prev {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "previous_init_addr_word != last_init_addr_word_prev",
                ));
            } else if public_values.previous_finalize_addr_word != last_finalize_addr_word_prev {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "previous_finalize_addr_word != last_finalize_addr_word_prev",
                ));
            }
            last_init_addr_word_prev = public_values.last_init_addr_word;
            last_finalize_addr_word_prev = public_values.last_finalize_addr_word;
        }

        // Digest constraints.
        //
        // Initialization:
        // - `committed_value_digest` should be zero.
        // - `deferred_proofs_digest` should be zero.
        //
        // Transition:
        // - If `committed_value_digest_prev` is not zero, then `committed_value_digest` equal
        //  `committed_value_digest_prev`. Otherwise, `committed_value_digest` should equal zero.
        // - If `deferred_proofs_digest_prev` is not zero, then `deferred_proofs_digest` should be
        //   `deferred_proofs_digest_prev`. Otherwise, `deferred_proofs_digest` should be zero.
        // - If it's not a shard with "CPU", then `committed_value_digest` should not change.
        // - If it's not a shard with "CPU", then `deferred_proofs_digest` should not change.
        let zero_committed_value_digest = [[BabyBear::zero(); 4]; PV_DIGEST_NUM_WORDS];
        let zero_deferred_proofs_digest = [BabyBear::zero(); POSEIDON_NUM_WORDS];
        let mut committed_value_digest_prev = zero_committed_value_digest;
        let mut deferred_proofs_digest_prev = zero_deferred_proofs_digest;
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
                shard_proof.public_values.as_slice().borrow();
            if committed_value_digest_prev != zero_committed_value_digest
                && public_values.committed_value_digest != committed_value_digest_prev
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "committed_value_digest != committed_value_digest_prev",
                ));
            } else if deferred_proofs_digest_prev != zero_deferred_proofs_digest
                && public_values.deferred_proofs_digest != deferred_proofs_digest_prev
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "deferred_proofs_digest != deferred_proofs_digest_prev",
                ));
            } else if public_values.execution_shard == public_values.next_execution_shard
                && public_values.committed_value_digest != committed_value_digest_prev
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "committed_value_digest != committed_value_digest_prev",
                ));
            } else if public_values.execution_shard == public_values.next_execution_shard
                && public_values.deferred_proofs_digest != deferred_proofs_digest_prev
            {
                return Err(MachineVerifierError::InvalidPublicValues(
                    "deferred_proofs_digest != deferred_proofs_digest_prev",
                ));
            }
            committed_value_digest_prev = public_values.committed_value_digest;
            deferred_proofs_digest_prev = public_values.deferred_proofs_digest;
        }

        // Verify that the number of shards is not too large.
        if proof.0.len() >= 1 << 24 {
            return Err(MachineVerifierError::TooManyShards);
        }

        // Verify the global cumulative sum is correct.
        let initial_global_cumulative_sum = vk.initial_global_cumulative_sum;
        let mut cumulative_sum = initial_global_cumulative_sum;
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<[_; 4], Word<_>, [_; 4], _> =
                shard_proof.public_values.as_slice().borrow();
            cumulative_sum = cumulative_sum + public_values.global_cumulative_sum;
        }
        if !cumulative_sum.is_zero() {
            return Err(MachineVerifierError::InvalidPublicValues(
                "global cumulative sum is not zero",
            ));
        }

        // Verify the shard proofs.
        for (i, shard_proof) in proof.0.iter().enumerate() {
            let span = tracing::debug_span!("Verify shard proof", i).entered();
            let mut challenger = self.core_prover.verifier().challenger();
            vk.observe_into(&mut challenger);
            self.core_prover
                .verifier()
                .verify_shard(vk, shard_proof, &mut challenger)
                .map_err(MachineVerifierError::InvalidShardProof)?;
            span.exit();
        }

        Ok(())
    }

    /// Verify a compressed proof.
    pub fn verify_compressed(
        &self,
        proof: &SP1ReduceProof<BabyBearPoseidon2>,
        vk: &SP1VerifyingKey,
    ) -> Result<(), MachineVerifierConfigError<CoreSC>> {
        let SP1ReduceProof { vk: compress_vk, proof } = proof;
        let mut challenger = self.recursion_prover.verifier().challenger();
        compress_vk.observe_into(&mut challenger);
        self.recursion_prover
            .verifier()
            .verify_shard(compress_vk, proof, &mut challenger)
            .map_err(MachineVerifierError::InvalidShardProof)?;

        // Validate public values
        let public_values: &RecursionPublicValues<_> = proof.public_values.as_slice().borrow();

        if !is_recursion_public_values_valid(public_values) {
            return Err(MachineVerifierError::InvalidPublicValues(
                "recursion public values are invalid",
            ));
        }

        if public_values.vk_root != self.recursion_prover.recursion_vk_root {
            return Err(MachineVerifierError::InvalidPublicValues("vk_root mismatch"));
        }

        if self.recursion_prover.vk_verification
            && !self.recursion_prover.recursion_vk_map.contains_key(&compress_vk.hash_babybear())
        {
            return Err(MachineVerifierError::InvalidVerificationKey);
        }

        // `is_complete` should be 1. In the reduce program, this ensures that the proof is fully
        // reduced.
        if public_values.is_complete != BabyBear::one() {
            return Err(MachineVerifierError::InvalidPublicValues("is_complete is not 1"));
        }

        // Verify that the proof is for the sp1 vkey we are expecting.
        let vkey_hash = vk.hash_babybear();
        if public_values.sp1_vk_digest != vkey_hash {
            return Err(MachineVerifierError::InvalidPublicValues("sp1 vk hash mismatch"));
        }

        Ok(())
    }

    /// Verify a shrink proof.
    pub fn verify_shrink(
        &self,
        proof: &SP1ReduceProof<BabyBearPoseidon2>,
        vk: &SP1VerifyingKey,
    ) -> Result<(), MachineVerifierConfigError<CoreSC>> {
        let SP1ReduceProof { vk: compress_vk, proof } = proof;
        let mut challenger = self.recursion_prover.shrink_verifier().challenger();
        compress_vk.observe_into(&mut challenger);
        self.recursion_prover
            .shrink_verifier()
            .verify_shard(compress_vk, proof, &mut challenger)
            .map_err(MachineVerifierError::InvalidShardProof)?;

        // Validate public values
        let public_values: &RecursionPublicValues<_> = proof.public_values.as_slice().borrow();

        if !is_recursion_public_values_valid(public_values) {
            return Err(MachineVerifierError::InvalidPublicValues(
                "recursion public values are invalid",
            ));
        }

        if public_values.vk_root != self.recursion_prover.recursion_vk_root {
            return Err(MachineVerifierError::InvalidPublicValues("vk_root mismatch"));
        }

        if self.recursion_prover.vk_verification
            && !self.recursion_prover.recursion_vk_map.contains_key(&compress_vk.hash_babybear())
        {
            return Err(MachineVerifierError::InvalidVerificationKey);
        }

        // `is_complete` should be 1. In the reduce program, this ensures that the proof is fully
        // reduced.
        if public_values.is_complete != BabyBear::one() {
            return Err(MachineVerifierError::InvalidPublicValues("is_complete is not 1"));
        }

        // Verify that the proof is for the sp1 vkey we are expecting.
        let vkey_hash = vk.hash_babybear();
        if public_values.sp1_vk_digest != vkey_hash {
            return Err(MachineVerifierError::InvalidPublicValues("sp1 vk hash mismatch"));
        }

        Ok(())
    }

    /// Verify a wrap bn254 proof.
    pub fn verify_wrap_bn254(
        &self,
        proof: &SP1ReduceProof<Bn254JaggedConfig>,
        vk: &SP1VerifyingKey,
    ) -> Result<(), MachineVerifierConfigError<OuterSC>> {
        let SP1ReduceProof { vk: _, proof } = proof;
        let wrap_vk = self.recursion_prover.wrap_keys().1;
        let mut challenger = self.recursion_prover.wrap_verifier().challenger();
        wrap_vk.observe_into(&mut challenger);
        self.recursion_prover
            .wrap_verifier()
            .verify_shard(&wrap_vk, proof, &mut challenger)
            .map_err(MachineVerifierError::InvalidShardProof)?;

        // Validate public values
        let public_values: &RootPublicValues<_> = proof.public_values.as_slice().borrow();
        if !is_root_public_values_valid(public_values) {
            return Err(MachineVerifierError::InvalidPublicValues(
                "root public values are invalid",
            ));
        }

        // Verify that the proof is for the sp1 vkey we are expecting.
        let vkey_hash = vk.hash_babybear();
        if *public_values.sp1_vk_digest() != vkey_hash {
            return Err(MachineVerifierError::InvalidPublicValues("sp1 vk hash mismatch"));
        }

        Ok(())
    }

    /// Verifies a PLONK proof using the circuit artifacts in the build directory.
    pub fn verify_plonk_bn254(
        &self,
        proof: &PlonkBn254Proof,
        vk: &SP1VerifyingKey,
        public_values: &SP1PublicValues,
        build_dir: &Path,
    ) -> Result<()> {
        let prover = PlonkBn254Prover::new();

        let vkey_hash = BigUint::from_str(&proof.public_inputs[0])?;
        let committed_values_digest = BigUint::from_str(&proof.public_inputs[1])?;
        let exit_code = BigUint::from_str(&proof.public_inputs[2])?;
        let vk_root = BigUint::from_str(&proof.public_inputs[3])?;

        // Verify the proof with the corresponding public inputs.
        prover.verify(
            proof,
            &vkey_hash,
            &committed_values_digest,
            &exit_code,
            &vk_root,
            build_dir,
        )?;

        verify_plonk_bn254_public_inputs(vk, public_values, &proof.public_inputs)?;

        Ok(())
    }

    /// Verifies a Groth16 proof using the circuit artifacts in the build directory.
    pub fn verify_groth16_bn254(
        &self,
        proof: &Groth16Bn254Proof,
        vk: &SP1VerifyingKey,
        public_values: &SP1PublicValues,
        build_dir: &Path,
    ) -> Result<()> {
        let prover = Groth16Bn254Prover::new();

        let vkey_hash = BigUint::from_str(&proof.public_inputs[0])?;
        let committed_values_digest = BigUint::from_str(&proof.public_inputs[1])?;
        let exit_code = BigUint::from_str(&proof.public_inputs[2])?;
        let vk_root = BigUint::from_str(&proof.public_inputs[3])?;

        // Verify the proof with the corresponding public inputs.
        prover.verify(
            proof,
            &vkey_hash,
            &committed_values_digest,
            &exit_code,
            &vk_root,
            build_dir,
        )?;

        verify_groth16_bn254_public_inputs(vk, public_values, &proof.public_inputs)?;

        Ok(())
    }
}

/// Verify the vk_hash and public_values_hash in the public inputs of the PlonkBn254Proof match
/// the expected values.
pub fn verify_plonk_bn254_public_inputs(
    vk: &SP1VerifyingKey,
    public_values: &SP1PublicValues,
    plonk_bn254_public_inputs: &[String],
) -> Result<()> {
    let expected_vk_hash = BigUint::from_str(&plonk_bn254_public_inputs[0])?;
    let expected_public_values_hash = BigUint::from_str(&plonk_bn254_public_inputs[1])?;

    let vk_hash = vk.hash_bn254().as_canonical_biguint();
    if vk_hash != expected_vk_hash {
        return Err(PlonkVerificationError::InvalidVerificationKey.into());
    }

    let public_values_hash = public_values.hash_bn254();
    if public_values_hash != expected_public_values_hash {
        return Err(PlonkVerificationError::InvalidPublicValues.into());
    }

    Ok(())
}

/// Verify the vk_hash and public_values_hash in the public inputs of the Groth16Bn254Proof
/// match the expected values.
pub fn verify_groth16_bn254_public_inputs(
    vk: &SP1VerifyingKey,
    public_values: &SP1PublicValues,
    groth16_bn254_public_inputs: &[String],
) -> Result<()> {
    let expected_vk_hash = BigUint::from_str(&groth16_bn254_public_inputs[0])?;
    let expected_public_values_hash = BigUint::from_str(&groth16_bn254_public_inputs[1])?;

    let vk_hash = vk.hash_bn254().as_canonical_biguint();
    if vk_hash != expected_vk_hash {
        return Err(Groth16VerificationError::InvalidVerificationKey.into());
    }

    let public_values_hash = public_values.hash_bn254();
    if public_values_hash != expected_public_values_hash {
        return Err(Groth16VerificationError::InvalidPublicValues.into());
    }
    verify_public_values(public_values, expected_public_values_hash)?;

    Ok(())
}

/// In SP1, a proof's public values can either be hashed with SHA2 or Blake3. In SP1 V4, there is no
/// metadata attached to the proof about which hasher function was used for public values hashing.
/// Instead, when verifying the proof, the public values are hashed with SHA2 and Blake3, and
/// if either matches the `expected_public_values_hash`, the verification is successful.
///
/// The security for this verification in SP1 V4 derives from the fact that both SHA2 and Blake3 are
/// designed to be collision resistant. It is computationally infeasible to find an input i1 for
/// SHA256 and an input i2 for Blake3 that the same hash value. Doing so would require breaking both
/// algorithms simultaneously.
fn verify_public_values(
    public_values: &SP1PublicValues,
    expected_public_values_hash: BigUint,
) -> Result<()> {
    // First, check if the public values are hashed with SHA256. If that fails, attempt hashing with
    // Blake3. If neither match, return an error.
    let sha256_public_values_hash = public_values.hash_bn254();
    if sha256_public_values_hash != expected_public_values_hash {
        let blake3_public_values_hash = public_values.hash_bn254_with_fn(blake3_hash);
        if blake3_public_values_hash != expected_public_values_hash {
            return Err(Groth16VerificationError::InvalidPublicValues.into());
        }
    }

    Ok(())
}

use crate::{
    components::SP1ProverComponents, CoreSC, InnerSC, SP1CoreProofData, SP1Prover, SP1VerifyingKey,
};
// use sp1_stark::DIGEST_SIZE;

impl<C: SP1ProverComponents> SubproofVerifier for SP1Prover<C> {
    fn verify_deferred_proof(
        &self,
        proof: &sp1_core_machine::reduce::SP1ReduceProof<InnerSC>,
        vk: &sp1_stark::MachineVerifyingKey<CoreSC>,
        _vk_hash: [u32; 8],
        _committed_value_digest: [u32; 8],
    ) -> Result<(), MachineVerifierConfigError<CoreSC>> {
        // Check that the vk hash matches the vk hash from the input.
        // if vk.hash_u32() != vk_hash {
        //     return Err(MachineVerifierError::InvalidPublicValues(
        //         "vk hash from syscall does not match vkey from input",
        //     ));
        // }
        // Check that proof is valid.
        self.verify_compressed(
            &SP1ReduceProof { vk: proof.vk.clone(), proof: proof.proof.clone() },
            &SP1VerifyingKey { vk: vk.clone() },
        )?;
        // // Check that the committed value digest matches the one from syscall
        // let public_values: &RecursionPublicValues<_> =
        //     proof.proof.public_values.as_slice().borrow();
        // for (i, word) in public_values.committed_value_digest.iter().enumerate() {
        //     if *word != committed_value_digest[i].into() {
        //         return Err(MachineVerifierError::InvalidPublicValues(
        //             "committed_value_digest does not match",
        //         ));
        //     }
        // }
        Ok(())
    }
}

//! Common test utilities shared across test modules.

#[cfg(test)]
pub mod tracegen_setup {

    pub const FIBONACCI_PROOF: &[u8] = include_bytes!("../fib_proof.bin");

    /// Macro to setup shrink circuit test data.
    ///
    /// A macro is nice here because we don't have to specify types, and we can easily switch to using core records without introducing complicated traits.
    #[macro_export]
    macro_rules! tracegen_setup {
        () => {{
            let client = sp1_sdk::CpuProver::new_unsound().await;
            let machine = sp1_prover::CompressAir::<$crate::config::Felt>::compress_machine();
            let inner = <sp1_sdk::CpuProver as sp1_sdk::Prover>::inner(&client);
            let prover = inner.prover();

            let compressed_proof: sp1_core_executor::SP1RecursionProof<_, sp1_prover::InnerSC> =
                bincode::deserialize($crate::test_utils::tracegen_setup::FIBONACCI_PROOF).unwrap();

            let sp1_core_executor::SP1RecursionProof { vk: compressed_vk, proof: compressed_proof } = compressed_proof;
            let input = sp1_recursion_circuit::machine::SP1ShapedWitnessValues {
                vks_and_proofs: vec![(compressed_vk.clone(), compressed_proof)],
                is_complete: true,
            };

            let input = prover.recursion().make_merkle_proofs(input);
            let witness = sp1_prover::SP1CircuitWitness::Shrink(input);

            let record = prover.recursion().execute(witness).unwrap();
            let program = record.program.clone();

            (machine, record, program)
        }};
    }

    pub use tracegen_setup;
}

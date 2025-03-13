#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use slop_baby_bear::BabyBear;
    use sp1_recursion_executor::{ExecutionRecord, RecursionProgram};
    use sp1_stark::{
        prover::{CpuProver, MachineProver, MachineProverOpts},
        BabyBearPoseidon2, Machine, MachineProof, MachineVerifier, MachineVerifierError,
        ShardVerifier,
    };
    use tokio::sync::mpsc;
    use tracing::Instrument;

    use crate::machine::RecursionAir;

    #[allow(unused_variables)]
    pub async fn run_test_recursion<const DEGREE: usize>(
        records: Vec<ExecutionRecord<BabyBear>>,
        machine: Machine<BabyBear, RecursionAir<BabyBear, DEGREE>>,
        program: RecursionProgram<BabyBear>,
    ) -> Result<MachineProof<BabyBearPoseidon2>, MachineVerifierError<BabyBearPoseidon2>> {
        let log_blowup = 1;
        let log_stacking_height = 21;
        let max_log_row_count = 21;
        let verifier = ShardVerifier::from_basefold_parameters(
            log_blowup,
            log_stacking_height,
            max_log_row_count,
            machine,
        );
        let prover = CpuProver::new(verifier.clone());

        let (pk, vk) = prover
            .setup(Arc::new(program))
            .instrument(tracing::debug_span!("setup").or_current())
            .await;
        let challenger = verifier.pcs_verifier.challenger();
        let (proof_tx, mut proof_rx) = tokio::sync::mpsc::unbounded_channel();
        let machine_prover_opts = MachineProverOpts::default();
        let prover = MachineProver::new(machine_prover_opts, &Arc::new(prover));
        let (records_tx, records_rx) = mpsc::channel::<ExecutionRecord<BabyBear>>(1);
        for record in records {
            records_tx.send(record).await.unwrap();
        }
        drop(records_tx);
        prover
            .prove_stream(Arc::new(pk), records_rx, proof_tx, challenger)
            .instrument(tracing::debug_span!("prove stream"))
            .await
            .unwrap();

        let mut shard_proofs = Vec::new();
        while let Some(proof) = proof_rx.recv().await {
            shard_proofs.push(proof);
        }

        assert!(shard_proofs.len() == 1);

        let proof = MachineProof { shard_proofs };

        let mut challenger = verifier.pcs_verifier.challenger();
        let machine_verifier = MachineVerifier::new(verifier);
        tracing::debug_span!("verify the proof")
            .in_scope(|| machine_verifier.verify(&vk, &proof, &mut challenger))?;
        Ok(proof)
    }
}

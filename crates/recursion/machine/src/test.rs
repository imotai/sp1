use std::sync::Arc;

use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use slop_merkle_tree::my_bb_16_perm;
use sp1_recursion_executor::{
    linear_program, Block, ExecutionRecord, Instruction, RecursionProgram, Runtime, D,
};
use sp1_stark::{
    prover::{CpuProver, MachineProver, MachineProverOpts},
    BabyBearPoseidon2, Machine, MachineProof, MachineVerifier, MachineVerifierError, ShardVerifier,
};
use tokio::sync::mpsc;
use tracing::Instrument;

use crate::machine::RecursionAir;

/// Runs the given program on machines that use the wide and skinny Poseidon2 chips.
pub async fn run_recursion_test_machines(
    program: RecursionProgram<BabyBear>,
    witness: Vec<Block<BabyBear>>,
) {
    // type A = RecursionAir<BabyBear, 3>;

    let mut runtime = Runtime::<
        BabyBear,
        BinomialExtensionField<BabyBear, D>,
        DiffusionMatrixBabyBear,
    >::new(Arc::new(program.clone()), my_bb_16_perm());
    runtime.witness_stream = witness.into();
    runtime.run().unwrap();

    // // Run with the poseidon2 wide chip.
    // let machine = A::machine_wide_with_all_chips();
    // run_test_recursion(vec![runtime.record.clone()], machine, program.clone()).await.unwrap();

    // // Run with the poseidon2 skinny chip.
    // let skinny_machine = B::machine_skinny_with_all_chips();
    // run_test_recursion(vec![runtime.record], skinny_machine, program).await.unwrap();
    // println!("ran proof gen with machine skinny");
}

/// Constructs a linear program and runs it on machines that use the wide and skinny Poseidon2 chips.
pub async fn test_recursion_linear_program(instrs: Vec<Instruction<BabyBear>>) {
    run_recursion_test_machines(linear_program(instrs).unwrap(), Vec::new()).await;
}

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

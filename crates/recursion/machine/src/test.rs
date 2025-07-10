use std::sync::Arc;

use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use slop_merkle_tree::my_bb_16_perm;
use sp1_recursion_executor::{
    linear_program, Block, ExecutionRecord, Instruction, RecursionProgram, Runtime, D,
};
use sp1_stark::{
    prover::CpuProverBuilder, BabyBearPoseidon2, Machine, MachineProof, MachineVerifier,
    MachineVerifierConfigError, ShardVerifier,
};
use tracing::Instrument;

use crate::machine::RecursionAir;

/// Runs the given program on machines that use the wide and skinny Poseidon2 chips.
pub async fn run_recursion_test_machines(
    program: RecursionProgram<BabyBear>,
    witness: Vec<Block<BabyBear>>,
) {
    type A = RecursionAir<BabyBear, 3, 2>;

    let mut runtime = Runtime::<
        BabyBear,
        BinomialExtensionField<BabyBear, D>,
        DiffusionMatrixBabyBear,
    >::new(Arc::new(program.clone()), my_bb_16_perm());
    runtime.witness_stream = witness.into();
    runtime.run().unwrap();

    // Run with the poseidon2 wide chip.
    let machine = A::machine_wide_with_all_chips();
    run_test_recursion(vec![runtime.record.clone()], machine, program.clone()).await.unwrap();

    // // Run with the poseidon2 skinny chip.
    // let skinny_machine = B::machine_skinny_with_all_chips();
    // run_test_recursion(vec![runtime.record], skinny_machine, program).await.unwrap();
    // println!("ran proof gen with machine skinny");
}

/// Constructs a linear program and runs it on machines that use the wide and skinny Poseidon2
/// chips.
pub async fn test_recursion_linear_program(instrs: Vec<Instruction<BabyBear>>) {
    run_recursion_test_machines(linear_program(instrs).unwrap(), Vec::new()).await;
}

pub async fn run_test_recursion<const DEGREE: usize, const VAR_EVENTS_PER_ROW: usize>(
    records: Vec<ExecutionRecord<BabyBear>>,
    machine: Machine<BabyBear, RecursionAir<BabyBear, DEGREE, VAR_EVENTS_PER_ROW>>,
    program: RecursionProgram<BabyBear>,
) -> Result<MachineProof<BabyBearPoseidon2>, MachineVerifierConfigError<BabyBearPoseidon2>> {
    let log_blowup = 1;
    let log_stacking_height = 22;
    let max_log_row_count = 21;
    let verifier = ShardVerifier::from_basefold_parameters(
        log_blowup,
        log_stacking_height,
        max_log_row_count,
        machine,
    );
    let prover = CpuProverBuilder::simple(verifier.clone()).build();

    let (pk, vk) = prover
        .setup(Arc::new(program), None)
        .instrument(tracing::debug_span!("setup").or_current())
        .await
        .unwrap();
    let pk = unsafe { pk.into_inner() };
    let mut shard_proofs = Vec::with_capacity(records.len());
    for record in records {
        let proof = prover.prove_shard(pk.clone(), record).await.unwrap();
        shard_proofs.push(proof);
    }

    assert!(shard_proofs.len() == 1);

    let proof = MachineProof { shard_proofs };

    let machine_verifier = MachineVerifier::new(verifier);
    tracing::debug_span!("verify the proof").in_scope(|| machine_verifier.verify(&vk, &proof))?;
    Ok(proof)
}

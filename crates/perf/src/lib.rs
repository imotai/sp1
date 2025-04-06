use std::{marker::PhantomData, sync::Arc, time::Duration};

use clap::ValueEnum;
use components::CudaSP1ProverComponents;
use csl_cuda::TaskScope;
use csl_machine::*;
use slop_baby_bear::BabyBear;
use sp1_core_executor::SP1Context;
use sp1_core_machine::{io::SP1Stdin, riscv::RiscvAir, utils::prove_core};
use sp1_prover::{CompressAir, SP1CoreProofData, SP1Prover};
use sp1_recursion_circuit::{
    basefold::{
        stacked::RecursiveStackedPcsVerifier, tcs::RecursiveMerkleTreeTcs,
        RecursiveBasefoldConfigImpl, RecursiveBasefoldVerifier,
    },
    jagged::{
        RecursiveJaggedConfigImpl, RecursiveJaggedEvalSumcheckConfig, RecursiveJaggedPcsVerifier,
    },
    shard::RecursiveShardVerifier,
};
use sp1_recursion_compiler::config::InnerConfig;
use sp1_stark::{BabyBearPoseidon2, MachineVerifier, ShardVerifier};
use tokio::time::Instant;
use tracing::Instrument;

pub use report::{write_measurements_to_csv, Measurement};

mod components;
mod report;

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum Stage {
    Core,
    Compress,
}

pub async fn make_measurement(
    name: &str,
    elf: &[u8],
    stdin: &SP1Stdin,
    stage: Stage,
    t: TaskScope,
) -> Measurement {
    let core_log_blowup = 1;
    let core_log_stacking_height = 21;
    let core_max_log_row_count = 22;

    let compress_log_blowup = 1;
    let compress_log_stacking_height = 20;
    let compress_max_log_row_count = 20;

    let (cycles, num_shards, core_time, compress_time) = match stage {
        Stage::Core => {
            let log_blowup = core_log_blowup;
            let log_stacking_height = core_log_stacking_height;
            let max_log_row_count = core_max_log_row_count;

            let machine = RiscvAir::machine();
            let verifier = ShardVerifier::from_basefold_parameters(
                log_blowup,
                log_stacking_height,
                max_log_row_count,
                machine.clone(),
            );
            let prover = new_cuda_prover_trivial_eval(verifier.clone(), t);

            let program = Arc::new(sp1_core_executor::Program::from(elf).unwrap());
            let (pk, vk) = prover.setup(program.clone()).await;

            let opts = gpu_prover_opts();
            let challenger = verifier.pcs_verifier.challenger();
            let time = Instant::now();
            let (proof, cycles) = prove_core(
                Arc::new(prover),
                Arc::new(pk),
                program,
                stdin,
                opts.core_opts,
                SP1Context::default(),
                challenger,
            )
            .instrument(tracing::info_span!("prove core"))
            .await
            .unwrap();
            let core_time = time.elapsed();

            let mut challenger = verifier.pcs_verifier.challenger();
            let machine_verifier = MachineVerifier::new(verifier);
            tracing::info_span!("verify the proof")
                .in_scope(|| machine_verifier.verify(&vk, &proof, &mut challenger))
                .unwrap();

            let cycles = cycles as usize;
            let num_shards = proof.shard_proofs.len();
            let compress_time = Duration::ZERO;
            (cycles, num_shards, core_time, compress_time)
        }
        Stage::Compress => {
            let machine = RiscvAir::machine();

            type SC = BabyBearPoseidon2;
            type C = InnerConfig;

            // Make a new SP1 prover

            let core_verifier = ShardVerifier::from_basefold_parameters(
                core_log_blowup,
                core_log_stacking_height,
                core_max_log_row_count,
                machine.clone(),
            );

            let recursive_core_verifier = {
                let recursive_verifier = RecursiveBasefoldVerifier {
                    fri_config: core_verifier
                        .pcs_verifier
                        .stacked_pcs_verifier
                        .pcs_verifier
                        .fri_config,
                    tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
                };
                let recursive_verifier =
                    RecursiveStackedPcsVerifier::new(recursive_verifier, core_log_stacking_height);

                let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
                    SC,
                    C,
                    RecursiveJaggedConfigImpl<
                        C,
                        SC,
                        RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
                    >,
                > {
                    stacked_pcs_verifier: recursive_verifier,
                    max_log_row_count: core_max_log_row_count,
                    jagged_evaluator: RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(
                        PhantomData,
                    ),
                };

                RecursiveShardVerifier {
                    machine,
                    pcs_verifier: recursive_jagged_verifier,
                    _phantom: std::marker::PhantomData,
                }
            };

            let machine = CompressAir::<BabyBear>::machine_wide_with_all_chips();
            let compress_shard_verifier = ShardVerifier::from_basefold_parameters(
                compress_log_blowup,
                compress_log_stacking_height,
                compress_max_log_row_count,
                machine.clone(),
            );

            let recursive_compress_verifier = {
                let recursive_verifier = RecursiveBasefoldVerifier {
                    fri_config: compress_shard_verifier
                        .pcs_verifier
                        .stacked_pcs_verifier
                        .pcs_verifier
                        .fri_config,
                    tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
                };
                let recursive_verifier = RecursiveStackedPcsVerifier::new(
                    recursive_verifier,
                    compress_log_stacking_height,
                );

                let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
                    SC,
                    C,
                    RecursiveJaggedConfigImpl<
                        C,
                        SC,
                        RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
                    >,
                > {
                    stacked_pcs_verifier: recursive_verifier,
                    max_log_row_count: compress_max_log_row_count,
                    jagged_evaluator: RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(
                        PhantomData,
                    ),
                };

                RecursiveShardVerifier {
                    machine,
                    pcs_verifier: recursive_jagged_verifier,
                    _phantom: std::marker::PhantomData,
                }
            };

            let core_prover =
                Arc::new(new_cuda_prover_sumcheck_eval(core_verifier.clone(), t.clone()));

            let compress_prover =
                Arc::new(new_cuda_prover_sumcheck_eval(compress_shard_verifier.clone(), t.clone()));
            let prover = SP1Prover::<CudaSP1ProverComponents>::new(
                core_prover,
                core_verifier,
                recursive_core_verifier,
                compress_prover,
                compress_shard_verifier,
                recursive_compress_verifier,
            );

            let prover = Arc::new(prover);

            let opts = gpu_prover_opts();

            let time = Instant::now();
            let (pk, program, vk) =
                prover.setup(elf).instrument(tracing::debug_span!("setup").or_current()).await;
            let _setup_time = time.elapsed();

            let pk = Arc::new(pk);

            tracing::info!("opts shard batch size: {}", opts.core_opts.shard_batch_size);
            let time = Instant::now();
            let core_proof = prover
                .clone()
                .prove_core(pk, program, stdin, opts, SP1Context::default())
                .instrument(tracing::info_span!("prove core"))
                .await
                .unwrap();
            let core_time = time.elapsed();

            let cycles = core_proof.cycles as usize;
            let num_shards = core_proof.proof.0.len();

            // // Serialize the proof and vk and save to a file.
            // let vk_bytes = bincode::serialize(&vk).unwrap();
            // let core_machine_proof = sp1_stark::MachineProof::<BabyBearPoseidon2> {
            //     shard_proofs: core_proof.proof.0.clone(),
            // };
            // let proof_bytes = bincode::serialize(&core_machine_proof).unwrap();
            // std::fs::write("vk.bin", vk_bytes).unwrap();
            // std::fs::write("proof.bin", proof_bytes).unwrap();

            // Verify the proof
            let core_proof_data = SP1CoreProofData(core_proof.proof.0.clone());
            prover.verify(&core_proof_data, &vk).unwrap();

            // Make the compress proof.
            let time = Instant::now();
            let compress_proof = prover
                .clone()
                .compress(&vk, core_proof, vec![], opts)
                .instrument(tracing::info_span!("compress"))
                .await
                .unwrap();
            let compress_time = time.elapsed();

            // Verify the compress proof
            prover.verify_compressed(&compress_proof, &vk).unwrap();

            // Serialize the compress proof and measure it's size.
            let compress_proof_bytes = bincode::serialize(&compress_proof).unwrap();
            let compress_proof_size = compress_proof_bytes.len();
            tracing::info!("compress proof size: {}", compress_proof_size);

            (cycles, num_shards, core_time, compress_time)
        }
    };

    Measurement {
        name: name.to_string(),
        cycles,
        num_shards,
        core_time,
        compress_time,
        shrink_time: Duration::ZERO,
        wrap_time: Duration::ZERO,
    }
}

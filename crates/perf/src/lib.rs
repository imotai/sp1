use std::{sync::Arc, time::Duration};

use clap::ValueEnum;
use csl_cuda::TaskScope;
use csl_prover::{local_gpu_opts, SP1CudaProverBuilder};
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::{local::LocalProver, SP1CoreProofData};
use tokio::time::Instant;
use tracing::Instrument;

pub use report::{write_measurements_to_csv, Measurement};

mod report;

pub const FIBONACCI_LONG_ELF: &[u8] =
    include_bytes!("../../prover/programs/fibonacci/riscv32im-succinct-zkvm-elf");

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum Stage {
    Core,
    Compress,
    Shrink,
    Wrap,
}

pub async fn make_measurement(
    name: &str,
    elf: &[u8],
    stdin: SP1Stdin,
    stage: Stage,
    t: TaskScope,
) -> Measurement {
    let recursion_cache_size = 5;
    let sp1_prover = SP1CudaProverBuilder::new(t.clone())
        .recursion_cache_size(recursion_cache_size)
        .max_reduce_arity(4)
        .build()
        .await;
    let mut opts = local_gpu_opts();
    opts.core_opts
        .retained_events_presets
        .insert(sp1_core_executor::RetainedEventsPreset::Bls12381Field);
    opts.core_opts
        .retained_events_presets
        .insert(sp1_core_executor::RetainedEventsPreset::Bn254Field);
    opts.core_opts.retained_events_presets.insert(sp1_core_executor::RetainedEventsPreset::Sha256);
    opts.core_opts.retained_events_presets.insert(sp1_core_executor::RetainedEventsPreset::U256);

    let prover = Arc::new(LocalProver::new(sp1_prover, opts));

    let time = Instant::now();
    let (pk, program, vk) = prover
        .prover()
        .core()
        .setup(elf)
        .instrument(tracing::debug_span!("setup").or_current())
        .await;
    let _setup_time = time.elapsed();

    let pk = unsafe { pk.into_inner() };

    let time = Instant::now();
    let core_proof = prover
        .clone()
        .prove_core(pk, program, stdin, SP1Context::default())
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
    prover.prover().verify(&core_proof_data, &vk).unwrap();

    if let Stage::Core = stage {
        return Measurement {
            name: name.to_string(),
            cycles,
            num_shards,
            core_time,
            compress_time: Duration::ZERO,
            shrink_time: Duration::ZERO,
            wrap_time: Duration::ZERO,
        };
    }

    // Make the compress proof.
    let time = Instant::now();
    let compress_proof = prover
        .clone()
        .compress(&vk, core_proof, vec![])
        .instrument(tracing::info_span!("compress"))
        .await
        .unwrap();
    let compress_time = time.elapsed();

    if let Stage::Compress = stage {
        return Measurement {
            name: name.to_string(),
            cycles,
            num_shards,
            core_time,
            compress_time,
            shrink_time: Duration::ZERO,
            wrap_time: Duration::ZERO,
        };
    }

    // Verify the compress proof
    prover.prover().verify_compressed(&compress_proof, &vk).unwrap();

    // Serialize the compress proof and measure it's size.
    let compress_proof_bytes = bincode::serialize(&compress_proof).unwrap();
    let compress_proof_size = compress_proof_bytes.len();
    tracing::info!("compress proof size: {}", compress_proof_size);

    let time = Instant::now();
    let shrunk = prover.shrink(compress_proof).await.unwrap();
    let shrink_time = time.elapsed();

    if let Stage::Shrink = stage {
        return Measurement {
            name: name.to_string(),
            cycles,
            num_shards,
            core_time,
            compress_time,
            shrink_time,
            wrap_time: Duration::ZERO,
        };
    }

    prover.prover().verify_shrink(&shrunk, &vk).unwrap();

    let time = Instant::now();
    let wrapped = prover.wrap(shrunk).await.unwrap();
    let wrap_time = time.elapsed();

    prover.prover().verify_wrap_bn254(&wrapped, &vk).unwrap();

    Measurement {
        name: name.to_string(),
        cycles,
        num_shards,
        core_time,
        compress_time,
        shrink_time,
        wrap_time,
    }
}

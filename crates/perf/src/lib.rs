use std::{sync::Arc, time::Duration};

use clap::ValueEnum;
use csl_cuda::TaskScope;
use csl_prover::{local_gpu_opts, SP1CudaProverBuilder};
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::utils::generate_nonce;
use sp1_prover::{local::LocalProver, shapes::DEFAULT_ARITY, SP1CoreProofData};
use tokio::time::Instant;
use tracing::Instrument;

pub use report::{write_measurements_to_csv, Measurement};

mod report;
pub mod telemetry;

pub const FIBONACCI_ELF: &[u8] =
    include_bytes!("../../prover/programs/fibonacci/riscv64im-succinct-zkvm-elf");
pub const KECCAK_ELF: &[u8] =
    include_bytes!("../../prover/programs/keccak/riscv64im-succinct-zkvm-elf");
pub const SHA2_ELF: &[u8] =
    include_bytes!("../../prover/programs/sha2/riscv64im-succinct-zkvm-elf");
pub const LOOP_ELF: &[u8] =
    include_bytes!("../../prover/programs/loop/riscv64im-succinct-zkvm-elf");
pub const POSEIDON2_ELF: &[u8] =
    include_bytes!("../../prover/programs/poseidon2/riscv64im-succinct-zkvm-elf");
pub const RSP_ELF: &[u8] = include_bytes!("../../prover/programs/rsp/riscv64im-succinct-zkvm-elf");

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
        .normalize_cache_size(recursion_cache_size)
        .set_max_compose_arity(DEFAULT_ARITY)
        .without_vk_verification()
        .build()
        .await;
    let opts = local_gpu_opts();

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
    let context = SP1Context::builder().proof_nonce(generate_nonce()).build();

    let core_proof = prover
        .clone()
        .prove_core(pk, program, stdin, context)
        .instrument(tracing::debug_span!("prove core"))
        .await
        .unwrap();
    let core_time = time.elapsed();

    let cycles = core_proof.cycles as usize;
    let num_shards = core_proof.proof.0.len();

    // // Serialize the proof and vk and save to a file.
    // let vk_bytes = bincode::serialize(&vk).unwrap();
    // let core_machine_proof = sp1_hypercube::MachineProof::<BabyBearPoseidon2> {
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
        .instrument(tracing::debug_span!("compress"))
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
    let mut file = std::fs::File::create("crates/prover-clean/fib_proof.bin").unwrap();
    bincode::serialize_into(&mut file, &compress_proof).unwrap();
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

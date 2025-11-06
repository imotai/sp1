use std::{sync::Arc, time::Duration};

use clap::ValueEnum;
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::utils::generate_nonce;
use sp1_prover::SP1ProverComponents;
use sp1_prover::{local::LocalProver, SP1CoreProofData};
use tokio::time::Instant;
use tracing::Instrument;

pub use report::{write_measurements_to_csv, Measurement};

mod report;
pub mod telemetry;

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum ProverBackend {
    /// Use the old prover implementation
    Old,
    /// Use the new prover-clean implementation
    ProverClean,
}

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
pub const RSP_ELF: &[u8] = include_bytes!("../programs/rsp/elf/rsp-client");

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum Stage {
    Core,
    Compress,
    Shrink,
    Wrap,
}

pub async fn make_measurement<C: SP1ProverComponents>(
    name: &str,
    elf: &[u8],
    stdin: SP1Stdin,
    stage: Stage,
    prover: Arc<LocalProver<C>>,
) -> Measurement {
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
            core_time: Some(core_time),
            compress_time: None,
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
            core_time: Some(core_time),
            compress_time: Some(compress_time),
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
            core_time: Some(core_time),
            compress_time: Some(compress_time),
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
        core_time: Some(core_time),
        compress_time: Some(compress_time),
        shrink_time,
        wrap_time,
    }
}

pub fn get_program_and_input(program: String, param: String) -> (Vec<u8>, SP1Stdin) {
    // If the program elf is local, load it.
    if let Some(program_path) = program.strip_prefix("local-") {
        if program_path == "fibonacci" {
            let mut stdin = SP1Stdin::new();
            let n = param.parse::<usize>().unwrap_or(1000);
            stdin.write(&n);
            return (FIBONACCI_ELF.to_vec(), stdin);
        } else if program_path == "loop" {
            let mut stdin = SP1Stdin::new();
            let n = param.parse::<usize>().unwrap_or(1000);
            stdin.write(&n);
            return (LOOP_ELF.to_vec(), stdin);
        } else if program_path == "sha2" {
            let mut stdin = SP1Stdin::new();
            stdin.write_vec(vec![0u8; param.parse::<usize>().unwrap_or(1000)]);
            return (SHA2_ELF.to_vec(), stdin);
        } else if program_path == "keccak" {
            let mut stdin = SP1Stdin::new();
            stdin.write_vec(vec![0u8; param.parse::<usize>().unwrap_or(1000)]);
            return (KECCAK_ELF.to_vec(), stdin);
        } else if program_path == "poseidon2" {
            let mut stdin = SP1Stdin::new();
            let n = param.parse::<usize>().unwrap_or(1000);
            stdin.write(&n);
            return (POSEIDON2_ELF.to_vec(), stdin);
        } else if program_path == "rsp" {
            let mut stdin = SP1Stdin::new();
            let client_input_path = format!("crates/perf/programs/rsp/input/{param}.bin");
            let client_input = std::fs::read(client_input_path).unwrap();
            stdin.write_vec(client_input);
            return (RSP_ELF.to_vec(), stdin);
        } else {
            panic!("invalid program path provided: {program}");
        }
    }

    // Otherwise, assume it's a program from the s3 bucket.
    // Download files from S3
    let s3_path = program;
    let output = std::process::Command::new("aws")
        .args(["s3", "cp", &format!("s3://sp1-testing-suite/{s3_path}/program.bin"), "program.bin"])
        .output()
        .unwrap();
    if !output.status.success() {
        panic!("failed to download program.bin");
    }
    let output = if param.is_empty() {
        std::process::Command::new("aws")
            .args(["s3", "cp", &format!("s3://sp1-testing-suite/{s3_path}/stdin.bin"), "stdin.bin"])
            .output()
            .unwrap()
    } else {
        std::process::Command::new("aws")
            .args([
                "s3",
                "cp",
                &format!("s3://sp1-testing-suite/{s3_path}/input/{param}.bin"),
                "stdin.bin",
            ])
            .output()
            .unwrap()
    };
    if !output.status.success() {
        panic!("failed to download stdin.bin");
    }

    let program_path = "program.bin";
    let stdin_path = "stdin.bin";
    let program = std::fs::read(program_path).unwrap();
    let stdin = std::fs::read(stdin_path).unwrap();
    let stdin: SP1Stdin = bincode::deserialize(&stdin).unwrap();

    // remove the files
    std::fs::remove_file(program_path).unwrap();
    std::fs::remove_file(stdin_path).unwrap();

    (program, stdin)
}

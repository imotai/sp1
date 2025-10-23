use std::{sync::Arc, time::Duration};

use clap::Parser;
use csl_perf::{
    telemetry, Measurement, FIBONACCI_ELF, KECCAK_ELF, LOOP_ELF, POSEIDON2_ELF, SHA2_ELF,
};
use csl_prover::{new_cuda_prover_sumcheck_eval, CudaSP1ProverComponents};
use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;

const RSP_CLIENT_ELF: &[u8] = include_bytes!("../../programs/rsp/elf/rsp-client");

use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover::{components::CoreProver, worker::SP1LocalNodeBuilder, SP1ProverComponents};
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "local-fibonacci")]
    pub program: String,
    #[arg(long, default_value = "1000")]
    pub param: u32,
    #[arg(long, default_value = "5")]
    pub splice_workers: usize,
    #[arg(long, default_value = "10")]
    pub splice_buffer: usize,
    #[arg(long, default_value = "10")]
    pub task_capacity: usize,
    #[arg(long, default_value = "false")]
    pub telemetry: bool,
}

fn get_program_and_input(program: String, param: u32) -> (Vec<u8>, SP1Stdin) {
    // If the program elf is local, load it.
    if let Some(program_path) = program.strip_prefix("local-") {
        if program_path == "fibonacci" {
            let mut stdin = SP1Stdin::new();
            let n = param;
            stdin.write(&n);
            return (FIBONACCI_ELF.to_vec(), stdin);
        } else if program_path == "loop" {
            let mut stdin = SP1Stdin::new();
            let n = param as usize;
            stdin.write(&n);
            return (LOOP_ELF.to_vec(), stdin);
        } else if program_path == "sha2" {
            let mut stdin = SP1Stdin::new();
            stdin.write_vec(vec![0u8; param as usize]);
            return (SHA2_ELF.to_vec(), stdin);
        } else if program_path == "keccak" {
            let mut stdin = SP1Stdin::new();
            stdin.write_vec(vec![0u8; param as usize]);
            return (KECCAK_ELF.to_vec(), stdin);
        } else if program_path == "poseidon2" {
            let mut stdin = SP1Stdin::new();
            let n = param as usize;
            stdin.write(&n);
            return (POSEIDON2_ELF.to_vec(), stdin);
        } else if program_path == "rsp" {
            let mut stdin = SP1Stdin::new();
            let client_input_path = format!("crates/perf/programs/rsp/input/{param}.bin");
            let client_input = std::fs::read(client_input_path).unwrap();
            stdin.write_vec(client_input);
            return (RSP_CLIENT_ELF.to_vec(), stdin);
        } else {
            panic!("invalid program path provided: {program}");
        }
    }

    // Otherwise, assume it's a program from the s3 bucket.
    // Download files from S3
    let s3_path = program;
    std::process::Command::new("aws")
        .args(["s3", "cp", &format!("s3://sp1-testing-suite/{s3_path}/program.bin"), "program.bin"])
        .output()
        .unwrap();
    std::process::Command::new("aws")
        .args(["s3", "cp", &format!("s3://sp1-testing-suite/{s3_path}/stdin.bin"), "stdin.bin"])
        .output()
        .unwrap();

    let program_path = "program.bin";
    let stdin_path = "stdin.bin";
    let program = std::fs::read(program_path).unwrap();
    let stdin = std::fs::read(stdin_path).unwrap();
    let stdin: SP1Stdin = bincode::deserialize(&stdin).unwrap();

    // // remove the files
    // std::fs::remove_file(program_path).unwrap();
    // std::fs::remove_file(stdin_path).unwrap();

    (program, stdin)
}

#[tokio::main]
#[allow(clippy::field_reassign_with_default)]
async fn main() {
    let args = Args::parse();

    // Initialize the tracer.
    if args.telemetry {
        let resource = Resource::new(vec![KeyValue::new("service.name", "csl-perf")]);
        telemetry::init(resource);
    } else {
        csl_tracing::init_tracer();
    }

    // Get the program and input.
    let (elf, stdin) = get_program_and_input(args.program.clone(), args.param);

    // Initialize the AirProver and permits
    let measurement = csl_cuda::spawn(move |t| async move {
        let core_verifier = CudaSP1ProverComponents::core_verifier();
        let core_air_prover: CoreProver<CudaSP1ProverComponents> =
            new_cuda_prover_sumcheck_eval(core_verifier.shard_verifier().clone(), t.clone());
        let core_air_prover = Arc::new(core_air_prover);
        let permits = ProverSemaphore::new(1);

        let client = SP1LocalNodeBuilder::<CudaSP1ProverComponents>::new()
            .with_core_air_prover(core_air_prover, permits)
            .build()
            .unwrap();

        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        let setup_time = time.elapsed();
        tracing::info!("setup time: {:?}", setup_time);

        let context = SP1Context::default();
        let time = tokio::time::Instant::now();
        let proof = client.prove_core(&elf, stdin, context).await.unwrap();
        let core_time = time.elapsed();
        tracing::info!("core proof time: {:?}", core_time);

        let cycles = proof.cycles as usize;
        let num_shards = proof.proof.0.len();

        // Verify the proof
        client.verifier().verify(&proof.proof, &vk).unwrap();

        Measurement {
            name: args.program,
            cycles,
            num_shards,
            core_time,
            compress_time: Duration::ZERO,
            shrink_time: Duration::ZERO,
            wrap_time: Duration::ZERO,
        }
    })
    .await
    .unwrap();

    if args.telemetry {
        tokio::task::spawn_blocking(opentelemetry::global::shutdown_tracer_provider).await.unwrap();
    }

    println!("{measurement}");
}

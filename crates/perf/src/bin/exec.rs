use std::sync::Arc;

use clap::Parser;
use csl_perf::{telemetry, FIBONACCI_ELF, KECCAK_ELF, LOOP_ELF, POSEIDON2_ELF, SHA2_ELF};
use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use sp1_core_executor::SP1CoreOpts;
use sp1_core_machine::io::SP1Stdin;

const RSP_CLIENT_ELF: &[u8] = include_bytes!("../../programs/rsp/elf/rsp-client");

use sp1_prover::worker::{
    ProofId, RequesterId, SP1CoreExecutor, SplicingEngine, SplicingWorker, TrivialWorkerClient,
};
use sp1_prover_types::{ArtifactClient, InMemoryArtifactClient};
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;
use tokio::sync::mpsc;

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
    let (elf, stdin) = get_program_and_input(args.program, args.param);

    // Initialize the artifact and worker clients
    let artifact_client = InMemoryArtifactClient::new();
    let worker_client = TrivialWorkerClient::new(args.task_capacity, artifact_client.clone());

    let splicing_workers = (0..args.splice_workers)
        .map(|_| SplicingWorker::new(artifact_client.clone(), worker_client.clone()))
        .collect::<Vec<_>>();

    let splicing_engine = Arc::new(SplicingEngine::new(splicing_workers, args.splice_buffer));

    let opts = SP1CoreOpts::default();
    let proof_id = ProofId::new("bench_pure_execution");
    let parent_id = None;
    let parent_context = None;
    let requester_id = RequesterId::new("bench_pure_execution");
    let common_input = artifact_client.create_artifact().expect("failed to create artifact");

    let (sender, mut receiver) = mpsc::unbounded_channel();

    let elf_artifact = artifact_client.create_artifact().expect("failed to create artifact");
    let elf_bytes = elf.to_vec();
    artifact_client.upload(&elf_artifact, elf_bytes).await.expect("failed to upload elf");

    let stdin = Arc::new(stdin);

    let mode_artifact = artifact_client.create_artifact().expect("failed to create artifact");
    artifact_client.upload(&mode_artifact, opts).await.expect("failed to upload opts");

    let executor = SP1CoreExecutor::new(
        splicing_engine,
        elf_artifact,
        stdin,
        common_input,
        proof_id,
        parent_id,
        parent_context,
        requester_id,
        sender,
        artifact_client,
        worker_client,
    );

    let counter_handle = tokio::task::spawn(async move {
        let mut shard_counter = 0;
        while receiver.recv().await.is_some() {
            shard_counter += 1;
        }
        println!("num shards: {shard_counter}");
    });

    // Execute and see the result
    let time = tokio::time::Instant::now();
    let result = executor.execute().await.expect("failed to execute");
    let time = time.elapsed();
    println!(
        "cycles: {}, execution time: {:?}, mhz: {}",
        result.cycles,
        time,
        result.cycles as f64 / (time.as_secs_f64() * 1_000_000.0)
    );

    // Make sure the counter is finished before exiting
    counter_handle.await.expect("counter task panicked");
}

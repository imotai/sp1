use std::sync::Arc;

use clap::Parser;
use csl_perf::{get_program_and_input, telemetry};
use csl_prover::local_gpu_opts;

use slop_algebra::AbstractField;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;

use sp1_core_machine::io::SP1Stdin;
use sp1_hypercube::{septic_digest::SepticDigest, MachineVerifyingKey};
use sp1_primitives::{Elf, SP1Field};
use sp1_prover::{
    worker::{
        CommonProverInput, ProofId, RequesterId, SP1CoreExecutor, SplicingEngine, SplicingWorker,
        TaskContext, TrivialWorkerClient,
    },
    SP1VerifyingKey,
};
use sp1_prover_types::{network_base_types::ProofMode, ArtifactClient, InMemoryArtifactClient};
use sp1_sdk::{MockProver, Prover};
use tokio::sync::mpsc;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "local-fibonacci")]
    pub program: String,
    #[arg(long, default_value = "")]
    pub param: String,
    #[arg(long, default_value = "10")]
    pub splice_workers: usize,
    #[arg(long, default_value = "10")]
    pub splice_buffer: usize,
    #[arg(long, default_value = "2")]
    pub send_workers: usize,
    #[arg(long, default_value = "2")]
    pub send_buffer_size: usize,
    #[arg(long, default_value = None)]
    pub chunk_size: Option<u64>,
    #[arg(long, default_value = "10")]
    pub task_capacity: usize,
    #[arg(long, default_value = "false")]
    pub telemetry: bool,
    #[arg(long, default_value = "gas")]
    pub mode: String,
}

// Executes a program similarly to the cluster controller.
async fn execute_node(args: Args, elf: Vec<u8>, stdin: SP1Stdin) {
    // Initialize the artifact and worker clients
    let artifact_client = InMemoryArtifactClient::new();
    let worker_client = TrivialWorkerClient::new(args.task_capacity, artifact_client.clone());

    let splicing_workers = (0..args.splice_workers)
        .map(|_| {
            SplicingWorker::new(
                artifact_client.clone(),
                worker_client.clone(),
                args.send_workers,
                args.send_buffer_size,
            )
        })
        .collect::<Vec<_>>();

    let splicing_engine = Arc::new(SplicingEngine::new(splicing_workers, args.splice_buffer));

    let proof_id = ProofId::new("bench_pure_execution");
    let parent_id = None;
    let parent_context = None;
    let requester_id = RequesterId::new("bench_pure_execution");

    let dummy_vk = MachineVerifyingKey {
        pc_start: [SP1Field::zero(); 3],
        initial_global_cumulative_sum: SepticDigest::zero(),
        preprocessed_commit: [SP1Field::zero(); 8],
        enable_untrusted_programs: SP1Field::zero(),
    };
    let dummy_vk = SP1VerifyingKey { vk: dummy_vk };

    let common_input = CommonProverInput {
        vk: dummy_vk,
        deferred_digest: [0; 8],
        mode: ProofMode::Core,
        num_deferred_proofs: 0,
        nonce: [0; 4],
    };
    let common_input_artifact =
        artifact_client.create_artifact().expect("failed to create artifact");
    artifact_client
        .upload(&common_input_artifact, common_input)
        .await
        .expect("failed to upload common input");

    let (sender, mut receiver) = mpsc::unbounded_channel();

    let elf_artifact = artifact_client.create_artifact().expect("failed to create artifact");
    let elf_bytes = elf.to_vec();
    artifact_client.upload(&elf_artifact, elf_bytes).await.expect("failed to upload elf");

    let stdin = Arc::new(stdin);

    let mut opts = local_gpu_opts();
    if let Some(chunk_size) = args.chunk_size {
        opts.minimal_trace_chunk_threshold = chunk_size;
    }
    let task_context = TaskContext { proof_id, parent_id, parent_context, requester_id };
    let global_memory_buffer_size = 2 * args.splice_workers;
    let executor = SP1CoreExecutor::new(
        splicing_engine,
        global_memory_buffer_size,
        elf_artifact,
        stdin,
        common_input_artifact,
        opts,
        0,
        task_context,
        sender,
        artifact_client,
        worker_client,
        None,
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

// Executes a program while measuring gas and prints the gas report.
async fn execute_gas(elf: Vec<u8>, stdin: SP1Stdin) {
    let prover = MockProver::new().await;

    let (_, report) = prover
        .execute(Elf::from(elf), stdin)
        .calculate_gas(true)
        .deferred_proof_verification(false)
        .await
        .unwrap();

    println!("{}", report);
}

#[tokio::main]
#[allow(clippy::field_reassign_with_default)]
async fn main() {
    let args = Args::parse();
    let args_clone = args.clone();

    // Initialize the tracer.
    if args.telemetry {
        let resource = Resource::new(vec![KeyValue::new("service.name", "csl-execute")]);
        telemetry::init(resource);
    } else {
        csl_tracing::init_tracer();
    }

    // Get the program and input.
    let (elf, stdin) = get_program_and_input(args.program, args.param);

    match args.mode.as_str() {
        "node" => execute_node(args_clone, elf, stdin).await,
        "gas" => execute_gas(elf, stdin).await,
        _ => panic!("invalid mode"),
    }
}

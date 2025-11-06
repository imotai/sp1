use std::time::Duration;

use clap::Parser;
use csl_perf::ProverBackend;
use csl_perf::{get_program_and_input, telemetry, Measurement, Stage};
use csl_prover::cuda_worker_builder;
use csl_prover::prover_clean_worker_builder;
use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use sp1_core_executor::SP1Context;
use sp1_prover::worker::SP1LocalNodeBuilder;
use sp1_prover_types::network_base_types::ProofMode;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "local-fibonacci")]
    pub program: String,
    #[arg(long, default_value = "")]
    pub param: String,
    #[arg(long, default_value = "false")]
    pub telemetry: bool,
    #[arg(long, default_value = "core")]
    pub stage: Stage,
    #[arg(long, short, default_value = "1")]
    pub num_iterations: usize,
    #[arg(long, default_value = "prover-clean")]
    pub backend: ProverBackend,
}

fn proof_mode_from_stage(stage: Stage) -> ProofMode {
    match stage {
        Stage::Core => ProofMode::Core,
        Stage::Compress => ProofMode::Compressed,
        _ => panic!("invalid stage provided: {stage:?}"),
    }
}

#[tokio::main]
#[allow(clippy::field_reassign_with_default)]
async fn main() {
    let args = Args::parse();

    // Load the environment variables.
    dotenv::dotenv().ok();

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
    let measurements = csl_cuda::spawn(move |t| async move {
        let client = match args.backend {
            ProverBackend::Old => {
                SP1LocalNodeBuilder::from_worker_client_builder(cuda_worker_builder(t.clone()))
                    .build()
                    .await
                    .unwrap()
            }
            ProverBackend::ProverClean => SP1LocalNodeBuilder::from_worker_client_builder(
                prover_clean_worker_builder(t.clone()).await,
            )
            .build()
            .await
            .unwrap(),
        };

        let time = tokio::time::Instant::now();
        let context = SP1Context::default();
        tracing::info!("executing the program");
        let (_, _, report) = client.execute(&elf, stdin.clone(), context.clone()).await.unwrap();
        let execute_time = time.elapsed();
        let cycles = report.total_instruction_count() as usize;
        tracing::info!("execute time: {:?}", execute_time);

        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        let setup_time = time.elapsed();
        tracing::info!("setup time: {:?}", setup_time);

        // Run the prover for a number of iterations.
        let mut measurements = Vec::with_capacity(args.num_iterations);
        for _ in 0..args.num_iterations {
            let mode = proof_mode_from_stage(args.stage);
            let stdin = stdin.clone();
            let context = context.clone();
            let time = tokio::time::Instant::now();
            tracing::info!("proving with mode: {mode:?}");
            let proof = client.prove_with_mode(&elf, stdin, context, mode).await.unwrap();
            let proof_time = time.elapsed();
            tracing::info!("proof time: {:?}", proof_time);
            // Verify the proof
            client.verify(&vk, &proof).unwrap();
            let num_shards = proof.num_shards().unwrap();

            let (core_time, compress_time) = match mode {
                ProofMode::Core => (Some(proof_time), None),
                ProofMode::Compressed => (None, Some(proof_time)),
                _ => (None, None),
            };

            let measurement = Measurement {
                name: args.program.clone(),
                cycles,
                num_shards,
                core_time,
                compress_time,
                shrink_time: Duration::ZERO,
                wrap_time: Duration::ZERO,
            };
            println!("{measurement}");
            measurements.push(measurement);
        }
        measurements
    })
    .await
    .unwrap();

    if args.telemetry {
        tokio::task::spawn_blocking(opentelemetry::global::shutdown_tracer_provider).await.unwrap();
    }

    println!("All {} measurements done", measurements.len());
}

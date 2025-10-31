use std::time::Duration;

use clap::Parser;
use csl_perf::{get_program_and_input, telemetry, Measurement, Stage};
use csl_prover::cuda_worker_builder;
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
    #[arg(long, default_value = "1000")]
    pub param: u32,
    #[arg(long, default_value = "false")]
    pub telemetry: bool,
    #[arg(long, default_value = "core")]
    pub stage: Stage,
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
        let client =
            SP1LocalNodeBuilder::from_worker_client_builder(cuda_worker_builder(t.clone()))
                .build()
                .await
                .unwrap();

        let time = tokio::time::Instant::now();
        let context = SP1Context::default();
        let (_, _, report) = client.execute(&elf, stdin.clone(), context.clone()).await.unwrap();
        let execute_time = time.elapsed();
        let cycles = report.total_instruction_count() as usize;
        tracing::info!("execute time: {:?}", execute_time);

        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        let setup_time = time.elapsed();
        tracing::info!("setup time: {:?}", setup_time);

        let time = tokio::time::Instant::now();

        let mode = proof_mode_from_stage(args.stage);

        tracing::info!("proving with mode: {mode:?}");
        let proof = client.prove_with_mode(&elf, stdin, context, mode).await.unwrap();
        let proof_time = time.elapsed();
        tracing::info!("proof time: {:?}", proof_time);

        // let cycles = proof.cycles as usize;
        // let num_shards = proof.proof.0.len();

        // Verify the proof
        client.verify(&vk, &proof).unwrap();

        let num_shards = proof.num_shards().unwrap();

        let (core_time, compress_time) = match mode {
            ProofMode::Core => (Some(proof_time), None),
            ProofMode::Compressed => (None, Some(proof_time)),
            _ => (None, None),
        };

        Measurement {
            name: args.program,
            cycles,
            num_shards,
            core_time,
            compress_time,
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

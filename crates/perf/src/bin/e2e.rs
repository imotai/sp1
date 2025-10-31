use std::sync::Arc;

use clap::{arg, Parser, ValueEnum};
use csl_perf::{get_program_and_input, make_measurement, telemetry, ProverBackend, Stage};
use csl_prover::{local_gpu_opts, SP1CudaProverBuilder, SP1ProverCleanBuilder};
use csl_tracing::init_tracer;
use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use sp1_prover::{local::LocalProver, shapes::DEFAULT_ARITY};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "local-fibonacci")]
    pub program: String,
    #[arg(long, default_value = "false")]
    pub skip_verify: bool,
    #[arg(long, default_value = "core")]
    pub stage: Stage,
    #[arg(long, default_value = "1000")]
    pub param: u32,
    #[arg(long, default_value = "nvtx")]
    pub trace: Trace,
    #[arg(long, default_value = "prover-clean")]
    pub backend: ProverBackend,
}

#[derive(Clone, Debug, ValueEnum)]
enum Trace {
    Nvtx,
    Telemetry,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize the tracer.
    match args.trace {
        Trace::Nvtx => init_tracer(),
        Trace::Telemetry => {
            let resource = Resource::new(vec![KeyValue::new("service.name", "csl-perf")]);
            telemetry::init(resource);
        }
    }

    let name = args.program.clone();
    let stage = args.stage;
    let (elf, stdin) = get_program_and_input(args.program, args.param);

    let measurement = csl_cuda::spawn(move |t| async move {
        let recursion_cache_size = 5;
        let opts = local_gpu_opts();

        match args.backend {
            ProverBackend::Old => {
                let sp1_prover = SP1CudaProverBuilder::new(t.clone())
                    .normalize_cache_size(recursion_cache_size)
                    .set_max_compose_arity(DEFAULT_ARITY)
                    .without_vk_verification()
                    .build()
                    .await;
                let prover = Arc::new(LocalProver::new(sp1_prover, opts));
                make_measurement(&name, &elf, stdin, stage, prover).await
            }
            ProverBackend::ProverClean => {
                let sp1_prover_clean = SP1ProverCleanBuilder::new(t.clone())
                    .await
                    .normalize_cache_size(recursion_cache_size)
                    .set_max_compose_arity(DEFAULT_ARITY)
                    .without_vk_verification()
                    .build()
                    .await;

                let prover = Arc::new(LocalProver::new(sp1_prover_clean, opts));
                make_measurement(&name, &elf, stdin, stage, prover).await
            }
        }
    })
    .await
    .unwrap();

    if let Trace::Telemetry = args.trace {
        tokio::task::spawn_blocking(opentelemetry::global::shutdown_tracer_provider).await.unwrap();
    }

    println!("{measurement}");
}

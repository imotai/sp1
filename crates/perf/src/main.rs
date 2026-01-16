use std::{
    future::Future,
    time::{Duration, Instant},
};

use clap::Parser;
use sp1_prover::ProverMode;
use sp1_sdk::{
    network::{signer::NetworkSigner, FulfillmentStrategy, NetworkMode},
    Elf, ProveRequest, Prover, ProverClient, ProvingKey, SP1Stdin,
};

#[derive(Parser, Clone)]
#[command(about = "Evaluate the performance of SP1 on programs.")]
struct Args {
    /// The program to evaluate.
    #[arg(short, long)]
    pub program: String,

    /// The input to the program being evaluated.
    #[arg(short, long)]
    pub stdin: String,

    /// The prover mode to use.
    #[arg(short, long)]
    pub mode: ProverMode,
}

#[tokio::main]
async fn main() {
    sp1_sdk::utils::setup_logger();

    let args = Args::parse();
    let elf = Elf::Dynamic(std::fs::read(args.program).expect("failed to read program").into());
    let stdin = std::fs::read(args.stdin).expect("failed to read stdin");
    let stdin: SP1Stdin = bincode::deserialize(&stdin).expect("failed to deserialize stdin");

    let performance_report = match args.mode {
        ProverMode::Cpu => {
            let (prover, prover_init_duration) =
                time_operation_fut(async || ProverClient::builder().cpu().build().await).await;

            let ((_, execution_report), execution_duration) = time_operation_fut(async || {
                prover.execute(elf.clone(), stdin.clone()).await.unwrap()
            })
            .await;

            let (pk, setup_duration) =
                time_operation_fut(async || prover.setup(elf).await.unwrap()).await;

            let (proof, prove_duration) =
                time_operation_fut(async || prover.prove(&pk, stdin).compressed().await.unwrap())
                    .await;

            let ((), verify_duration) = time_operation(|| {
                prover.verify(&proof, pk.verifying_key(), None).expect("verification failed")
            });

            PerfResult {
                cycles: execution_report.total_instruction_count(),
                execution_duration,
                prover_init_duration,
                setup_duration,
                prove_duration,
                verify_duration,
            }
        }
        ProverMode::Cuda => {
            let (prover, prover_init_duration) =
                time_operation_fut(async || ProverClient::builder().cuda().build().await).await;

            let ((_, execution_report), execution_duration) = time_operation_fut(async || {
                prover.execute(elf.clone(), stdin.clone()).await.unwrap()
            })
            .await;

            let (pk, setup_duration) =
                time_operation_fut(async || prover.setup(elf).await.unwrap()).await;

            let (proof, prove_duration) =
                time_operation_fut(async || prover.prove(&pk, stdin).compressed().await.unwrap())
                    .await;

            let ((), verify_duration) = time_operation(|| {
                prover.verify(&proof, pk.verifying_key(), None).expect("verification failed")
            });

            PerfResult {
                cycles: execution_report.total_instruction_count(),
                execution_duration,
                prover_init_duration,
                setup_duration,
                prove_duration,
                verify_duration,
            }
        }
        ProverMode::Network => {
            let private_key = std::env::var("NETWORK_PRIVATE_KEY")
                .expect("NETWORK_PRIVATE_KEY environment variable must be set");
            let signer = NetworkSigner::local(&private_key).expect("failed to create signer");
            let (prover, prover_init_duration) = time_operation_fut(async || {
                ProverClient::builder()
                    .network_for(NetworkMode::Mainnet)
                    .rpc_url("https://rpc.sepolia.succinct.xyz")
                    .signer(signer)
                    .build()
                    .await
            })
            .await;

            let ((_, execution_report), execution_duration) = time_operation_fut(async || {
                prover.execute(elf.clone(), stdin.clone()).await.unwrap()
            })
            .await;

            let (pk, setup_duration) =
                time_operation_fut(async || prover.setup(elf).await.unwrap()).await;

            let (proof, prove_duration) = time_operation_fut(async || {
                prover
                    .prove(&pk, stdin)
                    .strategy(FulfillmentStrategy::Auction)
                    .auction_timeout(Duration::from_secs(60))
                    .min_auction_period(1)
                    .cycle_limit(100_000_000_000)
                    .gas_limit(10_000_000_000)
                    .max_price_per_pgu(600_000_000)
                    .skip_simulation(true)
                    .compressed()
                    .await
                    .unwrap()
            })
            .await;

            let ((), verify_duration) = time_operation(|| {
                prover.verify(&proof, pk.verifying_key(), None).expect("verification failed")
            });

            PerfResult {
                cycles: execution_report.total_instruction_count(),
                execution_duration,
                prover_init_duration,
                setup_duration,
                prove_duration,
                verify_duration,
            }
        }
        ProverMode::Mock => unreachable!(),
    };

    println!("{performance_report:#?}");
}

#[derive(Default, Debug, Clone)]
#[allow(dead_code)]
struct PerfResult {
    pub cycles: u64,
    pub execution_duration: Duration,
    pub prover_init_duration: Duration,
    pub setup_duration: Duration,
    pub prove_duration: Duration,
    pub verify_duration: Duration,
}

pub async fn time_operation_fut<Fut, T, F>(operation: F) -> (T, std::time::Duration)
where
    Fut: Future<Output = T>,
    F: FnOnce() -> Fut,
{
    let start = Instant::now();
    let result = operation().await;
    let duration = start.elapsed();
    (result, duration)
}

pub fn time_operation<T>(operation: impl FnOnce() -> T) -> (T, std::time::Duration) {
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}

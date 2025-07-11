use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use clap::{command, Parser};
use sp1_core_executor::Program;
use sp1_prover::{
    local::{LocalProver, LocalProverOpts},
    CpuSP1ProverComponents, ProverMode, SP1ProverBuilder,
};
use sp1_sdk::{self, Elf, SP1Stdin};
use sp1_stark::MachineProof;

#[derive(Parser, Clone)]
#[command(about = "Evaluate the performance of SP1 on programs.")]
struct PerfArgs {
    /// The program to evaluate.
    #[arg(short, long)]
    pub program: String,

    /// The input to the program being evaluated.
    #[arg(short, long)]
    pub stdin: String,

    /// The prover mode to use.
    ///
    /// Provide this only in prove mode.
    #[arg(short, long)]
    pub mode: ProverMode,
}

#[derive(Default, Debug, Clone)]
#[allow(dead_code)]
struct PerfResult {
    pub cycles: u64,
    pub execution_duration: Duration,
    pub prove_core_duration: Duration,
    pub verify_core_duration: Duration,
    pub compress_duration: Duration,
    pub verify_compressed_duration: Duration,
    pub shrink_duration: Duration,
    pub verify_shrink_duration: Duration,
    pub wrap_duration: Duration,
    pub verify_wrap_duration: Duration,
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

pub fn time_operation<T, F>(operation: F) -> (T, std::time::Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}

#[tokio::main]
async fn main() {
    sp1_sdk::utils::setup_logger();
    let args = PerfArgs::parse();

    let elf = std::fs::read(args.program).expect("failed to read program");
    let elf: Elf = elf.into();
    let stdin = std::fs::read(args.stdin).expect("failed to read stdin");
    let stdin: SP1Stdin = bincode::deserialize(&stdin).expect("failed to deserialize stdin");
    let prover =
        SP1ProverBuilder::<CpuSP1ProverComponents>::new().without_vk_verification().build().await;

    let opts = LocalProverOpts::default();
    let prover = Arc::new(LocalProver::new(prover, opts));

    let (pk, _, vk) = prover.prover().core().setup(&elf).await;
    let pk = unsafe { pk.into_inner() };
    match args.mode {
        ProverMode::Cpu => {
            let (report, execution_duration) = time_operation_fut(|| async {
                prover.clone().execute(&elf, &stdin, Default::default())
            })
            .await;

            let cycles = report.expect("execution failed").2.total_instruction_count();
            let (core_proof, prove_core_duration) = time_operation_fut(|| async {
                prover
                    .clone()
                    .prove_core(
                        pk.clone(),
                        Arc::new(Program::from(&elf).unwrap()),
                        stdin.clone(),
                        Default::default(),
                    )
                    .await
                    .unwrap()
            })
            .await;

            let machine_proof: MachineProof<_> =
                MachineProof { shard_proofs: core_proof.proof.0.clone() };
            let (_, verify_core_duration) =
                time_operation(|| prover.prover().core().verifier().verify(&vk.vk, &machine_proof));

            let proofs = stdin.proofs.into_iter().map(|(proof, _)| proof).collect::<Vec<_>>();
            let (compress_proof, compress_duration) = time_operation_fut(|| async {
                prover.clone().compress(&vk, core_proof.clone(), proofs).await.unwrap()
            })
            .await;

            let (_, verify_compressed_duration) =
                time_operation(|| prover.prover().verify_compressed(&compress_proof, &vk));

            let (shrink_proof, shrink_duration) = time_operation_fut(|| async {
                prover.clone().shrink(compress_proof.clone()).await.unwrap()
            })
            .await;

            let (_, verify_shrink_duration) =
                time_operation(|| prover.prover().verify_shrink(&shrink_proof, &vk));

            let (wrapped_bn254_proof, wrap_duration) = time_operation_fut(|| async {
                prover.clone().wrap(shrink_proof.clone()).await.unwrap()
            })
            .await;

            let (_, verify_wrap_duration) =
                time_operation(|| prover.prover().verify_wrap_bn254(&wrapped_bn254_proof, &vk));

            let result = PerfResult {
                cycles,
                execution_duration,
                prove_core_duration,
                verify_core_duration,
                compress_duration,
                verify_compressed_duration,
                shrink_duration,
                verify_shrink_duration,
                wrap_duration,
                verify_wrap_duration,
            };

            println!("{result:?}");
        }
        ProverMode::Cuda => {
            // let server = SP1CudaProver::new(MoongateServer::default())
            //     .expect("failed to initialize CUDA prover");

            // let context = SP1Context::default();
            // let (report, execution_duration) =
            //     time_operation(|| prover.execute(&elf, &stdin, context.clone()));

            // let cycles = report.expect("execution failed").2.total_instruction_count();

            // let (_, _) = time_operation(|| server.setup(&elf).unwrap());

            // let (core_proof, prove_core_duration) =
            //     time_operation(|| server.prove_core(&stdin).unwrap());

            // let (_, verify_core_duration) = time_operation(|| {
            //     prover.verify(&core_proof.proof, &vk).expect("Proof verification failed")
            // });

            // let proofs = stdin.proofs.into_iter().map(|(proof, _)| proof).collect::<Vec<_>>();
            // let (compress_proof, compress_duration) =
            //     time_operation(|| server.compress(&vk, core_proof, proofs).unwrap());

            // let (_, verify_compressed_duration) =
            //     time_operation(|| prover.verify_compressed(&compress_proof, &vk));

            // let (shrink_proof, shrink_duration) =
            //     time_operation(|| server.shrink(compress_proof).unwrap());

            // let (_, verify_shrink_duration) =
            //     time_operation(|| prover.verify_shrink(&shrink_proof, &vk));

            // let (_, wrap_duration) = time_operation(|| server.wrap_bn254(shrink_proof).unwrap());

            // // TODO: FIX
            // //
            // // let (_, verify_wrap_duration) =
            // //     time_operation(|| prover.verify_wrap_bn254(&wrapped_bn254_proof, &vk));

            // let result = PerfResult {
            //     cycles,
            //     execution_duration,
            //     prove_core_duration,
            //     verify_core_duration,
            //     compress_duration,
            //     verify_compressed_duration,
            //     shrink_duration,
            //     verify_shrink_duration,
            //     wrap_duration,
            //     ..Default::default()
            // };

            // println!("{:?}", result);
        }
        ProverMode::Network => {
            // let prover = ProverClient::builder().network().build();
            // let (_, _) = time_operation(|| prover.execute(&elf, &stdin));

            // let prover = ProverClient::builder().network().build();

            // let (_, _) = time_operation(|| prover.execute(&elf, &stdin));

            // let use_groth16: bool = rand::thread_rng().gen();
            // if use_groth16 {
            //     let (proof, _) =
            //         time_operation(|| prover.prove(&pk, &stdin).groth16().run().unwrap());

            //     let (_, _) = time_operation(|| prover.verify(&proof, &vk));
            // } else {
            //     let (proof, _) =
            //         time_operation(|| prover.prove(&pk, &stdin).plonk().run().unwrap());

            //     let (_, _) = time_operation(|| prover.verify(&proof, &vk));
            // }
        }
        ProverMode::Mock => unreachable!(),
    };
}

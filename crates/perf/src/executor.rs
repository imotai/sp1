use std::time::{Duration, Instant};

// use clap::{command, Parser};
// use sp1_primitives::SP1Field;
// use sp1_core_executor::{Executor, ExecutorMode, Program, Trace};
// use sp1_core_machine::shape::CoreShapeConfig;
// use sp1_sdk::{self, SP1Stdin};
// use sp1_hypercube::SP1ProverOpts;
// use sp1_stark::SP1ProverOpts;
use clap::{command, Parser};
use sp1_core_executor::{Executor, ExecutorMode, MinimalExecutor, Program};
use sp1_sdk::{self, SP1Stdin};
use std::sync::Arc;

#[derive(Parser, Clone)]
#[command(about = "Evaluate the performance of SP1 on programs.")]
struct PerfArgs {
    /// The program to evaluate.
    #[arg(short, long)]
    pub program: String,

    /// The input to the program being evaluated.
    #[arg(short, long)]
    pub stdin: String,

    /// The executor mode to use.
    #[arg(short, long)]
    pub executor_mode: ExecutorMode,
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

pub fn time_operation<T, F: FnOnce() -> T>(operation: F) -> (T, std::time::Duration) {
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}
fn main() {
    let args = PerfArgs::parse();

    let elf = std::fs::read(args.program).expect("failed to read program");
    let stdin = std::fs::read(args.stdin).expect("failed to read stdin");
    let stdin: SP1Stdin = bincode::deserialize(&stdin).expect("failed to deserialize stdin");

    let program = Program::from(&elf).expect("failed to parse program");
    let program = Arc::new(program);
    let mut executor = Executor::new(program.clone(), Default::default());
    // executor.maximal_shapes = Some(maximal_shapes);
    executor.write_vecs(&stdin.buffer);
    for (proof, vkey) in stdin.proofs.iter() {
        executor.write_proof(proof.clone(), vkey.clone());
    }

    let mut minimal = MinimalExecutor::tracing(program, None);
    for input in stdin.buffer.iter() {
        minimal.with_input(input);
    }

    match args.executor_mode {
        ExecutorMode::Simple => {
            let (_, execution_duration) = time_operation(|| executor.run_fast());
            println!("Simple mode:");
            println!("cycles: {}", executor.state.global_clk);
            println!(
                "MHZ: {}",
                executor.state.global_clk as f64 / 1_000_000.0 / execution_duration.as_secs_f64()
            );

            let (_, execution_duration) = time_operation(|| minimal.execute_chunk());
            println!("Minimal mode:");
            println!("cycles: {}", minimal.global_clk());
            println!(
                "MHZ: {}",
                minimal.global_clk() as f64 / 1_000_000.0 / execution_duration.as_secs_f64()
            );

            minimal.reset();
            for input in stdin.buffer.iter() {
                minimal.with_input(input);
            }

            let (_, execution_duration) = time_operation(|| minimal.execute_chunk());
            println!("Minimal mode after reset:");
            println!("cycles: {}", minimal.global_clk());
            println!(
                "MHZ: {}",
                minimal.global_clk() as f64 / 1_000_000.0 / execution_duration.as_secs_f64()
            );
        } // ExecutorMode::Checkpoint => {
        //     let (_, execution_duration) = time_operation(|| executor.run_checkpoint(true));
        //     println!("Checkpoint mode:");
        //     println!("cycles: {}", executor.state.global_clk);
        //     println!(
        //         "MHZ: {}",
        //         executor.state.global_clk as f64 / 1_000_000.0 / execution_duration.as_secs_f64()
        //     );
        // }
        // ExecutorMode::Trace => {
        //     let (_, execution_duration) = time_operation(|| executor.run::<Trace>());
        //     println!("Trace mode:");
        //     println!("cycles: {}", executor.state.global_clk);
        //     println!(
        //         "MHZ: {}",
        //         executor.state.global_clk as f64 / 1_000_000.0 / execution_duration.as_secs_f64()
        //     );
        // }
        // ExecutorMode::ShapeCollection => unimplemented!(),
        _ => todo!(),
    }
}

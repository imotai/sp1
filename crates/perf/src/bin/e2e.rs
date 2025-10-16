use clap::{arg, Parser, ValueEnum};
use csl_perf::{
    make_measurement, telemetry, ProverBackend, Stage, FIBONACCI_ELF, KECCAK_ELF, LOOP_ELF,
    POSEIDON2_ELF, SHA2_ELF,
};
use csl_tracing::init_tracer;
use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use sp1_core_machine::io::SP1Stdin;

const RSP_CLIENT_ELF: &[u8] = include_bytes!("../../programs/rsp/elf/rsp-client");
// const RSP_CLIENT_INPUT: &[u8] = include_bytes!("../rsp/input/21000000.bin");

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
    match args.trace {
        Trace::Nvtx => init_tracer(),
        Trace::Telemetry => {
            let resource = Resource::new(vec![KeyValue::new("service.name", "csl-perf")]);
            telemetry::init(resource);
        }
    }

    let name = args.program.clone();
    let stage = args.stage;
    let backend = args.backend;
    let (elf, stdin) = get_program_and_input(args.program, args.param);

    let measurement = csl_cuda::spawn(move |t| async move {
        make_measurement(&name, &elf, stdin, stage, backend, t).await
    })
    .await
    .unwrap();

    if let Trace::Telemetry = args.trace {
        tokio::task::spawn_blocking(opentelemetry::global::shutdown_tracer_provider).await.unwrap();
    }

    println!("{measurement}");
}

use std::sync::Arc;

use clap::{arg, Parser};
use csl_perf::make_measurement;
use csl_tracing::init_tracer;
use sp1_core_executor::Program;
use sp1_core_machine::io::SP1Stdin;

const FIBONACCI_LONG_ELF: &[u8] =
    include_bytes!("../../machine/programs/fibonacci/riscv32im-succinct-zkvm-elf");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "local-fibonacci")]
    pub program: String,
    #[arg(long, default_value = "false")]
    pub skip_verify: bool,
}

fn get_program_and_input(program: String) -> (Arc<Program>, SP1Stdin) {
    // If the program elf is local, load it.
    if let Some(program_path) = program.strip_prefix("local-") {
        assert!(program_path == "fibonacci");
        let program = Arc::new(Program::from(FIBONACCI_LONG_ELF).unwrap());
        let stdin = SP1Stdin::new();
        return (program, stdin);
    }
    // Otherwise, assume it's a progra from the s3 bucket.
    // Download files from S3
    let s3_path = program;
    std::process::Command::new("aws")
        .args([
            "s3",
            "cp",
            &format!("s3://sp1-testing-suite/{}/program.bin", s3_path),
            "program.bin",
        ])
        .output()
        .unwrap();
    std::process::Command::new("aws")
        .args(["s3", "cp", &format!("s3://sp1-testing-suite/{}/stdin.bin", s3_path), "stdin.bin"])
        .output()
        .unwrap();

    let program_path = "program.bin";
    let stdin_path = "stdin.bin";
    let program = std::fs::read(program_path).unwrap();
    let stdin = std::fs::read(stdin_path).unwrap();
    let stdin: SP1Stdin = bincode::deserialize(&stdin).unwrap();

    // remove the files
    std::fs::remove_file(program_path).unwrap();
    std::fs::remove_file(stdin_path).unwrap();

    let program = Program::from(&program).unwrap();

    (Arc::new(program), stdin)
}

#[tokio::main]
async fn main() {
    init_tracer();

    let args = Args::parse();
    let name = args.program.clone();
    let (program, stdin) = get_program_and_input(args.program);

    let measurement =
        csl_cuda::spawn(|t| async move { make_measurement(&name, program, &stdin, t).await })
            .await
            .unwrap()
            .await
            .unwrap();
    println!("{}", measurement);
}

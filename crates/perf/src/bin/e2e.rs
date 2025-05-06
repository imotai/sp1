use clap::{arg, Parser};
use csl_perf::{make_measurement, Stage, FIBONACCI_LONG_ELF};
use csl_tracing::init_tracer;
use sp1_core_machine::io::SP1Stdin;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "local-fibonacci")]
    pub program: String,
    #[arg(long, default_value = "false")]
    pub skip_verify: bool,
    #[arg(long, default_value = "core")]
    pub stage: Stage,
}

fn get_program_and_input(program: String) -> (Vec<u8>, SP1Stdin) {
    // If the program elf is local, load it.
    if let Some(program_path) = program.strip_prefix("local-") {
        assert!(program_path == "fibonacci");
        let stdin = SP1Stdin::new();
        return (FIBONACCI_LONG_ELF.to_vec(), stdin);
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

    // // remove the files
    // std::fs::remove_file(program_path).unwrap();
    // std::fs::remove_file(stdin_path).unwrap();

    (program, stdin)
}

#[tokio::main]
async fn main() {
    init_tracer();

    let args = Args::parse();
    let name = args.program.clone();
    let stage = args.stage;
    let (elf, stdin) = get_program_and_input(args.program);

    let measurement =
        csl_cuda::spawn(
            move |t| async move { make_measurement(&name, &elf, stdin, stage, t).await },
        )
        .await
        .unwrap();
    println!("{}", measurement);
}

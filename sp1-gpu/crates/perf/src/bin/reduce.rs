use clap::Parser;
// use sp1_gpu_perf::FIBONACCI_LONG_ELF;
use sp1_gpu_tracing::init_tracer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "2")]
    pub arity: usize,
}

fn main() {
    init_tracer();

    // let args = Args::parse();
    // let arity = args.arity;
    // let elf = FIBONACCI_LONG_ELF;

    // Get a core proof for the ELF.

    // compute a single recursion.

    // For pairs of proofs, compute the reduce times.
}

use clap::Parser;
use csl_cuda::run_in_place;
use server::Server;

use tracing_subscriber::{filter::EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

mod server;

#[derive(Debug, Parser)]
struct Args {
    #[clap(long)]
    version: bool,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::Registry::default()
        .with(fmt::layer())
        .with(EnvFilter::builder().parse("cuslop=debug").unwrap())
        .try_init()
        .unwrap();

    let args = Args::parse();

    if args.version {
        println!("{}", sp1_primitives::SP1_VERSION);
        return;
    }

    let cuda_device_id = std::env::var("CUDA_VISIBLE_DEVICES")
        .expect("CUDA_VISIBLE_DEVICES must be set")
        .parse()
        .expect("Expected only one CUDA device as a u32");

    let server = Server { cuda_device_id };

    if let Err(e) = run_in_place(|scope| server.run(scope)).await.await {
        eprintln!("Error running server: {}", e);
    }
}

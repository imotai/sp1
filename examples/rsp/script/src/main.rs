use alloy_primitives::B256;
use clap::Parser;
use rsp_client_executor::{io::ClientExecutorInput, CHAIN_ID_ETH_MAINNET};
use std::path::PathBuf;

use sp1_prover::{components::CpuSP1ProverComponents, SP1ProverBuilder};

/// The ELF we want to execute inside the zkVM.
const ELF: &[u8] = include_elf!("rsp-program");

#[derive(Parser, Debug)]
struct Args {
    /// Whether or not to generate a proof.
    #[arg(long, default_value_t = false)]
    prove: bool,
}

fn load_input_from_cache(chain_id: u64, block_number: u64) -> ClientExecutorInput {
    let cache_path = PathBuf::from(format!("./input/{}/{}.bin", chain_id, block_number));
    let mut cache_file = std::fs::File::open(cache_path).unwrap();
    let client_input: ClientExecutorInput = bincode::deserialize_from(&mut cache_file).unwrap();

    client_input
}

#[tokio::main]
async fn main() {
    utils::setup_logger();
    let sp1_prover = SP1ProverBuilder::<CpuSP1ProverComponents>::cpu().build().await;
    let opts = LocalProverOpts {
        core_opts: SP1CoreOpts {
            retained_events_presets: [RetainedEventsPreset::Sha256].into(),
            ..Default::default()
        },
        ..Default::default()
    };
    let prover = Arc::new(LocalProver::new(sp1_prover, opts));

    let (pk, program, vk) = prover
        .prover()
        .core()
        .setup(ELF)
        .instrument(tracing::debug_span!("setup").or_current())
        .await;
}

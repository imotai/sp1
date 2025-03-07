use std::{sync::Arc, time::Duration};

use csl_cuda::TaskScope;
use csl_machine::*;
use sp1_core_executor::{Program, SP1Context};
use sp1_core_machine::{io::SP1Stdin, riscv::RiscvAir, utils::prove_core};
use sp1_stark::{MachineVerifier, ShardVerifier};
use tokio::time::Instant;
use tracing::Instrument;

pub use report::{write_measurements_to_csv, Measurement};

mod report;

pub async fn make_measurement(
    name: &str,
    program: Arc<Program>,
    stdin: &SP1Stdin,
    t: TaskScope,
) -> Measurement {
    let log_blowup = 1;
    let log_stacking_height = 21;
    let max_log_row_count = 21;
    let machine = RiscvAir::machine();
    let verifier = ShardVerifier::from_basefold_parameters(
        log_blowup,
        log_stacking_height,
        max_log_row_count,
        machine,
    );

    let prover = new_cuda_prover(verifier.clone(), t);

    let opts = gpu_prover_opts().core_opts;

    let time = Instant::now();
    let (pk, vk) =
        prover.setup(program.clone()).instrument(tracing::debug_span!("setup").or_current()).await;
    let _setup_time = time.elapsed();

    let challenger = verifier.pcs_verifier.challenger();
    tracing::info!("opts shard batch size: {}", opts.shard_batch_size);
    let time = Instant::now();
    let (proof, cycles) = prove_core(
        Arc::new(prover),
        Arc::new(pk),
        program,
        stdin,
        opts,
        SP1Context::default(),
        challenger,
    )
    .instrument(tracing::debug_span!("prove core"))
    .await
    .unwrap();
    let core_time = time.elapsed();

    let mut challenger = verifier.pcs_verifier.challenger();
    let machine_verifier = MachineVerifier::new(verifier);
    tracing::debug_span!("verify the proof")
        .in_scope(|| machine_verifier.verify(&vk, &proof, &mut challenger))
        .unwrap();

    Measurement {
        name: name.to_string(),
        cycles: cycles as usize,
        num_shards: proof.shard_proofs.len(),
        core_time,
        compress_time: Duration::ZERO,
        shrink_time: Duration::ZERO,
        wrap_time: Duration::ZERO,
    }
}

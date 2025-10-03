use crate::tracegen::JaggedTraceMle;
use core::pin::pin;
use std::collections::{BTreeMap, BTreeSet};
use std::future::ready;
use std::ops::Range;
use std::sync::Arc;

use csl_cuda::{IntoDevice, TaskScope};
use csl_tracegen::CudaTracegenAir;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use slop_air::BaseAir;
use slop_algebra::Field;
use slop_alloc::{Backend, Buffer, CopyIntoBackend, CpuBackend, HasBackend, Slice, ToHost};
use slop_multilinear::Mle;

use sp1_hypercube::{air::MachineAir, Machine};

use sp1_hypercube::{Chip, MachineRecord};

use crate::config::{Ext, Felt};
use crate::jagged::JaggedTransferError;
use crate::DenseData;
use crate::JaggedMle;

pub fn commit(jagged_trace_mle: &JaggedTraceMle<Felt, TaskScope>, use_preprocessed: bool) {
    // let map = if use_preprocessed {
    //     &jagged_trace_mle.dense_data.preprocessed_table_index
    // } else {
    //     &jagged_trace_mle.dense_data.main_table_index
    // };
    todo!();
}
#[cfg(test)]
mod tests {

    use crate::config::Ext;
    use csl_cuda::{run_in_place, TaskScope};
    use csl_tracegen::CudaTraceGenerator;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use serial_test::serial;
    use slop_multilinear::Point;
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};
    use std::sync::Arc;

    use crate::test_utils::tracegen_setup;

    #[serial]
    #[tokio::test]
    async fn test_commit_matches() {
        let (machine, record, program) = tracegen_setup::setup().await;

        let mut rng = StdRng::seed_from_u64(4);
        run_in_place(|scope| async move {
            const CORE_MAX_LOG_ROW_COUNT: u32 = 22;
            // TODO: this belongs somewhere else.
            const CORE_MAX_TRACE_SIZE: u32 = 1 << 29;
            let z_row: Point<Ext, _> = Point::rand(&mut rng, CORE_MAX_LOG_ROW_COUNT);

            let semaphore = ProverSemaphore::new(1);

            // Generate traces using the host tracegen.
            let trace_generator = CudaTraceGenerator::new_in(machine.clone(), scope.clone());
            let warmup_traces = trace_generator
                .generate_traces(
                    program.clone(),
                    record.clone(),
                    CORE_MAX_LOG_ROW_COUNT as usize,
                    semaphore.clone(),
                )
                .await;

            println!(
                "warmup traces generated: {:?}",
                warmup_traces.main_trace_data.shard_chips.len()
            );
        })
        .await;
    }
}

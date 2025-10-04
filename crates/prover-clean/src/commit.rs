use crate::tracegen::JaggedTraceMle;

use csl_cuda::TaskScope;

use crate::config::Felt;

pub fn commit(_jagged_trace_mle: &JaggedTraceMle<Felt, TaskScope>, _use_preprocessed: bool) {
    todo!();
}
#[cfg(test)]
mod tests {

    use csl_cuda::run_in_place;
    use csl_tracegen::CudaTraceGenerator;
    use serial_test::serial;
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};

    use crate::test_utils::tracegen_setup;

    #[serial]
    #[tokio::test]
    async fn test_commit_matches() {
        let (machine, record, program) = tracegen_setup::setup().await;

        run_in_place(|scope| async move {
            const CORE_MAX_LOG_ROW_COUNT: u32 = 22;

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

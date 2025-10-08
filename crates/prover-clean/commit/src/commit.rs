use std::{iter::once, sync::Arc};

use crate::encoder::{encode_batch, SpparkDftKoalaBear};
use csl_cuda::TaskScope;
use cslpc_merkle_tree::Poseidon2KoalaBear16CudaProver;
use cslpc_prover::ProverCleanStackedPcsProverData;
use cslpc_utils::{traces::JaggedTraceMle, Felt, VirtualTensor, GC};
use slop_algebra::AbstractField;
use slop_challenger::IopCtx;
use slop_jagged::JaggedProverData;
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction as _};

pub type Digest = <GC as IopCtx>::Digest;
// TODO: this needs to be generic. for shrink and wrap this is 4. see crates/prover/src/recursion/components.rs for constant values.
const LOG_BLOWUP: usize = 1;

/// TODO: document
pub async fn commit_multilinears(
    jagged_trace_mle: Arc<JaggedTraceMle<Felt, TaskScope>>,
    max_log_row_count: u32,
    log_stacking_height: u32,
    use_preprocessed: bool,
) -> (Digest, JaggedProverData<GC, ProverCleanStackedPcsProverData<GC>>) {
    let (index, padding, virtual_tensor) = if use_preprocessed {
        (
            &jagged_trace_mle.dense().preprocessed_table_index,
            jagged_trace_mle.dense().preprocessed_padding,
            jagged_trace_mle.preprocessed_virtual_tensor(log_stacking_height),
        )
    } else {
        (
            &jagged_trace_mle.dense().main_table_index,
            jagged_trace_mle.dense().main_padding,
            jagged_trace_mle.main_virtual_tensor(log_stacking_height),
        )
    };
    let (mut row_counts, mut column_counts) = (
        index.values().map(|x| x.poly_size).collect::<Vec<_>>(),
        index.values().map(|x| x.num_polys).collect::<Vec<_>>(),
    );

    let encoder = SpparkDftKoalaBear::default();

    let encoded_messages = encode_batch(encoder, LOG_BLOWUP as u32, virtual_tensor).unwrap();

    let virtual_tensor = VirtualTensor::from_tensor(encoded_messages.as_ref());

    // Commit to the tensors.
    let tensor_prover = Poseidon2KoalaBear16CudaProver::default();

    let (commitment, tcs_data) = tensor_prover.commit_tensors(virtual_tensor).await.unwrap();

    let data = ProverCleanStackedPcsProverData {
        merkle_tree_tcs_data: tcs_data,
        interleaved_mles: jagged_trace_mle.clone(),
        codeword_mle: Arc::new(encoded_messages.as_buffer().clone()),
    };

    let num_added_cols = padding.div_ceil(1 << max_log_row_count).max(1);

    row_counts.push(1 << max_log_row_count);
    row_counts.push(padding - (num_added_cols - 1) * (1 << max_log_row_count));
    column_counts.push(num_added_cols - 1);
    column_counts.push(1);

    let (hasher, compressor) = GC::default_hasher_and_compressor();

    let hash = hasher.hash_iter(
        once(Felt::from_canonical_u32(row_counts.len() as u32))
            .chain(row_counts.clone().into_iter().map(|x| Felt::from_canonical_u32(x as u32)))
            .chain(column_counts.clone().into_iter().map(|x| Felt::from_canonical_u32(x as u32))),
    );

    let final_commitment = compressor.compress([commitment, hash]);

    let jagged_prover_data = JaggedProverData {
        pcs_prover_data: data,
        row_counts: Arc::new(row_counts),
        column_counts: Arc::new(column_counts),
        padding_column_count: num_added_cols,
        original_commitment: commitment,
    };

    (final_commitment, jagged_prover_data)
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use csl_cuda::run_in_place;
    use csl_jagged::Poseidon2KoalaBearJaggedCudaProverComponents;
    use csl_tracegen::CudaTraceGenerator;
    use serial_test::serial;
    use slop_jagged::{JaggedPcsVerifier, JaggedProver, KoalaBearPoseidon2};

    use slop_koala_bear::KoalaBearDegree4Duplex;
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};

    use crate::commit::{commit_multilinears, LOG_BLOWUP};
    use cslpc_tracegen::full_tracegen;
    use cslpc_tracegen::test_utils::tracegen_setup::{
        self, CORE_MAX_LOG_ROW_COUNT, CORE_MAX_TRACE_SIZE, LOG_STACKING_HEIGHT,
    };

    #[serial]
    #[tokio::test]
    async fn test_commit_matches() {
        let (machine, record, program) = tracegen_setup::setup().await;

        type JC = KoalaBearPoseidon2;
        type Prover =
            JaggedProver<KoalaBearDegree4Duplex, Poseidon2KoalaBearJaggedCudaProverComponents>;

        run_in_place(|scope| async move {
            let semaphore = ProverSemaphore::new(1);
            // Generate traces using the host tracegen.
            let trace_generator = CudaTraceGenerator::new_in(machine.clone(), scope.clone());
            let old_traces = trace_generator
                .generate_traces(
                    program.clone(),
                    record.clone(),
                    CORE_MAX_LOG_ROW_COUNT as usize,
                    semaphore.clone(),
                )
                .await;

            println!("warmup traces generated: {:?}", old_traces.main_trace_data.shard_chips.len());

            let num_rounds = 2;

            let jagged_verifier = JaggedPcsVerifier::<_, JC>::new(
                LOG_BLOWUP,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT as usize,
                num_rounds,
            );

            // Commit to preprocessed and main using the old prover.
            let jagged_prover = Prover::from_verifier(&jagged_verifier);

            let preprocessed_message = old_traces.preprocessed_traces.values().cloned().collect();
            let main_message = old_traces.main_trace_data.traces.values().cloned().collect();

            let (old_preprocessed_commitment, old_preprocessed_data) =
                jagged_prover.commit_multilinears(preprocessed_message).await.ok().unwrap();
            let (old_main_commitment, old_main_data) =
                jagged_prover.commit_multilinears(main_message).await.ok().unwrap();

            // Commit to preprocessed and main using the new prover.
            // Do tracegen with the new setup.
            let record = Arc::new(record);
            let (_public_values, jagged_trace_data, _chip_set) = full_tracegen(
                &machine,
                program.clone(),
                record.clone(),
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                &scope,
            )
            .await;

            let jagged_trace_data = Arc::new(jagged_trace_data);

            let (new_preprocessed_commitment, new_preprocessed_data) = commit_multilinears(
                jagged_trace_data.clone(),
                CORE_MAX_LOG_ROW_COUNT,
                LOG_STACKING_HEIGHT,
                true,
            )
            .await;

            let (new_main_commitment, new_main_data) = commit_multilinears(
                jagged_trace_data.clone(),
                CORE_MAX_LOG_ROW_COUNT,
                LOG_STACKING_HEIGHT,
                false,
            )
            .await;

            assert_eq!(old_preprocessed_commitment, new_preprocessed_commitment);
            assert_eq!(old_main_commitment, new_main_commitment);
            assert_eq!(old_preprocessed_data.row_counts, new_preprocessed_data.row_counts);
            assert_eq!(old_preprocessed_data.column_counts, new_preprocessed_data.column_counts);
            assert_eq!(
                old_preprocessed_data.padding_column_count,
                new_preprocessed_data.padding_column_count
            );
            assert_eq!(
                old_preprocessed_data.original_commitment,
                new_preprocessed_data.original_commitment
            );
            assert_eq!(old_main_data.row_counts, new_main_data.row_counts);
            assert_eq!(old_main_data.column_counts, new_main_data.column_counts);
            assert_eq!(old_main_data.padding_column_count, new_main_data.padding_column_count);
            assert_eq!(old_main_data.original_commitment, new_main_data.original_commitment);
        })
        .await;
    }
}

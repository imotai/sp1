use std::{iter::once, sync::Arc};

use csl_cuda::TaskScope;
use cslpc_basefold::{ProverCleanFriCudaProver, ProverCleanStackedPcsProverData};
use cslpc_merkle_tree::{SingleLayerMerkleTreeProverError, TcsProverClean};
use cslpc_utils::{traces::JaggedTraceMle, Ext, Felt};
use slop_algebra::AbstractField;
use slop_alloc::HasBackend;
use slop_challenger::IopCtx;
use slop_jagged::JaggedProverData;
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction as _};
use slop_tensor::Tensor;

/// TODO: document
pub async fn commit_multilinears<GC: IopCtx<F = Felt, EF = Ext>, P: TcsProverClean<GC>>(
    jagged_trace_mle: &JaggedTraceMle<Felt, TaskScope>,
    max_log_row_count: u32,
    use_preprocessed: bool,
    basefold_prover: &ProverCleanFriCudaProver<GC, P, Felt>,
) -> Result<
    (GC::Digest, JaggedProverData<GC, ProverCleanStackedPcsProverData<GC>>),
    SingleLayerMerkleTreeProverError,
> {
    let (index, padding, dst) = if use_preprocessed {
        (
            &jagged_trace_mle.dense().preprocessed_table_index,
            jagged_trace_mle.dense().preprocessed_padding,
            Tensor::<Felt, TaskScope>::with_sizes_in(
                [
                    jagged_trace_mle.dense().preprocessed_offset >> basefold_prover.log_height,
                    1 << (basefold_prover.log_height + basefold_prover.config.log_blowup()),
                ],
                jagged_trace_mle.dense().dense.backend().clone(),
            ),
        )
    } else {
        (
            &jagged_trace_mle.dense().main_table_index,
            jagged_trace_mle.dense().main_padding,
            Tensor::<Felt, TaskScope>::with_sizes_in(
                [
                    jagged_trace_mle.dense().main_size() >> basefold_prover.log_height,
                    1 << (basefold_prover.log_height + basefold_prover.config.log_blowup()),
                ],
                jagged_trace_mle.dense().dense.backend().clone(),
            ),
        )
    };
    let (mut row_counts, mut column_counts) = (
        index.values().map(|x| x.poly_size).collect::<Vec<_>>(),
        index.values().map(|x| x.num_polys).collect::<Vec<_>>(),
    );

    let (commitment, data) =
        basefold_prover.encode_and_commit(use_preprocessed, jagged_trace_mle, dst).await?;

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

    Ok((final_commitment, jagged_prover_data))
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use csl_cuda::run_in_place;
    use csl_jagged::Poseidon2KoalaBearJaggedCudaProverComponents;
    use csl_tracegen::CudaTraceGenerator;
    use cslpc_basefold::ProverCleanFriCudaProver;
    use cslpc_merkle_tree::Poseidon2KoalaBear16CudaProver;
    use cslpc_tracegen::test_utils::tracegen_setup::{
        self, CORE_MAX_LOG_ROW_COUNT, LOG_STACKING_HEIGHT,
    };
    use cslpc_tracegen::{full_tracegen, CORE_MAX_TRACE_SIZE};
    use cslpc_utils::TestGC;
    use serial_test::serial;
    use slop_challenger::IopCtx;
    use slop_jagged::{JaggedPcsVerifier, JaggedProver, KoalaBearPoseidon2};
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};

    use crate::commit::commit_multilinears;
    #[serial]
    #[tokio::test]
    async fn test_commit_matches() {
        let (machine, record, program) = tracegen_setup::setup().await;

        type JC = KoalaBearPoseidon2;
        type Prover = JaggedProver<TestGC, Poseidon2KoalaBearJaggedCudaProverComponents>;

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
                1,
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

            let tcs_prover = Poseidon2KoalaBear16CudaProver::default();

            let basefold_prover = ProverCleanFriCudaProver::<TestGC, _, <TestGC as IopCtx>::F>::new(
                tcs_prover,
                jagged_verifier.pcs_verifier.pcs_verifier.fri_config,
                LOG_STACKING_HEIGHT as usize,
            );

            let (new_preprocessed_commitment, new_preprocessed_data) = commit_multilinears(
                &jagged_trace_data,
                CORE_MAX_LOG_ROW_COUNT,
                true,
                &basefold_prover,
            )
            .await
            .unwrap();

            let (new_main_commitment, new_main_data) = commit_multilinears(
                &jagged_trace_data,
                CORE_MAX_LOG_ROW_COUNT,
                false,
                &basefold_prover,
            )
            .await
            .unwrap();

            assert_eq!(old_preprocessed_data.row_counts, new_preprocessed_data.row_counts);
            assert_eq!(old_preprocessed_data.column_counts, new_preprocessed_data.column_counts);
            assert_eq!(
                old_preprocessed_data.padding_column_count,
                new_preprocessed_data.padding_column_count
            );
            assert_eq!(old_main_data.row_counts, new_main_data.row_counts);
            assert_eq!(old_main_data.column_counts, new_main_data.column_counts);
            assert_eq!(old_main_data.padding_column_count, new_main_data.padding_column_count);
            assert_eq!(
                old_preprocessed_data.original_commitment,
                new_preprocessed_data.original_commitment
            );
            assert_eq!(old_main_data.original_commitment, new_main_data.original_commitment);
            assert_eq!(old_preprocessed_commitment, new_preprocessed_commitment);
            assert_eq!(old_main_commitment, new_main_commitment);
        })
        .await;
    }
}

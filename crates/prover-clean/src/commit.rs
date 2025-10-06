use std::marker::PhantomData;
use std::{iter::once, sync::Arc};

use crate::{config::GC, tracegen::JaggedTraceMle};

use crate::config::Felt;
use csl_basefold::{CudaDftEncoder, FriCudaProver, GrindingPowCudaProver};
use csl_cuda::TaskScope;
use csl_jagged::Poseidon2KoalaBearJaggedCudaProverComponents;
use csl_merkle_tree::Poseidon2KoalaBear16CudaProver;
use slop_algebra::AbstractField;
use slop_basefold::FriConfig;
use slop_challenger::IopCtx;
use slop_jagged::JaggedProverData;
use slop_symmetric::CryptographicHasher;

pub type Digest = <GC as IopCtx>::Digest;
// TODO: this needs to be generic. for shrink and wrap this is 4. see crates/prover/src/recursion/components.rs for constant values.
const LOG_BLOWUP: usize = 1;

pub async fn commit_multilinears(
    jagged_trace_mle: &JaggedTraceMle<Felt, TaskScope>,
    max_log_row_count: u32,
    log_stacking_height: u32,
    use_preprocessed: bool,
) -> (Digest, JaggedProverData<GC, Poseidon2KoalaBearJaggedCudaProverComponents>) {
    // let fri_config = FriConfig::auto(LOG_BLOWUP, 84);
    // let pow_prover = PowProver {};
    // let tcs_prover = TcsProver::default();
    // let encoder = CudaDftEncoder { config: fri_config, dft: SpparkDftKoalaBear::default() };
    // let fri_prover = FriCudaProver::<_, _>(PhantomData);

    let index = if use_preprocessed {
        &jagged_trace_mle.dense().preprocessed_table_index
    } else {
        &jagged_trace_mle.dense().main_table_index
    };
    let (mut row_counts, mut column_counts) = (
        index.values().map(|x| x.poly_size).collect::<Vec<_>>(),
        index.values().map(|x: &crate::tracegen::TraceOffset| x.num_polys).collect::<Vec<_>>(),
    );

    todo!();

    // // Check the validity of the input multilinears.
    // for padded_mle in multilinears.iter() {
    //     // Check that the number of variables matches what the prover expects.
    //     assert_eq!(padded_mle.num_variables(), self.max_log_row_count as u32);
    // }

    // Because of the padding in the stacked PCS, it's necessary to add a "dummy columns" in the
    // jagged commitment scheme to pad the area to the next multiple of the stacking height.
    // We do this in the form of two dummy tables, one with the maximum number of rows and possibly
    // multiple columns, and one with a single column and the remaining number of "leftover"
    // values.

    // // To commit to the batch of padded Mles, the underlying PCS prover commits to the dense
    // // representation of all of these Mles (i.e. a single "giga" Mle consisting of all the
    // // entries of all the individual Mles),
    // // padding the total area to the next multiple of the stacking height.

    // let interleaved_mles =
    //     self.stacker.interleave_multilinears(multilinears, self.log_stacking_height).await;
    // Encode the guts of the mle via Reed-Solomon encoding.

    // let encoded_messages = encoder.encode_batch(mles.clone()).await.unwrap();

    // // Commit to the encoded messages.
    // let (commitment, tcs_prover_data) = self
    //     .tcs_prover
    //     .commit_tensors(encoded_messages.clone())
    //     .await
    //     .map_err(BaseFoldConfigProverError::<GC, C>::TcsCommitError)?;

    // Ok((commitment, BasefoldProverData { encoded_messages, tcs_prover_data }));

    // let prover_data = StackedPcsProverData { pcs_batch_data, interleaved_mles };

    // // let (commitment, data, num_added_vals) =
    // //     self.pcs_prover.commit_multilinear(message).await.unwrap();

    // let num_added_cols = num_added_vals.div_ceil(1 << max_log_row_count).max(1);

    // row_counts.push(1 << max_log_row_count);
    // row_counts.push(num_added_vals - (num_added_cols - 1) * (1 << max_log_row_count));
    // column_counts.push(num_added_cols - 1);
    // column_counts.push(1);

    // let (hasher, compressor) = GC::default_hasher_and_compressor();

    // let hash = hasher.hash_iter(
    //     once(Felt::from_canonical_u32(row_counts.len() as u32))
    //         .chain(row_counts.clone().into_iter().map(|x| Felt::from_canonical_u32(x as u32)))
    //         .chain(column_counts.clone().into_iter().map(|x| Felt::from_canonical_u32(x as u32))),
    // );

    // let final_commitment = compressor.compress([commitment, hash]);

    // let jagged_prover_data = JaggedProverData {
    //     pcs_prover_data: data,
    //     row_counts: Arc::new(row_counts),
    //     column_counts: Arc::new(column_counts),
    //     padding_column_count: num_added_cols,
    //     original_commitment: commitment,
    // };

    // Ok((final_commitment, jagged_prover_data))
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

    use crate::{
        commit::commit_multilinears,
        test_utils::tracegen_setup,
        test_utils::tracegen_setup::{
            CORE_MAX_LOG_ROW_COUNT, CORE_MAX_TRACE_SIZE, LOG_STACKING_HEIGHT,
        },
        tracegen::full_tracegen,
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

            let log_blowup = 1;
            let log_stacking_height = 11;
            let num_rounds = 2;

            let jagged_verifier = JaggedPcsVerifier::<_, JC>::new(
                log_blowup,
                log_stacking_height,
                CORE_MAX_LOG_ROW_COUNT as usize,
                num_rounds,
            );

            // Commit to preprocessed and main using the old prove
            let jagged_prover = Prover::from_verifier(&jagged_verifier);

            let preprocessed_message = old_traces.preprocessed_traces.values().cloned().collect();
            let main_message = old_traces.main_trace_data.traces.values().cloned().collect();

            let (old_preprocessed_commitment, old_preprocessed_data) =
                jagged_prover.commit_multilinears(preprocessed_message).await.ok().unwrap();
            let (old_main_commitment, old_main_data) =
                jagged_prover.commit_multilinears(main_message).await.ok().unwrap();

            // Commit to preprocessed and main using the new prove
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

            let (new_preprocessed_commitment, new_preprocessed_data) = commit_multilinears(
                &jagged_trace_data,
                CORE_MAX_LOG_ROW_COUNT,
                LOG_STACKING_HEIGHT,
                true,
            )
            .await;
            let (new_main_commitment, new_main_data) = commit_multilinears(
                &jagged_trace_data,
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

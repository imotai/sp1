use std::{marker::PhantomData, sync::Arc};

use csl_basefold::BasefoldCudaConfig;
use csl_cuda::{TaskScope, ToDevice};
use csl_jagged::JaggedAssistSumAsPolyGPUImpl;
use cslpc_merkle_tree::{SingleLayerMerkleTreeProverError, TcsProverClean};
use cslpc_zerocheck::primitives::evaluate_traces;
use slop_algebra::AbstractField;
use slop_alloc::{Buffer, HasBackend, ToHost};
use slop_basefold::Poseidon2KoalaBear16BasefoldConfig;
use slop_challenger::{FieldChallenger, IopCtx};
use slop_commit::Rounds;
use slop_jagged::{
    JaggedEvalProver, JaggedEvalSumcheckProver, JaggedLittlePolynomialProverParams, JaggedPcsProof,
    JaggedProverData, JaggedProverError, KoalaBearPoseidon2,
};
use slop_multilinear::{Evaluations, Mle, MleEval, Point};
use slop_stacked::StackedPcsProof;
use slop_tensor::Tensor;
use sp1_hypercube::{
    air::MachineAir,
    prover::{AirProver, PreprocessedData, ProverPermit, ProverSemaphore, ProvingKey},
    Machine, MachineConfig, MachineVerifyingKey, ShardProof,
};
use thiserror::Error;

use cslpc_basefold::{ProverCleanFriCudaProver, ProverCleanStackedPcsProverData};
use cslpc_jagged_sumcheck::{generate_jagged_sumcheck_poly, jagged_sumcheck};
use cslpc_utils::{Ext, Felt, JaggedTraceMle, GC};
use tracing::Instrument;

/// A prover for the hypercube STARK, given a configuration.
pub struct CudaShardProver<GC: IopCtx, P: TcsProverClean<GC>> {
    pub max_log_row_count: usize,
    pub log_stacking_height: u32,
    pub basefold_prover: ProverCleanFriCudaProver<GC, P, GC::F>,
    _marker: PhantomData<GC>,
}

pub struct CudaShardProverData<GC: IopCtx> {
    /// The preprocessed traces.
    pub preprocessed_traces: Arc<JaggedTraceMle<Felt, TaskScope>>,
    /// The pcs data for the preprocessed traces.
    pub preprocessed_data: JaggedProverData<GC, ProverCleanStackedPcsProverData<GC>>,
}

impl<GC: IopCtx, Config: MachineConfig<GC>, Air: MachineAir<GC::F>, P: TcsProverClean<GC>>
    AirProver<GC, Config, Air> for CudaShardProver<GC, P>
{
    type PreprocessedData = CudaShardProverData<GC>;

    fn machine(&self) -> &Machine<GC::F, Air> {
        todo!()
    }

    /// Setup a shard, using a verifying key if provided.
    async fn setup_from_vk(
        &self,
        _program: Arc<Air::Program>,
        _vk: Option<MachineVerifyingKey<GC, Config>>,
        _prover_permits: ProverSemaphore,
    ) -> (PreprocessedData<ProvingKey<GC, Config, Air, Self>>, MachineVerifyingKey<GC, Config>)
    {
        todo!()
    }

    /// Setup and prove a shard.
    async fn setup_and_prove_shard(
        &self,
        _program: Arc<Air::Program>,
        _record: Air::Record,
        _vk: Option<MachineVerifyingKey<GC, Config>>,
        _prover_permits: ProverSemaphore,
        _challenger: &mut GC::Challenger,
    ) -> (MachineVerifyingKey<GC, Config>, ShardProof<GC, Config>, ProverPermit) {
        todo!()
    }

    /// Prove a shard with a given proving key.
    async fn prove_shard_with_pk(
        &self,
        _pk: Arc<ProvingKey<GC, Config, Air, Self>>,
        _record: Air::Record,
        _prover_permits: ProverSemaphore,
        _challenger: &mut GC::Challenger,
    ) -> (ShardProof<GC, Config>, ProverPermit) {
        todo!()
    }
}

// An error type for cuda jagged prover
#[derive(Debug, Error)]
pub enum CudaShardProverError {}

impl<P: TcsProverClean<GC>> CudaShardProver<GC, P> {
    /// Commit to a batch of padded multilinears.
    ///
    /// The jagged polynomial commitments scheme is able to commit to sparse polynomials having
    /// very few or no real rows.
    /// **Note** the padding values will be ignored and treated as though they are zero.
    pub async fn commit_multilinears(
        &self,
        multilinears: Arc<JaggedTraceMle<Felt, TaskScope>>,
        use_preprocessed_data: bool,
    ) -> Result<
        (<GC as IopCtx>::Digest, JaggedProverData<GC, ProverCleanStackedPcsProverData<GC>>),
        JaggedProverError<SingleLayerMerkleTreeProverError>,
    > {
        cslpc_commit::commit_multilinears::<P>(
            multilinears,
            self.max_log_row_count as u32,
            self.log_stacking_height,
            use_preprocessed_data,
            &self.basefold_prover,
        )
        .await
        .map_err(JaggedProverError::BatchPcsProverError)
    }

    pub async fn round_batch_evaluations(
        &self,
        stacked_point: &Point<Ext>,
        jagged_trace_mle: Arc<JaggedTraceMle<Felt, TaskScope>>,
    ) -> Rounds<Evaluations<Ext, TaskScope>> {
        let backend = jagged_trace_mle.backend();
        let evaluations = evaluate_traces(&jagged_trace_mle, stacked_point).await;

        async fn mle_eval_from_slice(
            slice: &[Ext],
            backend: &TaskScope,
        ) -> MleEval<Ext, TaskScope> {
            let buf = Buffer::from(slice.to_vec());
            let tensor = Tensor::from(buf);
            let tensor_device = tensor.to_device_in(backend).await.unwrap();
            MleEval::new(tensor_device)
        }

        let mut evals_so_far = 0;
        let mut preprocessed_evaluations = Vec::new();

        for offset in jagged_trace_mle.dense().preprocessed_table_index.values() {
            if offset.poly_size == 0 {
                let zeros = vec![Ext::zero(); offset.num_polys];
                let mle_eval = mle_eval_from_slice(&zeros, backend).await;
                preprocessed_evaluations.push(mle_eval);
            } else {
                // Make an `MleEval` for this table.
                let slice = &evaluations[evals_so_far..evals_so_far + offset.num_polys];
                let mle_eval = mle_eval_from_slice(slice, backend).await;
                preprocessed_evaluations.push(mle_eval);
                evals_so_far += offset.num_polys;
            }
        }
        let preprocessed_evaluations =
            preprocessed_evaluations.into_iter().collect::<Evaluations<_, _>>();

        // Skip the padding column, if it exists.
        evals_so_far = jagged_trace_mle.dense().preprocessed_cols;
        let mut main_evaluations = Vec::new();
        for offset in jagged_trace_mle.dense().main_table_index.values() {
            if offset.poly_size == 0 {
                let zeros = vec![Ext::zero(); offset.num_polys];
                let mle_eval = mle_eval_from_slice(&zeros, backend).await;
                main_evaluations.push(mle_eval);
            } else {
                // Make an `MleEval` for this table.
                let slice = &evaluations[evals_so_far..evals_so_far + offset.num_polys];
                let mle_eval = mle_eval_from_slice(slice, backend).await;
                main_evaluations.push(mle_eval);
                evals_so_far += offset.num_polys;
            }
        }
        let main_evaluations = main_evaluations.into_iter().collect::<Evaluations<_, _>>();

        Rounds::from_iter([preprocessed_evaluations, main_evaluations])
    }

    pub async fn round_stacked_evaluations(
        &self,
        stacked_point: &Point<Ext>,
        jagged_trace_mle: Arc<JaggedTraceMle<Felt, TaskScope>>,
    ) -> Rounds<Evaluations<Ext, TaskScope>> {
        let backend = jagged_trace_mle.backend();
        let log_stacking_height = stacked_point.len();
        let stacking_height = 1 << log_stacking_height;
        let preprocessed_stacked_size =
            jagged_trace_mle.dense().preprocessed_offset / stacking_height;
        let total_preprocessed_size = stacking_height * preprocessed_stacked_size;

        let device_point = stacked_point.to_device_in(backend).await.unwrap();

        // todo: remove this assert, it's kinda useless
        assert!(total_preprocessed_size == jagged_trace_mle.dense().preprocessed_offset);

        let main_stacked_size = jagged_trace_mle.dense().main_size() / stacking_height;
        let total_main_size = stacking_height * main_stacked_size;
        let mut preprocessed_buffer =
            Buffer::with_capacity_in(total_preprocessed_size, backend.clone());
        unsafe {
            preprocessed_buffer.set_len(total_preprocessed_size);
        }
        let dst_slice = &mut preprocessed_buffer[..];
        let src_slice = &jagged_trace_mle.dense().dense[0..total_preprocessed_size];
        unsafe {
            dst_slice.copy_from_slice(src_slice, backend).unwrap();
        }
        let preprocessed_tensor = Tensor::from(preprocessed_buffer);
        let preprocessed_tensor =
            preprocessed_tensor.reshape([preprocessed_stacked_size, stacking_height]);
        let preprocessed_mle = Mle::new(preprocessed_tensor);

        let mut main_buffer = Buffer::with_capacity_in(total_main_size, backend.clone());
        unsafe {
            main_buffer.set_len(total_main_size);
        }
        let dst_slice = &mut main_buffer[..];
        let src_slice = &jagged_trace_mle.dense().dense[total_preprocessed_size..];
        unsafe {
            dst_slice.copy_from_slice(src_slice, backend).unwrap();
        }
        let main_tensor = Tensor::from(main_buffer);
        let main_tensor = main_tensor.reshape([main_stacked_size, stacking_height]);
        let main_mle = Mle::new(main_tensor);

        let (preprocessed_evaluations, main_evaluations) =
            tokio::join!(preprocessed_mle.eval_at(&device_point), main_mle.eval_at(&device_point));

        let preprocessed_evaluations =
            Evaluations { round_evaluations: vec![preprocessed_evaluations] };

        let main_evaluations = Evaluations { round_evaluations: vec![main_evaluations] };

        Rounds::from_iter([preprocessed_evaluations, main_evaluations])
    }

    /// Prove trusted evaluations.
    pub async fn prove_trusted_evaluations(
        &self,
        eval_point: Point<Ext>,
        evaluation_claims: Rounds<Evaluations<Ext, TaskScope>>,
        prover_data: Rounds<JaggedProverData<GC, ProverCleanStackedPcsProverData<GC>>>, // todo: both contain arcs to the same underlying trace
        challenger: &mut <GC as IopCtx>::Challenger,
    ) -> Result<JaggedPcsProof<GC, KoalaBearPoseidon2>, JaggedProverError<CudaShardProverError>>
    {
        let num_col_variables = prover_data
            .iter()
            .map(|data| data.column_counts.iter().sum::<usize>())
            .sum::<usize>()
            .next_power_of_two()
            .ilog2();
        let z_col = (0..num_col_variables)
            .map(|_| challenger.sample_ext_element::<Ext>())
            .collect::<Point<_>>();

        let z_row = eval_point.clone();

        let backend = evaluation_claims[0][0].backend().clone();

        // First, allocate a buffer for all of the column claims on device.
        let total_column_claims = evaluation_claims
            .iter()
            .map(|evals| evals.iter().map(|evals| evals.num_polynomials()).sum::<usize>())
            .sum::<usize>();

        // Add in the dummy padding columns added during the stacked PCS commitment.
        let total_len = total_column_claims
            + prover_data.iter().map(|data| data.padding_column_count).sum::<usize>();

        let mut column_claims: Buffer<Ext, TaskScope> =
            Buffer::with_capacity_in(total_len, backend.clone());

        // Then, copy the column claims from the evaluation claims into the buffer, inserting extra
        // zeros for the dummy columns.
        for (column_claim_round, data) in evaluation_claims.into_iter().zip(prover_data.iter()) {
            for column_claim in column_claim_round.into_iter() {
                column_claims
                    .extend_from_device_slice(column_claim.into_evaluations().as_buffer())?;
            }
            column_claims
                .extend_from_host_slice(vec![Ext::zero(); data.padding_column_count].as_slice())?;
        }

        assert!(prover_data
            .iter()
            .flat_map(|data| data.row_counts.iter())
            .all(|x| *x <= 1 << self.max_log_row_count));

        // Collect the jagged polynomial parameters.
        let params = JaggedLittlePolynomialProverParams::new(
            prover_data
                .iter()
                .flat_map(|data| {
                    data.row_counts
                        .iter()
                        .copied()
                        .zip(data.column_counts.iter().copied())
                        .flat_map(|(row_count, column_count)| {
                            std::iter::repeat_n(row_count, column_count)
                        })
                })
                .collect(),
            self.max_log_row_count,
        );

        // Generate the jagged sumcheck proof.
        let z_row_backend = z_row.copy_into(&backend);
        let z_col_backend = z_col.copy_into(&backend);

        let all_mles = prover_data.last().unwrap().pcs_prover_data.interleaved_mles.clone();

        let eq_z_row = Mle::partial_lagrange(&z_row_backend).await;
        let eq_z_col = Mle::partial_lagrange(&z_col_backend).await;

        // The overall evaluation claim of the sparse polynomial is inferred from the individual
        // table claims.
        let column_claims: Mle<Ext, TaskScope> = Mle::from_buffer(column_claims);

        let sumcheck_claims = column_claims.eval_at(&z_col_backend).await;
        let sumcheck_claims_host = sumcheck_claims.to_host().await.unwrap();
        let sumcheck_claim = sumcheck_claims_host[0];

        let sumcheck_poly = generate_jagged_sumcheck_poly(all_mles.clone(), eq_z_col, eq_z_row);

        // TODO: why are component_poly_evals unused?
        let (sumcheck_proof, _component_poly_evals) =
            jagged_sumcheck(sumcheck_poly, challenger, sumcheck_claim)
                .instrument(tracing::debug_span!("jagged sumcheck"))
                .await;

        let final_eval_point = sumcheck_proof.point_and_eval.0.clone();

        let jagged_eval_prover: JaggedEvalSumcheckProver<
            Felt,
            JaggedAssistSumAsPolyGPUImpl<
                Felt,
                Ext,
                <Poseidon2KoalaBear16BasefoldConfig as BasefoldCudaConfig<GC>>::DeviceChallenger,
            >,
            _,
            _,
        > = JaggedEvalSumcheckProver::default();

        let jagged_eval_proof = jagged_eval_prover
            .prove_jagged_evaluation(
                &params,
                &z_row,
                &z_col,
                &final_eval_point,
                challenger,
                backend.clone(),
            )
            .instrument(tracing::debug_span!("jagged evaluation proof"))
            .await;

        let (row_counts, column_counts): (Rounds<_>, Rounds<_>) = prover_data
            .iter()
            .map(|data| {
                (Clone::clone(data.row_counts.as_ref()), Clone::clone(data.column_counts.as_ref()))
            })
            .unzip();

        let original_commitments: Rounds<_> =
            prover_data.iter().map(|data| data.original_commitment).collect();

        let stacked_prover_data =
            prover_data.into_iter().map(|data| data.pcs_prover_data).collect::<Rounds<_>>();

        let final_eval_point = sumcheck_proof.point_and_eval.0.clone();

        let (_, stack_point) = final_eval_point
            .split_at(final_eval_point.dimension() - self.log_stacking_height as usize);
        // let stack_point = stack_point.copy_into(&backend);

        let batch_evaluations =
            self.round_stacked_evaluations(&stack_point, all_mles.clone()).await;

        let mut host_batch_evaluations = Rounds::new();
        for round_evals in batch_evaluations.iter() {
            let mut host_round_evals = vec![];
            for eval in round_evals.iter() {
                let host_eval = eval.to_host().await.unwrap();
                host_round_evals.extend(host_eval);
            }
            let host_round_evals = Evaluations::new(vec![host_round_evals.into()]);
            host_batch_evaluations.push(host_round_evals);
        }

        for round in batch_evaluations.iter() {
            for claim in round.iter() {
                let host_claim = claim.to_host().await.unwrap();
                for evaluation in host_claim.iter() {
                    challenger.observe_ext_element(*evaluation);
                }
            }
        }

        let pcs_proof = self
            .basefold_prover
            .prove_trusted_evaluations_basefold(
                stack_point,
                batch_evaluations,
                stacked_prover_data,
                challenger,
            )
            .await
            .unwrap();

        let row_counts_and_column_counts: Rounds<Vec<(usize, usize)>> = row_counts
            .into_iter()
            .zip(column_counts.into_iter())
            .map(|(r, c)| r.into_iter().zip(c.into_iter()).collect())
            .collect();

        let host_batch_evaluations = host_batch_evaluations
            .into_iter()
            .map(|round| round.into_iter().flatten().collect::<MleEval<_>>())
            .collect::<Rounds<_>>();

        let stacked_pcs_proof =
            StackedPcsProof { pcs_proof, batch_evaluations: host_batch_evaluations };

        Ok(JaggedPcsProof {
            pcs_proof: stacked_pcs_proof,
            sumcheck_proof,
            jagged_eval_proof,
            params: params.into_verifier_params(),
            row_counts_and_column_counts,
            merkle_tree_commitments: original_commitments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use csl_cuda::run_in_place;
    use cslpc_merkle_tree::Poseidon2KoalaBear16CudaProver;
    use cslpc_tracegen::full_tracegen;
    use cslpc_tracegen::test_utils::tracegen_setup::{
        self, CORE_MAX_LOG_ROW_COUNT, CORE_MAX_TRACE_SIZE, LOG_STACKING_HEIGHT,
    };
    use serial_test::serial;
    use slop_basefold::BasefoldVerifier;
    use slop_jagged::JaggedPcsVerifier;
    use slop_multilinear::MultilinearPcsChallenger;
    use sp1_hypercube::SP1CoreJaggedConfig;

    #[tokio::test]
    #[serial]
    async fn test_prove_trusted_evaluations() {
        let (machine, record, program) = tracegen_setup::setup().await;
        run_in_place(|scope| async move {
            // *********** Generate traces using the host tracegen. ***********
            let (_public_values, jagged_trace_data, _shard_chips) = full_tracegen(
                &machine,
                program.clone(),
                Arc::new(record),
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                &scope,
            )
            .await;

            let jagged_trace_data = Arc::new(jagged_trace_data);

            let verifier = BasefoldVerifier::<GC>::new(1, 2);

            let basefold_prover = ProverCleanFriCudaProver::<GC, _, Felt>::new(
                Poseidon2KoalaBear16CudaProver::default(),
                verifier.fri_config,
                LOG_STACKING_HEIGHT as usize,
            );

            let shard_prover = CudaShardProver {
                max_log_row_count: CORE_MAX_LOG_ROW_COUNT as usize,
                log_stacking_height: LOG_STACKING_HEIGHT,
                basefold_prover,
                _marker: PhantomData,
            };

            let mut challenger = GC::default_challenger();

            let eval_point = challenger.sample_point(CORE_MAX_LOG_ROW_COUNT);

            let evaluation_claims =
                shard_prover.round_batch_evaluations(&eval_point, jagged_trace_data.clone()).await;

            let (preprocessed_digest, preprocessed_prover_data) =
                shard_prover.commit_multilinears(jagged_trace_data.clone(), true).await.unwrap();

            let (main_digest, main_prover_data) =
                shard_prover.commit_multilinears(jagged_trace_data.clone(), false).await.unwrap();

            let prover_data = Rounds::from_iter([preprocessed_prover_data, main_prover_data]);

            let mut prover_challenger = challenger.clone();
            let proof = shard_prover
                .prove_trusted_evaluations(
                    eval_point.clone(),
                    evaluation_claims.clone(),
                    prover_data,
                    &mut prover_challenger,
                )
                .await
                .unwrap();

            let jagged_verifier = JaggedPcsVerifier::<_, SP1CoreJaggedConfig>::new(
                1,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT as usize,
                2,
            );

            let mut all_evaluations = Vec::new();
            for round_evals in evaluation_claims.iter() {
                let mut host_evals = Vec::new();
                for eval in round_evals.iter() {
                    let host_eval = eval.to_host().await.unwrap();
                    host_evals.extend_from_slice(host_eval.evaluations().as_buffer().as_slice());
                }
                let buf = Buffer::from(host_evals);
                let mle_eval = MleEval::new(Tensor::from(buf));
                all_evaluations.push(mle_eval);
            }

            let mut verifier_challenger = challenger.clone();
            jagged_verifier
                .verify_trusted_evaluations(
                    &[preprocessed_digest, main_digest],
                    eval_point,
                    &all_evaluations,
                    &proof,
                    &mut verifier_challenger,
                )
                .unwrap();
        })
        .await;
    }
}

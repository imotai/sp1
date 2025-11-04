use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, marker::PhantomData, sync::Arc};

use slop_algebra::{
    extension::BinomialExtensionField, AbstractExtensionField, AbstractField, ExtensionField,
    TwoAdicField,
};
use slop_alloc::{Buffer, HasBackend, IntoHost};
use slop_basefold::{BasefoldProof, FriConfig};
use slop_basefold_prover::{host_fold_even_odd, BasefoldProverError};
use slop_challenger::{CanObserve, CanSampleBits, FieldChallenger, IopCtx};
use slop_commit::{Message, Rounds};
use slop_koala_bear::KoalaBear;
use slop_merkle_tree::MerkleTreeOpeningAndProof;
use slop_multilinear::{Evaluations, Mle, MleEval, MleFoldBackend, Point};
use slop_tensor::{Tensor, TransposeBackend};

use csl_cuda::{
    args,
    sys::{
        basefold::{
            batch_koala_bear_base_ext_kernel, batch_koala_bear_base_ext_kernel_flattened,
            flatten_to_base_koala_bear_base_ext_kernel,
            transpose_even_odd_koala_bear_base_ext_kernel,
        },
        runtime::KernelPtr,
    },
    IntoDevice, TaskScope,
};
use cslpc_merkle_tree::{MerkleTreeProverData, SingleLayerMerkleTreeProverError, TcsProverClean};
use cslpc_utils::{Ext, Felt, JaggedTraceMle, TraceDenseData};

use crate::{
    encode_batch, DeviceGrindingChallenger, GrindingPowCudaProver, ProverCleanStackedPcsProverData,
    SpparkDftKoalaBear,
};

/// # Safety
///
pub unsafe trait MleBatchKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn batch_mle_kernel() -> KernelPtr;
}

/// # Safety
///
pub unsafe trait RsCodeWordBatchKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn batch_rs_codeword_kernel() -> KernelPtr;
}

/// # Safety
pub unsafe trait RsCodeWordTransposeKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn transpose_even_odd_kernel() -> KernelPtr;
}

/// # Safety
pub unsafe trait MleFlattenKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn flatten_to_base_kernel() -> KernelPtr;
}

#[derive(
    Debug, Clone, Default, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FriCudaProver<GC>(PhantomData<GC>);

impl<GC: IopCtx> FriCudaProver<GC>
where
    GC::F: TwoAdicField,
    GC::EF: TwoAdicField,
    TaskScope: MleBatchKernel<GC::F, GC::EF>
        + RsCodeWordBatchKernel<GC::F, GC::EF>
        + TransposeBackend<GC::EF>,
{
}

pub struct ProverCleanFriCudaProver<GC, P, F> {
    pub tcs_prover: P,
    pub config: FriConfig<F>,
    pub log_height: u32,
    _marker: PhantomData<GC>,
}

impl<GC: IopCtx<F = Felt, EF = Ext>, P> ProverCleanFriCudaProver<GC, P, GC::F>
where
    GC::F: TwoAdicField,
    GC::EF: ExtensionField<GC::F> + TwoAdicField,
    P: TcsProverClean<GC>,

    TaskScope: MleBatchKernel<GC::F, GC::EF>
        + RsCodeWordBatchKernel<GC::F, GC::EF>
        + MleFoldBackend<GC::EF>
        + TransposeBackend<GC::F>
        + TransposeBackend<GC::EF>
        + RsCodeWordTransposeKernel<GC::F, GC::EF>
        + MleFlattenKernel<GC::F, GC::EF>,
{
    pub fn new(tcs_prover: P, config: FriConfig<GC::F>, log_height: u32) -> Self {
        Self { tcs_prover, config, log_height, _marker: PhantomData }
    }
    pub async fn encode_and_commit(
        &self,
        use_preprocessed: bool,
        jagged_trace_mle: &JaggedTraceMle<Felt, TaskScope>,
        mut dst: Tensor<Felt, TaskScope>,
    ) -> Result<
        (<GC as IopCtx>::Digest, ProverCleanStackedPcsProverData<GC>),
        SingleLayerMerkleTreeProverError,
    > {
        let encoder = SpparkDftKoalaBear::default();

        unsafe {
            dst.assume_init();
        }

        let virtual_tensor = if use_preprocessed {
            jagged_trace_mle.preprocessed_virtual_tensor(self.log_height)
        } else {
            jagged_trace_mle.main_virtual_tensor(self.log_height)
        };

        encode_batch(encoder, self.config.log_blowup as u32, virtual_tensor, &mut dst).unwrap();

        // Commit to the tensors.

        let (commitment, tcs_data) = self.tcs_prover.commit_tensors(&dst).await?;

        Ok((
            commitment,
            ProverCleanStackedPcsProverData {
                merkle_tree_tcs_data: tcs_data,
                codeword_mle: Arc::new(dst),
            },
        ))
    }
    pub async fn batch(
        &self,
        batching_challenge: GC::EF,
        mles: &TraceDenseData<GC::F, TaskScope>,
        codewords: Message<Tensor<Felt, TaskScope>>,
        evaluation_claims: Vec<MleEval<GC::EF, TaskScope>>,
    ) -> (Mle<GC::EF, TaskScope>, Tensor<GC::F, TaskScope>, GC::EF) {
        let log_stacking_height = self.log_height;
        // Compute all the batch challenge powers.
        let total_num_polynomials = codewords.iter().map(|c| c.sizes()[0]).sum::<usize>();

        let mut batch_challenge_powers =
            batching_challenge.powers().take(total_num_polynomials).collect::<Vec<_>>();

        // Compute the random linear combination of the MLEs of the columns of the matrices
        let num_variables = log_stacking_height;
        let codeword_size = (codewords.first().unwrap()).sizes()[1];
        let scope: TaskScope = mles.backend().clone();
        let mut batch_mle =
            Mle::new(Tensor::<GC::EF, TaskScope>::zeros_in([1, 1 << num_variables], scope.clone()));
        let mut batch_codeword = Tensor::<GC::F, TaskScope>::zeros_in(
            [<GC::EF as AbstractExtensionField<GC::F>>::D, codeword_size],
            scope.clone(),
        );

        unsafe {
            let block_dim = 256;
            let grid_dim = (1usize << num_variables).div_ceil(block_dim);
            let batch_size = total_num_polynomials;
            let powers_device =
                Buffer::from(batch_challenge_powers.clone()).into_device_in(&scope).await.unwrap();
            let mle_args = args!(
                mles.dense.as_ptr(),
                batch_mle.guts_mut().as_mut_ptr(),
                powers_device.as_ptr(),
                (1 << num_variables) as usize,
                batch_size
            );
            scope
                .launch_kernel(TaskScope::batch_mle_kernel(), grid_dim, block_dim, &mle_args, 0)
                .unwrap();
        }

        for codeword in codewords.iter() {
            let batch_size = codeword.sizes()[0];
            let mut powers = batch_challenge_powers;
            batch_challenge_powers = powers.split_off(batch_size);
            let powers_device = Buffer::from(powers.clone()).into_device_in(&scope).await.unwrap();

            let block_dim = 256;
            let grid_dim = codeword_size.div_ceil(block_dim);
            let codeword_args = args!(
                codeword.as_ptr(),
                batch_codeword.as_mut_ptr(),
                powers_device.as_ptr(),
                codeword_size,
                batch_size
            );
            unsafe {
                scope
                    .launch_kernel(
                        TaskScope::batch_rs_codeword_kernel(),
                        grid_dim,
                        block_dim,
                        &codeword_args,
                        0,
                    )
                    .unwrap();
            }
        }

        // Compute the batched evaluation claim.
        let mut batch_eval_claim = GC::EF::zero();
        let mut power = GC::EF::one();
        for batch_claims in evaluation_claims {
            let claims = batch_claims.into_host().await.unwrap();
            for value in claims.evaluations().as_slice() {
                batch_eval_claim += power * *value;
                power *= batching_challenge;
            }
        }

        (batch_mle, batch_codeword, batch_eval_claim)
    }

    async fn commit_phase_round(
        &self,
        current_mle: Mle<GC::EF, TaskScope>,
        current_codeword: Tensor<GC::F, TaskScope>,
        challenger: &mut GC::Challenger,
    ) -> Result<
        (
            GC::EF,
            Mle<GC::EF, TaskScope>,
            Tensor<GC::F, TaskScope>,
            GC::Digest,
            Tensor<GC::F, TaskScope>,
            MerkleTreeProverData<GC::Digest>,
        ),
        SingleLayerMerkleTreeProverError,
    > {
        // Perform a single round of the FRI commit phase, returning the commitment, folded
        // codeword, and folding parameter.
        // On CPU, the current codeword is in row-major form, which means that in order to put
        // even and odd entries together all we need to do is rehsape it to multiply the number of
        // columns by 2 and divide the number of rows by 2.
        let codeword_size = current_codeword.sizes()[1];
        let batch_size = current_codeword.sizes()[0];
        let scope = current_codeword.backend().clone();

        let mut leaves = Tensor::with_sizes_in([batch_size * 2, codeword_size / 2], scope.clone());
        let output_codeword_size = codeword_size / 2;
        let block_dim = 256;
        let grid_dim = output_codeword_size.div_ceil(block_dim);
        unsafe {
            let args = args!(current_codeword.as_ptr(), leaves.as_mut_ptr(), output_codeword_size);
            leaves.assume_init();
            scope
                .launch_kernel(
                    TaskScope::transpose_even_odd_kernel(),
                    grid_dim,
                    block_dim,
                    &args,
                    0,
                )
                .unwrap();
        }

        let (commit, prover_data) = self.tcs_prover.commit_tensors(&leaves).await?;
        // Observe the commitment.
        challenger.observe(commit);

        let beta: GC::EF = challenger.sample_ext_element();

        // Fold the mle.
        let folded_mle = current_mle.fold(beta).await;
        let folded_num_variables = folded_mle.num_variables();

        if folded_num_variables < 4 {
            let current_codeword_vec = current_codeword.transpose().into_host().await.unwrap();
            let current_codeword_vec =
                current_codeword_vec.into_buffer().into_extension::<GC::EF>().into_vec();
            let folded_codeword_vec = host_fold_even_odd(current_codeword_vec, beta);
            let folded_codeword_storage =
                Buffer::from(folded_codeword_vec).flatten_to_base::<GC::F>();
            let mut new_size = current_codeword.sizes().to_vec();
            new_size[1] /= 2;
            let folded_codeword =
                folded_codeword_storage.into_device_in(folded_mle.backend()).await.unwrap();
            let folded_codeword =
                Tensor::from(folded_codeword).reshape([new_size[1], new_size[0]]).transpose();
            return Ok((beta, folded_mle, folded_codeword, commit, leaves, prover_data));
        }

        let folded_height = 1 << folded_num_variables;
        let mut folded_mle_flattened = Tensor::<GC::F, TaskScope>::with_sizes_in(
            [<GC::EF as AbstractExtensionField<GC::F>>::D, folded_height],
            scope.clone(),
        );

        let mut folded_codeword = Tensor::<GC::F, TaskScope>::zeros_in(
            [<GC::EF as AbstractExtensionField<GC::F>>::D, folded_height << self.config.log_blowup],
            scope.clone(),
        );

        let block_dim = 256;
        let grid_dim = folded_height.div_ceil(block_dim);
        unsafe {
            let args =
                args!(folded_mle.guts().as_ptr(), folded_mle_flattened.as_mut_ptr(), folded_height);
            folded_mle_flattened.assume_init();
            scope
                .launch_kernel(TaskScope::flatten_to_base_kernel(), grid_dim, block_dim, &args, 0)
                .unwrap();
        }
        let encoder = SpparkDftKoalaBear::default();
        encode_batch(
            encoder,
            self.config.log_blowup as u32,
            folded_mle_flattened.as_view(),
            &mut folded_codeword,
        )
        .unwrap();

        Ok((beta, folded_mle, folded_codeword, commit, leaves, prover_data))
    }

    async fn final_poly(&self, final_codeword: Tensor<GC::F, TaskScope>) -> GC::EF {
        let final_codeword_host = final_codeword.into_host().await.unwrap();
        let final_codeword_transposed = final_codeword_host.transpose();
        GC::EF::from_base_slice(
            &final_codeword_transposed.storage.as_slice()
                [0..(<GC::EF as AbstractExtensionField<GC::F>>::D)],
        )
    }

    #[inline]
    pub async fn prove_trusted_evaluations_basefold(
        &self,
        mut eval_point: Point<GC::EF>,
        evaluation_claims: Rounds<Evaluations<GC::EF, TaskScope>>,
        mles: &JaggedTraceMle<GC::F, TaskScope>,
        prover_data: Rounds<&ProverCleanStackedPcsProverData<GC>>,
        challenger: &mut GC::Challenger,
    ) -> Result<
        BasefoldProof<GC>,
        BasefoldProverError<
            SingleLayerMerkleTreeProverError,
            Infallible,
            SingleLayerMerkleTreeProverError,
        >,
    >
    where
        GC::Challenger: DeviceGrindingChallenger<Witness = GC::F>,
    {
        let encoded_messages = prover_data
            .iter()
            .map(|data| data.codeword_mle.clone())
            .collect::<Message<Tensor<_, _>>>();

        let evaluation_claims = evaluation_claims.into_iter().flatten().collect::<Vec<_>>();

        // Sample a batching challenge and batch the mles and codewords.
        let batching_challenge: GC::EF = challenger.sample_ext_element();
        // Batch the mles and codewords.
        let (mle_batch, codeword_batch, batched_eval_claim) =
            self.batch(batching_challenge, mles.dense(), encoded_messages, evaluation_claims).await;
        // From this point on, run the BaseFold protocol on the random linear combination codeword,
        // the random linear combination multilinear, and the random linear combination of the
        // evaluation claims.
        let mut current_mle = mle_batch;
        let mut current_codeword = codeword_batch;
        // Initialize the vecs that go into a BaseFoldProof.
        let log_len = current_mle.num_variables();
        let mut univariate_messages: Vec<[GC::EF; 2]> = vec![];
        let mut fri_commitments = vec![];
        let mut commit_phase_data = vec![];
        let mut current_batched_eval_claim = batched_eval_claim;
        let mut commit_phase_values = vec![];

        assert_eq!(
            current_mle.num_variables(),
            eval_point.dimension() as u32,
            "eval point dimension mismatch"
        );
        for _ in 0..eval_point.dimension() {
            // Compute claims for `g(X_0, X_1, ..., X_{d-1}, 0)` and `g(X_0, X_1, ..., X_{d-1}, 1)`.
            let last_coord = eval_point.remove_last_coordinate();
            let zero_values = current_mle.fixed_at_zero(&eval_point).await;
            let zero_val = zero_values[0];
            let one_val = (current_batched_eval_claim - zero_val) / last_coord + zero_val;
            let uni_poly = [zero_val, one_val];
            univariate_messages.push(uni_poly);

            uni_poly.iter().for_each(|elem| challenger.observe_ext_element(*elem));

            // Perform a single round of the FRI commit phase, returning the commitment, folded
            // codeword, and folding parameter.
            let (beta, folded_mle, folded_codeword, commitment, leaves, prover_data) = self
                .commit_phase_round(current_mle, current_codeword, challenger)
                .await
                .map_err(BasefoldProverError::CommitPhaseError)?;

            fri_commitments.push(commitment);
            commit_phase_data.push(prover_data);
            commit_phase_values.push(leaves);

            current_mle = folded_mle;
            current_codeword = folded_codeword;
            current_batched_eval_claim = zero_val + beta * one_val;
        }

        let final_poly = self.final_poly(current_codeword).await;
        challenger.observe_ext_element(final_poly);

        let fri_config = self.config;
        let pow_bits = fri_config.proof_of_work_bits;
        let pow_witness = GrindingPowCudaProver::grind(challenger, pow_bits).await;
        // FRI Query Phase.
        let query_indices: Vec<usize> = (0..fri_config.num_queries)
            .map(|_| challenger.sample_bits(log_len as usize + fri_config.log_blowup()))
            .collect();

        // Open the original polynomials at the query indices.
        let mut component_polynomials_query_openings_and_proofs = vec![];
        for prover_data in prover_data {
            let ProverCleanStackedPcsProverData { merkle_tree_tcs_data, codeword_mle, .. } =
                prover_data;
            let values = self
                .tcs_prover
                .compute_openings_at_indices(codeword_mle.as_ref(), &query_indices)
                .await;
            let proof = self
                .tcs_prover
                .prove_openings_at_indices(merkle_tree_tcs_data, &query_indices)
                .await
                .map_err(BasefoldProverError::TcsCommitError)?;
            let opening = MerkleTreeOpeningAndProof::<GC> { values, proof };
            component_polynomials_query_openings_and_proofs.push(opening);
        }

        // Provide openings for the FRI query phase.
        let mut query_phase_openings_and_proofs = vec![];
        let mut indices = query_indices;
        for (leaves, data) in commit_phase_values.into_iter().zip_eq(commit_phase_data) {
            for index in indices.iter_mut() {
                *index >>= 1;
            }
            let values = self.tcs_prover.compute_openings_at_indices(&leaves, &indices).await;

            let proof = self
                .tcs_prover
                .prove_openings_at_indices(&data, &indices)
                .await
                .map_err(BasefoldProverError::TcsCommitError)?;
            let opening = MerkleTreeOpeningAndProof { values, proof };
            query_phase_openings_and_proofs.push(opening);
        }

        Ok(BasefoldProof {
            univariate_messages,
            fri_commitments,
            component_polynomials_query_openings_and_proofs,
            query_phase_openings_and_proofs,
            final_poly,
            pow_witness,
        })
    }
}

unsafe impl MleBatchKernel<KoalaBear, BinomialExtensionField<KoalaBear, 4>> for TaskScope {
    fn batch_mle_kernel() -> KernelPtr {
        unsafe { batch_koala_bear_base_ext_kernel() }
    }
}

unsafe impl RsCodeWordBatchKernel<KoalaBear, BinomialExtensionField<KoalaBear, 4>> for TaskScope {
    fn batch_rs_codeword_kernel() -> KernelPtr {
        unsafe { batch_koala_bear_base_ext_kernel_flattened() }
    }
}

unsafe impl RsCodeWordTransposeKernel<KoalaBear, BinomialExtensionField<KoalaBear, 4>>
    for TaskScope
{
    fn transpose_even_odd_kernel() -> KernelPtr {
        unsafe { transpose_even_odd_koala_bear_base_ext_kernel() }
    }
}

unsafe impl MleFlattenKernel<KoalaBear, BinomialExtensionField<KoalaBear, 4>> for TaskScope {
    fn flatten_to_base_kernel() -> KernelPtr {
        unsafe { flatten_to_base_koala_bear_base_ext_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;
    use std::sync::Arc;

    use csl_basefold::Poseidon2KoalaBear16BasefoldCudaProverComponents;
    use csl_cuda::{run_in_place, ToDevice};
    use csl_tracegen::CudaTraceGenerator;
    use cslpc_merkle_tree::Poseidon2KoalaBear16CudaProver;
    use slop_alloc::ToHost;
    use slop_basefold::BasefoldVerifier;
    use slop_basefold_prover::BasefoldProver;
    use slop_commit::Message;
    use slop_koala_bear::KoalaBearDegree4Duplex;
    use slop_multilinear::{Mle, MultilinearPcsBatchVerifier};
    use slop_stacked::{FixedRateInterleave, InterleaveMultilinears};
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};

    use cslpc_tracegen::test_utils::tracegen_setup::{
        self, CORE_MAX_LOG_ROW_COUNT, LOG_STACKING_HEIGHT,
    };
    use cslpc_tracegen::{full_tracegen, CORE_MAX_TRACE_SIZE};
    use cslpc_utils::{Felt, TestGC};
    use futures::{stream, StreamExt};

    use super::*;

    #[tokio::test]
    async fn test_prover_clean_basefold() {
        let (machine, record, program) = tracegen_setup::setup().await;

        run_in_place(|scope| async move {
            let verifier = BasefoldVerifier::<KoalaBearDegree4Duplex>::new(1, 2);
            let old_prover = BasefoldProver::<
                KoalaBearDegree4Duplex,
                Poseidon2KoalaBear16BasefoldCudaProverComponents,
            >::new(&verifier);

            let new_prover = ProverCleanFriCudaProver::<TestGC, _, Felt> {
                tcs_prover: Poseidon2KoalaBear16CudaProver::default(),
                config: verifier.fri_config,
                log_height: LOG_STACKING_HEIGHT,
                _marker: PhantomData::<TestGC>,
            };

            // Generate traces using the host tracegen.
            let semaphore = ProverSemaphore::new(1);
            let trace_generator = CudaTraceGenerator::new_in(machine.clone(), scope.clone());
            let old_traces = trace_generator
                .generate_traces(
                    program.clone(),
                    record.clone(),
                    CORE_MAX_LOG_ROW_COUNT as usize,
                    semaphore.clone(),
                )
                .await;

            let preprocessed_traces = old_traces.preprocessed_traces.clone();

            let message = preprocessed_traces
                .into_iter()
                .filter_map(|mle| mle.1.into_inner())
                .map(|x| Clone::clone(x.as_ref()))
                .collect::<Message<Mle<_, _>>>();

            let interleaver = FixedRateInterleave::<Felt, TaskScope>::new(32);

            let interleaved_message =
                interleaver.interleave_multilinears(message, LOG_STACKING_HEIGHT).await;

            let interleaved_message =
                interleaved_message.into_iter().map(|x| x.as_ref().clone()).collect::<Message<_>>();

            let (old_preprocessed_commitment, old_preprocessed_prover_data) =
                old_prover.commit_mles(interleaved_message.clone()).await.unwrap();

            let new_semaphore = ProverSemaphore::new(1);
            let capacity = CORE_MAX_TRACE_SIZE as usize;
            let mut buffer: Vec<MaybeUninit<Felt>> = Vec::with_capacity(capacity);
            unsafe { buffer.set_len(capacity) };
            let boxed: Box<[MaybeUninit<Felt>]> = buffer.into_boxed_slice();
            let buffer = Box::into_pin(boxed);
            let (_, new_traces, _, _) = full_tracegen(
                &machine,
                program,
                Arc::new(record),
                buffer.as_ptr() as usize,
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT,
                &scope,
                new_semaphore,
            )
            .await;

            let dst = Tensor::<Felt, TaskScope>::with_sizes_in(
                [
                    new_traces.0.dense().preprocessed_offset >> LOG_STACKING_HEIGHT,
                    1 << (LOG_STACKING_HEIGHT as usize + verifier.fri_config.log_blowup()),
                ],
                scope.clone(),
            );

            let (new_preprocessed_commit, new_preprocessed_prover_data) =
                new_prover.encode_and_commit(true, &new_traces, dst).await.unwrap();

            assert_eq!(new_preprocessed_commit, old_preprocessed_commitment);

            let dst = Tensor::<Felt, TaskScope>::with_sizes_in(
                [
                    new_traces.0.dense().main_size() >> LOG_STACKING_HEIGHT,
                    1 << (LOG_STACKING_HEIGHT as usize + verifier.fri_config.log_blowup()),
                ],
                scope.clone(),
            );

            let (new_main_commit, new_main_prover_data) =
                new_prover.encode_and_commit(false, &new_traces, dst).await.unwrap();
            let message = old_traces
                .main_trace_data
                .traces
                .into_iter()
                .filter_map(|mle| mle.1.into_inner())
                .map(|x| Clone::clone(x.as_ref()))
                .collect::<Message<Mle<_, _>>>();

            let interleaved_message_2 =
                interleaver.interleave_multilinears(message, LOG_STACKING_HEIGHT).await;

            let (old_main_commitment, old_main_prover_data) =
                old_prover.commit_mles(interleaved_message_2.clone()).await.unwrap();

            assert_eq!(new_main_commit, old_main_commitment);

            let mut rng = rand::thread_rng();

            let eval_point_host = Point::<Ext>::rand(&mut rng, LOG_STACKING_HEIGHT);

            let eval_point = eval_point_host.to_device_in(&scope).await.unwrap();

            let evaluation_claims_1 = stream::iter(interleaved_message.clone())
                .then(async |mle| mle.eval_at(&eval_point).await)
                .collect::<Vec<_>>()
                .await;

            let evaluation_claims_1 = Evaluations { round_evaluations: evaluation_claims_1 };

            let evaluation_claims_2 = stream::iter(interleaved_message_2.clone())
                .then(async |mle| mle.eval_at(&eval_point).await)
                .collect::<Vec<_>>()
                .await;

            let host_evaluation_claims_1 = stream::iter(evaluation_claims_1.iter())
                .then(async |mle| mle.to_host().await.unwrap())
                .collect::<Vec<_>>()
                .await;

            let host_evaluation_claims_2 = stream::iter(evaluation_claims_2.iter())
                .then(async |mle| mle.to_host().await.unwrap())
                .collect::<Vec<_>>()
                .await;

            let flattened_evaluation_claims = vec![
                MleEval::new(
                    host_evaluation_claims_1.into_iter().flat_map(|x| x.to_vec()).collect(),
                ),
                MleEval::new(
                    host_evaluation_claims_2.into_iter().flat_map(|x| x.to_vec()).collect(),
                ),
            ];

            let evaluation_claims_2 = Evaluations { round_evaluations: evaluation_claims_2 };

            let mut challenger = KoalaBearDegree4Duplex::default_challenger();

            scope.synchronize().await.unwrap();
            let now = std::time::Instant::now();

            let basefold_proof = old_prover
                .prove_trusted_mle_evaluations(
                    eval_point_host.clone(),
                    vec![interleaved_message, interleaved_message_2].into_iter().collect(),
                    vec![evaluation_claims_1.clone(), evaluation_claims_2.clone()]
                        .into_iter()
                        .collect(),
                    vec![old_preprocessed_prover_data, old_main_prover_data].into_iter().collect(),
                    &mut challenger,
                )
                .await
                .unwrap();

            scope.synchronize().await.unwrap();
            println!("Old proof time: {:?}", now.elapsed());

            let mut challenger = KoalaBearDegree4Duplex::default_challenger();

            scope.synchronize().await.unwrap();

            let now = std::time::Instant::now();

            let new_basefold_proof = new_prover
                .prove_trusted_evaluations_basefold(
                    eval_point_host.clone(),
                    [evaluation_claims_1, evaluation_claims_2].into_iter().collect(),
                    &new_traces,
                    [&new_preprocessed_prover_data, &new_main_prover_data].into_iter().collect(),
                    &mut challenger,
                )
                .await
                .unwrap();

            scope.synchronize().await.unwrap();
            println!("New proof time: {:?}", now.elapsed());

            for (i, (a, b)) in basefold_proof
                .univariate_messages
                .iter()
                .zip_eq(new_basefold_proof.univariate_messages.iter())
                .enumerate()
            {
                assert_eq!(a, b, "Failure on message from round {}", i);
            }

            for (i, (a, b)) in basefold_proof
                .fri_commitments
                .iter()
                .zip_eq(new_basefold_proof.fri_commitments.iter())
                .enumerate()
            {
                assert_eq!(a, b, "Failure on FRI commitment from round {}", i);
            }

            assert_eq!(
                basefold_proof.final_poly, new_basefold_proof.final_poly,
                "Failure on final poly"
            );

            // Because the grinding is technically non-deterministic, the proof-of-work witnesses
            // do not need to be the same. Therefore, all the query indices are not necessarily the
            // same between the new and old proofs. However, the new proof should still verify.

            verifier
                .verify_trusted_evaluations(
                    &[new_preprocessed_commit, new_main_commit],
                    eval_point_host,
                    &flattened_evaluation_claims,
                    &new_basefold_proof,
                    &mut KoalaBearDegree4Duplex::default_challenger(),
                )
                .unwrap();
        })
        .await
        .await
        .unwrap();
    }
}

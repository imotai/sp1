use std::{marker::PhantomData, sync::Arc};

use csl_cuda::{
    args,
    sys::{
        basefold::{
            batch_baby_bear_base_ext_kernel, batch_baby_bear_base_ext_kernel_flattened,
            flatten_to_base_baby_bear_base_ext_kernel,
            transpose_even_odd_baby_bear_base_ext_kernel,
        },
        runtime::KernelPtr,
    },
    IntoDevice, TaskScope,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, TwoAdicField};
use slop_alloc::{Buffer, HasBackend, IntoHost};
use slop_baby_bear::BabyBear;
use slop_basefold::RsCodeWord;
use slop_basefold_prover::{
    host_fold_even_odd, BasefoldBatcher, FriIoppProver, ReedSolomonEncoder,
};
use slop_challenger::{CanObserve, FieldChallenger};
use slop_commit::{Message, TensorCs, TensorCsProver};
use slop_futures::OwnedBorrow;
use slop_multilinear::{Mle, MleEval, MleFoldBackend};
use slop_tensor::{Tensor, TransposeBackend};

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
pub struct FriCudaProver<E, P>(pub PhantomData<(E, P)>);

impl<F, EF, E, P> BasefoldBatcher<F, EF, E, TaskScope> for FriCudaProver<E, P>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    E: ReedSolomonEncoder<F, TaskScope> + Clone,
    P: TensorCsProver<TaskScope>,
    TaskScope: MleBatchKernel<F, EF> + RsCodeWordBatchKernel<F, EF> + TransposeBackend<EF>,
{
    async fn batch<M, Code>(
        &self,
        batching_challenge: EF,
        mles: Message<M>,
        codewords: Message<Code>,
        evaluation_claims: Vec<MleEval<EF, TaskScope>>,
        _encoder: &E,
    ) -> (Mle<EF, TaskScope>, RsCodeWord<F, TaskScope>, EF)
    where
        M: OwnedBorrow<Mle<F, TaskScope>>,
        Code: OwnedBorrow<RsCodeWord<F, TaskScope>>,
    {
        // Compute all the batch challenge powers.
        let total_num_polynomials = mles
            .iter()
            .map(|mle| {
                let mle: &Mle<F, TaskScope> = (**mle).borrow();
                mle.num_polynomials()
            })
            .sum::<usize>();
        let mut batch_challenge_powers =
            batching_challenge.powers().take(total_num_polynomials).collect::<Vec<_>>();

        // Compute the random linear combination of the MLEs of the columns of the matrices
        let num_variables = (mles.first().unwrap()).borrow().num_variables() as usize;
        let codeword_size = (codewords.first().unwrap()).borrow().data.sizes()[1];
        let scope: TaskScope = (mles.first().unwrap()).borrow().backend().clone();
        let mut batch_mle =
            Mle::new(Tensor::<EF, TaskScope>::zeros_in([1, 1 << num_variables], scope.clone()));
        let mut batch_codeword =
            Tensor::<F, TaskScope>::zeros_in([EF::D, codeword_size], scope.clone());
        for (mle, codeword) in mles.iter().zip_eq(codewords.iter()) {
            let mle: &Mle<F, TaskScope> = (**mle).borrow();
            let batch_size = mle.num_polynomials();
            let mut powers = batch_challenge_powers;
            batch_challenge_powers = powers.split_off(batch_size);
            let powers_device = Buffer::from(powers.clone()).into_device_in(&scope).await.unwrap();

            unsafe {
                let block_dim = 256;
                let grid_dim = (1usize << num_variables).div_ceil(block_dim);
                let mle_args = args!(
                    mle.guts().as_ptr(),
                    batch_mle.guts_mut().as_mut_ptr(),
                    powers_device.as_ptr(),
                    (1 << num_variables) as usize,
                    batch_size
                );
                scope
                    .launch_kernel(TaskScope::batch_mle_kernel(), grid_dim, block_dim, &mle_args, 0)
                    .unwrap();
            }
            let codeword: &RsCodeWord<F, TaskScope> = (**codeword).borrow();
            let block_dim = 256;
            let grid_dim = codeword_size.div_ceil(block_dim);
            let codeword_args = args!(
                codeword.data.as_ptr(),
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
        let mut batch_eval_claim = EF::zero();
        let mut power = EF::one();
        for batch_claims in evaluation_claims {
            let claims = batch_claims.into_host().await.unwrap();
            for value in claims.evaluations().as_slice() {
                batch_eval_claim += power * *value;
                power *= batching_challenge;
            }
        }

        let batch_codeword = RsCodeWord::new(batch_codeword);
        (batch_mle, batch_codeword, batch_eval_claim)
    }
}

impl<F, EF, Tcs, Challenger, E, P> FriIoppProver<F, EF, Tcs, Challenger, E, TaskScope>
    for FriCudaProver<E, P>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Tcs: TensorCs<Data = F>,
    Challenger: FieldChallenger<F> + CanObserve<Tcs::Commitment> + 'static + Send + Sync,
    E: ReedSolomonEncoder<F, TaskScope> + Clone,
    P: TensorCsProver<TaskScope, Cs = Tcs>,
    TaskScope: MleBatchKernel<F, EF>
        + RsCodeWordBatchKernel<F, EF>
        + MleFoldBackend<EF>
        + TransposeBackend<F>
        + TransposeBackend<EF>
        + RsCodeWordTransposeKernel<F, EF>
        + MleFlattenKernel<F, EF>,
{
    type FriProverError = P::ProverError;
    type TcsProver = P;
    type Encoder = E;

    async fn commit_phase_round(
        &self,
        current_mle: Mle<EF, TaskScope>,
        current_codeword: RsCodeWord<F, TaskScope>,
        encoder: &Self::Encoder,
        tcs_prover: &Self::TcsProver,
        challenger: &mut Challenger,
    ) -> Result<
        (
            EF,
            Mle<EF, TaskScope>,
            RsCodeWord<F, TaskScope>,
            <Tcs as TensorCs>::Commitment,
            Arc<slop_tensor::Tensor<F, TaskScope>>,
            <Self::TcsProver as TensorCsProver<TaskScope>>::ProverData,
        ),
        Self::FriProverError,
    > {
        // Perform a single round of the FRI commit phase, returning the commitment, folded
        // codeword, and folding parameter.
        // On CPU, the current codeword is in row-major form, which means that in order to put
        // even and odd entries together all we need to do is rehsape it to multiply the number of
        // columns by 2 and divide the number of rows by 2.
        let codeword_size = current_codeword.data.sizes()[1];
        let batch_size = current_codeword.data.sizes()[0];
        let scope = current_codeword.backend().clone();

        let mut leaves = Tensor::with_sizes_in([batch_size * 2, codeword_size / 2], scope.clone());
        let output_codeword_size = codeword_size / 2;
        let block_dim = 256;
        let grid_dim = output_codeword_size.div_ceil(block_dim);
        unsafe {
            let args =
                args!(current_codeword.data.as_ptr(), leaves.as_mut_ptr(), output_codeword_size);
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

        let leaves = Message::<Tensor<F, TaskScope>>::from(vec![leaves]);
        let (commit, prover_data) = tcs_prover.commit_tensors(leaves.clone()).await?;
        // Observe the commitment.
        challenger.observe(commit.clone());

        let beta: EF = challenger.sample_ext_element();

        // Fold the mle.
        let folded_mle = current_mle.fold(beta).await;
        let folded_num_variables = folded_mle.num_variables();

        if folded_num_variables < 4 {
            let current_codeword_vec = current_codeword.data.transpose().into_host().await.unwrap();
            let current_codeword_vec =
                current_codeword_vec.into_buffer().into_extension::<EF>().into_vec();
            let folded_codeword_vec = host_fold_even_odd(current_codeword_vec, beta);
            let folded_codeword_storage = Buffer::from(folded_codeword_vec).flatten_to_base::<F>();
            let mut new_size = current_codeword.data.sizes().to_vec();
            new_size[1] /= 2;
            let folded_codeword =
                folded_codeword_storage.into_device_in(folded_mle.backend()).await.unwrap();
            let folded_codeword =
                Tensor::from(folded_codeword).reshape([new_size[1], new_size[0]]).transpose();
            let folded_codeword = RsCodeWord::new(folded_codeword);
            return Ok((beta, folded_mle, folded_codeword, commit, leaves[0].clone(), prover_data));
        }

        let folded_height = 1 << folded_num_variables;
        let mut folded_mle_flattened =
            Tensor::<F, TaskScope>::with_sizes_in([EF::D, folded_height], scope.clone());
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
        let folded_mle_flattened =
            Message::<Mle<F, TaskScope>>::from(vec![Mle::new(folded_mle_flattened)]);
        let folded_codeword = encoder.encode_batch(folded_mle_flattened).await.unwrap();

        let folded_codeword = RsCodeWord::clone(&folded_codeword[0]);
        Ok((beta, folded_mle, folded_codeword, commit, leaves[0].clone(), prover_data))
    }

    async fn final_poly(&self, final_codeword: RsCodeWord<F, TaskScope>) -> EF {
        let final_codeword_host = final_codeword.data.into_host().await.unwrap();
        let final_codeword_transposed = final_codeword_host.transpose();
        EF::from_base_slice(&final_codeword_transposed.storage.as_slice()[0..EF::D])
    }
}

unsafe impl MleBatchKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn batch_mle_kernel() -> KernelPtr {
        unsafe { batch_baby_bear_base_ext_kernel() }
    }
}

unsafe impl RsCodeWordBatchKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn batch_rs_codeword_kernel() -> KernelPtr {
        unsafe { batch_baby_bear_base_ext_kernel_flattened() }
    }
}

unsafe impl RsCodeWordTransposeKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn transpose_even_odd_kernel() -> KernelPtr {
        unsafe { transpose_even_odd_baby_bear_base_ext_kernel() }
    }
}

unsafe impl MleFlattenKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn flatten_to_base_kernel() -> KernelPtr {
        unsafe { flatten_to_base_baby_bear_base_ext_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use futures::{future::join_all, prelude::*};
    use rand::Rng;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_basefold_prover::{BasefoldProver, Poseidon2BabyBear16BasefoldCpuProverComponents};
    use slop_multilinear::Point;

    use crate::Poseidon2BabyBear16BasefoldCudaProverComponents;

    use super::*;

    #[tokio::test]
    async fn test_basefold_bacther() {
        let mut rng = rand::thread_rng();

        let num_variables = 6;
        let widths = [14];
        let log_blowup = 1;
        let codeword_size = 1 << (num_variables + log_blowup as u32);
        let point = Point::<BinomialExtensionField<BabyBear, 4>>::rand(&mut rng, num_variables);

        type C = Poseidon2BabyBear16BasefoldConfig;
        type CudaProver = BasefoldProver<Poseidon2BabyBear16BasefoldCudaProverComponents>;
        type HostProver = BasefoldProver<Poseidon2BabyBear16BasefoldCpuProverComponents>;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let verifier = BasefoldVerifier::<C>::new(log_blowup);
        let cuda_prover = CudaProver::new(&verifier);
        let host_prover = HostProver::new(&verifier);

        let host_mles = widths
            .iter()
            .map(|&num_polynomials| Mle::<BabyBear>::rand(&mut rng, num_polynomials, num_variables))
            .collect::<Vec<_>>();

        let codeward_futures = host_mles.iter().map(|mle| async {
            host_prover
                .encoder
                .encode_batch(Message::<Mle<BabyBear>>::from(vec![mle.clone()]))
                .await
                .unwrap()[0]
                .clone()
        });

        let host_codewords = join_all(codeward_futures)
            .await
            .into_iter()
            .map(|codeword| Arc::into_inner(codeword).unwrap())
            .collect::<Vec<_>>();

        let host_eval_claims = stream::iter(host_mles.iter())
            .then(|mle| mle.eval_at(&point))
            .collect::<Vec<_>>()
            .await;

        let batching_challenge = rng.gen::<EF>();
        // Batch the mles and codewords on the host.
        let host_mles_message = host_mles.iter().cloned().collect::<Message<_>>();
        let host_codewords_message = host_codewords.iter().cloned().collect::<Message<_>>();
        let (host_batch_mle, host_batch_codeword, host_eval_claim) = host_prover
            .fri_prover
            .batch(
                batching_challenge,
                host_mles_message,
                host_codewords_message,
                host_eval_claims.clone(),
                &host_prover.encoder,
            )
            .await;

        let (batch_mle, batch_codeword, eval_claim) = csl_cuda::task()
            .await
            .unwrap()
            .run(|t| async move {
                let mut mles = vec![];
                let mut codewords = vec![];
                let mut eval_claims = vec![];
                for mle in host_mles {
                    mles.push(t.into_device(mle).await.unwrap());
                }
                for codeword in host_codewords {
                    let codeword = t.into_device(codeword).await.unwrap();
                    assert_eq!(codeword.data.sizes()[1], codeword_size);
                    codewords.push(codeword);
                }
                for eval_claim in host_eval_claims {
                    eval_claims.push(t.into_device(eval_claim).await.unwrap());
                }
                let mles = Message::<Mle<BabyBear, TaskScope>>::from(mles);
                let codewords = Message::<RsCodeWord<BabyBear, TaskScope>>::from(codewords);
                let (batch_mle, batch_codeword, eval_claim) = cuda_prover
                    .fri_prover
                    .batch(batching_challenge, mles, codewords, eval_claims, &cuda_prover.encoder)
                    .await;
                let batch_mle = batch_mle.into_host().await.unwrap();
                let batch_codeword = batch_codeword.into_host().await.unwrap();
                (batch_mle, batch_codeword, eval_claim)
            })
            .await
            .await
            .unwrap();

        // Compare the results of the host and cuda provers.
        for (exp, val) in
            host_batch_mle.guts().as_slice().iter().zip_eq(batch_mle.guts().as_slice().iter())
        {
            assert_eq!(exp, val);
        }
        for (exp, val) in
            host_batch_codeword.data.as_slice().iter().zip_eq(batch_codeword.data.as_slice().iter())
        {
            assert_eq!(exp, val);
        }

        assert_eq!(host_eval_claim, eval_claim);
    }

    #[tokio::test]
    async fn test_commit_phase_round() {
        let mut rng = rand::thread_rng();

        let log_blowup = 1;

        for num_variables in 1..16 {
            type EF = BinomialExtensionField<BabyBear, 4>;

            let initial_mle = Mle::<EF>::rand(&mut rng, 1, num_variables);
            type C = Poseidon2BabyBear16BasefoldConfig;
            type CudaProver = BasefoldProver<Poseidon2BabyBear16BasefoldCudaProverComponents>;
            type HostProver = BasefoldProver<Poseidon2BabyBear16BasefoldCpuProverComponents>;

            let verifier = BasefoldVerifier::<C>::new(log_blowup);
            let cuda_prover = CudaProver::new(&verifier);
            let host_prover = HostProver::new(&verifier);

            let initial_mle_flattened =
                initial_mle.guts().clone().into_buffer().flatten_to_base::<BabyBear>();
            let initial_mle_flat_tensor =
                Tensor::from(initial_mle_flattened).reshape([1 << num_variables, 4]);
            let initial_mle_flat_tensor =
                Message::<Mle<BabyBear>>::from(vec![Mle::new(initial_mle_flat_tensor)]);
            let initial_codeword =
                host_prover.encoder.encode_batch(initial_mle_flat_tensor).await.unwrap();

            let initial_codeword = RsCodeWord::clone(&initial_codeword[0]);

            // Batch the mles and codewords on the host.
            let mut challenger = verifier.challenger();
            let (host_beta, host_folded_mle, host_folded_codeword, host_commit, _, _) = host_prover
                .fri_prover
                .commit_phase_round(
                    initial_mle.clone(),
                    initial_codeword.clone(),
                    &host_prover.encoder,
                    &host_prover.tcs_prover,
                    &mut challenger,
                )
                .await
                .unwrap();

            let mut challenger = verifier.challenger();
            let (beta, folded_mle, folded_codeword, commit, _, _) = csl_cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let initial_mle = t.into_device(initial_mle).await.unwrap();
                    let initial_codeword = t.into_device(initial_codeword.clone()).await.unwrap();
                    let (beta, folded_mle, folded_codeword, commit, leaves, prover_data) =
                        cuda_prover
                            .fri_prover
                            .commit_phase_round(
                                initial_mle,
                                initial_codeword,
                                &cuda_prover.encoder,
                                &cuda_prover.tcs_prover,
                                &mut challenger,
                            )
                            .await
                            .unwrap();
                    let folded_mle_to_host = folded_mle.into_host().await.unwrap();
                    let folded_codeword_to_host = folded_codeword.into_host().await.unwrap();
                    (beta, folded_mle_to_host, folded_codeword_to_host, commit, leaves, prover_data)
                })
                .await
                .await
                .unwrap();

            // Compare the results of the host and cuda provers.
            assert_eq!(beta, host_beta);
            for (exp, val) in
                host_folded_mle.guts().as_slice().iter().zip_eq(folded_mle.guts().as_slice().iter())
            {
                assert_eq!(exp, val);
            }

            assert_eq!(host_commit, commit);

            assert_eq!(host_folded_codeword.data.sizes(), folded_codeword.data.sizes());
            for (exp, val) in host_folded_codeword
                .data
                .as_slice()
                .iter()
                .zip_eq(folded_codeword.data.as_slice().iter())
            {
                assert_eq!(exp, val);
            }
        }
    }
}

use slop_basefold::Poseidon2Bn254FrBasefoldConfig;
use slop_bn254::Bn254Fr;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use derive_where::derive_where;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field, TwoAdicField};
use slop_alloc::{Backend, CpuBackend};
use slop_baby_bear::BabyBear;
use slop_basefold::BasefoldVerifier;
use slop_basefold::{BasefoldConfig, BasefoldProof, Poseidon2BabyBear16BasefoldConfig, RsCodeWord};
use slop_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use slop_commit::{ComputeTcsOpenings, Message, Rounds, TensorCs, TensorCsOpening, TensorCsProver};
use slop_dft::p3::Radix2DitParallel;
use slop_futures::OwnedBorrow;
use slop_merkle_tree::{
    FieldMerkleTreeProver, MerkleTreeTcs, Poseidon2BabyBear16Prover, Poseidon2BabyBearConfig,
    Poseidon2Bn254Config, OUTER_DIGEST_SIZE,
};
use slop_multilinear::{
    Evaluations, Mle, MleBaseBackend, MleEvaluationBackend, MleFixedAtZeroBackend,
    MultilinearPcsProver, Point,
};
use slop_tensor::Tensor;
use thiserror::Error;

use crate::{
    BasefoldBatcher, CpuDftEncoder, FriCpuProver, FriIoppProver, GrindingPowProver, PowProver,
    ReedSolomonEncoder,
};

/// The components required for a Basefold prover.
pub trait BasefoldProverComponents: Clone + Send + Sync + 'static + Debug {
    type F: TwoAdicField;
    type EF: ExtensionField<Self::F>;
    type A: Backend
        + MleBaseBackend<Self::EF>
        + MleFixedAtZeroBackend<Self::EF, Self::EF>
        + MleEvaluationBackend<Self::F, Self::EF>;
    type Tcs: TensorCs<Data = Self::F>;

    type Challenger: FieldChallenger<Self::F>
        + GrindingChallenger
        + CanObserve<<Self::Tcs as TensorCs>::Commitment>
        + Send
        + Sync
        + 'static;

    /// The Basefold configuration for which we can create proof for.
    type Config: BasefoldConfig<
        F = Self::F,
        EF = Self::EF,
        Tcs = Self::Tcs,
        Challenger = Self::Challenger,
        Commitment = <<Self::Config as BasefoldConfig>::Tcs as TensorCs>::Commitment,
    >;

    /// The encoder for encoding the Mle guts into codewords.
    type Encoder: ReedSolomonEncoder<Self::F, Self::A> + Clone + Debug + Send + Sync + 'static;
    /// The prover for the FRI proximity test.
    type FriProver: FriIoppProver<
            Self::F,
            Self::EF,
            Self::Tcs,
            Self::Challenger,
            Self::Encoder,
            Self::A,
            Encoder = Self::Encoder,
            TcsProver = Self::TcsProver,
        > + Send
        + Debug
        + Sync
        + 'static;
    /// The TCS prover for committing to the encoded messages.
    type TcsProver: TensorCsProver<Self::A, Cs = Self::Tcs>
        + ComputeTcsOpenings<Self::A, Cs = Self::Tcs>
        + Debug
        + 'static
        + Send
        + Sync;
    /// The prover for the proof-of-work grinding phase.
    type PowProver: PowProver<Self::Challenger> + Debug + Send + Sync + 'static;
}

pub trait DefaultBasefoldProver: BasefoldProverComponents + Sized {
    fn default_prover(verifier: &BasefoldVerifier<Self::Config>) -> BasefoldProver<Self>;
}

#[derive(Serialize, Deserialize)]
#[derive_where(Debug, Clone; <C::TcsProver as TensorCsProver<C::A>>::ProverData: Debug + Clone)]
#[serde(bound(
    serialize = "<C::TcsProver as TensorCsProver<C::A>>::ProverData: Serialize, RsCodeWord<C::F, C::A>: Serialize",
    deserialize = "<C::TcsProver as TensorCsProver<C::A>>::ProverData: Deserialize<'de>, RsCodeWord<C::F, C::A>: Deserialize<'de>"
))]
pub struct BasefoldProverData<C: BasefoldProverComponents> {
    pub tcs_prover_data: <C::TcsProver as TensorCsProver<C::A>>::ProverData,
    pub encoded_messages: Message<RsCodeWord<C::F, C::A>>,
}

#[derive(Error)]
pub enum BasefoldProverError<C: BasefoldProverComponents> {
    #[error("Commit error: {0}")]
    TcsCommitError(<C::TcsProver as TensorCsProver<C::A>>::ProverError),
    #[error("Encoder error: {0}")]
    EncoderError(<C::Encoder as ReedSolomonEncoder<C::F, C::A>>::Error),
    #[error("Commit phase error: {0}")]
    #[allow(clippy::type_complexity)]
    CommitPhaseError(
        <C::FriProver as FriIoppProver<C::F, C::EF, C::Tcs, C::Challenger, C::Encoder, C::A>>::FriProverError,
    ),
}

impl<C: BasefoldProverComponents> std::fmt::Debug for BasefoldProverError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BasefoldProverError::TcsCommitError(e) => write!(f, "Tcs commit error: {e}"),
            BasefoldProverError::EncoderError(e) => write!(f, "Encoder error: {e}"),
            BasefoldProverError::CommitPhaseError(e) => write!(f, "Commit phase error: {e}"),
        }
    }
}

/// A prover for the BaseFold protocol.
///
/// The [BasefoldProver] struct implements the interactive parts of the Basefold PCS while
/// abstracting some of the key parts.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasefoldProver<C: BasefoldProverComponents> {
    pub encoder: C::Encoder,
    pub fri_prover: C::FriProver,
    pub tcs_prover: C::TcsProver,
    pub pow_prover: C::PowProver,
}

impl<C: BasefoldProverComponents> MultilinearPcsProver for BasefoldProver<C> {
    type F = C::F;
    type EF = C::EF;
    type Commitment = <C::Config as BasefoldConfig>::Commitment;
    type Proof = BasefoldProof<C::Config>;
    type Challenger = C::Challenger;
    type Verifier = BasefoldVerifier<C::Config>;
    type ProverData = BasefoldProverData<C>;
    type A = C::A;
    type ProverError = BasefoldProverError<C>;

    async fn commit_multilinears(
        &self,
        mles: Message<Mle<Self::F, Self::A>>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        self.commit_mles(mles).await
    }

    async fn prove_trusted_evaluations(
        &self,
        eval_point: Point<Self::EF>,
        mle_rounds: Rounds<Message<Mle<Self::F, Self::A>>>,
        evaluation_claims: Rounds<Evaluations<Self::EF, Self::A>>,
        prover_data: Rounds<Self::ProverData>,
        challenger: &mut Self::Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        self.prove_trusted_mle_evaluations(
            eval_point,
            mle_rounds,
            evaluation_claims,
            prover_data,
            challenger,
        )
        .await
    }
}

impl<C: BasefoldProverComponents> BasefoldProver<C> {
    #[inline]
    pub const fn from_parts(
        encoder: C::Encoder,
        fri_prover: C::FriProver,
        tcs_prover: C::TcsProver,
        pow_prover: C::PowProver,
    ) -> Self {
        Self { encoder, fri_prover, tcs_prover, pow_prover }
    }

    #[inline]
    pub fn new(verifier: &BasefoldVerifier<C::Config>) -> Self
    where
        C: DefaultBasefoldProver,
    {
        C::default_prover(verifier)
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    pub async fn commit_mles<M>(
        &self,
        mles: Message<M>,
    ) -> Result<
        (<C::Config as BasefoldConfig>::Commitment, BasefoldProverData<C>),
        BasefoldProverError<C>,
    >
    where
        M: OwnedBorrow<Mle<C::F, C::A>>,
    {
        // Encode the guts of the mle via Reed-Solomon encoding.

        let encoded_messages = self.encoder.encode_batch(mles.clone()).await.unwrap();

        // Commit to the encoded messages.
        let (commitment, tcs_prover_data) = self
            .tcs_prover
            .commit_tensors(encoded_messages.clone())
            .await
            .map_err(BasefoldProverError::<C>::TcsCommitError)?;

        Ok((commitment, BasefoldProverData { encoded_messages, tcs_prover_data }))
    }

    #[inline]
    pub async fn prove_trusted_mle_evaluations(
        &self,
        mut eval_point: Point<C::EF>,
        mle_rounds: Rounds<Message<Mle<C::F, C::A>>>,
        evaluation_claims: Rounds<Evaluations<C::EF, C::A>>,
        prover_data: Rounds<BasefoldProverData<C>>,
        challenger: &mut C::Challenger,
    ) -> Result<BasefoldProof<C::Config>, BasefoldProverError<C>> {
        // Get all the mles from all rounds in order.
        let mles = mle_rounds
            .iter()
            .flat_map(|round| round.clone().into_iter())
            .collect::<Message<Mle<_, _>>>();

        let encoded_messages = prover_data
            .iter()
            .flat_map(|data| data.encoded_messages.iter().cloned())
            .collect::<Message<RsCodeWord<_, _>>>();

        let evaluation_claims = evaluation_claims.into_iter().flatten().collect::<Vec<_>>();

        // Sample a batching challenge and batch the mles and codewords.
        let batching_challenge: C::EF = challenger.sample_ext_element();
        // Batch the mles and codewords.
        let (mle_batch, codeword_batch, batched_eval_claim) = self
            .fri_prover
            .batch(batching_challenge, mles, encoded_messages, evaluation_claims, &self.encoder)
            .await;
        // From this point on, run the BaseFold protocol on the random linear combination codeword,
        // the random linear combination multilinear, and the random linear combination of the
        // evaluation claims.
        let mut current_mle = mle_batch;
        let mut current_codeword = codeword_batch;
        // Initialize the vecs that go into a BaseFoldProof.
        let log_len = current_mle.num_variables();
        let mut univariate_messages: Vec<[C::EF; 2]> = vec![];
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
                .fri_prover
                .commit_phase_round(
                    current_mle,
                    current_codeword,
                    &self.encoder,
                    &self.tcs_prover,
                    challenger,
                )
                .await
                .map_err(BasefoldProverError::CommitPhaseError)?;

            fri_commitments.push(commitment);
            commit_phase_data.push(prover_data);
            commit_phase_values.push(leaves);

            current_mle = folded_mle;
            current_codeword = folded_codeword;
            current_batched_eval_claim = zero_val + beta * one_val;
        }

        let final_poly = self.fri_prover.final_poly(current_codeword).await;
        challenger.observe_ext_element(final_poly);

        let fri_config = self.encoder.config();
        let pow_bits = fri_config.proof_of_work_bits;
        let pow_witness = self.pow_prover.grind(challenger, pow_bits).await;
        // FRI Query Phase.
        let query_indices: Vec<usize> = (0..fri_config.num_queries)
            .map(|_| challenger.sample_bits(log_len as usize + fri_config.log_blowup()))
            .collect();

        // Open the original polynomials at the query indices.
        let mut component_polynomials_query_openings = vec![];
        for prover_data in prover_data {
            let BasefoldProverData { encoded_messages, tcs_prover_data } = prover_data;
            let values =
                self.tcs_prover.compute_openings_at_indices(encoded_messages, &query_indices).await;
            let proof = self
                .tcs_prover
                .prove_openings_at_indices(tcs_prover_data, &query_indices)
                .await
                .map_err(BasefoldProverError::<C>::TcsCommitError)
                .unwrap();
            let opening = TensorCsOpening::<C::Tcs>::new(values, proof);
            component_polynomials_query_openings.push(opening);
        }

        // Provide openings for the FRI query phase.
        let mut query_phase_openings = vec![];
        let mut indices = query_indices;
        for (leaves, data) in commit_phase_values.into_iter().zip_eq(commit_phase_data) {
            for index in indices.iter_mut() {
                *index >>= 1;
            }
            let leaves: Message<Tensor<C::F, C::A>> = leaves.into();
            let values = self.tcs_prover.compute_openings_at_indices(leaves, &indices).await;

            let proof = self
                .tcs_prover
                .prove_openings_at_indices(data, &indices)
                .await
                .map_err(BasefoldProverError::<C>::TcsCommitError)?;
            let opening = TensorCsOpening::<C::Tcs>::new(values, proof);
            query_phase_openings.push(opening);
        }

        Ok(BasefoldProof {
            univariate_messages,
            fri_commitments,
            component_polynomials_query_openings,
            query_phase_openings,
            final_poly,
            pow_witness,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord, Serialize, Deserialize)]
pub struct Poseidon2BabyBear16BasefoldCpuProverComponents;

impl BasefoldProverComponents for Poseidon2BabyBear16BasefoldCpuProverComponents {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type A = CpuBackend;
    type Tcs = MerkleTreeTcs<Poseidon2BabyBearConfig>;
    type Challenger = <Poseidon2BabyBear16BasefoldConfig as BasefoldConfig>::Challenger;
    type Config = Poseidon2BabyBear16BasefoldConfig;
    type Encoder = CpuDftEncoder<BabyBear, Radix2DitParallel>;
    type FriProver = FriCpuProver<Self::Encoder, Self::TcsProver>;
    type TcsProver = FieldMerkleTreeProver<
        <BabyBear as Field>::Packing,
        <BabyBear as Field>::Packing,
        Poseidon2BabyBearConfig,
        8,
    >;
    type PowProver = GrindingPowProver;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord, Serialize, Deserialize)]
pub struct Poseidon2Bn254BasefoldCpuProverComponents;

impl BasefoldProverComponents for Poseidon2Bn254BasefoldCpuProverComponents {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type A = CpuBackend;
    type Tcs = MerkleTreeTcs<Poseidon2Bn254Config>;
    type Challenger = <Poseidon2Bn254FrBasefoldConfig as BasefoldConfig>::Challenger;
    type Config = Poseidon2Bn254FrBasefoldConfig;
    type Encoder = CpuDftEncoder<BabyBear, Radix2DitParallel>;
    type FriProver = FriCpuProver<Self::Encoder, Self::TcsProver>;
    type TcsProver =
        FieldMerkleTreeProver<BabyBear, Bn254Fr, Poseidon2Bn254Config, OUTER_DIGEST_SIZE>;
    type PowProver = GrindingPowProver;
}

impl DefaultBasefoldProver for Poseidon2BabyBear16BasefoldCpuProverComponents {
    fn default_prover(
        verifier: &BasefoldVerifier<Poseidon2BabyBear16BasefoldConfig>,
    ) -> BasefoldProver<Self> {
        let encoder =
            CpuDftEncoder { config: verifier.fri_config, dft: Arc::new(Radix2DitParallel) };
        let fri_prover = FriCpuProver::<
            CpuDftEncoder<BabyBear, Radix2DitParallel>,
            FieldMerkleTreeProver<
                <BabyBear as Field>::Packing,
                <BabyBear as Field>::Packing,
                Poseidon2BabyBearConfig,
                8,
            >,
        >(PhantomData);

        let tcs_prover = Poseidon2BabyBear16Prover::default();
        let pow_prover = GrindingPowProver;
        BasefoldProver { encoder, fri_prover, tcs_prover, pow_prover }
    }
}

impl DefaultBasefoldProver for Poseidon2Bn254BasefoldCpuProverComponents {
    fn default_prover(
        verifier: &BasefoldVerifier<Poseidon2Bn254FrBasefoldConfig>,
    ) -> BasefoldProver<Self> {
        let encoder =
            CpuDftEncoder { config: verifier.fri_config, dft: Arc::new(Radix2DitParallel) };
        let fri_prover = FriCpuProver::<
            CpuDftEncoder<BabyBear, Radix2DitParallel>,
            FieldMerkleTreeProver<BabyBear, Bn254Fr, Poseidon2Bn254Config, OUTER_DIGEST_SIZE>,
        >(PhantomData);

        let tcs_prover = FieldMerkleTreeProver::<
            BabyBear,
            Bn254Fr,
            Poseidon2Bn254Config,
            OUTER_DIGEST_SIZE,
        >::default();
        let pow_prover = GrindingPowProver;
        BasefoldProver { encoder, fri_prover, tcs_prover, pow_prover }
    }
}

#[cfg(test)]
mod tests {
    use futures::prelude::*;
    use rand::thread_rng;
    use slop_baby_bear::BabyBear;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_multilinear::MultilinearPcsVerifier;

    use super::*;

    #[tokio::test]
    async fn test_basefold_prover_backend() {
        type C = Poseidon2BabyBear16BasefoldConfig;
        type Prover = BasefoldProver<Poseidon2BabyBear16BasefoldCpuProverComponents>;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let num_variables = 16;
        let round_widths = [vec![16, 10, 14], vec![20, 78, 34], vec![10, 10]];
        let log_blowup = 1;

        let mut rng = thread_rng();
        let round_mles = round_widths
            .iter()
            .map(|widths| {
                widths
                    .iter()
                    .map(|&w| Mle::<BabyBear>::rand(&mut rng, w, num_variables))
                    .collect::<Message<_>>()
            })
            .collect::<Rounds<_>>();

        let verifier = BasefoldVerifier::<C>::new(log_blowup);
        let prover = Prover::new(&verifier);

        let mut challenger = verifier.challenger();
        let mut commitments = vec![];
        let mut prover_data = Rounds::new();
        let mut eval_claims = Rounds::new();
        let point = Point::<EF>::rand(&mut rng, num_variables);
        for mles in round_mles.iter() {
            let (commitment, data) = prover.commit_mles(mles.clone()).await.unwrap();
            challenger.observe(commitment);
            commitments.push(commitment);
            prover_data.push(data);
            let evaluations = stream::iter(mles.iter())
                .then(|mle| mle.eval_at(&point))
                .collect::<Evaluations<_>>()
                .await;
            eval_claims.push(evaluations);
        }

        let proof = prover
            .prove_trusted_mle_evaluations(
                point.clone(),
                round_mles,
                eval_claims.clone(),
                prover_data,
                &mut challenger,
            )
            .await
            .unwrap();

        let mut challenger = verifier.challenger();
        for commitment in commitments.iter() {
            challenger.observe(*commitment);
        }
        verifier
            .verify_trusted_evaluations(&commitments, point, &eval_claims, &proof, &mut challenger)
            .unwrap();
    }
}

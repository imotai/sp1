use derive_where::derive_where;
use futures::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

use slop_algebra::{AbstractField, ExtensionField, Field};
use slop_alloc::{HasBackend, ToHost};
use slop_challenger::{CanObserve, FieldChallenger, Synchronizable};
use slop_commit::{Message, Rounds};
use slop_multilinear::{
    Evaluations, Mle, MleBaseBackend, MleEvaluationBackend, MultilinearPcsProver, PaddedMle, Point,
};
use slop_stacked::{
    FixedRateInterleaveBackend, InterleaveMultilinears, StackedPcsProver, StackedPcsProverData,
    StackedPcsProverError,
};
use slop_sumcheck::{
    reduce_sumcheck_to_evaluation, ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend,
    SumcheckPolyBackend,
};
use thiserror::Error;

use crate::{
    HadamardProduct, JaggedConfig, JaggedEvalConfig, JaggedEvalProver,
    JaggedLittlePolynomialProverParams, JaggedPcsProof, JaggedPcsVerifier, JaggedSumcheckProver,
};

pub trait JaggedBackend<F: Field, EF: ExtensionField<F>>:
    MleBaseBackend<F>
    + MleBaseBackend<EF>
    + MleEvaluationBackend<F, EF>
    + FixedRateInterleaveBackend<F>
    + ComponentPolyEvalBackend<HadamardProduct<F, EF, Self>, EF>
    + ComponentPolyEvalBackend<HadamardProduct<EF, EF, Self>, EF>
    + SumcheckPolyBackend<HadamardProduct<EF, EF, Self>, EF>
    + SumCheckPolyFirstRoundBackend<HadamardProduct<F, EF, Self>, EF, NextRoundPoly: Send + Sync>
{
}

impl<F, EF, A> JaggedBackend<F, EF> for A
where
    F: Field,
    EF: ExtensionField<F>,
    A: MleBaseBackend<F>
        + MleBaseBackend<EF>
        + MleEvaluationBackend<F, EF>
        + FixedRateInterleaveBackend<F>
        + ComponentPolyEvalBackend<HadamardProduct<F, EF, Self>, EF>
        + ComponentPolyEvalBackend<HadamardProduct<EF, EF, Self>, EF>
        + SumcheckPolyBackend<HadamardProduct<EF, EF, Self>, EF>
        + SumCheckPolyFirstRoundBackend<HadamardProduct<F, EF, Self>, EF>,
    <A as SumCheckPolyFirstRoundBackend<HadamardProduct<F, EF, Self>, EF>>::NextRoundPoly:
        Send + Sync,
{
}

pub trait JaggedProverComponents: Clone + Send + Sync + 'static + Debug {
    type F: Field;
    type EF: ExtensionField<Self::F>;
    type A: JaggedBackend<Self::F, Self::EF>;
    type Challenger: FieldChallenger<Self::F>
        + CanObserve<Self::Commitment>
        + 'static
        + Send
        + Sync
        + Clone
        + Synchronizable;

    type Commitment: 'static + Clone + Send + Sync + Serialize + DeserializeOwned;

    type BatchPcsProof: 'static + Clone + Send + Sync + Serialize + DeserializeOwned;

    type Config: JaggedConfig<
            F = Self::F,
            EF = Self::EF,
            Commitment = Self::Commitment,
            Challenger = Self::Challenger,
            BatchPcsProof = Self::BatchPcsProof,
        >
        + 'static
        + Send
        + Sync
        + Clone
        + Debug;

    type JaggedSumcheckProver: JaggedSumcheckProver<Self::F, Self::EF, Self::A>;

    type BatchPcsProver: MultilinearPcsProver<
        F = Self::F,
        EF = Self::EF,
        A = Self::A,
        Commitment = Self::Commitment,
        Challenger = Self::Challenger,
        Verifier = <Self::Config as JaggedConfig>::BatchPcsVerifier,
        Proof = Self::BatchPcsProof,
    >;
    type Stacker: InterleaveMultilinears<Self::F, Self::A>;

    type JaggedEvalProver: JaggedEvalProver<
            Self::EF,
            Self::Challenger,
            EvalConfig = <Self::Config as JaggedConfig>::JaggedEvaluator,
            EvalProof = <<Self::Config as JaggedConfig>::JaggedEvaluator as JaggedEvalConfig<
                Self::EF,
                Self::Challenger,
            >>::JaggedEvalProof,
        >
        + 'static
        + Send
        + Sync;
}

#[derive(Debug, Clone)]
pub struct JaggedProver<C: JaggedProverComponents> {
    stacked_pcs_prover: StackedPcsProver<C::BatchPcsProver, C::Stacker>,
    jagged_sumcheck_prover: C::JaggedSumcheckProver,
    jagged_eval_prover: C::JaggedEvalProver,
    pub max_log_row_count: usize,
}

#[derive(Serialize, Deserialize)]
#[derive_where(Debug, Clone; StackedPcsProverData<C::BatchPcsProver>: Debug + Clone)]
#[serde(bound(
    serialize = "StackedPcsProverData<C::BatchPcsProver>: Serialize",
    deserialize = "StackedPcsProverData<C::BatchPcsProver>: Deserialize<'de>"
))]
pub struct JaggedProverData<C: JaggedProverComponents> {
    pub stacked_pcs_prover_data: StackedPcsProverData<C::BatchPcsProver>,
    pub row_counts: Arc<Vec<usize>>,
    pub column_counts: Arc<Vec<usize>>,
}

#[derive(Debug, Error)]
pub enum JaggedProverError<C: JaggedProverComponents> {
    #[error("batch pcs prover error")]
    BatchPcsProverError(
        StackedPcsProverError<<C::BatchPcsProver as MultilinearPcsProver>::ProverError>,
    ),
}

pub trait DfeaultJaggedProver: JaggedProverComponents {
    fn prover_from_verifier(verifier: &JaggedPcsVerifier<Self::Config>) -> JaggedProver<Self>;
}

impl<C: JaggedProverComponents> JaggedProver<C> {
    pub const fn new(
        max_log_row_count: usize,
        stacked_pcs_prover: StackedPcsProver<C::BatchPcsProver, C::Stacker>,
        jagged_sumcheck_prover: C::JaggedSumcheckProver,
        jagged_eval_prover: C::JaggedEvalProver,
    ) -> Self {
        Self { stacked_pcs_prover, jagged_sumcheck_prover, jagged_eval_prover, max_log_row_count }
    }

    pub fn from_verifier(verifier: &JaggedPcsVerifier<C::Config>) -> Self
    where
        C: DfeaultJaggedProver,
    {
        C::prover_from_verifier(verifier)
    }

    #[inline]
    pub const fn log_stacking_height(&self) -> u32 {
        self.stacked_pcs_prover.log_stacking_height
    }

    /// Commit to a batch of padded multilinears.
    ///
    /// The jagged polyniomial commitments scheme is able to commit to sparse polynomials having
    /// very few or no real rows.
    /// **Note** the padding values will be ignored and treated as though they are zero.
    pub async fn commit_multilinears(
        &self,
        multilinears: Vec<PaddedMle<C::F, C::A>>,
    ) -> Result<(C::Commitment, JaggedProverData<C>), JaggedProverError<C>> {
        let mut row_counts = multilinears.iter().map(|x| x.num_real_entries()).collect::<Vec<_>>();
        let mut column_counts =
            multilinears.iter().map(|x| x.num_polynomials()).collect::<Vec<_>>();
        // TODO: why is this here?
        column_counts.push(1);

        // Check the vality of the input multilinears.
        for padded_mle in multilinears.iter() {
            // Check that the number of variables matches what the prover expects.
            assert_eq!(padded_mle.num_variables(), self.max_log_row_count as u32);
        }

        // TODO: more comments
        let next_multiple = multilinears
            .iter()
            .map(|mle| mle.num_real_entries() * mle.num_polynomials())
            .sum::<usize>()
            .next_multiple_of(1 << self.log_stacking_height());

        let next_multiple = if next_multiple > 0 {
            next_multiple
        } else {
            1 << self.stacked_pcs_prover.log_stacking_height
        };

        row_counts.push(
            next_multiple
                - multilinears
                    .iter()
                    .map(|mle| mle.num_real_entries() * mle.num_polynomials())
                    .sum::<usize>(),
        );

        // Collect all the multilinears that have at least one non-zero entry into a commit message
        // for the dense PCS.
        let message =
            multilinears.into_iter().filter_map(|mle| mle.into_inner()).collect::<Message<_>>();

        let (commitment, data) =
            self.stacked_pcs_prover.commit_multilinears(message).await.unwrap();

        let jagged_prover_data = JaggedProverData {
            stacked_pcs_prover_data: data,
            row_counts: Arc::new(row_counts),
            column_counts: Arc::new(column_counts),
        };

        Ok((commitment, jagged_prover_data))
    }

    pub async fn prove_trusted_evaluations(
        &self,
        eval_point: Point<C::EF>,
        evaluation_claims: Rounds<Evaluations<C::EF, C::A>>,
        prover_data: Rounds<JaggedProverData<C>>,
        challenger: &mut C::Challenger,
    ) -> Result<JaggedPcsProof<C::Config>, JaggedProverError<C>> {
        let num_col_variables = prover_data
            .iter()
            .map(|data| data.column_counts.iter().sum::<usize>())
            .sum::<usize>()
            .next_power_of_two()
            .ilog2();
        let z_col = (0..num_col_variables)
            .map(|_| challenger.sample_ext_element::<C::EF>())
            .collect::<Point<_>>();

        let z_row = eval_point;

        // Collect the claims for the different polynomials.
        let mut column_claims = stream::iter(evaluation_claims.iter().flatten())
            .then(|evals| evals.to_host())
            .flat_map(|evals| stream::iter(evals.unwrap()))
            .collect::<Vec<C::EF>>()
            .await;

        let insertion_points = prover_data
            .iter()
            .map(|data| data.column_counts.iter().sum::<usize>() - 1)
            .scan(0, |state, x| {
                *state += x;
                Some(*state)
            })
            .collect::<Vec<_>>();

        for insertion_point in insertion_points.iter().rev().skip(1) {
            column_claims.insert(*insertion_point, C::EF::zero());
        }

        column_claims.resize(column_claims.len().next_power_of_two(), C::EF::zero());

        assert!(prover_data
            .iter()
            .flat_map(|data| data.row_counts.iter())
            .all(|x| *x <= 1 << self.max_log_row_count));

        let row_data =
            prover_data.iter().map(|data| data.row_counts.clone()).collect::<Rounds<_>>();
        let column_data =
            prover_data.iter().map(|data| data.column_counts.clone()).collect::<Rounds<_>>();

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
                            std::iter::repeat(row_count).take(column_count)
                        })
                })
                .collect(),
            self.max_log_row_count,
        );

        // Generate the jagged sumcheck proof.
        let backend = prover_data[0].stacked_pcs_prover_data.interleaved_mles[0].backend().clone();

        let z_row_backend = z_row.copy_into(&backend);
        let z_col_backend = z_col.copy_into(&backend);

        let all_mles = prover_data
            .iter()
            .map(|data| data.stacked_pcs_prover_data.interleaved_mles.clone())
            .collect::<Rounds<_>>();

        let sumcheck_poly = self
            .jagged_sumcheck_prover
            .jagged_sumcheck_poly(
                all_mles,
                &params,
                row_data,
                column_data,
                &z_row_backend,
                &z_col_backend,
            )
            .await;

        // The overall evaluation claim of the sparse polynomial is inferred from the individual
        // table claims.
        let column_claims = Mle::from(column_claims);
        let sumcheck_claim = column_claims.eval_at(&z_col).await[0];

        let (sumcheck_proof, component_poly_evals) = reduce_sumcheck_to_evaluation(
            vec![sumcheck_poly],
            challenger,
            vec![sumcheck_claim],
            1,
            C::EF::one(),
        )
        .await;

        let final_eval_point = sumcheck_proof.point_and_eval.0.clone();

        let jagged_eval_proof = self
            .jagged_eval_prover
            .prove_jagged_evaluation(&params, &z_row, &z_col, &final_eval_point, challenger)
            .await;

        let (_, stack_point) = final_eval_point
            .split_at(final_eval_point.dimension() - self.log_stacking_height() as usize);
        let stack_point = stack_point.copy_into(&backend);
        let batch_evaluations = stream::iter(prover_data.iter())
            .then(|data| {
                self.stacked_pcs_prover
                    .round_batch_evaluations(&stack_point, &data.stacked_pcs_prover_data)
            })
            .collect::<Rounds<_>>()
            .await;

        let stacked_prover_data =
            prover_data.into_iter().map(|data| data.stacked_pcs_prover_data).collect::<Rounds<_>>();

        let stacked_pcs_proof = self
            .stacked_pcs_prover
            .prove_trusted_evaluation(
                final_eval_point,
                component_poly_evals[0][0],
                stacked_prover_data,
                batch_evaluations,
                challenger,
            )
            .await
            .unwrap();

        Ok(JaggedPcsProof {
            stacked_pcs_proof,
            sumcheck_proof,
            jagged_eval_proof,
            params: params.into_verifier_params(),
        })
    }
}

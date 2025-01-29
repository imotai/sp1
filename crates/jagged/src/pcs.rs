use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, ExtensionField, Field,
};
use slop_challenger::{CanObserve, CanSample, FieldChallenger};
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use slop_multilinear::{
    MainTraceProverData, Mle, MultilinearPcsBatchProver, MultilinearPcsBatchVerifier,
    MultilinearPcsProver, MultilinearPcsVerifier, Point, StackedPcsProver, StackedPcsVerifier,
};
use slop_sumcheck::{
    partially_verify_sumcheck_proof, PartialSumcheckProof, SumcheckError, SumcheckPoly,
    SumcheckPolyBase, SumcheckPolyFirstRound,
};
use slop_sumcheck_prover::reduce_sumcheck_to_evaluation;
use slop_utils::{log2_ceil_usize, log2_strict_usize};

use std::iter::repeat;

use crate::{JaggedLittlePolynomialProverParams, JaggedLittlePolynomialVerifierParams};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JaggedPcsError<PcsError, SumcheckError> {
    PcsError(PcsError),
    SumcheckError(SumcheckError),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JaggedPcsProof<Proof, EF: AbstractField> {
    pub pcs_proof: Proof,
    pub sumcheck_proof: PartialSumcheckProof<EF>,
    pub params: JaggedLittlePolynomialVerifierParams<EF>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JaggedPcs<Pcs> {
    pub pcs: Pcs,
    pub max_log_row_count: usize,
}

impl<Pcs: MultilinearPcsVerifier> MultilinearPcsBatchVerifier for JaggedPcs<Pcs>
where
    Pcs::Challenger: CanObserve<Pcs::F> + CanSample<Pcs::EF>,
    Pcs::EF: Field,
    Pcs::Proof: Serialize + DeserializeOwned + Clone,
{
    type F = Pcs::F;

    type EF = Pcs::EF;

    type Proof = JaggedPcsProof<Pcs::Proof, Pcs::EF>;

    type Commitment = Pcs::Commitment;

    type Error = JaggedPcsError<Pcs::Error, SumcheckError>;

    type Challenger = Pcs::Challenger;

    fn verify_trusted_evaluations(
        &self,
        point: Point<Self::EF>,
        evaluation_claims: &[&[Self::EF]],
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        let JaggedPcsProof { pcs_proof, sumcheck_proof, params } = proof;
        let z_col = Point::new(
            (0..params.max_log_column_count)
                .map(|_| challenger.sample_ext_element::<Self::EF>())
                .collect(),
        );

        let z_row = point;

        let z_tab = Point::new(
            (0..log2_ceil_usize(params.column_counts.len()))
                .map(|_| challenger.sample_ext_element::<Self::EF>())
                .collect(),
        );

        let table_claims = evaluation_claims
            .iter()
            .map(|x| {
                Mle::eval_at_point(
                    &x.iter()
                        .copied()
                        .chain(repeat(Self::EF::zero()))
                        .take(1 << z_col.dimension())
                        .collect::<Vec<_>>()
                        .into(),
                    &z_col,
                )
            })
            .collect::<Vec<_>>();

        let sumcheck_claim = Mle::eval_at_point(&table_claims.into(), &z_tab);

        assert!(sumcheck_claim == sumcheck_proof.claimed_sum);

        partially_verify_sumcheck_proof(sumcheck_proof, challenger)
            .map_err(JaggedPcsError::SumcheckError)?;

        let expected_eval = sumcheck_proof.point_and_eval.1
            / params.full_jagged_little_polynomial_evaluation(
                &z_tab,
                &z_row,
                &z_col,
                &sumcheck_proof.point_and_eval.0,
            );

        <Pcs as MultilinearPcsVerifier>::verify_trusted_evaluation(
            &self.pcs,
            sumcheck_proof.point_and_eval.0.clone(),
            expected_eval,
            commitment,
            pcs_proof,
            challenger,
        )
        .map_err(JaggedPcsError::PcsError)
    }
}

#[derive(Debug, Clone)]
pub struct JaggedSumcheckPoly<K: AbstractField, EK: AbstractExtensionField<K>> {
    pub mle: Mle<K>,
    pub jagged_evals: Mle<EK>,
}

impl<K: Field, EK: ExtensionField<K>> JaggedSumcheckPoly<K, EK> {
    pub fn new(
        mle: Mle<K>,
        jagged_params: JaggedLittlePolynomialProverParams,
        z_tab: &Point<EK>,
        z_row: &Point<EK>,
        z_col: &Point<EK>,
    ) -> Self {
        Self {
            mle,
            jagged_evals: jagged_params
                .partial_jagged_little_polynomial_evaluation(z_tab, z_row, z_col),
        }
    }
}

impl<Pcs: MultilinearPcsBatchProver> MultilinearPcsBatchProver for JaggedPcs<StackedPcsProver<Pcs>>
where
    Pcs::MultilinearProverData:
        MainTraceProverData<RowMajorMatrix<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>>,
    <Pcs::PCS as MultilinearPcsBatchVerifier>::Challenger: CanObserve<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>
        + CanSample<<Pcs::PCS as MultilinearPcsBatchVerifier>::EF>,
    <Pcs::PCS as MultilinearPcsBatchVerifier>::Proof: Clone + Serialize + DeserializeOwned,
{
    type PCS = JaggedPcs<StackedPcsVerifier<Pcs::PCS>>;

    type MultilinearProverData = (Pcs::MultilinearProverData, Vec<usize>, Vec<usize>);

    fn commit_multilinears(
        &self,
        data: Vec<RowMajorMatrix<<Self::PCS as MultilinearPcsBatchVerifier>::F>>,
    ) -> (<Self::PCS as MultilinearPcsBatchVerifier>::Commitment, Self::MultilinearProverData) {
        let row_counts = data.iter().map(|x| x.height()).collect::<Vec<_>>();

        let column_counts = data.iter().map(|x| x.width()).collect::<Vec<_>>();
        let (commitment, data) =
            self.pcs.commit_multilinear(data.into_iter().map(|x| x.values).collect());
        (commitment, (data, row_counts, column_counts))
    }

    fn prove_trusted_evaluations(
        &self,
        eval_point: Point<<Self::PCS as MultilinearPcsBatchVerifier>::EF>,
        expected_evals: &[&[<Self::PCS as MultilinearPcsBatchVerifier>::EF]],
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsBatchVerifier>::Challenger,
    ) -> <Self::PCS as MultilinearPcsBatchVerifier>::Proof {
        let z_col = Point::new(
            (0..log2_strict_usize(*prover_data.2.iter().max().unwrap()))
                .map(|_| {
                    challenger
                        .sample_ext_element::<<Self::PCS as MultilinearPcsBatchVerifier>::EF>()
                })
                .collect(),
        );

        let z_row = eval_point;

        let z_tab = Point::new(
            (0..log2_ceil_usize(prover_data.2.len()))
                .map(|_| {
                    challenger
                        .sample_ext_element::<<Self::PCS as MultilinearPcsBatchVerifier>::EF>()
                })
                .collect(),
        );

        let table_claims = tracing::info_span!("compute expected evaluations").in_scope(|| {
            expected_evals
                .iter()
                .map(|x| {
                    Mle::eval_at_point(
                        &x.iter()
                            .copied()
                            .chain(repeat(<Self::PCS as MultilinearPcsBatchVerifier>::EF::zero()))
                            .take(*prover_data.2.iter().max().unwrap())
                            .collect::<Vec<_>>()
                            .into(),
                        &z_col,
                    )
                })
                .collect::<Vec<_>>()
        });

        let params = JaggedLittlePolynomialProverParams::new(
            prover_data.1,
            prover_data.2,
            self.max_log_row_count,
        );

        let (base_data, table_data) = prover_data.0.split_off_main_traces();

        let sumcheck_poly = tracing::info_span!("make sumcheck poly").in_scope(|| {
            JaggedSumcheckPoly::<
                <Self::PCS as MultilinearPcsBatchVerifier>::F,
                <Self::PCS as MultilinearPcsBatchVerifier>::EF,
            >::new(
                tracing::info_span!("assemble traces").in_scope(|| {
                    table_data
                        .iter()
                        // TODO: Remove this clone.
                        .flat_map(|x| x.clone().transpose().values.into_iter())
                        .collect::<Vec<_>>()
                        .into()
                }),
                params.clone(),
                &z_tab,
                &z_row,
                &z_col,
            )
        });

        // The overall evaluation claim of the sparse polynomial is inferred from the individual
        // table claims.
        let sumcheck_claim = Mle::eval_at_point(&table_claims.into(), &z_tab);

        let lambda = challenger.sample();

        let (sumcheck_proof, component_poly_evals) = tracing::info_span!("generate sumcheck proof")
            .in_scope(|| {
                reduce_sumcheck_to_evaluation(
                    vec![sumcheck_poly],
                    challenger,
                    vec![sumcheck_claim],
                    1,
                    lambda,
                )
            });

        let orig_prover_data = Pcs::MultilinearProverData::reconstitute(base_data, table_data);

        let pcs_proof = tracing::info_span!("generate stacked pcs proof").in_scope(|| {
            self.pcs.prove_trusted_evaluation(
                sumcheck_proof.point_and_eval.0.clone(),
                component_poly_evals[0][0],
                orig_prover_data,
                challenger,
            )
        });

        JaggedPcsProof { pcs_proof, sumcheck_proof, params: params.into_verifier_params() }
    }
}

impl<K: Field, EK: ExtensionField<K>> JaggedSumcheckPoly<K, EK> {
    fn sum_as_poly_in_last_variable(
        &self,
        claim: Option<EK>,
    ) -> slop_algebra::UnivariatePolynomial<EK> {
        assert!(self.mle.num_variables() == self.jagged_evals.num_variables());
        assert!(self.mle.num_variables() > 0);

        // The sumcheck polynomial is a multi-quadratic polynomial, so three evaluations are needed.
        let eval_0 = self
            .jagged_evals
            .guts
            .par_iter()
            .step_by(2)
            .zip(self.mle.guts.par_iter().step_by(2))
            .map(|(x, y)| *x * *y)
            .sum();

        let eval_1 = claim.map(|x| x - eval_0).unwrap_or(
            self.jagged_evals
                .guts
                .par_iter()
                .skip(1)
                .step_by(2)
                .zip(self.mle.guts.par_iter().skip(1).step_by(2))
                .map(|(x, y)| *x * *y)
                .sum(),
        );

        let eval_half: EK = self
            .jagged_evals
            .guts
            .par_iter()
            .step_by(2)
            .zip(self.jagged_evals.guts.par_iter().skip(1).step_by(2))
            .zip(self.mle.guts.par_iter().step_by(2))
            .zip(self.mle.guts.par_iter().skip(1).step_by(2))
            .map(|(((je_0, je_1), mle_0), mle_1)| (*je_0 + *je_1) * (*mle_0 + *mle_1))
            .sum();

        interpolate_univariate_polynomial(
            &[
                EK::from_canonical_u16(0),
                EK::from_canonical_u16(1),
                EK::from_canonical_u16(2).inverse(),
            ],
            &[eval_0, eval_1, eval_half * EK::from_canonical_u16(4).inverse()],
        )
    }

    fn n_variables(&self) -> usize {
        assert!(self.mle.num_variables() == self.jagged_evals.num_variables());
        self.mle.num_variables()
    }

    fn get_component_poly_evals(&self) -> Vec<EK> {
        assert!(self.mle.num_variables() == 0);
        vec![EK::from_base(self.mle.guts[0]), self.jagged_evals.guts[0]]
    }
}

impl<K: Field, EK: ExtensionField<K>> SumcheckPolyBase<EK> for JaggedSumcheckPoly<K, EK> {
    fn n_variables(&self) -> usize {
        self.n_variables()
    }

    fn get_component_poly_evals(&self) -> Vec<EK> {
        self.get_component_poly_evals()
    }
}

impl<EK: Field> SumcheckPoly<EK> for JaggedSumcheckPoly<EK, EK> {
    fn fix_last_variable(self, alpha: EK) -> Self {
        let new_mle = self.mle.fix_last_variable(alpha);
        let new_jagged_evals = self.jagged_evals.fix_last_variable(alpha);

        Self { mle: new_mle, jagged_evals: new_jagged_evals }
    }

    fn sum_as_poly_in_last_variable(
        &self,
        claim: Option<EK>,
    ) -> slop_algebra::UnivariatePolynomial<EK> {
        self.sum_as_poly_in_last_variable(claim)
    }
}

impl<K: Field, EK: ExtensionField<K>> SumcheckPolyFirstRound<EK> for JaggedSumcheckPoly<K, EK> {
    fn fix_t_variables(self, alpha: EK, t: usize) -> impl SumcheckPoly<EK> {
        assert!(t == 1);
        let new_mle: Mle<EK> = self.mle.sc_fix_last_variable(alpha);
        let new_jagged_evals = self.jagged_evals.fix_last_variable(alpha);

        JaggedSumcheckPoly::<EK, EK> { mle: new_mle, jagged_evals: new_jagged_evals }
    }

    fn sum_as_poly_in_last_t_variables(
        &self,
        _claim: Option<EK>,
        t: usize,
    ) -> slop_algebra::UnivariatePolynomial<EK> {
        assert!(t == 1);
        self.sum_as_poly_in_last_variable(None)
    }
}

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, ExtensionField, Field,
};
use slop_challenger::{CanObserve, FieldChallenger};
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use slop_multilinear::{Mle, MultilinearPcsProver, MultilinearPcsVerifier, Point};
use slop_sumcheck::{
    partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation, ComponentPoly,
    PartialSumcheckProof, SumcheckError, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound,
};
use slop_utils::log2_ceil_usize;

use std::{iter::repeat, sync::Arc};

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

impl<Pcs> JaggedPcs<Pcs> {
    pub fn new(pcs: Pcs, max_log_row_count: usize) -> Self {
        Self { pcs, max_log_row_count }
    }
}

impl<Pcs: MultilinearPcsVerifier> JaggedPcs<Pcs>
where
    Pcs::Challenger: FieldChallenger<Pcs::F> + CanObserve<Pcs::Commitment>,
    Pcs::EF: Field,
    Pcs::Proof: Serialize + DeserializeOwned + Clone,
{
    pub fn verify_trusted_evaluations(
        &self,
        point: Point<Pcs::EF>,
        evaluation_claims: &[&[&[Pcs::EF]]],
        commitment: &[Pcs::Commitment],
        proof: &JaggedPcsProof<Pcs::Proof, Pcs::EF>,
        // `insertion_points` is the running sum of the number of matrices in each commit. Its length
        // should be one less than the number of commitments (the finalization commitment is not included).
        insertion_points: &[usize],
        challenger: &mut Pcs::Challenger,
    ) -> Result<(), JaggedPcsError<Pcs::Error, SumcheckError>> {
        let JaggedPcsProof { pcs_proof, sumcheck_proof, params } = proof;
        let z_col = (0..log2_ceil_usize(params.col_prefix_sums.len() - 1))
            .map(|_| challenger.sample_ext_element::<Pcs::EF>())
            .collect::<Point<_>>();

        let z_row = point;

        let mut column_claims = evaluation_claims
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<_>>();

        // For each commit, Rizz needed a commitment to a vector of length a multiple of
        // 1 << self.pcs.log_stacking_height, and this is achieved by adding a single column of zeroes
        // as the last matrix of the commitment. We insert these "artificial" zeroes into the evaluation
        // claims.
        for insertion_point in insertion_points.iter().rev() {
            column_claims.insert(*insertion_point, Pcs::EF::zero());
        }

        column_claims.resize(column_claims.len().next_power_of_two(), Pcs::EF::zero());

        let sumcheck_claim = Mle::eval_at(&column_claims.into(), &z_col).to_vec()[0];

        assert!(sumcheck_claim == sumcheck_proof.claimed_sum);

        let _lambda = challenger.sample_ext_element::<Pcs::EF>();

        partially_verify_sumcheck_proof(sumcheck_proof, challenger)
            .map_err(JaggedPcsError::SumcheckError)?;

        let expected_eval = tracing::info_span!("compute q evaluation claim").in_scope(|| {
            sumcheck_proof.point_and_eval.1
                / params.full_jagged_little_polynomial_evaluation(
                    &z_row,
                    &z_col,
                    &sumcheck_proof.point_and_eval.0,
                )
        });

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

pub struct MachineJaggedPcs<'a, Pcs: MultilinearPcsVerifier> {
    pub pcs: &'a JaggedPcs<Pcs>,
    pub column_counts_by_round: Vec<Vec<usize>>,
}

impl<'a, Pcs: MultilinearPcsVerifier> MachineJaggedPcs<'a, Pcs>
where
    Pcs::Challenger: FieldChallenger<Pcs::F> + CanObserve<Pcs::Commitment>,
    Pcs::EF: Field,
    Pcs::Proof: Serialize + DeserializeOwned + Clone,
{
    pub fn new(pcs: &'a JaggedPcs<Pcs>, column_counts_by_round: Vec<Vec<usize>>) -> Self {
        Self { pcs, column_counts_by_round }
    }

    pub fn verify_trusted_evaluations(
        &self,
        point: Point<Pcs::EF>,
        evaluation_claims: &[&[&[Pcs::EF]]],
        commitments: &[Pcs::Commitment],
        proof: &JaggedPcsProof<Pcs::Proof, Pcs::EF>,
        challenger: &mut Pcs::Challenger,
    ) -> Result<(), JaggedPcsError<Pcs::Error, SumcheckError>> {
        self.pcs.verify_trusted_evaluations(
            point,
            evaluation_claims,
            commitments,
            proof,
            &self
                .column_counts_by_round
                .iter()
                .scan(0, |state, y| {
                    *state += y.iter().sum::<usize>();
                    Some(*state)
                })
                .collect::<Vec<_>>(),
            challenger,
        )
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
        z_row: &Point<EK>,
        z_col: &Point<EK>,
    ) -> Self {
        Self {
            mle,
            jagged_evals: jagged_params.partial_jagged_little_polynomial_evaluation(z_row, z_col),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaggedProverData<ProverData> {
    base_data: ProverData,
    pub row_counts: Vec<usize>,
    pub column_counts: Vec<usize>,
}

impl<Pcs: MultilinearPcsBatchProver> JaggedPcs<StackedPcsProver<Pcs>>
where
    Pcs::MultilinearProverData:
        MainTraceProverData<RowMajorMatrix<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>>,
    <Pcs::PCS as MultilinearPcsBatchVerifier>::Challenger: FieldChallenger<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>
        + CanObserve<<Pcs::PCS as MultilinearPcsBatchVerifier>::Commitment>,
    <Pcs::PCS as MultilinearPcsBatchVerifier>::Proof: Clone + Serialize + DeserializeOwned,
{
    pub fn commit_multilinears(
        &self,
        data: Vec<RowMajorMatrix<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>>,
    ) -> (
        <Pcs::PCS as MultilinearPcsBatchVerifier>::Commitment,
        JaggedProverData<Pcs::MultilinearProverData>,
    ) {
        let mut row_counts = data.iter().map(|x| x.height()).collect::<Vec<_>>();

        let mut column_counts = data.iter().map(|x| x.width()).collect::<Vec<_>>();

        column_counts.push(1);

        let next_multiple = data
            .iter()
            .map(|mat| mat.values.len())
            .sum::<usize>()
            .next_multiple_of(1 << self.pcs.log_stacking_height);

        let next_multiple =
            if next_multiple > 0 { next_multiple } else { 1 << self.pcs.log_stacking_height };

        row_counts.push(next_multiple - data.iter().map(|mat| mat.values.len()).sum::<usize>());

        let (commitment, data) = self.pcs.commit_multilinear(
            data.into_iter()
                .map(|x| if x.height() != 0 { x.transpose().values } else { x.values })
                .collect(),
        );
        (commitment, JaggedProverData { base_data: data, row_counts, column_counts })
    }

    pub fn prove_trusted_evaluations(
        &self,
        eval_point: Point<<Pcs::PCS as MultilinearPcsBatchVerifier>::EF>,
        expected_evals: &[&[&[<Pcs::PCS as MultilinearPcsBatchVerifier>::EF]]],
        prover_data: &[Arc<JaggedProverData<Pcs::MultilinearProverData>>],
        challenger: &mut <Pcs::PCS as MultilinearPcsBatchVerifier>::Challenger,
    ) -> JaggedPcsProof<StackedProof<Pcs::PCS>, <Pcs::PCS as MultilinearPcsBatchVerifier>::EF> {
        let z_col = (0..log2_ceil_usize(
            prover_data.iter().map(|data| data.column_counts.iter().sum::<usize>()).sum::<usize>(),
        ))
            .map(|_| {
                challenger.sample_ext_element::<<Pcs::PCS as MultilinearPcsBatchVerifier>::EF>()
            })
            .collect::<Point<_>>();

        let z_row = eval_point;

        let mut column_claims =
            tracing::info_span!("compute expected evaluations").in_scope(|| {
                expected_evals
                    .iter()
                    .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
                    .collect::<Vec<_>>()
            });

        let insertion_points = prover_data
            .iter()
            .map(|data| data.column_counts.iter().sum::<usize>() - 1)
            .scan(0, |state, x| {
                *state += x;
                Some(*state)
            })
            .collect::<Vec<_>>();

        for insertion_point in insertion_points.iter().rev().skip(1) {
            column_claims
                .insert(*insertion_point, <Pcs::PCS as MultilinearPcsBatchVerifier>::EF::zero());
        }

        column_claims.resize(
            column_claims.len().next_power_of_two(),
            <Pcs::PCS as MultilinearPcsBatchVerifier>::EF::zero(),
        );

        assert!(prover_data
            .iter()
            .flat_map(|data| data.row_counts.iter())
            .all(|x| *x <= 1 << self.max_log_row_count));

        let params = JaggedLittlePolynomialProverParams::new(
            prover_data
                .iter()
                .flat_map(|data| {
                    data.row_counts
                        .iter()
                        .copied()
                        .zip(data.column_counts.iter().copied())
                        .flat_map(|(row_count, column_count)| repeat(row_count).take(column_count))
                })
                .collect(),
            self.max_log_row_count,
        );

        let (_, table_data): (Vec<_>, Vec<_>) =
            prover_data.iter().map(|x| x.base_data.split_off_main_traces()).unzip();

        // TODO: Eliminate this clone.
        let mut table_data_vec = table_data
            .iter()
            .flat_map(|y| y.iter())
            .flat_map(|x| x.clone().transpose().values.into_iter())
            .collect::<Vec<_>>();

        table_data_vec.resize(
            table_data_vec.len().next_power_of_two(),
            <Pcs::PCS as MultilinearPcsBatchVerifier>::F::zero(),
        );

        let sumcheck_poly = tracing::info_span!("make sumcheck poly").in_scope(|| {
            JaggedSumcheckPoly::<
                <Pcs::PCS as MultilinearPcsBatchVerifier>::F,
                <Pcs::PCS as MultilinearPcsBatchVerifier>::EF,
            >::new(
                tracing::info_span!("assemble traces").in_scope(|| table_data_vec.into()),
                params.clone(),
                &z_row,
                &z_col,
            )
        });

        // The overall evaluation claim of the sparse polynomial is inferred from the individual
        // table claims.
        let column_claims = Mle::from(column_claims);
        let sumcheck_claim = column_claims.eval_at(&z_col)[0];

        let lambda = challenger.sample_ext_element();

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

        let pcs_proof = tracing::info_span!("generate stacked pcs proof").in_scope(|| {
            self.pcs.prove_trusted_evaluation(
                sumcheck_proof.point_and_eval.0.clone(),
                component_poly_evals[0][0],
                prover_data.iter().map(|x| &x.base_data).collect(),
                challenger,
            )
        });

        JaggedPcsProof { pcs_proof, sumcheck_proof, params: params.into_verifier_params() }
    }

    pub fn finalize(
        &self,
        data: &mut Vec<Arc<JaggedProverData<Pcs::MultilinearProverData>>>,
        commits: &mut Vec<Com<Pcs::PCS>>,
        challenger: &mut <Pcs::PCS as MultilinearPcsBatchVerifier>::Challenger,
    ) {
        let total_area = data
            .iter()
            .map(|x| {
                x.column_counts
                    .iter()
                    .zip(x.row_counts.iter())
                    .map(|(col_count, row_count)| (col_count * row_count))
                    .sum::<usize>()
            })
            .sum::<usize>();
        let needed_zeroes = total_area.next_power_of_two() - total_area;

        let new_matrix = RowMajorMatrix::new(
            repeat(<Pcs::PCS as MultilinearPcsBatchVerifier>::F::zero())
                .take(needed_zeroes)
                .collect(),
            needed_zeroes / (1 << self.pcs.log_stacking_height),
        );

        if needed_zeroes != 0 {
            let (commit, new_data) = self.commit_multilinears(vec![new_matrix]);
            challenger.observe(commit.clone());
            commits.push(commit);
            data.push(Arc::new(new_data));
        }
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
        let (eval_0, eval_half) = rayon::join(
            || {
                self.jagged_evals
                    .guts()
                    .as_slice()
                    .par_iter()
                    .step_by(2)
                    .zip(self.mle.guts().as_slice().par_iter().step_by(2))
                    .map(|(x, y)| *x * *y)
                    .sum()
            },
            || {
                self.jagged_evals
                    .guts()
                    .as_slice()
                    .par_iter()
                    .step_by(2)
                    .zip(self.jagged_evals.guts().as_slice().par_iter().skip(1).step_by(2))
                    .zip(self.mle.guts().as_slice().par_iter().step_by(2))
                    .zip(self.mle.guts().as_slice().par_iter().skip(1).step_by(2))
                    .map(|(((je_0, je_1), mle_0), mle_1)| (*je_0 + *je_1) * (*mle_0 + *mle_1))
                    .sum::<EK>()
            },
        );

        let eval_1 = claim.map(|x| x - eval_0).unwrap_or(
            self.jagged_evals
                .guts()
                .as_slice()
                .par_iter()
                .skip(1)
                .step_by(2)
                .zip(self.mle.guts().as_slice().par_iter().skip(1).step_by(2))
                .map(|(x, y)| *x * *y)
                .sum(),
        );

        interpolate_univariate_polynomial(
            &[
                EK::from_canonical_u16(0),
                EK::from_canonical_u16(1),
                EK::from_canonical_u16(2).inverse(),
            ],
            &[eval_0, eval_1, eval_half * EK::from_canonical_u16(4).inverse()],
        )
    }

    fn n_variables(&self) -> u32 {
        assert!(self.mle.num_variables() == self.jagged_evals.num_variables());
        self.mle.num_variables()
    }

    fn get_component_poly_evals(&self) -> Vec<EK> {
        assert!(self.mle.num_variables() == 0);
        vec![EK::from_base(*self.mle.guts()[[0, 0]]), *self.jagged_evals.guts()[[0, 0]]]
    }
}

impl<K: Field, EK: ExtensionField<K>> SumcheckPolyBase for JaggedSumcheckPoly<K, EK> {
    fn n_variables(&self) -> u32 {
        self.n_variables()
    }
}

impl<K: Field, EK: ExtensionField<K>> ComponentPoly<EK> for JaggedSumcheckPoly<K, EK> {
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
        let new_mle: Mle<EK> = self.mle.fix_last_variable(alpha);
        let new_jagged_evals = self.jagged_evals.fix_last_variable(alpha);

        JaggedSumcheckPoly::<EK, EK> { mle: new_mle, jagged_evals: new_jagged_evals }
    }

    fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<EK>,
        t: usize,
    ) -> slop_algebra::UnivariatePolynomial<EK> {
        assert!(t == 1);
        self.sum_as_poly_in_last_variable(claim)
    }
}

use std::marker::PhantomData;
use std::sync::Arc;

use itertools::Itertools;
use slop_algebra::{
    interpolate_univariate_polynomial, ExtensionField, Field, UnivariatePolynomial,
};
use slop_alloc::{Backend, Buffer};
use slop_multilinear::{Mle, Point};
use slop_sumcheck::{ComponentPoly, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound};
use slop_utils::log2_ceil_usize;

use super::JaggedAssistSumAsPoly;

/// A struct that represents the polynomial that is used to evaluate the sumcheck.
pub struct JaggedEvalSumcheckPoly<
    F: Field,
    EF: ExtensionField<F>,
    BPE: JaggedAssistSumAsPoly<F, EF, A> + Send + Sync,
    A: Backend,
> {
    /// Batch evaluator of the branching program.
    bp_batch_eval: BPE,
    /// The random point generated during the sumcheck proving time.
    rho: Point<EF>,
    /// The z_col point.
    z_col: Point<EF>,
    /// This is a concatenation of the bitstring of t_c and t_{c+1} for every column c.
    merged_prefix_sums: Arc<Vec<Point<F>>>,
    /// The evaluations of the z_col at the merged prefix sums.
    z_col_eq_vals: Vec<EF>,
    /// The sumcheck round number that this poly is used in.
    round_num: usize,
    /// The intermediate full evaluations of the eq polynomials.
    intermediate_eq_full_evals: Vec<EF>,
    /// The half value.
    half: EF,

    _marker: PhantomData<A>,
}

impl<
        F: Field,
        EF: ExtensionField<F>,
        A: Backend,
        BPE: JaggedAssistSumAsPoly<F, EF, A> + Send + Sync,
    > JaggedEvalSumcheckPoly<F, EF, BPE, A>
{
    pub async fn new(
        z_row: Point<EF>,
        z_col: Point<EF>,
        z_index: Point<EF>,
        prefix_sums: Vec<usize>,
    ) -> Self {
        let log_m = log2_ceil_usize(*prefix_sums.last().unwrap());
        let col_prefix_sums: Vec<Point<F>> =
            prefix_sums.iter().map(|&x| Point::from_usize(x, log_m + 1)).collect();

        // Generate all of the merged prefix sums.
        let merged_prefix_sums = col_prefix_sums
            .windows(2)
            .map(|prefix_sums| {
                let mut merged_prefix_sum = prefix_sums[0].clone();
                merged_prefix_sum.extend(&prefix_sums[1]);
                merged_prefix_sum
            })
            .collect_vec();

        // Generate all of the z_col partial lagrange mle.
        let z_col_partial_lagrange = Mle::blocking_partial_lagrange(&z_col);

        // Condense the merged_prefix_sums and z_col_eq_vals for empty tables.
        let (merged_prefix_sums, z_col_eq_vals): (Vec<Point<F>>, Vec<EF>) = merged_prefix_sums
            .iter()
            .zip(z_col_partial_lagrange.guts().as_slice())
            .group_by(|(merged_prefix_sum, _)| *merged_prefix_sum)
            .into_iter()
            .map(|(merged_prefix_sum, group)| {
                let group_elements =
                    group.into_iter().map(|(_, z_col_eq_val)| *z_col_eq_val).collect_vec();
                (merged_prefix_sum.clone(), group_elements.into_iter().sum::<EF>())
            })
            .unzip();

        let merged_prefix_sums_len = merged_prefix_sums.len();
        assert!(merged_prefix_sums_len == z_col_eq_vals.len());

        let merged_prefix_sums = Arc::new(merged_prefix_sums);

        let half = EF::two().inverse();
        let bp_batch_eval =
            BPE::new(z_row, z_index, merged_prefix_sums.clone(), z_col_eq_vals.clone()).await;

        Self {
            bp_batch_eval,
            rho: Point::new(Buffer::default()),
            z_col,
            merged_prefix_sums,
            round_num: 0,
            z_col_eq_vals,
            intermediate_eq_full_evals: vec![EF::one(); merged_prefix_sums_len],
            half,
            _marker: PhantomData,
        }
    }
}

impl<
        F: Field,
        EF: ExtensionField<F>,
        BPE: JaggedAssistSumAsPoly<F, EF, A> + Send + Sync,
        A: Backend,
    > SumcheckPolyBase for JaggedEvalSumcheckPoly<F, EF, BPE, A>
{
    fn num_variables(&self) -> u32 {
        self.merged_prefix_sums.first().unwrap().dimension() as u32
    }
}

impl<
        F: Field,
        EF: ExtensionField<F>,
        BPE: JaggedAssistSumAsPoly<F, EF, A> + Send + Sync,
        A: Backend,
    > ComponentPoly<EF> for JaggedEvalSumcheckPoly<F, EF, BPE, A>
{
    async fn get_component_poly_evals(&self) -> Vec<EF> {
        Vec::new()
    }
}

impl<
        F: Field,
        EF: ExtensionField<F>,
        BPE: JaggedAssistSumAsPoly<F, EF, A> + Send + Sync,
        A: Backend,
    > SumcheckPolyFirstRound<EF> for JaggedEvalSumcheckPoly<F, EF, BPE, A>
{
    type NextRoundPoly = JaggedEvalSumcheckPoly<F, EF, BPE, A>;

    async fn fix_t_variables(self, alpha: EF, _t: usize) -> Self::NextRoundPoly {
        self.fix_last_variable(alpha).await
    }

    async fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert!(t == 1);
        self.sum_as_poly_in_last_variable(claim).await
    }
}

impl<
        F: Field,
        EF: ExtensionField<F>,
        BPE: JaggedAssistSumAsPoly<F, EF, A> + Send + Sync,
        A: Backend,
    > SumcheckPoly<EF> for JaggedEvalSumcheckPoly<F, EF, BPE, A>
{
    async fn fix_last_variable(self, alpha: EF) -> Self {
        // Add alpha to the rho point.
        let mut rho = self.rho.clone();
        rho.add_dimension(alpha);

        let merged_prefix_sum_dim = self.merged_prefix_sums[0].dimension();

        // Update the intermediate full eq evals.
        let updated_intermediate_eq_full_evals = self
            .merged_prefix_sums
            .iter()
            .zip_eq(self.intermediate_eq_full_evals.iter())
            .map(|(merged_prefix_sum, intermediate_eq_full_eval)| {
                let x_i = merged_prefix_sum
                    .values()
                    .get(merged_prefix_sum_dim - 1 - self.round_num)
                    .unwrap();
                *intermediate_eq_full_eval
                    * ((alpha * *x_i) + (EF::one() - alpha) * (EF::one() - *x_i))
            })
            .collect_vec();

        Self {
            bp_batch_eval: self.bp_batch_eval,
            rho,
            merged_prefix_sums: self.merged_prefix_sums,
            z_col: self.z_col,
            round_num: self.round_num + 1,
            z_col_eq_vals: self.z_col_eq_vals,
            intermediate_eq_full_evals: updated_intermediate_eq_full_evals,
            half: self.half,
            _marker: PhantomData,
        }
    }

    async fn sum_as_poly_in_last_variable(&self, claim: Option<EF>) -> UnivariatePolynomial<EF> {
        let (y_0, y_half) = self
            .bp_batch_eval
            .sum_as_poly(
                self.round_num,
                &self.z_col_eq_vals,
                &self.intermediate_eq_full_evals,
                &self.rho,
            )
            .await;

        // Infer the value at x = 1 from the claim.
        let y_1 = claim.unwrap() - y_0;

        interpolate_univariate_polynomial(&[EF::zero(), EF::one(), self.half], &[y_0, y_1, y_half])
    }
}

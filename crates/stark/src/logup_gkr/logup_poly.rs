use itertools::izip;
use rayon::prelude::*;
use slop_algebra::{ExtensionField, Field};

use slop_algebra::{interpolate_univariate_polynomial, UnivariatePolynomial};
use slop_alloc::{Buffer, CanCopyFrom, CpuBackend, HasBackend};
use slop_multilinear::{Mle, MleBaseBackend, PaddedMle, Point, PointBackend};
use slop_sumcheck::{
    ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend, SumcheckPolyBackend, SumcheckPolyBase,
};
use slop_tensor::Tensor;

/// An error that occurs during the verification.
/// Polynomial for each GKR round.
#[derive(Debug, Clone)]
pub struct LogupGkrPoly<
    NumeratorType: Field,
    K: ExtensionField<NumeratorType>,
    B: PointBackend<K> = CpuBackend,
> {
    /// The random challenge point at which the polynomial is evaluated.
    pub point: Point<K>,
    /// The MLE for the numerator polynomial with last var fixed to 0.
    pub numerator_0: PaddedMle<NumeratorType, B>,
    /// The MLE for the numerator polynomial with last var fixed to 1.
    pub numerator_1: PaddedMle<NumeratorType, B>,
    /// The MLE for the denominator polynomial with last var fixed to 0.
    pub denom_0: PaddedMle<K, B>,
    /// The MLE for the denominator polynomial with last var fixed to 1.
    pub denom_1: PaddedMle<K, B>,
    /// The lambda value for combining the numerator and denominator polynomials.
    pub lambda: K,
    /// The adjustment factor from the constant part of the eq polynomial.
    pub eq_adjustment: K,
    /// The inverse of 8.
    pub one_eighth: K,
    /// The inverse of 2.
    pub one_half: K,
    /// The randomness to batch the sumchecks across the interactions.
    pub batching_randomness: K,
    /// Precomputed powers of the batching randomness.
    pub batching_randomness_powers: Vec<K>,
    /// Precomputed powers of the batching randomness, on device.
    pub batching_randomness_powers_device: Tensor<K, B>,
    /// The sum of the `batching_randomness_powers`. Used for adjustments to the sums from padding.
    pub batching_randomness_powers_sum: K,
}

impl<
        NumeratorType: Field,
        K: ExtensionField<NumeratorType>,
        B: PointBackend<K> + MleBaseBackend<NumeratorType>,
    > HasBackend for LogupGkrPoly<NumeratorType, K, B>
{
    type Backend = B;

    fn backend(&self) -> &Self::Backend {
        self.numerator_0.padding_values().backend()
    }
}

impl<
        NumeratorType: Field,
        K: ExtensionField<NumeratorType>,
        B: PointBackend<K>
            + CanCopyFrom<Buffer<K>, CpuBackend, Output = Buffer<K, B>>
            + MleBaseBackend<K>
            + MleBaseBackend<NumeratorType>,
    > LogupGkrPoly<NumeratorType, K, B>
{
    /// Creates a new `LogupGkrPoly`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_batching_randomness_powers(
        point: Point<K>,
        numerator_0: PaddedMle<NumeratorType, B>,
        numerator_1: PaddedMle<NumeratorType, B>,
        denom_0: PaddedMle<K, B>,
        denom_1: PaddedMle<K, B>,
        lambda: K,
        eq_adjustment: K,
        batching_randomness: K,
        batching_randomness_powers: Vec<K>,
        batching_randomness_powers_device: Tensor<K, B>,
        batching_randomness_powers_sum: K,
    ) -> Self {
        assert_eq!(numerator_0.num_variables(), numerator_1.num_variables());
        assert_eq!(numerator_0.num_variables(), denom_0.num_variables());
        assert_eq!(numerator_0.num_variables(), denom_1.num_variables());
        let one_eighth = K::one() / K::from_canonical_usize(8);
        let one_half = K::one() / K::from_canonical_usize(2);

        Self {
            point,
            numerator_0,
            numerator_1,
            denom_0,
            denom_1,
            lambda,
            eq_adjustment,
            one_eighth,
            one_half,
            batching_randomness,
            batching_randomness_powers,
            batching_randomness_powers_device,
            batching_randomness_powers_sum,
        }
    }

    /// Creates a new `LogupGkrPoly`, computing once and for all the powers of the batching randomness.
    ///
    /// In subsequent iterations of sumcheck, `new_with_batching_randomness_powers` should be used.
    pub async fn new(
        point: Point<K>,
        numerators: (PaddedMle<NumeratorType, B>, PaddedMle<NumeratorType, B>),
        denoms: (PaddedMle<K, B>, PaddedMle<K, B>),
        lambda: K,
        eq_adjustment: K,
        batching_randomness: K,
    ) -> Self {
        let (numerator_0, numerator_1) = numerators;
        let (denom_0, denom_1) = denoms;

        let num_interactions = numerator_0.num_polynomials();
        let (
            batching_randomness_powers,
            batching_randomness_powers_device,
            batching_randomness_powers_sum,
        ) = if numerator_0.inner().is_some() {
            let batching_randomness_powers: Vec<_> =
                batching_randomness.powers().take(num_interactions).collect();
            let batching_randomness_powers_device: Tensor<K, B> =
                <B as CanCopyFrom<Buffer<K>, CpuBackend>>::copy_into(
                    numerator_0.backend(),
                    Buffer::from(batching_randomness_powers.clone()),
                )
                .await
                .unwrap()
                .into();

            let batching_randomness_powers_sum = batching_randomness_powers.iter().copied().sum();

            (
                batching_randomness_powers,
                batching_randomness_powers_device,
                batching_randomness_powers_sum,
            )
        } else {
            (
                vec![],
                <B as CanCopyFrom<Buffer<K>, CpuBackend>>::copy_into(
                    numerator_0.backend(),
                    Buffer::from(vec![]),
                )
                .await
                .unwrap()
                .into(),
                (K::one() - batching_randomness.exp_u64(num_interactions as u64))
                    / (K::one() - batching_randomness),
            )
        };

        Self::new_with_batching_randomness_powers(
            point,
            numerator_0,
            numerator_1,
            denom_0,
            denom_1,
            lambda,
            eq_adjustment,
            batching_randomness,
            batching_randomness_powers,
            batching_randomness_powers_device,
            batching_randomness_powers_sum,
        )
    }
}

impl<NumeratorType: Field, K: ExtensionField<NumeratorType>> LogupGkrPoly<NumeratorType, K> {
    fn iter_or_default<T: Copy>(
        slice: &[T],
        num_entries: usize,
        default_value: T,
    ) -> impl Iterator<Item = T> + '_ {
        slice
            .iter()
            .copied()
            .skip(num_entries)
            .chain(std::iter::repeat(default_value).take(num_entries))
    }
    #[allow(clippy::too_many_lines)]
    fn pair_sums(&self, eq_guts: &[K]) -> Vec<(K, K)> {
        assert!(self.numerator_0.inner().is_some());
        assert!(self.numerator_1.inner().is_some());
        assert!(self.denom_0.inner().is_some());
        assert!(self.denom_1.inner().is_some());
        let num_polynomials = self.numerator_0.num_polynomials();
        let num_rows = self.numerator_0.num_real_entries();
        let eq_guts_iter = eq_guts.par_iter().take(num_rows.div_ceil(2));
        let numerator_0_iter = self
            .numerator_0
            .inner()
            .as_ref()
            .unwrap()
            .guts()
            .as_slice()
            .par_chunks(2 * num_polynomials);
        let numerator_1_iter = self
            .numerator_1
            .inner()
            .as_ref()
            .unwrap()
            .guts()
            .as_slice()
            .par_chunks(2 * num_polynomials);
        let denom_0_iter = self
            .denom_0
            .inner()
            .as_ref()
            .unwrap()
            .guts()
            .as_slice()
            .par_chunks(2 * num_polynomials);
        let denom_1_iter = self
            .denom_1
            .inner()
            .as_ref()
            .unwrap()
            .guts()
            .as_slice()
            .par_chunks(2 * num_polynomials);
        eq_guts_iter
            .zip(numerator_0_iter)
            .zip(numerator_1_iter)
            .zip(denom_0_iter)
            .zip(denom_1_iter)
            .map(
                |((((eq, numerator_0_chunk), numerator_1_chunk), denom_0_chunk), denom_1_chunk)| {
                    izip!(
                        numerator_0_chunk.iter().copied().take(num_polynomials),
                        Self::iter_or_default(
                            numerator_0_chunk,
                            num_polynomials,
                            NumeratorType::zero()
                        ),
                        numerator_1_chunk.iter().copied().take(num_polynomials),
                        Self::iter_or_default(
                            numerator_1_chunk,
                            num_polynomials,
                            NumeratorType::zero()
                        ),
                        denom_0_chunk.iter().copied().take(num_polynomials),
                        Self::iter_or_default(denom_0_chunk, num_polynomials, K::one()),
                        denom_1_chunk.iter().copied().take(num_polynomials),
                        Self::iter_or_default(denom_1_chunk, num_polynomials, K::one())
                    )
                    .map(
                        |(
                            numerator_0_0,
                            numerator_0_1,
                            numerator_1_0,
                            numerator_1_1,
                            denom_0_0,
                            denom_0_1,
                            denom_1_0,
                            denom_1_1,
                        )| {
                            let denom_0_half = denom_0_1 + denom_0_0;
                            let denom_1_half = denom_1_1 + denom_1_0;
                            let numerator_0_half = numerator_0_1 + numerator_0_0;
                            let numerator_1_half = numerator_1_1 + numerator_1_0;

                            (
                                *eq * ((denom_0_0 * (self.lambda * denom_1_0 + numerator_1_0))
                                    + denom_1_0 * numerator_0_0),
                                *eq * ((denom_0_half
                                    * (self.lambda * denom_1_half + numerator_1_half))
                                    + denom_1_half * numerator_0_half),
                            )
                        },
                    )
                    .collect::<Vec<_>>()
                },
            )
            .reduce(
                || vec![(K::zero(), K::zero()); num_polynomials],
                |accum, pairs| {
                    accum
                        .iter()
                        .copied()
                        .zip(pairs.iter().copied())
                        .map(|((accum_0, accum_1), (y_0, y_half))| {
                            (accum_0 + y_0, accum_1 + y_half)
                        })
                        .collect()
                },
            )
    }

    /// The sumcheck sum function for the CPU implementation of the `LogupGkr` polynomial.
    pub async fn sum_as_poly_in_last_variable(&self, claim: Option<K>) -> UnivariatePolynomial<K> {
        assert!(self.num_variables() > 0);
        let claim = claim.expect("claim must be provided");

        let mut rest = self.point.clone();

        let last_coordinate = rest.remove_last_coordinate();
        let one_minus_first_coord = K::one() - last_coordinate;

        let mut xs = Vec::new();
        let mut ys = Vec::new();

        let num_rows = self.numerator_0.num_real_entries();

        let num_polynomials = self.numerator_0.num_polynomials();
        let denom_polynomials = self.denom_0.num_polynomials();
        assert!(num_polynomials == denom_polynomials);

        let ((mut y_0, mut y_half), eq_guts_sum) = match self.numerator_0.inner() {
            Some(_) => {
                // TODO: Don't compute such a large MLE, only need to compute the part corresponding
                // to the "real variables".
                let partial_lagrange: Mle<K> = Mle::partial_lagrange(&rest).await;
                let eq_guts = partial_lagrange.guts().as_slice();
                let pair_sums = self.pair_sums(eq_guts);
                let (y_0, y_half) = pair_sums
                    .iter()
                    .copied()
                    .zip(self.batching_randomness_powers.iter().copied())
                    .fold((K::zero(), K::zero()), |(accum_0, accum_1), ((y_0, y_half), power)| {
                        (accum_0 + power * y_0, accum_1 + power * y_half)
                    });

                ((y_0, y_half), eq_guts.par_iter().copied().take(num_rows.div_ceil(2)).sum::<K>())
            }
            None => ((K::zero(), K::zero()), K::zero()),
        };

        // TODO: A lot of this logic is common to the GPU implementation, and should be refactored.
        let half_point = (0..rest.dimension()).map(|_| K::two().inverse()).collect::<Point<_>>();
        let powers_sum = self.batching_randomness_powers_sum;
        let rest_eq_guts_sum = Mle::full_lagrange_eval(&rest, &half_point)
            * NumeratorType::from_canonical_u32(1 << rest.dimension())
            - eq_guts_sum;
        y_0 += rest_eq_guts_sum * powers_sum * self.lambda;

        y_half += rest_eq_guts_sum * K::from_canonical_u32(4) * powers_sum * self.lambda;

        xs.push(K::zero());
        y_0 *= one_minus_first_coord;
        ys.push(y_0);

        y_half *= self.one_eighth;
        xs.push(self.one_half);
        ys.push(y_half);

        let y_1 = (claim / self.eq_adjustment) - y_0;
        xs.push(K::one());
        ys.push(y_1);

        let point_elements = self.point.to_vec();
        let point_last = point_elements.last().unwrap();
        let b_const = (K::one() - *point_last) / (K::one() - point_last.double());
        xs.push(b_const);
        ys.push(K::zero());

        let ys = ys.iter().map(|y| *y * self.eq_adjustment).collect::<Vec<_>>();

        interpolate_univariate_polynomial(&xs, &ys)
    }

    /// The fix last variable implementation for the GKR polynomial. Essentially fixes the last
    /// the last variable for all of the constituent polynomials.
    pub async fn fix_last_variable(self, alpha: K) -> LogupGkrPoly<K, K> {
        let mut new_point = self.point.clone();
        let last_coord = new_point.remove_last_coordinate();
        let prod = alpha * last_coord;

        let eq_adjustment = self.eq_adjustment * (K::one() - alpha - last_coord + prod + prod);

        let numerator_0 = self.numerator_0.fix_last_variable(alpha).await;
        let numerator_1 = self.numerator_1.fix_last_variable(alpha).await;
        let denom_0 = self.denom_0.fix_last_variable(alpha).await;
        let denom_1 = self.denom_1.fix_last_variable(alpha).await;

        LogupGkrPoly::<K, K>::new(
            new_point,
            (numerator_0, numerator_1),
            (denom_0, denom_1),
            self.lambda,
            eq_adjustment,
            self.batching_randomness,
        )
        .await
    }
}

impl<
        N: Field,
        K: ExtensionField<N>,
        B: PointBackend<K> + MleBaseBackend<K> + MleBaseBackend<N>,
    > SumcheckPolyBase for LogupGkrPoly<N, K, B>
{
    fn num_variables(&self) -> u32 {
        assert!(self.numerator_0.num_variables() == self.numerator_1.num_variables());
        assert!(self.numerator_0.num_variables() == self.denom_0.num_variables());
        assert!(self.numerator_0.num_variables() == self.denom_1.num_variables());
        self.numerator_0.num_variables()
    }
}

impl<N, K> ComponentPolyEvalBackend<LogupGkrPoly<N, K>, K> for CpuBackend
where
    N: Field,
    K: ExtensionField<N>,
{
    async fn get_component_poly_evals(poly: &LogupGkrPoly<N, K>) -> Vec<K> {
        assert!(poly.num_variables() == 0);
        izip!(
            poly.numerator_0
                .inner()
                .as_ref()
                .map(|x| x.guts().as_slice().iter())
                .unwrap_or(vec![N::zero(); poly.numerator_0.num_polynomials()].as_slice().iter()),
            poly.numerator_1
                .inner()
                .as_ref()
                .map(|x| x.guts().as_slice().iter())
                .unwrap_or(vec![N::zero(); poly.numerator_1.num_polynomials()].as_slice().iter()),
            poly.denom_0
                .inner()
                .as_ref()
                .map(|x| x.guts().as_slice().iter())
                .unwrap_or(vec![K::one(); poly.denom_0.num_polynomials()].as_slice().iter()),
            poly.denom_1
                .inner()
                .as_ref()
                .map(|x| x.guts().as_slice().iter())
                .unwrap_or(vec![K::one(); poly.denom_1.num_polynomials()].as_slice().iter()),
        )
        .flat_map(|(n0, n1, d0, d1)| [K::from(*n0), K::from(*n1), *d0, *d1].into_iter())
        .collect()
    }
}

impl<K: ExtensionField<NumeratorType>, NumeratorType: Field>
    SumCheckPolyFirstRoundBackend<LogupGkrPoly<NumeratorType, K>, K> for CpuBackend
{
    type NextRoundPoly = LogupGkrPoly<K, K>;
    async fn fix_t_variables(
        poly: LogupGkrPoly<NumeratorType, K>,
        alpha: K,
        t: usize,
    ) -> Self::NextRoundPoly {
        assert!(t == 1);
        poly.fix_last_variable(alpha).await
    }

    async fn sum_as_poly_in_last_t_variables(
        poly: &LogupGkrPoly<NumeratorType, K>,
        claim: Option<K>,
        t: usize,
    ) -> UnivariatePolynomial<K> {
        assert!(t == 1);
        poly.sum_as_poly_in_last_variable(claim).await
    }
}

impl<EF: Field> SumcheckPolyBackend<LogupGkrPoly<EF, EF>, EF> for CpuBackend {
    async fn fix_last_variable(poly: LogupGkrPoly<EF, EF>, alpha: EF) -> LogupGkrPoly<EF, EF> {
        LogupGkrPoly::fix_last_variable(poly, alpha).await
    }

    async fn sum_as_poly_in_last_variable(
        poly: &LogupGkrPoly<EF, EF>,
        claim: Option<EF>,
    ) -> UnivariatePolynomial<EF> {
        LogupGkrPoly::sum_as_poly_in_last_variable(poly, claim).await
    }
}

use itertools::Itertools;

use slop_algebra::{rlc_univariate_polynomials, ExtensionField, Field, UnivariatePolynomial};
use slop_challenger::{CanObserve, CanSample};
use slop_multilinear::Point;

use slop_sumcheck::{PartialSumcheckProof, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound};

/// Proves a sumcheck for any sumcheckable polynomial, by reducing it to a claim about the
/// evaluation of the polynomial at a point.
///
///  # Panics
///  Will panic if the polynomial has zero variables.
pub fn reduce_sumcheck_to_evaluation<
    F: Field,
    EF: ExtensionField<F>,
    Challenger: CanSample<EF> + CanObserve<F>,
>(
    polys: Vec<impl SumcheckPolyFirstRound<EF>>,
    challenger: &mut Challenger,
    claims: Vec<EF>,
    t: usize,
    lambda: EF,
) -> (PartialSumcheckProof<EF>, Vec<Vec<EF>>) {
    assert!(!polys.is_empty());
    // Check that all the polynomials have the same number of variables.

    let n_variables = polys[0].n_variables();

    // Check that all the polynomials have the same number of variables.
    assert!(polys.iter().all(|poly| poly.n_variables() == n_variables));

    // The first round will process the first t variables, so we need to ensure that there are at least t variables.
    assert!(n_variables >= t);

    // The point at which the reduced sumcheck proof should be evaluated.
    let mut point = vec![];

    // The univariate poly messages.  This will be a rlc of the polys' univariate polys.
    let mut univariate_poly_msgs: Vec<UnivariatePolynomial<EF>> = vec![];

    let mut uni_polys = tracing::debug_span!("sum_as_poly_in_first_variable first round")
        .in_scope(|| {
            polys
                .iter()
                .zip(claims.iter())
                .map(|(poly, claim)| poly.sum_as_poly_in_last_t_variables(Some(*claim), t))
        })
        .collect_vec();
    let mut rlc_uni_poly = rlc_univariate_polynomials(&uni_polys, lambda);
    let coefficients =
        rlc_uni_poly.coefficients.iter().flat_map(|x| x.as_base_slice()).copied().collect_vec();
    challenger.observe_slice(&coefficients);

    univariate_poly_msgs.push(rlc_uni_poly);

    let alpha: EF = challenger.sample();
    point.insert(0, alpha);
    let mut polys_cursor = tracing::debug_span!("fix_first_variable first round")
        .in_scope(|| polys.into_iter().map(|poly| poly.fix_t_variables(alpha, t)))
        .collect_vec();

    // The multi-variate polynomial used at the start of each sumcheck round.
    for _ in t..n_variables {
        // Get the round claims from the last round's univariate poly messages.
        let round_claims = uni_polys.iter().map(|poly| poly.eval_at_point(*point.first().unwrap()));

        uni_polys = tracing::debug_span!("sum_as_poly_in_first_variable")
            .in_scope(|| {
                polys_cursor
                    .iter()
                    .zip_eq(round_claims)
                    .map(|(poly, round_claim)| poly.sum_as_poly_in_last_variable(Some(round_claim)))
            })
            .collect_vec();
        rlc_uni_poly = rlc_univariate_polynomials(&uni_polys, lambda);
        challenger.observe_slice(
            &rlc_uni_poly
                .coefficients
                .iter()
                .flat_map(|x| x.as_base_slice())
                .copied()
                .collect::<Vec<_>>(),
        );

        univariate_poly_msgs.push(rlc_uni_poly);

        let alpha: EF = challenger.sample();
        point.insert(0, alpha);
        polys_cursor = tracing::debug_span!("fix_first_variable").in_scope(|| {
            polys_cursor.into_iter().map(|poly| poly.fix_last_variable(alpha)).collect_vec()
        });
    }

    let evals = tracing::debug_span!("eval_at_point")
        .in_scope(|| uni_polys.iter().map(|poly| poly.eval_at_point(*point.first().unwrap())))
        .collect_vec();

    let component_poly_evals =
        polys_cursor.iter().map(|poly| poly.get_component_poly_evals()).collect_vec();

    (
        PartialSumcheckProof {
            univariate_polys: univariate_poly_msgs,
            claimed_sum: claims.into_iter().fold(EF::zero(), |acc, x| acc * lambda + x),
            point_and_eval: (
                Point::new(point),
                evals.into_iter().fold(EF::zero(), |acc, x| acc * lambda + x),
            ),
        },
        component_poly_evals,
    )
}

#[cfg(test)]
mod tests {
    use slop_baby_bear::{BabyBear, DiffusionMatrixBabyBear};

    use rand::{thread_rng, Rng};
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_challenger::{CanSample, DuplexChallenger};
    use slop_multilinear::{partial_lagrange_eval, Mle, Point};
    use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use slop_sumcheck::{partially_verify_sumcheck_proof, SumcheckError};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type Challenger = DuplexChallenger<BabyBear, Perm, 16, 8>;

    #[test]
    fn test_reduce_sumcheck_to_evaluation() {
        let guts: Mle<F> = vec![
            F::from_canonical_u32(2),
            F::from_canonical_u32(3),
            F::from_canonical_u32(2),
            F::from_canonical_u32(3),
            F::from_canonical_u32(2),
            F::from_canonical_u32(3),
            F::from_canonical_u32(2),
            F::from_canonical_u32(3),
        ]
        .into();
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut thread_rng(),
        );
        let mut challenger = Challenger::new(perm.clone());
        let lambda = challenger.sample();

        let claim: EF = guts.guts.iter().map(|x| EF::from(*x)).sum();
        let (proof, _) = super::reduce_sumcheck_to_evaluation::<F, EF, _>(
            vec![guts],
            &mut challenger,
            vec![claim],
            1,
            lambda,
        );

        let mut challenger = Challenger::new(perm.clone());
        let _lambda: EF = challenger.sample();
        assert!(partially_verify_sumcheck_proof::<F, EF, _>(&proof, &mut challenger).is_ok());

        assert_eq!(proof.univariate_polys.len(), 3);
        assert_eq!(proof.claimed_sum, EF::from_canonical_u32(20));
    }

    #[test]
    fn valid_sumcheck_tests() {
        let max_num_variables = 23;
        for _ in 0..100 {
            let perm = Perm::new_from_rng_128(
                Poseidon2ExternalMatrixGeneral,
                DiffusionMatrixBabyBear,
                &mut thread_rng(),
            );

            for num_variables in 1..max_num_variables {
                let guts: Mle<F> =
                    (0..1 << num_variables).map(|_| thread_rng().gen()).collect::<Vec<_>>().into();
                let claim = guts.guts.iter().map(|x| EF::from(*x)).sum();

                let mut challenger = Challenger::new(perm.clone());
                let lambda = challenger.sample();

                let (proof, _) = crate::reduce_sumcheck_to_evaluation::<F, EF, _>(
                    vec![guts],
                    &mut challenger,
                    vec![claim],
                    1,
                    lambda,
                );

                let mut challenger = Challenger::new(perm.clone());
                let _lambda: EF = challenger.sample();
                assert!(
                    partially_verify_sumcheck_proof::<F, EF, _>(&proof, &mut challenger).is_ok()
                );
                assert_eq!(proof.univariate_polys.len(), num_variables);
            }
        }
    }

    #[test]
    fn invalid_sumcheck_tests() {
        let max_num_variables = 21;
        for _ in 0..100 {
            let perm = Perm::new_from_rng_128(
                Poseidon2ExternalMatrixGeneral,
                DiffusionMatrixBabyBear,
                &mut thread_rng(),
            );
            for num_variables in 1..max_num_variables {
                let guts: Mle<F> =
                    (0..1 << num_variables).map(|_| thread_rng().gen()).collect::<Vec<_>>().into();
                let mut challenger = Challenger::new(perm.clone());
                let lambda = challenger.sample();
                let claim: F = guts.guts.iter().copied().sum();
                let (mut proof, _) = crate::reduce_sumcheck_to_evaluation(
                    vec![guts],
                    &mut challenger,
                    vec![claim],
                    1,
                    lambda,
                );
                assert_eq!(proof.univariate_polys.len(), num_variables);

                assert!(partially_verify_sumcheck_proof(&proof, &mut challenger).is_err());

                proof.claimed_sum += F::one();

                let mut challenger = Challenger::new(perm.clone());
                let _lambda: EF = challenger.sample();

                assert_eq!(
                    partially_verify_sumcheck_proof(&proof, &mut challenger),
                    Err(SumcheckError::InconsistencyWithClaimedSum)
                );

                proof.claimed_sum -= F::one();
                proof.point_and_eval.1 += F::one();

                let mut challenger = Challenger::new(perm.clone());
                let _lambda: EF = challenger.sample();

                assert_eq!(
                    partially_verify_sumcheck_proof(&proof, &mut challenger),
                    Err(SumcheckError::InconsistencyWithEval)
                );

                proof.point_and_eval.1 -= F::one();

                if num_variables > 1 {
                    proof.univariate_polys[1].coefficients[0] += F::one();
                    let mut challenger = Challenger::new(perm.clone());
                    let _lambda: EF = challenger.sample();

                    assert_eq!(
                        partially_verify_sumcheck_proof(&proof, &mut challenger),
                        Err(SumcheckError::SumcheckRoundInconsistency)
                    );
                }
            }
        }
    }

    #[test]
    fn test_partial_lagrange_fix_first_variable() {
        let num_variables = 3;
        let mut rng = thread_rng();

        let point: Point<F> = Point::new((0..num_variables).map(|_| rng.gen()).collect());

        let partial_lagrange: Mle<_> = partial_lagrange_eval(&point).into();

        let (rest, last) = point.split_at_last();

        let smaller_partial_lagrange: Mle<_> = partial_lagrange_eval(&rest).into();

        let random_coefficient: F = rng.gen();

        assert_eq!(
            partial_lagrange.sc_fix_last_variable(random_coefficient),
            smaller_partial_lagrange
                * (random_coefficient * last + (F::one() - random_coefficient) * (F::one() - last))
        );
    }
}

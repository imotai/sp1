use itertools::Itertools;

use slop_challenger::{CanObserve, CanSample};
// TODO: Remove or replace with a more principled approach before going to production.
// use p3_util::{
//     enable_bb_ops_counting, enable_ext_ops_counting, enable_permute_counting,
//     get_bb_ops_invocation_count, get_ext_ops_invocation_count, get_permute_invocation_count,
// };
use slop_algebra::{ExtensionField, Field, UnivariatePolynomial};
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
    poly: impl SumcheckPolyFirstRound<EF>,
    challenger: &mut Challenger,
    claim: EF,
) -> (PartialSumcheckProof<EF>, Vec<EF>) {
    let n_vars = poly.n_variables();
    assert!(n_vars > 0);

    // The point at which the reduced sumcheck proof should be evaluated.
    let mut point = vec![];
    // The univariate poly messages.
    let mut univariate_polys: Vec<UnivariatePolynomial<EF>> = vec![];

    let uni_poly_message = tracing::debug_span!("sum_as_poly_in_first_variable first round")
        .in_scope(|| poly.sum_as_poly_in_first_variable(Some(claim)));
    univariate_polys.push(uni_poly_message.clone());

    challenger.observe_slice(
        &uni_poly_message
            .coefficients
            .iter()
            .flat_map(|x| x.as_base_slice())
            .copied()
            .collect_vec(),
    );

    let alpha: EF = challenger.sample();
    point.push(alpha);
    let mut poly_cursor = tracing::debug_span!("fix_first_variable first round")
        .in_scope(|| poly.fix_first_variable(alpha));

    // The multi-variate polynomial used at the start of each sumcheck round.
    for i in 1..n_vars {
        let round_claim = univariate_polys[i - 1].eval_at_point(*point.last().unwrap());

        let uni_poly_message = tracing::debug_span!("sum_as_poly_in_first_variable")
            .in_scope(|| poly_cursor.sum_as_poly_in_first_variable(Some(round_claim)));
        univariate_polys.push(uni_poly_message.clone());

        challenger.observe_slice(
            &uni_poly_message
                .coefficients
                .iter()
                .flat_map(|x| x.as_base_slice())
                .copied()
                .collect::<Vec<_>>(),
        );

        let alpha: EF = challenger.sample();
        point.push(alpha);
        poly_cursor = tracing::debug_span!("fix_first_variable")
            .in_scope(|| poly_cursor.fix_first_variable(alpha));
    }

    let eval = tracing::debug_span!("eval_at_point")
        .in_scope(|| univariate_polys.last().unwrap().eval_at_point(*point.last().unwrap()));

    // if output {
    //     let (
    //         ext_add,
    //         ext_add_field,
    //         ext_sub,
    //         ext_sub_field,
    //         ext_mul,
    //         ext_mul_field,
    //         ext_div,
    //         ext_div_field,
    //     ) = get_ext_ops_invocation_count();
    //     let (bb_add, bb_sub, bb_mul, bb_div) = get_bb_ops_invocation_count();
    //     let permute_count = get_permute_invocation_count();

    //     println!(
    //         "ext add ops = {ext_add}, ext add field = {ext_add_field}, ext sub ops = {ext_sub},
    // ext sub field = {ext_sub_field}, ext mul ops = {ext_mul}, ext mul field = {ext_mul_field},
    // ext div ops = {ext_div}, ext div field = {ext_div_field}, bb add ops = {bb_add}, bb sub ops =
    // {bb_sub}, bb mul ops = {bb_mul}, bb div ops = {bb_div}, permute count = {permute_count}"
    //     );
    // }

    let component_poly_evals = poly_cursor.get_component_poly_evals();

    (
        PartialSumcheckProof {
            univariate_polys,
            claimed_sum: claim,
            point_and_eval: (Point::new(point), eval),
        },
        component_poly_evals,
    )
}

#[cfg(test)]
mod tests {
    use slop_baby_bear::{BabyBear, DiffusionMatrixBabyBear};

    use rand::{thread_rng, Rng};
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_challenger::DuplexChallenger;
    use slop_multilinear::{partial_lagrange_eval, Mle, Point};
    use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use slop_sumcheck::{partially_verify_sumcheck_proof, SumcheckError, SumcheckPoly};

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

        let claim: EF = guts.guts.iter().map(|x| EF::from(*x)).sum();
        let (proof, _) =
            super::reduce_sumcheck_to_evaluation::<F, EF, _>(guts, &mut challenger, claim);

        let mut challenger = Challenger::new(perm.clone());
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

                let (proof, _) = crate::reduce_sumcheck_to_evaluation::<F, EF, _>(
                    guts,
                    &mut Challenger::new(perm.clone()),
                    claim,
                );

                let mut challenger = Challenger::new(perm.clone());
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
                let claim: F = guts.guts.iter().copied().sum();
                let (mut proof, _) =
                    crate::reduce_sumcheck_to_evaluation(guts, &mut challenger, claim);
                assert_eq!(proof.univariate_polys.len(), num_variables);

                assert!(partially_verify_sumcheck_proof(&proof, &mut challenger).is_err());

                proof.claimed_sum += F::one();

                assert_eq!(
                    partially_verify_sumcheck_proof(&proof, &mut Challenger::new(perm.clone())),
                    Err(SumcheckError::InconsistencyWithClaimedSum)
                );

                proof.claimed_sum -= F::one();
                proof.point_and_eval.1 += F::one();

                assert_eq!(
                    partially_verify_sumcheck_proof(&proof, &mut Challenger::new(perm.clone())),
                    Err(SumcheckError::InconsistencyWithEval)
                );

                proof.point_and_eval.1 -= F::one();

                if num_variables > 1 {
                    proof.univariate_polys[1].coefficients[0] += F::one();
                    assert_eq!(
                        partially_verify_sumcheck_proof(&proof, &mut Challenger::new(perm.clone())),
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

        let (first, rest) = point.split_at_first();

        let smaller_partial_lagrange: Mle<_> = partial_lagrange_eval(&rest).into();

        let random_coefficient: F = rng.gen();

        assert_eq!(
            partial_lagrange.fix_first_variable(random_coefficient),
            smaller_partial_lagrange
                * (random_coefficient * first
                    + (F::one() - random_coefficient) * (F::one() - first))
        );
    }
}

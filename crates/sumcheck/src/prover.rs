use itertools::Itertools;
use p3_challenger::{CanObserve, CanSample};
// use p3_util::{
//     enable_bb_ops_counting, enable_ext_ops_counting, enable_permute_counting,
//     get_bb_ops_invocation_count, get_ext_ops_invocation_count, get_permute_invocation_count,
// };
use spl_algebra::{ExtensionField, Field, UnivariatePolynomial};
use spl_multi_pcs::Point;

use crate::{PartialSumcheckProof, SumcheckPoly};

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
    poly: Box<dyn SumcheckPoly<EF>>,
    challenger: &mut Challenger,
    claim: EF,
) -> (PartialSumcheckProof<EF>, Box<dyn SumcheckPoly<EF>>) {
    let n_vars = poly.n_variables();
    assert!(n_vars > 0);

    // The point at which the reduced sumcheck proof should be evaluated.
    let mut point = vec![];
    // The univariate poly messages.
    let mut univariate_polys: Vec<UnivariatePolynomial<EF>> = vec![];

    // The multi-variate polynomial used at the start of each sumcheck round.
    let mut poly_cursor: Box<dyn SumcheckPoly<EF>> = poly;

    for i in 0..n_vars {
        let round_claim = if i == 0 {
            claim
        } else {
            univariate_polys[i - 1].eval_at_point(*point.last().unwrap())
        };

        let uni_poly_message = tracing::debug_span!("sum_as_poly_in_first_variable")
            .in_scope(|| poly_cursor.sum_as_poly_in_first_variable(Some(round_claim)));
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
        poly_cursor = tracing::debug_span!("fix_first_variable")
            .in_scope(|| poly_cursor.fix_first_variable(alpha));
    }

    let eval = tracing::debug_span!("eval_at_point").in_scope(|| {
        univariate_polys
            .last()
            .unwrap()
            .eval_at_point(*point.last().unwrap())
    });

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

    (
        PartialSumcheckProof {
            univariate_polys,
            claimed_sum: claim,
            point_and_eval: (Point::new(point), eval),
        },
        poly_cursor,
    )
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};

    use p3_challenger::DuplexChallenger;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use rand::thread_rng;
    use spl_algebra::{extension::BinomialExtensionField, AbstractField};
    use spl_multi_pcs::Mle;

    use crate::partially_verify_sumcheck_proof;

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
        let (proof, _) = super::reduce_sumcheck_to_evaluation::<F, EF, _>(
            Box::new(guts),
            &mut challenger,
            claim,
        );

        let mut challenger = Challenger::new(perm.clone());
        assert!(partially_verify_sumcheck_proof::<F, EF, _>(&proof, &mut challenger).is_ok());

        assert_eq!(proof.univariate_polys.len(), 3);
        assert_eq!(proof.claimed_sum, EF::from_canonical_u32(20));
    }
}

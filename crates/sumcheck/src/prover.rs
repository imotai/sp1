use itertools::Itertools;

use slop_algebra::{rlc_univariate_polynomials, ExtensionField, Field, UnivariatePolynomial};
use slop_challenger::FieldChallenger;

use crate::{ComponentPoly, PartialSumcheckProof, SumcheckPoly, SumcheckPolyFirstRound};

/// Proves a sumcheck for any sumcheckable polynomial, by reducing it to a claim about the
/// evaluation of the polynomial at a point.
///
///  # Panics
///  Will panic if the polynomial has zero variables.
pub fn reduce_sumcheck_to_evaluation<
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
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
    assert!(n_variables >= t as u32);

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

    let alpha: EF = challenger.sample_ext_element();
    point.insert(0, alpha);
    let mut polys_cursor = tracing::debug_span!("fix_first_variable first round")
        .in_scope(|| polys.into_iter().map(|poly| poly.fix_t_variables(alpha, t)))
        .collect_vec();

    // The multi-variate polynomial used at the start of each sumcheck round.
    for _ in t..n_variables as usize {
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

        let alpha: EF = challenger.sample_ext_element();
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
                point.into(),
                evals.into_iter().fold(EF::zero(), |acc, x| acc * lambda + x),
            ),
        },
        component_poly_evals,
    )
}

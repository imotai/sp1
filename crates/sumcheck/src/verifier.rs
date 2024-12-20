use itertools::Itertools;
use p3_challenger::{CanObserve, CanSample};
use spl_algebra::{ExtensionField, Field};
use spl_multilinear::Point;

use crate::PartialSumcheckProof;

#[derive(Debug, Eq, PartialEq)]
pub enum SumcheckError {
    InvalidProofShape,
    SumcheckRoundInconsistency,
    InconsistencyWithClaimedSum,
    InconsistencyWithEval,
}
/// Verifies that a PartialSumcheckProof is correct up until the evaluation claim.
pub fn partially_verify_sumcheck_proof<
    F: Field,
    EF: ExtensionField<F>,
    Challenger: CanObserve<F> + CanSample<EF>,
>(
    proof: &PartialSumcheckProof<EF>,
    challenger: &mut Challenger,
) -> Result<(), SumcheckError> {
    let num_variables = proof.univariate_polys.len();
    let mut alpha_point = Point::default();

    // Checks for the correct proof shape.
    if num_variables != proof.point_and_eval.0.dimension() {
        return Err(SumcheckError::InvalidProofShape);
    }

    // There is a way to structure a sumcheck proof so that this check is not needed, but it doesn't
    // actually save the verifier work.
    let first_poly = &proof.univariate_polys[0];
    if first_poly.eval_one_plus_eval_zero() != proof.claimed_sum {
        return Err(SumcheckError::InconsistencyWithClaimedSum);
    }

    challenger.observe_slice(
        &first_poly.coefficients.iter().flat_map(|x| x.as_base_slice()).copied().collect_vec(),
    );
    let mut previous_poly = first_poly;

    for poly in proof.univariate_polys.iter().skip(1) {
        let alpha = challenger.sample();
        alpha_point.add_dimension(alpha);
        let expected_eval = previous_poly.eval_at_point(alpha);
        if expected_eval != poly.eval_one_plus_eval_zero() {
            return Err(SumcheckError::SumcheckRoundInconsistency);
        }
        challenger.observe_slice(
            &poly.coefficients.iter().flat_map(|x| x.as_base_slice()).copied().collect_vec(),
        );
        previous_poly = poly;
    }

    let alpha = challenger.sample();
    alpha_point.add_dimension(alpha);

    // Check that the randomness generated for the prover is the same as the one obtained by the
    // verifier. There is a way to structure a sumcheck proof so that this check is not needed,
    // but it doesn't actually save the verifier work.
    if alpha_point != proof.point_and_eval.0 {
        return Err(SumcheckError::InvalidProofShape);
    }

    // Check that the evaluation claim implied by the last univariate polynomial matches the
    // evaluation claim in the proof struct.
    // There is a way to structure a sumcheck proof so that this check is not needed, but it doesn't
    // actually save the verifier work.
    if previous_poly.eval_at_point(alpha) != proof.point_and_eval.1 {
        return Err(SumcheckError::InconsistencyWithEval);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};

    use p3_challenger::DuplexChallenger;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use rand::{thread_rng, Rng};
    use spl_algebra::{extension::BinomialExtensionField, AbstractField};
    use spl_multilinear::{partial_lagrange_eval, Mle, Point};

    use crate::{partially_verify_sumcheck_proof, SumcheckPoly};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type Challenger = DuplexChallenger<BabyBear, Perm, 16, 8>;

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
                    Box::new(guts),
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
                    crate::reduce_sumcheck_to_evaluation(Box::new(guts), &mut challenger, claim);
                assert_eq!(proof.univariate_polys.len(), num_variables);

                assert!(partially_verify_sumcheck_proof(&proof, &mut challenger).is_err());

                proof.claimed_sum += F::one();

                assert_eq!(
                    partially_verify_sumcheck_proof(&proof, &mut Challenger::new(perm.clone())),
                    Err(crate::SumcheckError::InconsistencyWithClaimedSum)
                );

                proof.claimed_sum -= F::one();
                proof.point_and_eval.1 += F::one();

                assert_eq!(
                    partially_verify_sumcheck_proof(&proof, &mut Challenger::new(perm.clone())),
                    Err(crate::SumcheckError::InconsistencyWithEval)
                );

                proof.point_and_eval.1 -= F::one();

                if num_variables > 1 {
                    proof.univariate_polys[1].coefficients[0] += F::one();
                    assert_eq!(
                        partially_verify_sumcheck_proof(&proof, &mut Challenger::new(perm.clone())),
                        Err(crate::SumcheckError::SumcheckRoundInconsistency)
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
            *partial_lagrange.fix_first_variable(random_coefficient).mle()[0],
            smaller_partial_lagrange
                * (random_coefficient * first
                    + (F::one() - random_coefficient) * (F::one() - first))
        );
    }
}

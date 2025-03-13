use crate::{
    challenger::{CanObserveVariable, FieldChallengerVariable},
    BabyBearFriConfigVariable, CircuitConfig,
};
use p3_baby_bear::BabyBear;
use slop_algebra::UnivariatePolynomial;
use slop_multilinear::Point;
use slop_sumcheck::PartialSumcheckProof;
use sp1_recursion_compiler::{
    ir::Felt,
    prelude::{Builder, Ext, SymbolicExt},
};

pub fn verify_sumcheck<C: CircuitConfig<F = BabyBear>, SC: BabyBearFriConfigVariable<C>>(
    builder: &mut Builder<C>,
    challenger: &mut SC::FriChallengerVariable,
    proof: PartialSumcheckProof<Ext<C::F, C::EF>>,
) {
    let num_variables = proof.univariate_polys.len();
    let mut alpha_point: Point<SymbolicExt<C::F, C::EF>> = Point::default();

    assert_eq!(num_variables, proof.point_and_eval.0.dimension());

    let first_poly = proof.univariate_polys[0].clone();
    let first_poly_symbolic: UnivariatePolynomial<SymbolicExt<C::F, C::EF>> =
        UnivariatePolynomial {
            coefficients: first_poly
                .coefficients
                .clone()
                .into_iter()
                .map(|c| c.into())
                .collect::<Vec<_>>(),
        };
    builder.assert_ext_eq(first_poly_symbolic.eval_one_plus_eval_zero(), proof.claimed_sum);

    let coeffs: Vec<Felt<C::F>> =
        first_poly.coefficients.iter().flat_map(|x| C::ext2felt(builder, *x)).collect::<Vec<_>>();

    challenger.observe_slice(builder, coeffs);

    let mut previous_poly = first_poly_symbolic;
    for poly in proof.univariate_polys.iter().skip(1) {
        let alpha = challenger.sample_ext(builder);
        alpha_point.add_dimension(alpha.into());
        let poly_symbolic: UnivariatePolynomial<SymbolicExt<C::F, C::EF>> = UnivariatePolynomial {
            coefficients: poly
                .coefficients
                .clone()
                .into_iter()
                .map(|c| c.into())
                .collect::<Vec<_>>(),
        };
        let expected_eval = previous_poly.eval_at_point(alpha.into());
        builder.assert_ext_eq(expected_eval, poly_symbolic.eval_one_plus_eval_zero());

        let coeffs: Vec<Felt<C::F>> =
            poly.coefficients.iter().flat_map(|x| C::ext2felt(builder, *x)).collect::<Vec<_>>();
        challenger.observe_slice(builder, coeffs);
        previous_poly = poly_symbolic;
    }

    let alpha = challenger.sample_ext(builder);
    alpha_point.add_dimension(alpha.into());

    alpha_point.iter().zip(proof.point_and_eval.0.iter()).for_each(|(d, p)| {
        builder.assert_ext_eq(*d, *p);
    });

    builder.assert_ext_eq(previous_poly.eval_at_point(alpha.into()), proof.point_and_eval.1);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::challenger::DuplexChallengerVariable;
    use p3_baby_bear::DiffusionMatrixBabyBear;
    use p3_field::AbstractField;
    use rand::rngs::OsRng;
    use rand::thread_rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_algebra::AbstractExtensionField;
    use slop_challenger::DuplexChallenger;
    use slop_jagged::BabyBearPoseidon2;
    use slop_merkle_tree::{my_bb_16_perm, Perm};
    use slop_multilinear::{full_geq, Mle};
    use slop_sumcheck::reduce_sumcheck_to_evaluation;
    use sp1_recursion_compiler::{
        circuit::{AsmBuilder, AsmCompiler, AsmConfig, CircuitV2Builder},
        config::InnerConfig,
        ir::{Builder, Ext, SymbolicExt},
    };
    use sp1_recursion_executor::Runtime;
    use zkhash::ark_ff::UniformRand;

    type F = BabyBear;
    type SC = BabyBearPoseidon2;
    type C = InnerConfig;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[tokio::test]
    async fn test_sumcheck() {
        let mut rng = thread_rng();

        let mle = Mle::<BabyBear>::rand(&mut rng, 1, 10);

        let default_perm = my_bb_16_perm();
        let mut challenger = DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm.clone());

        let claim = EF::from_base(mle.guts().as_slice().iter().copied().sum::<BabyBear>());

        let (sumcheck_proof, _) = reduce_sumcheck_to_evaluation::<BabyBear, EF, _>(
            vec![mle.clone()],
            &mut challenger,
            vec![claim],
            1,
            EF::one(),
        )
        .await;

        let (point, eval_claim) = sumcheck_proof.point_and_eval.clone();
        let evaluation = mle.eval_at(&point).await[0];
        assert_eq!(evaluation, eval_claim);

        let mut builder = Builder::<C>::default();

        let proof_polys_variable: Vec<UnivariatePolynomial<Ext<F, EF>>> = sumcheck_proof
            .univariate_polys
            .iter()
            .map(|poly| UnivariatePolynomial {
                coefficients: poly
                    .coefficients
                    .iter()
                    .map(|c| builder.constant(*c))
                    .collect::<Vec<_>>(),
            })
            .collect::<Vec<_>>();
        let claimed_sum_variable: Ext<F, EF> = builder.constant(sumcheck_proof.claimed_sum);
        let point_and_eval_variable: (Point<Ext<F, EF>>, Ext<F, EF>) = (
            Point::from(
                sumcheck_proof
                    .point_and_eval
                    .0
                    .iter()
                    .copied()
                    .map(|x| builder.constant(x))
                    .collect::<Vec<_>>(),
            ),
            builder.constant(sumcheck_proof.point_and_eval.1),
        );

        let sumcheck_proof_variable = PartialSumcheckProof {
            univariate_polys: proof_polys_variable,
            claimed_sum: claimed_sum_variable,
            point_and_eval: point_and_eval_variable,
        };

        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);
        verify_sumcheck::<C, SC>(&mut builder, &mut challenger_variable, sumcheck_proof_variable);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program, my_bb_16_perm());
        runtime.run().unwrap();
    }

    #[tokio::test]
    async fn test_sumcheck_failure() {
        let mut rng = thread_rng();

        let mle = Mle::<BabyBear>::rand(&mut rng, 1, 10);

        let default_perm = my_bb_16_perm();
        let mut challenger = DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm.clone());

        let claim = EF::from_base(mle.guts().as_slice().iter().copied().sum::<BabyBear>());

        let (sumcheck_proof, _) = reduce_sumcheck_to_evaluation::<BabyBear, EF, _>(
            vec![mle.clone()],
            &mut challenger,
            vec![claim],
            1,
            EF::one(),
        )
        .await;

        let (point, eval_claim) = sumcheck_proof.point_and_eval.clone();
        let evaluation = mle.eval_at(&point).await[0];
        assert_eq!(evaluation, eval_claim);

        let mut builder = Builder::<C>::default();

        let mut proof_polys_variable: Vec<UnivariatePolynomial<Ext<F, EF>>> = sumcheck_proof
            .univariate_polys
            .iter()
            .map(|poly| UnivariatePolynomial {
                coefficients: poly
                    .coefficients
                    .iter()
                    .map(|c| builder.constant(*c))
                    .collect::<Vec<_>>(),
            })
            .collect::<Vec<_>>();

        // Modify the first polynomial to make the sumcheck fail
        proof_polys_variable[0].coefficients[0] = builder.constant(EF::one());

        let claimed_sum_variable: Ext<F, EF> = builder.constant(sumcheck_proof.claimed_sum);
        let point_and_eval_variable: (Point<Ext<F, EF>>, Ext<F, EF>) = (
            Point::from(
                sumcheck_proof
                    .point_and_eval
                    .0
                    .iter()
                    .copied()
                    .map(|x| builder.constant(x))
                    .collect::<Vec<_>>(),
            ),
            builder.constant(sumcheck_proof.point_and_eval.1),
        );

        let sumcheck_proof_variable = PartialSumcheckProof {
            univariate_polys: proof_polys_variable,
            claimed_sum: claimed_sum_variable,
            point_and_eval: point_and_eval_variable,
        };

        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);
        verify_sumcheck::<C, SC>(&mut builder, &mut challenger_variable, sumcheck_proof_variable);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program, my_bb_16_perm());
        runtime.run().expect_err("Sumcheck should fail");
    }

    #[test]
    fn test_eval_at_point() {
        let mut rng = OsRng;
        let mut builder = AsmBuilder::<F, EF>::default();
        let exts = builder.hint_exts_v2(3);
        let point = builder.hint_ext_v2();
        let univariate_poly =
            UnivariatePolynomial { coefficients: vec![exts[0], exts[1], exts[2]] };
        let univariate_poly_symbolic: UnivariatePolynomial<SymbolicExt<F, EF>> =
            UnivariatePolynomial {
                coefficients: univariate_poly.coefficients.iter().map(|c| (*c).into()).collect(),
            };
        let expected_eval = univariate_poly_symbolic.eval_at_point(point.into());
        builder.assert_ext_eq(expected_eval, exts[0] + exts[1] * point + exts[2] * point * point);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program, my_bb_16_perm());
        let coeffs = (0..3).map(|_| F::rand(&mut rng)).collect::<Vec<_>>();
        let point = F::rand(&mut rng);
        runtime.witness_stream =
            [vec![coeffs[0].into(), coeffs[1].into(), coeffs[2].into()], vec![point.into()]]
                .concat()
                .into();
        runtime.run().unwrap();
    }

    #[test]
    fn test_eq_eval() {
        let mut builder = AsmBuilder::<F, EF>::default();
        let vec_1: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let vec_2: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let point_1 = Point::from(vec_1);
        let point_2 = Point::from(vec_2);
        let eq_eval = Mle::full_lagrange_eval(&point_1, &point_2);
        let one: Ext<F, EF> = builder.constant(EF::one());
        builder.assert_ext_eq(eq_eval, one);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program, my_bb_16_perm());
        runtime.witness_stream =
            [vec![F::zero().into(), F::one().into()], vec![F::zero().into(), F::one().into()]]
                .concat()
                .into();
        runtime.run().unwrap();
    }

    #[test]
    fn test_full_geq() {
        let mut builder = AsmBuilder::<F, EF>::default();
        let vec_1: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let vec_2: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let point_1 = Point::from(vec_1);
        let point_2 = Point::from(vec_2);
        let geq_eval = full_geq(&point_1, &point_2);
        let one: Ext<F, EF> = builder.constant(EF::one());
        builder.assert_ext_eq(geq_eval, one);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program, my_bb_16_perm());
        runtime.witness_stream =
            [vec![F::zero().into(), F::one().into()], vec![F::one().into(), F::zero().into()]]
                .concat()
                .into();
        runtime.run().unwrap();
    }
}

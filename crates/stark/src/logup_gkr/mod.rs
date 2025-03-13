mod execution;
mod logup_poly;
mod prover;
mod types;
mod verifier;

pub use execution::*;
pub use logup_poly::*;
pub(crate) use prover::*;
pub use types::*;
pub(crate) use verifier::*;

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use rand::Rng;
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_alloc::CpuBackend;
    use slop_baby_bear::BabyBear;
    use slop_jagged::{
        BabyBearPoseidon2, JaggedPcsVerifier, Poseidon2BabyBearJaggedCpuProverComponents,
    };
    use slop_matrix::dense::RowMajorMatrix;
    use slop_multilinear::{Mle, PaddedMle, Padding};

    use crate::{
        generate_mles,
        logup_gkr::{
            prover::generate_gkr_proof, verifier::verify_gkr_rounds, GkrPointAndEvals,
            GkrProofWithoutOpenings,
        },
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    type C = BabyBearPoseidon2;

    #[tokio::test]
    async fn logup_gkr_works() {
        const LOG_N: usize = 18;
        let mut rng = rand::thread_rng();
        let num_polynomials = 8;

        let numerator_mles = PaddedMle::with_minimal_padding(
            Arc::new(
                RowMajorMatrix::new(
                    (0..num_polynomials * ((1 << LOG_N) + 7))
                        .map(|_| rng.gen::<F>())
                        .collect::<Vec<_>>(),
                    num_polynomials,
                )
                .into(),
            ),
            Padding::Constant((F::zero(), num_polynomials, CpuBackend)),
        );

        let denom_mles = PaddedMle::with_minimal_padding(
            Arc::new(
                RowMajorMatrix::new(
                    (0..num_polynomials * ((1 << LOG_N) + 7))
                        .map(|_| rng.gen::<EF>())
                        .collect::<Vec<_>>(),
                    num_polynomials,
                )
                .into(),
            ),
            Padding::Constant((EF::one(), num_polynomials, CpuBackend)),
        );

        test_gkr_logup(numerator_mles, denom_mles).await;

        let numerator_mles = PaddedMle::new(
            None,
            (LOG_N + 1) as u32,
            Padding::Constant((F::zero(), num_polynomials, CpuBackend)),
        );
        let denom_mles = PaddedMle::new(
            None,
            (LOG_N + 1) as u32,
            Padding::Constant((EF::one(), num_polynomials, CpuBackend)),
        );
        test_gkr_logup(numerator_mles, denom_mles).await;
    }

    async fn test_gkr_logup(numerator_mles: PaddedMle<F>, denom_mles: PaddedMle<EF>) {
        let log_blowup = 1;
        let log_stacking_height = 21;
        let max_log_row_count = 21;
        let verifier =
            JaggedPcsVerifier::<C>::new(log_blowup, log_stacking_height, max_log_row_count);
        // let prover = CpuProver::new(verifier.clone());

        let mut challenger = verifier.challenger();

        let GkrProofWithoutOpenings {
            prover_messages,
            sc_proofs,
            numerator_claims: perm_numerator_claim,
            denom_claims: perm_denom_claim,
            challenge: _challenge_point,
        } = generate_gkr_proof::<Poseidon2BabyBearJaggedCpuProverComponents>(
            numerator_mles.clone(),
            denom_mles.clone(),
            &mut challenger,
        )
        .await;

        let mut challenger = verifier.challenger();
        let GkrPointAndEvals { point, numerator_evals, denom_evals } =
            verify_gkr_rounds::<BabyBearPoseidon2>(
                prover_messages.as_slice(),
                sc_proofs.as_slice(),
                perm_numerator_claim.as_slice(),
                perm_denom_claim.as_slice(),
                &mut challenger,
            )
            .unwrap();

        // Check that the evaluations produced by GKR are correct.
        assert_eq!(
            numerator_evals,
            numerator_mles.eval_at(&point).await.evaluations().as_buffer().to_vec()
        );
        assert_eq!(
            denom_evals,
            denom_mles.eval_at(&point).await.evaluations().as_buffer().to_vec()
        );
    }

    #[tokio::test]
    async fn logup_generate_mles() {
        const LOG_N: usize = 22;
        let mut rng = rand::thread_rng();

        let numerator_mles = PaddedMle::with_minimal_padding(
            Arc::new((0..1 << LOG_N).map(|_| rng.gen::<F>()).collect::<Mle<_>>()),
            Padding::Constant((F::zero(), 1, CpuBackend)),
        );
        let denom_mles = PaddedMle::with_minimal_padding(
            Arc::new((0..1 << LOG_N).map(|_| rng.gen::<EF>()).collect::<Mle<_>>()),
            Padding::Constant((EF::one(), 1, CpuBackend)),
        );

        let start_time = Instant::now();
        let _ = generate_mles::<Poseidon2BabyBearJaggedCpuProverComponents>(
            numerator_mles.clone(),
            denom_mles.clone(),
        )
        .await;

        let duration = start_time.elapsed();
        tracing::debug!("Duration: {duration:?}");
    }
}

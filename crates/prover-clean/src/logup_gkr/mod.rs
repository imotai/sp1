//! Each table has some associated lookups. Evaluating lookups for each table has a "jagged" structure, since each table might have a different height.
//! Then, once we have run GKR for every table to completion, we join the results together with the "interactions layers".
//!
use csl_cuda::ToDevice;
use slop_algebra::AbstractField;
use slop_alloc::HasBackend;
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, Point};

use sp1_hypercube::LogupGkrRoundProof;

use crate::{
    config::{Ext, Felt},
    logup_gkr::utils::FirstLayerPolynomial,
};
mod execution;
mod interactions;
mod layer;
mod sumcheck;
mod utils;

pub use utils::{
    generate_test_data, get_polys_from_layer, random_first_layer, FirstGkrLayer, FirstLayerData,
    GkrCircuitLayer, GkrLayer, GkrTestData, LogupRoundPolynomial, PolynomialLayer,
};

pub use sumcheck::{
    bench_materialized_sumcheck, first_round_sumcheck, materialized_round_sumcheck,
};

pub use execution::{extract_outputs, gkr_transition};

async fn prove_materialized_round<C: FieldChallenger<Felt>>(
    layer: GkrLayer,
    eval_point: &Point<Ext>,
    numerator_eval: Ext,
    denominator_eval: Ext,
    challenger: &mut C,
) -> LogupGkrRoundProof<Ext> {
    let lambda = challenger.sample_ext_element::<Ext>();
    let (interaction_point, row_point) =
        eval_point.split_at(layer.num_interaction_variables as usize);
    let backend = layer.jagged_mle.backend().clone();
    let interaction_point = interaction_point.to_device_in(&backend).await.unwrap();
    let row_point = row_point.to_device_in(&backend).await.unwrap();
    let eq_interaction = Mle::partial_lagrange(&interaction_point).await;
    let eq_row = Mle::partial_lagrange(&row_point).await;
    let sumcheck_poly = LogupRoundPolynomial {
        layer: PolynomialLayer::CircuitLayer(layer),
        eq_row,
        eq_interaction,
        lambda,
        eq_adjustment: Ext::one(),
        padding_adjustment: Ext::one(),
        point: eval_point.clone(),
    };
    let claim = numerator_eval * lambda + denominator_eval;

    // Produce the sumcheck proof.
    let (sumcheck_proof, openings) =
        sumcheck::materialized_round_sumcheck(sumcheck_poly, challenger, claim).await;
    let [numerator_0, numerator_1, denominator_0, denominator_1] = openings.try_into().unwrap();

    LogupGkrRoundProof { numerator_0, numerator_1, denominator_0, denominator_1, sumcheck_proof }
}

async fn prove_first_round<C: FieldChallenger<Felt>>(
    layer: FirstGkrLayer,
    eval_point: &Point<Ext>,
    numerator_eval: Ext,
    denominator_eval: Ext,
    challenger: &mut C,
) -> LogupGkrRoundProof<Ext> {
    let lambda = challenger.sample_ext_element::<Ext>();
    let (interaction_point, row_point) =
        eval_point.split_at(layer.num_interaction_variables as usize);
    let backend = layer.jagged_mle.backend();
    let interaction_point = interaction_point.to_device_in(backend).await.unwrap();
    let row_point = row_point.to_device_in(backend).await.unwrap();
    let eq_interaction = Mle::partial_lagrange(&interaction_point).await;
    let eq_row = Mle::partial_lagrange(&row_point).await;

    let sumcheck_poly =
        FirstLayerPolynomial { layer, eq_row, eq_interaction, lambda, point: eval_point.clone() };

    let claim = numerator_eval * lambda + denominator_eval;
    // Produce the sumcheck proof.
    let (sumcheck_proof, openings) =
        sumcheck::first_round_sumcheck(sumcheck_poly, challenger, claim).await;
    let [numerator_0, numerator_1, denominator_0, denominator_1] = openings.try_into().unwrap();
    LogupGkrRoundProof { numerator_0, numerator_1, denominator_0, denominator_1, sumcheck_proof }
}

pub async fn prove_round<C: FieldChallenger<Felt>>(
    circuit: GkrCircuitLayer,
    eval_point: &Point<Ext>,
    numerator_eval: Ext,
    denominator_eval: Ext,
    challenger: &mut C,
) -> LogupGkrRoundProof<Ext> {
    match circuit {
        GkrCircuitLayer::Materialized(layer) => {
            prove_materialized_round(
                layer,
                eval_point,
                numerator_eval,
                denominator_eval,
                challenger,
            )
            .await
        }
        GkrCircuitLayer::FirstLayer(layer) => {
            prove_first_round(layer, eval_point, numerator_eval, denominator_eval, challenger).await
        }
        GkrCircuitLayer::FirstLayerVirtual(_) => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use crate::logup_gkr::utils::{
        generate_test_data, get_polys_from_layer, random_first_layer, GkrTestData,
    };
    use itertools::Itertools;
    use slop_alloc::ToHost;
    use slop_basefold::{BasefoldVerifier, Poseidon2KoalaBear16BasefoldConfig};
    use slop_sumcheck::partially_verify_sumcheck_proof;

    use crate::logup_gkr::execution::{extract_outputs, gkr_transition, layer_transition};

    use super::*;

    use rand::{rngs::StdRng, SeedableRng};

    #[tokio::test]
    async fn test_logup_gkr_circuit_transition() {
        let mut rng = StdRng::seed_from_u64(1);

        let interaction_col_sizes: Vec<u32> =
            vec![(1 << 10) + 32, (1 << 10) - 2, 1 << 6, 1 << 8, (1 << 10) + 2];
        let (layer, test_data) = generate_test_data(&mut rng, interaction_col_sizes, None).await;
        let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

        let GkrLayer {
            jagged_mle,
            interaction_col_sizes,
            num_interaction_variables,
            num_row_variables,
        } = layer;

        csl_cuda::spawn(move |t| async move {
            let jagged_mle = jagged_mle.into_device(&t).await.unwrap();

            let layer = GkrLayer {
                jagged_mle,
                interaction_col_sizes,
                num_interaction_variables,
                num_row_variables,
            };

            // Test a single transition.
            let next_layer = layer_transition(&layer).await;

            let GkrLayer {
                jagged_mle: next_layer_data,
                interaction_col_sizes,
                num_interaction_variables,
                num_row_variables,
            } = next_layer;

            let next_layer_data = next_layer_data.into_host().await.unwrap();

            let next_layer_host = GkrLayer {
                jagged_mle: next_layer_data,
                interaction_col_sizes,
                num_interaction_variables,
                num_row_variables,
            };

            let next_layer_data = get_polys_from_layer(&next_layer_host).await;

            let next_numerator_0 = next_layer_data.numerator_0;
            let next_numerator_1 = next_layer_data.numerator_1;
            let next_denominator_0 = next_layer_data.denominator_0;
            let next_denominator_1 = next_layer_data.denominator_1;

            let next_n_values = next_numerator_0
                .guts()
                .as_slice()
                .iter()
                .interleave(next_numerator_1.guts().as_slice())
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(next_n_values.len(), numerator_0.guts().as_slice().len());
            let next_d_values = next_denominator_0
                .guts()
                .as_slice()
                .iter()
                .interleave(next_denominator_1.guts().as_slice())
                .copied()
                .collect::<Vec<_>>();

            for (i, (((((next_n, next_d), n_0), n_1), d_0), d_1)) in next_n_values
                .iter()
                .zip_eq(next_d_values)
                .zip_eq(numerator_0.guts().as_slice())
                .zip_eq(numerator_1.guts().as_slice())
                .zip_eq(denominator_0.guts().as_slice())
                .zip_eq(denominator_1.guts().as_slice())
                .enumerate()
            {
                assert_eq!(next_d, *d_0 * *d_1, "failed at index {i}");
                assert_eq!(*next_n, *n_0 * *d_1 + *n_1 * *d_0, "failed at index {i}");
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_logup_gkr_round_prover() {
        let mut rng = StdRng::seed_from_u64(1);

        type Config = Poseidon2KoalaBear16BasefoldConfig;

        let verifier = BasefoldVerifier::<_, Config>::new(1);
        let get_challenger = move || verifier.clone().challenger();

        let interaction_col_sizes: Vec<u32> = vec![
            99064, 99064, 99064, 188896, 188896, 188896, 85256, 107776, 107776, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 56360, 56360, 56360, 56360, 56360, 56360, 4,
            169496, 169496, 169496, 169496, 169496,
        ];
        // let interaction_col_sizes: Vec<u32> = vec![32, 32, 32, 32];
        let layer = random_first_layer(&mut rng, interaction_col_sizes, Some(19)).await;
        println!("generated test data");

        let FirstGkrLayer {
            jagged_mle,
            interaction_col_sizes,
            num_interaction_variables,
            num_row_variables,
        } = layer;

        println!("num row variables: {}", num_row_variables);

        let first_eval_point = Point::<Ext>::rand(&mut rng, num_interaction_variables + 1);

        csl_cuda::spawn(move |t| async move {
            let jagged_mle = jagged_mle.into_device(&t).await.unwrap();

            let layer = FirstGkrLayer {
                jagged_mle,
                interaction_col_sizes,
                num_interaction_variables,
                num_row_variables,
            };
            let layer = GkrCircuitLayer::FirstLayer(layer);

            t.synchronize().await.unwrap();
            let time = tokio::time::Instant::now();
            let mut layers = vec![layer];
            for _ in 0..num_row_variables - 1 {
                let layer = gkr_transition(layers.last().unwrap()).await;
                layers.push(layer);
            }
            t.synchronize().await.unwrap();
            println!("trace generation time: {:?}", time.elapsed());

            let time = tokio::time::Instant::now();
            layers.reverse();
            let first_layer =
                if let GkrCircuitLayer::Materialized(first_layer) = layers.first().unwrap() {
                    first_layer
                } else {
                    panic!("first layer not correct");
                };
            assert_eq!(first_layer.num_row_variables, 1);

            let output = extract_outputs(first_layer, num_interaction_variables).await;
            println!("time to extract values: {:?}", time.elapsed());

            let first_point_device = first_eval_point.to_device_in(&t).await.unwrap();
            let first_numerator_eval =
                output.numerator.eval_at(&first_point_device).await.to_host().await.unwrap()[0];
            let first_denominator_eval =
                output.denominator.eval_at(&first_point_device).await.to_host().await.unwrap()[0];

            let mut challenger = get_challenger();
            t.synchronize().await.unwrap();
            let time = tokio::time::Instant::now();
            let mut round_proofs = Vec::new();
            // Follow the GKR protocol layer by layer.
            let mut numerator_eval = first_numerator_eval;
            let mut denominator_eval = first_denominator_eval;
            let mut eval_point = first_eval_point.clone();

            for layer in layers {
                let round_proof = prove_round(
                    layer,
                    &eval_point,
                    numerator_eval,
                    denominator_eval,
                    &mut challenger,
                )
                .await;

                // Observe the prover message.
                challenger.observe_ext_element(round_proof.numerator_0);
                challenger.observe_ext_element(round_proof.numerator_1);
                challenger.observe_ext_element(round_proof.denominator_0);
                challenger.observe_ext_element(round_proof.denominator_1);
                // Get the evaluation point for the claims.
                eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();
                // Sample the last coordinate.
                let last_coordinate = challenger.sample_ext_element::<Ext>();
                // Compute the evaluation of the numerator and denominator at the last coordinate.
                numerator_eval = round_proof.numerator_0
                    + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
                denominator_eval = round_proof.denominator_0
                    + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
                eval_point.add_dimension_back(last_coordinate);
                // Add the round proof to the total
                round_proofs.push(round_proof);
            }
            t.synchronize().await.unwrap();
            println!("proof generation time: {:?}", time.elapsed());

            // Follow the GKR protocol layer by layer.
            let mut challenger = get_challenger();
            let mut numerator_eval = first_numerator_eval;
            let mut denominator_eval = first_denominator_eval;
            let mut eval_point = first_eval_point;
            let num_proofs = round_proofs.len();
            println!("Num rounds: {num_proofs}");
            for (i, round_proof) in round_proofs.iter().enumerate() {
                // Get the batching challenge for combining the claims.
                let lambda = challenger.sample_ext_element::<Ext>();
                // Check that the claimed sum is consitent with the previous round values.
                let expected_claim = numerator_eval * lambda + denominator_eval;
                assert_eq!(round_proof.sumcheck_proof.claimed_sum, expected_claim);
                // Verify the sumcheck proof.
                partially_verify_sumcheck_proof(
                    &round_proof.sumcheck_proof,
                    &mut challenger,
                    i + num_interaction_variables as usize + 1,
                    3,
                )
                .unwrap();
                // Verify that the evaluation claim is consistent with the prover messages.
                let (point, final_eval) = round_proof.sumcheck_proof.point_and_eval.clone();
                let eq_eval = Mle::full_lagrange_eval(&point, &eval_point);
                let numerator_sumcheck_eval = round_proof.numerator_0 * round_proof.denominator_1
                    + round_proof.numerator_1 * round_proof.denominator_0;
                let denominator_sumcheck_eval =
                    round_proof.denominator_0 * round_proof.denominator_1;
                let expected_final_eval =
                    eq_eval * (numerator_sumcheck_eval * lambda + denominator_sumcheck_eval);

                assert_eq!(final_eval, expected_final_eval, "Failure in round {i}");

                // Observe the prover message.
                challenger.observe_ext_element(round_proof.numerator_0);
                challenger.observe_ext_element(round_proof.numerator_1);
                challenger.observe_ext_element(round_proof.denominator_0);
                challenger.observe_ext_element(round_proof.denominator_1);

                // Get the evaluation point for the claims.
                eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();

                // Sample the last coordinate and add to the point.
                let last_coordinate = challenger.sample_ext_element::<Ext>();
                eval_point.add_dimension_back(last_coordinate);
                // Update the evaluation of the numerator and denominator at the last coordinate.
                numerator_eval = round_proof.numerator_0
                    + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
                denominator_eval = round_proof.denominator_0
                    + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
            }
        })
        .await
        .unwrap();
    }
}

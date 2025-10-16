//! Each chip may have some associated lookups. We place each chip's traces together in a single "jagged" buffer, since each chip may have a different height.
//! Then, once we have run GKR for every chip to completion, we join the results together with the "interactions layers".
//!
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
};

use csl_cuda::{TaskScope, ToDevice};
use itertools::Itertools;
use slop_algebra::AbstractField;
use slop_alloc::{CanCopyFromRef, HasBackend, ToHost};
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, MultilinearPcsChallenger, Point};
use tracing::Instrument;

use sp1_hypercube::{
    air::MachineAir, Chip, ChipEvaluation, LogUpEvaluations, LogUpGkrOutput, LogupGkrProof,
    LogupGkrRoundProof,
};

use crate::tracegen::generate_gkr_circuit;
use cslpc_utils::traces::JaggedTraceMle;
use cslpc_utils::{Ext, Felt};
use cslpc_zerocheck::primitives::round_batch_evaluations;
mod execution;
mod interactions;
mod layer;
mod sumcheck;
mod tracegen;
mod utils;

pub use utils::*;

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
    let claim = numerator_eval * lambda + denominator_eval;
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
    let claim = numerator_eval * lambda + denominator_eval;
    let (interaction_point, row_point) =
        eval_point.split_at(layer.num_interaction_variables as usize);

    let backend = layer.jagged_mle.backend();
    let interaction_point = interaction_point.to_device_in(backend).await.unwrap();
    let row_point = row_point.to_device_in(backend).await.unwrap();
    let eq_interaction = Mle::partial_lagrange(&interaction_point).await;
    let eq_row = Mle::partial_lagrange(&row_point).await;

    let sumcheck_poly =
        FirstLayerPolynomial { layer, eq_row, eq_interaction, lambda, point: eval_point.clone() };

    // Produce the sumcheck proof.
    let (sumcheck_proof, openings) =
        sumcheck::first_round_sumcheck(sumcheck_poly, challenger, claim).await;
    let [numerator_0, numerator_1, denominator_0, denominator_1] = openings.try_into().unwrap();
    LogupGkrRoundProof { numerator_0, numerator_1, denominator_0, denominator_1, sumcheck_proof }
}

pub async fn prove_round<'a, C: FieldChallenger<Felt>>(
    circuit: GkrCircuitLayer<'a>,
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

/// Proves the GKR circuit, layer by layer.
pub async fn prove_gkr_circuit<'a, C: FieldChallenger<Felt>>(
    numerator_value: Ext,
    denominator_value: Ext,
    eval_point: Point<Ext>,
    mut circuit: LogUpCudaCircuit<'a, TaskScope>,
    challenger: &mut C,
) -> (Point<Ext>, Vec<LogupGkrRoundProof<Ext>>) {
    let mut round_proofs = Vec::new();
    // Follow the GKR protocol layer by layer.
    let mut numerator_eval = numerator_value;
    let mut denominator_eval = denominator_value;
    let mut eval_point = eval_point;
    while let Some(layer) = circuit.next().await {
        // Generate the round proof.
        let round_proof =
            prove_round(layer, &eval_point, numerator_eval, denominator_eval, challenger).await;

        // Observe the prover message.
        challenger.observe_ext_element::<Ext>(round_proof.numerator_0);
        challenger.observe_ext_element::<Ext>(round_proof.numerator_1);
        challenger.observe_ext_element::<Ext>(round_proof.denominator_0);
        challenger.observe_ext_element::<Ext>(round_proof.denominator_1);

        // Get the evaluation point for the claims of the next round.
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
    (eval_point, round_proofs)
}

/// End-to-end proves lookups for a given trace.
pub async fn prove_logup_gkr<A: MachineAir<Felt>, C: MultilinearPcsChallenger<Felt>>(
    chips: &BTreeSet<Chip<Felt, A>>,
    jagged_trace_data: &JaggedTraceMle<Felt, TaskScope>,
    num_row_variables: u32,
    alpha: Ext,
    beta_seed: Point<Ext>,
    challenger: &mut C,
) -> LogupGkrProof<Ext> {
    let backend = jagged_trace_data.backend().clone();
    let num_interactions =
        chips.iter().map(|chip| chip.sends().len() + chip.receives().len()).sum::<usize>();
    let num_interaction_variables = num_interactions.next_power_of_two().ilog2();

    // Run the GKR circuit and get the output.
    let (output, circuit) = generate_gkr_circuit(
        chips,
        jagged_trace_data,
        num_row_variables,
        alpha,
        beta_seed,
        backend,
    )
    .await;

    let LogUpGkrOutput { numerator, denominator } = &output;

    // Copy the output to host and observe the claims.
    let host_numerator = numerator.to_host().await.unwrap();
    let host_denominator = denominator.to_host().await.unwrap();
    for (n, d) in
        host_numerator.guts().as_slice().iter().zip_eq(host_denominator.guts().as_slice().iter())
    {
        challenger.observe_ext_element(*n);
        challenger.observe_ext_element(*d);
    }
    let output_host = LogUpGkrOutput { numerator: host_numerator, denominator: host_denominator };

    // TODO: instead calculate from number of interactions.
    let initial_number_of_variables = numerator.num_variables();
    assert_eq!(initial_number_of_variables, num_interaction_variables + 1);
    let first_eval_point = challenger.sample_point::<Ext>(initial_number_of_variables);

    // Follow the GKR protocol layer by layer.
    let first_point = numerator.backend().copy_to(&first_eval_point).await.unwrap();
    let first_point_eq = Mle::partial_lagrange(&first_point).await;
    let first_numerator_eval =
        numerator.eval_at_eq(&first_point_eq).await.to_host().await.unwrap()[0];
    let first_denominator_eval =
        denominator.eval_at_eq(&first_point_eq).await.to_host().await.unwrap()[0];

    let (eval_point, round_proofs) = prove_gkr_circuit(
        first_numerator_eval,
        first_denominator_eval,
        first_eval_point,
        circuit,
        challenger,
    )
    .instrument(tracing::info_span!("prove GKR circuit"))
    .await;

    // Get the evaluations for each chip at the evaluation point of the last round.
    // We accomplish this by doing jagged fix last variable on the evaluation point.
    let eval_point = eval_point.last_k(num_row_variables as usize);
    let chip_evaluations = round_batch_evaluations(&eval_point, jagged_trace_data).await;
    let [preprocessed, main] = chip_evaluations.rounds.try_into().unwrap();

    let mut chip_evaluations = BTreeMap::new();

    let mut preprocessed_so_far = 0;

    for (chip, main_evals) in chips.iter().zip_eq(main.iter()) {
        let openings = ChipEvaluation {
            main_trace_evaluations: main_evals.to_host().await.unwrap(),
            preprocessed_trace_evaluations: if chip.preprocessed_width() != 0 {
                let res = Some(preprocessed[preprocessed_so_far].to_host().await.unwrap());
                preprocessed_so_far += 1;
                res
            } else {
                None
            },
        };

        // Observe the openings.
        if let Some(prep_eval) = openings.preprocessed_trace_evaluations.as_ref() {
            for eval in prep_eval.deref().iter() {
                challenger.observe_ext_element(*eval);
            }
        }
        for eval in openings.main_trace_evaluations.deref().iter() {
            challenger.observe_ext_element(*eval);
        }

        chip_evaluations.insert(chip.name(), openings);
    }

    let logup_evaluations = LogUpEvaluations { point: eval_point, chip_openings: chip_evaluations };

    LogupGkrProof { circuit_output: output_host, round_proofs, logup_evaluations }
}

#[cfg(test)]
mod tests {
    use crate::utils::{
        generate_test_data, get_polys_from_layer, jagged_first_gkr_layer_to_device,
        jagged_gkr_layer_to_device, jagged_gkr_layer_to_host, random_first_layer, GkrTestData,
    };
    use csl_cuda::run_in_place;
    use cslpc_tracegen::{
        full_tracegen,
        test_utils::tracegen_setup::{self, CORE_MAX_LOG_ROW_COUNT, LOG_STACKING_HEIGHT},
        CORE_MAX_TRACE_SIZE,
    };
    use cslpc_utils::TestGC;
    use itertools::Itertools;
    use serial_test::serial;
    use slop_alloc::ToHost;
    use slop_challenger::{FieldChallenger, IopCtx};
    use slop_sumcheck::partially_verify_sumcheck_proof;
    use sp1_hypercube::ShardVerifier;
    use std::sync::Arc;

    use crate::execution::{extract_outputs, gkr_transition, layer_transition};

    use super::*;

    use rand::{rngs::StdRng, SeedableRng};

    #[tokio::test]
    #[serial]
    async fn test_logup_gkr_circuit_transition() {
        let mut rng = StdRng::seed_from_u64(1);

        let interaction_row_counts: Vec<u32> =
            vec![(1 << 10) + 32, (1 << 10) - 2, 1 << 6, 1 << 8, (1 << 10) + 2];
        let (layer, test_data) = generate_test_data(&mut rng, interaction_row_counts, None).await;
        let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

        let GkrLayer { jagged_mle, num_interaction_variables, num_row_variables } = layer;

        csl_cuda::spawn(move |t| async move {
            let jagged_mle = jagged_gkr_layer_to_device(jagged_mle, &t).await;

            let layer = GkrLayer { jagged_mle, num_interaction_variables, num_row_variables };

            // Test a single transition.
            let next_layer = layer_transition(&layer).await;

            let GkrLayer {
                jagged_mle: next_layer_data,
                num_interaction_variables,
                num_row_variables,
            } = next_layer;

            let next_layer_data = jagged_gkr_layer_to_host(next_layer_data).await;

            let next_layer_host = GkrLayer {
                jagged_mle: next_layer_data,
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
    #[serial]
    async fn test_logup_gkr_round_prover() {
        let mut rng = StdRng::seed_from_u64(1);

        let get_challenger = move || TestGC::default_challenger();

        let interaction_row_counts: Vec<u32> = vec![
            99064, 99064, 99064, 188896, 188896, 188896, 85256, 107776, 107776, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 56360, 56360, 56360, 56360, 56360, 56360, 4,
            169496, 169496, 169496, 169496, 169496,
        ];
        let layer = random_first_layer(&mut rng, interaction_row_counts, Some(19)).await;
        println!("generated test data");

        let FirstGkrLayer { jagged_mle, num_interaction_variables, num_row_variables } = layer;

        println!("num row variables: {}", num_row_variables);

        let first_eval_point = Point::<Ext>::rand(&mut rng, num_interaction_variables + 1);

        csl_cuda::spawn(move |t| async move {
            let jagged_mle = jagged_first_gkr_layer_to_device(jagged_mle, &t).await;

            let layer = FirstGkrLayer { jagged_mle, num_interaction_variables, num_row_variables };
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

    #[tokio::test]
    #[serial]
    async fn test_logup_gkr_e2e() {
        let (machine, record, program) = tracegen_setup::setup().await;

        run_in_place(|scope| async move {
            // *********** Generate traces using the host tracegen. ***********
            let (public_values, jagged_trace_data, shard_chips) = full_tracegen(
                &machine,
                program.clone(),
                Arc::new(record),
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                &scope,
            )
            .await;

            // *********** Generate LogupGKR traces and prove end to end ***********
            let mut challenger = TestGC::default_challenger();

            let alpha = challenger.sample_ext_element();
            let max_interaction_arity = shard_chips
                .iter()
                .flat_map(|c| c.sends().iter().chain(c.receives().iter()))
                .map(|i| i.values.len() + 1)
                .max()
                .unwrap();
            let beta_seed_dim = max_interaction_arity.next_power_of_two().ilog2();
            let beta_seed = challenger.sample_point(beta_seed_dim);
            let pv_challenge: Ext = challenger.sample_ext_element();

            let shard_verifier: ShardVerifier<TestGC, _, _> =
                ShardVerifier::from_basefold_parameters(
                    1,
                    LOG_STACKING_HEIGHT,
                    CORE_MAX_LOG_ROW_COUNT as usize,
                    machine.clone(),
                );

            let cumulative_sum: Ext = shard_verifier
                .verify_public_values(pv_challenge, &alpha, &beta_seed, &public_values)
                .unwrap();

            let shard_chips = machine.smallest_cluster(&shard_chips).unwrap();
            let mut prover_challenger = challenger.clone();
            let proof = super::prove_logup_gkr(
                shard_chips,
                &jagged_trace_data,
                CORE_MAX_LOG_ROW_COUNT,
                alpha,
                beta_seed.clone(),
                &mut prover_challenger,
            )
            .await;
            let prover_challenge: Ext = prover_challenger.sample_ext_element();

            let degrees = shard_chips
                .iter()
                .map(|c| {
                    let poly_size = jagged_trace_data.main_poly_height(&c.name()).unwrap();

                    let threshold_point =
                        Point::from_usize(poly_size, CORE_MAX_LOG_ROW_COUNT as usize + 1);
                    (c.name(), threshold_point)
                })
                .collect();

            let mut verifier_challenger = challenger.clone();
            sp1_hypercube::LogUpGkrVerifier::verify_logup_gkr(
                shard_chips,
                &degrees,
                alpha,
                &beta_seed,
                -cumulative_sum,
                CORE_MAX_LOG_ROW_COUNT as usize,
                &proof,
                &mut verifier_challenger,
            )
            .unwrap();

            // Assert the prover and verifier have the same challenger state.
            let verifier_challenge: Ext = verifier_challenger.sample_ext_element();
            assert_eq!(verifier_challenge, prover_challenge);
        })
        .await;
    }
}

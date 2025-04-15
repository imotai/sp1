// #[cfg(test)]
// mod tests {
//     use std::sync::Arc;

//     use csl_cuda::TaskScope;
//     use csl_tracing::init_tracer;
//     use slop_baby_bear::BabyBear;
//     use sp1_core_executor::{Executor, Program, SP1Context, Trace};
//     use sp1_core_machine::{io::SP1Stdin, riscv::RiscvAir, utils::prove_core};
//     use sp1_primitives::io::SP1PublicValues;
//     use sp1_stark::{
//         AirOpenedValues, BabyBearPoseidon2, ChipOpenedValues, LogupGkrProof, MachineProof,
//         MachineVerifier, MachineVerifierError, SP1CoreOpts, ShardProof, ShardVerifier,
//     };

//     const FIBONACCI_LONG_ELF: &[u8] =
//         include_bytes!("../programs/fibonacci/riscv32im-succinct-zkvm-elf");
//     use tracing::Instrument;

//     use crate::{gpu_prover_opts, new_cuda_prover_sumcheck_eval};

//     /// The canonical entry point for testing a [`Program`] and [`SP1Stdin`] with a [`MachineProver`].
//     pub async fn run_test(
//         program: Program,
//         inputs: SP1Stdin,
//         scope: TaskScope,
//     ) -> Result<
//         (MachineProof<BabyBearPoseidon2>, SP1PublicValues),
//         MachineVerifierError<BabyBearPoseidon2>,
//     > {
//         let mut runtime = Executor::new(Arc::new(program), SP1CoreOpts::default());
//         runtime.write_vecs(&inputs.buffer);
//         runtime.run::<Trace>().unwrap();
//         let public_values = SP1PublicValues::from(&runtime.state.public_values_stream);

//         let proof = run_test_core(runtime, inputs, scope).await?;
//         Ok((proof, public_values))
//     }

//     #[allow(unused_variables)]
//     pub async fn run_test_core(
//         runtime: Executor<'static>,
//         inputs: SP1Stdin,
//         scope: TaskScope,
//     ) -> Result<MachineProof<BabyBearPoseidon2>, MachineVerifierError<BabyBearPoseidon2>> {
//         let log_blowup = 1;
//         let log_stacking_height = 21;
//         let max_log_row_count = 22;
//         let machine = RiscvAir::machine();
//         let verifier = ShardVerifier::from_basefold_parameters(
//             log_blowup,
//             log_stacking_height,
//             max_log_row_count,
//             machine,
//         );
//         let prover = new_cuda_prover_sumcheck_eval(verifier.clone(), scope);

//         let (pk, vk) = prover
//             .setup(runtime.program.clone())
//             .instrument(tracing::debug_span!("setup").or_current())
//             .await;
//         let challenger = verifier.pcs_verifier.challenger();
//         let opts = gpu_prover_opts().core_opts;

//         let (proof, _) = prove_core(
//             Arc::new(prover),
//             Arc::new(pk),
//             runtime.program.clone(),
//             &inputs,
//             opts,
//             SP1Context::default(),
//             challenger,
//         )
//         .instrument(tracing::debug_span!("prove core"))
//         .await
//         .unwrap();

//         let mut challenger = verifier.pcs_verifier.challenger();
//         let machine_verifier = MachineVerifier::new(verifier);
//         tracing::debug_span!("verify the proof")
//             .in_scope(|| machine_verifier.verify(&vk, &proof, &mut challenger))
//             .unwrap();
//         Ok(proof)
//     }

//     #[tokio::test(flavor = "multi_thread")]
//     async fn test_long_fibonacci() {
//         init_tracer();
//         let program = Program::from(FIBONACCI_LONG_ELF).unwrap();
//         let stdin = SP1Stdin::new();
//         csl_cuda::spawn(|t| async move { run_test(program, stdin, t.clone()).await.unwrap() })
//             .await
//             .unwrap()
//             .await
//             .unwrap();
//     }

//     use sp1_stark::air::MachineAir;
//     #[tokio::test(flavor = "multi_thread")]
//     async fn test_dummy_proof() {
//         init_tracer();
//         let program = Program::from(FIBONACCI_LONG_ELF).unwrap();
//         let stdin = SP1Stdin::new();
//         let (proof, _) =
//             csl_cuda::spawn(|t| async move { run_test(program, stdin, t.clone()).await.unwrap() })
//                 .await
//                 .unwrap()
//                 .await
//                 .unwrap();

//         let riscv_machine = RiscvAir::<BabyBear>::machine();
//         let shard_chip_names =
//             proof.shard_proofs[0].shard_chips.iter().cloned().collect::<Vec<_>>();
//         let shard_chips = riscv_machine
//             .chips()
//             .iter()
//             .filter(|chip| shard_chip_names.contains(&chip.air.name()))
//             .cloned()
//             .collect();

//         let log_blowup = 1;
//         let log_stacking_height = 21;
//         let max_log_row_count = 22;

//         let preprocessed_multiple =
//             proof.shard_proofs[0].evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[0]
//                 .round_evaluations[0]
//                 .num_polynomials();
//         let main_multiple =
//             proof.shard_proofs[0].evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[1]
//                 .round_evaluations[0]
//                 .num_polynomials();

//         let dummy_proof = sp1_recursion_circuit::dummy::dummy_shard_proof(
//             shard_chips,
//             max_log_row_count,
//             log_blowup,
//             log_stacking_height,
//             &[preprocessed_multiple, main_multiple],
//         );

//         let ShardProof {
//             shard_chips,
//             public_values,
//             logup_gkr_proof,
//             zerocheck_proof,
//             opened_values,
//             ..
//         } = proof.shard_proofs[0].clone();

//         let ShardProof {
//             shard_chips: dummy_shard_chips,
//             public_values: dummy_public_values,
//             logup_gkr_proof: dummy_logup_gkr_proof,
//             zerocheck_proof: dummy_zerocheck_proof,
//             opened_values: dummy_opened_values,
//             ..
//         } = dummy_proof;

//         assert_eq!(dummy_public_values.len(), public_values.len());
//         assert_eq!(dummy_shard_chips, shard_chips);

//         // Check the zerocheck proof shape.
//         assert_eq!(
//             dummy_zerocheck_proof.univariate_polys.len(),
//             zerocheck_proof.univariate_polys.len()
//         );
//         for (dummy_poly, poly) in dummy_zerocheck_proof
//             .univariate_polys
//             .iter()
//             .zip(zerocheck_proof.univariate_polys.iter())
//         {
//             assert_eq!(dummy_poly.coefficients.len(), poly.coefficients.len());
//         }
//         assert_eq!(
//             dummy_zerocheck_proof.point_and_eval.0.dimension(),
//             zerocheck_proof.point_and_eval.0.dimension()
//         );

//         // Check the logup gkr proof shape.
//         let LogupGkrProof {
//             circuit_output: dummy_circuit_output,
//             round_proofs: dummy_round_proofs,
//             logup_evaluations: dummy_logup_evaluations,
//         } = dummy_logup_gkr_proof;

//         let LogupGkrProof { circuit_output, round_proofs, logup_evaluations } = logup_gkr_proof;

//         assert_eq!(
//             dummy_circuit_output.numerator.num_non_zero_entries(),
//             circuit_output.numerator.num_non_zero_entries()
//         );

//         assert_eq!(
//             dummy_circuit_output.denominator.num_non_zero_entries(),
//             circuit_output.denominator.num_non_zero_entries()
//         );
//         assert_eq!(dummy_logup_evaluations.point.dimension(), logup_evaluations.point.dimension());

//         assert_eq!(dummy_round_proofs.len(), round_proofs.len());

//         for (dummy_round_proof, round_proof) in dummy_round_proofs.iter().zip(round_proofs.iter()) {
//             assert_eq!(
//                 dummy_round_proof.sumcheck_proof.point_and_eval.0.dimension(),
//                 round_proof.sumcheck_proof.point_and_eval.0.dimension()
//             );
//             assert_eq!(
//                 dummy_round_proof.sumcheck_proof.univariate_polys.len(),
//                 round_proof.sumcheck_proof.univariate_polys.len()
//             );
//             for (dummy_poly, poly) in dummy_round_proof
//                 .sumcheck_proof
//                 .univariate_polys
//                 .iter()
//                 .zip(round_proof.sumcheck_proof.univariate_polys.iter())
//             {
//                 assert_eq!(dummy_poly.coefficients.len(), poly.coefficients.len());
//             }
//         }

//         // Check that the logup GKR proofs have the same chips for the chip_openings field.
//         for key in dummy_logup_evaluations.chip_openings.keys() {
//             assert!(logup_evaluations.chip_openings.contains_key(key));
//         }

//         for key in logup_evaluations.chip_openings.keys() {
//             assert!(dummy_logup_evaluations.chip_openings.contains_key(key));
//         }

//         for (chip, opening) in dummy_logup_evaluations.chip_openings.iter() {
//             assert_eq!(
//                 opening.main_trace_evaluations.len(),
//                 logup_evaluations.chip_openings[chip].main_trace_evaluations.len()
//             );

//             if let Some(prep_evaluations) = opening.preprocessed_trace_evaluations.as_ref() {
//                 assert_eq!(
//                     prep_evaluations.len(),
//                     logup_evaluations.chip_openings[chip]
//                         .preprocessed_trace_evaluations
//                         .as_ref()
//                         .unwrap()
//                         .len()
//                 );
//             } else {
//                 assert!(logup_evaluations.chip_openings[chip]
//                     .preprocessed_trace_evaluations
//                     .is_none());
//             }
//         }

//         // Check that the opened values shapes match.
//         assert_eq!(dummy_opened_values.chips.len(), opened_values.chips.len());

//         for (dummy_chip, chip) in dummy_opened_values.chips.iter().zip(opened_values.chips.iter()) {
//             let ChipOpenedValues {
//                 preprocessed: AirOpenedValues { local: dummy_prep_local, next: dummy_prep_next },
//                 main: AirOpenedValues { local: dummy_main_local, next: dummy_main_next },
//                 degree: dummy_degree,
//                 ..
//             } = dummy_chip;

//             let ChipOpenedValues {
//                 preprocessed: AirOpenedValues { local: prep_local, next: prep_next },
//                 main: AirOpenedValues { local: main_local, next: main_next },
//                 degree,
//                 ..
//             } = chip;

//             assert_eq!(dummy_degree.dimension(), degree.dimension());
//             assert_eq!(dummy_prep_local.len(), prep_local.len());
//             assert_eq!(dummy_prep_next.len(), prep_next.len());
//             assert_eq!(dummy_main_local.len(), main_local.len());
//             assert_eq!(dummy_main_next.len(), main_next.len());
//         }

//         // The jagged PCS proof gets checked in a unit test on its own, so we don't check it here.
//     }
// }

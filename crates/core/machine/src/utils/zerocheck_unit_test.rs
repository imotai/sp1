use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{ExecutionRecord, Program};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{MachineAir, SP1AirBuilder};

use crate::utils::{next_multiple_of_32, zeroed_f_vec};

/// The number of main trace columns for `AddiChip`.
pub const NUM_MINIMAL_ADD_COLS: usize = size_of::<MinimalAddCols<u8>>();

/// A chip that implements addition for the opcode ADDI.
#[derive(Default, Clone)]
pub struct MinimalAddChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct MinimalAddCols<T> {
    op_a: T,
    op_b: T,
    op_c: T,
}

impl<F> BaseAir<F> for MinimalAddChip {
    fn width(&self) -> usize {
        NUM_MINIMAL_ADD_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for MinimalAddChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "MinimalAdd".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.addi_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the rows for the trace.
        let chunk_size = std::cmp::max(input.addi_events.len() / num_cpus::get(), 1);
        let merged_events = input.addi_events.iter().collect::<Vec<_>>();
        // let padded_nb_rows = <MinimalAddChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(merged_events.len() * NUM_MINIMAL_ADD_COLS);

        values.chunks_mut(chunk_size * NUM_MINIMAL_ADD_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MINIMAL_ADD_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MinimalAddCols<F> = row.borrow_mut();

                    if idx < merged_events.len() {
                        let event = merged_events[idx];
                        cols.op_a = F::from_canonical_u32(event.0.a);
                        cols.op_b = F::from_canonical_u32(event.0.b);
                        cols.op_c = F::from_canonical_u32(event.0.c);
                    }
                });
            },
        );

        let padded_row_template = {
            let mut row = [F::zero(); NUM_MINIMAL_ADD_COLS];
            let cols: &mut MinimalAddCols<F> = row.as_mut_slice().borrow_mut();
            cols.op_a = F::one();
            cols.op_b = F::zero();
            cols.op_c = F::zero();
            row
        };

        debug_assert!(padded_row_template.len() == NUM_MINIMAL_ADD_COLS);
        for i in input.shift_left_events.len() * NUM_MINIMAL_ADD_COLS..values.len() {
            values[i] = padded_row_template[i % NUM_MINIMAL_ADD_COLS];
        }

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_MINIMAL_ADD_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.addi_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for MinimalAddChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MinimalAddCols<AB::Var> = (*local).borrow();

        builder.assert_eq(local.op_a, local.op_b + local.op_c + AB::Expr::one());
    }
}

// #[cfg(test)]
// mod tests {
//     #![allow(clippy::print_stdout)]

//     use std::sync::Arc;

//     use itertools::Itertools;
//     use p3_uni_stark::get_symbolic_constraints;
//     use rand::Rng;
//     use slop_air::Air;
//     use slop_algebra::{extension::BinomialExtensionField, AbstractField};
//     use slop_alloc::CpuBackend;
//     use slop_basefold::{BasefoldConfig, DefaultBasefoldConfig,
// Poseidon2BabyBear16BasefoldConfig};     use slop_matrix::dense::RowMajorMatrixView;
//     use slop_multilinear::{full_geq, Mle, PaddedMle, Padding, Point, VirtualGeq};
//     use slop_sumcheck::{partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation};
//     use sp1_core_executor::{
//         events::{AluEvent, MemoryReadRecord, MemoryRecordEnum},
//         ExecutionRecord, ITypeRecord, Instruction, Opcode, Program, DEFAULT_PC_INC,
//     };
//     use sp1_stark::{
//         air::MachineAir,
//         prover::{ZeroCheckPoly, ZerocheckCpuProverData, ZerocheckProverData},
//         AirOpenedValues, BabyBearPoseidon2, Chip, ChipOpenedValues, ConstraintSumcheckFolder,
//         ShardVerifier, PROOF_MAX_NUM_PVS,
//     };

//     use crate::alu::minimal_add::{MinimalAddChip, NUM_MINIMAL_ADD_COLS};

//     type F = slop_baby_bear::BabyBear;
//     type EF = BinomialExtensionField<F, 4>;

//     #[tokio::test]
//     async fn test_zerocheck() {
//         let mut rng = rand::thread_rng();
//         let air = MinimalAddChip::default();
//         let num_real_entries = 65;
//         let num_variables = 7;

//         let mut shard = ExecutionRecord::default();

//         let instructions =
//             vec![Instruction::new(Opcode::ADDI, 29, 0, 5, false, true); num_real_entries];
//         let program = Program::new(instructions, 0, 0);

//         shard.program = Arc::new(program);

//         for i in 0..num_real_entries {
//             let operand_1 = rand::thread_rng().gen_range(0..(u16::MAX as u32));
//             let operand_2 = rand::thread_rng().gen_range(0..(u16::MAX as u32));

//             let result = operand_1.wrapping_add(operand_2) + 1;

//             shard.addi_events.push((
//                 AluEvent::new(
//                     (i as u32) * DEFAULT_PC_INC,
//                     0,
//                     Opcode::ADDI,
//                     result,
//                     operand_1,
//                     operand_2,
//                     false,
//                 ),
//                 ITypeRecord {
//                     a: MemoryRecordEnum::Read(MemoryReadRecord {
//                         value: 0,
//                         shard: 0,
//                         timestamp: 0,
//                         prev_shard: 0,
//                         prev_timestamp: 0,
//                     }),
//                     b: MemoryRecordEnum::Read(MemoryReadRecord {
//                         value: 0,
//                         shard: 0,
//                         timestamp: 0,
//                         prev_shard: 0,
//                         prev_timestamp: 0,
//                     }),
//                 },
//             ));
//         }

//         let virtually_padded_trace = MinimalAddChip::generate_trace(
//             &MinimalAddChip,
//             &shard,
//             &mut ExecutionRecord::default(),
//         );

//         assert!(<MinimalAddChip as MachineAir<F>>::preprocessed_width(&air) == 0);

//         let alpha = rng.gen::<EF>();
//         let gkr_power = rng.gen::<EF>();

//         let num_constraints = get_symbolic_constraints::<F, _>(&air, 0, PROOF_MAX_NUM_PVS).len();

//         let mut alpha_powers = alpha.powers().take(num_constraints).collect::<Vec<_>>();

//         alpha_powers.reverse();

//         let gkr_powers = gkr_power.powers().take(NUM_MINIMAL_ADD_COLS).collect::<Vec<_>>();

//         let prover_data = ZerocheckCpuProverData::default();

//         let main_trace = PaddedMle::new(
//             Some(Arc::new(virtually_padded_trace.clone().into())),
//             num_variables,
//             Padding::Constant((F::zero(), NUM_MINIMAL_ADD_COLS, CpuBackend)),
//         );

//         let virtual_geq =
//             VirtualGeq::new(num_real_entries as u32, F::one(), F::zero(), num_variables);

//         let air_data = prover_data
//             .round_prover(
//                 Arc::new(air),
//                 Arc::new(vec![F::zero(); PROOF_MAX_NUM_PVS]),
//                 Arc::new(alpha_powers.clone()),
//                 Arc::new(gkr_powers.clone()),
//             )
//             .await;

//         let dummy_main = vec![F::zero(); NUM_MINIMAL_ADD_COLS];

//         let mut folder = ConstraintSumcheckFolder {
//             preprocessed: RowMajorMatrixView::new_row(&[]),
//             main: RowMajorMatrixView::new_row(&dummy_main),
//             accumulator: EF::zero(),
//             public_values: &vec![F::zero(); PROOF_MAX_NUM_PVS],
//             constraint_index: 0,
//             powers_of_alpha: &alpha_powers,
//         };

//         let air = MinimalAddChip::default();

//         air.eval(&mut folder);
//         let padded_row_adjustment = folder.accumulator;

//         let zeta = Point::rand(&mut rng, num_variables);

//         let gkr_openings = main_trace.eval_at(&zeta).await;

//         let sumcheck_claim = gkr_openings
//             .evaluations()
//             .as_slice()
//             .iter()
//             .zip_eq(gkr_powers.iter())
//             .map(|(a, b)| *a * *b)
//             .sum::<EF>();

//         let zerocheck_poly = ZeroCheckPoly::<F, F, EF, _, CpuBackend>::new(
//             air_data,
//             zeta.clone(),
//             None,
//             main_trace.clone(),
//             EF::one(),
//             EF::zero(),
//             padded_row_adjustment,
//             virtual_geq,
//         );

//         let claims = vec![sumcheck_claim];
//         let t = 1;
//         let lambda = EF::zero();

//         let mut challenger = Poseidon2BabyBear16BasefoldConfig::default_challenger(
//             &Poseidon2BabyBear16BasefoldConfig::default_verifier(1),
//         );

//         let (proof, column_openings) =
//             reduce_sumcheck_to_evaluation(vec![zerocheck_poly], &mut challenger, claims, t,
// lambda)                 .await;

//         let chip_eval_claim = proof.point_and_eval.1;
//         let chip_eval_point = proof.point_and_eval.0.clone();

//         let column_openings = &column_openings[0];

//         assert_eq!(column_openings, &main_trace.eval_at(&chip_eval_point).await.to_vec());

//         let opening = ChipOpenedValues::<F, EF> {
//             preprocessed: AirOpenedValues { local: vec![], next: vec![] },
//             main: AirOpenedValues { local: column_openings.clone(), next: vec![] },
//             local_cumulative_sum: EF::zero(),
//             degree: Point::from_usize(num_real_entries as usize, num_variables as usize + 1),
//         };

//         let openings_batch = column_openings
//             .iter()
//             .zip_eq(gkr_powers.iter())
//             .map(|(opening, power)| *opening * *power)
//             .sum::<EF>();

//         let public_values = vec![F::zero(); PROOF_MAX_NUM_PVS];

//         let zerocheck_eq_val = Mle::full_lagrange_eval(&zeta, &chip_eval_point);

//         let padded_row_adjustment =
//             ShardVerifier::<BabyBearPoseidon2, _>::compute_padded_row_adjustment(
//                 &Chip::new(MinimalAddChip::default()),
//                 alpha,
//                 &public_values,
//             );

//         let mut point_extended = chip_eval_point.clone();
//         point_extended.add_dimension(EF::zero());

//         let geq_val = full_geq(&opening.degree, &point_extended);

//         let eval = ShardVerifier::<BabyBearPoseidon2, _>::eval_constraints(
//             &Chip::new(MinimalAddChip::default()),
//             &opening,
//             alpha,
//             &public_values,
//         );

//         let constraint_eval = eval - padded_row_adjustment * geq_val;

//         let mut challenger = Poseidon2BabyBear16BasefoldConfig::default_challenger(
//             &Poseidon2BabyBear16BasefoldConfig::default_verifier(1),
//         );

//         partially_verify_sumcheck_proof(&proof, &mut challenger).unwrap();
//         assert_eq!(chip_eval_claim, zerocheck_eq_val * (constraint_eval + openings_batch));
//     }
// }

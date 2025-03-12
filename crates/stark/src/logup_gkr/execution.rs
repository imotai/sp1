use std::sync::Arc;

use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::*;
use slop_algebra::{ExtensionField, Field, Powers};
use slop_alloc::CpuBackend;
use slop_matrix::dense::RowMajorMatrix;
use slop_multilinear::{PaddedMle, Padding};

use crate::Interaction;

pub(crate) fn generate_interaction_evals<F: Field, EF: ExtensionField<F>>(
    preprocessed_row: &[F],
    main_row: &[F],
    interaction: &Interaction<F>,
    is_send: bool,
    alpha: EF,
    betas: &Powers<EF>,
) -> (F, EF) {
    let mut denominator = alpha;
    let mut betas = betas.clone();
    denominator += betas.next().unwrap() * EF::from_canonical_usize(interaction.argument_index());
    for (columns, beta) in interaction.values.iter().zip(betas) {
        let apply = columns.apply::<F, F>(preprocessed_row, main_row);
        denominator += beta * apply;
    }
    let mut mult = interaction.multiplicity.apply::<F, F>(preprocessed_row, main_row);

    if !is_send {
        mult = -mult;
    }

    (mult, denominator)
}

/// Given the preprocessed and main traces of the protocol, generate the numerator and denominator
/// multilinear polynomials that are inputs into the GKR protocol.
#[allow(clippy::too_many_lines)]
pub fn generate_gkr_input_mles<F: Field, EF: ExtensionField<F>>(
    preprocessed: Option<&PaddedMle<F>>,
    main: &PaddedMle<F>,
    interactions: &[(&Interaction<F>, bool)],
    alpha: EF,
    betas: &Powers<EF>,
    log_max_row_height: usize,
) -> (PaddedMle<F>, PaddedMle<EF>) {
    let height: usize = main.num_real_entries();
    let num_interactions = interactions.len();

    if height == 0 {
        return (
            PaddedMle::new(
                None,
                log_max_row_height as u32,
                slop_multilinear::Padding::Constant((F::zero(), num_interactions, CpuBackend)),
            ),
            PaddedMle::new(
                None,
                log_max_row_height as u32,
                Padding::Constant((EF::one(), num_interactions, CpuBackend)),
            ),
        );
    }

    let mut numerator_evals = vec![F::zero(); height * num_interactions];
    let mut denom_evals = vec![EF::one(); height * num_interactions];

    match preprocessed {
        Some(prep) => {
            numerator_evals
                .par_chunks_exact_mut(num_interactions)
                .zip_eq(denom_evals.par_chunks_exact_mut(num_interactions))
                .zip_eq(
                    prep.inner()
                        .as_ref()
                        .unwrap()
                        .guts()
                        .as_slice()
                        .par_chunks(prep.num_polynomials())
                        .zip(
                            main.inner()
                                .as_ref()
                                .unwrap()
                                .guts()
                                .as_slice()
                                .par_chunks(main.num_polynomials()),
                        ),
                )
                .for_each(|((numerator_chunk, denom_chunk), (prep_row, main_row))| {
                    interactions
                        .iter()
                        .zip(numerator_chunk.iter_mut())
                        .zip(denom_chunk.iter_mut())
                        .for_each(|(((interaction, is_send), numerator_val), denom_val)| {
                            let (numerator_input, denom_input) = generate_interaction_evals(
                                prep_row,
                                main_row,
                                interaction,
                                *is_send,
                                alpha,
                                betas,
                            );
                            *numerator_val = numerator_input;
                            *denom_val = denom_input;
                        });
                });
        }
        None => {
            numerator_evals
                .par_chunks_exact_mut(num_interactions)
                .zip_eq(denom_evals.par_chunks_exact_mut(num_interactions))
                .zip(
                    main.inner()
                        .as_ref()
                        .unwrap()
                        .guts()
                        .as_slice()
                        .par_chunks(main.num_polynomials()),
                )
                .for_each(|((numerator_chunk, denom_chunk), main_row)| {
                    interactions
                        .iter()
                        .zip(numerator_chunk.iter_mut())
                        .zip(denom_chunk.iter_mut())
                        .for_each(|(((interaction, is_send), numerator_val), denom_val)| {
                            let (numerator_input, denom_input) = generate_interaction_evals(
                                &[],
                                main_row,
                                interaction,
                                *is_send,
                                alpha,
                                betas,
                            );
                            *numerator_val = numerator_input;
                            *denom_val = denom_input;
                        });
                });
        }
    }

    (
        PaddedMle::new(
            Some(Arc::new(RowMajorMatrix::new(numerator_evals, num_interactions).into())),
            log_max_row_height as u32,
            Padding::Constant((F::zero(), num_interactions, CpuBackend)),
        ),
        PaddedMle::new(
            Some(Arc::new(RowMajorMatrix::new(denom_evals, num_interactions).into())),
            log_max_row_height as u32,
            Padding::Constant((EF::one(), num_interactions, CpuBackend)),
        ),
    )
}

use std::sync::Arc;

use itertools::izip;
use rayon::prelude::*;
use slop_algebra::{ExtensionField, Field, Powers};
use slop_alloc::{Backend, CpuBackend};
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

/// A collection of multilinear polynomials produced from running the `LogUp` circuit and passed in
/// to the GKR sumcheck instances.
pub struct GkrMle<F, EF, B: Backend = CpuBackend> {
    /// The zero evaluations of the numerator multilinear polynomial.
    pub numerator_0: PaddedMle<F, B>,
    /// The one evaluations of the numerator multilinear polynomial.
    pub numerator_1: PaddedMle<F, B>,
    /// The zero evaluations of the denominator multilinear polynomial.
    pub denom_0: PaddedMle<EF, B>,
    /// The one evaluations of the denominator multilinear polynomial.
    pub denom_1: PaddedMle<EF, B>,
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
) -> GkrMle<F, EF> {
    let height: usize = main.num_real_entries();
    let num_interactions = interactions.len();

    if height == 0 {
        let numerator_0 = PaddedMle::new(
            None,
            (log_max_row_height - 1) as u32,
            slop_multilinear::Padding::Constant((F::zero(), num_interactions, CpuBackend)),
        );
        let numerator_1 = PaddedMle::new(
            None,
            (log_max_row_height - 1) as u32,
            slop_multilinear::Padding::Constant((F::zero(), num_interactions, CpuBackend)),
        );
        let denom_0 = PaddedMle::new(
            None,
            (log_max_row_height - 1) as u32,
            slop_multilinear::Padding::Constant((EF::one(), num_interactions, CpuBackend)),
        );
        let denom_1 = PaddedMle::new(
            None,
            (log_max_row_height - 1) as u32,
            slop_multilinear::Padding::Constant((EF::one(), num_interactions, CpuBackend)),
        );
        return GkrMle { numerator_0, numerator_1, denom_0, denom_1 };
    }

    assert!(height != 1);

    let mut numerator_0_evals = vec![F::zero(); height / 2 * num_interactions];
    let mut numerator_1_evals = vec![F::zero(); height / 2 * num_interactions];
    let mut denom_0_evals = vec![EF::one(); height / 2 * num_interactions];
    let mut denom_1_evals = vec![EF::one(); height / 2 * num_interactions];

    match preprocessed {
        Some(prep) => {
            (
                numerator_0_evals.par_chunks_exact_mut(num_interactions),
                numerator_1_evals.par_chunks_exact_mut(num_interactions),
                denom_0_evals.par_chunks_exact_mut(num_interactions),
                denom_1_evals.par_chunks_exact_mut(num_interactions),
                prep.inner()
                    .as_ref()
                    .unwrap()
                    .guts()
                    .as_slice()
                    .par_chunks(2 * prep.num_polynomials()),
                main.inner()
                    .as_ref()
                    .unwrap()
                    .guts()
                    .as_slice()
                    .par_chunks(2 * main.num_polynomials()),
            )
                .into_par_iter()
                .for_each(
                    |(
                        numerator_0_chunk,
                        numerator_1_chunk,
                        denom_0_chunk,
                        denom_1_chunk,
                        prep_rows,
                        main_rows,
                    )| {
                        let (prep_row_0, prep_row_1) = prep_rows.split_at(prep.num_polynomials());
                        let (main_row_0, main_row_1) = main_rows.split_at(main.num_polynomials());

                        izip!(
                            interactions.iter(),
                            numerator_0_chunk.iter_mut(),
                            denom_0_chunk.iter_mut(),
                            numerator_1_chunk.iter_mut(),
                            denom_1_chunk.iter_mut()
                        )
                        .for_each(
                            |(
                                (interaction, is_send),
                                numerator_0_val,
                                denom_0_val,
                                numerator_1_val,
                                denom_1_val,
                            ): (_, &mut _, &mut _, &mut _, &mut _)| {
                                let (numerator_input, denom_input) = generate_interaction_evals(
                                    prep_row_0,
                                    main_row_0,
                                    interaction,
                                    *is_send,
                                    alpha,
                                    betas,
                                );
                                *numerator_0_val = numerator_input;
                                *denom_0_val = denom_input;

                                let (numerator_input, denom_input) = generate_interaction_evals(
                                    prep_row_1,
                                    main_row_1,
                                    interaction,
                                    *is_send,
                                    alpha,
                                    betas,
                                );
                                *numerator_1_val = numerator_input;
                                *denom_1_val = denom_input;
                            },
                        );
                    },
                );
        }
        None => {
            (
                numerator_0_evals.par_chunks_exact_mut(num_interactions),
                numerator_1_evals.par_chunks_exact_mut(num_interactions),
                denom_0_evals.par_chunks_exact_mut(num_interactions),
                denom_1_evals.par_chunks_exact_mut(num_interactions),
                main.inner()
                    .as_ref()
                    .unwrap()
                    .guts()
                    .as_slice()
                    .par_chunks_exact(2 * main.num_polynomials()),
            )
                .into_par_iter()
                .for_each(
                    |(
                        numerator_0_chunk,
                        numerator_1_chunk,
                        denom_0_chunk,
                        denom_1_chunk,
                        main_rows,
                    )| {
                        let (main_row_0, main_row_1) = main_rows.split_at(main.num_polynomials());
                        izip!(
                            interactions.iter(),
                            numerator_0_chunk.iter_mut(),
                            denom_0_chunk.iter_mut(),
                            numerator_1_chunk.iter_mut(),
                            denom_1_chunk.iter_mut()
                        )
                        .for_each(
                            |(
                                (interaction, is_send),
                                numerator_0_val,
                                denom_0_val,
                                numerator_1_val,
                                denom_1_val,
                            ): (_, &mut _, &mut _, &mut _, &mut _)| {
                                let (numerator_input, denom_input) = generate_interaction_evals(
                                    &[],
                                    main_row_0,
                                    interaction,
                                    *is_send,
                                    alpha,
                                    betas,
                                );
                                *numerator_0_val = numerator_input;
                                *denom_0_val = denom_input;

                                let (numerator_input, denom_input) = generate_interaction_evals(
                                    &[],
                                    main_row_1,
                                    interaction,
                                    *is_send,
                                    alpha,
                                    betas,
                                );
                                *numerator_1_val = numerator_input;
                                *denom_1_val = denom_input;
                            },
                        );
                    },
                );
        }
    }

    let numerator_0 = PaddedMle::new(
        Some(Arc::new(RowMajorMatrix::new(numerator_0_evals, num_interactions).into())),
        (log_max_row_height - 1) as u32,
        Padding::Constant((F::zero(), num_interactions, CpuBackend)),
    );
    let numerator_1 = PaddedMle::new(
        Some(Arc::new(RowMajorMatrix::new(numerator_1_evals, num_interactions).into())),
        (log_max_row_height - 1) as u32,
        Padding::Constant((F::zero(), num_interactions, CpuBackend)),
    );
    let denom_0 = PaddedMle::new(
        Some(Arc::new(RowMajorMatrix::new(denom_0_evals, num_interactions).into())),
        (log_max_row_height - 1) as u32,
        Padding::Constant((EF::one(), num_interactions, CpuBackend)),
    );
    let denom_1 = PaddedMle::new(
        Some(Arc::new(RowMajorMatrix::new(denom_1_evals, num_interactions).into())),
        (log_max_row_height - 1) as u32,
        Padding::Constant((EF::one(), num_interactions, CpuBackend)),
    );

    GkrMle { numerator_0, numerator_1, denom_0, denom_1 }
}

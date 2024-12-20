use itertools::{izip, Itertools};
use std::marker::PhantomData;

use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{LagrangeSelectors, Pcs, PolynomialSpace};
use p3_uni_stark::{StarkGenericConfig, Val};

use spl_algebra::{AbstractExtensionField, AbstractField};
use spl_multilinear::{MultilinearPcs, Point};

use crate::{
    air_types::{AirOpenedValues, ChipOpenedValues},
    folder::VerifierConstraintFolder,
    prover::LOG_QUOTIENT_DEGREE,
    types::{
        Dom, MultivariateAdapterAir, MultivariateAdapterError, MultivariateAdapterPCS,
        MultivariateAdapterProof,
    },
};

/// An air that knows how many columns there are in the "adapter" trace.
pub trait AdapterAir<F>: BaseAir<F> {
    fn adapter_width(&self) -> usize;
}

impl<SC: StarkGenericConfig> MultilinearPcs<SC::Challenger> for MultivariateAdapterPCS<SC> {
    type Proof = MultivariateAdapterProof<SC>;

    type Commitment = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment;

    type Error = MultivariateAdapterError;

    type F = Val<SC>;

    type EF = SC::Challenge;

    fn verify_evaluations(
        &self,
        point: spl_multilinear::Point<SC::Challenge>,
        eval_claims: &[SC::Challenge],
        main_commit: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut SC::Challenger,
    ) -> Result<(), Self::Error> {
        let MultivariateAdapterProof {
            adapter_commit,
            quotient_commit,
            opening_proof,
            opened_values,
            ..
        } = proof;

        let pcs = self.config.pcs();

        challenger.observe(main_commit.clone());

        let log_degrees = opened_values.iter().map(|val| val.log_degree).collect::<Vec<_>>();

        let log_quotient_degrees =
            opened_values.iter().map(|_| LOG_QUOTIENT_DEGREE).collect::<Vec<_>>();

        let trace_domains = log_degrees
            .iter()
            .map(|log_degree| pcs.natural_domain_for_degree(1 << log_degree))
            .collect::<Vec<_>>();

        // Sample the batch challenge.
        let batch_challenge = challenger.sample_ext_element::<SC::Challenge>();

        challenger.observe(adapter_commit.clone());

        // Sample the constraint folding challenge.
        let alpha = challenger.sample_ext_element::<SC::Challenge>();

        // Observe the quotient commitments.
        challenger.observe(quotient_commit.clone());

        // Sample the opening point for the quotient polynomial check.
        let zeta = challenger.sample_ext_element::<SC::Challenge>();

        let parent_span = tracing::debug_span!("gather domains, points, and openings");
        let (
            main_domains_points_and_opens,
            adapter_domains_points_and_opens,
            quotient_chunk_domains,
            quotient_domains_points_and_opens,
        ) = parent_span.in_scope(|| {
            let main_domains_points_and_opens = trace_domains
                .iter()
                .zip_eq(opened_values.iter())
                .map(|(domain, values)| {
                    (
                        *domain,
                        vec![
                            (zeta, values.main.local.clone()),
                            (domain.next_point(zeta).unwrap(), values.main.next.clone()),
                        ],
                    )
                })
                .collect::<Vec<_>>();

            let adapter_domains_points_and_opens = trace_domains
                .iter()
                .zip_eq(opened_values.iter())
                .map(|(domain, values)| {
                    (
                        *domain,
                        vec![
                            (zeta, values.adapter.local.clone()),
                            (domain.next_point(zeta).unwrap(), values.adapter.next.clone()),
                        ],
                    )
                })
                .collect::<Vec<_>>();

            let quotient_chunk_domains = trace_domains
                .iter()
                .zip_eq(log_degrees)
                .zip_eq(log_quotient_degrees)
                .map(|((domain, log_degree), log_quotient_degree)| {
                    let quotient_degree = 1 << log_quotient_degree;
                    let quotient_domain =
                        domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
                    quotient_domain.split_domains(quotient_degree)
                })
                .collect::<Vec<_>>();

            let quotient_domains_points_and_opens =
                proof
                    .opened_values
                    .iter()
                    .zip_eq(quotient_chunk_domains.iter())
                    .flat_map(|(values, qc_domains)| {
                        values.quotient.iter().zip_eq(qc_domains).map(move |(values, q_domain)| {
                            (*q_domain, vec![(zeta, values.clone())])
                        })
                    })
                    .collect::<Vec<_>>();
            (
                main_domains_points_and_opens,
                adapter_domains_points_and_opens,
                quotient_chunk_domains,
                quotient_domains_points_and_opens,
            )
        });

        // Verify the opening proofs.
        let parent_span = tracing::debug_span!("verify univariate openings");
        parent_span.in_scope(move || {
            self.config
                .pcs()
                .verify(
                    vec![
                        (main_commit.clone(), main_domains_points_and_opens),
                        (adapter_commit.clone(), adapter_domains_points_and_opens),
                        (quotient_commit.clone(), quotient_domains_points_and_opens),
                    ],
                    opening_proof,
                    challenger,
                )
                .map_err(|_| MultivariateAdapterError::PcsError)
        })?;

        // Verify the constraint evaluations.
        for (trace_domain, qc_domains, values) in
            izip!(trace_domains, quotient_chunk_domains, opened_values.iter())
        {
            // Verify the shape of the opening arguments matches the expected values.
            let span = tracing::debug_span!("verify opening shape");
            span.in_scope(|| {
                self.verify_opening_shape(
                    MultivariateAdapterAir { batch_size: self.batch_size },
                    values,
                )
            })
            .map_err(|_| MultivariateAdapterError::ShapeMismatch)?;

            // Verify the constraint evaluation.

            let span = tracing::debug_span!("verify constraints");
            span.in_scope(|| {
                Self::verify_constraints::<MultivariateAdapterAir>(
                    &MultivariateAdapterAir { batch_size: self.batch_size },
                    values,
                    trace_domain,
                    qc_domains,
                    zeta,
                    alpha,
                    eval_claims,
                    point.clone(),
                    batch_challenge,
                )
            })
            .map_err(|_| MultivariateAdapterError::Verification)?;
        }

        Ok(())
    }
}

impl<SC: StarkGenericConfig> MultivariateAdapterPCS<SC> {
    pub fn verify_opening_shape<A: AdapterAir<Val<SC>>>(
        &self,
        air: A,
        opening: &ChipOpenedValues<SC::Challenge>,
    ) -> Result<(), MultivariateAdapterError> {
        // Verify that the main width matches the expected value for the chip.
        if opening.main.local.len() != air.width() || opening.main.next.len() != air.width() {
            return Err(MultivariateAdapterError::ShapeMismatch);
        }

        // Verify that the adapter width matches the expected value for the chip.
        if opening.adapter.local.len() != air.adapter_width() * SC::Challenge::D
            || opening.adapter.next.len() != air.adapter_width() * SC::Challenge::D
        {
            return Err(MultivariateAdapterError::ShapeMismatch);
        }

        // Verift that the number of quotient chunks matches the expected value for the chip.
        if opening.quotient.len() != 1 << (LOG_QUOTIENT_DEGREE) {
            return Err(MultivariateAdapterError::ShapeMismatch);
        }
        // For each quotient chunk, verify that the number of elements is equal to the degree of the
        // challenge extension field over the value field.
        for slice in &opening.quotient {
            if slice.len() != SC::Challenge::D {
                return Err(MultivariateAdapterError::ShapeMismatch);
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    pub fn verify_constraints<A>(
        air: &A,
        opening: &ChipOpenedValues<SC::Challenge>,
        trace_domain: Dom<SC>,
        qc_domains: Vec<Dom<SC>>,
        zeta: SC::Challenge,
        alpha: SC::Challenge,
        expected_evals: &[SC::Challenge],
        eval_point: Point<SC::Challenge>,
        batch_challenge: SC::Challenge,
    ) -> Result<(), MultivariateAdapterError>
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        let sels = trace_domain.selectors_at_point(zeta);

        // Recompute the quotient at zeta from the chunks.
        let quotient = Self::recompute_quotient(opening, &qc_domains, zeta);
        // Calculate the evaluations of the constraints at zeta.
        let folded_constraints = Self::eval_constraints::<A>(
            air,
            eval_point,
            opening,
            &sels,
            alpha,
            expected_evals,
            batch_challenge,
        );

        // Check that the constraints match the quotient, i.e.
        // folded_constraints(zeta) / Z_H(zeta) = quotient(zeta)
        if folded_constraints * sels.inv_zeroifier == quotient {
            Ok(())
        } else {
            Err(MultivariateAdapterError::Verification)
        }
    }

    /// Evaluates the constraints for a chip and opening.
    pub fn eval_constraints<A>(
        air: &A,
        eval_point: Point<SC::Challenge>,
        opening: &ChipOpenedValues<SC::Challenge>,
        selectors: &LagrangeSelectors<SC::Challenge>,
        alpha: SC::Challenge,
        expected_evals: &[SC::Challenge],
        batch_challenge: SC::Challenge,
    ) -> SC::Challenge
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        // Reconstruct the prmutation opening values as extention elements.
        let unflatten = |v: &[SC::Challenge]| {
            v.chunks_exact(SC::Challenge::D)
                .map(|chunk| {
                    chunk.iter().enumerate().map(|(e_i, &x)| SC::Challenge::monomial(e_i) * x).sum()
                })
                .collect::<Vec<SC::Challenge>>()
        };

        let adapter_opening = AirOpenedValues {
            local: unflatten(&opening.adapter.local),
            next: unflatten(&opening.adapter.next),
        };

        let mut folder = VerifierConstraintFolder::<SC> {
            main: opening.main.view(),
            adapter: adapter_opening.view(),
            is_first_row: selectors.is_first_row,
            is_last_row: selectors.is_last_row,
            is_transition: selectors.is_transition,
            alpha,
            expected_evals,
            accumulator: SC::Challenge::zero(),
            evaluation_point: eval_point,
            batch_challenge,
            _marker: PhantomData,
        };

        air.eval(&mut folder);

        folder.accumulator
    }

    /// Recomputes the quotient for a chip and opening.
    pub fn recompute_quotient(
        opening: &ChipOpenedValues<SC::Challenge>,
        qc_domains: &[Dom<SC>],
        zeta: SC::Challenge,
    ) -> SC::Challenge {
        use spl_algebra::Field;

        let zps = qc_domains
            .iter()
            .enumerate()
            .map(|(i, domain)| {
                qc_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other_domain)| {
                        other_domain.zp_at_point(zeta)
                            * other_domain.zp_at_point(domain.first_point()).inverse()
                    })
                    .product::<SC::Challenge>()
            })
            .collect_vec();

        opening
            .quotient
            .iter()
            .enumerate()
            .map(|(ch_i, ch)| {
                assert_eq!(ch.len(), SC::Challenge::D);
                ch.iter()
                    .enumerate()
                    .map(|(e_i, &c)| zps[ch_i] * SC::Challenge::monomial(e_i) * c)
                    .sum::<SC::Challenge>()
            })
            .sum::<SC::Challenge>()
    }
}

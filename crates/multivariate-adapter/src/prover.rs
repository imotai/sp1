use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::iter::once;

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    stack::VerticalPair,
    Matrix,
};
use p3_uni_stark::{PackedChallenge, PackedVal, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;

use spl_algebra::{AbstractExtensionField, AbstractField, PackedValue};
use spl_multi_pcs::Point;

use crate::{
    air_types::{AirOpenedValues, ChipOpenedValues},
    folder::ProverConstraintFolder,
    types::{
        generate_adapter_trace, MultivariateAdapterAir, MultivariateAdapterPCS,
        MultivariateAdapterProof,
    },
};

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

type Dom<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

type ProverData<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::ProverData;

type PointAndEvals<SC> =
    (Point<<SC as StarkGenericConfig>::Challenge>, Vec<<SC as StarkGenericConfig>::Challenge>);

pub struct AdapterProver<SC: StarkGenericConfig> {
    pub(crate) pcs: MultivariateAdapterPCS<SC>,
}

pub const LOG_QUOTIENT_DEGREE: usize = 1;

#[allow(clippy::type_complexity)]
impl<SC: StarkGenericConfig> AdapterProver<SC> {
    pub fn new(pcs: MultivariateAdapterPCS<SC>) -> Self {
        Self { pcs }
    }

    pub fn prove_evaluation(
        &self,
        data: RowMajorMatrix<Val<SC>>,
        eval_point: Point<SC::Challenge>,
        expected_evals: Vec<SC::Challenge>,
        prover_data: ProverData<SC>,
        main_commit: Com<SC>,
        challenger: &mut SC::Challenger,
    ) -> MultivariateAdapterProof<SC> {
        let log_degree = log2_strict_usize(data.height());
        let degree = 1 << log_degree;

        // Observe the main commitment.
        challenger.observe(main_commit.clone());

        let trace_domain = self.pcs.config.pcs().natural_domain_for_degree(degree);

        let log_quotient_degree = LOG_QUOTIENT_DEGREE;

        let batch_randomness: SC::Challenge = challenger.sample_ext_element();
        let adapter_trace = generate_adapter_trace::<SC>(&data, &eval_point, batch_randomness);

        let parent_span = tracing::debug_span!("commit adapter trace");
        let (adapter_trace_commit, adapter_trace_data) = parent_span.in_scope(|| {
            self.pcs.config.pcs().commit(vec![(trace_domain, adapter_trace.flatten_to_base())])
        });

        challenger.observe(adapter_trace_commit.clone());

        // Compute the quotient polynomial for the multivariate adapter.

        let quotient_domain =
            trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));

        let parent_span = tracing::debug_span!("compute ldes");
        let (data_on_quotient_domain, adapter_trace_on_quotient_domain) =
            parent_span.in_scope(|| {
                (
                    self.pcs
                        .config
                        .pcs()
                        .get_evaluations_on_domain(&prover_data, 0, quotient_domain)
                        .to_row_major_matrix(),
                    self.pcs
                        .config
                        .pcs()
                        .get_evaluations_on_domain(&adapter_trace_data, 0, quotient_domain)
                        .to_row_major_matrix(),
                )
            });

        // The constraint folding challenge.
        let alpha: SC::Challenge = challenger.sample_ext_element::<SC::Challenge>();

        // Compute the quotient values.
        let quotient_values = tracing::debug_span!("compute quotient values").in_scope(|| {
            Self::quotient_values(
                MultivariateAdapterAir { batch_size: data.width() },
                trace_domain,
                quotient_domain,
                &(eval_point.clone(), expected_evals),
                data_on_quotient_domain,
                adapter_trace_on_quotient_domain,
                [alpha, batch_randomness],
            )
        });

        // Split the quotient values and commit to them.
        let quotient_degree = 1 << log_quotient_degree;
        let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
        let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
        let qc_domains = quotient_domain.split_domains(quotient_degree);

        let num_quotient_chunks = quotient_chunks.len();

        let parent_span = tracing::debug_span!("commit quotient values");
        let (quotient_commit, quotient_data) = parent_span.in_scope(|| {
            self.pcs.config.pcs().commit(qc_domains.into_iter().zip(quotient_chunks).collect())
        });
        challenger.observe(quotient_commit.clone());

        // Compute the quotient argument.
        let zeta: SC::Challenge = challenger.sample_ext_element();

        let trace_opening_points = vec![vec![zeta, trace_domain.next_point(zeta).unwrap()]];

        // Compute quotient opening points, open every chunk at zeta.
        let quotient_opening_points =
            (0..num_quotient_chunks).map(|_| vec![zeta]).collect::<Vec<_>>();

        let parent_span = tracing::debug_span!("open commitments");
        let (openings, opening_proof) = parent_span.in_scope(|| {
            self.pcs.config.pcs().open(
                vec![
                    (&prover_data, trace_opening_points.clone()),
                    (&adapter_trace_data, trace_opening_points),
                    (&quotient_data, quotient_opening_points),
                ],
                challenger,
            )
        });

        // Collect the opened values for each chip.
        let [main_values, adapter_values, mut quotient_values] = openings.try_into().unwrap();
        assert!(main_values.len() == 1);

        let main_opened_values = main_values
            .into_iter()
            .map(|op| {
                let [local, next] = op.try_into().unwrap();
                AirOpenedValues { local, next }
            })
            .collect::<Vec<_>>();
        let adapter_opened_values = adapter_values
            .into_iter()
            .map(|op| {
                let [local, next] = op.try_into().unwrap();
                AirOpenedValues { local, next }
            })
            .collect::<Vec<_>>();
        let mut quotient_opened_values = Vec::new();
        let degree = 1 << log_quotient_degree;
        let slice = quotient_values.drain(0..degree);
        quotient_opened_values.push(slice.map(|mut op| op.pop().unwrap()).collect::<Vec<_>>());

        let opened_values = main_opened_values
            .into_iter()
            .zip_eq(adapter_opened_values)
            .zip_eq(quotient_opened_values)
            .zip_eq(once(log_degree))
            .map(|(((main, adapter), quotient), log_degree)| ChipOpenedValues {
                main,
                adapter,
                quotient,
                log_degree,
            })
            .collect::<Vec<_>>();

        MultivariateAdapterProof {
            adapter_commit: adapter_trace_commit,
            quotient_commit,
            opening_proof,
            opened_values,
        }
    }

    /// Compute the evaluations of the quotient polynomial for the multivariate adapter.
    fn quotient_values<A, Mat>(
        air: A,
        trace_domain: Dom<SC>,
        quotient_domain: Dom<SC>,
        point_and_evals: &PointAndEvals<SC>,
        trace_on_quotient_domain: Mat,
        adapter_trace_on_quotient_domain: Mat,
        random_challenges: [SC::Challenge; 2],
    ) -> Vec<SC::Challenge>
    where
        A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
        Mat: Matrix<Val<SC>> + Sync,
    {
        let (alpha, batch_challenge) = (random_challenges[0], random_challenges[1]);
        let quotient_size = quotient_domain.size();
        let main_width = trace_on_quotient_domain.width();
        let adapter_width = adapter_trace_on_quotient_domain.width();
        let sels = trace_domain.selectors_on_coset(quotient_domain);

        let (eval_point, expected_evals) = point_and_evals;

        let qdb =
            log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
        let next_step = 1 << qdb;

        let ext_degree = SC::Challenge::D;

        assert!(
            quotient_size >= PackedVal::<SC>::WIDTH,
            "quotient size is too small: got {}, expected at least {}",
            quotient_size,
            PackedVal::<SC>::WIDTH,
        );

        (0..quotient_size)
            .into_par_iter()
            .step_by(PackedVal::<SC>::WIDTH)
            .flat_map_iter(|i_start| {
                let wrap = |i| i % quotient_size;
                let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

                let is_first_row =
                    *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
                let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
                let is_transition =
                    *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
                let inv_zeroifier =
                    *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range.clone()]);

                let local: Vec<_> = (0..main_width)
                    .map(|col| {
                        PackedVal::<SC>::from_fn(|offset| {
                            trace_on_quotient_domain.get(wrap(i_start + offset), col)
                        })
                    })
                    .collect();
                let next: Vec<_> = (0..main_width)
                    .map(|col| {
                        PackedVal::<SC>::from_fn(|offset| {
                            trace_on_quotient_domain.get(wrap(i_start + next_step + offset), col)
                        })
                    })
                    .collect();

                let adapter_local: Vec<_> = (0..adapter_width)
                    .step_by(ext_degree)
                    .map(|col| {
                        PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                adapter_trace_on_quotient_domain
                                    .get(wrap(i_start + offset), col + i)
                            })
                        })
                    })
                    .collect();

                let adapter_next: Vec<_> = (0..adapter_width)
                    .step_by(ext_degree)
                    .map(|col| {
                        PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                adapter_trace_on_quotient_domain
                                    .get(wrap(i_start + next_step + offset), col + i)
                            })
                        })
                    })
                    .collect();

                let accumulator = PackedChallenge::<SC>::zero();
                let mut folder = ProverConstraintFolder {
                    main: VerticalPair::new(
                        RowMajorMatrixView::new_row(&local),
                        RowMajorMatrixView::new_row(&next),
                    ),
                    adapter: VerticalPair::new(
                        RowMajorMatrixView::new_row(&adapter_local),
                        RowMajorMatrixView::new_row(&adapter_next),
                    ),
                    is_first_row,
                    is_last_row,
                    is_transition,
                    alpha,
                    accumulator,
                    expected_evals: expected_evals
                        .iter()
                        .copied()
                        .map(PackedChallenge::<SC>::from_f)
                        .collect(),
                    evaluation_point: &eval_point.0,
                    batch_challenge: PackedChallenge::<SC>::from_f(batch_challenge),
                };
                let span = tracing::debug_span!("eval constraints");
                span.in_scope(|| air.eval(&mut folder));

                // quotient(x) = constraints(x) / Z_H(x)
                let quotient = folder.accumulator * inv_zeroifier;

                // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
                (0..PackedVal::<SC>::WIDTH).map(move |idx_in_packing| {
                    let quotient_value = (0..<SC::Challenge as AbstractExtensionField<Val<SC>>>::D)
                        .map(|coeff_idx| {
                            quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing]
                        })
                        .collect::<Vec<_>>();
                    SC::Challenge::from_base_slice(&quotient_value)
                })
            })
            .collect()
    }

    pub fn commit(&self, data: RowMajorMatrix<Val<SC>>) -> (Com<SC>, ProverData<SC>) {
        let domain = self.pcs.config.pcs().natural_domain_for_degree(data.height());
        self.pcs.config.pcs().commit(vec![(domain, data)])
    }
}

#[cfg(test)]
pub mod tests {

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_fri::{FriConfig, TwoAdicFriPcs};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_uni_stark::StarkConfig;
    use rand::{thread_rng, Rng};
    use spl_algebra::{
        extension::BinomialExtensionField, AbstractExtensionField, AbstractField, Field,
    };
    use spl_multi_pcs::{Mle, MultilinearPcs, Point};
    use spl_utils::setup_logger;

    use crate::prover::AdapterProver;

    use super::MultivariateAdapterPCS;

    type Val = BabyBear;
    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    type Challenge = BinomialExtensionField<Val, 4>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Dft = Radix2DitParallel;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    #[test]
    fn test_generate_adapter_trace() {
        let data = vec![Val::one(), Val::one()];
        let eval_point = Point::new(vec![Val::two()]);
        let trace = crate::types::generate_adapter_trace::<MyConfig>(
            &RowMajorMatrix::new(data, 1),
            &Point(eval_point.0.iter().map(|x| Challenge::from_base(*x)).collect()),
            Challenge::zero(),
        );
        println!("{:?}", trace);
    }

    fn test_adapter_stark_batch_size<const BATCH_SIZE: usize, const NUM_VARIABLES: usize>() {
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut thread_rng(),
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let dft = Dft {};

        let mut batch_vals = vec![];
        for _ in 0..BATCH_SIZE {
            let vals = (0..(1 << NUM_VARIABLES)).map(|_| thread_rng().gen()).collect_vec();
            batch_vals.push(vals);
        }
        let eval_point: Point<Challenge> =
            Point::new(std::array::from_fn::<_, NUM_VARIABLES, _>(|_| thread_rng().gen()).to_vec());
        let fri_config = FriConfig {
            log_blowup: 2,
            num_queries: 28,
            proof_of_work_bits: 8,
            mmcs: challenge_mmcs,
        };
        let pcs = Pcs::new(NUM_VARIABLES, dft, val_mmcs, fri_config);
        let config = MyConfig::new(pcs);
        let mut challenger = Challenger::new(perm.clone());

        let pcs = MultivariateAdapterPCS { config, batch_size: BATCH_SIZE };

        let prover = AdapterProver::new(pcs);

        let flattened = RowMajorMatrix::new(
            batch_vals.clone().into_iter().flatten().collect(),
            1 << NUM_VARIABLES,
        )
        .transpose();

        let (commitment, data) = prover.commit(flattened.clone());

        let mles = batch_vals.into_iter().map(Into::into).collect_vec();

        let expected_evals = Mle::eval_batch_at_point(&mles, &eval_point);

        let proof = tracing::debug_span!("prove opening").in_scope(|| {
            prover.prove_evaluation(
                flattened,
                Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
                expected_evals.clone(),
                data,
                commitment,
                &mut challenger,
            )
        });

        tracing::debug_span!("verify opening proof").in_scope(|| {
            prover
                .pcs
                .verify(
                    Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
                    expected_evals,
                    commitment,
                    &proof,
                    &mut Challenger::new(perm.clone()),
                )
                .unwrap()
        });
    }

    #[test]
    fn test_adapter_stark() {
        setup_logger();
        test_adapter_stark_batch_size::<1, 8>();
        test_adapter_stark_batch_size::<2, 8>();
        test_adapter_stark_batch_size::<4, 7>();
        test_adapter_stark_batch_size::<25, 18>();
    }

    #[test]
    #[should_panic]
    fn test_adapter_stark_fails_on_non_matching_commitment() {
        setup_logger();
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut thread_rng(),
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let dft = Dft {};

        let vals = std::array::from_fn::<BabyBear, 8, _>(|_| thread_rng().gen()).to_vec();

        let eval_point: Point<Challenge> =
            Point::new(std::array::from_fn::<_, 3, _>(|_| thread_rng().gen()).to_vec());
        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 28,
            proof_of_work_bits: 8,
            mmcs: challenge_mmcs,
        };
        let pcs = Pcs::new(3, dft, val_mmcs, fri_config);
        let config = MyConfig::new(pcs);
        let mut challenger = Challenger::new(perm.clone());

        let pcs = MultivariateAdapterPCS { config, batch_size: 2 };

        let prover = AdapterProver::new(pcs);

        let (commit, data) = prover.commit(RowMajorMatrix::new(vals.clone(), 2));
        let mle = Mle::new(vals);
        let expected_eval = mle.eval_at_point(&eval_point);

        let proof = tracing::debug_span!("prove opening").in_scope(|| {
            prover.prove_evaluation(
                RowMajorMatrix::new(mle.into(), 1),
                Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
                vec![Challenge::from_base(expected_eval)],
                data,
                commit,
                &mut challenger,
            )
        });

        tracing::debug_span!("verify opening").in_scope(|| {
            prover
                .pcs
                .verify(
                    Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
                    // Put a wrong value here to make sure the verification fails.
                    vec![Challenge::from_base(Val::from_canonical_u16(0xDEAD))],
                    commit,
                    &proof,
                    &mut Challenger::new(perm.clone()),
                )
                .unwrap()
        });
    }
}

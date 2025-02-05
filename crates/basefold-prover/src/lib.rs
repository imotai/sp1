mod configs;

pub use configs::*;

use std::fmt::Debug;

use itertools::Itertools;
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::{ExtensionField, Field, TwoAdicField};
use slop_basefold::{BaseFoldPcs, BaseFoldProof};
use slop_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use slop_commit::{ExtensionMmcs, Mmcs};
use slop_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use slop_fri::{fold_even_odd, CommitPhaseProofStep, FriConfig, PowersReducer, QueryProof};
use slop_matrix::{
    bitrev::BitReversableMatrix,
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use slop_multilinear::{
    MainTraceProverData, Mle, MultilinearPcsBatchProver, MultilinearPcsBatchVerifier, Point,
};

pub type MatrixOpening<F> = Vec<F>;

#[derive(Debug, Clone)]
pub struct BaseFoldProver<K: TwoAdicField, EK: ExtensionField<K>, InnerMmcs: Mmcs<K>, Challenger>
where
    Challenger: GrindingChallenger
        + FieldChallenger<K>
        + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>,
    <ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment: Debug,
{
    pub(crate) pcs: BaseFoldPcs<K, EK, InnerMmcs, Challenger>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BaseFoldProverData<K, ProverData> {
    vals: Vec<RowMajorMatrix<K>>,
    pub data: ProverData,
}

impl<K: Clone, ProverData> MainTraceProverData<RowMajorMatrix<K>>
    for BaseFoldProverData<K, ProverData>
{
    type BaseProverData = ProverData;
    fn split_off_main_traces(&self) -> (&ProverData, &[RowMajorMatrix<K>]) {
        (&self.data, &self.vals)
    }
}

#[allow(clippy::type_complexity)]
impl<K: TwoAdicField, EK: TwoAdicField + ExtensionField<K>, InnerMmcs: Mmcs<K>, Challenger>
    BaseFoldProver<K, EK, InnerMmcs, Challenger>
where
    Challenger: GrindingChallenger
        + FieldChallenger<K>
        + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>,
    <ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment: Debug,
{
    pub fn new(pcs: BaseFoldPcs<K, EK, InnerMmcs, Challenger>) -> Self {
        Self { pcs }
    }

    /// Considering the columns of the matrices in `vals` as univariate polynomials in the evaluation
    /// basis, compute and commit to the LDE of these polynomials.
    pub fn commit(
        &self,
        vals: Vec<RowMajorMatrix<K>>,
    ) -> (InnerMmcs::Commitment, BaseFoldProverData<K, InnerMmcs::ProverData<RowMajorMatrix<K>>>)
    {
        // The parameter `vals` is the vector of coefficients of a `vals.width` univariate polynomials
        // of degree < d, and we compute the evluations of this polynomial on a domain of size
        // `config.blowup()*d`

        // Extend vals by zero before doing an FFT.
        let to_commit = tracing::info_span!("zero pad").in_scope(|| {
            vals.iter()
                .cloned()
                .map(|mut mat| {
                    mat.values
                        .resize(mat.values.len() << self.pcs.fri_config().log_blowup, K::zero());
                    mat
                })
                .collect::<Vec<_>>()
        });

        // Compute the evaluations of `vals` on the appropriate domain.
        let mats = tracing::info_span!("compute LDE").in_scope(|| {
            to_commit
                .into_iter()
                .map(|mat| Radix2DitParallel.dft_batch(mat).bit_reverse_rows())
                .collect::<Vec<_>>()
        });

        let (commit, data) =
            tracing::info_span!("Mmcs commit").in_scope(|| self.pcs.inner_mmcs().commit(mats));

        (commit, BaseFoldProverData { vals, data })
    }

    /// Given a collection of matrices `data.vals`, a point `eval_point`, and claimed evaluations
    /// `expected_evals`, generate a proof that the evaluations of the multilinear extensions of the
    /// columns of the matrices in `data.vals` at `eval_point` are as claimed.
    ///
    /// We batch the claims into a single one about the random linear combination of all the columns
    /// of the matrices in `data.vals`, and the corresponding random linear combination of the
    /// evaluation claims, and run BaseFold on the single polynomial.
    ///
    ///
    /// Thinking of `vals` as a univariate polynomial in the coefficient basis, the prover commits
    /// to the Reed-Solomon encoding of `vals` (as a vector). Letting g be the multilinear
    /// extension and `X_0, ... , X_{d-1}` be the coordinates of `eval_point`, the prover:
    /// 1. Computes `g(X_0, X_1, ..., X_{d-1}, 0)` and `g(X_0, X_1, ..., X_{d-1}, 1)`, and sends its
    ///    claims about these openings as messages to the verifier.
    /// 2. Takes a random linear combination of the above two messages to produce an equivalent
    ///    evaluation claim about a multilinear polynomial in one fewer variable. On the univariate
    ///    side, this corresponds to folding in FRI, and the prover sends a commitment to the folded
    ///    univariate polynomial.
    /// 3. Repeats the two above steps until reduced to a claim about a 0-variate multilinear and a
    ///    degree 0-univariate polynomial, which should agree. It sends the value of this constant
    ///    polynomial.
    /// 4. Answers queries to its vector commitments as in FRI.
    ///
    /// # Panics
    /// Panics if the heights of the matrices are not equal.
    pub fn prove_evaluations(
        &self,
        data: Vec<&BaseFoldProverData<K, InnerMmcs::ProverData<RowMajorMatrix<K>>>>,
        mut eval_point: Point<EK>,
        expected_evals: &[&[&[EK]]],
        challenger: &mut Challenger,
    ) -> BaseFoldProof<K, EK, InnerMmcs::Commitment, Challenger::Witness, InnerMmcs::Proof, InnerMmcs>
    where
        Challenger: GrindingChallenger
            + FieldChallenger<K>
            + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>,
        <ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment: Debug,
    {
        // Get the RS-encoded columns of the original matrices.
        let matrices = data
            .iter()
            .flat_map(|round_data| {
                self.pcs
                    .inner_mmcs()
                    .get_matrices(&round_data.data)
                    .into_iter()
                    .map(|m| m.as_view())
            })
            .collect::<Vec<_>>();

        // Get the original matrices.
        let orig_matrices = data
            .iter()
            .flat_map(|round_data| round_data.vals.iter().map(|m| m.as_view()))
            .collect::<Vec<_>>();

        // Ensure all matrices have the same height.
        assert!(matrices.iter().map(|mat| mat.height()).unique().count() == 1);
        let encoded_height = matrices[0].height();

        let batching_challenge: EK = challenger.sample_ext_element();

        // Precompute all the powers of the batching challenge that will be used.
        let total_width = matrices.iter().map(|mat| mat.width()).sum::<usize>();
        let batch_challenge_powers =
            batching_challenge.powers().take(total_width).collect::<Vec<_>>();

        let batch_reducer = PowersReducer::new(batching_challenge, total_width);
        // Batch all the evaluation claims together in a random linear combination.
        let mut current_batched_eval_claim: EK =
            tracing::info_span!("batch evals").in_scope(|| {
                expected_evals
                    .iter()
                    .flat_map(|eval_set| eval_set.iter())
                    .flat_map(|eval| eval.iter())
                    .zip(batch_challenge_powers.clone())
                    .map(|(elem, power)| power * *elem)
                    .sum()
            });

        let mut curr_batch_power = 0;

        let mut current = vec![EK::zero(); encoded_height];

        // Batch all of the RS codewords into a single one.
        tracing::info_span!("batch encoded matrices").in_scope(|| {
            (0..matrices.len()).for_each(|i| {
                current
                    .par_iter_mut()
                    .zip_eq(matrices[i].values.par_chunks(matrices[i].width))
                    .for_each(|(a, row)| {
                        *a += batching_challenge.exp_u64(curr_batch_power as u64)
                            * batch_reducer.reduce_base(row)
                    });
                curr_batch_power += matrices[i].width();
            })
        });

        let mut curr_batch_power = 0;
        // Compute the random linear combination of the MLEs of the columns of the matrices.
        let mut current_mle_vec: Vec<EK> =
            vec![EK::zero(); encoded_height >> self.pcs.fri_config().log_blowup];
        tracing::info_span!("batch MLEs").in_scope(|| {
            orig_matrices.iter().for_each(|mat| {
                current_mle_vec.par_iter_mut().zip(mat.par_rows()).for_each(|(a, row)| {
                    *a += batching_challenge.exp_u64(curr_batch_power as u64)
                        * batch_reducer.reduce_base(row.collect::<Vec<_>>().as_slice())
                });
                curr_batch_power += mat.width();
            })
        });

        let mut current_mle: Mle<EK> = current_mle_vec.into();

        let log_len = current_mle.num_variables();

        // From this point on, run the BaseFold protocol on the random linear combination codeword,
        // the random linear combination multilinear, and the random linear combination of the
        // evaluation claims.

        // Initialize the vecs that go into a BaseFoldProof.
        let mut univariate_polys: Vec<[EK; 2]> = vec![];
        let mut commits = vec![];
        let mut commit_phase_data = vec![];

        for i in 0..eval_point.dimension() {
            // Compute claims for `g(X_0, X_1, ..., X_{d-1}, 0)` and `g(X_0, X_1, ..., X_{d-1}, 1)`.
            let last_coord = eval_point.remove_last_coordinate();
            let zero_val = tracing::info_span!("sum", round = i)
                .in_scope(|| current_mle.fixed_at_zero(&eval_point));
            let one_val = (current_batched_eval_claim - zero_val) / last_coord + zero_val;
            let uni_poly = [zero_val, one_val];
            univariate_polys.push(uni_poly);

            uni_poly.iter().for_each(|elem| challenger.observe_ext_element(*elem));

            // Perform a single round of the FRI commit phase, returning the commitment, folded
            // codeword, and folding parameter.
            let commit;
            let prover_data;
            let beta;
            (current, commit, prover_data, beta) = tracing::info_span!("commit phase", round = i)
                .in_scope(|| commit_phase_single_round(self.pcs.fri_config(), current, challenger));
            commits.push(commit);
            commit_phase_data.push(prover_data);

            // Combine the two halves (last variable evaluated to 0 and 1) into a single
            // multivariate to pass into the next round. Equivalent to the FRI-fold
            // stemp on the univariate side.
            current_mle =
                tracing::info_span!("fold mle", round = i).in_scope(|| current_mle.fold(beta));
            current_batched_eval_claim = zero_val + beta * one_val;
        }

        // As in FRI, the last codeword should be an encoding of a constant polynomial, with
        // `1<<log_blowup` entries.
        debug_assert_eq!(current.len(), 1 << self.pcs.fri_config().log_blowup);
        for elem in current[1..].iter() {
            debug_assert_eq!(*elem, current[0])
        }

        challenger.observe_ext_element(current[0]);

        let pow_witness = challenger.grind(self.pcs.fri_config().proof_of_work_bits);

        // FRI Query Phase.
        let query_indices: Vec<usize> = (0..self.pcs.fri_config().num_queries)
            .map(|_| challenger.sample_bits(log_len as usize + self.pcs.fri_config().log_blowup))
            .collect();

        let query_openings: Vec<Vec<(Vec<MatrixOpening<K>>, _)>> =
            tracing::info_span!("open queries").in_scope(|| {
                query_indices
                    .iter()
                    .map(|&index| {
                        data.iter()
                            .map(|round_data| {
                                self.pcs.inner_mmcs().open_batch(index, &round_data.data)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });

        let query_proofs = tracing::info_span!("answer queries").in_scope(|| {
            query_indices
                .iter()
                .map(|&index| answer_query(self.pcs.fri_config(), &commit_phase_data, index))
                .collect::<Vec<_>>()
        });

        BaseFoldProof {
            univariate_messages: univariate_polys,
            commitments: commits,
            query_phase_proofs: query_proofs,
            pow_witness,
            final_poly: current[0],
            query_openings,
        }
    }
}

impl<
        K: TwoAdicField,
        EK: TwoAdicField + ExtensionField<K>,
        InnerMmcs: Mmcs<K>,
        Challenger: GrindingChallenger
            + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>
            + FieldChallenger<K>,
    > MultilinearPcsBatchProver for BaseFoldProver<K, EK, InnerMmcs, Challenger>
where
    <ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment: Eq + Debug,
    <InnerMmcs as Mmcs<K>>::ProverData<RowMajorMatrix<K>>: Clone + Serialize + DeserializeOwned,
{
    type MultilinearProverData = BaseFoldProverData<K, InnerMmcs::ProverData<RowMajorMatrix<K>>>;

    type PCS = BaseFoldPcs<K, EK, InnerMmcs, Challenger>;

    fn commit_multilinears(
        &self,
        data: Vec<RowMajorMatrix<K>>,
    ) -> (<Self::PCS as MultilinearPcsBatchVerifier>::Commitment, Self::MultilinearProverData) {
        self.commit(data)
    }

    fn prove_trusted_evaluations(
        &self,
        eval_point: Point<EK>,
        expected_evals: &[&[&[EK]]],
        prover_data: Vec<&Self::MultilinearProverData>,
        challenger: &mut Challenger,
    ) -> <Self::PCS as MultilinearPcsBatchVerifier>::Proof {
        BaseFoldProver::prove_evaluations(self, prover_data, eval_point, expected_evals, challenger)
    }
}

/// A single round of the FRI commit phase. This function commits to the prover message, samples the
/// round challenge, and folds the polynomial. This function is repeated in the Plonky3 FRI prover
/// during the commit phase.
pub fn commit_phase_single_round<
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: CanObserve<M::Commitment> + FieldChallenger<F>,
>(
    config: &FriConfig<M>,
    current: Vec<EF>,
    challenger: &mut Challenger,
) -> (Vec<EF>, M::Commitment, M::ProverData<DenseMatrix<EF>>, EF) {
    // TODO: optimize this, so that we don't need to clone `current`. For example, we could compute
    // the two halves that go into folding, make `leaves` from `current`, then fold the two halves.
    let leaves = RowMajorMatrix::new(current.clone(), 2);
    let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
    challenger.observe(commit.clone());

    let beta: EF = challenger.sample_ext_element();
    (fold_even_odd(current, beta), commit, prover_data, beta)
}

/// Answer a single FRI query (open the vector commitment and the commit phase messages at a given
/// index).
///
/// Returns: the opening of the vector commitment at the given index, and a FRI query proof.
/// Modified from Plonky3 to be compatible with opening only a single vector.
pub fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> (F, QueryProof<F, M>)
where
    F: Field,
    M: Mmcs<F>,
{
    let mut opening = F::zero();
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index_pair, commit);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row[index_i_sibling % 2];
            if i == 0 {
                opening = opened_row[index_i % 2];
            }

            CommitPhaseProofStep { sibling_value, opening_proof }
        })
        .collect();

    (opening, QueryProof { commit_phase_openings })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::Rng;

    use slop_baby_bear::{my_perm, BabyBear, DiffusionMatrixBabyBear};
    use slop_challenger::{CanObserve, DuplexChallenger};
    use slop_commit::{ExtensionMmcs, Pcs};
    use slop_dft::Radix2DitParallel;
    use slop_fri::{FriConfig, TwoAdicFriPcs};
    use slop_jagged::MachineJaggedPcs;
    use slop_matrix::dense::RowMajorMatrix;
    use slop_merkle_tree::FieldMerkleTreeMmcs;
    use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use slop_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use slop_utils::log2_strict_usize;

    use slop_algebra::{extension::BinomialExtensionField, AbstractField, Field};
    use slop_basefold::BaseFoldPcs;
    use slop_multilinear::{
        Mle, MultilinearPcsBatchVerifier, MultilinearPcsProver, MultilinearPcsVerifier, Point,
        StackedPcsProver, StackedPcsVerifier,
    };
    use slop_utils::setup_logger;

    use crate::{default_jagged_basefold_config, BaseFoldProver};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        FieldMerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;
    type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    #[test]
    fn test_prover() {
        setup_logger();
        let mut rng = rand::thread_rng();
        let log_num_columns = 7;
        let num_columns = 1 << log_num_columns;
        let num_matrices = 3;

        (1..20).for_each(|num_variables| {
            println!("Testing an instance with {} variables.", num_variables);

            let vals = tracing::info_span!("construct big vecs").in_scope(|| {
                (0..num_matrices)
                    .map(|_| {
                        let mut vec = vec![F::zero(); 1 << (num_variables + log_num_columns)];
                        vec.iter_mut().for_each(|v| *v = rng.gen::<F>());
                        vec
                    })
                    .collect::<Vec<_>>()
            });
            let perm = Perm::new_from_rng_128(
                Poseidon2ExternalMatrixGeneral,
                DiffusionMatrixBabyBear,
                &mut rng,
            );
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let inner_mmcs = ValMmcs::new(hash, compress);
            let mmcs = ChallengeMmcs::new(inner_mmcs.clone());
            let config = FriConfig {
                log_blowup: 1,
                num_queries: 100,
                proof_of_work_bits: 8,
                mmcs: mmcs.clone(),
            };
            let cloned_config =
                FriConfig { log_blowup: 1, num_queries: 100, proof_of_work_bits: 8, mmcs };

            let pcs = BaseFoldPcs::<F, EF, ValMmcs, Challenger>::new(config, inner_mmcs.clone());

            let new_eval_point = (0..num_variables).map(|_| rng.gen::<EF>()).collect::<Point<_>>();

            let mats = tracing::info_span!("construct matrices").in_scope(|| {
                vals.into_iter()
                    .map(|vals| RowMajorMatrix::new(vals, num_columns))
                    .collect::<Vec<_>>()
            });

            let expected_evals = tracing::info_span!("evaluate matrices at point").in_scope(|| {
                mats.iter()
                    .map(|mat| Mle::eval_matrix_at_point(mat, &new_eval_point))
                    .collect::<Vec<_>>()
            });

            let prover = BaseFoldProver::new(pcs);

            let cloned_mats = mats.clone();

            let fri_pcs =
                TwoAdicFriPcs::<F, Radix2DitParallel, ValMmcs, ExtensionMmcs<F, EF, ValMmcs>>::new(
                    num_variables,
                    Radix2DitParallel,
                    inner_mmcs,
                    cloned_config,
                );

            let (_, fri_data) = tracing::info_span!("Plonky3 commit").in_scope(|| {
                let domain = <TwoAdicFriPcs<
                    F,
                    Radix2DitParallel,
                    ValMmcs,
                    ExtensionMmcs<F, EF, ValMmcs>,
                > as Pcs<EF, Challenger>>::natural_domain_for_degree(
                    &fri_pcs, 1<<num_variables
                );
                <TwoAdicFriPcs<
                    F,
                    Radix2DitParallel,
                    ValMmcs,
                    ExtensionMmcs<F, EF, ValMmcs>,
                > as Pcs<EF, Challenger>>::commit(&fri_pcs,cloned_mats.into_iter().map(|mat| (domain, mat)).collect())
            });

            let (commit_1, data_1) = tracing::info_span!("commit").in_scope(|| prover.commit(mats[0..2].to_vec()));
            let (commit_2, data_2) = tracing::info_span!("commit").in_scope(|| prover.commit(mats[2..].to_vec()));

            let proof = tracing::info_span!("prove evaluations").in_scope(|| {
                prover.prove_evaluations(
                    vec![&data_1, &data_2],
                    new_eval_point.clone(),
                    &[&expected_evals.iter().map(Vec::as_slice).collect::<Vec<_>>()],
                    &mut Challenger::new(perm.clone()),
                )
            });

            tracing::info_span!("prove Plonky3 evaluations").in_scope(|| {
                fri_pcs.open(
                    vec![(&fri_data, (0..num_matrices).map(|_| vec![rng.gen::<EF>()]).collect())],
                    &mut Challenger::new(perm.clone()),
                );
            });

            tracing::info_span!("verify evaluations").in_scope(|| {
                prover
                    .pcs
                    .verify_trusted_evaluations(
                        new_eval_point,
                        &[&expected_evals[0..2].iter().map(Vec::as_slice).collect::<Vec<_>>(), &expected_evals[2..].iter().map(Vec::as_slice).collect::<Vec<_>>()],
                        &[commit_1, commit_2],
                        &proof,
                        &mut Challenger::new(perm.clone()),
                    )
                    .unwrap();
            });
        });
    }

    #[test]
    fn test_stacked_prover() {
        setup_logger();
        let mut rng = rand::thread_rng();
        let num_columns = (1 << 5) + 1;
        let num_matrices = 4;
        let log_stacking_height = 21;

        // Test: 1 as a potential degenerate edge case, 12 and 13 because they are on opposite sides
        // of the tables being larger than 1<<log_stacking_height, and 20-22 because they are
        // representative of the real-world use case.
        [1, 12, 13, 14, 18].iter().for_each(|num_variables| {
            println!("Testing an instance with {} variables.", num_variables);

            let vals = tracing::info_span!("construct big vecs").in_scope(|| {
                (0..num_matrices)
                    .map(|_| {
                        let mut vec = vec![F::zero(); num_columns * (1 << num_variables)];
                        vec.iter_mut().for_each(|v| *v = rng.gen::<F>());
                        vec
                    })
                    .collect::<Vec<_>>()
            });

            let round_1 = vals[0..1].to_vec();
            let round_2 = vals[1..].to_vec();
            let perm = Perm::new_from_rng_128(
                Poseidon2ExternalMatrixGeneral,
                DiffusionMatrixBabyBear,
                &mut rng,
            );
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let inner_mmcs = ValMmcs::new(hash, compress);
            let mmcs = ChallengeMmcs::new(inner_mmcs.clone());
            let config = FriConfig {
                log_blowup: 1,
                num_queries: 100,
                proof_of_work_bits: 8,
                mmcs: mmcs.clone(),
            };
            let cloned_config =
                FriConfig { log_blowup: 1, num_queries: 100, proof_of_work_bits: 8, mmcs };

            let pcs = BaseFoldPcs::<F, EF, ValMmcs, Challenger>::new(config, inner_mmcs.clone());

            let pcs_clone =
                BaseFoldPcs::<F, EF, ValMmcs, Challenger>::new(cloned_config, inner_mmcs);

            let stacked_verifier = StackedPcsVerifier { pcs: pcs_clone, log_stacking_height };

            let total_area = (((round_1.len() * (1 << num_variables) * num_columns) as u64)
                .next_multiple_of(1 << log_stacking_height)
                + ((round_2.len() * (1 << num_variables) * num_columns) as u64)
                    .next_multiple_of(1 << log_stacking_height))
            .next_power_of_two() as usize;

            let new_eval_point =
                (0..log2_strict_usize(total_area)).map(|_| rng.gen::<EF>()).collect::<Point<_>>();

            let prover = BaseFoldProver::new(pcs);

            let stacked_prover = StackedPcsProver { pcs: prover, log_stacking_height };

            let (commit_1, data_1) = tracing::info_span!("commit")
                .in_scope(|| stacked_prover.commit_multilinear(round_1));

            let (commit_2, data_2) = tracing::info_span!("commit")
                .in_scope(|| stacked_prover.commit_multilinear(round_2));

            let proof = tracing::info_span!("prove evaluations").in_scope(|| {
                stacked_prover.prove_trusted_evaluation(
                    new_eval_point.clone(),
                    rng.gen::<EF>(),
                    vec![&data_1, &data_2],
                    &mut Challenger::new(perm.clone()),
                )
            });

            let (front_half, _) =
                new_eval_point.split_at(new_eval_point.dimension() - log_stacking_height);

            let eval_claim = proof
                .evaluations
                .iter()
                .flatten()
                .flatten()
                .cloned()
                .collect::<Mle<_>>()
                .eval_at(&front_half)[0];

            tracing::info_span!("verify evaluations").in_scope(|| {
                stacked_verifier
                    .verify_trusted_evaluation(
                        new_eval_point,
                        eval_claim,
                        &[commit_1, commit_2],
                        &proof,
                        &mut Challenger::new(perm.clone()),
                    )
                    .unwrap();
            });
        });
    }

    #[test]
    fn test_jagged_prover() {
        setup_logger();
        let column_counts = [1 << 1, 1 << 2, 1 << 7, 1 << 1];
        let row_counts = [1 << 13, 1 << 7, (1 << 19) + 7, 7];

        let log_stacking_height = 18;

        let log_max_row_count = 23;

        let batch_split_point = 1;

        let mut rng = rand::thread_rng();

        let matrices = column_counts
            .iter()
            .zip(row_counts.iter())
            .map(|(column_count, row_count)| {
                RowMajorMatrix::new(
                    (0..(*row_count * *column_count)).map(|_| rng.gen::<F>()).collect(),
                    *column_count,
                )
            })
            .collect::<Vec<_>>();

        let (jagged_prover, jagged_verifier) =
            default_jagged_basefold_config(log_stacking_height, log_max_row_count);

        let jagged_verifier = MachineJaggedPcs::new(
            jagged_verifier,
            vec![
                column_counts[0..batch_split_point].to_vec(),
                column_counts[batch_split_point..].to_vec(),
            ],
        );

        let mut challenger = Challenger::new(my_perm().clone());

        let (commit_1, data_1) =
            jagged_prover.commit_multilinears(matrices[0..batch_split_point].to_vec());

        challenger.observe(commit_1);

        let (commit_2, data_2) =
            jagged_prover.commit_multilinears(matrices[batch_split_point..].to_vec());

        challenger.observe(commit_2);

        let mut data = vec![Arc::new(data_1), Arc::new(data_2)];

        let mut commits = vec![commit_1, commit_2];

        jagged_prover.finalize(&mut data, &mut commits, &mut challenger);

        let eval_point = (0..log_max_row_count).map(|_| rng.gen::<EF>()).collect::<Point<_>>();

        let eval_claims = matrices
            .iter()
            .map(|mat| Mle::eval_matrix_at_point(mat, &eval_point))
            .collect::<Vec<_>>();

        // We pass a fresh challenger into the prover and verifier, though in production we need to
        // use one that has observed the commitments.
        let proof = jagged_prover.prove_trusted_evaluations(
            eval_point.clone(),
            &[&eval_claims.iter().map(Vec::as_slice).collect::<Vec<_>>()],
            &data,
            &mut challenger.clone(),
        );

        let result = jagged_verifier.verify_trusted_evaluations(
            eval_point,
            &[&eval_claims.iter().map(Vec::as_slice).collect::<Vec<_>>()],
            &commits,
            &proof,
            &mut challenger,
        );

        println!("Result: {:?}", result);

        assert!(result.is_ok());
    }
}

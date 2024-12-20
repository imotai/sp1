use std::fmt::Debug;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_fri::{fold_even_odd, CommitPhaseProofStep, FriConfig, QueryProof};
use p3_matrix::{
    bitrev::BitReversableMatrix,
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use spl_algebra::{ExtensionField, Field, TwoAdicField};
use spl_multi_pcs::MultilinearPcsProver;

use crate::{
    BaseFoldPcs, BaseFoldProof, BaseFoldProver, BaseFoldProverData, Point, RsBaseFoldConfig,
};

/// A single round of the FRI commit phase. This function commits to the prover message, samples the
/// round challenge, and folds the polynomial. This function is repeated in the Plonky3 FRI prover
/// during the commit phase.
pub fn commit_phase_single_round<
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: CanObserve<M::Commitment> + FieldChallenger<F>,
>(
    config: &RsBaseFoldConfig<M>,
    current: Vec<EF>,
    challenger: &mut Challenger,
) -> (Vec<EF>, M::Commitment, M::ProverData<DenseMatrix<EF>>, EF) {
    // TODO: optimize this, so that we don't need to clone `current`. For example, we could compute
    // the two halves that go into folding, make `leaves` from `current`, then fold the two halves.
    let leaves = RowMajorMatrix::new(current.clone(), 2);
    let (commit, prover_data) = config.fri_config.mmcs.commit_matrix(leaves);
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

#[allow(clippy::type_complexity)]
impl<K: TwoAdicField, M: Mmcs<K>, Challenger> BaseFoldProver<K, M, Challenger>
where
    Challenger: GrindingChallenger + FieldChallenger<K> + CanObserve<M::Commitment>,
    M::Commitment: Debug,
{
    pub fn new(pcs: BaseFoldPcs<K, M, Challenger>) -> Self {
        Self { pcs }
    }

    pub fn commit(
        &self,
        vals: Vec<K>,
    ) -> (M::Commitment, BaseFoldProverData<K, M::ProverData<RowMajorMatrix<K>>>) {
        // The parameter `vals` is the vector of coefficients of a univariate polynomial of degree
        // < d, and we compute the evluations of this polynomial on a domain of size
        // `config.blowup()*d`

        let len = vals.len();

        // Extend `vals` by zero to have the correct length.
        let to_commit = vals
            .clone()
            .into_iter()
            .chain(
                std::iter::repeat(K::zero())
                    .take(len * (1 << (self.pcs.config.fri_config.log_blowup - 1))),
            )
            .collect_vec();

        // Compute the evaluations of `vals` on the appropriate domain.
        let mut mat = Radix2Dit::default()
            .coset_dft_batch(RowMajorMatrix::new(to_commit, 1), K::one())
            .bit_reverse_rows()
            .to_row_major_matrix();

        mat.width = 2;

        let (commit, data) = self.pcs.config.fri_config.mmcs.commit_matrix(mat);

        (commit, BaseFoldProverData { vals: vals.into(), data })
    }

    /// Given a vector of evaluations of a polynomial `vals` at a point `eval_point`, the prover
    /// generates a proof about the evaluation of the multilinear extension of `vals` at
    /// `eval_point`.
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
    pub fn prove_evaluation(
        &self,
        data: BaseFoldProverData<K, M::ProverData<RowMajorMatrix<K>>>,
        // TODO: support batches.
        mut eval_point: Point<K>,
        _expected_eval: K,
        challenger: &mut Challenger,
    ) -> BaseFoldProof<K, M, Challenger::Witness>
    where
        Challenger: GrindingChallenger + FieldChallenger<K> + CanObserve<M::Commitment>,
        M::Commitment: Debug,
    {
        let matrices = self.pcs.config.fri_config.mmcs.get_matrices(&data.data);

        debug_assert_eq!(matrices.len(), 1);

        let mut current = matrices[0].values.clone();

        let mut current_mle = data.vals;

        let log_len = current_mle.num_variables();

        // Initialize the vecs that go into a BaseFoldProof.
        let mut univariate_polys: Vec<[K; 2]> = vec![];
        let mut commits = vec![];
        let mut data = vec![];

        for _ in 0..eval_point.dimension() {
            // Compute claims for `g(X_0, X_1, ..., X_{d-1}, 0)` and `g(X_0, X_1, ..., X_{d-1}, 1)`.
            // TODO: I think there's a lot repeated work between these rounds that can be cut out.
            eval_point.remove_last_coordinate();
            let uni_poly = current_mle.fixed_evaluations(&eval_point);
            univariate_polys.push(uni_poly);

            challenger.observe_slice(&uni_poly);

            // Perform a single round of the FRI commit phase, returning the commitment, folded
            // codeword, and folding parameter.
            let commit;
            let prover_data;
            let beta;
            (current, commit, prover_data, beta) =
                commit_phase_single_round(&self.pcs.config, current, challenger);
            commits.push(commit);
            data.push(prover_data);

            // Combine the two halves (last variable evaluated to 0 and 1) into a single
            // multivariate to pass into the next round. Equivalent to the FRI-fold
            // stemp on the univariate side.
            current_mle = current_mle.random_linear_combination(beta);
        }

        // As in FRI, the last codeword should be an encoding of a constant polynomial, with
        // `1<<log_blowup` entries.
        debug_assert_eq!(current.len(), 1 << self.pcs.config.fri_config.log_blowup);
        for elem in current[1..].iter() {
            debug_assert_eq!(*elem, current[0])
        }

        let pow_witness = challenger.grind(self.pcs.config.fri_config.proof_of_work_bits);

        // FRI Query Phase.
        let query_indices: Vec<usize> = (0..self.pcs.config.fri_config.num_queries)
            .map(|_| challenger.sample_bits(log_len + self.pcs.config.fri_config.log_blowup))
            .collect();

        let query_proofs = query_indices
            .iter()
            .map(|&index| answer_query(&self.pcs.config.fri_config, &data, index))
            .collect_vec();

        BaseFoldProof {
            univariate_messages: univariate_polys,
            commitments: commits,
            query_phase_proofs: query_proofs,
            pow_witness,
            final_poly: current[0],
        }
    }
}

impl<
        K: TwoAdicField,
        M: Mmcs<K>,
        Challenger: GrindingChallenger + CanObserve<M::Commitment> + FieldChallenger<K>,
    > MultilinearPcsProver<K, K, Challenger, BaseFoldPcs<K, M, Challenger>>
    for BaseFoldProver<K, M, Challenger>
where
    M::Commitment: Eq + Debug,
{
    type OpeningProof = BaseFoldProof<K, M, Challenger::Witness>;

    type MultilinearProverData = BaseFoldProverData<K, M::ProverData<RowMajorMatrix<K>>>;

    type MultilinearCommitment = M::Commitment;

    fn commit(
        &self,
        data: RowMajorMatrix<K>,
    ) -> (Self::MultilinearCommitment, Self::MultilinearProverData) {
        self.commit(data.values)
    }

    fn prove_evaluations(
        &self,
        eval_point: Point<K>,
        expected_evals: &[K],
        prover_data: Self::MultilinearProverData,
        challenger: &mut Challenger,
    ) -> Self::OpeningProof {
        BaseFoldProver::prove_evaluation(
            self,
            prover_data,
            eval_point,
            expected_evals[0],
            challenger,
        )
    }
}

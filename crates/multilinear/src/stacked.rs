//! API for stacked multilinear polynomial commitment schemes.
//!
//!
//! For any multilinear PCS that can commit to batches of matrices of the same height (considering
//! the columns of those matrices as evaluations of multilinear polynomials on the Boolean hypercube),
//! and then prove joint evaluations of those multililiner polynomials at the same point, this module
//! provides functionality that can commit to heterogeneous batches of matrices (considering that
//! batch as a single mutlilinear polynomial in many variables), and then prove evaluations of that
//! multilinear polynomial at a point.
//!
//! This is implemented by making a virtual vector consisting of the concatenation of all of the
//! data in the matrices in the batch, splitting that vector up into vectors of a prescribed size,
//! and then using the underlying PCS to commit to and prove evaluations of those vectors. The
//! verifier then computes the expected multilinear evaluation of the larger vector by using a
//! multilinear evaluation algorithm in a smaller number of variables). This is essentially the
//! the interleaving algorithm of `Ligero`(https://eprint.iacr.org/2022/1608).

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::Field;
use slop_matrix::dense::RowMajorMatrix;

use crate::{
    MainTraceProverData, Mle, MultilinearPcsBatchProver, MultilinearPcsBatchVerifier,
    MultilinearPcsProver, MultilinearPcsVerifier,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackedPcsVerifier<Pcs>
where
    Pcs: MultilinearPcsBatchVerifier,
{
    pub pcs: Pcs,
    pub log_stacking_height: usize,
}

impl<Pcs: MultilinearPcsBatchVerifier> StackedPcsVerifier<Pcs> {
    pub fn new(pcs: Pcs, log_stacking_height: usize) -> Self {
        Self { pcs, log_stacking_height }
    }
}

#[derive(Debug)]
pub enum StackedPcsError<Error> {
    PcsError(Error),
    StackingError,
}

pub struct StackedPcsProver<Pcs>
where
    Pcs: MultilinearPcsBatchProver,
    Pcs::MultilinearProverData:
        MainTraceProverData<RowMajorMatrix<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>>,
{
    pub pcs: Pcs,
    pub log_stacking_height: usize,
}

impl<Pcs> StackedPcsProver<Pcs>
where
    Pcs: MultilinearPcsBatchProver,
    Pcs::MultilinearProverData:
        MainTraceProverData<RowMajorMatrix<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>>,
{
    pub fn new(pcs: Pcs, log_stacking_height: usize) -> Self {
        Self { pcs, log_stacking_height }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackedPcsProof<PcsProof, EF> {
    pub pcs_proof: PcsProof,
    pub evaluations: Vec<Vec<Vec<EF>>>,
}

impl<Pcs> MultilinearPcsVerifier for StackedPcsVerifier<Pcs>
where
    Pcs: MultilinearPcsBatchVerifier,
    Pcs::EF: Field,
    Pcs::Proof: Clone + Serialize + DeserializeOwned,
{
    type F = Pcs::F;

    type EF = Pcs::EF;

    // "Hints" for the intermediate evaluations used in Blaze/Ligero-style proving, plus the proofs
    // that the intermediate evaluations are correct.
    type Proof = StackedPcsProof<Pcs::Proof, Pcs::EF>;

    type Commitment = Pcs::Commitment;

    type Error = StackedPcsError<Pcs::Error>;

    type Challenger = Pcs::Challenger;

    fn verify_trusted_evaluation(
        &self,
        point: crate::Point<Self::EF>,
        evaluation_claim: Self::EF,
        commitments: &[Self::Commitment],
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        let (front_half, back_half) = point.split_at(point.dimension() - self.log_stacking_height);

        let evaluations_mle =
            proof.evaluations.iter().flatten().flatten().cloned().collect::<Mle<_>>();
        if evaluation_claim != evaluations_mle.eval_at(&front_half)[0] {
            return Err(StackedPcsError::StackingError);
        }

        let inner_refs = proof
            .evaluations
            .iter()
            .map(|evals| evals.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let middle_refs = inner_refs.iter().map(Vec::as_slice).collect::<Vec<_>>();

        self.pcs
            .verify_untrusted_evaluations(
                back_half,
                &middle_refs,
                commitments,
                &proof.pcs_proof,
                challenger,
            )
            .map_err(StackedPcsError::PcsError)
    }
}

impl<Pcs: MultilinearPcsBatchProver> MultilinearPcsProver for StackedPcsProver<Pcs>
where
    Pcs::MultilinearProverData:
        MainTraceProverData<RowMajorMatrix<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>>,
    <Pcs::PCS as MultilinearPcsBatchVerifier>::Proof: Clone + Serialize + DeserializeOwned,
{
    type PCS = StackedPcsVerifier<Pcs::PCS>;

    type MultilinearProverData = Pcs::MultilinearProverData;

    fn commit_multilinear(
        &self,
        data: Vec<Vec<<Self::PCS as MultilinearPcsVerifier>::F>>,
    ) -> (<Self::PCS as MultilinearPcsVerifier>::Commitment, Self::MultilinearProverData) {
        let new_data = tracing::info_span!("restack matrices")
            .in_scope(|| restack_matrices(data, self.log_stacking_height));
        let new_data_vec: Vec<_> = new_data.collect();

        let (commit, data) = self.pcs.commit_multilinears(new_data_vec.clone());

        (commit, data)
    }

    fn prove_trusted_evaluation(
        &self,
        eval_point: crate::Point<<Self::PCS as MultilinearPcsVerifier>::EF>,
        _expected_eval: <Self::PCS as MultilinearPcsVerifier>::EF,
        prover_data: Vec<&Self::MultilinearProverData>,
        challenger: &mut <Self::PCS as MultilinearPcsVerifier>::Challenger,
    ) -> <Self::PCS as MultilinearPcsVerifier>::Proof {
        let (_, matrices) = prover_data
            .iter()
            .map(|data| data.split_off_main_traces())
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let (_, rest) = eval_point.split_at(eval_point.dimension() - self.log_stacking_height);

        // TODO: This may be computed at an earlier part of the stack, and we can refactor the API
        // to avoid recomputing it.
        let evaluations = tracing::info_span!("eval matrices at point").in_scope(|| {
            matrices
                .iter()
                .map(|mats| {
                    mats.iter().map(|mat| Mle::eval_matrix_at_point(mat, &rest)).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });

        let refs = evaluations
            .iter()
            .map(|evals| evals.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let middle_refs = refs.iter().map(Vec::as_slice).collect::<Vec<_>>();

        StackedPcsProof {
            pcs_proof: self.pcs.prove_untrusted_evaluations(
                rest,
                &middle_refs,
                prover_data,
                challenger,
            ),
            evaluations,
        }
    }
}

/// Given a `Vec` of matrices, consider them as a single long vector (concatenate all the underlying
/// `values` vectors) and split them into vectors of size `1 << log_stacking_height`, which are then
/// assembled into `RowMajorMatrix` instances with height `1 << log_stacking_height`.
fn restack_matrices<F: Copy + Default + Send + Sync>(
    matrices: Vec<Vec<F>>,
    log_stacking_height: usize,
) -> impl Iterator<Item = RowMajorMatrix<F>> + Clone {
    let mut out = vec![];
    let mut buffer: Vec<Vec<F>> = vec![];

    for mut matrix in matrices.into_iter() {
        // If the buffer is non-empty and a multiple of the stacking height, flush it.
        if needed_length(&buffer, log_stacking_height) == 0 {
            flush_buffer(&mut buffer, &mut out, log_stacking_height);
        }

        let needed_length = needed_length(&buffer, log_stacking_height);

        // If the current matrix is too small to fill the buffer, add it to the buffer and do nothing
        // else.
        if matrix.len() < needed_length {
            buffer.push(matrix);
        } else {
            // Split off the needed length from `matrix` and add it to the buffer. At this point,
            // there will be enough elements in the buffer to create a `RowMajorMatrix` of height
            // `1 << log_stacking_height`. If it's possible to throw in more multiples of
            // `1 << log_stacking_height` from `matrix`, do so.
            let num_further_multiples = (matrix.len() - needed_length) / (1 << log_stacking_height);
            let hi = matrix
                .split_off(needed_length + num_further_multiples * (1 << log_stacking_height));

            buffer.push(matrix);
            flush_buffer(&mut buffer, &mut out, log_stacking_height);

            // If there are any elements leftover after splitting, add them to the buffer and flush it.
            if { hi.len() } > 0 {
                buffer = vec![hi];
                flush_buffer(&mut buffer, &mut out, log_stacking_height);
            }
        }
    }

    // In case there are any elements left in the buffer, pad it with zeroes and flush it.
    let needed_zeroes = buffer
        .iter()
        .map(|mat| mat.len())
        .sum::<usize>()
        .next_multiple_of(1 << log_stacking_height)
        - buffer.iter().map(|mat| mat.len()).sum::<usize>();
    let new_vec = vec![F::default(); needed_zeroes];
    buffer.push(new_vec);
    flush_buffer(&mut buffer, &mut out, log_stacking_height);

    out.into_iter()
}

/// Given a buffer of matrices, flush it into a `RowMajorMatrix` if it has a total length that is a
/// non-zero multiple of `1 << log_stacking_height`.
fn flush_buffer<F: Clone + Default + Send + Sync>(
    buffer: &mut Vec<Vec<F>>,
    out: &mut Vec<RowMajorMatrix<F>>,
    log_stacking_height: usize,
) {
    let total_buffer_length = buffer.iter().map(|mat| mat.len()).sum::<usize>();
    // If the buffer contains enough elements to create a `RowMajorMatrix` of height
    // `1 << log_stacking_height`, then do so, add it to out, and clear the buffer.
    if total_buffer_length % (1 << log_stacking_height) == 0 && total_buffer_length > 0 {
        let new_mat = RowMajorMatrix::new(buffer.concat(), 1 << log_stacking_height);
        let new_mat = new_mat.transpose();

        out.push(new_mat);
        buffer.clear();
    }
}

/// Helper function to determine how many elements are needed in the before before it contains a
/// a multiple of `1 << log_stacking_height` elements.
fn needed_length<F>(buffer: &[Vec<F>], log_stacking_height: usize) -> usize {
    let next_multiple = buffer
        .iter()
        .map(|mat| mat.len())
        .sum::<usize>()
        .next_multiple_of(1 << log_stacking_height);

    let next_multiple = if next_multiple > 0 { next_multiple } else { 1 << log_stacking_height };
    next_multiple - buffer.iter().map(|mat| mat.len()).sum::<usize>()
}

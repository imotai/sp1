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
use std::iter::once;

use p3_matrix::dense::RowMajorMatrix;
use slop_algebra::{AbstractField, Field};

use crate::{
    MainTraceProverData, Mle, MultilinearPcsBatchProver, MultilinearPcsBatchVerifier,
    MultilinearPcsProver, MultilinearPcsVerifier,
};

pub struct StackedPcsVerifier<Pcs>
where
    Pcs: MultilinearPcsBatchVerifier,
{
    pub pcs: Pcs,
    pub log_stacking_height: usize,
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

impl<Pcs> MultilinearPcsVerifier for StackedPcsVerifier<Pcs>
where
    Pcs: MultilinearPcsBatchVerifier,
    Pcs::EF: Field,
{
    type F = Pcs::F;

    type EF = Pcs::EF;

    // "Hints" for the intermediate evaluations used in Blaze/Ligero-style proving, plus the proofs
    // that the intermediate evaluations are correct.
    type Proof = (Pcs::Proof, Vec<Vec<Pcs::EF>>);

    type Commitment = Pcs::Commitment;

    type Error = StackedPcsError<Pcs::Error>;

    type Challenger = Pcs::Challenger;

    fn verify_trusted_evaluation(
        &self,
        point: crate::Point<Self::EF>,
        evaluation_claim: Self::EF,
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        // For the stacked scheme, one only ever evaluates one polynomial at a time.

        // NOTE FOR JAGGED PCS: This corresponds to the identification of [0,2^{n-1}] with n-bit
        // strings where the most-significant bits are listed last.
        let (front_half, back_half) = point.split_at(self.log_stacking_height);

        if evaluation_claim
            != Mle::<Pcs::EF>::eval_at_point::<Pcs::EF>(
                &proof.1.iter().flatten().cloned().collect::<Vec<_>>().into(),
                &back_half,
            )
        {
            return Err(StackedPcsError::StackingError);
        }
        self.pcs
            .verify_untrusted_evaluations(
                front_half,
                &proof.1.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                commitment,
                &proof.0,
                challenger,
            )
            .map_err(StackedPcsError::PcsError)
    }
}

impl<Pcs: MultilinearPcsBatchProver> MultilinearPcsProver for StackedPcsProver<Pcs>
where
    Pcs::MultilinearProverData:
        MainTraceProverData<RowMajorMatrix<<Pcs::PCS as MultilinearPcsBatchVerifier>::F>>,
{
    type PCS = StackedPcsVerifier<Pcs::PCS>;

    type MultilinearProverData = Pcs::MultilinearProverData;

    type MultilinearCommitment = Pcs::MultilinearCommitment;

    fn commit_multilinear(
        &self,
        data: Vec<Vec<<Self::PCS as MultilinearPcsVerifier>::F>>,
    ) -> (Self::MultilinearCommitment, Self::MultilinearProverData) {
        let new_data = restack_matrices(data, self.log_stacking_height);
        let total_width = new_data.clone().map(|mat| mat.width).sum::<usize>();
        let new_data_vec: Vec<_> = if !total_width.is_power_of_two() {
            let extra_zeroes = (total_width * (1 << self.log_stacking_height)).next_power_of_two()
                - (total_width * (1 << self.log_stacking_height));
            new_data
                .chain(once(RowMajorMatrix::new(
                    vec![<Self::PCS as MultilinearPcsVerifier>::F::zero(); extra_zeroes],
                    (extra_zeroes) / (1 << self.log_stacking_height),
                )))
                .collect()
        } else {
            new_data.collect()
        };
        // assert!(new_data_vec.iter().map(|mat| mat.width).sum::<usize>().is_power_of_two());
        let (commit, data) = self.pcs.commit_multilinears(new_data_vec);

        (commit, data)
    }

    fn prove_trusted_evaluation(
        &self,
        eval_point: crate::Point<<Self::PCS as MultilinearPcsVerifier>::EF>,
        _expected_eval: <Self::PCS as MultilinearPcsVerifier>::EF,
        prover_data: Self::MultilinearProverData,
        challenger: &mut <Self::PCS as MultilinearPcsVerifier>::Challenger,
    ) -> <Self::PCS as MultilinearPcsVerifier>::Proof {
        let (prover_data, matrices) = prover_data.split_off_main_traces();
        let front_portion = eval_point.first_k_points(self.log_stacking_height);

        // TODO: This may be computed at an earlier part of the stack, and we can refactor the API
        // to avoid recomputing it.
        let evaluations = tracing::info_span!("eval matrices at point").in_scope(|| {
            matrices
                .iter()
                .map(|mat| Mle::eval_matrix_at_point(mat, &front_portion))
                .collect::<Vec<_>>()
        });

        (
            self.pcs.prove_untrusted_evaluations(
                front_portion,
                &evaluations.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                Self::MultilinearProverData::reconstitute(prover_data, matrices),
                challenger,
            ),
            evaluations,
        )
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
        let new_mat =
            RowMajorMatrix::new(buffer.concat(), total_buffer_length / (1 << log_stacking_height));
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

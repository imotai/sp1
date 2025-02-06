use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

/// Enum the will encapsulated either an owned `RowMajorMatrix` or a reference to one.
/// The zerocheck poly used in the sumcheck's first round will have a reference to the matrices, since
/// the trace will be needed for other parts of the prover (namely the opening proofs).  All other
/// zerocheck polys will have owned matrices, since they are the only users of those matrices.
pub(crate) enum ZeroCheckPolyVals<'a, K: Field> {
    Reference(&'a RowMajorMatrix<K>),
    Owned(RowMajorMatrix<K>),
}

impl<'a, K: Field> ZeroCheckPolyVals<'a, K> {
    /// Returns a reference to the underlying matrix.
    pub fn matrix_ref(&'a self) -> &'a RowMajorMatrix<K> {
        match self {
            ZeroCheckPolyVals::Reference(matrix) => matrix,
            ZeroCheckPolyVals::Owned(matrix) => matrix,
        }
    }
}

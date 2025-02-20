use p3_field::Field;
use slop_alloc::Backend;
use slop_multilinear::Mle;

/// Enum the will encapsulated either an owned `RowMajorMatrix` or a reference to one.
/// The zerocheck poly used in the sumcheck's first round will have a reference to the matrices, since
/// the trace will be needed for other parts of the prover (namely the opening proofs).  All other
/// zerocheck polys will have owned matrices, since they are the only users of those matrices.
pub(crate) enum ZeroCheckPolyVals<'a, K: Field, A: Backend> {
    Reference(&'a Mle<K, A>),
    Owned(Mle<K, A>),
}

impl<'a, K: Field, A: Backend> ZeroCheckPolyVals<'a, K, A> {
    /// Returns a reference to the underlying mle.
    pub fn mle_ref(&'a self) -> &'a Mle<K, A> {
        match self {
            ZeroCheckPolyVals::Reference(mle) => mle,
            ZeroCheckPolyVals::Owned(mle) => mle,
        }
    }
}

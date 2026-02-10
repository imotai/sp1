use slop_algebra::AbstractField;
use slop_alloc::CpuBackend;

use crate::Tensor;

/// Add a scalar value to all elements of a tensor in place.
pub fn add_assign<T: AbstractField>(lhs: &mut Tensor<T, CpuBackend>, rhs: T) {
    let lhs = lhs.as_mut_slice();
    for elem in lhs.iter_mut() {
        *elem += rhs.clone();
    }
}

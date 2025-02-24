use p3_field::Field;

/// An implementation of `batch_multiplicative_inverse` that operates in place.
#[allow(dead_code)]
pub fn batch_multiplicative_inverse_inplace<F: Field>(values: &mut [F]) {
    // Check if values are zero and construct a new vector with only nonzero values.
    let mut nonzero_values = Vec::with_capacity(values.len());
    let mut indices = Vec::with_capacity(values.len());
    for (i, value) in values.iter().copied().enumerate() {
        if value.is_zero() {
            continue;
        }
        nonzero_values.push(value);
        indices.push(i);
    }

    // Compute the multiplicative inverse of nonzero values.
    let inverse_nonzero_values = p3_field::batch_multiplicative_inverse(&nonzero_values);

    // Reconstruct the original vector.
    for (i, index) in indices.into_iter().enumerate() {
        values[index] = inverse_nonzero_values[i];
    }
}

pub(crate) fn block_on<T>(fut: impl std::future::Future<Output = T>) -> T {
    use tokio::task::block_in_place;

    // Handle case if we're already in an tokio runtime.
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        block_in_place(|| handle.block_on(fut))
    } else {
        // Otherwise create a new runtime.
        let rt = tokio::runtime::Runtime::new().expect("Failed to create a new runtime");
        rt.block_on(fut)
    }
}

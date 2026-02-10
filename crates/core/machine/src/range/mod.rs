pub mod air;
pub mod columns;
pub mod trace;

use core::borrow::BorrowMut;
use slop_algebra::Field;
use std::{marker::PhantomData, mem::MaybeUninit};

use self::columns::{RangePreprocessedCols, NUM_RANGE_PREPROCESSED_COLS};
use crate::range::trace::NUM_ROWS;

/// A chip for range checking a limb with maximum number of bits.
#[derive(Debug, Clone, Copy, Default)]
pub struct RangeChip<F>(PhantomData<F>);

impl<F: Field> RangeChip<F> {
    /// Creates the preprocessed range table trace.
    pub fn trace(buffer: &mut [MaybeUninit<F>]) {
        let buffer_ptr = buffer.as_mut_ptr() as *mut F;
        let values = unsafe {
            core::slice::from_raw_parts_mut(buffer_ptr, NUM_RANGE_PREPROCESSED_COLS * NUM_ROWS)
        };

        // Set the first row to (0, 0).
        let col: &mut RangePreprocessedCols<F> = values[0..2].borrow_mut();
        col.a = F::zero();
        col.bits = F::zero();

        // For `0 <= bits <= 16`, put `(a, bits)` with `0 <= a < 2^bits` into the trace.
        for bits in 0..=16 {
            for a in 0..(1 << bits) {
                let row_index = (1 << bits) + a;
                let col: &mut RangePreprocessedCols<F> =
                    values[row_index * 2..(row_index + 1) * 2].borrow_mut();
                col.a = F::from_canonical_usize(a);
                col.bits = F::from_canonical_usize(bits);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use sp1_primitives::SP1Field;
    use std::time::Instant;

    use super::*;

    #[test]
    pub fn test_trace_and_map() {
        let mut vec: Vec<SP1Field> = Vec::with_capacity(NUM_ROWS * NUM_RANGE_PREPROCESSED_COLS);
        let start = Instant::now();
        RangeChip::<SP1Field>::trace(vec.spare_capacity_mut());
        println!("trace and map: {:?}", start.elapsed());
    }
}

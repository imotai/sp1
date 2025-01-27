use csl_device::{Buffer, Tensor};
use slop_algebra::Field;

use crate::Point;

impl<F: Field> From<slop_multilinear::Point<F>> for Point<F> {
    fn from(point: slop_multilinear::Point<F>) -> Self {
        Self::new(Tensor::from(Buffer::from_vec(point.to_vec())))
    }
}

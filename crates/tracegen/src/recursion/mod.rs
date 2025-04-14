use sp1_recursion_machine::RecursionAir;

use crate::{CudaTracegenAir, F};

impl<const DEGREE: usize> CudaTracegenAir<F> for RecursionAir<F, DEGREE> {}

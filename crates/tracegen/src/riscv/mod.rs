mod global;

use csl_cuda::TaskScope;
use slop_alloc::mem::CopyError;
use slop_multilinear::Mle;
use sp1_core_machine::riscv::RiscvAir;

use crate::{CudaTracegenAir, F};

impl CudaTracegenAir<F> for RiscvAir<F> {
    fn supports_device_tracegen(&self) -> bool {
        match self {
            RiscvAir::Global(chip) => chip.supports_device_tracegen(),
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => false,
        }
    }

    async fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        scope: &TaskScope,
    ) -> Result<Mle<F, TaskScope>, CopyError> {
        match self {
            RiscvAir::Global(chip) => chip.generate_trace_device(input, output, scope).await,
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => unimplemented!(),
        }
    }
}

use csl_air::{codegen_cuda_eval, instruction::Instruction16, SymbolicProverFolder};
use csl_cuda::{
    sys::{
        runtime::KernelPtr,
        zerocheck::{
            constraint_poly_eval_1024_baby_bear_extension_kernel,
            constraint_poly_eval_1024_baby_bear_kernel,
            constraint_poly_eval_128_baby_bear_extension_kernel,
            constraint_poly_eval_128_baby_bear_kernel,
            constraint_poly_eval_256_baby_bear_extension_kernel,
            constraint_poly_eval_256_baby_bear_kernel,
            constraint_poly_eval_32_baby_bear_extension_kernel,
            constraint_poly_eval_32_baby_bear_kernel,
            constraint_poly_eval_512_baby_bear_extension_kernel,
            constraint_poly_eval_512_baby_bear_kernel,
            constraint_poly_eval_64_baby_bear_extension_kernel,
            constraint_poly_eval_64_baby_bear_kernel, interpolate_row_baby_bear_extension_kernel,
            interpolate_row_baby_bear_kernel,
        },
    },
    TaskScope,
};
use slop_air::Air;
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field};
use slop_alloc::{mem::CopyError, Backend, Buffer, CopyToBackend, CpuBackend, HasBackend};
use slop_baby_bear::BabyBear;
use sp1_stark::air::MachineAir;

pub trait InterpolateRowKernel<K: Field> {
    fn interpolate_row_kernel() -> KernelPtr;
}

impl InterpolateRowKernel<BabyBear> for TaskScope {
    fn interpolate_row_kernel() -> KernelPtr {
        unsafe { interpolate_row_baby_bear_kernel() }
    }
}

impl InterpolateRowKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn interpolate_row_kernel() -> KernelPtr {
        unsafe { interpolate_row_baby_bear_extension_kernel() }
    }
}

pub trait ConstraintPolyEvalKernel<K: Field> {
    fn constraint_poly_eval_kernel(memory_size: usize) -> KernelPtr;
}

impl ConstraintPolyEvalKernel<BabyBear> for TaskScope {
    fn constraint_poly_eval_kernel(memory_size: usize) -> KernelPtr {
        match memory_size {
            0..=32 => unsafe { constraint_poly_eval_32_baby_bear_kernel() },
            33..=64 => unsafe { constraint_poly_eval_64_baby_bear_kernel() },
            65..=128 => unsafe { constraint_poly_eval_128_baby_bear_kernel() },
            129..=256 => unsafe { constraint_poly_eval_256_baby_bear_kernel() },
            257..=512 => unsafe { constraint_poly_eval_512_baby_bear_kernel() },
            513..=1024 => unsafe { constraint_poly_eval_1024_baby_bear_kernel() },
            _ => unreachable!(),
        }
    }
}

impl ConstraintPolyEvalKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn constraint_poly_eval_kernel(memory_size: usize) -> KernelPtr {
        match memory_size {
            0..=32 => unsafe { constraint_poly_eval_32_baby_bear_extension_kernel() },
            33..=64 => unsafe { constraint_poly_eval_64_baby_bear_extension_kernel() },
            65..=128 => unsafe { constraint_poly_eval_128_baby_bear_extension_kernel() },
            129..=256 => unsafe { constraint_poly_eval_256_baby_bear_extension_kernel() },
            257..=512 => unsafe { constraint_poly_eval_512_baby_bear_extension_kernel() },
            513..=1024 => unsafe { constraint_poly_eval_1024_baby_bear_extension_kernel() },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct EvalProgram<F, EF, B: Backend = CpuBackend> {
    pub operations: Buffer<Instruction16, B>,
    pub f_ctr: u32,
    pub f_constants: Buffer<F, B>,
    pub ef_constants: Buffer<EF, B>,
}

impl EvalProgram<BabyBear, BinomialExtensionField<BabyBear, 4>> {
    pub fn compile<A>(air: &A) -> Self
    where
        A: MachineAir<BabyBear> + for<'a> Air<SymbolicProverFolder<'a>>,
    {
        let (operations, f_ctr, _, f_constants, ef_constants) = codegen_cuda_eval(air);
        let operations = Buffer::from(operations);
        let f_constants = Buffer::from(f_constants);
        let ef_constants = Buffer::from(ef_constants);
        Self { operations, f_ctr, f_constants, ef_constants }
    }
}

impl<F, EF, B: Backend> HasBackend for EvalProgram<F, EF, B> {
    type Backend = B;

    #[inline]
    fn backend(&self) -> &Self::Backend {
        self.operations.backend()
    }
}

impl<F, EF> CopyToBackend<TaskScope, CpuBackend> for EvalProgram<F, EF, CpuBackend>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Output = EvalProgram<F, EF, TaskScope>;

    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        Ok(EvalProgram {
            operations: self.operations.copy_to_backend(backend).await?,
            f_ctr: self.f_ctr,
            f_constants: self.f_constants.copy_to_backend(backend).await?,
            ef_constants: self.ef_constants.copy_to_backend(backend).await?,
        })
    }
}

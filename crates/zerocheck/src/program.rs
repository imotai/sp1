use csl_air::{
    air_block::BlockAir, codegen_cuda_eval, instruction::Instruction16, SymbolicProverFolder,
};
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
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field};
use slop_alloc::{mem::CopyError, Backend, Buffer, CopyToBackend, CpuBackend, HasBackend};
use slop_baby_bear::BabyBear;

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
    pub constraint_indices: Buffer<u32, B>,
    pub operations: Buffer<Instruction16, B>,
    pub operations_indices: Buffer<u32, B>,
    pub f_ctr: u32,
    pub f_constants: Buffer<F, B>,
    pub f_constants_indices: Buffer<u32, B>,
    pub ef_constants: Buffer<EF, B>,
    pub ef_constants_indices: Buffer<u32, B>,
}

impl EvalProgram<BabyBear, BinomialExtensionField<BabyBear, 4>> {
    pub fn compile<A>(air: &A) -> Self
    where
        A: for<'a> BlockAir<SymbolicProverFolder<'a>>,
    {
        let (
            constraint_indices,
            operations,
            operations_indices,
            f_constants,
            f_constants_indices,
            ef_constants,
            ef_constants_indices,
            f_ctr,
            _,
        ) = codegen_cuda_eval(air);
        let constraint_indices = Buffer::from(constraint_indices);
        let operations = Buffer::from(operations);
        let operations_indices = Buffer::from(operations_indices);
        let f_constants = Buffer::from(f_constants);
        let f_constants_indices = Buffer::from(f_constants_indices);
        let ef_constants = Buffer::from(ef_constants);
        let ef_constants_indices = Buffer::from(ef_constants_indices);
        Self {
            constraint_indices,
            operations,
            operations_indices,
            f_ctr,
            f_constants,
            f_constants_indices,
            ef_constants,
            ef_constants_indices,
        }
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
            constraint_indices: self.constraint_indices.copy_to_backend(backend).await?,
            operations: self.operations.copy_to_backend(backend).await?,
            operations_indices: self.operations_indices.copy_to_backend(backend).await?,
            f_ctr: self.f_ctr,
            f_constants: self.f_constants.copy_to_backend(backend).await?,
            f_constants_indices: self.f_constants_indices.copy_to_backend(backend).await?,
            ef_constants: self.ef_constants.copy_to_backend(backend).await?,
            ef_constants_indices: self.ef_constants_indices.copy_to_backend(backend).await?,
        })
    }
}

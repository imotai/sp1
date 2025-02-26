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
use slop_algebra::{extension::BinomialExtensionField, Field};
use slop_alloc::Buffer;
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

#[derive(Clone, Debug)]
pub struct EvalProgram<F, EF> {
    pub operations: Buffer<Instruction16>,
    pub f_ctr: u32,
    pub f_constants: Buffer<F>,
    pub ef_constants: Buffer<EF>,
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

use std::{collections::HashMap, marker::PhantomData, time::Instant};

use csl_air::{codegen_cuda_eval, instruction::Instruction16, SymbolicProverFolder};
use csl_sys::{
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
};
use once_cell::sync::Lazy;
use slop_air::Air;
use slop_algebra::{extension::BinomialExtensionField, Field};
use slop_baby_bear::BabyBear;
use sp1_core_machine::riscv::RiscvAir;
use sp1_stark::{
    air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig, StarkMachine,
};

use crate::TaskScope;

mod sum_as_poly;

static EVAL_PROGRAM_GENERATOR: Lazy<
    DeviceConstraintPolyEvalProgramGenerator<BabyBearPoseidon2, RiscvAir<BabyBear>>,
> = Lazy::new(|| {
    let start_time = Instant::now();
    let risc_v_machine = RiscvAir::<BabyBear>::machine(BabyBearPoseidon2::default());
    let risc_v_eval_program_generator =
        DeviceConstraintPolyEvalProgramGenerator::new(&risc_v_machine);
    let risc_v_time = start_time.elapsed();
    println!("RISC-V eval program generator time: {:?}", risc_v_time);
    risc_v_eval_program_generator
});

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

#[allow(clippy::type_complexity)]
#[derive(Clone, Debug)]
pub struct DeviceConstraintPolyEvalProgramGenerator<SC: StarkGenericConfig, A> {
    eval_programs: HashMap<String, (Vec<Instruction16>, u32, Vec<SC::Val>, Vec<SC::Challenge>)>,
    _marker: PhantomData<(SC, A)>,
}

#[allow(clippy::type_complexity)]
impl<SC, A> DeviceConstraintPolyEvalProgramGenerator<SC, A>
where
    SC: StarkGenericConfig<Val = BabyBear, Challenge = BinomialExtensionField<BabyBear, 4>>,
    A: MachineAir<BabyBear> + for<'a> Air<SymbolicProverFolder<'a>>,
{
    pub fn new(machine: &StarkMachine<SC, A>) -> Self {
        let mut eval_programs = HashMap::new();
        for chip in machine.chips() {
            let (operations, f_ctr, _, f_constants, ef_constants) = codegen_cuda_eval(chip);
            eval_programs
                .insert(chip.name().to_owned(), (operations, f_ctr, f_constants, ef_constants));
        }
        Self { eval_programs, _marker: PhantomData }
    }

    pub fn get_eval_program(
        &self,
        chip_name: &str,
    ) -> &(Vec<Instruction16>, u32, Vec<SC::Val>, Vec<SC::Challenge>) {
        self.eval_programs.get(chip_name).unwrap()
    }
}

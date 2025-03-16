use std::{ops::Mul, sync::Arc};

use crate::{
    args,
    sys::{logup_gkr::gkr_tracegen_kernel, runtime::KernelPtr},
    TaskScope,
};
use slop_air::PairCol;
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field, Powers, PrimeField};
use slop_alloc::{Backend, Buffer, CopyToBackend, CpuBackend, HasBackend};
use slop_baby_bear::BabyBear;
use slop_multilinear::{Mle, PaddedMle, Padding};
use slop_tensor::Tensor;
use sp1_stark::Interaction;

impl<F: Field> From<PairCol> for PairColDevice<F> {
    fn from(value: PairCol) -> Self {
        match value {
            PairCol::Preprocessed(column_idx) => {
                Self { column_idx, is_preprocessed: true, weight: F::one() }
            }
            PairCol::Main(column_idx) => {
                Self { column_idx, is_preprocessed: false, weight: F::one() }
            }
        }
    }
}

impl<F: Field> Mul<F> for PairColDevice<F> {
    type Output = PairColDevice<F>;

    fn mul(self, rhs: F) -> Self::Output {
        PairColDevice {
            column_idx: self.column_idx,
            is_preprocessed: self.is_preprocessed,
            weight: self.weight * rhs,
        }
    }
}

impl<F: Field> DeviceInteractions<F, CpuBackend> {
    pub fn new(sends: &[Interaction<F>], receives: &[Interaction<F>]) -> Self {
        let mut values_ptr = vec![];
        let mut values_col_weights_ptr = vec![];
        let mut multiplicities_ptr = vec![];
        let mut arg_indices = vec![];
        let mut is_send = vec![];
        let mut mult_col_weights = vec![];
        let mut mult_constants = vec![];
        let mut values_col_weights = vec![];
        let mut values_constants = vec![];

        let num_interactions = sends.len() + receives.len();

        let mut curr_values_ptr = 0;
        let mut curr_values_col_weight_ptr = 0;
        let mut curr_mult_ptr = 0;

        // Put all of the interactions (for both send/receives) into a single list.
        // The ordering of the interactions is important to match with the CPU prover's ordering.
        // It should local sends, local receives.
        let interactions = {
            let sends = sends.iter().map(move |i| (i, true));
            let receives = receives.iter().map(move |i| (i, false));
            sends.chain(receives)
        };

        for (interaction, is_send_flag) in interactions {
            // Register the values
            values_ptr.push(curr_values_ptr);
            for value in interaction.values.iter() {
                values_col_weights_ptr.push(curr_values_col_weight_ptr);
                for (col, weight) in value.column_weights.iter() {
                    let col = PairColDevice::<F>::from(*col) * *weight;
                    values_col_weights.push(col);
                    curr_values_col_weight_ptr += 1;
                }
                values_constants.push(value.constant);
                curr_values_ptr += 1;
            }

            // Register the multiplicity values
            multiplicities_ptr.push(curr_mult_ptr);
            for (col, weight) in interaction.multiplicity.column_weights.iter() {
                let col = PairColDevice::<F>::from(*col) * *weight;
                mult_col_weights.push(col);
                curr_mult_ptr += 1;
            }
            mult_constants.push(interaction.multiplicity.constant);

            arg_indices.push(F::from_canonical_usize(interaction.argument_index()));

            is_send.push(is_send_flag);
        }

        values_col_weights_ptr.push(curr_values_col_weight_ptr);
        values_ptr.push(curr_values_ptr);
        multiplicities_ptr.push(curr_mult_ptr);

        Self {
            values_ptr: values_ptr.into(),
            values_col_weights_ptr: values_col_weights_ptr.into(),
            multiplicities_ptr: multiplicities_ptr.into(),
            values_col_weights: values_col_weights.into(),
            values_constants: values_constants.into(),
            mult_col_weights: mult_col_weights.into(),
            mult_constants: mult_constants.into(),
            arg_indices: arg_indices.into(),
            is_send: is_send.into(),
            num_interactions,
        }
    }
}

impl<F: Field, A: Backend> HasBackend for DeviceInteractions<F, A> {
    type Backend = A;

    fn backend(&self) -> &Self::Backend {
        self.values_col_weights_ptr.backend()
    }
}

impl<F: Field> DeviceInteractions<F, CpuBackend> {
    async fn to_device_interactions(
        &self,
        backend: &TaskScope,
    ) -> Result<DeviceInteractions<F, TaskScope>, slop_alloc::mem::CopyError> {
        let device_values_ptr = self.values_ptr.copy_to_backend(backend).await?;
        let device_multiplicities_ptr = self.multiplicities_ptr.copy_to_backend(backend).await?;
        let device_values_col_weights_ptr =
            self.values_col_weights_ptr.copy_to_backend(backend).await?;
        let device_values_col_weights = self.values_col_weights.copy_to_backend(backend).await?;
        let device_values_constants = self.values_constants.copy_to_backend(backend).await?;
        let device_mult_col_weights = self.mult_col_weights.copy_to_backend(backend).await?;
        let device_mult_constants = self.mult_constants.copy_to_backend(backend).await?;
        let device_arg_indices = self.arg_indices.copy_to_backend(backend).await?;
        let device_is_send = self.is_send.copy_to_backend(backend).await?;
        let num_interactions = self.num_interactions;

        Ok(DeviceInteractions {
            values_ptr: device_values_ptr,
            multiplicities_ptr: device_multiplicities_ptr,
            values_col_weights_ptr: device_values_col_weights_ptr,
            values_col_weights: device_values_col_weights,
            values_constants: device_values_constants,
            mult_col_weights: device_mult_col_weights,
            mult_constants: device_mult_constants,
            arg_indices: device_arg_indices,
            is_send: device_is_send,
            num_interactions,
        })
    }
}

impl<F: Field> DeviceInteractions<F, TaskScope> {
    fn as_ptr(&self) -> DeviceInteractionsPointer<F> {
        DeviceInteractionsPointer {
            values_ptr: self.values_ptr.as_ptr(),
            multiplicities_ptr: self.multiplicities_ptr.as_ptr(),
            values_col_weights_ptr: self.values_col_weights_ptr.as_ptr(),
            values_col_weights: self.values_col_weights.as_ptr(),
            values_constants: self.values_constants.as_ptr(),
            mult_col_weights: self.mult_col_weights.as_ptr(),
            mult_constants: self.mult_constants.as_ptr(),
            arg_indices: self.arg_indices.as_ptr(),
            is_send: self.is_send.as_ptr(),
            num_interactions: self.num_interactions,
        }
    }
}

unsafe impl<F: Field> Send for DeviceInteractionsPointer<F> {}
unsafe impl<F: Field> Sync for DeviceInteractionsPointer<F> {}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PairColDevice<F> {
    column_idx: usize,
    is_preprocessed: bool,
    weight: F,
}

/// An interaction for a lookup or a permutation argument.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceInteractions<F: Field, A: Backend> {
    pub values_ptr: Buffer<usize, A>,
    pub multiplicities_ptr: Buffer<usize, A>,
    pub values_col_weights_ptr: Buffer<usize, A>,

    pub values_col_weights: Buffer<PairColDevice<F>, A>,
    pub values_constants: Buffer<F, A>,

    pub mult_col_weights: Buffer<PairColDevice<F>, A>,
    pub mult_constants: Buffer<F, A>,

    pub arg_indices: Buffer<F, A>,
    pub is_send: Buffer<bool, A>,

    pub num_interactions: usize,
}

#[repr(C)]
pub struct DeviceInteractionsPointer<F: Field> {
    pub values_ptr: *const usize,
    pub multiplicities_ptr: *const usize,
    pub values_col_weights_ptr: *const usize,

    pub values_col_weights: *const PairColDevice<F>,
    pub values_constants: *const F,

    pub mult_col_weights: *const PairColDevice<F>,
    pub mult_constants: *const F,

    pub arg_indices: *const F,
    pub is_send: *const bool,

    pub num_interactions: usize,
}

pub trait GKRTracegenKernel<K, EK> {
    fn gkr_tracegen() -> KernelPtr;
}

impl GKRTracegenKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn gkr_tracegen() -> KernelPtr {
        unsafe { gkr_tracegen_kernel() }
    }
}

pub async fn generate_gkr_input_mles<F: PrimeField, EF: ExtensionField<F>>(
    preprocessed: Option<&PaddedMle<F, TaskScope>>,
    main: &PaddedMle<F, TaskScope>,
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    alpha: EF,
    betas: &Powers<EF>,
    log_max_row_height: usize,
) -> (PaddedMle<F, TaskScope>, PaddedMle<EF, TaskScope>)
where
    TaskScope: GKRTracegenKernel<F, EF>,
{
    let backend = main.padding_values().backend();

    let height = main.num_real_entries();
    let width = sends.len() + receives.len();

    if main.num_real_entries() == 0 {
        return (
            PaddedMle::new(
                None,
                log_max_row_height as u32,
                Padding::Constant((F::zero(), width, backend.clone())),
            ),
            PaddedMle::new(
                None,
                log_max_row_height as u32,
                Padding::Constant((EF::one(), width, backend.clone())),
            ),
        );
    }

    let host_interactions = DeviceInteractions::new(sends, receives);

    let device_interactions = host_interactions.to_device_interactions(backend).await.unwrap();

    // backend.synchronize().await.unwrap();

    let mut numer: Tensor<F, TaskScope> = Tensor::with_sizes_in([width, height], backend.clone());

    let mut denom: Tensor<EF, TaskScope> = Tensor::with_sizes_in([width, height], backend.clone());

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = 1;
    let grid_size = (grid_size_x, grid_size_y, 1);

    // backend.synchronize().await.unwrap();
    let backend = backend.clone();
    let backend = backend.clone();
    match preprocessed {
        Some(preprocessed) => unsafe {
            let device_interactions_ptr = device_interactions.as_ptr();

            let args = args!(
                device_interactions_ptr,
                numer.as_mut_ptr(),
                denom.as_mut_ptr(),
                preprocessed.inner().as_ref().unwrap().guts().as_ptr(),
                main.inner().as_ref().unwrap().guts().as_ptr(),
                alpha,
                betas.clone().take(2).collect::<Vec<_>>()[1],
                height
            );

            numer.assume_init();
            denom.assume_init();

            backend
                .launch_kernel(
                    <TaskScope as GKRTracegenKernel<F, EF>>::gkr_tracegen(),
                    grid_size,
                    (BLOCK_SIZE, 1, 1),
                    &args,
                    0,
                )
                .unwrap();
        },
        None => unsafe {
            let device_interactions_ptr = device_interactions.as_ptr();
            let null_ptr: *const F = std::ptr::null();

            let args = args!(
                device_interactions_ptr,
                numer.as_mut_ptr(),
                denom.as_mut_ptr(),
                null_ptr,
                main.inner().as_ref().unwrap().guts().as_ptr(),
                alpha,
                betas.clone().take(2).collect::<Vec<_>>()[1],
                height
            );
            numer.assume_init();
            denom.assume_init();

            backend
                .launch_kernel(
                    <TaskScope as GKRTracegenKernel<F, EF>>::gkr_tracegen(),
                    grid_size,
                    (BLOCK_SIZE, 1, 1),
                    &args,
                    0,
                )
                .unwrap();
        },
    }

    (
        PaddedMle::new(
            Some(Arc::new(Mle::new(numer))),
            log_max_row_height as u32,
            Padding::Constant((F::zero(), width, backend.clone())),
        ),
        PaddedMle::new(
            Some(Arc::new(Mle::new(denom))),
            log_max_row_height as u32,
            Padding::Constant((EF::one(), width, backend.clone())),
        ),
    )
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use rand::{thread_rng, Rng};
    use slop_air::{Air, BaseAir};
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_alloc::{CpuBackend, IntoHost};
    use slop_baby_bear::BabyBear;
    use slop_matrix::dense::RowMajorMatrix;
    use slop_multilinear::{Mle, MleEval, PaddedMle, Padding};
    use slop_tensor::Tensor;
    use sp1_stark::{
        air::{InteractionScope, MachineAir},
        Chip,
    };
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    use sp1_core_executor::{Instruction, Opcode, Program};
    use sp1_core_machine::{memory::MemoryLocalChip, riscv::ByteChip};
    use sp1_stark::InteractionBuilder;

    use crate::generate_gkr_input_mles;

    pub fn simple_memory_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0x12348765, false, true),
            // SW and LW
            Instruction::new(Opcode::SW, 29, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LW, 28, 0, 0x27654320, false, true),
            // LBU
            Instruction::new(Opcode::LBU, 27, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LBU, 26, 0, 0x27654321, false, true),
            Instruction::new(Opcode::LBU, 25, 0, 0x27654322, false, true),
            Instruction::new(Opcode::LBU, 24, 0, 0x27654323, false, true),
            // LB
            Instruction::new(Opcode::LB, 23, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LB, 22, 0, 0x27654321, false, true),
            // LHU
            Instruction::new(Opcode::LHU, 21, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LHU, 20, 0, 0x27654322, false, true),
            // LU
            Instruction::new(Opcode::LH, 19, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LH, 18, 0, 0x27654322, false, true),
            // SB
            Instruction::new(Opcode::ADD, 17, 0, 0x38276525, false, true),
            // Save the value 0x12348765 into address 0x43627530
            Instruction::new(Opcode::SW, 29, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627530, false, true),
            Instruction::new(Opcode::LW, 16, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627531, false, true),
            Instruction::new(Opcode::LW, 15, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627532, false, true),
            Instruction::new(Opcode::LW, 14, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627533, false, true),
            Instruction::new(Opcode::LW, 13, 0, 0x43627530, false, true),
            // SH
            // Save the value 0x12348765 into address 0x43627530
            Instruction::new(Opcode::SW, 29, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SH, 17, 0, 0x43627530, false, true),
            Instruction::new(Opcode::LW, 12, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SH, 17, 0, 0x43627532, false, true),
            Instruction::new(Opcode::LW, 11, 0, 0x43627530, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    async fn test_chip<A>(chip: Chip<F, A>, real_main_trace: bool)
    where
        A: BaseAir<F> + MachineAir<F, Program = Program> + Air<InteractionBuilder<F>>,
    {
        let mut rng = thread_rng();

        let program = simple_memory_program();
        let num_rows = 1 << 16;

        let log_num_padded_rows = 18;

        let padding_values: MleEval<F> = MleEval::new(Tensor::zeros_in([chip.width()], CpuBackend));

        let prep_padding_values: MleEval<F> =
            MleEval::new(Tensor::zeros_in([chip.preprocessed_width()], CpuBackend));

        let prep_padding_values_clone = prep_padding_values.clone();

        let preprocessed_trace = chip.generate_preprocessed_trace(&program);

        println!(
            "Preprocessed trace width: {}",
            preprocessed_trace.as_ref().map(|x| x.width).unwrap_or(0)
        );

        let preprocessed_trace_clone = preprocessed_trace.clone();

        let chip_clone = chip.clone();

        for interaction in chip.sends().iter().chain(chip.receives().iter()) {
            println!("{:?}", interaction.values);
        }

        // Generate a random trace.
        let mut main_trace = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());
        for val in main_trace.values.iter_mut() {
            *val = rng.gen::<F>();
        }
        let main_trace_clone = main_trace.clone();

        let padding_values_clone = padding_values.clone();

        // Get randomness.
        let alpha = rng.gen::<EF>();
        let beta = rng.gen::<EF>();
        // Transfer perm and main traces to the device.

        let (numerator_d, denom_d) = crate::task()
            .await
            .unwrap()
            .run(|t| async move {
                let main_trace_device = t.into_device(Mle::from(main_trace.clone())).await.unwrap();
                let padding_values_device =
                    Arc::new(t.into_device(padding_values.clone()).await.unwrap());
                let prep_padding_values_device =
                    Arc::new(t.into_device(prep_padding_values.clone()).await.unwrap());
                let padded_main = PaddedMle::new(
                    if real_main_trace { Some(Arc::new(main_trace_device)) } else { None },
                    log_num_padded_rows,
                    Padding::Generic(padding_values_device.clone()),
                );

                let prep_device = if let Some(preprocessed_trace) = preprocessed_trace.clone() {
                    Some(t.into_device(Mle::from(preprocessed_trace.clone())).await.unwrap())
                } else {
                    None
                };

                let padded_prep = prep_device.map(|prep| {
                    PaddedMle::new(
                        if real_main_trace { Some(Arc::new(prep)) } else { None },
                        log_num_padded_rows,
                        Padding::Generic(prep_padding_values_device.clone()),
                    )
                });

                generate_gkr_input_mles(
                    padded_prep.as_ref(),
                    &padded_main,
                    &chip
                        .sends()
                        .iter()
                        .filter(|x| x.scope == InteractionScope::Local)
                        .cloned()
                        .collect::<Vec<_>>(),
                    &chip
                        .receives()
                        .iter()
                        .filter(|x| x.scope == InteractionScope::Local)
                        .cloned()
                        .collect::<Vec<_>>(),
                    alpha,
                    &beta.powers(),
                    log_num_padded_rows as usize,
                )
                .await
            })
            .await
            .await
            .unwrap();

        let numerator_h = numerator_d.into_host().await.unwrap();
        let denom_h = denom_d.into_host().await.unwrap();

        let padded_main = PaddedMle::new(
            if real_main_trace { Some(Arc::new(main_trace_clone.into())) } else { None },
            log_num_padded_rows,
            Padding::Generic(Arc::new(padding_values_clone)),
        );

        let padded_prep = preprocessed_trace_clone.map(|prep| {
            PaddedMle::new(
                if real_main_trace { Some(Arc::new(prep.into())) } else { None },
                log_num_padded_rows,
                Padding::Generic(Arc::new(prep_padding_values_clone)),
            )
        });

        let (expected_numer, expected_denom) = sp1_stark::generate_gkr_input_mles(
            padded_prep.as_ref(),
            &padded_main,
            chip_clone
                .sends()
                .iter()
                .map(|interaction| (interaction, true))
                .chain(chip_clone.receives().iter().map(|interaction| (interaction, false)))
                .collect::<Vec<_>>()
                .as_slice(),
            alpha,
            &beta.powers(),
            log_num_padded_rows as usize,
        );
        assert_eq!(numerator_h.num_real_entries(), expected_numer.num_real_entries());
        assert_eq!(denom_h.num_real_entries(), expected_denom.num_real_entries());
        assert_eq!(numerator_h.num_polynomials(), expected_numer.num_polynomials());
        assert_eq!(denom_h.num_polynomials(), expected_denom.num_polynomials());

        assert_eq!(numerator_h.num_variables(), expected_numer.num_variables());
        assert_eq!(denom_h.num_variables(), expected_denom.num_variables());

        // Compare the values to the host values.
        for (i, (exp, res)) in expected_numer
            .inner()
            .as_ref()
            .map(|x| x.guts().as_slice())
            .unwrap_or(&[])
            .iter()
            .zip_eq(numerator_h.inner().as_ref().map(|x| x.guts().as_slice()).unwrap_or(&[]).iter())
            .enumerate()
        {
            assert_eq!(exp, res, "numer values failed at index {}", i);
        }

        for (i, (exp, res)) in expected_denom
            .inner()
            .as_ref()
            .map(|x| x.guts().as_slice())
            .unwrap_or(&[])
            .iter()
            .zip_eq(denom_h.inner().as_ref().map(|x| x.guts().as_slice()).unwrap_or(&[]).iter())
            .enumerate()
        {
            assert_eq!(exp, res, "denom values failed at index {}", i);
        }

        // for (i, (exp, res)) in expected_numer
        //     .padding_values()
        //     .to_host()
        //     .await
        //     .unwrap()
        //     .to_vec()
        //     .iter()
        //     .zip_eq(numerator_h.padding_values().to_host().await.unwrap().to_vec().iter())
        //     .enumerate()
        // {
        //     assert_eq!(exp, res, "numer padding values failed at index {}", i);
        // }

        // for (i, (exp, res)) in expected_denom
        //     .padding_values()
        //     .to_host()
        //     .await
        //     .unwrap()
        //     .to_vec()
        //     .iter()
        //     .zip_eq(denom_h.padding_values().to_host().await.unwrap().to_vec().iter())
        //     .enumerate()
        // {
        //     assert_eq!(exp, res, "denom padding values failed at index {}", i);
        // }
    }

    #[tokio::test]
    async fn test_memory_local_chip() {
        let chip = Chip::new(MemoryLocalChip::new());
        test_chip(chip.clone(), true).await;
        test_chip(chip, false).await;

        let chip = Chip::new(ByteChip::default());
        test_chip(chip.clone(), true).await;

        test_chip(chip, false).await;
    }
}

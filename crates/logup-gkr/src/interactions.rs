use csl_cuda::TaskScope;
use slop_air::PairCol;
use slop_algebra::Field;
use slop_alloc::{mem::CopyError, Backend, Buffer, CopyToBackend, CpuBackend, HasBackend};
use sp1_hypercube::Interaction;
use std::ops::Mul;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PairColDevice<F> {
    column_idx: usize,
    is_preprocessed: bool,
    weight: F,
}

#[repr(C)]
pub struct InteractionsRaw<F> {
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

/// An interaction for a lookup or a permutation argument.
#[derive(Debug)]
#[repr(C)]
pub struct Interactions<F, A: Backend> {
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

impl<F: Field> Interactions<F, CpuBackend> {
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

impl<F: Field> Interactions<F, TaskScope> {
    pub fn as_raw(&self) -> InteractionsRaw<F> {
        InteractionsRaw {
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

// impl<F: Field, EF: ExtensionField<F>> CanCopy

impl<F: Field> CopyToBackend<TaskScope, CpuBackend> for Interactions<F, CpuBackend> {
    type Output = Interactions<F, TaskScope>;
    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let (
            device_values_ptr,
            device_multiplicities_ptr,
            device_values_col_weights_ptr,
            device_values_col_weights,
            device_values_constants,
            device_mult_col_weights,
            device_mult_constants,
            device_arg_indices,
            device_is_send,
        ) = tokio::join!(
            async { self.values_ptr.copy_to_backend(backend).await.unwrap() },
            async { self.multiplicities_ptr.copy_to_backend(backend).await.unwrap() },
            async { self.values_col_weights_ptr.copy_to_backend(backend).await.unwrap() },
            async { self.values_col_weights.copy_to_backend(backend).await.unwrap() },
            async { self.values_constants.copy_to_backend(backend).await.unwrap() },
            async { self.mult_col_weights.copy_to_backend(backend).await.unwrap() },
            async { self.mult_constants.copy_to_backend(backend).await.unwrap() },
            async { self.arg_indices.copy_to_backend(backend).await.unwrap() },
            async { self.is_send.copy_to_backend(backend).await.unwrap() },
        );

        let num_interactions = self.num_interactions;

        Ok(Interactions {
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

impl<F: Field, A: Backend> HasBackend for Interactions<F, A> {
    type Backend = A;

    fn backend(&self) -> &Self::Backend {
        self.values_col_weights_ptr.backend()
    }
}

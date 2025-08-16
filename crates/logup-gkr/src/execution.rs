use std::{
    collections::{BTreeMap, BTreeSet},
    iter::once,
    marker::PhantomData,
    ptr,
    sync::Arc,
};

use futures::{prelude::*, stream::FuturesUnordered};

use csl_cuda::{
    args,
    sys::{
        logup_gkr::{
            logup_gkr_circuit_transition_koala_bear_extension, logup_gkr_extract_output_koala_bear,
            logup_gkr_first_layer_transition_koala_bear,
            logup_gkr_populate_last_circuit_layer_koala_bear,
        },
        runtime::KernelPtr,
    },
    TaskScope, ToDevice,
};
use itertools::Itertools;
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field};
use slop_alloc::{Buffer, HasBackend};
use slop_koala_bear::KoalaBear;
use slop_multilinear::Mle;
use slop_tensor::Tensor;
use sp1_hypercube::{
    air::MachineAir, prover::Traces, Chip, LogUpGkrOutput, LogUpGkrTraceGenerator,
};

use crate::{
    FirstGkrLayer, FirstLayerData, GkrCircuitLayer, GkrInputData, GkrLayer, Interactions,
    LogUpCudaCircuit,
};

pub struct LogUpGkrCudaTraceGenerator<F, EF, A>(PhantomData<(F, EF, A)>);

impl<F, EF, A> Default for LogUpGkrCudaTraceGenerator<F, EF, A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// # Safety
/// TODO
pub unsafe trait CircuitTransitionKernel<EF> {
    fn circuit_transition_kernel() -> KernelPtr;
}

/// # Safety
/// TODO
pub unsafe trait FirstLayerTransitionKernel<F, EF> {
    fn first_layer_transition_kernel() -> KernelPtr;
}

/// # Safety
/// TODO
pub unsafe trait PopulateLastCircuitLayerKernel<F, EF> {
    fn populate_last_circuit_layer_kernel() -> KernelPtr;
}

/// # Safety
/// TODO
pub unsafe trait ExtractOutputKernel<EF> {
    fn extract_output_kernel() -> KernelPtr;
}

impl<F, EF, A> LogUpGkrCudaTraceGenerator<F, EF, A>
where
    F: Field,
    EF: ExtensionField<F>,
    TaskScope: CircuitTransitionKernel<EF> + FirstLayerTransitionKernel<F, EF>,
{
    pub fn extract_outputs(
        &self,
        layer: &GkrLayer<EF, TaskScope>,
        num_interaction_variables: u32,
    ) -> LogUpGkrOutput<EF, TaskScope> {
        let height = layer.layer.sizes()[2];
        let output_height = 1 << (num_interaction_variables + 1);
        let backend = layer.layer.backend();

        let mut numerator = Mle::uninit(1, output_height, backend);
        let mut denominator = Mle::uninit(1, output_height, backend);

        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 4;
        let grid_height = output_height.div_ceil(2);
        let grid_size_x = grid_height.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size = (grid_size_x, 1, 1);
        let block_dim = BLOCK_SIZE;

        unsafe {
            numerator.assume_init();
            denominator.assume_init();
            let args = args!(
                layer.layer.as_ptr(),
                layer.interaction_data.as_ptr(),
                layer.interaction_start_indices.as_ptr(),
                numerator.guts_mut().as_mut_ptr(),
                denominator.guts_mut().as_mut_ptr(),
                height,
                grid_height
            );
            backend
                .launch_kernel(TaskScope::extract_output_kernel(), grid_size, block_dim, &args, 0)
                .unwrap();
        }

        LogUpGkrOutput { numerator, denominator }
    }

    pub(crate) async fn generate_first_layer(
        &self,
        input_data: &GkrInputData<F, EF>,
    ) -> FirstGkrLayer<F, EF>
    where
        TaskScope: PopulateLastCircuitLayerKernel<F, EF>,
    {
        let first_trace = input_data.traces.values().next().unwrap();
        let num_row_variables = first_trace.num_variables() - 1;
        let backend = first_trace.backend().clone();
        let mut dimensions = Vec::with_capacity(input_data.interactions.len());
        let interaction_row_counts = input_data
            .interactions
            .iter()
            .flat_map(|(name, interactions)| {
                let real_height = input_data.traces.get(name).unwrap().num_real_entries();
                let height = std::cmp::max(real_height, 1);
                let height = height.div_ceil(4);
                dimensions.push(sp1_hypercube::log2_ceil_usize(height) + 1);
                vec![height as u32; interactions.num_interactions]
            })
            .collect::<Vec<_>>();
        let interaction_start_indices = once(0)
            .chain(interaction_row_counts.iter().scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            }))
            .collect::<Buffer<_>>();
        let height = interaction_start_indices.last().copied().unwrap() as usize;

        let interaction_start_indices =
            interaction_start_indices.to_device_in(&backend).await.unwrap();
        let mut interaction_data = Buffer::<u32, _>::with_capacity_in(height, backend.clone());
        let mut numerator = Tensor::<F, _>::with_sizes_in([2, 2, height], backend.clone());
        let mut denominator = Tensor::<EF, _>::with_sizes_in([2, 2, height], backend.clone());

        const BLOCK_SIZE: usize = 256;
        const ROW_STRIDE: usize = 8;
        const INTERACTION_STRIDE: usize = 4;

        let mut interaction_offset = 0;
        let handles = FuturesUnordered::new();
        for ((name, interactions), dimension) in input_data.interactions.iter().zip_eq(dimensions) {
            let trace = input_data.traces.get(name).unwrap().clone();
            let preprocessed_trace = input_data.preprocessed_traces.get(name).cloned();
            let alpha = input_data.alpha;
            let beta = input_data.beta;
            let interactions = interactions.clone();
            let num_interactions = interactions.num_interactions;
            let interaction_start_indices = unsafe { interaction_start_indices.owned_unchecked() };
            let mut interaction_data = unsafe { interaction_data.owned_unchecked() };
            let mut numerator = unsafe { numerator.owned_unchecked() };
            let mut denominator = unsafe { denominator.owned_unchecked() };
            let handle = backend.run_in_place(move |s| async move {
                let real_height = trace.num_real_entries();
                assert_eq!(real_height % 2, 0);
                let is_padding = real_height == 0;

                let matrix_height = std::cmp::max(real_height, 2);
                let half_height = matrix_height.div_ceil(2);

                let interactions = interactions.to_device_in(&s).await.unwrap();

                let block_dim = BLOCK_SIZE;
                let grid_size = (
                    half_height.div_ceil(BLOCK_SIZE * ROW_STRIDE),
                    num_interactions.div_ceil(INTERACTION_STRIDE),
                    1,
                );
                unsafe {
                    let preprocessed_ptr = preprocessed_trace
                        .map(|t| t.inner().as_ref().unwrap().guts().as_ptr())
                        .unwrap_or(ptr::null());
                    let main_ptr =
                        trace.inner().as_ref().map(|m| m.guts().as_ptr()).unwrap_or(ptr::null());

                    let args = args!(
                        interactions.as_raw(),
                        interaction_start_indices.as_ptr(),
                        interaction_data.as_mut_ptr(),
                        numerator.as_mut_ptr(),
                        denominator.as_mut_ptr(),
                        preprocessed_ptr,
                        main_ptr,
                        alpha,
                        beta,
                        interaction_offset,
                        real_height,
                        half_height,
                        height,
                        dimension,
                        is_padding
                    );
                    s.launch_kernel(
                        TaskScope::populate_last_circuit_layer_kernel(),
                        grid_size,
                        block_dim,
                        &args,
                        0,
                    )
                    .unwrap();
                }
            });
            handles.push(handle);

            interaction_offset += num_interactions;
        }

        handles.collect::<Vec<_>>().await;

        unsafe {
            interaction_data.assume_init();
            numerator.assume_init();
            denominator.assume_init();
        }

        let data = FirstLayerData { numerator, denominator };

        let num_interaction_variables = interaction_offset.next_power_of_two().ilog2();

        FirstGkrLayer {
            layer: data,
            interaction_data,
            interaction_start_indices,
            interaction_row_counts,
            num_row_variables,
            num_interaction_variables,
        }
    }

    pub(crate) async fn gkr_transition(
        &self,
        layer: &GkrCircuitLayer<F, EF>,
    ) -> GkrCircuitLayer<F, EF>
    where
        TaskScope: CircuitTransitionKernel<EF> + FirstLayerTransitionKernel<F, EF>,
    {
        match layer {
            GkrCircuitLayer::FirstLayer(layer) => {
                GkrCircuitLayer::Materialized(self.first_layer_transition(layer).await)
            }
            GkrCircuitLayer::Materialized(layer) => {
                GkrCircuitLayer::Materialized(self.layer_transition(layer).await)
            }
            GkrCircuitLayer::FirstLayerVirtual(_) => {
                unreachable!()
            }
        }
    }

    pub(crate) async fn layer_transition(&self, layer: &GkrLayer<EF>) -> GkrLayer<EF>
    where
        TaskScope: CircuitTransitionKernel<EF>,
    {
        let backend = layer.layer.backend();
        let height = layer.layer.sizes()[2];

        // If this is not the last layer, we need to fix the last variable and create a
        // new circuit layer.
        let output_interaction_row_counts =
            layer.interaction_row_counts.iter().map(|count| count.div_ceil(2)).collect::<Vec<_>>();
        // The output indices is just the prefix sum of the interaction row counts.
        let output_interaction_start_indices = once(0)
            .chain(output_interaction_row_counts.iter().scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            }))
            .collect::<Buffer<_>>();
        let output_height = output_interaction_start_indices.last().copied().unwrap() as usize;
        let output_interaction_start_indices =
            output_interaction_start_indices.to_device_in(backend).await.unwrap();

        // Create a new layer
        let mut output_layer: Tensor<EF, TaskScope> =
            Tensor::with_sizes_in([4, 2, output_height], backend.clone());
        let mut output_interaction_data: Buffer<u32, TaskScope> =
            Buffer::with_capacity_in(output_height, backend.clone());

        // populate the new layer
        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 32;
        let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size = (grid_size_x, 1, 1);
        let block_dim = BLOCK_SIZE;
        unsafe {
            output_layer.assume_init();
            output_interaction_data.assume_init();
            let args = args!(
                layer.layer.as_ptr(),
                layer.interaction_data.as_ptr(),
                layer.interaction_start_indices.as_ptr(),
                output_layer.as_mut_ptr(),
                output_interaction_data.as_mut_ptr(),
                output_interaction_start_indices.as_ptr(),
                height,
                output_height
            );
            backend
                .launch_kernel(
                    TaskScope::circuit_transition_kernel(),
                    grid_size,
                    block_dim,
                    &args,
                    0,
                )
                .unwrap();
        }

        GkrLayer {
            layer: output_layer,
            interaction_data: output_interaction_data,
            interaction_start_indices: output_interaction_start_indices,
            interaction_row_counts: output_interaction_row_counts,
            num_row_variables: layer.num_row_variables - 1,
            num_interaction_variables: layer.num_interaction_variables,
        }
    }

    pub(crate) async fn first_layer_transition(&self, layer: &FirstGkrLayer<F, EF>) -> GkrLayer<EF>
    where
        TaskScope: FirstLayerTransitionKernel<F, EF>,
    {
        let backend = layer.layer.numerator.backend();
        let height = layer.layer.numerator.sizes()[2];

        // If this is not the last layer, we need to fix the last variable and create a
        // new circuit layer.
        let output_interaction_row_counts =
            layer.interaction_row_counts.iter().map(|count| count.div_ceil(2)).collect::<Vec<_>>();
        // The output indices is just the prefix sum of the interaction row counts.
        let output_interaction_start_indices = once(0)
            .chain(output_interaction_row_counts.iter().scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            }))
            .collect::<Buffer<_>>();
        let output_height = output_interaction_start_indices.last().copied().unwrap() as usize;
        let output_interaction_start_indices =
            output_interaction_start_indices.to_device_in(backend).await.unwrap();

        // Create a new layer
        let mut output_layer: Tensor<EF, TaskScope> =
            Tensor::with_sizes_in([4, 2, output_height], backend.clone());
        let mut output_interaction_data: Buffer<u32, TaskScope> =
            Buffer::with_capacity_in(output_height, backend.clone());

        // populate the new layer
        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 32;
        let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size = (grid_size_x, 1, 1);
        let block_dim = BLOCK_SIZE;
        unsafe {
            output_layer.assume_init();
            output_interaction_data.assume_init();
            let args = args!(
                layer.layer.numerator.as_ptr(),
                layer.layer.denominator.as_ptr(),
                layer.interaction_data.as_ptr(),
                layer.interaction_start_indices.as_ptr(),
                output_layer.as_mut_ptr(),
                output_interaction_data.as_mut_ptr(),
                output_interaction_start_indices.as_ptr(),
                height,
                output_height
            );
            backend
                .launch_kernel(
                    TaskScope::first_layer_transition_kernel(),
                    grid_size,
                    block_dim,
                    &args,
                    0,
                )
                .unwrap();
        }

        GkrLayer {
            layer: output_layer,
            interaction_data: output_interaction_data,
            interaction_start_indices: output_interaction_start_indices,
            interaction_row_counts: output_interaction_row_counts,
            num_row_variables: layer.num_row_variables - 1,
            num_interaction_variables: layer.num_interaction_variables,
        }
    }
}

impl<F, EF, A> LogUpGkrTraceGenerator<F, EF, A, TaskScope> for LogUpGkrCudaTraceGenerator<F, EF, A>
where
    F: Field,
    EF: ExtensionField<F>,
    A: MachineAir<F>,
    TaskScope: CircuitTransitionKernel<EF>
        + FirstLayerTransitionKernel<F, EF>
        + PopulateLastCircuitLayerKernel<F, EF>
        + ExtractOutputKernel<EF>,
{
    type Circuit = LogUpCudaCircuit<F, EF, A>;

    async fn generate_gkr_circuit(
        &self,
        chips: &BTreeSet<Chip<F, A>>,
        preprocessed_traces: Traces<F, TaskScope>,
        traces: Traces<F, TaskScope>,
        _public_values: Vec<F>,
        alpha: EF,
        beta: EF,
    ) -> (LogUpGkrOutput<EF, TaskScope>, Self::Circuit) {
        let interactions = chips
            .iter()
            .map(|chip| {
                let interactions = Interactions::new(chip.sends(), chip.receives());
                (chip.name(), Arc::new(interactions))
            })
            .collect::<BTreeMap<_, _>>();
        let input_data = GkrInputData { interactions, preprocessed_traces, traces, alpha, beta };

        let mut materialized_layers = Vec::new();

        let first_layer = self.generate_first_layer(&input_data).await;
        let num_row_variables = first_layer.num_row_variables;
        let num_interaction_variables = first_layer.num_interaction_variables;

        let first_layer = GkrCircuitLayer::FirstLayer(first_layer);
        let layer = self.gkr_transition(&first_layer).await;
        drop(first_layer);
        // materialized_layers.push(first_layer);
        materialized_layers.push(layer);
        for _ in 0..num_row_variables - 2 {
            let layer = self.gkr_transition(materialized_layers.last().unwrap()).await;
            materialized_layers.push(layer);
        }

        let last_layer = if let GkrCircuitLayer::Materialized(last_layer) =
            materialized_layers.last().unwrap()
        {
            last_layer
        } else {
            panic!("last layer not correct");
        };
        assert_eq!(last_layer.num_row_variables, 1);

        let output = self.extract_outputs(last_layer, num_interaction_variables);

        let circuit_generator = Some(Self::default());
        let circuit = LogUpCudaCircuit {
            circuit_generator,
            materialized_layers,
            input_data,
            num_virtual_layers: 1,
        };

        (output, circuit)
    }
}

unsafe impl CircuitTransitionKernel<BinomialExtensionField<KoalaBear, 4>> for TaskScope {
    fn circuit_transition_kernel() -> KernelPtr {
        unsafe { logup_gkr_circuit_transition_koala_bear_extension() }
    }
}

unsafe impl FirstLayerTransitionKernel<KoalaBear, BinomialExtensionField<KoalaBear, 4>>
    for TaskScope
{
    fn first_layer_transition_kernel() -> KernelPtr {
        unsafe { logup_gkr_first_layer_transition_koala_bear() }
    }
}

unsafe impl PopulateLastCircuitLayerKernel<KoalaBear, BinomialExtensionField<KoalaBear, 4>>
    for TaskScope
{
    fn populate_last_circuit_layer_kernel() -> KernelPtr {
        unsafe { logup_gkr_populate_last_circuit_layer_koala_bear() }
    }
}

unsafe impl ExtractOutputKernel<BinomialExtensionField<KoalaBear, 4>> for TaskScope {
    fn extract_output_kernel() -> KernelPtr {
        unsafe { logup_gkr_extract_output_koala_bear() }
    }
}

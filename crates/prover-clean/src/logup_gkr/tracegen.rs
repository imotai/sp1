use std::{
    collections::{BTreeMap, BTreeSet},
    iter::once,
    sync::Arc,
};

use csl_cuda::{
    args, sys::prover_clean::prover_clean_logup_gkr_populate_last_circuit_layer, TaskScope,
    ToDevice,
};
use futures::{stream::FuturesUnordered, StreamExt};
use slop_alloc::{Buffer, HasBackend};
use slop_multilinear::{Mle, Point};
use slop_tensor::Tensor;
use sp1_hypercube::{air::MachineAir, Chip, LogUpGkrOutput};

use crate::{
    config::{Ext, Felt},
    logup_gkr::{
        extract_outputs, gkr_transition,
        interactions::Interactions,
        layer::{JaggedFirstGkrLayer, JaggedFirstGkrMle},
        utils::{FirstGkrLayer, GkrInputData, LogUpCudaCircuit},
        GkrCircuitLayer,
    },
    tracegen::JaggedTraceMle,
};

/// Generates the first layer of the GKR circuit.
///
/// Processes all of the chip interaction information and traces into GKR circuit format.
pub async fn generate_first_layer(input_data: &GkrInputData, backend: &TaskScope) -> FirstGkrLayer {
    let num_row_variables = input_data.num_row_variables - 1;

    // interaction_row_counts iterates through the traces by chip order, and returns column sizes for each interaction.
    let interaction_row_counts = input_data
        .interactions
        .iter()
        .flat_map(|(name, interactions)| {
            let real_height = input_data.main_poly_height(name);
            // For padding reasons, `height` always needs to be at least 2.
            let height = std::cmp::max(real_height, 8);
            // Divide by 2 because each row has even height, so we only store length / 2.
            // Divide by 2 again because numerator(x, 0) and numerator(x, 1) are stored separately.
            let height = height.div_ceil(4);
            vec![height as u32; interactions.num_interactions]
        })
        .collect::<Vec<_>>();

    // interaction_start_indices is a prefix sum of interaction_row_counts.
    let interaction_start_indices = once(0)
        .chain(interaction_row_counts.iter().scan(0u32, |acc, x| {
            *acc += x;
            Some(*acc)
        }))
        .collect::<Buffer<_>>();
    let height = interaction_start_indices.last().copied().unwrap() as usize;

    let interaction_start_indices = interaction_start_indices.to_device_in(backend).await.unwrap();
    let mut interaction_data = Buffer::<u32, _>::with_capacity_in(height, backend.clone());
    let mut numerator = Tensor::<Felt, _>::with_sizes_in([2, 1, height * 2], backend.clone());
    let mut denominator = Tensor::<Ext, _>::with_sizes_in([2, 1, height * 2], backend.clone());

    const BLOCK_SIZE: usize = 256;
    const ROW_STRIDE: usize = 8;
    const INTERACTION_STRIDE: usize = 4;

    let beta = input_data.beta_seed.clone();
    let beta = beta.to_device_in(backend).await.unwrap();
    let betas = Mle::partial_lagrange(&beta).await;

    // Generate traces per chip, sorted by chip name.
    let mut interaction_offset = 0;
    let handles = FuturesUnordered::new();
    for (name, interactions) in input_data.interactions.iter() {
        let alpha = input_data.alpha;
        let interactions = interactions.clone();
        let num_interactions = interactions.num_interactions;
        let interaction_start_indices = unsafe { interaction_start_indices.owned_unchecked() };
        let mut interaction_data = unsafe { interaction_data.owned_unchecked() };
        let mut numerator = unsafe { numerator.owned_unchecked() };
        let mut denominator = unsafe { denominator.owned_unchecked() };
        let betas = unsafe { betas.owned_unchecked() };

        let handle = backend.run_in_place(move |scope| async move {
            let real_height = input_data.main_poly_height(name);

            // To fit the padding requirement, each trace must have even height.
            assert_eq!(real_height % 2, 0);
            let is_padding = real_height == 0;

            // half_height is max(1, ceil(real_height / 2))
            let matrix_height = std::cmp::max(real_height, 2);
            let half_height = matrix_height.div_ceil(2);

            let interactions = interactions.to_device_in(&scope).await.unwrap();

            let block_dim = BLOCK_SIZE;
            let grid_size = (
                half_height.div_ceil(BLOCK_SIZE * ROW_STRIDE),
                num_interactions.div_ceil(INTERACTION_STRIDE),
                1,
            );
            unsafe {
                let preprocessed_ptr = input_data.preprocessed_ptr(name);
                let main_ptr = input_data.main_ptr(name);

                let args = args!(
                    interactions.as_raw(),
                    interaction_start_indices.as_ptr(),
                    interaction_data.as_mut_ptr(),
                    numerator.as_mut_ptr(),
                    denominator.as_mut_ptr(),
                    preprocessed_ptr,
                    main_ptr,
                    alpha,
                    betas.guts().as_ptr(),
                    interaction_offset,
                    real_height,
                    height,
                    is_padding
                );
                scope
                    .launch_kernel(
                        prover_clean_logup_gkr_populate_last_circuit_layer(),
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

    // Height is half of the actual height of the numerator tensor.
    let height = numerator.sizes()[2] / 2;
    let jagged_layer = JaggedFirstGkrLayer { numerator, denominator, height };

    let jagged_mle =
        JaggedFirstGkrMle::new(jagged_layer, interaction_data, interaction_start_indices);

    let num_interaction_variables = interaction_offset.next_power_of_two().ilog2();

    FirstGkrLayer {
        jagged_mle,
        interaction_col_sizes: interaction_row_counts,
        num_row_variables,
        num_interaction_variables,
    }
}

impl LogUpCudaCircuit<TaskScope> {
    pub async fn next(&mut self) -> Option<GkrCircuitLayer> {
        if let Some(layer) = self.materialized_layers.pop() {
            Some(layer)
        } else {
            if self.num_virtual_layers == 0 {
                return None;
            }
            assert!(self.num_virtual_layers == 1);
            // We need to generate the virtual layers and store them in the circuit.
            let layer = generate_first_layer(&self.input_data, self.backend()).await;
            self.num_virtual_layers = 0;
            Some(GkrCircuitLayer::FirstLayer(layer))
        }
    }
}

/// Generates a GKR circuit from the given chips and jagged trace data.
pub async fn generate_gkr_circuit<A: MachineAir<Felt>>(
    chips: &BTreeSet<Chip<Felt, A>>,
    jagged_trace_data: Arc<JaggedTraceMle<TaskScope>>,
    num_row_variables: u32,
    alpha: Ext,
    beta_seed: Point<Ext>,
    backend: TaskScope,
) -> (LogUpGkrOutput<Ext, TaskScope>, LogUpCudaCircuit<TaskScope>) {
    let interactions = chips
        .iter()
        .map(|chip| {
            let interactions = Interactions::new(chip.sends(), chip.receives());
            (chip.name(), Arc::new(interactions))
        })
        .collect::<BTreeMap<_, _>>();
    let input_data = GkrInputData {
        interactions,
        jagged_trace_data: jagged_trace_data.clone(),
        alpha,
        beta_seed,
        num_row_variables,
        backend: backend.clone(),
    };

    let mut materialized_layers = Vec::new();

    // Generate the first layer.
    let first_layer = generate_first_layer(&input_data, &backend).await;
    let num_row_variables = first_layer.num_row_variables;
    let num_interaction_variables = first_layer.num_interaction_variables;

    let first_layer = GkrCircuitLayer::FirstLayer(first_layer);
    let layer: GkrCircuitLayer = gkr_transition(&first_layer).await;
    drop(first_layer);

    // Transition from the previous layer to generate the next one.
    materialized_layers.push(layer);
    for _ in 0..num_row_variables - 2 {
        let layer = gkr_transition(materialized_layers.last().unwrap()).await;
        materialized_layers.push(layer);
    }

    let last_layer =
        if let GkrCircuitLayer::Materialized(last_layer) = materialized_layers.last().unwrap() {
            last_layer
        } else {
            panic!("last layer not correct");
        };
    assert_eq!(last_layer.num_row_variables, 1);

    // Extract the outputs from the last layer.
    let output = extract_outputs(last_layer, num_interaction_variables).await;
    let circuit = LogUpCudaCircuit { materialized_layers, input_data, num_virtual_layers: 1 };

    (output, circuit)
}

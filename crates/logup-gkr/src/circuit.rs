use std::{collections::BTreeMap, sync::Arc};

use csl_cuda::TaskScope;
use slop_algebra::{ExtensionField, Field};
use slop_alloc::{Backend, Buffer, CpuBackend};
use slop_tensor::Tensor;
use sp1_stark::{prover::Traces, LogUpGkrCircuit};

use crate::{
    CircuitTransitionKernel, FirstLayerTransitionKernel, Interactions, LogUpGkrCudaTraceGenerator,
    PopulateLastCircuitLayerKernel,
};

/// A layer of the GKR circuit.
///
/// This layer contains the polynomials p_0, p_1, q_0, q_1 evaluated at the layer size. The circuit
/// represents sparse values that can come from various chips with different sizes.
pub struct GkrLayerGeneric<Layer, B: Backend = TaskScope> {
    /// A tensor of shape [4, 2, layer_size.div_ceil(2)] such that:
    ///
    ///    numerator[x, b, 0] = layer[0, b, x]
    ///    numerator[x, b, 1] = layer[1, b, x]
    ///    denominator[x, b, 0] = layer[2, b, x]
    ///    denominator[x, b, 1] = layer[3, b, x]
    ///
    /// for `b` in `{0, 1}` and `x` is the size of the next layer of the trace.
    pub layer: Layer,
    /// A buffer of shape [layer_size.div_ceil(2)] which contains data about the row for each  
    /// element in the next layer of the trace. Namely,
    ///
    /// `interaction_data[i] = intraction_idx | `dimension`
    ///
    /// where `intraction_idx` constitues the first 24 bits is the the specific interaction index
    /// and `restricted_dimension` constitues the last 8 bits. The value of `dimension`
    /// is the current real dimension of the chip.
    pub interaction_data: Buffer<u32, B>,
    /// The start indices for the interactions in the long index layer_size.
    ///
    /// The buffer is of length equal to the total number of interactions.
    pub interaction_start_indices: Buffer<u32, B>,
    /// The number of rows / 2 for each interaction.
    pub interaction_row_counts: Vec<u32>,
    /// The number of row variables.
    pub num_row_variables: u32,
    /// The total number of interaction variables
    pub num_interaction_variables: u32,
}

#[allow(clippy::type_complexity)]
pub struct GkrInputData<F: Field, EF> {
    pub interactions: BTreeMap<String, Arc<Interactions<F, CpuBackend>>>,
    pub traces: Traces<F, TaskScope>,
    pub preprocessed_traces: Traces<F, TaskScope>,
    pub alpha: EF,
    pub beta: EF,
}

pub struct FirstLayerData<F, EF, B: Backend> {
    pub numerator: Tensor<F, B>,
    pub denominator: Tensor<EF, B>,
}

pub type GkrLayer<EF, B = TaskScope> = GkrLayerGeneric<Tensor<EF, B>, B>;

pub type FirstGkrLayer<F, EF, B = TaskScope> = GkrLayerGeneric<FirstLayerData<F, EF, B>, B>;

pub struct LogUpCudaCircuit<F: Field, EF, A> {
    pub circuit_generator: Option<LogUpGkrCudaTraceGenerator<F, EF, A>>,
    pub materialized_layers: Vec<GkrCircuitLayer<F, EF>>,
    pub input_data: GkrInputData<F, EF>,
    pub num_virtual_layers: usize,
}

pub enum GkrCircuitLayer<F: Field, EF> {
    Materialized(GkrLayer<EF>),
    FirstLayer(FirstGkrLayer<F, EF>),
    FirstLayerVirtual(GkrInputData<F, EF>),
}

impl<F: Field, EF: ExtensionField<F>, A> LogUpGkrCircuit for LogUpCudaCircuit<F, EF, A>
where
    TaskScope: PopulateLastCircuitLayerKernel<F, EF>
        + CircuitTransitionKernel<EF>
        + FirstLayerTransitionKernel<F, EF>,
    A: Send + Sync,
{
    type CircuitLayer = GkrCircuitLayer<F, EF>;

    async fn next(&mut self) -> Option<Self::CircuitLayer> {
        if let Some(layer) = self.materialized_layers.pop() {
            Some(layer)
        } else {
            if self.num_virtual_layers == 0 {
                return None;
            }
            assert!(self.num_virtual_layers == 1);
            // We need to generate the virtual layers and store them in the circuit.
            //              Generate the virtual layers and store them in the circuit.
            let layer = self
                .circuit_generator
                .as_ref()
                .unwrap()
                .generate_first_layer(&self.input_data)
                .await;
            self.num_virtual_layers = 0;
            Some(GkrCircuitLayer::FirstLayer(layer))
        }
    }

    fn num_layers(&self) -> usize {
        self.materialized_layers.len() + self.num_virtual_layers
    }
}

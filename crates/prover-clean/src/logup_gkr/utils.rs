use csl_cuda::TaskScope;
use slop_alloc::{Backend, CpuBackend};
use slop_tensor::Tensor;
use std::{collections::BTreeMap, iter::once};

use slop_algebra::AbstractField;
use slop_alloc::Buffer;
use slop_multilinear::{Mle, Point};
use sp1_hypercube::prover::Traces;
use std::sync::Arc;

use rand::Rng;

use crate::{
    config::{Ext, Felt},
    logup_gkr::{
        interactions::Interactions,
        layer::{JaggedFirstGkrLayer, JaggedGkrLayer},
    },
    DenseData, JaggedMle,
};

/// A layer of the GKR circuit.
///
/// This layer contains the polynomials p_0, p_1, q_0, q_1 evaluated at the layer size. The circuit
/// represents sparse values that can come from various chips with different sizes.
#[derive(Clone)]
pub struct GkrLayerGeneric<Layer: DenseData<B>, B: Backend = TaskScope> {
    /// A jagged MLE containing the polynomials p_0, p_1, q_0, q_1 evaluated at the layer size.
    pub jagged_mle: JaggedMle<Layer, B>,
    /// The number of rows / 2 for each interaction.
    pub interaction_col_sizes: Vec<u32>,
    /// The number of row variables.
    pub num_row_variables: u32,
    /// The total number of interaction variables
    pub num_interaction_variables: u32,
}

#[allow(clippy::type_complexity)]
pub struct GkrInputData {
    pub interactions: BTreeMap<String, Arc<Interactions<Felt, CpuBackend>>>,
    pub traces: Traces<Felt, TaskScope>,
    pub preprocessed_traces: Traces<Felt, TaskScope>,
    pub alpha: Ext,
    pub beta_seed: Point<Ext>,
}
pub struct FirstLayerData<F, EF, B: Backend> {
    pub numerator: Tensor<F, B>,
    pub denominator: Tensor<EF, B>,
}

pub type GkrLayer<B = TaskScope> = GkrLayerGeneric<JaggedGkrLayer<B>, B>;

pub type FirstGkrLayer<B = TaskScope> = GkrLayerGeneric<JaggedFirstGkrLayer<B>, B>;

/// A layer of the GKR circuit.
pub enum GkrCircuitLayer<B: Backend = TaskScope> {
    Materialized(GkrLayer<B>),
    FirstLayer(FirstGkrLayer<B>),
    FirstLayerVirtual(GkrInputData),
}

/// A polynomial layer of the GKR circuit.
#[derive(Clone)]
pub enum PolynomialLayer<B: Backend = TaskScope> {
    CircuitLayer(GkrLayer<B>),
    InteractionsLayer(Tensor<Ext, TaskScope>),
}

/// The first layer polynomial of the GKR circuit.
pub struct FirstLayerPolynomial {
    pub layer: FirstGkrLayer,
    pub eq_row: Mle<Ext, TaskScope>,
    pub eq_interaction: Mle<Ext, TaskScope>,
    pub lambda: Ext,
    pub point: Point<Ext>,
}

impl FirstLayerPolynomial {
    pub fn num_variables(&self) -> u32 {
        self.eq_row.num_variables() + self.eq_interaction.num_variables()
    }
}

#[derive(Clone)]
pub struct LogupRoundPolynomial {
    /// The values of the numerator and denominator polynomials.
    pub layer: PolynomialLayer,
    /// The partial lagrange evaluation for the row variables.
    pub eq_row: Mle<Ext, TaskScope>,
    /// The partial lagrange evaluation for the interaction variables.
    pub eq_interaction: Mle<Ext, TaskScope>,
    /// The correction term for the eq polynomial.
    pub eq_adjustment: Ext,
    /// The correction term for padding.
    pub padding_adjustment: Ext,
    /// The batching factor for the numerator and denominator claims.
    pub lambda: Ext,
    /// The random point for the current GKR round.
    pub point: Point<Ext>,
}

impl LogupRoundPolynomial {
    pub fn num_variables(&self) -> u32 {
        self.eq_row.num_variables() + self.eq_interaction.num_variables()
    }
}

pub struct GkrTestData {
    pub numerator_0: Mle<Ext, CpuBackend>,
    pub numerator_1: Mle<Ext, CpuBackend>,
    pub denominator_0: Mle<Ext, CpuBackend>,
    pub denominator_1: Mle<Ext, CpuBackend>,
}

/// Generates a random first layer from an rng, and some interaction row counts.
///
/// Padded to num_row_variables if provided.
pub async fn random_first_layer<R: Rng>(
    rng: &mut R,
    interaction_col_sizes: Vec<u32>,
    num_row_variables: Option<u32>,
) -> FirstGkrLayer<CpuBackend> {
    let max_row_variables =
        interaction_col_sizes.iter().max().copied().unwrap().next_power_of_two().ilog2() + 1;

    let num_row_variables = if let Some(num_vars) = num_row_variables {
        assert!(num_vars >= max_row_variables);
        num_vars
    } else {
        max_row_variables
    };

    let num_interaction_variables = interaction_col_sizes.len().next_power_of_two().ilog2();

    let interaction_start_indices = once(0)
        .chain(interaction_col_sizes.iter().scan(0u32, |acc, x| {
            *acc += x;
            Some(*acc)
        }))
        .collect::<Buffer<_>>();
    let height = interaction_start_indices.last().copied().unwrap() as usize;
    let col_index = interaction_col_sizes
        .iter()
        .enumerate()
        .flat_map(|(i, c)| vec![i as u32; *c as usize])
        .collect::<Buffer<_>>();

    let numerator = Tensor::<Felt>::rand(rng, [2, 1, height << 1]);
    let denominator = Tensor::<Ext>::rand(rng, [2, 1, height << 1]);
    let layer_data = JaggedFirstGkrLayer::new(numerator, denominator, height);

    let jagged_mle = JaggedMle::new(layer_data, col_index, interaction_start_indices);

    FirstGkrLayer {
        jagged_mle,
        interaction_col_sizes,
        num_interaction_variables,
        num_row_variables,
    }
}

/// Generates a random layer from an rng, and some interaction row counts.
///
/// Padded to num_row_variables if provided.
pub async fn random_layer<R: Rng>(
    rng: &mut R,
    interaction_col_sizes: Vec<u32>,
    num_row_variables: Option<u32>,
) -> GkrLayer<CpuBackend> {
    let max_row_variables =
        interaction_col_sizes.iter().max().copied().unwrap().next_power_of_two().ilog2() + 1;

    let num_row_variables = if let Some(num_vars) = num_row_variables {
        assert!(num_vars >= max_row_variables);
        num_vars
    } else {
        max_row_variables
    };

    let num_interaction_variables = interaction_col_sizes.len().next_power_of_two().ilog2();

    let interaction_start_indices = once(0)
        .chain(interaction_col_sizes.iter().scan(0u32, |acc, x| {
            *acc += x;
            Some(*acc)
        }))
        .collect::<Buffer<_>>();
    let height = interaction_start_indices.last().copied().unwrap() as usize;
    let col_index = interaction_col_sizes
        .iter()
        .enumerate()
        .flat_map(|(i, c)| {
            let data = i as u32;
            vec![data; *c as usize]
        })
        .collect::<Buffer<_>>();

    let layer_data = Tensor::<Ext>::rand(rng, [4, 1, 2 * height]);

    let jagged_gkr_layer = JaggedGkrLayer::new(layer_data, height);

    GkrLayer {
        jagged_mle: JaggedMle::new(jagged_gkr_layer, col_index, interaction_start_indices),
        interaction_col_sizes,
        num_interaction_variables,
        num_row_variables,
    }
}

/// Generates test data for a layer.
pub async fn generate_test_data<R: Rng>(
    rng: &mut R,
    interaction_col_sizes: Vec<u32>,
    num_row_variables: Option<u32>,
) -> (GkrLayer<CpuBackend>, GkrTestData) {
    let layer = random_layer(rng, interaction_col_sizes, num_row_variables).await;
    let test_data = get_polys_from_layer(&layer).await;
    (layer, test_data)
}

/// Gets nicely formatted numerator_0, numerator_1, denominator_0, denominator_1 polynomials from dense GkrLayer data.
/// Materializes padding for each row to 2^num_row_variables.
pub async fn get_polys_from_layer(layer: &GkrLayer<CpuBackend>) -> GkrTestData {
    let GkrLayer {
        jagged_mle: JaggedMle { dense_data: layer_data, .. },
        interaction_col_sizes,
        num_interaction_variables,
        num_row_variables,
        ..
    } = layer;

    let full_padded_height = 1usize << num_row_variables;
    let get_mle = |values: Vec<Ext>,
                   padding: Ext,
                   interaction_col_sizes: &[u32],
                   num_interaction_variables: u32,
                   full_padded_height: usize| {
        let total_size = (1 << num_interaction_variables) * full_padded_height;

        // Pre-allocate the entire result vector
        let mut result = vec![padding; total_size];

        // Calculate cumulative sizes to know where to read from values
        let mut read_offset = 0;

        // Process each interaction in forward order
        for (i, &row_count) in interaction_col_sizes.iter().enumerate() {
            let h = (row_count as usize) << 1;
            let write_start = i * full_padded_height;

            // // Copy the actual values
            result[write_start..write_start + h]
                .copy_from_slice(&values[read_offset..read_offset + h]);
            // The rest is already filled with padding

            read_offset += h;
        }

        // Padding polynomials are already in place (filled with padding value)
        Mle::from(result)
    };

    // Extract numerator_0, numerator_1, denominator_0, denominator_1 from the layer_data in parallel
    let data_0 = layer_data.layer.get(0).unwrap().get(0).unwrap().as_slice().to_vec();
    let data_1 = layer_data.layer.get(1).unwrap().get(0).unwrap().as_slice().to_vec();
    let data_2 = layer_data.layer.get(2).unwrap().get(0).unwrap().as_slice().to_vec();
    let data_3 = layer_data.layer.get(3).unwrap().get(0).unwrap().as_slice().to_vec();

    let interaction_col_sizes = interaction_col_sizes.clone();
    let irc1 = interaction_col_sizes.clone();
    let irc2 = interaction_col_sizes.clone();
    let irc3 = interaction_col_sizes.clone();
    let num_interaction_vars = *num_interaction_variables;

    let (numerator_0, numerator_1, denominator_0, denominator_1) = tokio::join!(
        tokio::task::spawn_blocking(move || get_mle(
            data_0,
            Ext::zero(),
            &interaction_col_sizes,
            num_interaction_vars,
            full_padded_height
        )),
        tokio::task::spawn_blocking(move || get_mle(
            data_1,
            Ext::zero(),
            &irc1,
            num_interaction_vars,
            full_padded_height
        )),
        tokio::task::spawn_blocking(move || get_mle(
            data_2,
            Ext::one(),
            &irc2,
            num_interaction_vars,
            full_padded_height
        )),
        tokio::task::spawn_blocking(move || get_mle(
            data_3,
            Ext::one(),
            &irc3,
            num_interaction_vars,
            full_padded_height
        ))
    );

    GkrTestData {
        numerator_0: numerator_0.unwrap(),
        numerator_1: numerator_1.unwrap(),
        denominator_0: denominator_0.unwrap(),
        denominator_1: denominator_1.unwrap(),
    }
}

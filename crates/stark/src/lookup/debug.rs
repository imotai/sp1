use std::collections::BTreeMap;

use itertools::Itertools;
use slop_algebra::{AbstractField, Field, PackedValue, PrimeField32, PrimeField64};
use slop_alloc::CpuBackend;
use slop_baby_bear::BabyBear;
use slop_matrix::Matrix;
use slop_multilinear::Mle;

use crate::{
    air::InteractionScope,
    prover::{MachineProverComponents, MachineProvingKey, Traces},
    Chip, Machine, MachineConfig,
};

use super::InteractionKind;

use crate::air::MachineAir;

/// The data for an interaction.
#[derive(Debug)]
pub struct InteractionData<F: Field> {
    /// The chip name.
    pub chip_name: String,
    /// The kind of interaction.
    pub kind: InteractionKind,
    /// The row of the interaction.
    pub row: usize,
    /// The interaction number.
    pub interaction_number: usize,
    /// Whether the interaction is a send.
    pub is_send: bool,
    /// The multiplicity of the interaction.
    pub multiplicity: F,
}

/// Converts a vector of field elements to a string.
#[allow(clippy::needless_pass_by_value)]
#[must_use]
pub fn vec_to_string<F: Field>(vec: Vec<F>) -> String {
    let mut result = String::from("(");
    for (i, value) in vec.iter().enumerate() {
        if i != 0 {
            result.push_str(", ");
        }
        result.push_str(&value.to_string());
    }
    result.push(')');
    result
}

/// Display field elements as signed integers on the range `[-modulus/2, modulus/2]`.
///
/// This presentation is useful when debugging interactions as it makes it clear which
// interactions /// are `send` and which are `receive`.
fn field_to_int<F: PrimeField32>(x: F) -> i32 {
    let modulus = BabyBear::ORDER_U64;
    let val = x.as_canonical_u64();
    if val > modulus / 2 {
        val as i32 - modulus as i32
    } else {
        val as i32
    }
}

/// Debugs the interactions of a chip.
#[allow(clippy::type_complexity)]
#[allow(clippy::needless_pass_by_value)]
pub fn debug_interactions<F: Field, A: MachineAir<F>>(
    chip: &Chip<F, A>,
    preprocessed_traces: &Traces<F, CpuBackend>,
    traces: &Traces<F, CpuBackend>,
    interaction_kinds: Vec<InteractionKind>,
    scope: InteractionScope,
) -> (BTreeMap<String, Vec<InteractionData<F>>>, BTreeMap<String, F>) {
    let mut key_to_vec_data = BTreeMap::new();
    let mut key_to_count = BTreeMap::new();

    // let trace = chip.generate_trace(record, &mut A::Record::default());
    let main = traces.get(&chip.name()).cloned().unwrap();
    let pre_traces = preprocessed_traces.get(&chip.name()).cloned();

    // let main: Mle<C::F> = trace.clone().into();

    let height = main.clone().num_real_entries();

    let sends = chip.sends().iter().filter(|s| s.scope == scope).collect::<Vec<_>>();
    let receives = chip.receives().iter().filter(|r| r.scope == scope).collect::<Vec<_>>();

    let nb_send_interactions = sends.len();
    for row in 0..height {
        for (m, interaction) in sends.iter().chain(receives.iter()).enumerate() {
            if !interaction_kinds.contains(&interaction.kind) {
                continue;
            }
            let empty = vec![];
            let preprocessed_row = match pre_traces {
                Some(ref t) => t
                    .inner()
                    .as_ref()
                    .map_or(empty.as_slice(), |t| t.guts().get(row).unwrap().as_slice()),
                None => empty.as_slice(),
            };

            let is_send = m < nb_send_interactions;

            let main_row =
                main.inner().as_ref().unwrap().guts().get(row).unwrap().as_slice().to_vec();

            let multiplicity_eval: F = interaction.multiplicity.apply(preprocessed_row, &main_row);

            if !multiplicity_eval.is_zero() {
                let mut values = vec![];
                for value in &interaction.values {
                    let expr: F = value.apply(preprocessed_row, &main_row);
                    values.push(expr);
                }
                let key = format!(
                    "{} {} {}",
                    &interaction.scope.to_string(),
                    &interaction.kind.to_string(),
                    vec_to_string(values)
                );
                key_to_vec_data.entry(key.clone()).or_insert_with(Vec::new).push(InteractionData {
                    chip_name: chip.name(),
                    kind: interaction.kind,
                    row,
                    interaction_number: m,
                    is_send,
                    multiplicity: multiplicity_eval,
                });
                let current = key_to_count.entry(key.clone()).or_insert(F::zero());
                if is_send {
                    *current += multiplicity_eval;
                } else {
                    *current -= multiplicity_eval;
                }
            }
        }
    }

    (key_to_vec_data, key_to_count)
}

/// Calculate the number of times we send and receive each event of the given interaction type,
/// and print out the ones for which the set of sends and receives don't match.
#[allow(clippy::needless_pass_by_value)]
pub fn debug_interactions_with_all_chips<F, A>(
    chips: &[Chip<F, A>],
    // pkey: &MachineProvingKey<PC>,
    preprocessed_traces: &Traces<F, CpuBackend>,
    // shards: &[A::Record],
    traces: &Traces<F, CpuBackend>,
    interaction_kinds: Vec<InteractionKind>,
    scope: InteractionScope,
) -> bool
where
    F: Field,
    A: MachineAir<F>,
{
    // if scope == InteractionScope::Local {
    //     assert!(shards.len() == 1);
    // }

    let mut final_map = BTreeMap::new();
    let mut total = F::zero();

    // let chips = machine.chips();
    for chip in chips.iter() {
        let mut total_events = 0;
        // for shard in shards {
        //     if !chip.included(shard) {
        //         continue;
        //     }
        // eprintln!("{}", chip.name());
        let (_, count) = debug_interactions::<F, A>(
            chip,
            preprocessed_traces,
            traces,
            interaction_kinds.clone(),
            scope,
        );
        total_events += count.len();
        for (key, value) in count.iter() {
            let entry = final_map.entry(key.clone()).or_insert((F::zero(), BTreeMap::new()));
            entry.0 += *value;
            total += *value;
            *entry.1.entry(chip.name()).or_insert(F::zero()) += *value;
        }
        // }
        tracing::info!("{} chip has {} distinct events", chip.name(), total_events);
    }

    tracing::info!("Final counts below.");
    tracing::info!("==================");

    let mut any_nonzero = false;
    for (key, (value, chip_values)) in final_map.clone() {
        if !F::is_zero(&value) {
            tracing::info!("Interaction key: {} Send-Receive Discrepancy: {}", key, value);
            any_nonzero = true;
            for (chip, chip_value) in chip_values {
                tracing::info!(
                    " {} chip's send-receive discrepancy for this key is {}",
                    chip,
                    chip_value
                );
            }
        }
    }

    tracing::info!("==================");
    if !any_nonzero {
        tracing::info!("All chips have the same number of sends and receives.");
    } else {
        // tracing::info!("Positive values mean sent more than received.");
        // tracing::info!("Negative values mean received more than sent.");
        // if total != F::zero() {
        //     tracing::info!("Total send-receive discrepancy: {}", field_to_int(total));
        //     if field_to_int(total) > 0 {
        //         tracing::info!("you're sending more than you are receiving");
        //     } else {
        //         tracing::info!("you're receiving more than you are sending");
        //     }
        // } else {
        //     tracing::info!(
        //         "the total number of sends and receives match, but the keys don't match"
        //     );
        //     tracing::info!("check the arguments");
        // }
    }

    !any_nonzero
}

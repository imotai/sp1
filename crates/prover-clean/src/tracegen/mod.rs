use core::pin::pin;
use std::collections::{BTreeMap, BTreeSet};
use std::future::ready;
use std::ops::Range;
use std::sync::Arc;

use csl_cuda::TaskScope;
use csl_tracegen::CudaTracegenAir;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use slop_air::BaseAir;
use slop_alloc::{Backend, Buffer, CopyIntoBackend, HasBackend, Slice};
use slop_multilinear::Mle;
use sp1_hypercube::{air::MachineAir, prover::ProverPermit, prover::ProverSemaphore, Machine};

use sp1_hypercube::MachineRecord;
use tracing::debug_span;
use tracing::instrument::Instrument;

use crate::config::Felt;
use crate::DenseData;
use crate::JaggedMle;

pub type JaggedTraceMle<B> = JaggedMle<TraceDenseData<B>, B>;

/// Jagged representation of the traces.
pub struct TraceDenseData<B: Backend> {
    /// The dense representation of the traces.
    pub dense: Buffer<Felt, B>,
    /// The dense offset of the preprocessed traces.
    pub preprocessed_offset: usize,
    /// The total number of columns in the preprocessed traces.
    pub preprocessed_cols: usize,
    /// A mapping from chip name to the range of columns it occupies for preprocessed traces.
    pub preprocessed_table_index: BTreeMap<String, Range<usize>>,
    /// A mapping from chip name to the range of columns it occupies for main traces.
    pub main_table_index: BTreeMap<String, Range<usize>>,
    /// A permit, for limiting access to the GPU memory.
    pub permit: ProverPermit,
}

/// The raw pointer to the dense data, for use in CUDA FFI calls.
#[allow(dead_code)]
#[repr(C)]
pub struct TraceDenseDataRaw {
    dense: *const Felt,
}

/// The raw pointer to the dense data, for use in CUDA FFI calls.
#[allow(dead_code)]
#[repr(C)]
pub struct TraceDenseDataMutRaw {
    dense: *mut Felt,
}

impl<B: Backend> DenseData<B> for TraceDenseData<B> {
    type DenseDataRaw = TraceDenseDataRaw;
    type DenseDataMutRaw = TraceDenseDataMutRaw;

    fn as_ptr(&self) -> Self::DenseDataRaw {
        TraceDenseDataRaw { dense: self.dense.as_ptr() }
    }

    fn as_mut_ptr(&mut self) -> Self::DenseDataMutRaw {
        TraceDenseDataMutRaw { dense: self.dense.as_mut_ptr() }
    }
}

// ------------- The following logic is mostly copied from crates/tracegen/src/lib.rs -------------
// TODO: is this a reasonable upper bound on number of columns per trace? ~16k
pub const MAX_COLS_PER_TRACE: usize = 1 << 14;

/// The output of the host phase of the tracegen.
pub struct HostPhaseTracegen<A> {
    /// Which airs were touched in the host phase.
    pub device_airs: Vec<Arc<A>>,
    /// The traces generated in the host phase.
    pub host_traces: futures::channel::mpsc::UnboundedReceiver<(String, Mle<Felt>)>,
}

/// Traces generated
pub enum Trace<B: Backend = TaskScope> {
    // Real trace
    Real(Mle<Felt, B>),
    // Number of columns
    Padding(usize),
}

/// Sets up the jagged traces.
///
/// Returns the final offset and the final number of columns.
async fn setup_jagged_traces(
    dense_data: &mut Buffer<Felt, TaskScope>,
    col_index: &mut Buffer<u32, TaskScope>,
    start_indices: &mut Buffer<u32, TaskScope>,
    traces: BTreeMap<String, Trace>,
    initial_offset: usize,
    initial_cols: usize,
) -> (usize, usize, BTreeMap<String, Range<usize>>) {
    let mut offset = initial_offset;
    let mut cols_so_far = initial_cols;
    let mut table_index = BTreeMap::new();
    let backend = dense_data.backend().clone();
    for (i, (name, trace)) in traces.iter().enumerate() {
        match trace {
            Trace::Real(trace) => {
                let trace_buf = trace.guts().as_buffer();
                let trace_num_rows = trace.num_non_zero_entries();
                let trace_num_cols = trace.num_polynomials();
                assert_eq!(trace_num_rows * trace_num_cols, trace_buf.len());
                let trace_size = trace_buf.len();
                let dst_slice: &mut Slice<_, _> = &mut dense_data[offset..offset + trace_size];
                let src_slice: &Slice<_, _> = &trace_buf[..];
                unsafe {
                    dst_slice.copy_from_slice(src_slice, &backend).unwrap();
                }

                // Materialize the col_index on host, then copy to device.
                // TODO: trivial to construct on device. But I don't think the data is large enough for this to matter.
                let current_col_index = (0..trace_num_cols as u32)
                    .flat_map(|col| vec![col + cols_so_far as u32; trace_num_rows >> 1])
                    .collect::<Buffer<_>>();
                let current_col_index_device =
                    current_col_index.copy_into_backend(&backend).await.unwrap();
                let col_index_slice: &Slice<u32, _> = &current_col_index_device[..];

                let col_index_dst_slice = &mut col_index[offset >> 1..((offset + trace_size) >> 1)];
                unsafe {
                    col_index_dst_slice.copy_from_slice(col_index_slice, &backend).unwrap();
                }

                // Materialize the start indices on host, then copy to device.
                // If this is the last one, then also put the total number of columns onto the start indices.
                let mut current_start_indices = (0..trace_num_cols)
                    .map(|col| (offset + col * trace_num_rows) as u32 >> 1)
                    .collect::<Buffer<_>>();

                if i == traces.len() - 1 {
                    current_start_indices.push((offset + trace_size) as u32 >> 1);
                }

                let current_start_indices_device =
                    current_start_indices.copy_into_backend(&backend).await.unwrap();
                let start_indices_slice: &Slice<u32, _> = &current_start_indices_device[..];

                let start_indices_dst_slice = if i == traces.len() - 1 {
                    &mut start_indices[cols_so_far..cols_so_far + trace_num_cols + 1]
                } else {
                    &mut start_indices[cols_so_far..cols_so_far + trace_num_cols]
                };
                unsafe {
                    start_indices_dst_slice.copy_from_slice(start_indices_slice, &backend).unwrap();
                }

                table_index.insert(name.clone(), cols_so_far..cols_so_far + trace_num_cols);
                offset += trace_size;
                cols_so_far += trace_num_cols;
            }
            Trace::Padding(padding_cols) => {
                // Don't touch the dense data. This trace isn't real.
                // We just need to add some dummy values to start_indices.

                let current_start_indices_vec = if i == traces.len() - 1 {
                    vec![offset as u32 >> 1; *padding_cols + 1]
                } else {
                    vec![offset as u32 >> 1; *padding_cols]
                };
                let current_start_indices =
                    current_start_indices_vec.into_iter().collect::<Buffer<_>>();

                let current_start_indices_device =
                    current_start_indices.copy_into_backend(&backend).await.unwrap();
                let start_indices_slice: &Slice<u32, _> = &current_start_indices_device[..];

                let start_indices_dst_slice = if i == traces.len() - 1 {
                    &mut start_indices[cols_so_far..cols_so_far + *padding_cols + 1]
                } else {
                    &mut start_indices[cols_so_far..cols_so_far + *padding_cols]
                };
                unsafe {
                    start_indices_dst_slice.copy_from_slice(start_indices_slice, &backend).unwrap();
                }

                table_index.insert(name.clone(), cols_so_far..cols_so_far + *padding_cols);
                cols_so_far += *padding_cols;
            }
        }
    }
    (offset, cols_so_far, table_index)
}

fn host_preprocessed_tracegen<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    program: Arc<<A as MachineAir<Felt>>::Program>,
) -> HostPhaseTracegen<A> {
    // Split chips based on where we will generate their traces.
    let (device_airs, host_airs): (Vec<_>, Vec<_>) = machine
        .chips()
        .iter()
        .map(|chip| chip.air.clone())
        .partition(|air| air.supports_device_preprocessed_tracegen());

    // Spawn a rayon task to generate the traces on the CPU.
    // `traces` is a futures Stream that will immediately begin buffering traces.
    let (host_traces_tx, host_traces) = futures::channel::mpsc::unbounded();
    slop_futures::rayon::spawn(move || {
        host_airs.into_par_iter().for_each_with(host_traces_tx, |tx, air| {
            if let Some(trace) = air.generate_preprocessed_trace(&program) {
                tx.unbounded_send((air.name(), Mle::from(trace))).unwrap();
            }
        });
        // Make this explicit.
        // If we are the last users of the program, this will expensively drop it.
        drop(program);
    });
    HostPhaseTracegen { device_airs, host_traces }
}

async fn device_preprocessed_tracegen<A: CudaTracegenAir<Felt>>(
    program: Arc<<A as MachineAir<Felt>>::Program>,
    host_phase_tracegen: HostPhaseTracegen<A>,
    backend: &TaskScope,
) -> BTreeMap<String, Trace<TaskScope>> {
    let HostPhaseTracegen { device_airs, host_traces } = host_phase_tracegen;

    // Stream that, when polled, copies the host traces to the device.
    let copied_host_traces = pin!(host_traces.then(|(name, trace)| async move {
        (name, trace.copy_into_backend(backend).await.unwrap())
    }));
    // Stream that, when polled, copies events to the device and generates traces.
    let device_traces = device_airs
        .into_iter()
        .map(|air| {
            // We want to borrow the program and move the air.
            let program = program.as_ref();
            async move {
                let maybe_trace =
                    air.generate_preprocessed_trace_device(program, backend).await.unwrap();
                (air, maybe_trace)
            }
        })
        .collect::<FuturesUnordered<_>>()
        .filter_map(|(air, maybe_trace)| ready(maybe_trace.map(|trace| (air.name(), trace))));

    let named_traces = futures::stream_select!(copied_host_traces, device_traces)
        .map(|(name, trace)| (name, Trace::Real(trace)))
        .collect::<BTreeMap<_, _>>()
        .await;

    // If we're the last users of the program, expensively drop it in a separate task.
    // TODO: in general, figure out the best way to drop expensive-to-drop things.
    rayon::spawn(move || drop(program));

    named_traces
}

/// Corresponds to `generate_preprocessed_traces`.
pub async fn setup_tracegen<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    program: Arc<<A as MachineAir<Felt>>::Program>,
    max_trace_size: usize,
    prover_permits: ProverSemaphore,
    backend: &TaskScope,
) -> JaggedTraceMle<TaskScope> {
    // Generate traces on host.
    let host_phase_tracegen = host_preprocessed_tracegen(machine, Arc::clone(&program));

    // Wait for a prover to be available.
    let permit = prover_permits.acquire().instrument(debug_span!("acquire")).await.unwrap();

    // Now that the permit is acquired, we can begin the following two tasks:
    // - Copying host traces to the device.
    // - Generating traces on the device.
    let preprocessed_traces =
        device_preprocessed_tracegen(program, host_phase_tracegen, backend).await;

    // Allocate the big buffer for jagged traces.
    let mut dense_data: Buffer<Felt, TaskScope> =
        Buffer::with_capacity_in(max_trace_size, backend.clone());
    let mut col_index: Buffer<u32, TaskScope> =
        Buffer::with_capacity_in(max_trace_size >> 1, backend.clone());

    let mut start_indices: Buffer<u32, TaskScope> =
        Buffer::with_capacity_in(MAX_COLS_PER_TRACE, backend.clone());

    unsafe {
        dense_data.assume_init();
        col_index.assume_init();
        start_indices.assume_init();
    }

    // Put them in right places. Todo: parallelize.
    let (preprocessed_offset, preprocessed_cols, preprocessed_table_index) = setup_jagged_traces(
        &mut dense_data,
        &mut col_index,
        &mut start_indices,
        preprocessed_traces,
        0,
        0,
    )
    .await;

    let trace_dense_data: TraceDenseData<TaskScope> = TraceDenseData {
        dense: dense_data,
        preprocessed_offset,
        preprocessed_cols,
        preprocessed_table_index,
        main_table_index: BTreeMap::new(),
        permit,
    };

    JaggedMle { dense_data: trace_dense_data, col_index, start_indices }
}

fn host_main_tracegen<A>(
    machine: &Machine<Felt, A>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
) -> (HostPhaseTracegen<A>, BTreeMap<String, Trace<TaskScope>>)
where
    A: CudaTracegenAir<Felt>,
{
    // Set of chips we need to generate traces for.
    let chip_set = machine
        .chips()
        .iter()
        .filter(|chip| chip.included(&record))
        .cloned()
        .collect::<BTreeSet<_>>();

    // Split chips based on where we will generate their traces.
    let (device_airs, host_airs): (Vec<_>, Vec<_>) = chip_set
        .iter()
        .map(|chip| chip.air.clone())
        .partition(|c| c.supports_device_main_tracegen());

    // Spawn a rayon task to generate the traces on the CPU.
    // `host_traces` is a futures Stream that will immediately begin buffering traces.
    let (host_traces_tx, host_traces) = futures::channel::mpsc::unbounded();
    slop_futures::rayon::spawn(move || {
        host_airs.into_par_iter().for_each_with(host_traces_tx, |tx, air| {
            let trace = Mle::from(air.generate_trace(&record, &mut A::Record::default()));
            // Since it's unbounded, it will only error if the receiver is disconnected.
            tx.unbounded_send((air.name(), trace)).unwrap();
        });
        // Make this explicit.
        // If we are the last users of the record, this will expensively drop it.
        drop(record);
    });

    // Get the smallest cluster containing our tracegen chip set.
    let shard_chips = machine.smallest_cluster(&chip_set).unwrap().clone();
    // For every AIR in the cluster, make a (virtual) padded trace.
    let initial_traces = shard_chips
        .iter()
        .filter(|chip| !chip_set.contains(chip))
        .map(|chip| {
            let num_polynomials = chip.width();
            (chip.name(), Trace::Padding(num_polynomials))
        })
        .collect::<BTreeMap<_, _>>();

    (HostPhaseTracegen { device_airs, host_traces }, initial_traces)
}

/// Puts traces on device. Returns (traces, public values).
async fn device_main_tracegen<A: CudaTracegenAir<Felt>>(
    host_phase_tracegen: HostPhaseTracegen<A>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    initial_traces: BTreeMap<String, Trace>,
    backend: &TaskScope,
) -> (BTreeMap<String, Trace>, Vec<Felt>) {
    let HostPhaseTracegen { device_airs, host_traces } = host_phase_tracegen;

    // Stream that, when polled, copies the host traces to the device.
    let copied_host_traces = pin!(host_traces.then(|(name, trace)| async move {
        (name, trace.copy_into_backend(backend).await.unwrap())
    }));
    // Stream that, when polled, copies events to the device and generates traces.
    let device_traces = device_airs
        .into_iter()
        .map(|air| {
            // We want to borrow the record and move the chip.
            let record = record.as_ref();
            async move {
                let trace = air
                    .generate_trace_device(record, &mut A::Record::default(), backend)
                    .await
                    .unwrap();
                (air.name(), trace)
            }
        })
        .collect::<FuturesUnordered<_>>();

    let mut all_traces = initial_traces;

    // Combine the host and device trace streams and insert them into `all_traces`.
    futures::stream_select!(copied_host_traces, device_traces)
        .for_each(|(name, trace)| {
            all_traces.insert(name, Trace::Real(trace));
            ready(())
        })
        .await;

    // All traces are now generated, so the public values are ready.
    // That is, this value will have the correct global cumulative sum.
    let public_values = record.public_values::<Felt>();

    // If we're the last users of the record, expensively drop it in a separate task.
    // TODO: in general, figure out the best way to drop expensive-to-drop things.
    rayon::spawn(move || drop(record));

    (all_traces, public_values)
}

/// Corresponds to `generate_main_traces`.
/// Mutates jagged_traces in place, and returns public values.
pub async fn main_tracegen<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    jagged_traces: &mut JaggedTraceMle<TaskScope>,
    backend: &TaskScope,
) -> Vec<Felt> {
    // Start generating traces on host.
    let (host_phase_tracegen, initial_traces) = host_main_tracegen(machine, record.clone());

    // Now that the permit is acquired, we can begin the following two tasks:
    // - Copying host traces to the device.
    // - Generating traces on the device.
    let (traces, public_values) =
        device_main_tracegen(host_phase_tracegen, record, initial_traces, backend).await;

    // At this point, all traces are on device. Now we need to copy them into the Jagged MLE struct.
    let JaggedMle { dense_data: trace_dense_data, col_index, start_indices } = jagged_traces;

    let TraceDenseData {
        dense: dense_data,
        preprocessed_offset,
        preprocessed_cols,
        main_table_index,
        ..
    } = trace_dense_data;

    // Put them in right places. Todo: parallelize.
    let (final_offset, final_cols, new_main_table_index) = setup_jagged_traces(
        dense_data,
        col_index,
        start_indices,
        traces,
        *preprocessed_offset,
        *preprocessed_cols,
    )
    .await;

    *main_table_index = new_main_table_index;

    // Shrink the len of the dense data to match the actual size.
    unsafe {
        dense_data.set_len(final_offset);
        col_index.set_len(final_offset >> 1);
        start_indices.set_len(final_cols + 1);
    }
    public_values
}

pub async fn full_tracegen<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    program: Arc<<A as MachineAir<Felt>>::Program>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    max_trace_size: usize,
    prover_permits: ProverSemaphore,
    backend: &TaskScope,
) -> (Vec<Felt>, JaggedTraceMle<TaskScope>) {
    let mut jagged_mle =
        setup_tracegen(machine, program, max_trace_size, prover_permits, backend).await;
    let public_values = main_tracegen(machine, record, &mut jagged_mle, backend).await;
    (public_values, jagged_mle)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use csl_cuda::{run_in_place, sys::prover_clean::jagged_eval_kernel_chunked_felt, ToDevice};
    use csl_tracegen::CudaTraceGenerator;
    use slop_alloc::IntoHost;
    use slop_multilinear::{Mle, Point};
    use sp1_core_executor::SP1RecursionProof;
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};
    use sp1_prover::{CompressAir, InnerSC, SP1CircuitWitness};
    use sp1_recursion_circuit::machine::SP1ShapedWitnessValues;
    use sp1_sdk::{CpuProver, Prover};

    use crate::{
        config::Ext,
        tracegen::full_tracegen,
        zerocheck::{data::DenseBuffer, evaluate_jagged_mle_chunked},
        JaggedMle,
    };

    const FIBONACCI_PROOF: &[u8] = include_bytes!("../../fib_proof.bin");

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Takes a pre-generated proof and vk, and generates traces for the shrink program.
    /// Then, asserts that the jagged traces generated are the same as the traces in the old format.
    #[tokio::test]
    async fn test_jagged_tracegen() {
        // This is basically the same as LocalProver::shrink
        let client = CpuProver::new_unsound().await;
        let machine = CompressAir::compress_machine();
        let inner = client.inner();
        let prover = inner.prover();

        let compressed_proof: SP1RecursionProof<_, InnerSC> =
            bincode::deserialize(FIBONACCI_PROOF).unwrap();

        let SP1RecursionProof { vk: compressed_vk, proof: compressed_proof } = compressed_proof;
        let input = SP1ShapedWitnessValues {
            vks_and_proofs: vec![(compressed_vk.clone(), compressed_proof)],
            is_complete: true,
        };

        let input = prover.recursion().make_merkle_proofs(input);
        let witness = SP1CircuitWitness::Shrink(input);

        let record = prover.recursion().execute(witness).unwrap();
        let program = record.program.clone();

        // This tests core tracegen, which is more comprehensive, but since core records are so big,
        // it's not feasible to git commit them.

        // // Load the record and program from the test artifacts
        // let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        // let record_compressed_bytes = std::fs::read(format!("{}/record_1.bin", &root)).unwrap();
        // let record = zstd::decode_all(record_compressed_bytes.as_slice()).unwrap();
        // let record: sp1_core_executor::ExecutionRecord = bincode::deserialize(&record).unwrap();
        // let program = Arc::new(
        //     sp1_core_executor::Program::from_elf(&format!("{}/test_program.bin", &root)).unwrap(),
        // );

        // let machine = sp1_core_machine::riscv::RiscvAir::<crate::config::Felt>::machine();

        let mut rng = StdRng::seed_from_u64(4);
        run_in_place(|scope| async move {
            const CORE_MAX_LOG_ROW_COUNT: u32 = 22;
            const CORE_MAX_TRACE_SIZE: u32 = 1 << 29;
            let z_row: Point<Ext, _> = Point::rand(&mut rng, CORE_MAX_LOG_ROW_COUNT);

            let semaphore = ProverSemaphore::new(1);

            // Generate traces using the host tracegen.
            let trace_generator = CudaTraceGenerator::new_in(machine.clone(), scope.clone());
            let warmup_traces = trace_generator
                .generate_traces(
                    program.clone(),
                    record.clone(),
                    CORE_MAX_LOG_ROW_COUNT as usize,
                    semaphore.clone(),
                )
                .await;

            println!(
                "warmup traces generated: {:?}",
                warmup_traces.main_trace_data.shard_chips.len()
            );

            drop(warmup_traces);

            scope.synchronize().await.unwrap();
            let now = std::time::Instant::now();
            let old_traces = trace_generator
                .generate_traces(
                    program.clone(),
                    record.clone(),
                    CORE_MAX_LOG_ROW_COUNT as usize,
                    semaphore.clone(),
                )
                .await;
            scope.synchronize().await.unwrap();
            println!("old traces generated in {:?}", now.elapsed());

            let record = Arc::new(record);

            let mut num_cols = 0;
            let mut all_evals_host = vec![];

            // Evaluate all of the real traces at z_row. Concatenate evaluations into `all_evals_host`.
            for (_name, trace) in old_traces
                .preprocessed_traces
                .into_iter()
                .chain(old_traces.main_trace_data.traces.into_iter())
            {
                assert_eq!(trace.num_variables(), CORE_MAX_LOG_ROW_COUNT);
                if trace.num_real_entries() == 0 {
                    println!("fake trace: {}", _name);
                }

                let trace = trace.eval_at(&z_row).await;

                num_cols += trace.num_polynomials();
                let tensor = trace.into_evaluations();

                let evals_host = tensor.into_host().await.unwrap();
                all_evals_host.extend_from_slice(evals_host.as_buffer());
            }

            // Evaluate `all_evals_host` as an MLE at z_col.
            let all_evals_mle = Mle::from_buffer(all_evals_host.into());
            let num_col_variables = num_cols.next_power_of_two().ilog2();
            let z_col: Point<Ext, _> = Point::rand(&mut rng, num_col_variables);
            let old_tracegen_eval = all_evals_mle.eval_at(&z_col).await.evaluations().as_slice()[0];

            scope.synchronize().await.unwrap();
            drop(old_traces.main_trace_data.permit);
            let now = std::time::Instant::now();

            // Do tracegen with the new setup.
            let (_public_values, jagged_trace_data) = full_tracegen(
                &machine,
                program.clone(),
                record,
                CORE_MAX_TRACE_SIZE as usize,
                semaphore.clone(),
                &scope,
            )
            .await;

            println!("jagged trace data: {:#?}", jagged_trace_data.dense_data.main_table_index);

            scope.synchronize().await.unwrap();
            println!("new traces generated in {:?}", now.elapsed());

            let zerocheck_dense = DenseBuffer::new(jagged_trace_data.dense_data.dense);

            let zerocheck_jagged_mle = JaggedMle::new(
                zerocheck_dense,
                jagged_trace_data.col_index,
                jagged_trace_data.start_indices,
            );

            let num_dense_cols = zerocheck_jagged_mle.start_indices.len() - 1;

            let z_row_device = z_row.to_device_in(&scope).await.unwrap();
            let z_col_device = z_col.to_device_in(&scope).await.unwrap();

            let total_len = zerocheck_jagged_mle.dense_data.data.len() / 2;
            let zerocheck_eval = evaluate_jagged_mle_chunked(
                zerocheck_jagged_mle,
                z_row_device,
                z_col_device,
                num_dense_cols,
                total_len,
                jagged_eval_kernel_chunked_felt,
            )
            .await;

            let zerocheck_eval_host = zerocheck_eval.into_host().await.unwrap().as_slice()[0];

            assert_eq!(old_tracegen_eval, zerocheck_eval_host);
        })
        .await;
    }
}

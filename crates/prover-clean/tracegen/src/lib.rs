use core::pin::pin;
use itertools::Itertools;
use slop_alloc::mem::DeviceMemory;
use slop_futures::queue::Worker;
use slop_tensor::{Dimensions, Tensor};
use sp1_core_machine::global::GLOBAL_OFFSET_POS_COPY;
use std::collections::{BTreeMap, BTreeSet};
use std::future::ready;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;
use tokio::join;
use tokio::sync::Mutex;
use tracing::{instrument, Instrument};

use csl_cuda::sys::prover_clean::{
    count_and_add_kernel, fill_buffer, generate_col_index, generate_start_indices,
    sum_to_trace_kernel,
};
use csl_cuda::{args, PinnedBuffer, TaskScope};
use csl_tracegen::CudaTracegenAir;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use slop_air::BaseAir;
use slop_alloc::{Backend, Buffer, CopyIntoBackend, HasBackend, Slice};
use slop_challenger::IopCtx;
use slop_jagged::JaggedProverData;
use slop_multilinear::Mle;
use sp1_hypercube::prover::{ProverPermit, ProverSemaphore};

use sp1_core_executor::ELEMENT_THRESHOLD;
use sp1_hypercube::{air::MachineAir, Machine};
use sp1_hypercube::{Chip, ChipStatistics, MachineRecord};

use cslpc_basefold::ProverCleanStackedPcsProverData;
use cslpc_utils::{Felt, JaggedMle, JaggedTraceMle, TraceDenseData, TraceOffset};

pub mod test_utils;

// ------------- The following logic is mostly copied from crates/tracegen/src/lib.rs -------------
// TODO: is this a reasonable upper bound on number of columns per trace? ~16k
pub const MAX_COLS_PER_TRACE: usize = 1 << 14;
pub const CORE_MAX_TRACE_SIZE: u32 = (ELEMENT_THRESHOLD + (ELEMENT_THRESHOLD >> 1)) as u32;

/// The output of the host phase of the tracegen.
pub struct HostPhaseTracegen<A> {
    /// Which airs need to be generated on device.
    pub device_airs: Vec<Arc<A>>,
    /// The real traces generated in the host phase.
    pub host_traces: futures::channel::mpsc::UnboundedReceiver<(String, usize, usize, usize)>,
}

/// Information about the traces generated in the host phase.
pub struct HostPhaseShapeInfo<A> {
    /// The traces generated in the host phase.
    pub traces_by_name: BTreeMap<String, Trace<TaskScope>>,
    /// The set of chips we need to generate traces for.
    pub chip_set: BTreeSet<Chip<Felt, A>>,
}

/// Traces generated
pub enum Trace<B: Backend = TaskScope> {
    // Real trace
    Real(Mle<Felt, B>),
    // Number of columns
    Padding(usize),
}

impl Trace<TaskScope> {
    pub fn num_real_entries(&self) -> usize {
        match self {
            Trace::Real(mle) => mle.num_non_zero_entries(),
            Trace::Padding(_) => 0,
        }
    }
}

pub struct CudaShardProverData<GC: IopCtx, Air: MachineAir<GC::F>> {
    /// The preprocessed traces.
    pub preprocessed_traces: JaggedTraceMle<Felt, TaskScope>,
    /// The pcs data for the preprocessed traces.
    pub preprocessed_data: JaggedProverData<GC, ProverCleanStackedPcsProverData<GC>>,
    phantom: PhantomData<Air>,
}

impl<GC: IopCtx, Air: MachineAir<GC::F>> CudaShardProverData<GC, Air> {
    pub fn new(
        preprocessed_traces: JaggedTraceMle<Felt, TaskScope>,
        preprocessed_data: JaggedProverData<GC, ProverCleanStackedPcsProverData<GC>>,
    ) -> Self {
        Self { preprocessed_traces, preprocessed_data, phantom: PhantomData }
    }

    pub fn preprocessed_table_heights(&self) -> BTreeMap<String, usize> {
        self.preprocessed_traces
            .dense()
            .preprocessed_table_index
            .iter()
            .map(|(name, offset)| (name.clone(), offset.poly_size))
            .collect()
    }
}

fn fill_buf(dst: *mut u32, val: u32, len: usize, max_log_row_count: u32, backend: &TaskScope) {
    let args = args!(dst, val, max_log_row_count, len);
    const BLOCK_DIM: usize = 256;
    let grid_dim = len.div_ceil(BLOCK_DIM);

    unsafe {
        backend.launch_kernel(fill_buffer(), grid_dim, BLOCK_DIM, &args, 0).unwrap();
    }
}

fn count_and_add(dst: *mut u32, src: *const Felt, len: usize, backend: &TaskScope) {
    let args = args!(dst, src, len);
    const BLOCK_DIM: usize = 16;
    const NUM_BINS: usize = 256;
    let grid_dim: usize = 1024;

    let shared_mem = NUM_BINS * std::mem::size_of::<u32>() * BLOCK_DIM;

    unsafe {
        backend
            .launch_kernel(count_and_add_kernel(), grid_dim, BLOCK_DIM, &args, shared_mem)
            .unwrap();
    }
}

fn sum_to_trace(dst: *mut Felt, src: *const u32, backend: &TaskScope) {
    let args = args!(dst, src);
    const BLOCK_DIM: usize = 128;
    let grid_dim: usize = 32;

    unsafe {
        backend.launch_kernel(sum_to_trace_kernel(), grid_dim, BLOCK_DIM, &args, 0).unwrap();
    }
}

/// Sets up the jagged traces. TODO: can use fewer arguments by packing the mutable stuff into TraceDenseData.
///
/// Returns the final offset, the final number of columns, the amount of padding, and the table index.
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, level = "debug")]
async fn generate_jagged_traces(
    dense_data: &mut Buffer<Felt, TaskScope>,
    col_index: &mut Buffer<u32, TaskScope>,
    start_indices: &mut Buffer<u32, TaskScope>,
    column_heights: &mut Vec<u32>,
    traces: BTreeMap<String, Trace>,
    initial_offset: usize,
    initial_cols: usize,
    log_stacking_height: u32,
    max_log_row_count: u32,
) -> (usize, usize, usize, BTreeMap<String, TraceOffset>) {
    let mut offset = initial_offset;
    let mut cols_so_far = initial_cols;
    let mut table_index = BTreeMap::new();
    let backend = dense_data.backend().clone();
    column_heights.truncate(initial_cols);

    // Maps chip name -> (dense range, col index range, start indices range)
    let mut trace_offsets = BTreeMap::new();

    // First, get the offsets for each trace.
    for (name, trace) in traces.iter() {
        match trace {
            Trace::Real(trace) => {
                let trace_buf = trace.guts().as_buffer();
                let trace_num_rows = trace.num_non_zero_entries();
                let trace_num_cols = trace.num_polynomials();
                let trace_size = trace_buf.len();

                let dense_range = offset..offset + trace_size;
                let col_index_range = offset >> 1..((offset + trace_size) >> 1);
                let start_indices_range = cols_so_far..cols_so_far + trace_num_cols;

                trace_offsets.insert(
                    name.clone(),
                    (dense_range.clone(), col_index_range, start_indices_range),
                );

                let current_column_heights = vec![(trace_num_rows >> 1) as u32; trace_num_cols];
                column_heights.extend_from_slice(&current_column_heights);

                table_index.insert(
                    name.clone(),
                    TraceOffset {
                        dense_offset: dense_range,
                        poly_size: trace_num_rows,
                        num_polys: trace_num_cols,
                    },
                );
                offset += trace_size;
                cols_so_far += trace_num_cols;
            }
            Trace::Padding(padding_cols) => {
                column_heights.extend_from_slice(&vec![0; *padding_cols]);
                table_index.insert(
                    name.clone(),
                    TraceOffset {
                        dense_offset: offset..offset,
                        poly_size: 0,
                        num_polys: *padding_cols,
                    },
                );
                trace_offsets.insert(
                    name.clone(),
                    (
                        usize::MAX..usize::MAX,
                        usize::MAX..usize::MAX,
                        cols_so_far..cols_so_far + *padding_cols,
                    ),
                );
                cols_so_far += *padding_cols;
            }
        }
    }
    offset = initial_offset;
    cols_so_far = initial_cols;

    for (name, trace) in traces.iter() {
        let (dense_range, col_index_range, start_indices_range) = trace_offsets.get(name).unwrap();
        match trace {
            Trace::Real(trace) => {
                let trace_buf = trace.guts().as_buffer();
                let trace_num_rows = trace.num_non_zero_entries();
                let trace_num_cols = trace.num_polynomials();
                let trace_size = trace_buf.len();
                assert_eq!(trace_num_rows * trace_num_cols, trace_buf.len());

                tracing::trace_span!("dense buffer copy", chip = %name).in_scope(|| {
                    let dst_slice: &mut Slice<_, _> = &mut dense_data[dense_range.clone()];
                    let src_slice: &Slice<_, _> = &trace_buf[..];
                    unsafe {
                        dst_slice.copy_from_slice(src_slice, &backend).unwrap();
                    }
                });

                tracing::trace_span!("col index gen", chip = %name).in_scope(|| unsafe {
                    let args = args!(
                        col_index.as_mut_ptr().add(col_index_range.start),
                        cols_so_far as u32,
                        trace_num_cols,
                        trace_num_rows
                    );
                    const BLOCK_SIZE: usize = 256;
                    let grid_dim = ((trace_num_cols * trace_num_rows) >> 1).div_ceil(BLOCK_SIZE);
                    backend
                        .launch_kernel(generate_col_index(), grid_dim, BLOCK_SIZE, &args, 0)
                        .unwrap();
                });

                tracing::trace_span!("start indices gen", chip = %name).in_scope(|| unsafe {
                    let args = args!(
                        start_indices.as_mut_ptr().add(start_indices_range.start),
                        offset,
                        trace_num_cols,
                        trace_num_rows
                    );
                    const BLOCK_SIZE: usize = 256;
                    let grid_dim = trace_num_cols.div_ceil(BLOCK_SIZE);
                    backend
                        .launch_kernel(generate_start_indices(), grid_dim, BLOCK_SIZE, &args, 0)
                        .unwrap();
                });

                offset += trace_size;
                cols_so_far += trace_num_cols;
            }
            Trace::Padding(padding_cols) => {
                // Don't touch the dense data. This trace isn't real.
                // We just need to add some dummy values to start_indices.

                tracing::trace_span!("padding start indices gen and copy", chip = %name, padding_cols = %padding_cols).in_scope(
                    || unsafe {
                        fill_buf(start_indices.as_mut_ptr().add(start_indices_range.start), (offset >> 1) as u32, *padding_cols, max_log_row_count, &backend);
                    },
                );

                cols_so_far += *padding_cols;
            }
        }
    }

    let result = async {
        // Now, pad the dense data with 0's to the next multiple of 2^log_stacking_height.
        let next_multiple = offset.next_multiple_of(1 << log_stacking_height);
        let num_added_vals = next_multiple - offset;
        let num_added_cols = num_added_vals.div_ceil(1 << max_log_row_count);
        let remainder = num_added_vals % (1 << max_log_row_count);
        if next_multiple == offset {
            // TODO: this is buggy right now, add another elt to start indices.
            println!("WARNING: unexpected exact multiple of 2^log_stacking_height");
            return (next_multiple, cols_so_far, 0, table_index);
        }
        let dst_dense_slice = &mut dense_data[offset..next_multiple];
        let dst_col_idx_slice = &mut col_index[offset >> 1..(next_multiple >> 1)];
        let dst_start_idx_slice = &mut start_indices[cols_so_far..cols_so_far + 1 + num_added_cols];

        unsafe {
            backend
                .write_bytes(
                    dst_dense_slice.as_mut_ptr() as *mut u8,
                    0u8,
                    (next_multiple - offset) * size_of::<Felt>(),
                )
                .unwrap();
        }

        fill_buf(
            dst_col_idx_slice.as_mut_ptr(),
            cols_so_far as u32,
            (next_multiple - offset) >> 1,
            max_log_row_count,
            &backend,
        );

        let mut start_idx_vec = vec![(offset >> 1) as u32];

        for i in 0..num_added_cols - 1 {
            start_idx_vec
                .push((offset >> 1) as u32 + ((i + 1) * (1 << (max_log_row_count - 1))) as u32);
        }

        start_idx_vec.push((next_multiple >> 1) as u32);

        let start_idx = Buffer::from(start_idx_vec).copy_into_backend(&backend).await.unwrap();

        unsafe {
            dst_start_idx_slice.copy_from_slice(&start_idx, &backend).unwrap();
        }

        column_heights
            .extend((0..num_added_cols - 1).map(|_| (1 << (max_log_row_count - 1)) as u32));
        column_heights.push((remainder >> 1) as u32);
        cols_so_far += num_added_cols;
        (next_multiple, cols_so_far, next_multiple - offset, table_index)
    }
    .instrument(tracing::trace_span!("final padding"))
    .await;

    result
}

#[instrument(skip_all, level = "debug")]
fn host_preprocessed_tracegen<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    buffer_ptr: usize,
    program: Arc<<A as MachineAir<Felt>>::Program>,
) -> (HostPhaseTracegen<A>, usize) {
    // Split chips based on where we will generate their traces.
    let (device_airs, host_airs): (Vec<_>, Vec<_>) = machine
        .chips()
        .iter()
        .map(|chip| chip.air.clone())
        .partition(|air| air.supports_device_preprocessed_tracegen());

    let mut total_size = 0;
    let mut jobs = Vec::new();
    for air in host_airs.iter() {
        let height = air.preprocessed_num_rows(&program);
        let width = air.preprocessed_width();
        jobs.push((air.clone(), total_size, height, width));
        if let Some(height) = height {
            total_size += height * width;
        }
    }

    // Spawn a rayon task to generate the traces on the CPU.
    // `traces` is a futures Stream that will immediately begin buffering traces.
    let (host_traces_tx, host_traces) = futures::channel::mpsc::unbounded();
    tokio::task::spawn_blocking(move || {
        jobs.into_par_iter().for_each_with(host_traces_tx, |tx, (air, offset, height, width)| {
            let base_ptr = buffer_ptr as *mut MaybeUninit<Felt>;
            if let Some(height) = height {
                let trace_len = height * width;
                let slice: &mut [MaybeUninit<Felt>] =
                    unsafe { std::slice::from_raw_parts_mut(base_ptr.add(offset), trace_len) };
                air.generate_preprocessed_trace_into(&program, slice);
                let start_pointer = unsafe { base_ptr.add(offset) as usize };
                // Since it's unbounded, it will only error if the receiver is disconnected.
                tx.unbounded_send((air.name(), start_pointer, height, width)).unwrap();
            }
        });
        // Make this explicit.
        // If we are the last users of the program, this will expensively drop it.
        drop(program);
    });
    (HostPhaseTracegen { device_airs, host_traces }, total_size)
}

#[instrument(skip_all, level = "debug")]
async fn device_preprocessed_tracegen<A: CudaTracegenAir<Felt>>(
    program: Arc<<A as MachineAir<Felt>>::Program>,
    host_phase_tracegen: HostPhaseTracegen<A>,
    backend: &TaskScope,
) -> BTreeMap<String, Trace<TaskScope>> {
    let HostPhaseTracegen { device_airs, host_traces } = host_phase_tracegen;

    // Stream that, when polled, copies the host traces to the device.
    let copied_host_traces =
        pin!(host_traces.then(|(name, start_pointer, height, width)| async move {
            let inner_name = name.clone();
            let trace_len = height * width;
            let mut storage: Buffer<Felt, TaskScope> =
                Buffer::with_capacity_in(trace_len, backend.clone());
            let slice =
                unsafe { std::slice::from_raw_parts_mut(start_pointer as *mut Felt, trace_len) };
            storage.extend_from_host_slice(slice).unwrap();
            let dims: Dimensions = [height, width].try_into().unwrap();
            let guts = Tensor { storage, dimensions: dims }.transpose();
            (inner_name, Mle::new(guts))
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
    tokio::task::spawn_blocking(move || drop(program));

    named_traces
}

async fn allocate_and_initialize_traces(
    preprocessed_traces: BTreeMap<String, Trace<TaskScope>>,
    max_trace_size: usize,
    log_stacking_height: u32,
    max_log_row_count: u32,
    backend: &TaskScope,
) -> JaggedTraceMle<Felt, TaskScope> {
    let total_bytes = max_trace_size * std::mem::size_of::<Felt>()
        + (max_trace_size >> 1) * std::mem::size_of::<u32>()
        + MAX_COLS_PER_TRACE * std::mem::size_of::<u32>();

    let total_gb = total_bytes as f64 / (1 << 30) as f64;
    tracing::debug!("Allocating {:?} GB of traces", total_gb);
    let mut dense_data: Buffer<Felt, TaskScope> =
        Buffer::with_capacity_in(max_trace_size, backend.clone());
    let mut col_index: Buffer<u32, TaskScope> =
        Buffer::with_capacity_in(max_trace_size >> 1, backend.clone());

    let mut start_indices: Buffer<u32, TaskScope> =
        Buffer::with_capacity_in(MAX_COLS_PER_TRACE, backend.clone());

    let mut column_heights: Vec<u32> = Vec::with_capacity(MAX_COLS_PER_TRACE);

    unsafe {
        dense_data.assume_init();
        col_index.assume_init();
        start_indices.assume_init();
    }

    // Put them in right places. Todo: parallelize.
    let (preprocessed_offset, preprocessed_cols, preprocessed_padding, preprocessed_table_index) =
        generate_jagged_traces(
            &mut dense_data,
            &mut col_index,
            &mut start_indices,
            &mut column_heights,
            preprocessed_traces,
            0,
            0,
            log_stacking_height,
            max_log_row_count,
        )
        .await;

    let trace_dense_data: TraceDenseData<Felt, TaskScope> = TraceDenseData {
        dense: dense_data,
        preprocessed_offset,
        preprocessed_cols,
        preprocessed_table_index,
        main_table_index: BTreeMap::new(),
        preprocessed_padding,
        main_padding: 0,
    };

    JaggedTraceMle(JaggedMle {
        dense_data: trace_dense_data,
        col_index,
        start_indices,
        column_heights,
    })
}

fn update_global_dependencies(
    dense_data: &mut Buffer<Felt, TaskScope>,
    main_table_index: &BTreeMap<String, TraceOffset>,
) {
    let global_trace_offset = main_table_index.get("Global").unwrap();
    let global_dependencies_offset = global_trace_offset.dense_offset.start
        + global_trace_offset.poly_size * GLOBAL_OFFSET_POS_COPY;
    let len = global_trace_offset.poly_size;
    let byte_trace_offset = main_table_index.get("Byte").unwrap().dense_offset.start;

    let backend = dense_data.backend().clone();
    let mut cnt_buf = Tensor::<u32, TaskScope>::zeros_in([320], backend.clone());

    count_and_add(
        cnt_buf.as_mut_ptr(),
        unsafe { dense_data.as_ptr().add(global_dependencies_offset) },
        len,
        &backend,
    );

    sum_to_trace(
        unsafe { dense_data.as_mut_ptr().add(byte_trace_offset) },
        cnt_buf.as_ptr(),
        &backend,
    );
}

async fn copy_main_jagged_traces(
    traces: BTreeMap<String, Trace<TaskScope>>,
    jagged_traces: &mut JaggedTraceMle<Felt, TaskScope>,
    log_stacking_height: u32,
    max_log_row_count: u32,
    global_dependencies_opt: bool,
) {
    // At this point, all traces are on device. Now we need to copy them into the Jagged MLE struct.
    let JaggedMle { dense_data: trace_dense_data, col_index, start_indices, column_heights } =
        &mut **jagged_traces;

    let TraceDenseData {
        dense: dense_data,
        preprocessed_offset,
        preprocessed_cols,
        main_table_index,
        main_padding,
        ..
    } = trace_dense_data;

    unsafe {
        dense_data.set_len(dense_data.capacity());
        col_index.set_len(col_index.capacity());
        start_indices.set_len(start_indices.capacity());
    }

    // Put them in right places. Todo: parallelize.
    let (final_offset, final_cols, final_main_padding, new_main_table_index) =
        generate_jagged_traces(
            dense_data,
            col_index,
            start_indices,
            column_heights,
            traces,
            *preprocessed_offset,
            *preprocessed_cols,
            log_stacking_height,
            max_log_row_count,
        )
        .await;

    *main_table_index = new_main_table_index;
    *main_padding = final_main_padding;

    if main_table_index.contains_key("Global") && global_dependencies_opt {
        update_global_dependencies(dense_data, main_table_index);
    }

    // Shrink the len of the dense data to match the actual size.
    unsafe {
        dense_data.set_len(final_offset);
        col_index.set_len(final_offset >> 1);
        start_indices.set_len(final_cols + 1);
    }
}

/// Corresponds to `generate_preprocessed_traces`.
#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
pub async fn setup_tracegen<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    program: Arc<<A as MachineAir<Felt>>::Program>,
    buffer_ptr: usize,
    max_trace_size: usize,
    log_stacking_height: u32,
    max_log_row_count: u32,
    prover_permit: ProverSemaphore,
    backend: &TaskScope,
) -> (JaggedTraceMle<Felt, TaskScope>, ProverPermit) {
    // Generate traces on host.
    let (host_phase_tracegen, _) =
        host_preprocessed_tracegen(machine, buffer_ptr, Arc::clone(&program));

    let permit = prover_permit.acquire().await.unwrap();
    // - Copying host traces to the device.
    // - Generating traces on the device.
    let preprocessed_traces =
        device_preprocessed_tracegen(program, host_phase_tracegen, backend).await;

    let jagged_traces = allocate_and_initialize_traces(
        preprocessed_traces,
        max_trace_size,
        log_stacking_height,
        max_log_row_count,
        backend,
    )
    .await;

    (jagged_traces, permit)
}

#[allow(clippy::too_many_arguments)]
pub async fn setup_tracegen_permit<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    program: Arc<<A as MachineAir<Felt>>::Program>,
    buffer: &Worker<PinnedBuffer<Felt>>,
    max_trace_size: usize,
    log_stacking_height: u32,
    max_log_row_count: u32,
    prover_permit: ProverSemaphore,
    backend: &TaskScope,
) -> (JaggedTraceMle<Felt, TaskScope>, ProverPermit) {
    let (jagged_traces, permit) = setup_tracegen(
        machine,
        program,
        buffer.as_ptr() as usize,
        max_trace_size,
        log_stacking_height,
        max_log_row_count,
        prover_permit,
        backend,
    )
    .await;
    (jagged_traces, permit)
}

/// Returns a tuple of (host phase tracegen, shape info).
#[instrument(skip_all, level = "debug")]
fn host_main_tracegen<A>(
    machine: &Machine<Felt, A>,
    buffer_ptr: usize,
    start_idx: usize,
    record: Arc<<A as MachineAir<Felt>>::Record>,
) -> (HostPhaseTracegen<A>, HostPhaseShapeInfo<A>)
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

    let mut total_size = start_idx;
    let mut jobs = Vec::new();
    for air in host_airs.iter() {
        jobs.push((air.clone(), total_size));
        total_size += air.num_rows(&record).unwrap() * air.width();
    }

    let outer_span = tracing::Span::current();

    // Spawn a blocking task to generate the traces on the CPU.
    // `host_traces` is a futures Stream that will immediately begin buffering traces.
    let (host_traces_tx, host_traces) = futures::channel::mpsc::unbounded();
    tokio::task::spawn_blocking(move || {
        {
            jobs.into_par_iter().for_each_with(host_traces_tx, |tx, (air, offset)| {
                tracing::trace_span!(parent: &outer_span, "chip host main tracegen", chip = %air.name()).in_scope(
                    || {
                        let base_ptr = buffer_ptr as *mut MaybeUninit<Felt>;
                        let height = air.num_rows(&record).unwrap();
                        let width = air.width();
                        let trace_len = height * width;
                        let slice: &mut [MaybeUninit<Felt>] = unsafe {
                            std::slice::from_raw_parts_mut(base_ptr.add(offset), trace_len)
                        };
                        air.generate_trace_into(&record, &mut A::Record::default(), slice);
                        let start_pointer = unsafe { base_ptr.add(offset) as usize };
                        // Since it's unbounded, it will only error if the receiver is disconnected.
                        tx.unbounded_send((air.name(), start_pointer, height, width)).unwrap();
                    },
                );
            });
            // Make this explicit.
            // If we are the last users of the record, this will expensively drop it.
            drop(record);
        }
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

    let host_phase_shape_info = HostPhaseShapeInfo { traces_by_name: initial_traces, chip_set };

    let host_phase_tracegen = HostPhaseTracegen { device_airs, host_traces };

    (host_phase_tracegen, host_phase_shape_info)
}

/// Puts traces on device. Returns (traces, public values).
#[instrument(skip_all, level = "debug")]
async fn device_main_tracegen<A: CudaTracegenAir<Felt>>(
    host_phase_tracegen: HostPhaseTracegen<A>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    initial_traces: BTreeMap<String, Trace>,
    backend: &TaskScope,
) -> (BTreeMap<String, Trace>, Vec<Felt>) {
    let HostPhaseTracegen { device_airs, host_traces } = host_phase_tracegen;

    let outer_span = tracing::Span::current();
    // Stream that, when polled, copies the host traces to the device.
    let copied_host_traces = pin!(host_traces.then(|(name, start_pointer, height, width)| {
        let inner_name = name.clone();
        let trace_len = height * width;
        let mut storage: Buffer<Felt, TaskScope> =
            Buffer::with_capacity_in(trace_len, backend.clone());
        let slice =
            unsafe { std::slice::from_raw_parts_mut(start_pointer as *mut Felt, trace_len) };
        storage.extend_from_host_slice(slice).unwrap();
        let dims: Dimensions = [height, width].try_into().unwrap();
        let guts = Tensor { storage, dimensions: dims }.transpose();
        async move { (inner_name, Mle::new(guts)) }
    }
    .instrument(
        tracing::trace_span!(parent: &outer_span, "copy host trace to device", chip = %name)
    )));

    // Stream that, when polled, copies events to the device and generates traces.
    let device_traces = device_airs
        .into_iter()
        .map(|air| {
            // We want to borrow the record and move the chip.
            let record = record.as_ref();
            let outer_span = outer_span.clone();
            async move {
                let trace = air
                    .generate_trace_device(record, &mut A::Record::default(), backend)
                    .instrument(tracing::trace_span!(parent: &outer_span, "device chip tracegen", chip = %air.name()))
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
        .instrument(tracing::debug_span!("wait for device traces"))
        .await;

    // All traces are now generated, so the public values are ready.
    // That is, this value will have the correct global cumulative sum.
    let public_values = record.public_values::<Felt>();

    // If we're the last users of the record, expensively drop it in a separate task.
    // TODO: in general, figure out the best way to drop expensive-to-drop things.
    tokio::spawn(async move { drop(record) });

    (all_traces, public_values)
}

/// Corresponds to `generate_main_traces`.
/// Mutates jagged_traces in place, and returns public values.
#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
pub async fn main_tracegen<GC: IopCtx<F = Felt>, A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    jagged_traces: &Mutex<CudaShardProverData<GC, A>>,
    buffer: &Worker<PinnedBuffer<Felt>>,
    log_stacking_height: u32,
    max_log_row_count: u32,
    backend: &TaskScope,
    prover_permit: ProverSemaphore,
    global_dependencies_opt: bool,
) -> (Vec<Felt>, BTreeSet<Chip<Felt, A>>, ProverPermit) {
    // Start generating traces on host.
    let (host_phase_tracegen, host_phase_shape_info) =
        host_main_tracegen(machine, buffer.as_ptr() as usize, 0, record.clone());

    let HostPhaseShapeInfo { traces_by_name: initial_traces, chip_set } = host_phase_shape_info;
    let permit =
        prover_permit.acquire().instrument(tracing::debug_span!("acquire permit")).await.unwrap();
    let mut jagged_traces = jagged_traces.lock().await;

    // Now that the permit is acquired, we can begin the following two tasks:
    // - Copying host traces to the device.
    // - Generating traces on the device.
    let (traces, public_values) =
        device_main_tracegen(host_phase_tracegen, record, initial_traces, backend).await;

    log_chip_stats(machine, &chip_set, &traces);

    copy_main_jagged_traces(
        traces,
        &mut jagged_traces.preprocessed_traces,
        log_stacking_height,
        max_log_row_count,
        global_dependencies_opt,
    )
    .await;

    (public_values, chip_set, permit)
}

#[allow(clippy::too_many_arguments)]
pub async fn main_tracegen_permit<GC: IopCtx<F = Felt>, A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    jagged_traces: &Mutex<CudaShardProverData<GC, A>>,
    buffer: &Worker<PinnedBuffer<Felt>>,
    log_stacking_height: u32,
    max_log_row_count: u32,
    backend: &TaskScope,
    prover_permit: ProverSemaphore,
    global_dependencies_opt: bool,
) -> (Vec<Felt>, BTreeSet<Chip<Felt, A>>, ProverPermit) {
    let (public_values, chip_set, permit) = main_tracegen(
        machine,
        record,
        jagged_traces,
        buffer,
        log_stacking_height,
        max_log_row_count,
        backend,
        prover_permit,
        global_dependencies_opt,
    )
    .await;

    (public_values, chip_set, permit)
}

/// Does tracegen for both preprocessed and main.
///
/// TODO: output a `MainTraceData` (from prover-clean/prover/types.rs)
#[instrument(skip_all, level = "debug")]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub async fn full_tracegen<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    program: Arc<<A as MachineAir<Felt>>::Program>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    buffer: &Worker<PinnedBuffer<Felt>>,
    max_trace_size: usize,
    log_stacking_height: u32,
    max_log_row_count: u32,
    backend: &TaskScope,
    prover_permits: ProverSemaphore,
    global_dependencies_opt: bool,
) -> (Vec<Felt>, JaggedTraceMle<Felt, TaskScope>, BTreeSet<Chip<Felt, A>>, ProverPermit) {
    let (prep_host_phase_tracegen, start_idx) =
        host_preprocessed_tracegen(machine, buffer.as_ptr() as usize, program.clone());

    let (main_host_phase_tracegen, HostPhaseShapeInfo { traces_by_name: initial_traces, chip_set }) =
        host_main_tracegen(machine, buffer.as_ptr() as usize, start_idx, record.clone());

    // Wait for a prover to be available.
    let permit =
        prover_permits.acquire().instrument(tracing::debug_span!("acquire")).await.unwrap();

    // Now that the permit is acquired, we can begin the following two tasks:
    // - Copying host traces to the device.
    // - Generating traces on the device.

    let (preprocessed_traces, (main_traces, public_values)) = join!(
        device_preprocessed_tracegen(program, prep_host_phase_tracegen, backend),
        device_main_tracegen(main_host_phase_tracegen, record.clone(), initial_traces, backend)
    );

    log_chip_stats(machine, &chip_set, &main_traces);

    let mut jagged_mle = allocate_and_initialize_traces(
        preprocessed_traces,
        max_trace_size,
        log_stacking_height,
        max_log_row_count,
        backend,
    )
    .await;

    copy_main_jagged_traces(
        main_traces,
        &mut jagged_mle,
        log_stacking_height,
        max_log_row_count,
        global_dependencies_opt,
    )
    .await;

    (public_values, jagged_mle, chip_set, permit)
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub async fn full_tracegen_permit<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    program: Arc<<A as MachineAir<Felt>>::Program>,
    record: Arc<<A as MachineAir<Felt>>::Record>,
    buffer: &Worker<PinnedBuffer<Felt>>,
    max_trace_size: usize,
    log_stacking_height: u32,
    max_log_row_count: u32,
    backend: &TaskScope,
    prover_permits: ProverSemaphore,
    global_dependencies_opt: bool,
) -> (Vec<Felt>, JaggedTraceMle<Felt, TaskScope>, BTreeSet<Chip<Felt, A>>, ProverPermit) {
    let (public_values, jagged_mle, chip_set, permit) = full_tracegen(
        machine,
        program,
        record,
        buffer,
        max_trace_size,
        log_stacking_height,
        max_log_row_count,
        backend,
        prover_permits,
        global_dependencies_opt,
    )
    .await;
    (public_values, jagged_mle, chip_set, permit)
}

fn log_chip_stats<A: CudaTracegenAir<Felt>>(
    machine: &Machine<Felt, A>,
    chip_set: &BTreeSet<Chip<Felt, A>>,
    traces: &BTreeMap<String, Trace<TaskScope>>,
) {
    let mut total_number_of_cells = 0;
    tracing::info!("Proving shard");
    for (chip, trace) in machine.smallest_cluster(chip_set).unwrap().iter().zip_eq(traces.values())
    {
        let height = trace.num_real_entries();
        let stats = ChipStatistics::new(chip, height);
        tracing::info!("{}", stats);
        total_number_of_cells += stats.total_number_of_cells();
    }

    tracing::info!(
        "Total number of cells: {}, number of variables: {}",
        total_number_of_cells,
        total_number_of_cells.next_power_of_two().ilog2(),
    );
}

#[cfg(test)]
mod tests {
    use slop_algebra::PrimeField32;
    use slop_futures::queue::WorkerQueue;
    use slop_tensor::Tensor;
    use std::sync::Arc;

    use csl_cuda::{run_in_place, sys::prover_clean::jagged_eval_kernel_chunked_felt, ToDevice};
    use csl_cuda::{PinnedBuffer, TaskScope};
    use csl_tracegen::CudaTraceGenerator;
    use cslpc_utils::{Ext, Felt};
    use cslpc_zerocheck::primitives::evaluate_jagged_mle_chunked;
    use serial_test::serial;
    use slop_algebra::AbstractField;
    use slop_alloc::{Buffer, CopyIntoBackend, IntoHost};
    use slop_multilinear::{Mle, Point};
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};

    use crate::{
        count_and_add, fill_buf, full_tracegen,
        test_utils::tracegen_setup::{self, CORE_MAX_LOG_ROW_COUNT, LOG_STACKING_HEIGHT},
        CORE_MAX_TRACE_SIZE,
    };

    use rand::SeedableRng;
    use rand::{rngs::StdRng, Rng};

    /// Takes a pre-generated proof and vk, and generates traces for the shrink program.
    /// Then, asserts that the jagged traces generated are the same as the traces in the old format.
    #[tokio::test]
    #[serial]
    async fn test_jagged_tracegen() {
        let (machine, record, program) = tracegen_setup::setup().await;

        let mut rng = StdRng::seed_from_u64(4);
        run_in_place(|scope| async move {
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

            println!("warmup traces generated");

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
            for trace in old_traces.preprocessed_traces.values() {
                assert_eq!(trace.num_variables(), CORE_MAX_LOG_ROW_COUNT);

                let trace = trace.eval_at(&z_row).await;

                num_cols += trace.num_polynomials();
                let tensor = trace.into_evaluations();

                let evals_host = tensor.into_host().await.unwrap();
                all_evals_host.extend_from_slice(evals_host.as_buffer());
            }

            // Add zero evaluation for preprocessed padding to next multiple of 2^log stacking height.
            num_cols += 1;
            all_evals_host.extend_from_slice(&[Ext::zero()]);

            for trace in old_traces.main_trace_data.traces.values() {
                assert_eq!(trace.num_variables(), CORE_MAX_LOG_ROW_COUNT);

                let trace = trace.eval_at(&z_row).await;

                num_cols += trace.num_polynomials();
                let tensor = trace.into_evaluations();

                let evals_host = tensor.into_host().await.unwrap();
                all_evals_host.extend_from_slice(evals_host.as_buffer());
            }

            num_cols += 1;
            all_evals_host.extend_from_slice(&[Ext::zero()]);

            // Evaluate `all_evals_host` as an MLE at z_col.
            let all_evals_mle = Mle::from_buffer(all_evals_host.into());
            let num_col_variables = num_cols.next_power_of_two().ilog2();
            let z_col: Point<Ext, _> = Point::rand(&mut rng, num_col_variables);
            let old_tracegen_eval = all_evals_mle.eval_at(&z_col).await.evaluations().as_slice()[0];

            scope.synchronize().await.unwrap();
            drop(old_traces.main_trace_data.permit);
            let now = std::time::Instant::now();

            let capacity = CORE_MAX_TRACE_SIZE as usize;
            let buffer = PinnedBuffer::<Felt>::with_capacity(capacity);
            let queue = Arc::new(WorkerQueue::new(vec![buffer]));
            let buffer = queue.pop().await.unwrap();

            // Do tracegen with the new setup.
            let (_public_values, jagged_trace_data, _chip_set_, _permit) = full_tracegen(
                &machine,
                program.clone(),
                record,
                &buffer,
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT,
                &scope,
                semaphore.clone(),
                false,
            )
            .await;

            scope.synchronize().await.unwrap();
            println!("new traces generated in {:?}", now.elapsed());

            let num_dense_cols = jagged_trace_data.start_indices.len() - 1;
            println!("num dense cols: {}", num_dense_cols);

            let z_row_device = z_row.to_device_in(&scope).await.unwrap();
            let z_col_device = z_col.to_device_in(&scope).await.unwrap();

            let total_len = jagged_trace_data.dense_data.dense.len() / 2;
            println!("total len: {}", total_len);
            let zerocheck_eval = evaluate_jagged_mle_chunked(
                jagged_trace_data,
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

    #[tokio::test]
    async fn test_fill_buf() {
        let mut rng = StdRng::seed_from_u64(5);
        let val = rng.gen::<u32>();

        let randoms = vec![val; 1024];

        run_in_place(|scope| async move {
            let copied_buf = Buffer::from(randoms).copy_into_backend(&scope).await.unwrap();

            let mut generated_buf: Buffer<u32, _> = Buffer::with_capacity_in(1024, scope.clone());
            fill_buf(generated_buf.as_mut_ptr(), val, 1024, 11, &scope);

            unsafe {
                generated_buf.set_len(1024);
            }

            let host_copied_buf = copied_buf.into_host().await.unwrap();
            let host_generated_buf = generated_buf.into_host().await.unwrap();

            assert_eq!(host_copied_buf.as_slice(), host_generated_buf.as_slice());
        })
        .await;
    }

    #[tokio::test]
    async fn test_count_and_add() {
        let mut rng = StdRng::seed_from_u64(5);
        let len = 1 << 20;
        let mut randoms = Vec::with_capacity(len * 6);
        for _ in 0..4 * len {
            randoms.push(Felt::from_canonical_u8(rng.gen::<u8>()));
        }
        for _ in 0..len {
            randoms.push(Felt::from_canonical_u8(rng.gen::<u8>() % 64));
        }
        for _ in 0..len {
            randoms.push(Felt::from_canonical_u8(rng.gen::<u8>() % 2));
        }
        let mut cnt = vec![0u32; 320];
        for i in 0..len {
            if randoms[5 * len + i].as_canonical_u32() == 1 {
                cnt[randoms[i].as_canonical_u32() as usize] += 1;
                cnt[randoms[len + i].as_canonical_u32() as usize] += 1;
                cnt[randoms[2 * len + i].as_canonical_u32() as usize] += 1;
                cnt[randoms[3 * len + i].as_canonical_u32() as usize] += 1;
                cnt[randoms[4 * len + i].as_canonical_u32() as usize + 256] += 1;
            }
        }

        run_in_place(|scope| async move {
            let random_buf = Buffer::from(randoms).to_device_in(&scope).await.unwrap();
            scope.synchronize().await.unwrap();

            let t = std::time::Instant::now();
            let mut cnt_buf = Tensor::<u32, TaskScope>::zeros_in([320], scope.clone());
            count_and_add(cnt_buf.as_mut_ptr(), random_buf.as_ptr(), len, &scope);
            let final_cnt_host = cnt_buf.into_host().await.unwrap().into_buffer().to_vec();

            println!("elapsed time for [1 << 20] x [6] elements: {:?}", t.elapsed());

            assert_eq!(cnt, final_cnt_host);
        })
        .await;
    }
}

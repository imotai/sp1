mod recursion;
mod riscv;

use core::future::Future;
use core::pin::pin;
use std::collections::BTreeSet;
use std::{collections::BTreeMap, sync::Arc};

use csl_cuda::TaskScope;
use futures::future::join_all;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use rayon::prelude::*;
use slop_air::BaseAir;
use slop_algebra::Field;
use slop_alloc::mem::CopyError;
use slop_alloc::CopyIntoBackend;
use slop_baby_bear::BabyBear;
use slop_multilinear::{Mle, PaddedMle};
use slop_tensor::TransposeBackend;
use sp1_stark::MachineRecord;
use sp1_stark::{
    air::MachineAir,
    prover::{ShardData, TraceGenerator, Traces},
    Machine,
};
use tokio::sync::{oneshot, Semaphore};

/// We currently only link to BabyBear-specialized trace generation FFI.
pub(crate) type F = BabyBear;

/// A trace generator that is GPU accelerated.
pub struct CudaTraceGenerator<F: Field, A> {
    machine: Machine<F, A>,
    trace_allocator: TaskScope,
}

impl<A: MachineAir<F>> CudaTraceGenerator<F, A> {
    /// Create a new trace generator.
    #[must_use]
    pub fn new_in(machine: Machine<F, A>, trace_allocator: TaskScope) -> Self {
        Self { machine, trace_allocator }
    }
}

impl<F, A> TraceGenerator<F, A, TaskScope> for CudaTraceGenerator<F, A>
where
    F: Field,
    A: CudaTracegenAir<F>,
    TaskScope: TransposeBackend<F>,
{
    fn machine(&self) -> &Machine<F, A> {
        &self.machine
    }

    fn allocator(&self) -> &TaskScope {
        &self.trace_allocator
    }

    async fn generate_preprocessed_traces(
        &self,
        program: Arc<<A as MachineAir<F>>::Program>,
        max_log_row_count: usize,
        prover_permits: Arc<Semaphore>,
    ) -> Traces<F, TaskScope> {
        // This function's contents are copied from the DefaultTraceGenerator.
        let airs = self.machine.chips().iter().map(|chip| chip.air.clone()).collect::<Vec<_>>();
        let (tx, rx) = oneshot::channel();
        // Spawn a rayon task to generate the traces on the CPU.
        slop_futures::rayon::spawn(move || {
            let named_preprocessed_traces = airs
                .par_iter()
                .filter_map(|air| {
                    let name = air.name();
                    let trace = air.generate_preprocessed_trace(&program);
                    trace.map(Mle::from).map(|tr| (name, tr))
                })
                .collect::<BTreeMap<_, _>>();
            tx.send(named_preprocessed_traces).ok().unwrap();
        });

        // Wait for a prover to be available.
        let permit = prover_permits.acquire_owned().await.unwrap();

        // Wait for the traces to be generated and copy them to the target backend.
        // Wait for traces.
        let named_preprocessed_traces = rx.await.unwrap();
        // Wait for the device to be available.
        let named_traces =
            join_all(named_preprocessed_traces.into_iter().map(|(name, trace)| async move {
                let trace = trace.copy_into_backend(&self.trace_allocator).await.unwrap();
                let padded_mle =
                    PaddedMle::padded_with_zeros(Arc::new(trace), max_log_row_count as u32);
                (name, padded_mle)
            }))
            .await
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        drop(permit);
        Traces { named_traces }
    }

    async fn generate_main_traces(
        &self,
        record: <A as MachineAir<F>>::Record,
        max_log_row_count: usize,
        prover_permits: Arc<Semaphore>,
    ) -> ShardData<F, A, TaskScope> {
        // Set of chips we need to generate traces for.
        let chip_set = self
            .machine
            .chips()
            .iter()
            .filter(|air| air.included(&record))
            .cloned()
            .collect::<BTreeSet<_>>();

        // Split chips based on where we will generate their traces.
        let (device_chips, host_chips): (Vec<_>, Vec<_>) =
            chip_set.iter().cloned().partition(|c| c.air.supports_device_tracegen());

        let record = Arc::new(record);

        // Spawn a rayon task to generate the traces on the CPU.
        // `host_traces` is a futures Stream that will immediately begin buffering traces.
        let (host_traces_tx, host_traces) = futures::channel::mpsc::unbounded();
        {
            let record = Arc::clone(&record);
            slop_futures::rayon::spawn(move || {
                host_chips.into_par_iter().for_each_with(host_traces_tx, |tx, chip| {
                    let trace =
                        Mle::from(chip.air.generate_trace(&record, &mut A::Record::default()));
                    // Since it's unbounded, it will only error if the receiver is disconnected.
                    tx.unbounded_send((chip, trace)).unwrap();
                })
            });
        }

        // Get the smallest cluster containing our tracegen chip set.
        let shard_chips = self.machine.smallest_cluster(&chip_set).unwrap().clone();
        // For every AIR in the cluster, make a (virtual) padded trace.
        let padded_traces = shard_chips
            .iter()
            .filter(|chip| !chip_set.contains(chip))
            .map(|chip| {
                let num_polynomials = chip.width();
                (
                    chip.name(),
                    PaddedMle::zeros_in(
                        num_polynomials,
                        max_log_row_count as u32,
                        &self.trace_allocator,
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();

        // Wait for a prover to be available.
        let permit = prover_permits.acquire_owned().await.unwrap();

        // Now that the permit is acquired, we can begin the following two tasks:
        // - Copying host traces to the device.
        // - Generating traces on the device.

        // Stream that, when polled, copies the host traces to the device.
        let copied_host_traces = pin!(host_traces.then(|(chip, trace)| async move {
            (chip.name(), trace.copy_into_backend(&self.trace_allocator).await.unwrap())
        }));
        // Stream that, when polled, copies events to the device and generates traces.
        let device_traces = device_chips
            .into_iter()
            .map(|chip| {
                // We want to borrow the record and move the chip.
                let record = &record;
                async move {
                    let trace = chip
                        .air
                        .generate_trace_device(
                            record,
                            &mut A::Record::default(),
                            &self.trace_allocator,
                        )
                        .await
                        .unwrap();
                    (chip.name(), trace)
                }
            })
            .collect::<FuturesUnordered<_>>();

        let mut all_traces = padded_traces;

        // Combine the host and device trace streams and insert them into `all_traces`.
        futures::stream_select!(copied_host_traces, device_traces)
            .for_each(|(name, trace)| {
                all_traces.insert(
                    name,
                    PaddedMle::padded_with_zeros(Arc::new(trace), max_log_row_count as u32),
                );
                core::future::ready(())
            })
            .await;

        // All traces are now generated, so the public values are ready.
        // That is, this value will have the correct global cumulative sum.
        let public_values = record.public_values::<F>();

        // There should not be any more Arcs to our record floating around.
        // We check this fact and carefully deallocate in a separate task.
        // TODO: in general, figure out the best way to drop expensive-to-drop things.
        rayon::spawn(move || drop(Arc::into_inner(record).unwrap()));

        let traces = Traces { named_traces: all_traces };

        ShardData { traces, public_values, permit, shard_chips }
    }
}

/// An AIR that potentially supports trace generation over the given field.
pub trait CudaTracegenAir<F: Field>: MachineAir<F> {
    /// Whether this AIR supports device trace generation.
    fn supports_device_tracegen(&self) -> bool {
        false
    }

    /// Generate the trace on the device.
    ///
    /// # Panics
    /// Panics if unsupported. See [`CudaTracegenAir::supports_device_tracegen`].
    #[allow(unused_variables)]
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<Mle<F, TaskScope>, CopyError>> + Send {
        #[allow(unreachable_code)]
        core::future::ready(unimplemented!())
    }
}

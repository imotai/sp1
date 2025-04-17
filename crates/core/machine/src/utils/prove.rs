use std::{
    borrow::Borrow,
    collections::BTreeMap,
    fs::File,
    io::{self, Seek, SeekFrom},
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc::{self, Receiver, Sender, UnboundedSender};

use crate::{executor::MachineExecutorBuilder, riscv::RiscvAir};
use thiserror::Error;

use p3_field::PrimeField32;
use sp1_stark::{
    air::PublicValues,
    prover::{
        MachineProverBuilder, MachineProverComponents, MachineProvingKey, ProverSemaphore,
        ShardProver,
    },
    Machine, MachineProof, MachineRecord, ShardProof, ShardVerifier, Word,
};

use crate::{io::SP1Stdin, utils::concurrency::TurnBasedSync};
use sp1_core_executor::{ExecutionState, SP1CoreOpts};

use sp1_core_executor::{
    subproof::NoOpSubproofVerifier, ExecutionError, ExecutionRecord, ExecutionReport, Executor,
    Program, SP1Context,
};

pub fn generate_checkpoints(
    mut runtime: Executor,
    checkpoints_tx: Sender<(usize, File, bool, u64)>,
) -> Result<Vec<u8>, SP1CoreProverError> {
    let mut index = 0;
    loop {
        // Enter the span.
        let span = tracing::debug_span!("batch");
        let _span = span.enter();

        // Execute the runtime until we reach a checkpoint.
        let (checkpoint, _, done) =
            runtime.execute_state(false).map_err(SP1CoreProverError::ExecutionError)?;

        // Save the checkpoint to a temp file.
        let mut checkpoint_file = tempfile::tempfile().map_err(SP1CoreProverError::IoError)?;
        checkpoint.save(&mut checkpoint_file).map_err(SP1CoreProverError::IoError)?;

        // Send the checkpoint.
        checkpoints_tx
            .blocking_send((index, checkpoint_file, done, runtime.state.global_clk))
            .unwrap();

        // If we've reached the final checkpoint, break out of the loop.
        if done {
            break Ok(runtime.state.public_values_stream);
        }

        // Update the index.
        index += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_records<F: PrimeField32>(
    machine: &Machine<F, RiscvAir<F>>,
    program: Arc<Program>,
    record_gen_sync: Arc<TurnBasedSync>,
    checkpoints_rx: Arc<Mutex<Receiver<(usize, File, bool, u64)>>>,
    records_tx: Sender<ExecutionRecord>,
    state: Arc<Mutex<PublicValues<u32, u32>>>,
    deferred: Arc<Mutex<ExecutionRecord>>,
    report_aggregate: Arc<Mutex<ExecutionReport>>,
    opts: SP1CoreOpts,
) {
    loop {
        let received = { checkpoints_rx.lock().unwrap().blocking_recv() };
        if let Some((index, mut checkpoint, done, _)) = received {
            let (mut records, report) = tracing::debug_span!("trace checkpoint")
                .in_scope(|| trace_checkpoint(program.clone(), &checkpoint, opts));

            // Trace the checkpoint and reconstruct the execution records.
            *report_aggregate.lock().unwrap() += report;
            checkpoint.seek(SeekFrom::Start(0)).expect("failed to seek to start of tempfile");

            // Wait for our turn to update the state.
            record_gen_sync.wait_for_turn(index);

            // Update the public values & prover state for the shards which contain
            // "cpu events".
            let mut state = state.lock().unwrap();
            for record in records.iter_mut() {
                state.shard += 1;
                state.execution_shard = record.public_values.execution_shard;
                state.next_execution_shard = record.public_values.execution_shard + 1;
                state.start_pc = record.public_values.start_pc;
                state.next_pc = record.public_values.next_pc;
                state.last_timestamp = record.public_values.last_timestamp;
                state.last_timestamp_inv =
                    F::from_canonical_u32(state.last_timestamp).inverse().as_canonical_u32();
                state.committed_value_digest = record.public_values.committed_value_digest;
                state.deferred_proofs_digest = record.public_values.deferred_proofs_digest;
                record.public_values = *state;
            }

            // Defer events that are too expensive to include in every shard.
            let mut deferred = deferred.lock().unwrap();
            for record in records.iter_mut() {
                deferred.append(&mut record.defer());
            }

            // See if any deferred shards are ready to be committed to.
            let mut deferred = deferred.split(done, None, opts.split_opts);
            log::info!("deferred {} records", deferred.len());

            // Update the public values & prover state for the shards which do not
            // contain "cpu events" before committing to them.
            state.execution_shard = state.next_execution_shard;
            for record in deferred.iter_mut() {
                state.shard += 1;
                state.previous_init_addr_word = record.public_values.previous_init_addr_word;
                state.last_init_addr_word = record.public_values.last_init_addr_word;
                state.previous_finalize_addr_word =
                    record.public_values.previous_finalize_addr_word;
                state.last_finalize_addr_word = record.public_values.last_finalize_addr_word;
                state.start_pc = state.next_pc;
                state.last_timestamp = 0;
                state.last_timestamp_inv = 0;
                state.next_execution_shard = state.execution_shard;
                record.public_values = *state;
            }
            records.append(&mut deferred);

            // Generate the dependencies.
            machine.generate_dependencies(&mut records, None);

            // Let another worker update the state.
            record_gen_sync.advance_turn();

            // Send the records to the prover.
            for record in records {
                records_tx.blocking_send(record).unwrap();
            }
        } else {
            break;
        }
    }
}

pub async fn prove_core<F, PC>(
    verifier: ShardVerifier<PC::Config, RiscvAir<F>>,
    prover: Arc<ShardProver<PC>>,
    pk: Arc<MachineProvingKey<PC>>,
    program: Arc<Program>,
    stdin: SP1Stdin,
    opts: SP1CoreOpts,
    context: SP1Context<'static>,
) -> Result<(MachineProof<PC::Config>, u64), SP1CoreProverError>
where
    PC: MachineProverComponents<F = F, Air = RiscvAir<F>, Record = ExecutionRecord>,
    F: PrimeField32,
{
    let (proof_tx, mut proof_rx) = tokio::sync::mpsc::unbounded_channel();

    let (_, cycles) =
        prove_core_stream(verifier, prover, pk, program, stdin, opts, context, proof_tx)
            .await
            .unwrap();

    let mut shard_proofs = BTreeMap::new();
    while let Some(proof) = proof_rx.recv().await {
        let public_values: &PublicValues<Word<F>, F> = proof.public_values.as_slice().borrow();
        shard_proofs.insert(public_values.shard, proof);
    }
    let shard_proofs = shard_proofs.into_values().collect();
    let proof = MachineProof { shard_proofs };

    Ok((proof, cycles))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn prove_core_stream<F, PC>(
    // TODO: clean this up
    verifier: ShardVerifier<PC::Config, RiscvAir<F>>,
    prover: Arc<ShardProver<PC>>,
    pk: Arc<MachineProvingKey<PC>>,
    program: Arc<Program>,
    stdin: SP1Stdin,
    opts: SP1CoreOpts,
    context: SP1Context<'static>,
    proof_tx: UnboundedSender<ShardProof<PC::Config>>,
) -> Result<(Vec<u8>, u64), SP1CoreProverError>
where
    PC: MachineProverComponents<F = F, Air = RiscvAir<F>, Record = ExecutionRecord>,
    F: PrimeField32,
{
    // TODO: get this from input
    let num_record_workers = 4;
    let num_trace_gen_workers = 4;
    let (records_tx, mut records_rx) = mpsc::channel::<ExecutionRecord>(num_record_workers);

    let machine_executor = MachineExecutorBuilder::<F>::new(opts, num_record_workers).build();

    let prover_permits = ProverSemaphore::new(opts.shard_batch_size);
    let prover = MachineProverBuilder::<PC>::new(verifier, vec![prover_permits], vec![prover])
        .num_workers(num_trace_gen_workers)
        .build();

    let prover_handle = tokio::spawn(async move {
        let mut handles = Vec::new();
        while let Some(record) = records_rx.recv().await {
            let handle = prover.prove_shard(pk.clone(), record);
            handles.push(handle);
        }
        for handle in handles {
            let proof = handle.await.unwrap();
            proof_tx.send(proof).unwrap();
        }
    });

    // Run the machine executor.
    let output = machine_executor.execute(program, stdin, context, records_tx).await.unwrap();

    // Wait for the prover to finish.
    prover_handle.await.unwrap();

    let pv_stream = output.public_value_stream;
    let cycles = output.cycles;

    Ok((pv_stream, cycles))
}

pub fn trace_checkpoint(
    program: Arc<Program>,
    file: &File,
    opts: SP1CoreOpts,
) -> (Vec<ExecutionRecord>, ExecutionReport) {
    let noop = NoOpSubproofVerifier;

    let mut reader = std::io::BufReader::new(file);
    let state: ExecutionState =
        bincode::deserialize_from(&mut reader).expect("failed to deserialize state");
    let mut runtime = Executor::recover(program, state, opts);

    // We already passed the deferred proof verifier when creating checkpoints, so the proofs were
    // already verified. So here we use a noop verifier to not print any warnings.
    runtime.subproof_verifier = Some(Arc::new(noop));

    // Execute from the checkpoint.
    let (records, _) = runtime.execute_record(true).unwrap();

    (records.into_iter().map(|r| *r).collect(), runtime.report)
}

#[derive(Error, Debug)]
pub enum SP1CoreProverError {
    #[error("failed to execute program: {0}")]
    ExecutionError(ExecutionError),
    #[error("io error: {0}")]
    IoError(io::Error),
    #[error("serialization error: {0}")]
    SerializationError(bincode::Error),
}

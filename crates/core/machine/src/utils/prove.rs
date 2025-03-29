use std::{
    borrow::Borrow,
    collections::BTreeMap,
    fs::File,
    io::{self, Seek, SeekFrom},
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc::{self, Receiver, Sender, UnboundedSender};

use crate::riscv::RiscvAir;
use thiserror::Error;

use p3_field::PrimeField32;
use sp1_stark::{
    air::PublicValues,
    prover::{
        MachineProver, MachineProverComponents, MachineProverOpts, MachineProvingKey, ShardProver,
    },
    Machine, MachineProof, MachineRecord, SP1CoreOpts, ShardProof, Word,
};

use crate::{io::SP1Stdin, utils::concurrency::TurnBasedSync};
use sp1_core_executor::ExecutionState;

use sp1_core_executor::{
    subproof::NoOpSubproofVerifier, ExecutionError, ExecutionRecord, ExecutionReport, Executor,
    Program, SP1Context,
};

fn generate_checkpoints(
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
fn generate_records<F: PrimeField32>(
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
                state.start_pc = record.public_values.start_pc;
                state.next_pc = record.public_values.next_pc;
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
            if !done {
                state.execution_shard += 1;
            }
            for record in deferred.iter_mut() {
                state.shard += 1;
                state.previous_init_addr_bits = record.public_values.previous_init_addr_bits;
                state.last_init_addr_bits = record.public_values.last_init_addr_bits;
                state.previous_finalize_addr_bits =
                    record.public_values.previous_finalize_addr_bits;
                state.last_finalize_addr_bits = record.public_values.last_finalize_addr_bits;
                state.start_pc = state.next_pc;
                record.public_values = *state;
            }
            records.append(&mut deferred);

            // Generate the dependencies.
            machine.generate_dependencies(&mut records, &opts, None);

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
    prover: Arc<ShardProver<PC>>,
    pk: Arc<MachineProvingKey<PC>>,
    program: Arc<Program>,
    stdin: &SP1Stdin,
    opts: SP1CoreOpts,
    context: SP1Context<'static>,
    challenger: PC::Challenger,
) -> Result<(MachineProof<PC::Config>, u64), SP1CoreProverError>
where
    PC: MachineProverComponents<F = F, Air = RiscvAir<F>, Record = ExecutionRecord>,
    F: PrimeField32,
{
    let (proof_tx, mut proof_rx) = tokio::sync::mpsc::unbounded_channel();

    let (_, cycles) =
        prove_core_stream(prover, pk, program, stdin, opts, context, proof_tx, challenger)
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
pub async fn prove_core_stream<F, PC>(
    // TODO: clean this up
    prover: Arc<ShardProver<PC>>,
    pk: Arc<MachineProvingKey<PC>>,
    program: Arc<Program>,
    stdin: &SP1Stdin,
    opts: SP1CoreOpts,
    context: SP1Context<'static>,
    proof_tx: UnboundedSender<ShardProof<PC::Config>>,
    challenger: PC::Challenger,
) -> Result<(Vec<u8>, u64), SP1CoreProverError>
where
    PC: MachineProverComponents<F = F, Air = RiscvAir<F>, Record = ExecutionRecord>,
    F: PrimeField32,
{
    // Setup the runtime.
    let mut runtime = Executor::with_context(program.clone(), opts, context);
    runtime.write_vecs(&stdin.buffer);
    for proof in stdin.proofs.iter() {
        let (proof, vk) = proof.clone();
        runtime.write_proof(proof, vk);
    }
    // Setup the machine prover.
    let machine_prover_opts = MachineProverOpts {
        num_prover_workers: opts.shard_batch_size,
        num_trace_workers: opts.trace_gen_workers,
        shard_data_channel_capacity: opts.shard_batch_size,
    };
    let machine = prover.machine().clone();
    let prover = MachineProver::new(machine_prover_opts, &prover);

    // Spawn the checkpoint generator task.
    let (checkpoints_tx, checkpoints_rx) =
        mpsc::channel::<(usize, File, bool, u64)>(opts.checkpoints_channel_capacity);
    let checkpoint_generator_handle =
        tokio::task::spawn_blocking(move || generate_checkpoints(runtime, checkpoints_tx));

    // Spawn the record generator workers.
    let checkpoints_rx = Arc::new(Mutex::new(checkpoints_rx));
    let record_gen_sync = Arc::new(TurnBasedSync::new());
    let (records_tx, records_rx) =
        mpsc::channel::<ExecutionRecord>(opts.records_and_traces_channel_capacity);
    let state = Arc::new(Mutex::new(PublicValues::<u32, u32>::default().reset()));
    let report_aggregate = Arc::new(Mutex::new(ExecutionReport::default()));
    let deferred = Arc::new(Mutex::new(ExecutionRecord::new(program.clone())));
    let mut record_gen_handles = Vec::new();
    for _ in 0..opts.trace_gen_workers {
        let worker_sync = record_gen_sync.clone();
        let checkpoints_rx = checkpoints_rx.clone();
        let records_tx = records_tx.clone();
        let state = Arc::clone(&state);
        let deferred = Arc::clone(&deferred);
        let report_aggregate = report_aggregate.clone();
        let program = program.clone();
        let machine = machine.clone();
        let handle = tokio::task::spawn_blocking(move || {
            generate_records(
                &machine,
                program,
                worker_sync,
                checkpoints_rx,
                records_tx,
                state,
                deferred,
                report_aggregate,
                opts,
            )
        });
        record_gen_handles.push(handle);
    }

    drop(records_tx);
    drop(deferred);

    // Run the prover and wait for all proofs to be sent.
    prover.prove_stream(pk, records_rx, proof_tx, challenger).await.unwrap();

    // Wait for the checkpoint generator to finish.
    let pv_stream: Vec<u8> = checkpoint_generator_handle.await.unwrap()?;

    // Get the cycles from the aggregate report.
    let report_aggregate = report_aggregate.lock().unwrap();
    let cycles = report_aggregate.total_instruction_count();
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

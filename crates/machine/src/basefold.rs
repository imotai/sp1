#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use csl_cuda::TaskScope;
    use csl_tracing::init_tracer;
    use sp1_core_executor::{Executor, Program, SP1Context, Trace};
    use sp1_core_machine::{io::SP1Stdin, riscv::RiscvAir, utils::prove_core};
    use sp1_primitives::io::SP1PublicValues;
    use sp1_stark::{
        BabyBearPoseidon2, MachineProof, MachineVerifier, MachineVerifierError, SP1CoreOpts,
        ShardVerifier,
    };

    const FIBONACCI_LONG_ELF: &[u8] =
        include_bytes!("../programs/fibonacci/riscv32im-succinct-zkvm-elf");
    use tracing::Instrument;

    use crate::{gpu_prover_opts, new_cuda_prover};

    /// The canonical entry point for testing a [`Program`] and [`SP1Stdin`] with a [`MachineProver`].
    pub async fn run_test(
        program: Program,
        inputs: SP1Stdin,
        scope: TaskScope,
    ) -> Result<SP1PublicValues, MachineVerifierError<BabyBearPoseidon2>> {
        let mut runtime = Executor::new(Arc::new(program), SP1CoreOpts::default());
        runtime.write_vecs(&inputs.buffer);
        runtime.run::<Trace>().unwrap();
        let public_values = SP1PublicValues::from(&runtime.state.public_values_stream);

        let _ = run_test_core(runtime, inputs, scope).await?;
        Ok(public_values)
    }

    #[allow(unused_variables)]
    pub async fn run_test_core(
        runtime: Executor<'static>,
        inputs: SP1Stdin,
        scope: TaskScope,
    ) -> Result<MachineProof<BabyBearPoseidon2>, MachineVerifierError<BabyBearPoseidon2>> {
        let log_blowup = 1;
        let log_stacking_height = 21;
        let max_log_row_count = 21;
        let machine = RiscvAir::machine();
        let verifier = ShardVerifier::from_basefold_parameters(
            log_blowup,
            log_stacking_height,
            max_log_row_count,
            machine,
        );
        let prover = new_cuda_prover(verifier.clone(), scope);

        let (pk, vk) = prover
            .setup(runtime.program.clone())
            .instrument(tracing::debug_span!("setup").or_current())
            .await;
        let challenger = verifier.pcs_verifier.challenger();
        let opts = gpu_prover_opts().core_opts;

        let (proof, _) = prove_core(
            Arc::new(prover),
            Arc::new(pk),
            runtime.program.clone(),
            &inputs,
            opts,
            SP1Context::default(),
            challenger,
        )
        .instrument(tracing::debug_span!("prove core"))
        .await
        .unwrap();

        let mut challenger = verifier.pcs_verifier.challenger();
        let machine_verifier = MachineVerifier::new(verifier);
        tracing::debug_span!("verify the proof")
            .in_scope(|| machine_verifier.verify(&vk, &proof, &mut challenger))
            .unwrap();
        Ok(proof)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_long_fibonacci() {
        init_tracer();
        let program = Program::from(FIBONACCI_LONG_ELF).unwrap();
        let stdin = SP1Stdin::new();
        csl_cuda::spawn(|t| async move { run_test(program, stdin, t.clone()).await.unwrap() })
            .await
            .unwrap()
            .await
            .unwrap();
    }
}

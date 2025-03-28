use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier},
    jagged::{
        JaggedPcsProofVariable, RecursiveJaggedConfig, RecursiveJaggedPcsVerifier,
        RecursiveMachineJaggedPcsVerifier,
    },
    witness::Witnessable,
    zerocheck::RecursiveVerifierConstraintFolder,
    BabyBearFriConfigVariable, CircuitConfig,
};
use p3_air::Air;
use slop_algebra::{extension::BinomialExtensionField, TwoAdicField};
use slop_baby_bear::BabyBear;
use slop_commit::Rounds;
use slop_multilinear::{Evaluations, MleEval};
use slop_sumcheck::PartialSumcheckProof;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Felt},
    prelude::Ext,
};
use sp1_recursion_executor::DIGEST_SIZE;
use sp1_stark::{
    air::MachineAir, septic_digest::SepticDigest, Machine, MachineConfig, ShardOpenedValues,
    ShardProof,
};

#[allow(clippy::type_complexity)]
pub struct ShardProofVariable<
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    SC: BabyBearFriConfigVariable<C> + Send + Sync,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    /// The commitments to main traces.
    pub main_commitment: SC::DigestVariable,
    /// The values of the traces at the final random point.
    pub opened_values: ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    /// The evaluation proof.
    pub evaluation_proof: JaggedPcsProofVariable<JC>,
    /// The zerocheck IOP proof.
    pub zerocheck_proof: PartialSumcheckProof<Ext<C::F, C::EF>>,
    /// The public values
    pub public_values: Vec<Felt<C::F>>,
    // TODO: The `LogUp+GKR` IOP proofs.
    // pub gkr_proofs: Vec<LogupGkrProof<Ext<C::F, C::EF>>>,
    pub shard_proof: ShardProof<SC>,
}

pub struct MachineVerifyingKeyVariable<
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    SC: BabyBearFriConfigVariable<C>,
> {
    pub pc_start: Felt<C::F>,
    /// The starting global digest of the program, after incorporating the initial memory.
    pub initial_global_cumulative_sum: SepticDigest<Felt<C::F>>,
    /// The preprocessed commitments.
    pub preprocessed_commit: Option<SC::DigestVariable>,
}
impl<C, SC> MachineVerifyingKeyVariable<C, SC>
where
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    SC: BabyBearFriConfigVariable<C>,
{
    /// Hash the verifying key + prep domains into a single digest.
    /// poseidon2( commit[0..8] || pc_start || initial_global_cumulative_sum || prep_domains[N].{log_n, .size, .shift, .g})
    pub fn hash(&self, builder: &mut Builder<C>) -> SC::DigestVariable
    where
        C::F: TwoAdicField,
        SC::DigestVariable: IntoIterator<Item = Felt<C::F>>,
    {
        // let prep_domains = self.chip_information.iter().map(|(_, domain, _)| domain);
        let num_inputs = DIGEST_SIZE + 1 + 14; //(4 * prep_domains.len());
        let mut inputs = Vec::with_capacity(num_inputs);
        if let Some(commit) = self.preprocessed_commit {
            inputs.extend(commit)
        }
        inputs.push(self.pc_start);
        inputs.extend(self.initial_global_cumulative_sum.0.x.0);
        inputs.extend(self.initial_global_cumulative_sum.0.y.0);
        // for domain in prep_domains {
        //     inputs.push(builder.eval(C::F::from_canonical_usize(domain.log_n)));
        //     let size = 1 << domain.log_n;
        //     inputs.push(builder.eval(C::F::from_canonical_usize(size)));
        //     let g = C::F::two_adic_generator(domain.log_n);
        //     inputs.push(builder.eval(domain.shift));
        //     inputs.push(builder.eval(g));
        // }

        SC::hash(builder, &inputs)
    }
}

/// A verifier for shard proofs.
pub struct StarkVerifier<
    A: MachineAir<C::F>,
    SC: BabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    /// The machine.
    pub machine: Machine<C::F, A>,
    /// The jagged pcs verifier.
    pub pcs_verifier: RecursiveJaggedPcsVerifier<SC, C, JC>,
    pub _phantom: std::marker::PhantomData<(C, SC, A, JC)>,
}

impl<C, SC, A, JC> StarkVerifier<A, SC, C, JC>
where
    A: MachineAir<C::F>,
    SC: BabyBearFriConfigVariable<C> + MachineConfig,
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    JC: RecursiveJaggedConfig<
        F = C::F,
        EF = C::EF,
        Circuit = C,
        Commitment = SC::DigestVariable,
        Challenger = SC::FriChallengerVariable,
        BatchPcsProof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>,
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
{
    pub fn verify_shard(
        &self,
        builder: &mut Builder<C>,
        vk: &MachineVerifyingKeyVariable<C, SC>,
        proof: &ShardProofVariable<C, SC, JC>,
        challenger: &mut SC::FriChallengerVariable,
    ) where
        A: for<'b> Air<RecursiveVerifierConstraintFolder<'b, C>>,
    {
        let ShardProofVariable {
            main_commitment,
            opened_values,
            evaluation_proof,
            zerocheck_proof,
            public_values,
            shard_proof,
        } = proof;

        //// Uncomment this when the GKR verification is finalized.
        // // Observe the public values.
        // for value in public_values[0..self.machine.num_pv_elts()].iter() {
        //     challenger.observe(builder, *value);
        // }
        // // Observe the main commitment.
        // challenger.observe(builder, *main_commitment);

        //// This is a hack until the GKR verification is finalized.
        let gkr_points_variable = shard_proof.testing_data.gkr_points.read(builder);
        let gkr_column_openings_variable = shard_proof
            .gkr_proofs
            .iter()
            .map(|gkr_proof| {
                let (main_openings, preprocessed_openings) = &gkr_proof.column_openings;
                let main_openings_variable = main_openings.read(builder);
                let preprocessed_openings_variable: MleEval<Ext<_, _>> = preprocessed_openings
                    .as_ref()
                    .map(MleEval::to_vec)
                    .unwrap_or_default()
                    .read(builder)
                    .into();
                (main_openings_variable, preprocessed_openings_variable)
            })
            .collect::<Vec<_>>();
        /////// End of hack.

        builder.cycle_tracker_v2_enter("verify-zerocheck");
        self.verify_zerocheck(
            builder,
            challenger,
            opened_values,
            zerocheck_proof,
            &gkr_points_variable,
            &gkr_column_openings_variable,
            public_values,
        );
        builder.cycle_tracker_v2_exit();

        // Verify the opening proof.
        let (preprocessed_openings_for_proof, main_openings_for_proof): (Vec<_>, Vec<_>) = proof
            .opened_values
            .chips
            .iter()
            .map(|opening| (opening.preprocessed.clone(), opening.main.clone()))
            .unzip();

        let preprocessed_openings = preprocessed_openings_for_proof
            .iter()
            .map(|x| x.local.iter().as_slice())
            .collect::<Vec<_>>();

        let main_openings = main_openings_for_proof
            .iter()
            .map(|x| x.local.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        let filtered_preprocessed_openings = preprocessed_openings
            .into_iter()
            .filter(|x| !x.is_empty())
            .map(|x| x.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        let preprocessed_column_count = filtered_preprocessed_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .collect::<Vec<_>>();

        let main_column_count =
            main_openings.iter().map(|table_openings| table_openings.len()).collect::<Vec<_>>();

        let only_has_main_commitment = vk.preprocessed_commit.is_none();

        let (commitments, column_counts, openings) = if only_has_main_commitment {
            (
                vec![*main_commitment],
                vec![main_column_count],
                Rounds { rounds: vec![main_openings] },
            )
        } else {
            (
                vec![vk.preprocessed_commit.unwrap(), *main_commitment],
                vec![preprocessed_column_count, main_column_count],
                Rounds { rounds: vec![filtered_preprocessed_openings, main_openings] },
            )
        };

        let machine_jagged_verifier =
            RecursiveMachineJaggedPcsVerifier::new(&self.pcs_verifier, column_counts);

        builder.cycle_tracker_v2_enter("jagged-verifier");
        machine_jagged_verifier.verify_trusted_evaluations(
            builder,
            &commitments,
            zerocheck_proof.point_and_eval.0.clone(),
            &openings,
            evaluation_proof,
            challenger,
        );
        builder.cycle_tracker_v2_exit();
    }
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, sync::Arc};

    use slop_algebra::extension::BinomialExtensionField;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_jagged::BabyBearPoseidon2;
    use sp1_core_executor::{Program, SP1Context};
    use sp1_core_machine::{
        io::SP1Stdin,
        riscv::RiscvAir,
        utils::{prove_core, setup_logger},
    };
    use sp1_recursion_compiler::{
        circuit::{AsmCompiler, AsmConfig},
        config::InnerConfig,
    };
    use sp1_recursion_machine::test::run_recursion_test_machines;
    use sp1_stark::{prover::CpuProver, SP1CoreOpts, ShardVerifier};

    use crate::{
        basefold::{stacked::RecursiveStackedPcsVerifier, tcs::RecursiveMerkleTreeTcs},
        challenger::DuplexChallengerVariable,
        jagged::{RecursiveJaggedConfigImpl, RecursiveJaggedEvalSumcheckConfig},
        witness::Witnessable,
    };

    use super::*;

    type F = BabyBear;
    type SC = BabyBearPoseidon2;
    type JC = RecursiveJaggedConfigImpl<
        C,
        SC,
        RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >;
    type C = InnerConfig;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type A = RiscvAir<BabyBear>;

    #[tokio::test]
    async fn test_verify_shard() {
        setup_logger();
        let program = Program::from(test_artifacts::FIBONACCI_ELF).unwrap();
        let log_blowup = 1;
        let log_stacking_height = 21;
        let max_log_row_count = 21;
        let machine = RiscvAir::machine();
        let verifier = ShardVerifier::from_basefold_parameters(
            log_blowup,
            log_stacking_height,
            max_log_row_count,
            machine.clone(),
        );
        let prover = CpuProver::new(verifier.clone());

        let (pk, vk) = prover.setup(Arc::new(program.clone())).await;

        let challenger = verifier.pcs_verifier.challenger();

        let (proof, _) = prove_core(
            Arc::new(prover),
            Arc::new(pk),
            Arc::new(program.clone()),
            &SP1Stdin::new(),
            SP1CoreOpts::default(),
            SP1Context::default(),
            challenger,
        )
        .await
        .unwrap();

        let shard_proof = proof.shard_proofs[0].clone();
        let challenger_state = shard_proof.testing_data.challenger_state.clone();

        let mut builder = Builder::<C>::default();

        //// This is a hack until the GKR verification is finalized.
        let mut challenger_variable =
            DuplexChallengerVariable::from_challenger(&mut builder, &challenger_state);
        /////// End of hack.

        let vk_variable = vk.read(&mut builder);
        let shard_proof_variable = shard_proof.read(&mut builder);

        let verifier = BasefoldVerifier::<Poseidon2BabyBear16BasefoldConfig>::new(log_blowup);
        let recursive_verifier = RecursiveBasefoldVerifier::<RecursiveBasefoldConfigImpl<C, SC>> {
            fri_config: verifier.fri_config,
            tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
        };
        let recursive_verifier =
            RecursiveStackedPcsVerifier::new(recursive_verifier, log_stacking_height);

        let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
            SC,
            C,
            RecursiveJaggedConfigImpl<
                C,
                SC,
                RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
            >,
        > {
            stacked_pcs_verifier: recursive_verifier,
            max_log_row_count,
            jagged_evaluator: RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(PhantomData),
        };

        let stark_verifier = StarkVerifier::<A, SC, C, JC> {
            machine,
            pcs_verifier: recursive_jagged_verifier,
            _phantom: std::marker::PhantomData,
        };

        builder.cycle_tracker_v2_enter("verify-shard");
        stark_verifier.verify_shard(
            &mut builder,
            &vk_variable,
            &shard_proof_variable,
            &mut challenger_variable,
        );
        builder.cycle_tracker_v2_exit();

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = compiler.compile_inner(block).validate().unwrap();

        let mut witness_stream = Vec::new();
        Witnessable::<AsmConfig<F, EF>>::write(&vk, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(&shard_proof, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(
            &shard_proof.testing_data.gkr_points,
            &mut witness_stream,
        );
        shard_proof.gkr_proofs.iter().for_each(|gkr_proof| {
            let (main_openings, preprocessed_openings) = &gkr_proof.column_openings;
            Witnessable::<AsmConfig<F, EF>>::write(main_openings, &mut witness_stream);
            let preprocessed_openings_unwrapped: MleEval<_> =
                preprocessed_openings.as_ref().map(MleEval::to_vec).unwrap_or_default().into();
            Witnessable::<AsmConfig<F, EF>>::write(
                &preprocessed_openings_unwrapped,
                &mut witness_stream,
            );
        });

        run_recursion_test_machines(program.clone(), witness_stream).await;
    }
}

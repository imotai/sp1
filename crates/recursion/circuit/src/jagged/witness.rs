use slop_jagged::{
    JaggedConfig, JaggedEvalConfig, JaggedLittlePolynomialVerifierParams, JaggedPcsProof,
    JaggedSumcheckEvalProof,
};
use sp1_recursion_compiler::ir::{Builder, Ext};

use crate::{
    witness::{WitnessWriter, Witnessable},
    AsRecursive, CircuitConfig,
};

use super::verifier::{JaggedPcsProofVariable, RecursiveJaggedConfig};

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for JaggedSumcheckEvalProof<T> {
    type WitnessVariable = JaggedSumcheckEvalProof<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        JaggedSumcheckEvalProof {
            branching_program_evals: self
                .branching_program_evals
                .iter()
                .map(|x| x.read(builder))
                .collect(),
            partial_sumcheck_proof: self.partial_sumcheck_proof.read(builder),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in &self.branching_program_evals {
            x.write(witness);
        }
        self.partial_sumcheck_proof.write(witness);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C>
    for JaggedLittlePolynomialVerifierParams<T>
{
    type WitnessVariable = JaggedLittlePolynomialVerifierParams<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        JaggedLittlePolynomialVerifierParams {
            col_prefix_sums: self
                .col_prefix_sums
                .iter()
                .map(|x| (*x).read(builder))
                .collect::<Vec<_>>(),
            next_col_prefix_sums: self
                .next_col_prefix_sums
                .iter()
                .map(|x| (*x).read(builder))
                .collect::<Vec<_>>(),
            max_log_row_count: self.max_log_row_count,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in &self.col_prefix_sums {
            x.write(witness);
        }
        for x in &self.next_col_prefix_sums {
            x.write(witness);
        }
    }
}

impl<C, JC, RecursiveStackedPcsProof, RecursiveJaggedEvalProof> Witnessable<C>
    for JaggedPcsProof<JC>
where
    C: CircuitConfig,
    JC: JaggedConfig<
            F = C::F,
            EF = C::EF,
            BatchPcsProof: Witnessable<C, WitnessVariable = RecursiveStackedPcsProof>,
        > + AsRecursive<C>,
    <<JC as JaggedConfig>::JaggedEvaluator as JaggedEvalConfig<
        C::EF,
        <JC as JaggedConfig>::Challenger,
    >>::JaggedEvalProof: Witnessable<C, WitnessVariable = RecursiveJaggedEvalProof>,
    JC::Recursive: RecursiveJaggedConfig<
        F = C::F,
        EF = C::EF,
        Circuit = C,
        BatchPcsProof = RecursiveStackedPcsProof,
        JaggedEvalProof = RecursiveJaggedEvalProof,
    >,
    C::EF: Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
{
    type WitnessVariable = JaggedPcsProofVariable<JC::Recursive>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        JaggedPcsProofVariable {
            stacked_pcs_proof: self.stacked_pcs_proof.read(builder),
            sumcheck_proof: self.sumcheck_proof.read(builder),
            jagged_eval_proof: self.jagged_eval_proof.read(builder),
            params: self.params.read(builder),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.stacked_pcs_proof.write(witness);
        self.sumcheck_proof.write(witness);
        self.jagged_eval_proof.write(witness);
        self.params.write(witness);
    }
}

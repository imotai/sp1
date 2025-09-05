use slop_challenger::IopCtx;
use slop_jagged::{
    JaggedConfig, JaggedLittlePolynomialVerifierParams, JaggedPcsProof, JaggedSumcheckEvalProof,
};
use slop_multilinear::MultilinearPcsVerifier;
use sp1_primitives::{SP1ExtensionField, SP1Field};
use sp1_recursion_compiler::ir::Builder;

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
            max_log_row_count: self.max_log_row_count,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in &self.col_prefix_sums {
            x.write(witness);
        }
    }
}

impl<GC, C, SC, RecursiveStackedPcsProof> Witnessable<C> for JaggedPcsProof<GC, SC>
where
    GC: IopCtx<F = SP1Field, EF = SP1ExtensionField>,
    C: CircuitConfig,
    SC: JaggedConfig<GC> + AsRecursive<C>,

    SC::Recursive: RecursiveJaggedConfig<
        F = SP1Field,
        EF = SP1ExtensionField,
        Circuit = C,
        BatchPcsProof = RecursiveStackedPcsProof,
        // JaggedEvalProof = RecursiveJaggedEvalProof,
    >,
    <SC::BatchPcsVerifier as MultilinearPcsVerifier<GC>>::Proof:
        Witnessable<C, WitnessVariable = RecursiveStackedPcsProof>,
{
    type WitnessVariable = JaggedPcsProofVariable<SC::Recursive>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let params = self.params.read(builder);
        let sumcheck_proof = self.sumcheck_proof.read(builder);
        let jagged_eval_proof = self.jagged_eval_proof.read(builder);
        let stacked_pcs_proof = self.stacked_pcs_proof.read(builder);
        let added_columns = self.added_columns.clone();

        JaggedPcsProofVariable {
            stacked_pcs_proof,
            sumcheck_proof,
            jagged_eval_proof,
            params,
            added_columns,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.params.write(witness);
        self.sumcheck_proof.write(witness);
        self.jagged_eval_proof.write(witness);
        self.stacked_pcs_proof.write(witness);
    }
}

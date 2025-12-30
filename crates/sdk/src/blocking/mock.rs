//! # Mock Prover
//!
//! A mock prover that can be used for testing.

use sp1_core_machine::io::SP1Stdin;
use sp1_prover::{
    worker::{SP1LightNode, SP1NodeCore},
    Groth16Bn254Proof, PlonkBn254Proof, SP1VerifyingKey,
};

use crate::{
    blocking::{
        block_on,
        cpu::CPUProverError,
        prover::{BaseProveRequest, ProveRequest, Prover},
    },
    SP1Proof, SP1ProofWithPublicValues, SP1ProvingKey, SP1VerificationError, StatusCode,
};

/// A mock prover that can be used for testing.
#[derive(Clone)]
pub struct MockProver {
    inner: SP1LightNode,
}

impl Default for MockProver {
    fn default() -> Self {
        let node = block_on(SP1LightNode::new());
        Self { inner: node }
    }
}

impl MockProver {
    /// Create a new mock prover.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Prover for MockProver {
    type ProvingKey = SP1ProvingKey;

    type Error = CPUProverError;

    type ProveRequest<'a> = MockProveRequest<'a>;

    fn inner(&self) -> &SP1NodeCore {
        self.inner.inner()
    }

    fn prove<'a>(&'a self, pk: &'a Self::ProvingKey, stdin: SP1Stdin) -> Self::ProveRequest<'a> {
        MockProveRequest { base: BaseProveRequest::new(self, pk, stdin) }
    }

    fn setup(&self, elf: sp1_build::Elf) -> Result<Self::ProvingKey, Self::Error> {
        let vk = block_on(self.inner.setup(&elf))?;
        Ok(SP1ProvingKey { vk, elf })
    }

    fn verify(
        &self,
        proof: &SP1ProofWithPublicValues,
        _vkey: &SP1VerifyingKey,
        _status_code: Option<StatusCode>,
    ) -> Result<(), SP1VerificationError> {
        match &proof.proof {
            SP1Proof::Plonk(PlonkBn254Proof { public_inputs: _, .. }) => {
                todo!()
                // verify_plonk_bn254_public_inputs(vkey, &bundle.public_values, public_inputs)
                // .map_err(SP1VerificationError::Plonk)
            }
            SP1Proof::Groth16(Groth16Bn254Proof { public_inputs: _, .. }) => {
                todo!()
                // verify_groth16_bn254_public_inputs(vkey, &bundle.public_values, public_inputs)
                // .map_err(SP1VerificationError::Groth16)
            }
            _ => Ok(()),
        }
    }
}

/// A mock prove request that can be used for testing.
pub struct MockProveRequest<'a> {
    pub(crate) base: BaseProveRequest<'a, MockProver>,
}

impl<'a> ProveRequest<'a, MockProver> for MockProveRequest<'a> {
    fn base(&mut self) -> &mut BaseProveRequest<'a, MockProver> {
        &mut self.base
    }

    fn run(self) -> Result<SP1ProofWithPublicValues, CPUProverError> {
        let BaseProveRequest { prover, pk, mode, stdin, context_builder } = self.base;
        let mut req = prover.execute(pk.elf.clone(), stdin);
        req.context_builder = context_builder;
        let (public_values, _) = req.run()?;
        Ok(SP1ProofWithPublicValues::create_mock_proof(
            &pk.vk,
            public_values,
            mode,
            prover.version(),
        ))
    }
}

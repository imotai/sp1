use serde::{Deserialize, Serialize};
use slop_challenger::IopCtx;
use sp1_hypercube::{verify_merkle_proof, HashableKey, MachineVerifyingKey, MerkleProof};
use sp1_primitives::SP1GlobalContext;

/// The serialized recursion verifying key data for this SP1 version.
const VERIFIER_VK_DATA_BYTES: &[u8] = include_bytes!("../vk-artifacts/verifier_vks.bin");

#[derive(Clone, Serialize, Debug, PartialEq, Eq, Deserialize)]
pub struct VerifierRecursionVks {
    pub root: <SP1GlobalContext as IopCtx>::Digest,
    pub vk_verification: bool,
    pub num_keys: usize,
}

impl Default for VerifierRecursionVks {
    fn default() -> Self {
        bincode::deserialize(VERIFIER_VK_DATA_BYTES).unwrap()
    }
}

impl VerifierRecursionVks {
    pub fn vk_verification(&self) -> bool {
        self.vk_verification
    }

    pub fn root(&self) -> <SP1GlobalContext as IopCtx>::Digest {
        self.root
    }

    pub fn num_keys(&self) -> usize {
        self.num_keys
    }

    pub fn verify(
        &self,
        proof: &MerkleProof<SP1GlobalContext>,
        vk: &MachineVerifyingKey<SP1GlobalContext>,
    ) -> bool {
        if !self.vk_verification {
            return true;
        }
        verify_merkle_proof(proof, vk.hash_koalabear(), self.root).is_ok()
    }
}

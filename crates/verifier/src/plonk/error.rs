use thiserror::Error;

#[derive(Error, Debug)]
pub enum PlonkError {
    #[error("Beyond the modulus")]
    BeyondTheModulus,
    #[error("BSB22 Commitment number mismatch")]
    Bsb22CommitmentMismatch,
    #[error("Challenge already computed")]
    ChallengeAlreadyComputed,
    #[error("Challenge not found")]
    ChallengeNotFound,
    #[error("DST too large")]
    DSTTooLarge,
    #[error("Ell too large")]
    EllTooLarge,
    #[error("Inverse not found")]
    InverseNotFound,
    #[error("Invalid number of digests")]
    InvalidNumberOfDigests,
    #[error("Invalid verifying key")]
    InvalidVerifyingKey,
    #[error("Invalid witness")]
    InvalidWitness,
    #[error("Pairing check failed")]
    PairingCheckFailed,
    #[error("Previous challenge not computed")]
    PreviousChallengeNotComputed,
    #[error("Transcript error")]
    TranscriptError,
    #[error("Plonk vkey hash mismatch")]
    PlonkVkeyHashMismatch,
    #[error("Vkey root mismatch")]
    VkeyRootMismatch,
    #[error("Exit code mismatch")]
    ExitCodeMismatch,
    #[error("General error: {0}")]
    GeneralError(#[from] crate::error::Error),
}

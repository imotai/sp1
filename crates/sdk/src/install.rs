//! # SP1 Install
//!
//! A library for installing the SP1 circuit artifacts.

pub use sp1_prover::build::{
    download_file, groth16_circuit_artifacts_dir, install_circuit_artifacts,
    plonk_circuit_artifacts_dir, try_install_circuit_artifacts, CIRCUIT_ARTIFACTS_URL_BASE,
};

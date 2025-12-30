#![allow(clippy::print_stdout)] // okay to print to stdout: this is a build script

use itertools::Itertools;
use slop_algebra::{AbstractField, PrimeField32};
use slop_bn254::Bn254Fr;
use sp1_hypercube::{MachineVerifyingKey, ShardProof};
use sp1_primitives::{io::sha256_hash, SP1Field, SP1OuterGlobalContext};
use sp1_recursion_circuit::{
    hash::FieldHasherVariable,
    machine::{SP1ShapedWitnessValues, SP1WrapVerifier},
    utils::{koalabear_bytes_to_bn254, koalabears_proof_nonce_to_bn254, koalabears_to_bn254},
};
use sp1_recursion_compiler::{
    config::OuterConfig,
    constraints::{Constraint, ConstraintCompiler},
    ir::Builder,
};
use sp1_recursion_executor::RecursionPublicValues;
use sp1_recursion_gnark_ffi::{Groth16Bn254Prover, PlonkBn254Prover};
use std::{borrow::Borrow, path::PathBuf};

pub use sp1_recursion_circuit::witness::{OuterWitness, Witnessable};

use {
    futures::StreamExt,
    indicatif::{ProgressBar, ProgressStyle},
    reqwest::Client,
    std::cmp::min,
    tokio::io::AsyncWriteExt,
    tokio::process::Command,
};

use crate::{
    components::{CpuSP1ProverComponents, SP1ProverComponents},
    utils::words_to_bytes,
    OuterSC, SP1_CIRCUIT_VERSION,
};

/// Tries to build the PLONK artifacts inside the development directory.
pub fn try_build_plonk_bn254_artifacts_dev(
    template_vk: &MachineVerifyingKey<SP1OuterGlobalContext, OuterSC>,
    template_proof: &ShardProof<SP1OuterGlobalContext, OuterSC>,
) -> PathBuf {
    let build_dir = plonk_bn254_artifacts_dev_dir(template_vk);
    if build_dir.exists() {
        tracing::info!("[sp1] plonk bn254 found (build_dir: {})", build_dir.display());
    } else {
        tracing::info!(
            "[sp1] building plonk bn254 artifacts in development mode (build_dir: {})",
            build_dir.display()
        );
        build_plonk_bn254_artifacts(template_vk, template_proof, &build_dir);
    }
    build_dir
}

/// Tries to build the groth16 bn254 artifacts in the current environment.
pub fn try_build_groth16_bn254_artifacts_dev(
    template_vk: &MachineVerifyingKey<SP1OuterGlobalContext, OuterSC>,
    template_proof: &ShardProof<SP1OuterGlobalContext, OuterSC>,
) -> PathBuf {
    let build_dir = groth16_bn254_artifacts_dev_dir(template_vk);
    if build_dir.exists() {
        tracing::info!("[sp1] groth16 bn254 found (build_dir: {})", build_dir.display());
    } else {
        tracing::info!(
            "[sp1] building groth16 bn254 artifacts in development mode (build_dir: {})",
            build_dir.display()
        );
        build_groth16_bn254_artifacts(template_vk, template_proof, &build_dir);
    }
    build_dir
}

/// Gets the directory where the PLONK artifacts are installed in development mode.
pub fn plonk_bn254_artifacts_dev_dir(
    template_vk: &MachineVerifyingKey<SP1OuterGlobalContext, OuterSC>,
) -> PathBuf {
    let serialized_vk = bincode::serialize(template_vk).unwrap();
    let vk_hash_prefix = hex_prefix(sha256_hash(&serialized_vk));
    dirs::home_dir()
        .unwrap()
        .join(".sp1")
        .join("circuits")
        .join(format!("{vk_hash_prefix}-plonk-dev"))
}

/// Gets the directory where the groth16 artifacts are installed in development mode.
pub fn groth16_bn254_artifacts_dev_dir(
    template_vk: &MachineVerifyingKey<SP1OuterGlobalContext, OuterSC>,
) -> PathBuf {
    let serialized_vk = bincode::serialize(template_vk).unwrap();
    let vk_hash_prefix = hex_prefix(sha256_hash(&serialized_vk));
    dirs::home_dir()
        .unwrap()
        .join(".sp1")
        .join("circuits")
        .join(format!("{vk_hash_prefix}-groth16-dev"))
}

fn hex_prefix(input: Vec<u8>) -> String {
    format!("{:016x}", u64::from_be_bytes(input[..8].try_into().unwrap()))
}

/// Build the plonk bn254 artifacts to the given directory for the given verification key and
/// template proof.
pub fn build_plonk_bn254_artifacts(
    template_vk: &MachineVerifyingKey<SP1OuterGlobalContext, OuterSC>,
    template_proof: &ShardProof<SP1OuterGlobalContext, OuterSC>,
    build_dir: impl Into<PathBuf>,
) {
    let build_dir = build_dir.into();
    std::fs::create_dir_all(&build_dir).expect("failed to create build directory");
    let (constraints, witness) = build_constraints_and_witness(template_vk, template_proof);
    PlonkBn254Prover::build(constraints, witness, build_dir);
}

/// Build the groth16 bn254 artifacts to the given directory for the given verification key and
/// template proof.
pub fn build_groth16_bn254_artifacts(
    template_vk: &MachineVerifyingKey<SP1OuterGlobalContext, OuterSC>,
    template_proof: &ShardProof<SP1OuterGlobalContext, OuterSC>,
    build_dir: impl Into<PathBuf>,
) {
    let build_dir = build_dir.into();
    std::fs::create_dir_all(&build_dir).expect("failed to create build directory");
    let (constraints, witness) = build_constraints_and_witness(template_vk, template_proof);
    Groth16Bn254Prover::build(constraints, witness, build_dir);
}

/// Build the verifier constraints and template witness for the circuit.
pub fn build_constraints_and_witness(
    template_vk: &MachineVerifyingKey<SP1OuterGlobalContext, OuterSC>,
    template_proof: &ShardProof<SP1OuterGlobalContext, OuterSC>,
) -> (Vec<Constraint>, OuterWitness<OuterConfig>) {
    tracing::info!("building verifier constraints");
    let template_input = SP1ShapedWitnessValues {
        vks_and_proofs: vec![(template_vk.clone(), template_proof.clone())],
        is_complete: true,
    };
    let constraints =
        tracing::info_span!("wrap circuit").in_scope(|| build_outer_circuit(&template_input));

    let pv: &RecursionPublicValues<SP1Field> = template_proof.public_values.as_slice().borrow();
    let vkey_hash = koalabears_to_bn254(&pv.sp1_vk_digest);
    let committed_values_digest_bytes: [SP1Field; 32] =
        words_to_bytes(&pv.committed_value_digest).try_into().unwrap();
    let committed_values_digest = koalabear_bytes_to_bn254(&committed_values_digest_bytes);
    let exit_code = Bn254Fr::from_canonical_u32(pv.exit_code.as_canonical_u32());
    let vk_root = koalabears_to_bn254(&pv.vk_root);
    let proof_nonce = koalabears_proof_nonce_to_bn254(&pv.proof_nonce);
    tracing::info!("building template witness");
    let mut witness = OuterWitness::default();
    template_input.write(&mut witness);
    witness.write_committed_values_digest(committed_values_digest);
    witness.write_vkey_hash(vkey_hash);
    witness.write_exit_code(exit_code);
    witness.write_vk_root(vk_root);
    witness.write_proof_nonce(proof_nonce);
    (constraints, witness)
}

fn build_outer_circuit(
    template_input: &SP1ShapedWitnessValues<SP1OuterGlobalContext, OuterSC>,
) -> Vec<Constraint> {
    let wrap_verifier = CpuSP1ProverComponents::wrap_verifier();
    let wrap_verifier = wrap_verifier.shard_verifier();
    let recursive_wrap_verifier =
        crate::recursion::recursive_verifier::<_, _, OuterSC, OuterConfig>(wrap_verifier);

    let wrap_span = tracing::debug_span!("build wrap circuit").entered();
    let mut builder = Builder::<OuterConfig>::default();

    // Get the value of the vk.
    let template_vk = template_input.vks_and_proofs.first().unwrap().0.clone();
    // Get an input variable.
    let input = template_input.read(&mut builder);

    // Fix the `wrap_vk` value to be the same as the template `vk`. Since the chip information and
    // the ordering is already a constant, we just need to constrain the commitment and pc_start.

    // Get the vk variable from the input.
    let vk = &input.vks_and_proofs.first().unwrap().0;
    // Get the expected commitment.
    let expected_commitment: [_; 1] = template_vk.preprocessed_commit.into();
    let expected_commitment = expected_commitment.map(|x| builder.eval(x));
    // Constrain `commit` to be the same as the template `vk`.
    SP1OuterGlobalContext::assert_digest_eq(
        &mut builder,
        expected_commitment,
        vk.preprocessed_commit,
    );
    // Constrain `pc_start` to be the same as the template `vk`.
    for (vk_pc, template_vk_pc) in vk.pc_start.iter().zip_eq(template_vk.pc_start.iter()) {
        builder.assert_felt_eq(*vk_pc, *template_vk_pc);
    }
    // Verify the proof.
    SP1WrapVerifier::verify(&mut builder, &recursive_wrap_verifier, input);

    let mut backend = ConstraintCompiler::<OuterConfig>::default();
    let operations = backend.emit(builder.into_operations());
    wrap_span.exit();

    operations
}

/// The base URL for the S3 bucket containing the circuit artifacts.
pub const CIRCUIT_ARTIFACTS_URL_BASE: &str = "https://sp1-circuits.s3-us-east-2.amazonaws.com";

/// Whether use the development mode for the circuit artifacts.
pub(crate) fn use_development_mode() -> bool {
    // TODO: Change this after v6.0.0 binary release
    std::env::var("SP1_DEV").unwrap_or("true".to_string()) == "true"
}

/// The directory where the groth16 circuit artifacts will be stored.
#[must_use]
pub(crate) fn groth16_circuit_artifacts_dir() -> PathBuf {
    std::env::var("SP1_GROTH16_CIRCUIT_PATH")
        .map_or_else(
            |_| dirs::home_dir().unwrap().join(".sp1").join("circuits/groth16"),
            |path| path.parse().unwrap(),
        )
        .join(SP1_CIRCUIT_VERSION)
}

/// The directory where the plonk circuit artifacts will be stored.
#[must_use]
pub(crate) fn plonk_circuit_artifacts_dir() -> PathBuf {
    std::env::var("SP1_PLONK_CIRCUIT_PATH")
        .map_or_else(
            |_| dirs::home_dir().unwrap().join(".sp1").join("circuits/plonk"),
            |path| path.parse().unwrap(),
        )
        .join(SP1_CIRCUIT_VERSION)
}

/// Tries to install the groth16 circuit artifacts if they are not already installed.
#[must_use]
pub(crate) async fn try_install_circuit_artifacts(artifacts_type: &str) -> PathBuf {
    let build_dir = if artifacts_type == "groth16" {
        groth16_circuit_artifacts_dir()
    } else if artifacts_type == "plonk" {
        plonk_circuit_artifacts_dir()
    } else {
        unimplemented!("unsupported artifacts type: {}", artifacts_type);
    };

    if build_dir.exists() {
        eprintln!(
            "[sp1] {} circuit artifacts already seem to exist at {}. if you want to re-download them, delete the directory",
            artifacts_type,
            build_dir.display()
        );
    } else {
        tracing::info!(
            "[sp1] {} circuit artifacts for version {} do not exist at {}. downloading...",
            artifacts_type,
            SP1_CIRCUIT_VERSION,
            build_dir.display()
        );

        install_circuit_artifacts(build_dir.clone(), artifacts_type).await;
    }
    build_dir
}

/// Install the latest circuit artifacts.
///
/// This function will download the latest circuit artifacts from the S3 bucket and extract them
/// to the directory specified by [`groth16_bn254_artifacts_dir()`].
#[allow(clippy::needless_pass_by_value)]
pub(crate) async fn install_circuit_artifacts(build_dir: PathBuf, artifacts_type: &str) {
    // Create the build directory.
    std::fs::create_dir_all(&build_dir).expect("failed to create build directory");

    // Download the artifacts.
    let download_url =
        format!("{CIRCUIT_ARTIFACTS_URL_BASE}/{SP1_CIRCUIT_VERSION}-{artifacts_type}.tar.gz");

    // Create a tempfile with a name to store the tar in.
    let artifacts_tar_gz_file = tempfile::NamedTempFile::new().expect("failed to create tempfile");

    // Get the path of the tempfile.
    let tar_path =
        artifacts_tar_gz_file.path().to_str().expect("A named file should have a path").to_owned();

    // Create a tokio friendly file to write the tarball to.
    let mut file = tokio::fs::File::from_std(artifacts_tar_gz_file.into_file());

    // Download the file.
    let client = Client::builder().build().expect("failed to create reqwest client");
    download_file(&client, &download_url, &mut file).await.expect("failed to download file");

    // Extract the tarball to the build directory.
    let res = Command::new("tar")
        .args(["-Pxzf", &tar_path, "-C", build_dir.to_str().unwrap()])
        .output()
        .await
        .expect("failed to extract tarball");

    if !res.status.success() {
        panic!("[sp1] failed to extract tarball to {:?}", build_dir.to_str().unwrap());
    }

    eprintln!("[sp1] downloaded {} to {:?}", download_url, build_dir.to_str().unwrap());
}

/// Download the file with a progress bar that indicates the progress.
pub(crate) async fn download_file(
    client: &Client,
    url: &str,
    file: &mut (impl tokio::io::AsyncWrite + Unpin),
) -> std::result::Result<(), String> {
    let res = client.get(url).send().await.or(Err(format!("Failed to GET from '{}'", &url)))?;

    let total_size =
        res.content_length().ok_or(format!("Failed to get content length from '{}'", &url))?;

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
        .progress_chars("#>-"));

    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();
    while let Some(item) = stream.next().await {
        let chunk = item.or(Err("Error while downloading file"))?;
        file.write_all(&chunk).await.or(Err("Error while writing to file"))?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }
    pb.finish();

    Ok(())
}

#[cfg(test)]
mod tests {
    use sp1_core_executor::SP1Context;
    use sp1_core_machine::{io::SP1Stdin, utils::setup_logger};
    use sp1_prover_types::network_base_types::ProofMode;

    use crate::{
        verify::WRAP_VK_BYTES,
        worker::{cpu_worker_builder, SP1LocalNodeBuilder},
    };

    #[tokio::test]
    #[ignore = "should be invoked when changing the wrap circuit"]
    async fn set_wrap_vk_and_wrapped_proof() {
        setup_logger();

        let elf = test_artifacts::FIBONACCI_ELF;

        tracing::info!("initializing prover");
        let client = SP1LocalNodeBuilder::from_worker_client_builder(cpu_worker_builder())
            .build()
            .await
            .unwrap();

        tracing::info!("prove compressed");
        let stdin = SP1Stdin::new();
        let compressed_proof = client
            .prove_with_mode(&elf, stdin, SP1Context::default(), ProofMode::Compressed)
            .await
            .unwrap();

        tracing::info!("shrink wrap");
        let wrapped_proof = client.shrink_wrap(&compressed_proof.proof).await.unwrap();
        let wrap_vk = wrapped_proof.vk;
        let wrapped_proof = wrapped_proof.proof;

        let wrap_vk_bytes = bincode::serialize(&wrap_vk).unwrap();
        let wrapped_proof_bytes = bincode::serialize(&wrapped_proof).unwrap();
        std::fs::write("wrap_vk.bin", wrap_vk_bytes).unwrap();
        std::fs::write("wrapped_proof.bin", wrapped_proof_bytes).unwrap();
    }

    #[tokio::test]
    async fn test_wrap_vk() {
        setup_logger();

        tracing::info!("initializing prover");
        let client = SP1LocalNodeBuilder::from_worker_client_builder(cpu_worker_builder())
            .build()
            .await
            .unwrap();

        // Check that the wrap vk is the same as the one included in the binary.
        let client_wrap_vk = client.wrap_vk().clone();
        let expected_wrap_vk = bincode::deserialize(WRAP_VK_BYTES).unwrap();
        assert_eq!(client_wrap_vk, expected_wrap_vk);
    }
}

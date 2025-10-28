use std::fs;
use std::path::PathBuf;

pub const SP1_CIRCUIT_VERSION: &str = include_str!("../../SP1_VERSION");

// NOTE: for the tests to run correctly, the circuits have to be built at the correct directory.
fn main() {
    // Re-run build script if the build.rs itself changes
    println!("cargo:rerun-if-changed=build.rs");

    // Get the directories for artifacts
    let dev_dir = std::env::var("SP1_PLONK_CIRCUIT_PATH")
        .map_or_else(
            |_| dirs::home_dir().unwrap().join(".sp1").join("circuits/plonk"),
            |path| path.parse().unwrap(),
        )
        .join(SP1_CIRCUIT_VERSION);

    // Target directory for vk files in verifier crate
    let target_dir = PathBuf::from("bn254-vk");
    fs::create_dir_all(&target_dir).expect("Failed to create bn254-vk directory");

    // Handle PLONK VK
    let plonk_vk_source = dev_dir.join("plonk_vk.bin");
    let plonk_vk_target = target_dir.join("plonk_vk.bin");

    if plonk_vk_source.exists() {
        println!("cargo:warning=Copying plonk_vk.bin from {}", plonk_vk_source.display());
        fs::copy(&plonk_vk_source, &plonk_vk_target).expect("Failed to copy plonk_vk.bin");
    } else {
        eprintln!(
            "Warning: plonk_vk.bin not found at {}. \
            Please build the artifacts first by running the appropriate build script from sp1-prover.",
            plonk_vk_source.display()
        );
        eprintln!(
            "You can build them using: \
            cargo run --bin build_plonk_bn254 --release -- --build-dir {}",
            dev_dir.display()
        );
        // Don't fail the build, just warn - the existing files in bn254-vk/ will be used
    }

    let dev_dir = std::env::var("SP1_PLONK_CIRCUIT_PATH")
        .map_or_else(
            |_| dirs::home_dir().unwrap().join(".sp1").join("circuits/groth16"),
            |path| path.parse().unwrap(),
        )
        .join(SP1_CIRCUIT_VERSION);

    // Handle Groth16 VK
    let groth16_vk_source = dev_dir.join("groth16_vk.bin");
    let groth16_vk_target = target_dir.join("groth16_vk.bin");

    if groth16_vk_source.exists() {
        println!("cargo:warning=Copying groth16_vk.bin from {}", groth16_vk_source.display());
        fs::copy(&groth16_vk_source, &groth16_vk_target).expect("Failed to copy groth16_vk.bin");
    } else {
        eprintln!(
            "Warning: groth16_vk.bin not found at {}. \
            Please build the artifacts first by running the appropriate build script from sp1-prover.",
            groth16_vk_source.display()
        );
        eprintln!(
            "You can build them using: \
            cargo run --bin build_groth16_bn254 --release -- --build-dir {}",
            dev_dir.display()
        );
        // Don't fail the build, just warn - the existing files in bn254-vk/ will be used
    }

    // Re-run if the source files change
    println!("cargo:rerun-if-changed={}", plonk_vk_source.display());
    println!("cargo:rerun-if-changed={}", groth16_vk_source.display());
}

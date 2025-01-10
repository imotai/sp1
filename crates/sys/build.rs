use std::path::{Path, PathBuf};
use std::{env, fs};

fn add_src_files(build: &mut cc::Build, dir: &Path) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectories
            add_src_files(build, &path)?;

            // Add only .c, .cpp, or .cu files.
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("c")
            || path.extension().and_then(|ext| ext.to_str()) == Some("cpp")
            || path.extension().and_then(|ext| ext.to_str()) == Some("cu")
        {
            build.file(path);
        }
    }
    Ok(())
}

fn add_include_directories(build: &mut cc::Build, dir: &Path) -> std::io::Result<()> {
    let mut found_h_file = false;

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectories.
            add_include_directories(build, &path)?;
            // Allow h, hpp, or cuh files.
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("h")
            || path.extension().and_then(|ext| ext.to_str()) == Some("hpp")
            || path.extension().and_then(|ext| ext.to_str()) == Some("cuh")
        {
            found_h_file = true;
        }
    }

    // If this directory contains at least one header, add it as an include directory.
    if found_h_file {
        build.include(dir);
    }

    Ok(())
}

fn builder() -> cc::Build {
    let mut build = cc::Build::new();

    // Compiler flags.
    build
        .cuda(true)
        .flag("-std=c++20")
        .flag("-default-stream=per-thread")
        .flag("-Xcompiler")
        .flag("-fopenmp")
        .flag("-lnvToolsExt")
        .flag("-arch=sm_89")
        .flag("-ldl")
        .flag("-lnvToolsExt")
        .flag("--expt-relaxed-constexpr");

    build
}

fn main() {
    println!("cargo:rerun-if-changed=../../cuda/");
    println!("cargo:rerun-if-changed=../../sppark/");

    let nvcc = which::which("nvcc").expect("nvcc not found");

    let cuda_version =
        std::process::Command::new(nvcc).arg("--version").output().expect("failed to get version");
    if !cuda_version.status.success() {
        panic!("{:?}", cuda_version);
    }
    let cuda_version = String::from_utf8(cuda_version.stdout).unwrap();
    let x =
        cuda_version.find("release ").expect("can't find \"release X.Y,\" in --version output") + 8;
    let y = cuda_version[x..].find(',').expect("can't parse \"release X.Y,\" in --version output");
    let v = cuda_version[x..x + y].parse::<f32>().unwrap();
    if v < 12.0 {
        panic!("Unsupported CUDA version {} < 12.0", v);
    }

    // The crate directory.
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = crate_dir.join("../../cuda/");
    // Get a new builder with the correct flags.
    let mut build = builder();

    env::set_var("DEP_SPPARK_ROOT", "../../sppark/");
    if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
        let include = crate_dir.join(include);
        build.include(include);
        build.define("SPPARK", None);
        build.define("FEATURE_BABY_BEAR", None);
        build.file("../../sppark/rust/src/lib.cpp").file("../../sppark/util/all_gpus.cpp");
    }

    // let sppark_dir = crate_dir.join("../../sppark/");
    // // Add all the source files.
    // add_src_files(&mut build, &sppark_dir).expect("Failed to find c, cpp, or cu files");
    // // Add all header files.
    // add_include_directories(&mut build, &sppark_dir).expect("Failed to find h, hpp, or cuh files");
    // build.define("SPPARK", None);
    // build.define("FEATURE_BABY_BEAR", None);

    // Add all the source files.
    add_src_files(&mut build, &src_dir).expect("Failed to find c, cpp, or cu files");
    // Add all header files.
    add_include_directories(&mut build, &src_dir).expect("Failed to find h, hpp, or cuh files");

    // let sppark_dir = crate_dir.join("../../sppark/");
    // // Add all the source files.
    // add_src_files(&mut build, &sppark_dir).expect("Failed to find c, cpp, or cu files");
    // // Add all header files.
    // add_include_directories(&mut build, &sppark_dir).expect("Failed to find h, hpp, or cuh files");
    // build.define("SPPARK", None);
    // build.define("FEATURE_BABY_BEAR", None);

    // Compile the library.
    build.compile("sys-cuda");
}

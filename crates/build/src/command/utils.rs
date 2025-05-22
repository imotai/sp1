use anyhow::{Context, Result};
use cargo_metadata::semver;
use std::{
    io::{BufRead, BufReader},
    process::{exit, Command, Stdio},
    thread,
};

use crate::{BuildArgs, BUILD_TARGET};

/// Get the arguments to build the program with the arguments from the [`BuildArgs`] struct.
pub(crate) fn get_program_build_args(args: &BuildArgs) -> Vec<String> {
    let mut build_args = vec![
        "build".to_string(),
        "--release".to_string(),
        "--target".to_string(),
        BUILD_TARGET.to_string(),
    ];

    if args.ignore_rust_version {
        build_args.push("--ignore-rust-version".to_string());
    }

    build_args.push("-Ztrim-paths".to_string());

    for p in &args.packages {
        build_args.push("-p".to_string());
        build_args.push(p.to_string());
    }

    for b in &args.binaries {
        build_args.push("--bin".to_string());
        build_args.push(b.to_string());
    }

    if !args.features.is_empty() {
        build_args.push("--features".to_string());
        build_args.push(args.features.join(","));
    }

    if args.no_default_features {
        build_args.push("--no-default-features".to_string());
    }

    if args.locked {
        build_args.push("--locked".to_string());
    }

    build_args
}

/// Rust flags for compilation of C libraries.
pub(crate) fn get_rust_compiler_flags(args: &BuildArgs, version: &semver::Version) -> String {
    // Note: as of 1.81.0, the `-C passes=loweratomic` flag is deprecated, because of a change to
    // llvm.
    let atomic_lower_pass =
        if version > &semver::Version::new(1, 81, 0) { "lower-atomic" } else { "loweratomic" };

    // Check if rustflags already contains a passes flag
    let mut has_passes = false;
    let mut modified_rustflags = Vec::with_capacity(args.rustflags.len());
    let mut i = 0;

    while i < args.rustflags.len() {
        // Handle the case where passes is specified as two separate arguments: `-C passes=...`
        if i + 1 < args.rustflags.len()
            && args.rustflags[i] == "-C"
            && args.rustflags[i + 1].starts_with("passes=")
        {
            // Found existing passes flag
            let existing_passes = &args.rustflags[i + 1];

            // Check if the atomic pass is already included
            if existing_passes.contains(atomic_lower_pass) {
                modified_rustflags.push("-C".to_string());
                modified_rustflags.push(existing_passes.to_string());
            } else {
                // Append our atomic pass
                let combined_passes = format!("{},{}", existing_passes, atomic_lower_pass);
                modified_rustflags.push("-C".to_string());
                modified_rustflags.push(combined_passes);
            }

            has_passes = true;
            i += 2; // Skip the next item since we've processed it
        }
        // Handle the case where passes is specified as a single argument: `-Cpasses=...`
        else if args.rustflags[i].starts_with("-Cpasses=") {
            // Found existing passes flag
            let existing_passes = &args.rustflags[i];

            // Check if the atomic pass is already included
            if existing_passes.contains(atomic_lower_pass) {
                modified_rustflags.push(existing_passes.to_string());
            } else {
                // Append our atomic pass
                let combined_passes = format!("{},{}", existing_passes, atomic_lower_pass);
                modified_rustflags.push(combined_passes);
            }

            has_passes = true;
            i += 1;
        } else {
            // Copy the flag as is
            modified_rustflags.push(args.rustflags[i].clone());
            i += 1;
        }
    }

    // If no passes flag was found or atomic pass wasn't included, add our atomic pass
    if !has_passes {
        modified_rustflags.push("-C".to_string());
        modified_rustflags.push(format!("passes={}", atomic_lower_pass));
    }

    let rust_flags = ["-C", "link-arg=-Ttext=0x00200800", "-C", "panic=abort"];
    let rust_flags: Vec<_> =
        rust_flags.into_iter().chain(modified_rustflags.iter().map(String::as_str)).collect();

    rust_flags.join("\x1f")
}

/// Execute the command and handle the output depending on the context.
pub(crate) fn execute_command(mut command: Command, docker: bool) -> Result<()> {
    // Add necessary tags for stdout and stderr from the command.
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn command")?;
    let stdout = BufReader::new(child.stdout.take().unwrap());
    let stderr = BufReader::new(child.stderr.take().unwrap());

    // Add prefix to the output of the process depending on the context.
    let msg = match docker {
        true => "[sp1] [docker] ",
        false => "[sp1] ",
    };

    // Pipe stdout and stderr to the parent process with [docker] prefix
    let stdout_handle = thread::spawn(move || {
        stdout.lines().for_each(|line| {
            println!("{} {}", msg, line.unwrap());
        });
    });
    stderr.lines().for_each(|line| {
        eprintln!("{} {}", msg, line.unwrap());
    });
    stdout_handle.join().unwrap();

    // Wait for the child process to finish and check the result.
    let result = child.wait()?;
    if !result.success() {
        // Error message is already printed by cargo.
        exit(result.code().unwrap_or(1))
    }
    Ok(())
}

pub(crate) fn parse_rustc_version(version: &str) -> semver::Version {
    let version_string =
        version.split(" ").nth(1).expect("Can't parse rustc --version stdout").trim();

    semver::Version::parse(version_string).expect("Can't parse rustc --version stdout")
}

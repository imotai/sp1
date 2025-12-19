#![allow(unused)]

use cfg_if::cfg_if;
use std::{env, path::PathBuf, process::Command};

#[allow(deprecated)]
use bindgen::CargoCallbacks;

/// Build the go library, generate Rust bindings for the exposed functions, and link the library.
#[allow(clippy::uninlined_format_args)]
fn main() {
    cfg_if! {
        if #[cfg(feature = "native")] {
            println!("cargo:rerun-if-changed=go");
            // Define the output directory
            let out_dir = env::var("OUT_DIR").unwrap();
            let dest_path = PathBuf::from(&out_dir);
            let lib_name = "sp1gnark";
            let dest = dest_path.join(format!("lib{lib_name}.a"));

            println!("Building Go library at {}", dest.display());

            if cfg!(feature = "groth16-cuda") {
                println!("cargo:rustc-link-search=native=/usr/local/lib");
                println!("cargo:rustc-link-lib=dylib=icicle_device");
                println!("cargo:rustc-link-lib=dylib=icicle_field_bn254");
                println!("cargo:rustc-link-lib=dylib=icicle_curve_bn254");
                println!("cargo:rustc-link-lib=dylib=icicle_field_bls12_377");
                println!("cargo:rustc-link-lib=dylib=icicle_curve_bls12_377");
                println!("cargo:rustc-link-lib=dylib=icicle_field_bls12_381");
                println!("cargo:rustc-link-lib=dylib=icicle_curve_bls12_381");
                println!("cargo:rustc-link-lib=dylib=icicle_field_bw6_761");
                println!("cargo:rustc-link-lib=dylib=icicle_curve_bw6_761");
                // Ideally we would also set the RPATH/RUNPATH using the following
                // println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
                // Unfortunately, this doesn't work. See:
                // https://github.com/rust-lang/cargo/pull/9557#issuecomment-884302305
            }

            let tags = if cfg!(feature = "groth16-cuda") {
                "-tags=debug,icicle"
            } else {
                "-tags=debug"
            };

            let status = Command::new("go")
                .current_dir("go")
                .env("CGO_ENABLED", "1")
                .args([
                    "build",
                    tags,
                    "-o",
                    dest.to_str().unwrap(),
                    "-buildmode=c-archive",
                    ".",
                ])
                .status()
                .expect("Failed to build Go library");
            if !status.success() {
                panic!("Go build failed");
            }

            // Copy go/koalabear.h to OUT_DIR/koalabear.h
            let header_src = PathBuf::from("go/koalabear.h");
            let header_dest = dest_path.join("koalabear.h");
            std::fs::copy(header_src, header_dest).unwrap();

            // Generate bindings using bindgen
            let header_path = dest_path.join(format!("lib{lib_name}.h"));
            let bindings = bindgen::Builder::default()
                .header(header_path.to_str().unwrap())
                .generate()
                .expect("Unable to generate bindings");

            bindings
                .write_to_file(dest_path.join("bindings.rs"))
                .expect("Couldn't write bindings!");

            println!("Go library built");

            // Link the Go library
            println!("cargo:rustc-link-search=native={}", dest_path.display());
            println!("cargo:rustc-link-lib=static={lib_name}");

            // Static linking doesn't really work on macos, so we need to link some system libs
            if cfg!(target_os = "macos") {
                println!("cargo:rustc-link-lib=framework=CoreFoundation");
                println!("cargo:rustc-link-lib=framework=Security");
            }
        }
    }
}

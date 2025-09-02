
# CUSLOP Repository Guide

## Repository Overview
CUSLOP is a high-performance cryptographic proving system with CUDA GPU acceleration. This repository contains:

- **Rust crates** for high-level proving logic and APIs
- **CUDA kernels** for GPU-accelerated cryptographic operations
- **sppark integration** for optimized field arithmetic and NTT operations
- **FFI bindings** via the `csl-sys` crate to bridge Rust and CUDA code

## Key Architecture Components

### CUDA Build System
The project uses a **CMake + Makefile** build system for CUDA compilation:
- **Main Makefile** (`/Makefile`) - handles all CUDA source compilation with incremental builds
- **CMakeLists.txt** (`/crates/sys/CMakeLists.txt`) - bridges cmake crate and Makefile
- **build.rs** (`/crates/sys/build.rs`) - uses cmake crate instead of cc::Build for better incremental compilation

### Directory Structure
- `cuda/` - All CUDA source files (.cu, .cuh) organized by module:
  - `algebra/`, `basefold/`, `challenger/`, `jagged/`, `logup_gkr/`, etc.
  - 38 total CUDA source files across different cryptographic operations
- `sppark/` - External CUDA library for NTT kernels and field arithmetic
- `crates/sys/` - Main FFI crate that compiles all CUDA code into static library
- `target/cuda-build/` - Build artifacts directory for CUDA compilation

### Key Crates
- **csl-sys** - FFI bindings and CUDA compilation (the core integration point)
- **csl-cuda** - High-level Rust wrappers for CUDA operations
- **csl-prover-clean** - Main proving algorithms with GPU acceleration
- **csl-perf** - Performance benchmarking and testing tools
- **cuslop-server** - Server API for the proving system

## CUDA Development Notes

### Compilation Requirements
- **CUDA 12.0+** required (checked at build time)
- **Relocatable device code** (`-rdc=true`) enabled for cross-file device function calls
- **Architecture targeting** supports sm_80, sm_86, sm_89, sm_90, sm_100, sm_120
- **Device linking** step required after compilation for proper CUDA object linking

### Important Build Details
1. **cbindgen headers** must be generated before CUDA compilation starts
2. **Library linking order**: sys-cuda → cudart → cudadevrt → stdc++ → gomp → dl
3. **CUDA path detection** tries multiple common installation locations automatically
4. **Environment variables**:
   - `CUDA_ARCHS` - override default GPU architectures
   - `PROFILE_DEBUG_DATA=true` - enable CUDA debug symbols

### Common Build Issues
- **Device function linking errors**: Ensure `-rdc=true` flag is set
- **Missing CUDA libraries**: Check CUDA installation and library paths
- **cbindgen header not found**: Headers must be generated before compilation
- **C++ symbol errors**: Ensure stdc++ is linked for CUDA C++ code

## Development Workflow

### Build Commands
```bash
cargo build                    # Debug build
cargo build --release         # Optimized build
cargo build --profile lto     # LTO optimized build
```

### Module-Specific Builds (via Make)
```bash
make ntt                       # Build only NTT module
make basefold                  # Build only basefold module  
make clean                     # Clean all build artifacts
```

### Benchmarking
```bash
# Run end-to-end benchmarks
cargo run --release -p csl-perf --bin e2e -- --program fibonacci-20m --stage compress

# Run look-ahead benchmarks
cargo run --release -p csl-prover-clean --bin look_ahead_bench
```

## Performance Considerations
- **Incremental compilation** - Only changed CUDA files are recompiled
- **Parallel Make builds** - Multiple CUDA files compile simultaneously  
- **GPU architecture targeting** - Optimized for modern GPUs (Ampere, Hopper, etc.)
- **Memory pool management** - Custom CUDA memory allocators for efficiency

## Debugging Tips
- Use `PROFILE_DEBUG_DATA=true` to enable CUDA debug symbols
- Check `target/cuda-build/` for compilation artifacts and logs
- CUDA compilation warnings are preserved and displayed during build
- Use `make info` to see current build configuration

## Final touches
Make sure formatting and clippy checks are passing by running:
```bash
cargo +stable fmt --all -- --check
```

and 
```bash
cargo +stable clippy -- -D warnings -A incomplete-features
```

Read the error messages and fix the issues following the suggestions.
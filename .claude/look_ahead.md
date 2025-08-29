# Look Ahead Implementation Summary

## Overview
The look_ahead module implements a Hadamard product computation with GPU acceleration using CUDA kernels. It's part of the prover-clean crate and performs iterative rounds of computation on tensor data.

## Running the benchmark

```bash
cargo run --profile lto -p prover-clean --bin look_ahead_bench
```



## Key Components

### 1. Hadamard Struct
- **Location**: `crates/prover-clean/src/look_ahead.rs`
- **Purpose**: Manages two tensors (p and q) for Hadamard product computation
- **Key Methods**:
  - `new()`: Creates instance from existing tensors
  - `alloc()`: Allocates new tensors with specified size
  - `round()`: Performs one round of computation
  - `num_variables()`: Returns log2 of tensor length

### 2. RoundParams Struct
- **Purpose**: Runtime configuration for the round function (replaced const generics)
- **Fields**:
  - `fix_group`: Group size for fixed operations (valid: 1, 2, 4, 8)
  - `fix_tile`: Tile size for processing (typically 32)
  - `sum_group`: Group size for summation (typically 2)
  - `num_points`: Number of output points (typically 2)
  - `store_restricted`: Whether to store intermediate restricted results

### 3. CUDA Kernel Selection
- **Function**: `look_ahead_round_kernel()`
- **Mechanism**: Runtime selection based on parameter combinations
- **Supported Combinations** (7 total):
  - (1, 32, 2, 2, false) → round_kernel_1_32_2_2_false
  - (2, 32, 2, 2, true) → round_kernel_2_32_2_2_true
  - (2, 32, 2, 2, false) → round_kernel_2_32_2_2_false
  - (4, 32, 2, 2, true) → round_kernel_4_32_2_2_true
  - (4, 32, 2, 2, false) → round_kernel_4_32_2_2_false
  - (8, 32, 2, 2, true) → round_kernel_8_32_2_2_true
  - (8, 32, 2, 2, false) → round_kernel_8_32_2_2_false
- **Error Handling**: Panics on unsupported combinations

## Round Function Operation

### Input/Output
- **Input**: Self (Hadamard with tensors p and q), RoundParams
- **Output**: Tuple of (result tensor, optional restricted Hadamard)

### Processing Steps
1. Calculate reduced height (half of current tensor size)
2. Optionally allocate restricted storage based on `store_restricted`
3. Configure kernel parameters:
   - Block size: 256 threads
   - Stride: 2 (for reduction)
   - Grid dimensions based on tensor size
   - Shared memory size calculation
4. Launch CUDA kernel with appropriate parameters
5. Synchronize and wait for completion
6. Sum results along dimension 1
7. Transfer result to host memory
8. Return result and optional restricted data

### Memory Management
- Uses `TaskScope` for GPU memory management
- Handles null pointers when restricted storage not needed
- Efficient shared memory allocation for kernel execution

## Benchmark Usage Pattern

### Typical Workflow (from look_ahead_bench.rs)
1. **First Round**: Uses special parameters (fix_group=1) for initial processing
2. **Subsequent Rounds**: Uses standard parameters (fix_group=FIX_GROUP)
3. **Iteration**: Continues until `num_variables` rounds complete
4. **Restriction**: Each round potentially produces restricted (halved) data for next round

### Performance Considerations
- Warmup iterations before measurement
- Multiple test runs for averaging
- Tests various tensor sizes (2^10, 2^24, 2^26, 2^27 elements)

## Design Decisions

### Why Runtime Parameters (vs Const Generics)
- Originally used const generics for compile-time optimization
- Switched to runtime parameters for:
  - Flexibility in parameter selection
  - Reduced binary size (fewer instantiations)
  - Dynamic configuration capability

### Kernel Pre-compilation
- All kernel variants are pre-compiled in CUDA
- Runtime selection avoids JIT compilation overhead
- Trade-off: Limited to 7 specific parameter combinations

## Technical Details

### Constants
- `BLOCK_SIZE`: 256 (threads per block)
- `STRIDE`: 2 (for pairwise reduction)
- Warp size: 32 (standard CUDA)

### Grid/Block Configuration
- Grid dimension calculated from tensor size and tile configuration
- Shared memory sized based on tile and point parameters
- Tile height adjustment for boundary conditions

## Dependencies
- `csl_cuda`: CUDA system interface
- `slop_tensor`: Tensor operations
- `slop_alloc`: Memory allocation utilities
- External CUDA kernels from `csl_cuda::sys::prover_clean`


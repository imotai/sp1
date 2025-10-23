#include "tracegen.cuh"
#include "config.cuh"

__global__ void generateColIndex(
    uint32_t* col_index,
    uint32_t starting_cols,
    size_t trace_num_cols,
    size_t trace_num_rows) {

    size_t total = (trace_num_cols * trace_num_rows) >> 1;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {

        size_t col = i / (trace_num_rows >> 1);

        col_index[i] = col + starting_cols;
    }
}

__global__ void generateStartIndices(
    uint32_t* col_index,
    size_t offset,
    size_t trace_num_cols,
    size_t trace_num_rows) {


    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < trace_num_cols;
         i += blockDim.x * gridDim.x) {

        col_index[i] = (offset + i * trace_num_rows) >> 1;
    }
}

__global__ void fillBuffer(uint32_t* dst, uint32_t val, uint32_t max_log_row_count, size_t len) {

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        dst[i] = val + i/(1 << (max_log_row_count-1));
    }
}

extern "C" void* generate_col_index() { return (void*)generateColIndex; }
extern "C" void* generate_start_indices() { return (void*)generateStartIndices; }
extern "C" void* fill_buffer() { return (void*)fillBuffer; }

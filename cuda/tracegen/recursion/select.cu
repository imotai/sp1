#include "csl-cbindgen.hpp"

#include "fields/bb31_t.cuh"

template <class T>
__global__ void recursion_select_generate_preprocessed_trace_kernel(
    T *trace,
    uintptr_t trace_height,
    const csl_sys::SelectInstr<T> *instructions,
    uintptr_t nb_instructions)
{
    static const size_t COLUMNS =
        sizeof(csl_sys::SelectPreprocessedCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_instructions; i += blockDim.x * gridDim.x)
    {
        csl_sys::SelectPreprocessedCols<T> cols;
        const auto &instr = instructions[i];
        cols.is_real = T::one();
        cols.addrs = instr.addrs;
        cols.mult1 = instr.mult1;
        cols.mult2 = instr.mult2;

        const T *arr = reinterpret_cast<T *>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j)
        {
            trace[i + j * trace_height] = arr[j];
        }
    }
}

template <class T>
__global__ void recursion_select_generate_trace_kernel(
    T *trace,
    uintptr_t trace_height,
    const csl_sys::SelectEvent<T> *events,
    uintptr_t nb_events)
{
    static const size_t COLUMNS =
        sizeof(csl_sys::SelectCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x)
    {
        csl_sys::SelectCols<T> cols;
        cols.vals = events[i];

        const T *arr = reinterpret_cast<T *>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j)
        {
            trace[i + j * trace_height] = arr[j];
        }
    }
}

namespace csl_sys
{
    extern KernelPtr recursion_select_generate_preprocessed_trace_baby_bear_kernel()
    {
        return (KernelPtr)::recursion_select_generate_preprocessed_trace_kernel<bb31_t>;
    }
    extern KernelPtr recursion_select_generate_trace_baby_bear_kernel()
    {
        return (KernelPtr)::recursion_select_generate_trace_kernel<bb31_t>;
    }
}

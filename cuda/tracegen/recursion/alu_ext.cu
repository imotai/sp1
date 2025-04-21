#include "csl-cbindgen.hpp"

#include "fields/bb31_t.cuh"

template <class T>
__global__ void recursion_ext_alu_generate_preprocessed_trace_kernel(
    T *trace,
    uintptr_t trace_height,
    const csl_sys::ExtAluInstr<T> *instructions,
    uintptr_t nb_instructions)
{
    static const size_t COLUMNS =
        sizeof(csl_sys::ExtAluAccessCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_instructions; i += blockDim.x * gridDim.x)
    {
        csl_sys::ExtAluAccessCols<T> cols;
        const auto &instr = instructions[i];
        cols.addrs = instr.addrs;
        cols.is_add = T::zero();
        cols.is_sub = T::zero();
        cols.is_mul = T::zero();
        cols.is_div = T::zero();
        cols.mult = instr.mult;

        switch (instr.opcode)
        {
        case csl_sys::ExtAluOpcode::AddE:
            cols.is_add = T::one();
            break;
        case csl_sys::ExtAluOpcode::SubE:
            cols.is_sub = T::one();
            break;
        case csl_sys::ExtAluOpcode::MulE:
            cols.is_mul = T::one();
            break;
        case csl_sys::ExtAluOpcode::DivE:
            cols.is_div = T::one();
            break;
        }

        const T *arr = reinterpret_cast<T *>(&cols);
        size_t start =
            (i % csl_sys::NUM_EXT_ALU_ENTRIES_PER_ROW) * COLUMNS;
        for (size_t j = 0; j < COLUMNS; ++j)
        {
            trace[(i / csl_sys::NUM_EXT_ALU_ENTRIES_PER_ROW) + (j + start) * trace_height] = arr[j];
        }
    }
}

template <class T>
__global__ void recursion_ext_alu_generate_trace_kernel(
    T *trace,
    uintptr_t trace_height,
    const csl_sys::ExtAluEvent<T> *events,
    uintptr_t nb_events)
{
    static const size_t COLUMNS =
        sizeof(csl_sys::ExtAluValueCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x)
    {
        csl_sys::ExtAluValueCols<T> cols;
        cols.vals = events[i];

        const T *arr = reinterpret_cast<T *>(&cols);
        size_t start =
            (i % csl_sys::NUM_EXT_ALU_ENTRIES_PER_ROW) * COLUMNS;
        for (size_t j = 0; j < COLUMNS; ++j)
        {
            trace[(i / csl_sys::NUM_EXT_ALU_ENTRIES_PER_ROW) + (j + start) * trace_height] = arr[j];
        }
    }
}

namespace csl_sys
{
    extern KernelPtr recursion_ext_alu_generate_preprocessed_trace_baby_bear_kernel()
    {
        return (KernelPtr)::recursion_ext_alu_generate_preprocessed_trace_kernel<bb31_t>;
    }
    extern KernelPtr recursion_ext_alu_generate_trace_baby_bear_kernel()
    {
        return (KernelPtr)::recursion_ext_alu_generate_trace_kernel<bb31_t>;
    }
}

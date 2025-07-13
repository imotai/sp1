#include "csl-cbindgen.hpp"

#include "fields/bb31_t.cuh"

template <class T>
__global__ void recursion_prefix_sum_checks_generate_trace_kernel(
    T *trace,
    uintptr_t trace_height,
    const csl_sys::PrefixSumChecksEvent<T> *events,
    uintptr_t nb_events)
{
    static const size_t COLUMNS =
        sizeof(csl_sys::PrefixSumChecksCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x)
    {
        csl_sys::PrefixSumChecksCols<T> cols;
        const auto &event = events[i];
        cols.x1 = event.x1;
        cols.x2 = event.x2;
        cols.acc = event.acc;
        cols.new_acc = event.new_acc;
        cols.felt_acc = event.field_acc;
        cols.felt_new_acc = event.new_field_acc;

        const T *arr = reinterpret_cast<T *>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j)
        {
            trace[i + j * trace_height] = arr[j];
        }
    }
}

namespace csl_sys
{
    extern KernelPtr recursion_prefix_sum_checks_generate_trace_baby_bear_kernel()
    {
        return (KernelPtr)::recursion_prefix_sum_checks_generate_trace_kernel<bb31_t>;
    }
}

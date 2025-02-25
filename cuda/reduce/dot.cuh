#pragma once

#include "reduction.cuh"
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"

extern "C" void *partial_inner_product_baby_bear_kernel();

extern "C" void *partial_inner_product_baby_bear_extension_kernel();

extern "C" void *partial_inner_product_baby_bear_base_extension_kernel();

extern "C" void *partial_dot_baby_bear_kernel();

extern "C" void *partial_dot_baby_bear_extension_kernel();

extern "C" void *partial_dot_baby_bear_base_extension_kernel();

extern "C" void *dot_along_short_dimension_kernel_baby_bear_base_base();

extern "C" void *dot_along_short_dimension_kernel_baby_bear_base_extension();

extern "C" void *dot_along_short_dimension_kernel_baby_bear_extension_extension();

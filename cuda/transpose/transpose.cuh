#pragma once

namespace transpose
{
    extern "C" void *transpose_kernel_baby_bear();
    extern "C" void *transpose_kernel_u32();
    extern "C" void *transpose_kernel_baby_bear_digest();
    extern "C" void *transpose_kernel_u32_digest();
    extern "C" void *transpose_kernel_baby_bear_extension();
}
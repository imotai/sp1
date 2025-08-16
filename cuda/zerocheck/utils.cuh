#pragma once

#include <cstdint>
#include "../fields/kb31_extension_t.cuh"

struct Instruction {
    unsigned char opcode;
    unsigned char b_variant;
    unsigned char c_variant;
    unsigned short a;
    unsigned short b;
    unsigned short c;
};

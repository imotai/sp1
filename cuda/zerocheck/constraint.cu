#include <type_traits>

#include "./constraint.cuh"
#include "./folder.cuh"
#include "./utils.cuh"

#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"
#include "../reduce/reduction.cuh"

#define DEBUG_FLAG 0  // Set this to 0 or 1

#if DEBUG_FLAG == 1
    #define DEBUG(...) printf(__VA_ARGS__)
#else
    #define DEBUG(...)  // Do nothing
#endif

template<typename K, size_t MEMORY_SIZE>
__global__ void constraintPolyEval(
    size_t numAirBlocks,
    const uint32_t *__restrict__ constraintIndices,
    const Instruction* evalProgram,
    const uint32_t *__restrict__ evalProgramIndices,
    size_t evalProgramLen,
    const bb31_t *evalConstantsF,
    const uint32_t *__restrict__ evalConstantsFIndices,
    const bb31_extension_t *evalConstantsEF,
    const uint32_t *__restrict__ evalConstantsEFIndices,
    const bb31_extension_t *partialLagrange,
    const K *preprocessedTrace,
    size_t preprocessedWidth,
    const K *mainTrace,
    size_t mainWidth,
    size_t height,
    const bb31_extension_t *powersOfAlpha,
    const bb31_t *publicValues,
    const bb31_extension_t *batchingPowers,
    bb31_extension_t *__restrict__ constraintValues)
{
    K expr_f[MEMORY_SIZE];
    bb31_extension_t expr_ef[10];
    ConstraintFolder<K> folder = ConstraintFolder<K>();

    bb31_extension_t thread_sum = bb31_extension_t::zero();
    
    // This kernel assumes that a single block deals with a single `xValueIdx`.
    size_t xValueIdx = blockDim.z * blockIdx.z + threadIdx.z;

    for (size_t rowIdx = blockDim.x * blockIdx.x + threadIdx.x; rowIdx < height; rowIdx += blockDim.x * gridDim.x) {
        for (size_t airBlockIdx = blockDim.y * blockIdx.y + threadIdx.y; airBlockIdx < numAirBlocks; airBlockIdx += blockDim.y * gridDim.y) {
            size_t constraint_offset = constraintIndices[airBlockIdx];
            size_t program_start_idx = evalProgramIndices[airBlockIdx];
            size_t program_end_idx = evalProgramLen;
            if (airBlockIdx + 1 < numAirBlocks) {
                program_end_idx = evalProgramIndices[airBlockIdx + 1];
            }
            size_t f_constant_offset = evalConstantsFIndices[airBlockIdx];
            size_t ef_constant_offset = evalConstantsEFIndices[airBlockIdx];
            for (size_t i = 0; i < MEMORY_SIZE; i++) {
                expr_f[i] = K::zero();
            }
            for (size_t i = 0; i < 10; i++) {
                expr_ef[i] = bb31_extension_t::zero();
            }

            folder.preprocessed = preprocessedTrace;
            folder.preprocessed_width = preprocessedWidth;
            folder.main = mainTrace;
            folder.main_width = mainWidth;
            folder.height = height;
            folder.publicValues = publicValues;
            folder.powersOfAlpha = powersOfAlpha;
            folder.constraintIndex = 0;
            folder.accumulator = bb31_extension_t::zero();
            folder.rowIdx = rowIdx;
            folder.xValueIdx = xValueIdx;

            for (size_t i = program_start_idx; i < program_end_idx; i++) {
                Instruction instr = evalProgram[i];
                switch (instr.opcode) {
                    case 0:
                        DEBUG("EMPTY\n");
                        break;

                    case 1:
                        DEBUG("FAssignC: %d <- %d\n", instr.a, instr.b);
                        expr_f[instr.a] = evalConstantsF[f_constant_offset + instr.b];
                        break;
                    case 2:
                        DEBUG(
                            "FAssignV: %d <- (%d, %d)\n",
                            instr.a,
                            instr.b_variant,
                            instr.b
                        );
                        expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b);
                        break;
                    case 3:
                        DEBUG("FAssignE: %d <- %d\n", instr.a, instr.b);
                        expr_f[instr.a] = expr_f[instr.b];
                        break;
                    case 4:
                        DEBUG(
                            "FAddVC: %d <- %d + %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b
                        );
                        expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b)
                            + evalConstantsF[f_constant_offset + instr.c];
                        break;
                    case 5:
                        DEBUG(
                            "FAddVV: %d <- (%d, %d) + (%d, %d)\n",
                            instr.a,
                            instr.b_variant,
                            instr.b,
                            instr.c_variant,
                            instr.c
                        );
                        expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b)
                            + folder.var_f(instr.c_variant, instr.c);
                        break;
                    case 6:
                        DEBUG(
                            "FAddVE: %d <- (%d, %d) + %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b,
                            instr.c
                        );
                        expr_f[instr.a] =
                            folder.var_f(instr.b_variant, instr.b) + expr_f[instr.c];
                        break;

                    case 7:
                        DEBUG(
                            "FAddEC: %d <- %d + %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b
                        );
                        expr_f[instr.a] = expr_f[instr.b] + evalConstantsF[f_constant_offset + instr.c];
                        break;
                    case 8:
                        DEBUG(
                            "FAddEV: %d <- %d + (%d, %d)\n",
                            instr.a,
                            instr.b,
                            instr.c_variant,
                            instr.c
                        );
                        expr_f[instr.a] =
                            expr_f[instr.b] + folder.var_f(instr.c_variant, instr.c);
                        break;
                    case 9:
                        DEBUG("FAddEE: %d <- %d + %d\n", instr.a, instr.b, instr.c);
                        expr_f[instr.a] = expr_f[instr.b] + expr_f[instr.c];
                        break;
                    case 10:
                        DEBUG("FAddAssignE: %d <- %d\n", instr.a, instr.b);
                        expr_f[instr.a] += expr_f[instr.b];
                        break;

                    case 11:
                        DEBUG(
                            "FSubVC: %d <- %d - %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b
                        );
                        expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b)
                            - evalConstantsF[f_constant_offset + instr.c];
                        break;
                    case 12:
                        DEBUG(
                            "FSubVV: %d <- (%d, %d) - (%d, %d)\n",
                            instr.a,
                            instr.b_variant,
                            instr.b,
                            instr.c_variant,
                            instr.c
                        );
                        expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b)
                            - folder.var_f(instr.c_variant, instr.c);
                        break;
                    case 13:
                        DEBUG(
                            "FSubVE: %d <- (%d, %d) - %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b,
                            instr.c
                        );
                        expr_f[instr.a] =
                            folder.var_f(instr.b_variant, instr.b) - expr_f[instr.c];
                        break;

                    case 14:
                        DEBUG("FSubEC: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                        expr_f[instr.a] = expr_f[instr.b] - evalConstantsF[f_constant_offset + instr.c];
                        break;
                    case 15:
                        DEBUG(
                            "FSubEV: %d <- %d - (%d, %d)\n",
                            instr.a,
                            instr.b,
                            instr.c_variant,
                            instr.c
                        );
                        expr_f[instr.a] =
                            expr_f[instr.b] - folder.var_f(instr.c_variant, instr.c);
                        break;
                    case 16:
                        DEBUG("FSubEE: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                        expr_f[instr.a] = expr_f[instr.b] - expr_f[instr.c];
                        break;
                    case 17:
                        DEBUG("FSubAssignE: %d <- %d\n", instr.a, instr.b);
                        expr_f[instr.a] -= expr_f[instr.b];
                        break;

                    case 18:
                        DEBUG(
                            "FMulVC: %d <- %d * %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b
                        );
                        expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b)
                            * evalConstantsF[f_constant_offset + instr.c];
                        break;
                    case 19:
                        DEBUG(
                            "FMulVV: %d <- (%d, %d) * (%d, %d)\n",
                            instr.a,
                            instr.b_variant,
                            instr.b,
                            instr.c_variant,
                            instr.c
                        );
                        expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b)
                            * folder.var_f(instr.c_variant, instr.c);
                        break;
                    case 20:
                        DEBUG(
                            "FMulVE: %d <- (%d, %d) * %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b,
                            instr.c
                        );
                        expr_f[instr.a] =
                            folder.var_f(instr.b_variant, instr.b) * expr_f[instr.c];
                        break;

                    case 21:
                        DEBUG(
                            "FMulEC: %d <- %d * %d\n",
                            instr.a,
                            instr.b_variant,
                            instr.b
                        );
                        expr_f[instr.a] = expr_f[instr.b] * evalConstantsF[f_constant_offset + instr.c];
                        break;
                    case 22:
                        DEBUG(
                            "FMulEV: %d <- %d * (%d, %d)\n",
                            instr.a,
                            instr.b,
                            instr.c_variant,
                            instr.c
                        );
                        expr_f[instr.a] =
                            expr_f[instr.b] * folder.var_f(instr.c_variant, instr.c);
                        break;
                    case 23:
                        DEBUG("FMulEE: %d <- %d * %d\n", instr.a, instr.b, instr.c);
                        DEBUG(
                            "FMulEE Input: %d, %d\n",
                            expr_f[instr.b],
                            expr_f[instr.c]
                        );
                        expr_f[instr.a] = expr_f[instr.b] * expr_f[instr.c];
                        DEBUG("FMulEE Output: %d\n", expr_f[instr.a]);
                        break;
                    case 24:
                        DEBUG("FMulAssignE: %d <- %d\n", instr.a, instr.b);
                        expr_f[instr.a] *= expr_f[instr.b];
                        break;

                    case 25:
                        DEBUG("FNegE: %d <- -%d\n", instr.a, instr.b);
                        expr_f[instr.a] = -expr_f[instr.b];
                        break;

                    // case 26:
                    //     DEBUG("EAssignC: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] = evalConstantsEF[ef_constant_offset + instr.b];
                    //     break;
                    // case 27:
                    //     DEBUG(
                    //         "EAssignV: %d <- (%d, %d)\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b);
                    //     break;
                    // case 28:
                    //     DEBUG("EAssignE: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] = expr_ef[instr.b];
                    //     break;

                    // case 29:
                    //     DEBUG(
                    //         "EAddVC: %d <- %d + %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b)
                    //         + evalConstantsEF[ef_constant_offset + instr.c];
                    //     break;
                    // case 30:
                    //     DEBUG(
                    //         "EAddVV: %d <- (%d, %d) + (%d, %d)\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b,
                    //         instr.c_variant,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b)
                    //         + folder.var_ef(instr.c_variant, instr.c);
                    //     break;
                    // case 31:
                    //     DEBUG(
                    //         "EAddVE: %d <- (%d, %d) + %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] =
                    //         folder.var_ef(instr.b_variant, instr.b) + expr_ef[instr.c];
                    //     break;

                    // case 32:
                    //     DEBUG(
                    //         "EAddEC: %d <- %d + %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     expr_ef[instr.a] = expr_ef[instr.b] + evalConstantsEF[ef_constant_offset + instr.b];
                    //     break;
                    // case 33:
                    //     DEBUG(
                    //         "EAddEV: %d <- %d + (%d, %d)\n",
                    //         instr.a,
                    //         instr.b,
                    //         instr.c_variant,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] =
                    //         expr_ef[instr.b] + folder.var_ef(instr.c_variant, instr.c);
                    //     break;
                    // case 34:
                    //     DEBUG("EAddEE: %d <- %d + %d\n", instr.a, instr.b, instr.c);
                    //     expr_ef[instr.a] = expr_ef[instr.b] + expr_ef[instr.c];
                    //     break;
                    // case 35:
                    //     DEBUG("EAddAssignE: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] += expr_ef[instr.b];
                    //     break;

                    // case 36:
                    //     DEBUG(
                    //         "ESubVC: %d <- %d - %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b)
                    //         - evalConstantsEF[ef_constant_offset + instr.c];
                    //     break;
                    // case 37:
                    //     DEBUG(
                    //         "ESubVV: %d <- (%d, %d) - (%d, %d)\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b,
                    //         instr.c_variant,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b)
                    //         - folder.var_ef(instr.c_variant, instr.c);
                    //     break;
                    // case 38:
                    //     DEBUG(
                    //         "ESubVE: %d <- (%d, %d) - %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] =
                    //         folder.var_ef(instr.b_variant, instr.b) - expr_ef[instr.c];
                    //     break;

                    // case 39:
                    //     DEBUG(
                    //         "ESubEC: %d <- %d - %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     expr_ef[instr.a] = expr_ef[instr.b] - evalConstantsEF[ef_constant_offset + instr.b];
                    //     break;
                    // case 40:
                    //     DEBUG(
                    //         "ESubEV: %d <- %d - (%d, %d)\n",
                    //         instr.a,
                    //         instr.b,
                    //         instr.c_variant,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] =
                    //         expr_ef[instr.b] - folder.var_ef(instr.c_variant, instr.c);
                    //     break;
                    // case 41:
                    //     DEBUG("ESubEE: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                    //     expr_ef[instr.a] = expr_ef[instr.b] - expr_ef[instr.c];
                    //     break;
                    // case 42:
                    //     DEBUG("ESubAssignE: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] -= expr_ef[instr.b];
                    //     break;

                    // case 43:
                    //     DEBUG(
                    //         "EMulVC: %d <- %d * %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b)
                    //         * evalConstantsEF[ef_constant_offset + instr.c];
                    //     break;
                    // case 44:
                    //     DEBUG(
                    //         "EMulVV: %d <- (%d, %d) * (%d, %d)\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b,
                    //         instr.c_variant,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b)
                    //         * folder.var_ef(instr.c_variant, instr.c);
                    //     break;
                    // case 45:
                    //     DEBUG(
                    //         "EMulVE: %d <- (%d, %d) * %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] =
                    //         folder.var_ef(instr.b_variant, instr.b) * expr_ef[instr.c];
                    //     break;

                    // case 46:
                    //     DEBUG(
                    //         "EMulEC: %d <- %d * %d\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     expr_ef[instr.a] = expr_ef[instr.b] * evalConstantsEF[ef_constant_offset + instr.b];
                    //     break;
                    // case 47:
                    //     DEBUG(
                    //         "EMulEV: %d <- %d * (%d, %d)\n",
                    //         instr.a,
                    //         instr.b,
                    //         instr.c_variant,
                    //         instr.c
                    //     );
                    //     expr_ef[instr.a] =
                    //         expr_ef[instr.b] * folder.var_ef(instr.c_variant, instr.c);
                    //     break;
                    // case 48:
                    //     DEBUG("EMulEE: %d <- %d * %d\n", instr.a, instr.b, instr.c);
                    //     expr_ef[instr.a] = expr_ef[instr.b] * expr_ef[instr.c];
                    //     break;
                    // case 49:
                    //     DEBUG("EMulAssignE: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] *= expr_ef[instr.b];
                    //     break;

                    // case 50:
                    //     DEBUG("ENegE: %d <- -%d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] = bb31_extension_t::zero() - expr_ef[instr.b];
                    //     break;

                    // case 51:
                    //     DEBUG("EFFromE: %d <- %d\n", instr.a, instr.b);
                    //     if constexpr (std::is_same_v<K, bb31_t>) {
                    //         bb31_extension_t result;
                    //         result.value[0] = expr_f[instr.b];
                    //         result.value[1] = bb31_t {0};
                    //         result.value[2] = bb31_t {0};
                    //         result.value[3] = bb31_t {0};
                    //         expr_ef[instr.a] = result;
                    //     } else {
                    //         expr_ef[instr.a] = expr_ef[instr.b];
                    //     }
                    //     break;
                    // case 52:
                    //     DEBUG("EFAddEE: %d <- %d + %d\n", instr.a, instr.b, instr.c);
                    //     expr_ef[instr.a] = expr_ef[instr.b] + expr_f[instr.c];
                    //     break;
                    // case 53:
                    //     DEBUG("EFAddAssignE: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] += expr_f[instr.b];
                    //     break;
                    // case 54:
                    //     DEBUG("EFSubEE: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                    //     expr_ef[instr.a] = expr_ef[instr.b] - expr_f[instr.c];
                    //     break;
                    // case 55:
                    //     DEBUG("EFSubAssignE: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] -= expr_f[instr.b];
                    //     break;
                    // case 56:
                    //     DEBUG("EFMulEE: %d <- %d * %d\n", instr.a, instr.b, instr.c);
                    //     expr_ef[instr.a] = expr_ef[instr.b] * expr_f[instr.c];
                    //     break;
                    // case 57:
                    //     DEBUG("EFMulAssignE: %d <- %d\n", instr.a, instr.b);
                    //     expr_ef[instr.a] *= expr_f[instr.b];
                    //     break;
                    // case 58:
                    //     DEBUG(
                    //         "EFAsBaseSlice: %d <- (%d, %d)\n",
                    //         instr.a,
                    //         instr.b_variant,
                    //         instr.b
                    //     );
                    //     // UNSUPPORTED
                    //     break;

                    case 59:
                        DEBUG("FAssertZero: %d\n", instr.a);
                        folder.accumulator += (folder.powersOfAlpha[constraint_offset + folder.constraintIndex] * expr_f[instr.a]);
                        folder.constraintIndex++;
                        break;
                    // case 60:
                    //     DEBUG("EAssertZero: %d\n", instr.a);
                    //     folder.accumulator += (folder.powersOfAlpha[constraint_offset + folder.constraintIndex] * expr_ef[instr.a]);
                    //     folder.constraintIndex++;
                    //     break;
                }
            }

            bb31_extension_t gkr_correction = bb31_extension_t::zero();
            
            if (airBlockIdx == 0) {
                for (size_t i = 0; i < mainWidth; i++) {
                    gkr_correction += batchingPowers[i]*folder.var_f(4, i);
                }
                for (size_t i = 0; i < preprocessedWidth; i++) {
                    gkr_correction += batchingPowers[mainWidth + i]*folder.var_f(2,i);
                }
            }

            bb31_extension_t eq = bb31_extension_t::load(partialLagrange, rowIdx);
            thread_sum += (folder.accumulator + gkr_correction) * eq;
        }
    }

    extern __shared__ unsigned char memory[];
    bb31_extension_t *shared = reinterpret_cast<bb31_extension_t *>(memory);
    AddOp<bb31_extension_t> op;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    bb31_extension_t thread_block_sum = partialBlockReduce(block, tile, thread_sum, shared, op);

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        bb31_extension_t::store(constraintValues, (xValueIdx * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x, thread_block_sum);
    }
}

extern "C" void *constraint_poly_eval_32_baby_bear_kernel()
{
    return (void *)constraintPolyEval<bb31_t, 32>;
}

extern "C" void *constraint_poly_eval_64_baby_bear_kernel()
{
    return (void *)constraintPolyEval<bb31_t, 64>;
}

extern "C" void *constraint_poly_eval_128_baby_bear_kernel()
{
    return (void *)constraintPolyEval<bb31_t, 128>;
}

extern "C" void *constraint_poly_eval_256_baby_bear_kernel()
{
    return (void *)constraintPolyEval<bb31_t, 256>;
}

extern "C" void *constraint_poly_eval_512_baby_bear_kernel()
{
    return (void *)constraintPolyEval<bb31_t, 512>;
}

extern "C" void *constraint_poly_eval_1024_baby_bear_kernel()
{
    return (void *)constraintPolyEval<bb31_t, 1024>;
}

extern "C" void *constraint_poly_eval_32_baby_bear_extension_kernel()
{
    return (void *)constraintPolyEval<bb31_extension_t, 32>;
}

extern "C" void *constraint_poly_eval_64_baby_bear_extension_kernel()
{
    return (void *)constraintPolyEval<bb31_extension_t, 64>;
}

extern "C" void *constraint_poly_eval_128_baby_bear_extension_kernel()
{
    return (void *)constraintPolyEval<bb31_extension_t, 128>;
}

extern "C" void *constraint_poly_eval_256_baby_bear_extension_kernel()
{
    return (void *)constraintPolyEval<bb31_extension_t, 256>;
}

extern "C" void *constraint_poly_eval_512_baby_bear_extension_kernel()
{
    return (void *)constraintPolyEval<bb31_extension_t, 512>;
}

extern "C" void *constraint_poly_eval_1024_baby_bear_extension_kernel()
{
    return (void *)constraintPolyEval<bb31_extension_t, 1024>;
}
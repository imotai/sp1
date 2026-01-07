#include "csl-cbindgen.hpp"
#include "scan/scan.cuh"
#include "fields/kb31_septic_extension_t.cuh"

namespace csl_sys {
extern KernelPtr single_block_scan_kernel_large_bb31_septic_curve() {
    return (KernelPtr)scan_large::SingleBlockScan<bb31_septic_curve_t>;
}
extern KernelPtr scan_kernel_large_bb31_septic_curve() {
    return (KernelPtr)scan_large::Scan<bb31_septic_curve_t>;
}
} // namespace csl_sys
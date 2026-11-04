/*
 * The Imaging library
 *
 * The SIMD kernels of one instruction-set tier (design/SIMD_DISPATCH_PLAN.md):
 * every companion file is included here, and the tier's table of kernels
 * (SIMDKernels.h) is defined at the end. The file is compiled once per
 * tier, with the tier's compiler flags; everything in it is static but the
 * table.
 */

#include "Imaging.h"

#if defined(IMAGING_SIMD_TABLE)

#include "FilterSIMD.c"
#include "ReduceSIMD.c"
#include "ResampleSIMD.c"
#include "ConvertSIMD.c"

const ImagingSIMDKernels IMAGING_SIMD_TABLE = {
    .name = IMAGING_SIMD_TIER_NAME,
    .filter = &filter_simd_kernels,
    .reduce = &reduce_simd_kernels,
    .resample = &resample_simd_kernels,
    .convert = &convert_simd_kernels,
};

#endif

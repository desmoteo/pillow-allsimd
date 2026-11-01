/*
 * The Imaging library
 *
 * The choice of the SIMD kernels at run time (design/SIMD_DISPATCH_PLAN.md):
 * ImagingSIMD points at the kernel table of the tier the CPU supports,
 * among those compiled in (SIMDKernels.c). A build compiled for one
 * instruction set has one table; the scalar table has no kernels.
 */

#include "Imaging.h"

static const ImagingSIMDFilterKernels no_filter_kernels = {0};

const ImagingSIMDKernels ImagingSIMDKernels_scalar = {
    .name = "scalar",
    .filter = &no_filter_kernels,
};

const ImagingSIMDKernels *ImagingSIMD = &ImagingSIMDKernels_scalar;

#if defined(IMAGING_SIMD_TABLE)
extern const ImagingSIMDKernels IMAGING_SIMD_TABLE;
#endif

int
ImagingSIMDInit(const char **error) {
    *error = NULL;
#if defined(IMAGING_SIMD_TABLE)
    ImagingSIMD = &IMAGING_SIMD_TABLE;
#endif
    return 0;
}

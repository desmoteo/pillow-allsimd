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
static const ImagingSIMDReduceKernels no_reduce_kernels = {0};
static const ImagingSIMDResampleKernels no_resample_kernels = {0};
static const ImagingSIMDConvertKernels no_convert_kernels = {0};
static const ImagingSIMDColorLUTKernels no_colorlut_kernels = {0};
static const ImagingSIMDGeometryKernels no_geometry_kernels = {0};
static const ImagingSIMDTransformFastKernels no_transform_fast_kernels = {0};
static const ImagingSIMDBlurKernels no_blur_kernels = {0};

const ImagingSIMDKernels ImagingSIMDKernels_scalar = {
    .name = "scalar",
    .filter = &no_filter_kernels,
    .reduce = &no_reduce_kernels,
    .resample = &no_resample_kernels,
    .convert = &no_convert_kernels,
    .colorlut = &no_colorlut_kernels,
    .geometry = &no_geometry_kernels,
    .transform_fast = &no_transform_fast_kernels,
    .blur = &no_blur_kernels,
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

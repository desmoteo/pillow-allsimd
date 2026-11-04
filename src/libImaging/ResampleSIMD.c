/* The resampling kernels of one tier (resize, in Resample.c), compiled
   into SIMDKernels.c: the horizontal and vertical passes for 8-bit images
   (ResampleSIMDHorizontalConv.c, included twice, the second time with
   16-bit results for fast mode; ResampleSIMDVerticalConv.c), the passes
   for I, F and I;16 images in vectors of doubles (ResampleSIMD32bpc.c),
   and fast mode's vertical pass (ResampleFastSIMD.c). Resample.c keeps
   the portable kernels and uses one of these where the table has it. */

#include <math.h>
#include <stdint.h>

/* as in Resample.c */
#define ROUND_UP(f) ((int)((f) >= 0.0 ? (f) + 0.5F : (f) - 0.5F))

#include "ResampleSIMDHorizontalConv.c"
#define RESAMPLE_OUT16
#include "ResampleSIMDHorizontalConv.c"
#undef RESAMPLE_OUT16
#include "ResampleSIMDVerticalConv.c"
#include "ResampleSIMD32bpc.c"
#include "ResampleFast.h"
#include "ResampleFastSIMD.c"

#undef ROUND_UP

static const ImagingSIMDResampleKernels resample_simd_kernels = {
    .horizontal_8u = ImagingResampleHorizontalConvolution8u,
    .horizontal_8u4x = ImagingResampleHorizontalConvolution8u4x,
    .horizontal_8u_16 = ImagingResampleHorizontalConvolution8u_16,
    .horizontal_8u4x_16 = ImagingResampleHorizontalConvolution8u4x_16,
    .horizontal_1u = ImagingResampleHorizontalConvolution1u,
    .horizontal_1u4x = ImagingResampleHorizontalConvolution1u4x,
    .vertical_8u = ImagingResampleVerticalConvolution8u,
    .horizontal_f32 = resample_horizontal_f32,
    .horizontal_i32 = resample_horizontal_i32,
    .horizontal_u16le = resample_horizontal_u16le,
    .horizontal_u16be = resample_horizontal_u16be,
    .vertical_f32 = resample_vertical_f32,
    .vertical_i32 = resample_vertical_i32,
    .vertical_u16le = resample_vertical_u16le,
    .vertical_u16be = resample_vertical_u16be,
    .fast_vrow = resample_fast_vrow_simd,
};

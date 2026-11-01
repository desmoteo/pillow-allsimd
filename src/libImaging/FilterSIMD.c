/* The kernel filters' SIMD kernels (ImageFilter.Kernel, the built-in 3x3
   and 5x5 filters), compiled into SIMDKernels.c. Filter.c's filter_simd()
   chooses one of these for an 8-bit image from the active tier's table:
   the integer kernels for coefficients that fit 16 bits, the float ones
   otherwise. They write the whole image, the untouched border included. */

#if defined(IMAGING_SIMD_SSE4)
#include "FilterSIMD_3x3f_u8.c"
#include "FilterSIMD_3x3f_4u8.c"
#include "FilterSIMD_3x3i_4u8.c"
#include "FilterSIMD_5x5f_u8.c"
#include "FilterSIMD_5x5f_4u8.c"
#include "FilterSIMD_5x5i_4u8.c"
#endif
#if defined(IMAGING_SIMD_NEON)
#include "FilterSIMD_NxNf.c"
#endif
#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
#include "FilterSIMD_NxNi.c"
#endif

#if defined(IMAGING_SIMD_NEON)
/* Float kernels for 4-byte images. 1-byte images keep the generic code,
   which GCC vectorises well: on the Jetson Orin it was 1.5x faster than
   the NEON float kernel (CONTOUR on L: 15.5 vs 22.9 ms). 4-byte images
   gain 1.7x. */
static void
filter_3x3f_4u8_neon(Imaging imOut, Imaging im, const float *kernel, float offset) {
    ImagingFilterNxNf_neon(imOut, im, kernel, offset, 3, 4);
}

static void
filter_5x5f_4u8_neon(Imaging imOut, Imaging im, const float *kernel, float offset) {
    ImagingFilterNxNf_neon(imOut, im, kernel, offset, 5, 4);
}
#endif

static const ImagingSIMDFilterKernels filter_simd_kernels = {
#if defined(IMAGING_SIMD_SSE4)
    .i3x3_u8 = ImagingFilter3x3i_u8_simd,
    .i5x5_u8 = ImagingFilter5x5i_u8_simd,
    .i3x3_4u8 = ImagingFilter3x3i_4u8,
    .i5x5_4u8 = ImagingFilter5x5i_4u8,
    .f3x3_u8 = ImagingFilter3x3f_u8,
    .f5x5_u8 = ImagingFilter5x5f_u8,
    .f3x3_4u8 = ImagingFilter3x3f_4u8,
    .f5x5_4u8 = ImagingFilter5x5f_4u8,
#elif defined(IMAGING_SIMD_NEON)
    .i3x3_u8 = ImagingFilter3x3i_u8_simd,
    .i5x5_u8 = ImagingFilter5x5i_u8_simd,
    .i3x3_4u8 = ImagingFilter3x3i_4u8_simd,
    .i5x5_4u8 = ImagingFilter5x5i_4u8_simd,
    .f3x3_4u8 = filter_3x3f_4u8_neon,
    .f5x5_4u8 = filter_5x5f_4u8_neon,
#endif
};

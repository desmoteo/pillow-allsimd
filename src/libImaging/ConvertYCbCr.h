/* The constants of the YCbCr conversions, for ConvertYCbCr.c and its SIMD
   kernels (ConvertYCbCrSIMD.c). */
#ifndef __CONVERT_YCBCR_H__
#define __CONVERT_YCBCR_H__

#define SCALE 6 /* bits */

/* Each table above is T[v] = (int)(v * K * 64 + 0.5) for its coefficient
   K from the comment at the top (v - 128 for the tables of Cb and Cr in the
   conversion to RGB), truncated toward zero. In float32 the same
   expression gives every entry of every table, with or without a fused
   multiply-add (checked for all 256 values), so vectors compute the terms
   instead of looking them up. */
#define K_Y_R (0.299f * 64)
#define K_Y_G (0.587f * 64)
#define K_Y_B (0.114f * 64)
#define K_Cb_R (-0.16874f * 64)
#define K_Cb_G (-0.33126f * 64)
#define K_Cb_B (0.5f * 64)
#define K_Cr_G (-0.41869f * 64)
#define K_Cr_B (-0.08131f * 64)
#define K_R_Cr (1.402f * 64)
#define K_G_Cb (-0.34414f * 64)
#define K_G_Cr (-0.71414f * 64)
#define K_B_Cb (1.772f * 64)

/* Fast mode: JPEG's BT.601 full-range coefficients in 14 bits (see
   ImagingConvertRGB2YCbCrFast). */
#define YCC_K_YR 4899    /* 0.299 * 2^14 */
#define YCC_K_YG 9617    /* 0.587 */
#define YCC_K_YB 1868    /* 0.114 */
#define YCC_K_CBR -2765  /* -0.168736 */
#define YCC_K_CBG -5427  /* -0.331264 */
#define YCC_K_CRG -6860  /* -0.418688 */
#define YCC_K_CRB -1332  /* -0.081312 */
#define YCC_K_HALF 8192  /* 0.5 */
#define YCC_K_RCR 22970  /* 1.402 */
#define YCC_K_GCB -5638  /* -0.344136 */
#define YCC_K_GCR -11700 /* -0.714136 */
#define YCC_K_BCB 29032  /* 1.772 */
/* 128 << 14, plus the rounding half */
#define YCC_BIAS_C ((128 << 14) + (1 << 13))

#endif

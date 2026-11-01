/* Fast mode's resize (ResampleFast.c): the constants and the rounded
   product its portable and SIMD loops (ResampleFastSIMD.c) share. */
#ifndef __RESAMPLE_FAST_H__
#define __RESAMPLE_FAST_H__

#define RF_FRAC 6
/* The sums' bound: each product is within 2 of the value times the
   weight (1 from the weight's rounding, 1 from the product's), so a sum is
   at most 255 * 2^6 * RF_WSUM_MAX + 2 * taps, and adding 2^5 for the
   rounding keeps it below 32602. (The horizontal kernels' precision is at
   least 13 for weights up to 1.98, so coefs_precision - RF_FRAC >= 7.) */
#define RF_WSUM_MAX 1.98
#define RF_TAPS_MAX 128

static inline int
rf_mul(int a, int w) {
    return (a * w + (1 << 14)) >> 15;
}

#endif

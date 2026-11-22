/* Fast mode's resize (ResampleFast.c): the constants and the rounded
   product its portable and SIMD loops (ResampleFastSIMD.c) share. */
#ifndef __RESAMPLE_FAST_H__
#define __RESAMPLE_FAST_H__

#define RF_FRAC 6
/* The sums' bound: each product is within 2 of the value times the
   weight (1 from the weight's rounding, 1 from the product's), so a sum is
   at most 255 * 2^6 * RF_WSUM_MAX + 2 * taps, and adding 2^5 for the
   rounding keeps it below 32602. (The horizontal kernels' precision is at
   least 13 for weights up to 1.98, so coefs_precision - RF_FRAC >= 7.)
   The same bound on the horizontal weights keeps the horizontal sums
   within 16 bits before their clip: 255 * 2^6 * RF_WSUM_MAX is 32313. */
#define RF_WSUM_MAX 1.98
#define RF_TAPS_MAX 128

static inline int
rf_mul(int a, int w) {
    return (a * w + (1 << 14)) >> 15;
}

/* The same number from the two 16-bit halves of the product, for the
   portable loop: a * w = hi * 2^16 + lo with lo in 0 .. 65535, so
   rf_mul = 2 * hi + ((lo + 2^14) >> 15), and the last term is
   ((lo >> 14) + 1) >> 1. Equal to rf_mul() for every pair of 16-bit
   operands whose result fits 16 bits (all but -32768 * -32768; checked
   exhaustively). Written this way a compiler vectorizes it in 16-bit
   lanes with a high and a low multiply, twice the lanes of the 32-bit
   product and no widening. */
static inline INT16
rf_mul16(INT16 a, INT16 w) {
    INT16 hi = (INT16)(((int)a * w) >> 16);
    UINT16 lo = (UINT16)((unsigned)(UINT16)a * (UINT16)w);
    return (INT16)(2 * hi + (((lo >> 14) + 1) >> 1));
}

#endif

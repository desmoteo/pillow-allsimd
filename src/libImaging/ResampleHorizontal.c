/* The portable horizontal kernels for 8-bit images with 4-byte pixels,
   with the arithmetic of the SIMD ones (ResampleSIMDHorizontalConv.c):
   16-bit coefficients, 32-bit accumulators, rounding by adding half.
   Included twice by Resample.c: as is, with 8-bit results, and with
   RESAMPLE_OUT16 defined, for fast mode (ResampleFast.c), with 16-bit
   results that keep 6 more bits, as that file explains. */
#if defined(RESAMPLE_OUT16)
#define RH(name) name##_16
#define RH_OUT uint64_t
#else
#define RH(name) name
#define RH_OUT UINT32
#endif

#if defined(RESAMPLE_OUT16)
/* The 16-bit results are clipped to 0 .. 255 (in 2^-6 levels) and doubled
   a row at a time, after the sums were stored as they are: they fit 16
   bits, since resample_fast8() leaves weights that could overflow them to
   the exact passes (ResampleFast.h). Clipped one by one in the pixel loop
   they cost four times the 8-bit kernels' table lookup, and branches that
   mispredict at every clipped value; a row of 16-bit values the compiler
   clips in vectors. In chunks of a constant size, not inlined, for the
   reasons given at rf_vtap() in ResampleFast.c. */
#define RH_CHUNK 64

SIMD_NOINLINE void
rh_clip_chunk_16(INT16 *restrict v) {
    int i;
    for (i = 0; i < RH_CHUNK; i++) {
        int t = v[i];
        v[i] = (INT16)(2 * (t < 0 ? 0 : t > 255 << 6 ? 255 << 6 : t));
    }
}

static void
rh_clip_row_16(INT16 *v, int n) {
    int i = 0;
    for (; i + RH_CHUNK <= n; i += RH_CHUNK) {
        rh_clip_chunk_16(v + i);
    }
    for (; i < n; i++) {
        int t = v[i];
        v[i] = (INT16)(2 * (t < 0 ? 0 : t > 255 << 6 ? 255 << 6 : t));
    }
}

#define RH_CLIP(in, precision) ((INT16)((in) >> (precision)))
#define RH_FINISH(line, xsize) rh_clip_row_16((INT16 *)(line), (xsize) * 4)
#define RH_ELEM INT16
#define RH_FULL (255 << 6) /* before the row's doubling */
#else
#define RH_CLIP(in, precision) clip8(in, precision)
#define RH_FINISH(line, xsize)
#define RH_ELEM UINT8
#define RH_FULL 255
#endif

static void
RH(resample_horizontal_8u)(
    RH_OUT *restrict lineOut,
    UINT32 *restrict lineIn,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    int xmin, xmax, xx;
    /* a pointer's width: with an int, x * 4 could wrap under
       -fno-strict-overflow and the loop over the taps is not vectorized */
    ptrdiff_t x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        int ss0, ss1, ss2, ss3;
        UINT8 *in;
        RH_ELEM *out;
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        in = (UINT8 *)&lineIn[xmin];
        ss0 = ss1 = ss2 = ss3 = 1 << (coefs_precision - 1);
        for (x = 0; x < xmax; x++) {
            ss0 += in[x * 4 + 0] * k[x];
            ss1 += in[x * 4 + 1] * k[x];
            ss2 += in[x * 4 + 2] * k[x];
            ss3 += in[x * 4 + 3] * k[x];
        }
        out = (RH_ELEM *)&lineOut[xx];
        out[0] = RH_CLIP(ss0, coefs_precision);
        out[1] = RH_CLIP(ss1, coefs_precision);
        out[2] = RH_CLIP(ss2, coefs_precision);
        out[3] = RH_CLIP(ss3, coefs_precision);
    }
    RH_FINISH(lineOut, xsize);
}

/* The same for 3-band images: the fourth byte is padding, so it isn't
   summed (a quarter of the work), and gets 255 like decoded RGB pixels. */
static void
RH(resample_horizontal_8u3)(
    RH_OUT *restrict lineOut,
    UINT32 *restrict lineIn,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    int xmin, xmax, xx;
    /* a pointer's width: with an int, x * 4 could wrap under
       -fno-strict-overflow and the loop over the taps is not vectorized */
    ptrdiff_t x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        int ss0, ss1, ss2;
        UINT8 *in;
        RH_ELEM *out;
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        in = (UINT8 *)&lineIn[xmin];
        ss0 = ss1 = ss2 = 1 << (coefs_precision - 1);
        for (x = 0; x < xmax; x++) {
            ss0 += in[x * 4 + 0] * k[x];
            ss1 += in[x * 4 + 1] * k[x];
            ss2 += in[x * 4 + 2] * k[x];
        }
        out = (RH_ELEM *)&lineOut[xx];
        out[0] = RH_CLIP(ss0, coefs_precision);
        out[1] = RH_CLIP(ss1, coefs_precision);
        out[2] = RH_CLIP(ss2, coefs_precision);
        out[3] = RH_FULL;
    }
    RH_FINISH(lineOut, xsize);
}

static void
RH(resample_horizontal_8u4x)(
    RH_OUT *restrict lineOut0,
    RH_OUT *restrict lineOut1,
    RH_OUT *restrict lineOut2,
    RH_OUT *restrict lineOut3,
    UINT32 *restrict lineIn0,
    UINT32 *restrict lineIn1,
    UINT32 *restrict lineIn2,
    UINT32 *restrict lineIn3,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    RH(resample_horizontal_8u)(
        lineOut0, lineIn0, xsize, xbounds, kk, kmax, coefs_precision
    );
    RH(resample_horizontal_8u)(
        lineOut1, lineIn1, xsize, xbounds, kk, kmax, coefs_precision
    );
    RH(resample_horizontal_8u)(
        lineOut2, lineIn2, xsize, xbounds, kk, kmax, coefs_precision
    );
    RH(resample_horizontal_8u)(
        lineOut3, lineIn3, xsize, xbounds, kk, kmax, coefs_precision
    );
}

#undef RH_CLIP
#undef RH_FINISH
#undef RH_ELEM
#undef RH_FULL
#undef RH_OUT
#undef RH

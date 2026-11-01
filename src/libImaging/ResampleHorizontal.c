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
static inline INT16
RH(rh_clip)(int in, int precision) {
    in >>= precision;
    return (INT16)(2 * (in < 0 ? 0 : in > 255 << 6 ? 255 << 6 : in));
}
#define RH_CLIP(in, precision) RH(rh_clip)(in, precision)
#define RH_ELEM INT16
#define RH_FULL (255 << 7)
#else
#define RH_CLIP(in, precision) clip8(in, precision)
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
    int xmin, xmax, xx, x;
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
    int xmin, xmax, xx, x;
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
#undef RH_ELEM
#undef RH_FULL
#undef RH_OUT
#undef RH

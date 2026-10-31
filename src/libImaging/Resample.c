#include "Imaging.h"

#include <assert.h>
#include <math.h>

#define ROUND_UP(f) ((int)((f) >= 0.0 ? (f) + 0.5F : (f) - 0.5F))

struct filter {
    double (*filter)(double x);
    double support;
};

static inline double
box_filter(double x) {
    if (x > -0.5 && x <= 0.5) {
        return 1.0;
    }
    return 0.0;
}

static inline double
bilinear_filter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

static inline double
hamming_filter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    if (x == 0.0) {
        return 1.0;
    }
    if (x >= 1.0) {
        return 0.0;
    }
    x = x * M_PI;
    return sin(x) / x * (0.54f + 0.46f * cos(x));
}

static inline double
bicubic_filter(double x) {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
#define a -0.5
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
        return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0.0;
#undef a
}

static inline double
sinc_filter(double x) {
    if (x == 0.0) {
        return 1.0;
    }
    x = x * M_PI;
    return sin(x) / x;
}

static inline double
lanczos_filter(double x) {
    /* truncated sinc */
    if (-3.0 <= x && x < 3.0) {
        return sinc_filter(x) * sinc_filter(x / 3);
    }
    return 0.0;
}

static inline double
mks_2013_filter(double x) {
    /* https://johncostella.com/magic/ */
    if (x < 0) {
        x = -x;
    }
    if (x < 0.5) {
        return 17.0 / 16.0 - 7.0 / 4.0 * pow(x, 2);
    }
    if (x < 1.5) {
        return (1.0 - x) * (7.0 / 4.0 - x);
    }
    if (x < 2.5) {
        return -1.0 / 8.0 * pow(x - 5.0 / 2.0, 2);
    }
    return 0;
}

static inline double
mks_2021_filter(double x) {
    /* https://johncostella.com/magic/ */
    if (x < 0) {
        x = -x;
    }
    if (x < 0.5) {
        return 577.0 / 576.0 - 239.0 / 144.0 * pow(x, 2);
    }
    if (x < 1.5) {
        return 35.0 / 36.0 * (x - 1.0) * (x - 239.0 / 140.0);
    }
    if (x < 2.5) {
        return 1.0 / 6.0 * (x - 2.0) * (65.0 / 24.0 - x);
    }
    if (x < 3.5) {
        return 1.0 / 36.0 * (x - 3.0) * (x - 15.0 / 4.0);
    }
    if (x < 4.5) {
        return -1.0 / 288.0 * pow(x - 9.0 / 2.0, 2);
    }
    return 0;
}

static struct filter BOX = {box_filter, 0.5};
static struct filter BILINEAR = {bilinear_filter, 1.0};
static struct filter HAMMING = {hamming_filter, 1.0};
static struct filter BICUBIC = {bicubic_filter, 2.0};
static struct filter LANCZOS = {lanczos_filter, 3.0};
static struct filter MKS2013 = {mks_2013_filter, 2.5};
static struct filter MKS2021 = {mks_2021_filter, 4.5};

/* 8 bits for result. Filter can have negative areas.
   In one cases the sum of the coefficients will be negative,
   in the other it will be more than 1.0. That is why we need
   two extra bits for overflow and int type. */
#define PRECISION_BITS (32 - 8 - 2)
/* We use signed INT16 type to store coefficients. */
#define MAX_COEFS_PRECISION (16 - 1)

/* Handles values form -640 to 639. */
UINT8 _clip8_lookups[1280] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   2,   3,   4,   5,
    6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
    40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
    57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
    74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
    91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
    193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226,
    227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
};

UINT8 *clip8_lookups = &_clip8_lookups[640];

static inline UINT8
clip8(int in, int precision) {
    return clip8_lookups[in >> precision];
}

#include "ResampleSIMDHorizontalConv.c"
/* again, with 16-bit results for fast mode (ResampleFast.c) */
#define RESAMPLE_OUT16
#include "ResampleSIMDHorizontalConv.c"
#undef RESAMPLE_OUT16
#include "ResampleSIMDVerticalConv.c"
#include "ResampleSIMD32bpc.c"

static int
precompute_coeffs(
    int inSize,
    float in0,
    float in1,
    int outSize,
    struct filter *filterp,
    int **boundsp,
    double **kkp
) {
    double support, scale, filterscale;
    int xx, x, ksize, xmin, xmax;
    int *bounds;
    double *kk, *k;

    /* prepare for horizontal stretch */
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp->support * filterscale;

    /* maximum number of coeffs */
    ksize = (int)ceil(support) * 2 + 1;

    // check for overflow
    if (outSize > INT_MAX / (ksize * (int)sizeof(double))) {
        ImagingError_MemoryError();
        return 0;
    }

    /* coefficient buffer */
    /* malloc check ok, overflow checked above */
    kk = malloc(outSize * ksize * sizeof(double));
    if (!kk) {
        ImagingError_MemoryError();
        return 0;
    }

    /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
    bounds = malloc(outSize * 2 * sizeof(int));
    if (!bounds) {
        free(kk);
        ImagingError_MemoryError();
        return 0;
    }

    double inv_filterscale = 1.0 / filterscale;  // invariant over the loop
    for (xx = 0; xx < outSize; xx++) {
        double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;
        // Round the value
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        // Round the value
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) {
            xmax = inSize;
        }
        xmax -= xmin;
        k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = filterp->filter((x + xmin - center + 0.5) * inv_filterscale);
            k[x] = w;
            ww += w;
        }
        if (ww != 0.0) {
            for (x = 0; x < xmax; x++) {
                k[x] /= ww;
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < ksize; x++) {
            k[x] = 0;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    *boundsp = bounds;
    *kkp = kk;
    return ksize;
}

static int
normalize_coeffs_8bpc(int outSize, int ksize, double *prekk) {
    int x;
    INT16 *kk;
    int coefs_precision;
    double maxkk;

    // use the same buffer for normalized coefficients
    kk = (INT16 *)prekk;

    maxkk = prekk[0];
    for (x = 0; x < outSize * ksize; x++) {
        if (maxkk < prekk[x]) {
            maxkk = prekk[x];
        }
    }

    for (coefs_precision = 0; coefs_precision < PRECISION_BITS; coefs_precision += 1) {
        int next_value = (int)(0.5 + maxkk * (1 << (coefs_precision + 1)));
        // The next value will be outside of the range, so just stop
        if (next_value >= (1 << MAX_COEFS_PRECISION))
            break;
    }

    for (x = 0; x < outSize * ksize; x++) {
        if (prekk[x] < 0) {
            kk[x] = (int)(-0.5 + prekk[x] * (1 << coefs_precision));
        } else {
            kk[x] = (int)(0.5 + prekk[x] * (1 << coefs_precision));
        }
    }
    return coefs_precision;
}

/* For the resize of RGBA and LA images, which Pillow resizes premultiplied
   by alpha: the 8-bit passes premultiply the source rows they read (each
   row once, into `scratch`, four rows of the source's width) and remove the
   premultiplication from the rows they write, in place, while those are in
   the cache. The results are those of converting to RGBa or La, resizing
   and converting back. NULL members do nothing; the other passes ignore
   it. */
struct resample_alpha {
    ImagingShuffler premultiply;
    ImagingShuffler unpremultiply;
    UINT8 *scratch;
};

// Internal. Caller must guarantee imIn and imOut are distinct buffers.
static void
_ImagingResampleHorizontal_8bpc(
    Imaging imOut,
    Imaging imIn,
    int offset,
    int ksize,
    int *bounds,
    double *prekk,
    const struct resample_alpha *alpha
) {
    ImagingSectionCookie cookie;
    int ss0;
    int xx, yy, x, xmin, xmax;
    INT16 *k, *kk;
    int coefs_precision;

    MUTEX_LOCK(&imIn->mutex);

    // use the same buffer for normalized coefficients
    kk = (INT16 *)prekk;
    coefs_precision = normalize_coeffs_8bpc(imOut->xsize, ksize, prekk);

    // Sizes are invariant over the loop.
    int xsize = imOut->xsize, ysize = imOut->ysize;

    ImagingSectionEnter(&cookie);

    if (imIn->image8) {
        // Verify caller allocated distinct buffers.
        assert(imIn->image8 != imOut->image8);
        yy = 0;
#if defined(RESAMPLE_HORIZONTAL_1BAND_SIMD)
        for (; yy < ysize - 3; yy += 4) {
            ImagingResampleHorizontalConvolution1u4x(
                imOut->image8[yy],
                imOut->image8[yy + 1],
                imOut->image8[yy + 2],
                imOut->image8[yy + 3],
                imIn->image8[yy + offset],
                imIn->image8[yy + offset + 1],
                imIn->image8[yy + offset + 2],
                imIn->image8[yy + offset + 3],
                xsize,
                bounds,
                kk,
                ksize,
                coefs_precision
            );
        }
#endif
#if defined(RESAMPLE_HORIZONTAL_1BAND_SIMD)
        for (; yy < ysize; yy++) {
            ImagingResampleHorizontalConvolution1u(
                imOut->image8[yy],
                imIn->image8[yy + offset],
                xsize,
                bounds,
                kk,
                ksize,
                coefs_precision
            );
        }
#endif
        for (; yy < ysize; yy++) {
            // Restrict safe: imIn and imOut are distinct.
            UINT8 *restrict lineIn = imIn->image8[yy + offset];
            UINT8 *restrict lineOut = imOut->image8[yy];
            for (xx = 0; xx < xsize; xx++) {
                xmin = bounds[xx * 2 + 0];
                xmax = bounds[xx * 2 + 1];
                k = &kk[xx * ksize];
                ss0 = 1 << (coefs_precision - 1);
                for (x = 0; x < xmax; x++) {
                    ss0 += lineIn[x + xmin] * k[x];
                }
                lineOut[xx] = clip8(ss0, coefs_precision);
            }
        }
    } else if (imIn->type == IMAGING_TYPE_UINT8) {
        // Verify caller allocated distinct buffers.
        assert(imIn->image != imOut->image);
        yy = 0;
#if defined(RESAMPLE_HORIZONTAL_3BANDS_PORTABLE)
        if (imIn->bands == 3) {
            for (; yy < ysize; yy++) {
                ImagingResampleHorizontalConvolution8u3(
                    (UINT32 *)imOut->image32[yy],
                    (UINT32 *)imIn->image32[yy + offset],
                    xsize,
                    bounds,
                    kk,
                    ksize,
                    coefs_precision
                );
            }
        }
#endif
        // The source rows, or their premultiplied copies in the scratch rows
        UINT32 *in[4];
        int i, n;
        for (; yy < ysize; yy += n) {
            n = ysize - yy < 4 ? 1 : 4;
            for (i = 0; i < n; i++) {
                in[i] = (UINT32 *)imIn->image32[yy + offset + i];
                if (alpha && alpha->premultiply) {
                    UINT8 *row = alpha->scratch + (size_t)i * imIn->linesize;
                    alpha->premultiply(row, (UINT8 *)in[i], imIn->xsize);
                    in[i] = (UINT32 *)row;
                }
            }
            if (n == 4) {
                ImagingResampleHorizontalConvolution8u4x(
                    (UINT32 *)imOut->image32[yy],
                    (UINT32 *)imOut->image32[yy + 1],
                    (UINT32 *)imOut->image32[yy + 2],
                    (UINT32 *)imOut->image32[yy + 3],
                    in[0],
                    in[1],
                    in[2],
                    in[3],
                    xsize,
                    bounds,
                    kk,
                    ksize,
                    coefs_precision
                );
            } else {
                ImagingResampleHorizontalConvolution8u(
                    (UINT32 *)imOut->image32[yy],
                    in[0],
                    xsize,
                    bounds,
                    kk,
                    ksize,
                    coefs_precision
                );
            }
            if (alpha && alpha->unpremultiply) {
                for (i = 0; i < n; i++) {
                    UINT8 *row = (UINT8 *)imOut->image32[yy + i];
                    alpha->unpremultiply(row, row, xsize);
                }
            }
        }
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
}

// Internal. Caller must guarantee imIn and imOut are distinct buffers.
static void
_ImagingResampleVertical_8bpc(
    Imaging imOut,
    Imaging imIn,
    int offset,
    int ksize,
    int *bounds,
    double *prekk,
    const struct resample_alpha *alpha
) {
    ImagingSectionCookie cookie;
    int ss0;
    int xx, yy, y, ymin, ymax;
    INT16 *k, *kk;
    int coefs_precision;

    MUTEX_LOCK(&imIn->mutex);

    // use the same buffer for normalized coefficients
    kk = (INT16 *)prekk;
    coefs_precision = normalize_coeffs_8bpc(imOut->ysize, ksize, prekk);

    // Sizes are invariant over the loop.
    int xsize = imOut->xsize, ysize = imOut->ysize;

    ImagingSectionEnter(&cookie);

    if (imIn->image8) {
        // Verify caller allocated distinct buffers.
        assert(imIn->image8 != imOut->image8);
        // The vertical kernel works on bytes, so it handles 1-byte pixels
        // four at a time; the last xsize % 4 use the same arithmetic.
        int words = xsize / 4;
        for (yy = 0; yy < ysize; yy++) {
            // Restrict safe: imOut is a fresh allocation.
            UINT8 *restrict lineOut = imOut->image8[yy];
            k = &kk[yy * ksize];
            ymin = bounds[yy * 2 + 0];
            ymax = bounds[yy * 2 + 1];
            ImagingResampleVerticalConvolution8u(
                (UINT32 *)lineOut, imIn, ymin, ymax, k, coefs_precision, words
            );
            for (xx = words * 4; xx < xsize; xx++) {
                ss0 = 1 << (coefs_precision - 1);
                for (y = 0; y < ymax; y++) {
                    ss0 += ((UINT8)imIn->image8[y + ymin][xx]) * k[y];
                }
                lineOut[xx] = clip8(ss0, coefs_precision);
            }
        }
    } else if (imIn->type == IMAGING_TYPE_UINT8) {
        // Verify caller allocated distinct buffers.
        assert(imIn->image != imOut->image);
        for (yy = 0; yy < ysize; yy++) {
            k = &kk[yy * ksize];
            ymin = bounds[yy * 2 + 0];
            ymax = bounds[yy * 2 + 1];
            ImagingResampleVerticalConvolution8u(
                (UINT32 *)imOut->image32[yy],
                imIn,
                ymin,
                ymax,
                k,
                coefs_precision,
                xsize
            );
            if (alpha && alpha->unpremultiply) {
                UINT8 *row = (UINT8 *)imOut->image32[yy];
                alpha->unpremultiply(row, row, xsize);
            }
        }
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
}

// Internal. Caller must guarantee imIn and imOut are distinct buffers.
static void
_ImagingResampleHorizontal_16bpc(
    Imaging imOut,
    Imaging imIn,
    int offset,
    int ksize,
    int *bounds,
    double *kk,
    const struct resample_alpha *alpha
) {
    ImagingSectionCookie cookie;
    double ss;
    int xx, yy, x, xmin, xmax, ss_int;
    double *k;

    int bigendian = 0;
    if (
        imIn->mode == IMAGING_MODE_I_16N
#ifdef WORDS_BIGENDIAN
        || imIn->mode == IMAGING_MODE_I_16B
#endif
    ) {
        bigendian = 1;
    }

    MUTEX_LOCK(&imIn->mutex);

    // Sizes are invariant over the loop.
    int xsize = imOut->xsize, ysize = imOut->ysize;
    // Verify caller allocated distinct buffers. This function only uses image8.
    assert(imIn->image8 != imOut->image8);

    ImagingSectionEnter(&cookie);
#if defined(RESAMPLE_WIDE)
    if ((bigendian ? resample_horizontal_u16be : resample_horizontal_u16le)(
            imOut, imIn, offset, ksize, bounds, kk
        )) {
        MUTEX_UNLOCK(&imIn->mutex);
        ImagingSectionLeave(&cookie);
        return;
    }
#endif
    for (yy = 0; yy < ysize; yy++) {
        // Restrict safe: imIn and imOut are distinct.
        UINT8 *restrict lineIn = imIn->image8[yy + offset];
        UINT8 *restrict lineOut = imOut->image8[yy];
        for (xx = 0; xx < xsize; xx++) {
            xmin = bounds[xx * 2 + 0];
            xmax = bounds[xx * 2 + 1];
            k = &kk[xx * ksize];
            ss = 0.0;
            for (x = 0; x < xmax; x++) {
                ss += (lineIn[(x + xmin) * 2 + (bigendian ? 1 : 0)] +
                       (lineIn[(x + xmin) * 2 + (bigendian ? 0 : 1)] << 8)) *
                      k[x];
            }
            ss_int = CLIP16(ROUND_UP(ss));
            lineOut[xx * 2 + (bigendian ? 1 : 0)] = ss_int & 0xFF;
            lineOut[xx * 2 + (bigendian ? 0 : 1)] = ss_int >> 8;
        }
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
}

// Internal. Caller must guarantee imIn and imOut are distinct buffers.
static void
_ImagingResampleVertical_16bpc(
    Imaging imOut,
    Imaging imIn,
    int offset,
    int ksize,
    int *bounds,
    double *kk,
    const struct resample_alpha *alpha
) {
    ImagingSectionCookie cookie;
    double ss;
    int xx, yy, y, ymin, ymax, ss_int;
    double *k;

    int bigendian = 0;
    if (
        imIn->mode == IMAGING_MODE_I_16N
#ifdef WORDS_BIGENDIAN
        || imIn->mode == IMAGING_MODE_I_16B
#endif
    ) {
        bigendian = 1;
    }

    MUTEX_LOCK(&imIn->mutex);

    // Sizes are invariant over the loop.
    int xsize = imOut->xsize, ysize = imOut->ysize;
    // Verify caller allocated distinct buffers. This function only uses image8.
    assert(imIn->image8 != imOut->image8);

    ImagingSectionEnter(&cookie);
#if defined(RESAMPLE_WIDE)
    {
        (bigendian ? resample_vertical_u16be
                   : resample_vertical_u16le)(imOut, imIn, ksize, bounds, kk);
        MUTEX_UNLOCK(&imIn->mutex);
        ImagingSectionLeave(&cookie);
        return;
    }
#endif

    for (yy = 0; yy < ysize; yy++) {
        // imOut is a fresh allocation, so its row can be hoisted/restrict.
        UINT8 *restrict lineOut = imOut->image8[yy];
        ymin = bounds[yy * 2 + 0];
        ymax = bounds[yy * 2 + 1];
        k = &kk[yy * ksize];
        for (xx = 0; xx < xsize; xx++) {
            ss = 0.0;
            for (y = 0; y < ymax; y++) {
                ss += (imIn->image8[y + ymin][xx * 2 + (bigendian ? 1 : 0)] +
                       (imIn->image8[y + ymin][xx * 2 + (bigendian ? 0 : 1)] << 8)) *
                      k[y];
            }
            ss_int = CLIP16(ROUND_UP(ss));
            lineOut[xx * 2 + (bigendian ? 1 : 0)] = ss_int & 0xFF;
            lineOut[xx * 2 + (bigendian ? 0 : 1)] = ss_int >> 8;
        }
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
}

// Internal. Caller must guarantee imIn and imOut are distinct buffers.
static void
_ImagingResampleHorizontal_32bpc(
    Imaging imOut,
    Imaging imIn,
    int offset,
    int ksize,
    int *bounds,
    double *kk,
    const struct resample_alpha *alpha
) {
    ImagingSectionCookie cookie;
    double ss;
    int xx, yy, x, xmin, xmax;
    double *k;

    MUTEX_LOCK(&imIn->mutex);

    // Sizes are invariant over the loop.
    int xsize = imOut->xsize, ysize = imOut->ysize;
    // Verify caller allocated distinct buffers. This function only uses image32.
    assert(imIn->image32 != imOut->image32);

    ImagingSectionEnter(&cookie);
#if defined(RESAMPLE_WIDE)
    if (imIn->type == IMAGING_TYPE_FLOAT32
            ? resample_horizontal_f32(imOut, imIn, offset, ksize, bounds, kk)
            : imIn->type == IMAGING_TYPE_INT32 &&
                  resample_horizontal_i32(imOut, imIn, offset, ksize, bounds, kk)) {
        MUTEX_UNLOCK(&imIn->mutex);
        ImagingSectionLeave(&cookie);
        return;
    }
#endif

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            for (yy = 0; yy < ysize; yy++) {
                // Restrict safe: imIn and imOut are distinct.
                INT32 *restrict lineIn = imIn->image32[yy + offset];
                INT32 *restrict lineOut = imOut->image32[yy];
                for (xx = 0; xx < xsize; xx++) {
                    xmin = bounds[xx * 2 + 0];
                    xmax = bounds[xx * 2 + 1];
                    k = &kk[xx * ksize];
                    ss = 0.0;
                    for (x = 0; x < xmax; x++) {
                        ss += lineIn[x + xmin] * k[x];
                    }
                    lineOut[xx] = ROUND_UP(ss);
                }
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            for (yy = 0; yy < ysize; yy++) {
                // Restrict safe: imIn and imOut are distinct.
                FLOAT32 *restrict lineIn = (FLOAT32 *)imIn->image32[yy + offset];
                FLOAT32 *restrict lineOut = (FLOAT32 *)imOut->image32[yy];
                for (xx = 0; xx < xsize; xx++) {
                    xmin = bounds[xx * 2 + 0];
                    xmax = bounds[xx * 2 + 1];
                    k = &kk[xx * ksize];
                    ss = 0.0;
                    for (x = 0; x < xmax; x++) {
                        ss += lineIn[x + xmin] * k[x];
                    }
                    lineOut[xx] = ss;
                }
            }
            break;
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
}

// Internal. Caller must guarantee imIn and imOut are distinct buffers.
static void
_ImagingResampleVertical_32bpc(
    Imaging imOut,
    Imaging imIn,
    int offset,
    int ksize,
    int *bounds,
    double *kk,
    const struct resample_alpha *alpha
) {
    ImagingSectionCookie cookie;
    double ss;
    int xx, yy, y, ymin, ymax;
    double *k;

    MUTEX_LOCK(&imIn->mutex);

    // Sizes are invariant over the loop.
    int xsize = imOut->xsize, ysize = imOut->ysize;
    // Verify caller allocated distinct buffers. This function only uses image32.
    assert(imIn->image32 != imOut->image32);

    ImagingSectionEnter(&cookie);
#if defined(RESAMPLE_WIDE)
    if (imIn->type == IMAGING_TYPE_FLOAT32 || imIn->type == IMAGING_TYPE_INT32) {
        (imIn->type == IMAGING_TYPE_FLOAT32
             ? resample_vertical_f32
             : resample_vertical_i32)(imOut, imIn, ksize, bounds, kk);
        MUTEX_UNLOCK(&imIn->mutex);
        ImagingSectionLeave(&cookie);
        return;
    }
#endif

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            for (yy = 0; yy < ysize; yy++) {
                // imOut is a fresh allocation, so its row can be hoisted.
                INT32 *restrict lineOut = imOut->image32[yy];
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                k = &kk[yy * ksize];
                for (xx = 0; xx < xsize; xx++) {
                    ss = 0.0;
                    for (y = 0; y < ymax; y++) {
                        ss += IMAGING_PIXEL_I(imIn, xx, y + ymin) * k[y];
                    }
                    lineOut[xx] = ROUND_UP(ss);
                }
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            for (yy = 0; yy < ysize; yy++) {
                // imOut is a fresh allocation, so its row can be hoisted.
                FLOAT32 *restrict lineOut = (FLOAT32 *)imOut->image32[yy];
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                k = &kk[yy * ksize];
                for (xx = 0; xx < xsize; xx++) {
                    ss = 0.0;
                    for (y = 0; y < ymax; y++) {
                        ss += IMAGING_PIXEL_F(imIn, xx, y + ymin) * k[y];
                    }
                    lineOut[xx] = ss;
                }
            }
            break;
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
}

#include "ResampleFast.c"

typedef void (*ResampleFunction)(
    Imaging imOut,
    Imaging imIn,
    int offset,
    int ksize,
    int *bounds,
    double *kk,
    const struct resample_alpha *alpha
);

static Imaging
ImagingResampleInner(
    Imaging imIn,
    int xsize,
    int ysize,
    struct filter *filterp,
    float box[4],
    ResampleFunction ResampleHorizontal,
    ResampleFunction ResampleVertical,
    int alpha
);

/* Premultiplies an RGBA or LA image by alpha and removes that again, in
   place. Returns -1 when out of memory. */
static int
premultiply_round_trip(Imaging im) {
    ImagingShuffler premultiply = ImagingFindPremultiply(im->mode);
    ImagingShuffler unpremultiply = ImagingFindUnpremultiply(im->mode);
    UINT8 *row = malloc(im->linesize);
    int y;
    if (!row && im->linesize) {
        ImagingError_MemoryError();
        return -1;
    }
    for (y = 0; y < im->ysize; y++) {
        premultiply(row, (UINT8 *)im->image32[y], im->xsize);
        unpremultiply((UINT8 *)im->image32[y], row, im->xsize);
    }
    free(row);
    return 0;
}

static Imaging
resample(Imaging imIn, int xsize, int ysize, int filter, float box[4], int alpha) {
    struct filter *filterp;
    ResampleFunction ResampleHorizontal;
    ResampleFunction ResampleVertical;

    if (imIn->mode == IMAGING_MODE_P || imIn->mode == IMAGING_MODE_1) {
        return (Imaging)ImagingError_ModeError();
    }

    if (imIn->type == IMAGING_TYPE_I16) {
        ResampleHorizontal = _ImagingResampleHorizontal_16bpc;
        ResampleVertical = _ImagingResampleVertical_16bpc;
    } else if (imIn->image8) {
        ResampleHorizontal = _ImagingResampleHorizontal_8bpc;
        ResampleVertical = _ImagingResampleVertical_8bpc;
    } else {
        switch (imIn->type) {
            case IMAGING_TYPE_UINT8:
                ResampleHorizontal = _ImagingResampleHorizontal_8bpc;
                ResampleVertical = _ImagingResampleVertical_8bpc;
                break;
            case IMAGING_TYPE_INT32:
            case IMAGING_TYPE_FLOAT32:
                ResampleHorizontal = _ImagingResampleHorizontal_32bpc;
                ResampleVertical = _ImagingResampleVertical_32bpc;
                break;
            default:
                return (Imaging)ImagingError_ModeError();
        }
    }

    /* check filter */
    switch (filter) {
        case IMAGING_TRANSFORM_BOX:
            filterp = &BOX;
            break;
        case IMAGING_TRANSFORM_BILINEAR:
            filterp = &BILINEAR;
            break;
        case IMAGING_TRANSFORM_HAMMING:
            filterp = &HAMMING;
            break;
        case IMAGING_TRANSFORM_BICUBIC:
            filterp = &BICUBIC;
            break;
        case IMAGING_TRANSFORM_LANCZOS:
            filterp = &LANCZOS;
            break;
        case IMAGING_TRANSFORM_MKS2013:
            filterp = &MKS2013;
            break;
        case IMAGING_TRANSFORM_MKS2021:
            filterp = &MKS2021;
            break;
        default:
            return (Imaging)ImagingError_ValueError("unsupported resampling filter");
    }

    return ImagingResampleInner(
        imIn, xsize, ysize, filterp, box, ResampleHorizontal, ResampleVertical, alpha
    );
}

Imaging
ImagingResample(Imaging imIn, int xsize, int ysize, int filter, float box[4]) {
    return resample(imIn, xsize, ysize, filter, box, 0);
}

/* Resizes an RGBA or LA image as Pillow does, premultiplied by alpha, with
   the result in the same mode: the same pixels as converting to RGBa or La,
   resizing and converting back, without the two whole-image conversions. */
Imaging
ImagingResampleAlpha(Imaging imIn, int xsize, int ysize, int filter, float box[4]) {
    Imaging imOut;
    if (!ImagingFindPremultiply(imIn->mode) || !imIn->image32) {
        return (Imaging)ImagingError_ModeError();
    }
    // As the resize binding does: an integer box of the output's size is a
    // crop, premultiplied and back
    if (box[0] - (int)box[0] == 0 && box[2] - box[0] == xsize &&
        box[1] - (int)box[1] == 0 && box[3] - box[1] == ysize) {
        imOut = ImagingCrop(imIn, box[0], box[1], box[2], box[3]);
        if (imOut && premultiply_round_trip(imOut)) {
            ImagingDelete(imOut);
            return NULL;
        }
        return imOut;
    }
    return resample(imIn, xsize, ysize, filter, box, 1);
}

static Imaging
ImagingResampleInner(
    Imaging imIn,
    int xsize,
    int ysize,
    struct filter *filterp,
    float box[4],
    ResampleFunction ResampleHorizontal,
    ResampleFunction ResampleVertical,
    int alpha
) {
    Imaging imTemp = NULL;
    Imaging imOut = NULL;
    struct resample_alpha horiz_alpha = {NULL, NULL, NULL};
    struct resample_alpha vert_alpha = {NULL, NULL, NULL};
    ImagingShuffler premultiply = NULL, unpremultiply = NULL;

    int i, second_pass, need_horizontal, need_vertical, error = 0;
    int fast = ImagingGetFast() & IMAGING_FAST_ALL, fast_done = 0;
    int ybox_first, ybox_last;
    int ksize_horiz = 0, ksize_vert = 0;
    int *bounds_horiz, *bounds_vert;
    double *kk_horiz, *kk_vert;

    need_horizontal = xsize != imIn->xsize || box[0] || box[2] != xsize;
    need_vertical = ysize != imIn->ysize || box[1] || box[3] != ysize;

    // If height is being scaled down more than twice the amount the width is,
    // run the vertical pass first, to make the horizontal pass faster
    int horizontal_first =
        !((imIn->ysize - ysize) > 0 &&
          (imIn->ysize - ysize) > (imIn->xsize - xsize) * 2);

    if ((need_horizontal && horizontal_first) || need_vertical) {
        ksize_vert = precompute_coeffs(
            imIn->ysize, box[1], box[3], ysize, filterp, &bounds_vert, &kk_vert
        );
        if (!ksize_vert) {
            return NULL;
        }
    }
    if (need_horizontal) {
        if (horizontal_first) {
            // First used row in the source image
            ybox_first = bounds_vert[0];
            // Last used row in the source image
            ybox_last = bounds_vert[ysize * 2 - 2] + bounds_vert[ysize * 2 - 1];

            // Shift bounds for vertical pass
            if (ybox_first != 0) {
                for (i = 0; i < ysize; i++) {
                    bounds_vert[i * 2] -= ybox_first;
                }
            }
        }

        ksize_horiz = precompute_coeffs(
            imIn->xsize, box[0], box[2], xsize, filterp, &bounds_horiz, &kk_horiz
        );
        if (!ksize_horiz) {
            error = 1;
            goto end;
        }
    }

#define PASS(function, w, h, offset, ksize, bounds, kk, alphap) \
    second_pass = imTemp != NULL;                               \
    imTemp = ImagingNewDirty(imIn->mode, w, h);                 \
    if (!imTemp) {                                              \
        if (second_pass) {                                      \
            ImagingDelete(imIn);                                \
        }                                                       \
        error = 1;                                              \
        goto end;                                               \
    }                                                           \
    function(imTemp, imIn, offset, ksize, bounds, kk, alphap);  \
    if (second_pass) {                                          \
        ImagingDelete(imIn);                                    \
    }                                                           \
    imIn = imTemp;

    /* Fast mode: 8-bit images with 4-byte pixels, the passes fused and the
       vertical one in 16-bit lanes (ResampleFast.c), unless its sums could
       overflow */
    if (fast && horizontal_first && need_vertical && imIn->image32 &&
        imIn->type == IMAGING_TYPE_UINT8 && imIn->pixelsize == 4 &&
        ResampleHorizontal == _ImagingResampleHorizontal_8bpc) {
        fast_done = resample_fast8(
            &imOut,
            imIn,
            xsize,
            ysize,
            need_horizontal,
            ksize_horiz,
            bounds_horiz,
            kk_horiz,
            ksize_vert,
            bounds_vert,
            kk_vert,
            need_horizontal ? ybox_first : 0,
            alpha ? ImagingFindPremultiply(imIn->mode) : NULL,
            alpha ? ImagingFindUnpremultiply(imIn->mode) : NULL
        );
        if (fast_done) {
            goto end;
        }
    }

    if (alpha) {
        premultiply = ImagingFindPremultiply(imIn->mode);
        unpremultiply = ImagingFindUnpremultiply(imIn->mode);
        if (horizontal_first && need_horizontal) {
            // The horizontal pass premultiplies each source row it reads
            horiz_alpha.premultiply = premultiply;
            horiz_alpha.scratch = malloc(4 * (size_t)imIn->linesize);
            if (!horiz_alpha.scratch && imIn->linesize) {
                ImagingError_MemoryError();
                error = 1;
                goto end;
            }
        } else if (need_vertical) {
            // The vertical pass reads each row several times, so it reads a
            // premultiplied copy, which the first pass then deletes
            imTemp = ImagingNewDirty(imIn->mode, imIn->xsize, imIn->ysize);
            if (!imTemp) {
                error = 1;
                goto end;
            }
            for (i = 0; i < imIn->ysize; i++) {
                premultiply(
                    (UINT8 *)imTemp->image32[i], (UINT8 *)imIn->image32[i], imIn->xsize
                );
            }
            imIn = imTemp;
        }
        // The last pass removes the premultiplication from its rows
        if (horizontal_first ? need_vertical : !need_horizontal) {
            vert_alpha.unpremultiply = unpremultiply;
        } else {
            horiz_alpha.unpremultiply = unpremultiply;
        }
    }

    if (horizontal_first) {
        if (need_horizontal) {
            PASS(
                ResampleHorizontal,
                xsize,
                ybox_last - ybox_first,
                ybox_first,
                ksize_horiz,
                bounds_horiz,
                kk_horiz,
                &horiz_alpha
            );
        }
        if (need_vertical) {
            PASS(
                ResampleVertical,
                xsize,
                ysize,
                0,
                ksize_vert,
                bounds_vert,
                kk_vert,
                &vert_alpha
            );
        }
    } else {
        if (need_vertical) {
            PASS(
                ResampleVertical,
                imIn->xsize,
                ysize,
                0,
                ksize_vert,
                bounds_vert,
                kk_vert,
                &vert_alpha
            );
        }
        if (need_horizontal) {
            PASS(
                ResampleHorizontal,
                xsize,
                ysize,
                0,
                ksize_horiz,
                bounds_horiz,
                kk_horiz,
                &horiz_alpha
            );
        }
    }

end:
    free(horiz_alpha.scratch);
    if (ksize_horiz) {
        free(bounds_horiz);
        free(kk_horiz);
    }
    if (ksize_vert) {
        free(bounds_vert);
        free(kk_vert);
    }
    if (fast_done) {
        return imOut;
    }
    if (error) {
        return NULL;
    }

    if (imTemp) {
        imOut = imTemp;
    } else {
        // none of the previous steps are performed, copying
        imOut = ImagingCopy(imIn);
        // premultiplied and back, as resizing an RGBa or La copy would be
        if (imOut && alpha && premultiply_round_trip(imOut)) {
            ImagingDelete(imOut);
            return NULL;
        }
    }

    return imOut;
}

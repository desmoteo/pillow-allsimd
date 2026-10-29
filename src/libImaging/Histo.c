/*
 * The Python Imaging Library
 * $Id$
 *
 * histogram support
 *
 * history:
 * 1995-06-15 fl   Created.
 * 1996-04-05 fl   Fixed histogram for multiband images.
 * 1997-02-23 fl   Added mask support
 * 1998-07-01 fl   Added basic 32-bit float/integer support
 *
 * Copyright (c) 1997-2003 by Secret Labs AB.
 * Copyright (c) 1995-2003 by Fredrik Lundh.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

/* HISTOGRAM */
/* --------------------------------------------------------------------
 * Take a histogram of an image. Returns a histogram object containing
 * 256 slots per band in the input image.
 */

void
ImagingHistogramDelete(ImagingHistogram h) {
    if (h) {
        if (h->histogram) {
            free(h->histogram);
        }
        free(h);
    }
}

ImagingHistogram
ImagingHistogramNew(Imaging im) {
    ImagingHistogram h;

    /* Create histogram descriptor */
    h = calloc(1, sizeof(struct ImagingHistogramInstance));
    if (!h) {
        return (ImagingHistogram)ImagingError_MemoryError();
    }

    h->mode = im->mode;
    h->bands = im->bands;

    h->histogram = calloc(im->pixelsize, 256 * sizeof(long));
    if (!h->histogram) {
        free(h);
        return (ImagingHistogram)ImagingError_MemoryError();
    }

    return h;
}

#if defined(__aarch64__) || defined(_M_ARM64)
/* Counts the bytes of an unmasked 8-bit image into histogram, with 13.0's
   loops: on arm64 (Cortex-A78AE, GCC 12 and 14) they are 1.1x faster than
   the four tables used elsewhere. Returns 1. */
static int
histogram_u8(long *histogram, Imaging im) {
    int xsize = im->xsize, ysize = im->ysize;
    if (im->image8) {
        for (int y = 0; y < ysize; y++) {
            const UINT8 *in = im->image8[y];
            for (int x = 0; x < xsize; x++) {
                histogram[in[x]]++;
            }
        }
    } else if (im->bands == 2) {
        for (int y = 0; y < ysize; y++) {
            const UINT8 *in = (UINT8 *)im->image[y];
            for (int x = 0; x < xsize; x++, in += 4) {
                histogram[in[0]]++;
                histogram[in[3] + 256]++;
            }
        }
    } else if (im->bands == 3) {
        for (int y = 0; y < ysize; y++) {
            const UINT8 *in = (UINT8 *)im->image[y];
            for (int x = 0; x < xsize; x++, in += 4) {
                histogram[in[0]]++;
                histogram[in[1] + 256]++;
                histogram[in[2] + 512]++;
            }
        }
    } else {
        for (int y = 0; y < ysize; y++) {
            const UINT8 *in = (UINT8 *)im->image[y];
            for (int x = 0; x < xsize; x++, in += 4) {
                histogram[in[0]]++;
                histogram[in[1] + 256]++;
                histogram[in[2] + 512]++;
                histogram[in[3] + 768]++;
            }
        }
    }
    return 1;
}
#else
/* Counts the bytes of an unmasked 8-bit image into histogram. Neighbouring
   pixels go to different tables, which are added up at the end, so that
   increments of the same bin don't wait for each other. Returns 0 if out
   of memory. */
#define HISTOGRAM_TABLES 4
static int
histogram_u8(long *histogram, Imaging im) {
    int bins = im->pixelsize * 256;
    int x, y, i, k;
    long *t = calloc((size_t)HISTOGRAM_TABLES * bins, sizeof(long));
    if (!t) {
        return 0;
    }
    long *t0 = t, *t1 = t + bins, *t2 = t + 2 * bins, *t3 = t + 3 * bins;

    for (y = 0; y < im->ysize; y++) {
        const UINT8 *in = (UINT8 *)im->image[y];
        x = 0;
        if (im->image8) {
            for (; x < im->xsize - 3; x += 4) {
                t0[in[x]]++;
                t1[in[x + 1]]++;
                t2[in[x + 2]]++;
                t3[in[x + 3]]++;
            }
            for (; x < im->xsize; x++) {
                t0[in[x]]++;
            }
        } else {
            /* Bands 2 and 3 skip the padding bytes; nothing reads their
               counts. */
            int b1 = im->bands == 2 ? 3 : 1;
            int nb = im->bands == 2 ? 2 : im->bands;
#define HISTOGRAM_PIXEL(t, p)  \
    {                          \
        t[(p)[0]]++;           \
        t[(p)[b1] + 256]++;    \
        if (nb > 2) {          \
            t[(p)[2] + 512]++; \
        }                      \
        if (nb > 3) {          \
            t[(p)[3] + 768]++; \
        }                      \
    }
            for (; x < im->xsize - 3; x += 4, in += 16) {
                HISTOGRAM_PIXEL(t0, in)
                HISTOGRAM_PIXEL(t1, in + 4)
                HISTOGRAM_PIXEL(t2, in + 8)
                HISTOGRAM_PIXEL(t3, in + 12)
            }
            for (; x < im->xsize; x++, in += 4) {
                HISTOGRAM_PIXEL(t0, in)
            }
#undef HISTOGRAM_PIXEL
        }
    }
    for (k = 0; k < HISTOGRAM_TABLES; k++) {
        for (i = 0; i < bins; i++) {
            histogram[i] += t[k * bins + i];
        }
    }
    free(t);
    return 1;
}
#endif

/**
 * Compute a histogram of `im`'s value distribution,
 * optionally restricted to imMask.
 *
 * Contract: Both im and imMask are read-only.
 */
ImagingHistogram
ImagingGetHistogram(Imaging im, Imaging imMask, void *minmax) {
    ImagingSectionCookie cookie;
    ImagingHistogram h;
    INT32 imin, imax;
    FLOAT32 fmin, fmax, scale;

    if (!im) {
        return ImagingError_ModeError();
    }

    int xsize = im->xsize, ysize = im->ysize;
    if (imMask) {
        /* Validate mask */
        if (xsize != imMask->xsize || ysize != imMask->ysize) {
            return ImagingError_Mismatch();
        }
        if (imMask->mode != IMAGING_MODE_1 && imMask->mode != IMAGING_MODE_L) {
            return ImagingError_ValueError("bad transparency mask");
        }
    }

    h = ImagingHistogramNew(im);
    if (!h) {
        return NULL;
    }

    // restrict safe: im and imMask are both read-only here
    //                (they may even be the same image).
    // histogram is not restrict: with it, MSVC 14.51 (/O2 /GL) miscompiled
    // this function once it also calls histogram_u8, and the F histogram
    // returned NULL without an exception.

    long *histogram = h->histogram;

    if (imMask) {
        /* mask */
        if (im->image8) {
            ImagingSectionEnter(&cookie);
            MUTEX_LOCK(&im->mutex);
            for (int y = 0; y < ysize; y++) {
                UINT8 *restrict in = im->image8[y];
                UINT8 *restrict mask = imMask->image8[y];
                for (int x = 0; x < xsize; x++) {
                    if (mask[x] != 0) {
                        histogram[in[x]]++;
                    }
                }
            }
            MUTEX_UNLOCK(&im->mutex);
            ImagingSectionLeave(&cookie);
        } else { /* yes, we need the braces. C isn't Python! */
            if (im->type != IMAGING_TYPE_UINT8) {
                ImagingHistogramDelete(h);
                return ImagingError_ModeError();
            }
            ImagingSectionEnter(&cookie);
            MUTEX_LOCK(&im->mutex);
            for (int y = 0; y < ysize; y++) {
                UINT8 *restrict in = (UINT8 *)im->image32[y];
                UINT8 *restrict mask = imMask->image8[y];
                for (int x = 0; x < xsize; x++, in += 4) {
                    if (mask[x] != 0) {
                        histogram[*in]++;
                        if (im->bands == 2) {
                            histogram[*(in + 3) + 256]++;
                        } else {
                            histogram[*(in + 1) + 256]++;
                            histogram[*(in + 2) + 512]++;
                            histogram[*(in + 3) + 768]++;
                        }
                    }
                }
            }
            MUTEX_UNLOCK(&im->mutex);
            ImagingSectionLeave(&cookie);
        }
    } else {
        /* mask not given; process pixels in image */
        if (im->image8 || im->type == IMAGING_TYPE_UINT8) {
            ImagingSectionEnter(&cookie);
            MUTEX_LOCK(&im->mutex);
            int ok = histogram_u8(histogram, im);
            MUTEX_UNLOCK(&im->mutex);
            ImagingSectionLeave(&cookie);
            if (!ok) {
                ImagingHistogramDelete(h);
                return ImagingError_MemoryError();
            }
        } else {
            switch (im->type) {
                case IMAGING_TYPE_INT32:
                    if (!minmax) {
                        ImagingHistogramDelete(h);
                        return ImagingError_ValueError("min/max not given");
                    }
                    if (!xsize || !ysize) {
                        break;
                    }
                    memcpy(&imin, minmax, sizeof(imin));
                    memcpy(&imax, ((char *)minmax) + sizeof(imin), sizeof(imax));
                    if (imin >= imax) {
                        break;
                    }
                    ImagingSectionEnter(&cookie);
                    MUTEX_LOCK(&im->mutex);
                    scale = 255.0F / (imax - imin);
                    for (int y = 0; y < ysize; y++) {
                        INT32 *restrict in = im->image32[y];
                        for (int x = 0; x < xsize; x++) {
                            int i = (int)(((*in++) - imin) * scale);
                            if (i >= 0 && i < 256) {
                                histogram[i]++;
                            }
                        }
                    }
                    MUTEX_UNLOCK(&im->mutex);
                    ImagingSectionLeave(&cookie);
                    break;
                case IMAGING_TYPE_FLOAT32:
                    if (!minmax) {
                        ImagingHistogramDelete(h);
                        return ImagingError_ValueError("min/max not given");
                    }
                    if (!xsize || !ysize) {
                        break;
                    }
                    memcpy(&fmin, minmax, sizeof(fmin));
                    memcpy(&fmax, ((char *)minmax) + sizeof(fmin), sizeof(fmax));
                    if (fmin >= fmax) {
                        break;
                    }
                    ImagingSectionEnter(&cookie);
                    MUTEX_LOCK(&im->mutex);
                    scale = 255.0F / (fmax - fmin);
                    for (int y = 0; y < ysize; y++) {
                        FLOAT32 *restrict in = (FLOAT32 *)im->image32[y];
                        for (int x = 0; x < xsize; x++) {
                            int i = (int)(((*in++) - fmin) * scale);
                            if (i >= 0 && i < 256) {
                                histogram[i]++;
                            }
                        }
                    }
                    MUTEX_UNLOCK(&im->mutex);
                    ImagingSectionLeave(&cookie);
                    break;
            }
        }
    }

    return h;
}

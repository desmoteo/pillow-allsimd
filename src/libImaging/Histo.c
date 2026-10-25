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

ImagingHistogram
ImagingGetHistogram(Imaging im, Imaging imMask, void *minmax) {
    ImagingSectionCookie cookie;
    int x, y, i;
    ImagingHistogram h;
    INT32 imin, imax;
    FLOAT32 fmin, fmax, scale;

    if (!im) {
        return ImagingError_ModeError();
    }

    if (imMask) {
        /* Validate mask */
        if (im->xsize != imMask->xsize || im->ysize != imMask->ysize) {
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

    if (imMask) {
        /* mask */
        if (im->image8) {
            ImagingSectionEnter(&cookie);
            for (y = 0; y < im->ysize; y++) {
                for (x = 0; x < im->xsize; x++) {
                    if (imMask->image8[y][x] != 0) {
                        h->histogram[im->image8[y][x]]++;
                    }
                }
            }
            ImagingSectionLeave(&cookie);
        } else { /* yes, we need the braces. C isn't Python! */
            if (im->type != IMAGING_TYPE_UINT8) {
                ImagingHistogramDelete(h);
                return ImagingError_ModeError();
            }
            ImagingSectionEnter(&cookie);
            for (y = 0; y < im->ysize; y++) {
                UINT8 *in = (UINT8 *)im->image32[y];
                for (x = 0; x < im->xsize; x++, in += 4) {
                    if (imMask->image8[y][x] != 0) {
                        h->histogram[*in]++;
                        if (im->bands == 2) {
                            h->histogram[*(in + 3) + 256]++;
                        } else {
                            h->histogram[*(in + 1) + 256]++;
                            h->histogram[*(in + 2) + 512]++;
                            h->histogram[*(in + 3) + 768]++;
                        }
                    }
                }
            }
            ImagingSectionLeave(&cookie);
        }
    } else {
        /* mask not given; process pixels in image */
        if (im->image8 || im->type == IMAGING_TYPE_UINT8) {
            ImagingSectionEnter(&cookie);
            if (!histogram_u8(h->histogram, im)) {
                ImagingSectionLeave(&cookie);
                ImagingHistogramDelete(h);
                return ImagingError_MemoryError();
            }
            ImagingSectionLeave(&cookie);
        } else {
            switch (im->type) {
                case IMAGING_TYPE_INT32:
                    if (!minmax) {
                        ImagingHistogramDelete(h);
                        return ImagingError_ValueError("min/max not given");
                    }
                    if (!im->xsize || !im->ysize) {
                        break;
                    }
                    memcpy(&imin, minmax, sizeof(imin));
                    memcpy(&imax, ((char *)minmax) + sizeof(imin), sizeof(imax));
                    if (imin >= imax) {
                        break;
                    }
                    ImagingSectionEnter(&cookie);
                    scale = 255.0F / (imax - imin);
                    for (y = 0; y < im->ysize; y++) {
                        INT32 *in = im->image32[y];
                        for (x = 0; x < im->xsize; x++) {
                            i = (int)(((*in++) - imin) * scale);
                            if (i >= 0 && i < 256) {
                                h->histogram[i]++;
                            }
                        }
                    }
                    ImagingSectionLeave(&cookie);
                    break;
                case IMAGING_TYPE_FLOAT32:
                    if (!minmax) {
                        ImagingHistogramDelete(h);
                        return ImagingError_ValueError("min/max not given");
                    }
                    if (!im->xsize || !im->ysize) {
                        break;
                    }
                    memcpy(&fmin, minmax, sizeof(fmin));
                    memcpy(&fmax, ((char *)minmax) + sizeof(fmin), sizeof(fmax));
                    if (fmin >= fmax) {
                        break;
                    }
                    ImagingSectionEnter(&cookie);
                    scale = 255.0F / (fmax - fmin);
                    for (y = 0; y < im->ysize; y++) {
                        FLOAT32 *in = (FLOAT32 *)im->image32[y];
                        for (x = 0; x < im->xsize; x++) {
                            i = (int)(((*in++) - fmin) * scale);
                            if (i >= 0 && i < 256) {
                                h->histogram[i]++;
                            }
                        }
                    }
                    ImagingSectionLeave(&cookie);
                    break;
            }
        }
    }

    return h;
}

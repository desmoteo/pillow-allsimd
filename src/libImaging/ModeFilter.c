/*
 * The Python Imaging Library
 * $Id$
 *
 * mode filter
 *
 * history:
 * 2002-06-08 fl    Created (based on code from IFUNC95)
 * 2004-10-05 fl    Rewritten; use a simpler brute-force algorithm
 * 2026-10-27       Histogram updated by columns; small windows check only
 *                  their own values
 *
 * Copyright (c) Secret Labs AB 2002-2004.  All rights reserved.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

/* Adds sign (1 or -1) to the histogram counts of column x, rows y0 to y1. */
static inline void
mode_count_column(int *histogram, Imaging im, int x, int y0, int y1, int sign) {
    int yy;
    for (yy = y0; yy <= y1; yy++) {
        histogram[IMAGING_PIXEL_L(im, x, yy)] += sign;
    }
}

/* The highest count; a plain maximum, which compilers vectorise. */
static inline int
mode_max_count(const int *histogram) {
    int i, maxcount = 0;
    for (i = 0; i < 256; i++) {
        maxcount = histogram[i] > maxcount ? histogram[i] : maxcount;
    }
    return maxcount;
}

/* Windows with fewer values check those; larger ones scan the counts. */
#define MODE_WINDOW_LIMIT 16

Imaging
ImagingModeFilter(Imaging im, int size) {
    Imaging imOut;
    ImagingSectionCookie cookie;
    int x, y, i;
    int xx, yy;
    int x0, x1, y0, y1;
    int maxcount;
    UINT8 maxpixel;
    int histogram[256];

    if (!im || im->bands != 1 || im->type != IMAGING_TYPE_UINT8) {
        return (Imaging)ImagingError_ModeError();
    }

    imOut = ImagingNewDirty(im->mode, im->xsize, im->ysize);
    if (!imOut) {
        return NULL;
    }

    size = size / 2;

    ImagingSectionEnter(&cookie);
    for (y = 0; y < imOut->ysize; y++) {
        UINT8 *out = &IMAGING_PIXEL_L(imOut, 0, y);
        y0 = y - size < 0 ? 0 : y - size;
        y1 = y + size >= imOut->ysize ? imOut->ysize - 1 : y + size;

        /* The histogram of the window, updated by one column on each step. */
        memset(histogram, 0, sizeof(histogram));
        for (xx = 0; xx < size && xx < imOut->xsize; xx++) {
            mode_count_column(histogram, im, xx, y0, y1, 1);
        }

        for (x = 0; x < imOut->xsize; x++) {
            if (x - size - 1 >= 0) {
                mode_count_column(histogram, im, x - size - 1, y0, y1, -1);
            }
            if (x + size < imOut->xsize) {
                mode_count_column(histogram, im, x + size, y0, y1, 1);
            }
            x0 = x - size < 0 ? 0 : x - size;
            x1 = x + size >= imOut->xsize ? imOut->xsize - 1 : x + size;

            /* The most frequent value, the lowest one on a tie. Only values
               in the window can win, so small windows look at those
               instead of all 256 counts. */
            if ((x1 - x0 + 1) * (y1 - y0 + 1) < MODE_WINDOW_LIMIT) {
                maxpixel = 255;
                maxcount = 0;
                for (yy = y0; yy <= y1; yy++) {
                    UINT8 *in = &IMAGING_PIXEL_L(im, 0, yy);
                    for (xx = x0; xx <= x1; xx++) {
                        UINT8 v = in[xx];
                        if (histogram[v] > maxcount ||
                            (histogram[v] == maxcount && v < maxpixel)) {
                            maxcount = histogram[v];
                            maxpixel = v;
                        }
                    }
                }
            } else {
                maxcount = mode_max_count(histogram);
                for (i = 0; histogram[i] != maxcount; i++) {
                }
                maxpixel = (UINT8)i;
            }

            if (maxcount > 2) {
                out[x] = maxpixel;
            } else {
                out[x] = IMAGING_PIXEL_L(im, x, y);
            }
        }
    }
    ImagingSectionLeave(&cookie);

    ImagingCopyPalette(imOut, im);

    return imOut;
}

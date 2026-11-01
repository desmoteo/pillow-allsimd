/* PILusm, a gaussian blur and unsharp masking library for PIL
   By Kevin Cazabon, copyright 2003
   kevin_cazabon@hotmail.com
   kevin@cazabon.com */

/* Originally released under LGPL.  Graciously donated to PIL
   for distribution under the standard PIL license in 2009." */

#include "Imaging.h"

static inline UINT8
clip8(int in) {
    if (in >= 255) {
        return 255;
    }
    if (in <= 0) {
        return 0;
    }
    return (UINT8)in;
}

/* Sharpens one row of bytes in place: out holds the blurred row. */
static void
ImagingUnsharpMaskRow(
    UINT8 *out, const UINT8 *in, int size, int percent, int threshold
) {
    /* the active tier's kernel does the bytes it can (UnsharpMaskSIMD.c) */
    int x = IMAGING_SIMD_CALL(unsharp->row, out, in, size, percent, threshold), diff;

    for (; x < size; x++) {
        /* compare in/out pixels, apply sharpening */
        diff = in[x] - out[x];
        if (abs(diff) > threshold) {
            /* add the diff*percent to the original pixel */
            out[x] = clip8(in[x] + diff * percent / 100);
        } else {
            /* new pixel is the same as imIn */
            out[x] = in[x];
        }
    }
}

Imaging
ImagingUnsharpMask(
    Imaging imOut, Imaging imIn, float radius, int percent, int threshold
) {
    ImagingSectionCookie cookie;
    Imaging result;

    int y;

    /* First, do a gaussian blur on the image, putting results in imOut
       temporarily. All format checks are in gaussian blur. */
    result = ImagingGaussianBlur(imOut, imIn, radius, radius, 3);
    if (!result) {
        return NULL;
    }

    /* Now, go through each pixel, compare "normal" pixel to blurred
       pixel. If the difference is more than threshold values, apply
       the OPPOSITE correction to the amount of blur, multiplied by
       percent. */

    ImagingSectionEnter(&cookie);

    for (y = 0; y < imIn->ysize; y++) {
        ImagingUnsharpMaskRow(
            (UINT8 *)imOut->image[y],
            (UINT8 *)imIn->image[y],
            imIn->linesize,
            percent,
            threshold
        );
    }

    ImagingSectionLeave(&cookie);

    return imOut;
}

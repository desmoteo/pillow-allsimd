#include "Imaging.h"
#include <math.h>

#include "ColorLUT.h"

static inline UINT8
clip8(int in) {
    return clip8_lookups[(in + LUT_PRECISION_ROUNDING) >> LUT_PRECISION_BITS];
}

static inline void
interpolate3(INT16 out[3], const INT16 a[3], const INT16 b[3], INT16 shift) {
    out[0] = (a[0] * ((1 << LUT_SHIFT_BITS) - shift) + b[0] * shift) >> LUT_SHIFT_BITS;
    out[1] = (a[1] * ((1 << LUT_SHIFT_BITS) - shift) + b[1] * shift) >> LUT_SHIFT_BITS;
    out[2] = (a[2] * ((1 << LUT_SHIFT_BITS) - shift) + b[2] * shift) >> LUT_SHIFT_BITS;
}

static inline void
interpolate4(INT16 out[4], const INT16 a[4], const INT16 b[4], INT16 shift) {
    out[0] = (a[0] * ((1 << LUT_SHIFT_BITS) - shift) + b[0] * shift) >> LUT_SHIFT_BITS;
    out[1] = (a[1] * ((1 << LUT_SHIFT_BITS) - shift) + b[1] * shift) >> LUT_SHIFT_BITS;
    out[2] = (a[2] * ((1 << LUT_SHIFT_BITS) - shift) + b[2] * shift) >> LUT_SHIFT_BITS;
    out[3] = (a[3] * ((1 << LUT_SHIFT_BITS) - shift) + b[3] * shift) >> LUT_SHIFT_BITS;
}

/**
 * Apply a 3D colour lookup table to imIn, writing the result to imOut.
 *
 * imOut, imIn - images, should be the same size and may be the same image.
 *    Should have 3 or 4 channels.
 * table_channels - number of channels in the lookup table, 3 or 4.
 *    Should be less or equal than number of channels in imOut image;
 * size1D, size_2D and size3D - dimensions of provided table;
 * table - flat table,
 *    array with table_channels * size1D * size2D * size3D elements,
 *    where channels are changed first, then 1D, then 2D, then 3D.
 *    Each element is signed 16-bit int where 0 is lowest output value
 *    and 255 << LUT_PRECISION_BITS (16320) is highest value.
 *
 * Contract: unlike the other point-style loops in libImaging,
 * imIn and imOut MAY be the same image to support running in place.
 *
 * @return imOut on success or NULL on error.
 */
Imaging
ImagingColorLUT3D_linear(
    Imaging imOut,
    Imaging imIn,
    int table_channels,
    int size1D,
    int size2D,
    int size3D,
    INT16 *table
) {
    /* This float to int conversion doesn't have rounding
       error compensation (+0.5) for two reasons:
       1. As we don't hit the highest value,
          we can use one extra bit for precision.
       2. For every pixel, we interpolate 8 elements from the table:
          current and +1 for every dimension and their combinations.
          If we hit the upper cells from the table,
          +1 cells will be outside of the table.
          With this compensation we never hit the upper cells
          but this also doesn't introduce any noticeable difference. */
    UINT32 scale1D = (size1D - 1) / 255.0 * (1 << LUT_SCALE_BITS);
    UINT32 scale2D = (size2D - 1) / 255.0 * (1 << LUT_SCALE_BITS);
    UINT32 scale3D = (size3D - 1) / 255.0 * (1 << LUT_SCALE_BITS);
    int size1D_2D = size1D * size2D;
    int x, y;
    ImagingSectionCookie cookie;

    if (table_channels < 3 || table_channels > 4) {
        PyErr_SetString(PyExc_ValueError, "table_channels should be 3 or 4");
        return NULL;
    }

    if (imIn->type != IMAGING_TYPE_UINT8 || imOut->type != IMAGING_TYPE_UINT8 ||
        imIn->bands < 3 || imOut->bands < table_channels) {
        return (Imaging)ImagingError_ModeError();
    }

    /* In case we have one extra band in imOut and don't have in imIn.*/
    if (imOut->bands > table_channels && imOut->bands > imIn->bands) {
        return (Imaging)ImagingError_ModeError();
    }

    /* the SIMD kernel of the active tier (ColorLUTSIMD.c), if it has one */
    if (IMAGING_SIMD_CALL(
            colorlut->linear3d,
            imOut,
            imIn,
            table_channels,
            size1D,
            size2D,
            size3D,
            table
        )) {
        return imOut;
    }

    ImagingSectionEnter(&cookie);
    for (y = 0; y < imOut->ysize; y++) {
        UINT8 *rowIn = (UINT8 *)imIn->image[y];
        char *rowOut = (char *)imOut->image[y];
        for (x = 0; x < imOut->xsize; x++) {
            UINT32 index1D = rowIn[x * 4 + 0] * scale1D;
            UINT32 index2D = rowIn[x * 4 + 1] * scale2D;
            UINT32 index3D = rowIn[x * 4 + 2] * scale3D;
            INT16 shift1D =
                (LUT_SCALE_MASK & index1D) >> (LUT_SCALE_BITS - LUT_SHIFT_BITS);
            INT16 shift2D =
                (LUT_SCALE_MASK & index2D) >> (LUT_SCALE_BITS - LUT_SHIFT_BITS);
            INT16 shift3D =
                (LUT_SCALE_MASK & index3D) >> (LUT_SCALE_BITS - LUT_SHIFT_BITS);
            int idx = table_channels * table_index3D(
                                           index1D >> LUT_SCALE_BITS,
                                           index2D >> LUT_SCALE_BITS,
                                           index3D >> LUT_SCALE_BITS,
                                           size1D,
                                           size1D_2D
                                       );
            INT16 result[4], left[4], right[4];
            INT16 leftleft[4], leftright[4], rightleft[4], rightright[4];

            if (table_channels == 3) {
                UINT32 v;
                interpolate3(leftleft, &table[idx + 0], &table[idx + 3], shift1D);
                interpolate3(
                    leftright,
                    &table[idx + size1D * 3],
                    &table[idx + size1D * 3 + 3],
                    shift1D
                );
                interpolate3(left, leftleft, leftright, shift2D);

                interpolate3(
                    rightleft,
                    &table[idx + size1D_2D * 3],
                    &table[idx + size1D_2D * 3 + 3],
                    shift1D
                );
                interpolate3(
                    rightright,
                    &table[idx + size1D_2D * 3 + size1D * 3],
                    &table[idx + size1D_2D * 3 + size1D * 3 + 3],
                    shift1D
                );
                interpolate3(right, rightleft, rightright, shift2D);

                interpolate3(result, left, right, shift3D);

                v = MAKE_UINT32(
                    clip8(result[0]),
                    clip8(result[1]),
                    clip8(result[2]),
                    rowIn[x * 4 + 3]
                );
                memcpy(rowOut + x * sizeof(v), &v, sizeof(v));
            }

            if (table_channels == 4) {
                UINT32 v;
                interpolate4(leftleft, &table[idx + 0], &table[idx + 4], shift1D);
                interpolate4(
                    leftright,
                    &table[idx + size1D * 4],
                    &table[idx + size1D * 4 + 4],
                    shift1D
                );
                interpolate4(left, leftleft, leftright, shift2D);

                interpolate4(
                    rightleft,
                    &table[idx + size1D_2D * 4],
                    &table[idx + size1D_2D * 4 + 4],
                    shift1D
                );
                interpolate4(
                    rightright,
                    &table[idx + size1D_2D * 4 + size1D * 4],
                    &table[idx + size1D_2D * 4 + size1D * 4 + 4],
                    shift1D
                );
                interpolate4(right, rightleft, rightright, shift2D);

                interpolate4(result, left, right, shift3D);

                v = MAKE_UINT32(
                    clip8(result[0]),
                    clip8(result[1]),
                    clip8(result[2]),
                    clip8(result[3])
                );
                memcpy(rowOut + x * sizeof(v), &v, sizeof(v));
            }
        }
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}

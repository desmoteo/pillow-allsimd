#ifndef __SIMD_KERNELS_H__
#define __SIMD_KERNELS_H__

#include <stdint.h>

/* The table of SIMD kernels (design/SIMD_DISPATCH_PLAN.md).
 *
 * The SIMD code of each operation lives in a companion file
 * (<Area>SIMD.c), and every companion is compiled into SIMDKernels.c, one
 * translation unit per instruction-set tier, whose only export is a table
 * of this type. The scalar files call a kernel through ImagingSIMD, the
 * table of the tier chosen when _imaging loads (SIMDDispatch.c); an entry
 * is NULL where the tier has no kernel, and the scalar code runs.
 *
 * A row kernel takes the scalar loop's arguments, does the pixels the
 * vector loop can and returns how many, and the scalar file finishes the
 * row from there. A kernel for a whole operation returns 1 when it did
 * the work and 0 when it didn't handle the case. Kernels are called once
 * per row or per operation, never per pixel.
 */

/* Kernel filters (FilterSIMD.c): 3x3 and 5x5, for 1-byte and 4-byte
   pixels, with integer coefficients at a precision (i*) or float ones
   (f*). Each writes the whole output image. */
typedef struct {
    void (*i3x3_u8)(
        Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
    );
    void (*i5x5_u8)(
        Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
    );
    void (*i3x3_4u8)(
        Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
    );
    void (*i5x5_4u8)(
        Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
    );
    void (*f3x3_u8)(Imaging imOut, Imaging im, const float *kernel, float offset);
    void (*f5x5_u8)(Imaging imOut, Imaging im, const float *kernel, float offset);
    void (*f3x3_4u8)(Imaging imOut, Imaging im, const float *kernel, float offset);
    void (*f5x5_4u8)(Imaging imOut, Imaging im, const float *kernel, float offset);
} ImagingSIMDFilterKernels;

/* Integer box downscale (ReduceSIMD.c), 8-bit images with 4-byte (u8x4)
   or 1-byte (u8x1) pixels. Each returns 0 for scales it has no kernel
   for; otherwise it writes the output image but its last column and row
   (ImagingReduceCorners). */
typedef struct {
    int (*u8x4)(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale);
    int (*u8x1)(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale);
} ImagingSIMDReduceKernels;

/* Resampling (ResampleSIMD.c). The 8-bit horizontal kernels take a row of
   4-byte pixels (or four rows, which share the coefficient loads) and
   write the output row, 8-bit (8u) or 16-bit for fast mode (_16); the 1u
   ones a row of 1-byte pixels. The vertical kernel writes an output row
   of xsize 4-byte words from the input rows ymin .. ymin + ymax. The wide
   kernels do a whole pass of an I, F or I;16 image; a horizontal one
   returns 0 when out of memory. fast_vrow does fast mode's vertical pass
   for the bytes the vector loop can and returns how many. */
typedef void (*ImagingResampleRow8uFn)(
    UINT32 *out, UINT32 *in, int xsize, int *bounds, INT16 *kk, int kmax, int precision
);
typedef void (*ImagingResampleRows8uFn)(
    UINT32 *out0,
    UINT32 *out1,
    UINT32 *out2,
    UINT32 *out3,
    UINT32 *in0,
    UINT32 *in1,
    UINT32 *in2,
    UINT32 *in3,
    int xsize,
    int *bounds,
    INT16 *kk,
    int kmax,
    int precision
);
typedef void (*ImagingResampleRow16Fn)(
    uint64_t *out,
    UINT32 *in,
    int xsize,
    int *bounds,
    INT16 *kk,
    int kmax,
    int precision
);
typedef void (*ImagingResampleRows16Fn)(
    uint64_t *out0,
    uint64_t *out1,
    uint64_t *out2,
    uint64_t *out3,
    UINT32 *in0,
    UINT32 *in1,
    UINT32 *in2,
    UINT32 *in3,
    int xsize,
    int *bounds,
    INT16 *kk,
    int kmax,
    int precision
);
typedef void (*ImagingResampleRow1uFn)(
    UINT8 *out,
    const UINT8 *in,
    int xsize,
    int *bounds,
    INT16 *kk,
    int kmax,
    int precision
);
typedef void (*ImagingResampleRows1uFn)(
    UINT8 *out0,
    UINT8 *out1,
    UINT8 *out2,
    UINT8 *out3,
    const UINT8 *in0,
    const UINT8 *in1,
    const UINT8 *in2,
    const UINT8 *in3,
    int xsize,
    int *bounds,
    INT16 *kk,
    int kmax,
    int precision
);
typedef void (*ImagingResampleVertical8uFn)(
    UINT32 *out, Imaging imIn, int ymin, int ymax, INT16 *k, int precision, int xsize
);
typedef int (*ImagingResampleWideHFn)(
    Imaging imOut, Imaging imIn, int offset, int ksize, int *bounds, double *kk
);
typedef void (*ImagingResampleWideVFn)(
    Imaging imOut, Imaging imIn, int ksize, int *bounds, double *kk
);
typedef int (*ImagingResampleFastVrowFn)(
    UINT8 *out, INT16 **rows, const INT16 *w, int taps, int n
);

typedef struct {
    ImagingResampleRow8uFn horizontal_8u;
    ImagingResampleRows8uFn horizontal_8u4x;
    ImagingResampleRow16Fn horizontal_8u_16;
    ImagingResampleRows16Fn horizontal_8u4x_16;
    ImagingResampleRow1uFn horizontal_1u;
    ImagingResampleRows1uFn horizontal_1u4x;
    ImagingResampleVertical8uFn vertical_8u;
    ImagingResampleWideHFn horizontal_f32;
    ImagingResampleWideHFn horizontal_i32;
    ImagingResampleWideHFn horizontal_u16le;
    ImagingResampleWideHFn horizontal_u16be;
    ImagingResampleWideVFn vertical_f32;
    ImagingResampleWideVFn vertical_i32;
    ImagingResampleWideVFn vertical_u16le;
    ImagingResampleWideVFn vertical_u16be;
    ImagingResampleFastVrowFn fast_vrow;
} ImagingSIMDResampleKernels;

/* Mode conversions (ConvertSIMD.c, ConvertYCbCrSIMD.c): a row of pixels
   in, a row out. Each does the pixels its vector loop can and returns how
   many; Convert.c finishes the row from there. */
typedef int (*ImagingSIMDRowFn)(UINT8 *out, const UINT8 *in, int pixels);

typedef struct {
    ImagingSIMDRowFn l2rgb;
    ImagingSIMDRowFn rgb2l;
    ImagingSIMDRowFn rgb2la;
    ImagingSIMDRowFn rgb2i;
    ImagingSIMDRowFn rgba2la;
    ImagingSIMDRowFn rgb2hsv;
    ImagingSIMDRowFn rgb2hsv_fast;
    ImagingSIMDRowFn rgb2rgba;
    ImagingSIMDRowFn rgb2cmyk;
    ImagingSIMDRowFn premultiply;   /* RGBA to RGBa */
    ImagingSIMDRowFn unpremultiply; /* RGBa to RGBA */
    ImagingSIMDRowFn rgb2ycbcr;
    ImagingSIMDRowFn ycbcr2rgb;
    ImagingSIMDRowFn rgb2ycbcr_fast;
    ImagingSIMDRowFn ycbcr2rgb_fast;
} ImagingSIMDConvertKernels;

/* 3D colour lookup tables (ColorLUTSIMD.c): the whole image, with the
   arguments checked by the caller; returns 1. */
typedef struct {
    int (*linear3d)(
        Imaging imOut,
        Imaging imIn,
        int table_channels,
        int size1D,
        int size2D,
        int size3D,
        INT16 *table
    );
} ImagingSIMDColorLUTKernels;

/* Geometry (GeometrySIMD.c): the transposes (rotate 90 and 270, transpose,
   transverse) and row flips (flip left-right, rotate 180) of 8-bit and
   32-bit images, returning 0 for other pixel sizes; the bilinear and
   bicubic transform filters for 8-bit images with 3 or 4 bands; and
   ImagingGenericTransform's tile loops with those filters, returning 0
   for the images, filters and transforms they don't take. */
typedef struct {
    int (*transpose)(Imaging imOut, Imaging imIn, int flipx, int flipy);
    int (*flip_rows)(Imaging imOut, Imaging imIn, int flipy);
    /* ImagingTransformFilter, declared later in Imaging.h */
    int (*bilinear_rgb)(void *out, Imaging im, double x, double y);
    int (*bicubic_rgb)(void *out, Imaging im, double x, double y);
    int (*transform)(
        Imaging imOut,
        Imaging imIn,
        int x0,
        int y0,
        int x1,
        int y1,
        int method,
        void *transform_data,
        int filterid,
        int fill
    );
} ImagingSIMDGeometryKernels;

/* Fast mode's transforms (TransformFastSIMD.c): the whole region, tile by
   tile, from the map and the bicubic table ImagingTransformFast prepared;
   returns 1. */
struct ImagingTFMap;
typedef struct {
    int (*tiles)(
        Imaging imOut,
        Imaging imIn,
        const struct ImagingTFMap *map,
        INT16 (*cubic)[4],
        int x0,
        int y0,
        int x1,
        int y1,
        int bicubic,
        int fill
    );
} ImagingSIMDTransformFastKernels;

/* Box blur (BoxBlurSIMD.c). line32 blurs a row of 4-byte pixels with
   running sums, line32x2 two rows at once; direct does the box of each
   byte from the count source rows for the bytes it can and returns how
   many; interleave4 turns four 1-byte rows into a 4-byte row and
   deinterleave4 back; columns advances the vertical pass's running sums
   for the bytes it can and returns how many. */
struct ImagingBoxDirect;
typedef struct {
    void (*line32)(
        UINT32 *out,
        UINT32 *in,
        int lastx,
        int radius,
        int edgeA,
        int edgeB,
        UINT32 ww,
        UINT32 fw
    );
    void (*line32x2)(
        UINT32 *out0,
        UINT32 *out1,
        UINT32 *in0,
        UINT32 *in1,
        int lastx,
        int radius,
        int edgeA,
        int edgeB,
        UINT32 ww,
        UINT32 fw
    );
    int (*direct)(
        UINT8 *out,
        const UINT8 **src,
        int n,
        const struct ImagingBoxDirect *d,
        int count
    );
    void (*interleave4)(
        UINT32 *out, UINT8 *in0, UINT8 *in1, UINT8 *in2, UINT8 *in3, int xsize
    );
    void (*deinterleave4)(
        UINT8 *out0, UINT8 *out1, UINT8 *out2, UINT8 *out3, UINT32 *in, int xsize
    );
    int (*columns)(
        UINT8 *out,
        UINT32 *acc,
        const UINT8 *add,
        const UINT8 *sub,
        const UINT8 *far_row,
        int size,
        UINT32 ww,
        UINT32 fw
    );
} ImagingSIMDBlurKernels;

/* Row kernels that do the bytes their vector loops can and return how
   many; the scalar file finishes the row. Unsharp mask
   (UnsharpMaskSIMD.c): a row sharpened in place from its blurred copy.
   Fast mode's Gaussian blur (GaussianFastSIMD.c): the taps of radius r,
   weights w (w[0] at the centre), over the 2r + 1 sources. Alpha
   composite (AlphaCompositeSIMD.c): src over dst into out, which may be
   dst, rows of RGBA or LA pixels as bytes. Fast mode's blend
   (BlendSIMD.c): in1 + (in2 - in1) a15 / 2^15. ImageChops
   (ChopsSIMD.c): multiply (screen with invert = 255) and the saturating
   add (sign 1) or subtract (sign -1). */
typedef struct {
    int (*row)(UINT8 *out, const UINT8 *in, int size, int percent, int threshold);
} ImagingSIMDUnsharpKernels;

typedef struct {
    int (*taps)(UINT8 *out, const UINT8 **src, int r, const UINT16 *w, int n);
} ImagingSIMDGaussKernels;

typedef struct {
    int (*row)(UINT8 *out, const UINT8 *dst, const UINT8 *src, int xsize);
} ImagingSIMDCompositeKernels;

typedef struct {
    int (*fast_row)(UINT8 *out, const UINT8 *in1, const UINT8 *in2, int n, INT16 a15);
} ImagingSIMDBlendKernels;

typedef struct {
    int (*multiply_row)(
        UINT8 *out, const UINT8 *in1, const UINT8 *in2, int n, UINT8 invert
    );
    int (*saturate_row)(
        UINT8 *out, const UINT8 *in1, const UINT8 *in2, int n, int sign
    );
} ImagingSIMDChopsKernels;

/* PNG (ZipEncodeSIMD.c): the encoder's row filters do the bytes their
   vector loops can, from byte i on for average and Paeth, add their cost
   to *s and return how many they did; unfilter_up undoes the "up" filter
   on a whole row. Packing (PackSIMD.c, UnpackSIMD.c): RGB and BGR
   triplets from or into 4-byte pixels, the pixels the vector loop can,
   their count back. */
typedef struct {
    int (*cost_none)(const UINT8 *cur, int n, int *s);
    int (*filter_diff)(UINT8 *out, const UINT8 *cur, const UINT8 *ref, int n, int *s);
    int (*filter_average)(
        UINT8 *out, const UINT8 *cur, const UINT8 *prev, int n, int bpp, int i, int *s
    );
    int (*filter_paeth)(
        UINT8 *out, const UINT8 *cur, const UINT8 *prev, int n, int bpp, int i, int *s
    );
    void (*unfilter_up)(UINT8 *row, const UINT8 *prev, int len);
} ImagingSIMDZipKernels;

typedef struct {
    int (*rgb)(UINT8 *out, const UINT8 *in, int pixels);
    int (*bgr)(UINT8 *out, const UINT8 *in, int pixels);
} ImagingSIMDPackKernels;

typedef struct {
    int (*rgb)(UINT8 *out, const UINT8 *in, int pixels);
    int (*bgr)(UINT8 *out, const UINT8 *in, int pixels);
} ImagingSIMDUnpackKernels;

typedef struct ImagingSIMDKernels {
    /* "scalar", "sse4", "avx2", "avxvnni" or "neon" */
    const char *name;
    /* One group per area, never NULL: the scalar table's groups have
       every kernel NULL. */
    const ImagingSIMDFilterKernels *filter;
    const ImagingSIMDReduceKernels *reduce;
    const ImagingSIMDResampleKernels *resample;
    const ImagingSIMDConvertKernels *convert;
    const ImagingSIMDColorLUTKernels *colorlut;
    const ImagingSIMDGeometryKernels *geometry;
    const ImagingSIMDTransformFastKernels *transform_fast;
    const ImagingSIMDBlurKernels *blur;
    const ImagingSIMDUnsharpKernels *unsharp;
    const ImagingSIMDGaussKernels *gauss;
    const ImagingSIMDCompositeKernels *composite;
    const ImagingSIMDBlendKernels *blend;
    const ImagingSIMDChopsKernels *chops;
    const ImagingSIMDZipKernels *zip;
    const ImagingSIMDPackKernels *pack;
    const ImagingSIMDUnpackKernels *unpack;
} ImagingSIMDKernels;

/* The active tier's table; the scalar table until ImagingSIMDInit(). */
extern const ImagingSIMDKernels *ImagingSIMD;

/* Calls a kernel of the active tier, 0 when the tier has none. */
#define IMAGING_SIMD_CALL(kernel, ...) \
    (ImagingSIMD->kernel ? ImagingSIMD->kernel(__VA_ARGS__) : 0)

/* Chooses the tier for this CPU. Returns 0, or -1 with a message in
 *error when the build's instruction set isn't supported by the CPU. */
extern int
ImagingSIMDInit(const char **error);

extern const ImagingSIMDKernels ImagingSIMDKernels_scalar;

#endif

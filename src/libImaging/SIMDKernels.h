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

typedef struct ImagingSIMDKernels {
    /* "scalar", "sse4", "avx2", "avxvnni" or "neon" */
    const char *name;
    /* One group per area, never NULL: the scalar table's groups have
       every kernel NULL. */
    const ImagingSIMDFilterKernels *filter;
    const ImagingSIMDReduceKernels *reduce;
    const ImagingSIMDResampleKernels *resample;
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

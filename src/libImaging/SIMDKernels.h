#ifndef __SIMD_KERNELS_H__
#define __SIMD_KERNELS_H__

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

typedef struct ImagingSIMDKernels {
    /* "scalar", "sse4", "avx2", "avxvnni" or "neon" */
    const char *name;
    /* One group per area, never NULL: the scalar table's groups have
       every kernel NULL. */
    const ImagingSIMDFilterKernels *filter;
    const ImagingSIMDReduceKernels *reduce;
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

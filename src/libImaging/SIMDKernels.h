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

typedef struct ImagingSIMDKernels {
    /* "scalar", "sse4", "avx2", "avxvnni" or "neon" */
    const char *name;
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

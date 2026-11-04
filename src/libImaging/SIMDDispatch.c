/*
 * The Imaging library
 *
 * The choice of the SIMD kernels at run time (design/SIMD_DISPATCH_PLAN.md):
 * ImagingSIMD points at the kernel table of the highest tier this CPU
 * supports among those compiled in (SIMDKernels.c). A dispatch build on
 * x86 has the SSE4.1, AVX2 and AVX-VNNI tiers; a build for one
 * instruction set has that one; the scalar table, with no kernels, is the
 * fallback. The PILLOW_ALLSIMD_ISA environment variable caps the tier at
 * import (scalar, sse4, avx2, avxvnni, neon) and _imaging.set_simd()
 * changes it afterwards, for tests and benchmarks.
 */

#include "Imaging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const ImagingSIMDFilterKernels no_filter_kernels = {0};
static const ImagingSIMDReduceKernels no_reduce_kernels = {0};
static const ImagingSIMDResampleKernels no_resample_kernels = {0};
static const ImagingSIMDConvertKernels no_convert_kernels = {0};
static const ImagingSIMDColorLUTKernels no_colorlut_kernels = {0};
static const ImagingSIMDGeometryKernels no_geometry_kernels = {0};
static const ImagingSIMDTransformFastKernels no_transform_fast_kernels = {0};
static const ImagingSIMDBlurKernels no_blur_kernels = {0};
static const ImagingSIMDUnsharpKernels no_unsharp_kernels = {0};
static const ImagingSIMDGaussKernels no_gauss_kernels = {0};
static const ImagingSIMDCompositeKernels no_composite_kernels = {0};
static const ImagingSIMDBlendKernels no_blend_kernels = {0};
static const ImagingSIMDChopsKernels no_chops_kernels = {0};
static const ImagingSIMDZipKernels no_zip_kernels = {0};
static const ImagingSIMDPackKernels no_pack_kernels = {0};
static const ImagingSIMDUnpackKernels no_unpack_kernels = {0};
static const ImagingSIMDPointKernels no_point_kernels = {0};
static const ImagingSIMDBandsKernels no_bands_kernels = {0};
static const ImagingSIMDPasteKernels no_paste_kernels = {0};
static const ImagingSIMDDrawKernels no_draw_kernels = {0};
static const ImagingSIMDQuantKernels no_quant_kernels = {0};
static const ImagingSIMDRankKernels no_rank_kernels = {0};

const ImagingSIMDKernels ImagingSIMDKernels_scalar = {
    .name = "scalar",
    .filter = &no_filter_kernels,
    .reduce = &no_reduce_kernels,
    .resample = &no_resample_kernels,
    .convert = &no_convert_kernels,
    .colorlut = &no_colorlut_kernels,
    .geometry = &no_geometry_kernels,
    .transform_fast = &no_transform_fast_kernels,
    .blur = &no_blur_kernels,
    .unsharp = &no_unsharp_kernels,
    .gauss = &no_gauss_kernels,
    .composite = &no_composite_kernels,
    .blend = &no_blend_kernels,
    .chops = &no_chops_kernels,
    .zip = &no_zip_kernels,
    .pack = &no_pack_kernels,
    .unpack = &no_unpack_kernels,
    .point = &no_point_kernels,
    .bands = &no_bands_kernels,
    .paste = &no_paste_kernels,
    .draw = &no_draw_kernels,
    .quant = &no_quant_kernels,
    .rank = &no_rank_kernels,
};

const ImagingSIMDKernels *ImagingSIMD = &ImagingSIMDKernels_scalar;

#if defined(IMAGING_SIMD_DISPATCH)
extern const ImagingSIMDKernels ImagingSIMDKernels_sse4;
extern const ImagingSIMDKernels ImagingSIMDKernels_avx2;
#if defined(IMAGING_SIMD_DISPATCH_AVXVNNI)
extern const ImagingSIMDKernels ImagingSIMDKernels_avxvnni;
#endif
#elif defined(IMAGING_SIMD_TABLE)
extern const ImagingSIMDKernels IMAGING_SIMD_TABLE;
#endif

/* The tables compiled in, the scalar one first, then by instruction set */
static const ImagingSIMDKernels *const tiers[] = {
    &ImagingSIMDKernels_scalar,
#if defined(IMAGING_SIMD_DISPATCH)
    &ImagingSIMDKernels_sse4,
    &ImagingSIMDKernels_avx2,
#if defined(IMAGING_SIMD_DISPATCH_AVXVNNI)
    &ImagingSIMDKernels_avxvnni,
#endif
#elif defined(IMAGING_SIMD_TABLE)
    &IMAGING_SIMD_TABLE,
#endif
};
#define NTIERS ((int)(sizeof(tiers) / sizeof(tiers[0])))

/* What a tier needs of the CPU: 0 nothing (the scalar table, and NEON,
   part of the arm64 baseline), then SSE4.1, AVX2, AVX-VNNI; -1 for a name
   that isn't a tier's. */
static int
tier_level(const char *name) {
    if (!strcmp(name, "scalar") || !strcmp(name, "neon")) {
        return 0;
    }
    if (!strcmp(name, "sse4")) {
        return 1;
    }
    if (!strcmp(name, "avx2")) {
        return 2;
    }
    if (!strcmp(name, "avxvnni")) {
        return 3;
    }
    return -1;
}

/* The highest level this CPU supports. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
static void
cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    int t[4];
    __cpuidex(t, (int)leaf, (int)sub);
    memcpy(r, t, sizeof(t));
}
static int
ymm_enabled(void) {
    return (_xgetbv(0) & 6) == 6;
}
#else
#include <cpuid.h>
static void
cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
static int
ymm_enabled(void) {
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (eax & 6) == 6;
}
#endif

static int
cpu_level(void) {
    unsigned r[4], max;
    cpuid(0, 0, r);
    max = r[0];
    if (max < 1) {
        return 0;
    }
    cpuid(1, 0, r);
    if (!((r[2] >> 19) & 1)) { /* SSE4.1 */
        return 0;
    }
    /* AVX needs the OS to save the YMM registers: OSXSAVE, AVX, XCR0 */
    if (!((r[2] >> 27) & 1) || !((r[2] >> 28) & 1) || !ymm_enabled() || max < 7) {
        return 1;
    }
    cpuid(7, 0, r);
    if (!((r[1] >> 5) & 1)) { /* AVX2 */
        return 1;
    }
    if (r[0] < 1) { /* no sub-leaf 1 */
        return 2;
    }
    cpuid(7, 1, r);
    return ((r[0] >> 4) & 1) ? 3 : 2; /* AVX-VNNI */
}
#else
static int
cpu_level(void) {
    return 0;
}
#endif

static int cpu = -1; /* cpu_level(), once */

static const ImagingSIMDKernels *
tier_by_name(const char *name) {
    int i;
    for (i = 0; i < NTIERS; i++) {
        if (!strcmp(tiers[i]->name, name)) {
            return tiers[i];
        }
    }
    return NULL;
}

int
ImagingSIMDSelect(const char *name) {
    const ImagingSIMDKernels *t = tier_by_name(name);
    if (!t) {
        return -1;
    }
    if (cpu < 0) {
        cpu = cpu_level();
    }
    if (tier_level(name) > cpu) {
        return -2;
    }
    ImagingSIMD = t;
    return 0;
}

int
ImagingSIMDTiers(const char **names, int max) {
    int i;
    for (i = 0; i < NTIERS && i < max; i++) {
        names[i] = tiers[i]->name;
    }
    return i;
}

int
ImagingSIMDInit(const char **error) {
    static char message[200];
    const ImagingSIMDKernels *chosen = &ImagingSIMDKernels_scalar;
    const char *env;
    int i, cap;

    *error = NULL;
    cpu = cpu_level();
#if !defined(IMAGING_SIMD_DISPATCH) && defined(IMAGING_SIMD_TABLE)
    /* A build for one instruction set that this CPU lacks: a clear error
       rather than an illegal instruction later. */
    if (tier_level(IMAGING_SIMD_TABLE.name) > cpu) {
        snprintf(
            message,
            sizeof(message),
            "Pillow-AllSIMD was built for %s, which this CPU doesn't support: "
            "install the wheel, or build with -C simd=dispatch",
            IMAGING_SIMD_TABLE.name
        );
        *error = message;
        return -1;
    }
#endif
    cap = cpu;
    env = getenv("PILLOW_ALLSIMD_ISA");
    if (env && *env) {
        int level = tier_level(env);
        if (level >= 0 && level < cap) {
            cap = level;
        }
    }
    /* the highest tier within cap: the tiers are in order of level */
    for (i = 0; i < NTIERS; i++) {
        if (tier_level(tiers[i]->name) <= cap) {
            chosen = tiers[i];
        }
    }
    ImagingSIMD = chosen;
    return 0;
}

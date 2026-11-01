#ifndef __IMAGING_SIMD_H__
#define __IMAGING_SIMD_H__

#include <string.h>

/* Instruction set the SIMD kernels are compiled for. At most one of
   IMAGING_SIMD_SSE4 and IMAGING_SIMD_NEON is defined; IMAGING_SIMD_AVX2
   implies IMAGING_SIMD_SSE4, IMAGING_SIMD_AVXVNNI implies both. With none
   defined, the portable code is used.

   Three cases (design/SIMD_DISPATCH_PLAN.md):
   - IMAGING_SIMD_DISABLE: the portable code only.
   - IMAGING_SIMD_TIER, defined by a tier's translation unit of a dispatch
     build (SIMDKernels_sse4.c and friends, which include SIMDKernels.c):
     the tier's instruction set, whatever the compiler was told for the
     other files. GCC and clang must have the set enabled to compile its
     intrinsics (setup.py's per-file flags); MSVC compiles any.
   - IMAGING_SIMD_DISPATCH, defined for every file of a dispatch build:
     the portable code in the common files, the kernels in the tiers.
   - Otherwise, a build for one instruction set: the compiler's macros say
     which. */

#define IMAGING_SIMD_TIER_SSE4 1
#define IMAGING_SIMD_TIER_AVX2 2
#define IMAGING_SIMD_TIER_AVXVNNI 3
#define IMAGING_SIMD_TIER_NEON 10

#if defined(IMAGING_SIMD_DISABLE)

#elif defined(IMAGING_SIMD_TIER)

#if IMAGING_SIMD_TIER >= IMAGING_SIMD_TIER_SSE4 && \
    IMAGING_SIMD_TIER < IMAGING_SIMD_TIER_NEON
#define IMAGING_SIMD_SSE4
#include <emmintrin.h>
#include <mmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#if IMAGING_SIMD_TIER >= IMAGING_SIMD_TIER_AVX2
#define IMAGING_SIMD_AVX2
#include <immintrin.h>
#endif
#if IMAGING_SIMD_TIER >= IMAGING_SIMD_TIER_AVXVNNI
#define IMAGING_SIMD_AVXVNNI
#endif
#if !defined(_MSC_VER) &&                                                          \
    (!defined(__SSE4_1__) || (defined(IMAGING_SIMD_AVX2) && !defined(__AVX2__)) || \
     (defined(IMAGING_SIMD_AVXVNNI) && !defined(__AVXVNNI__)))
#error \
    "the tier's instruction set is not enabled for this file (setup.py's per-file flags)"
#endif
#elif IMAGING_SIMD_TIER == IMAGING_SIMD_TIER_NEON
#define IMAGING_SIMD_NEON
#include <arm_neon.h>
#endif

#elif defined(IMAGING_SIMD_DISPATCH)

#else

/* The Microsoft compiler doesn't limit intrinsics by target, and every
   x64 CPU it builds for in practice has SSE4.1. 32-bit x86 only guarantees
   SSE2, so it gets the scalar code. */
#if defined(__SSE4_1__) || \
    (defined(_MSC_VER) && defined(_M_X64) && !defined(_M_ARM64EC))
#define IMAGING_SIMD_SSE4
#include <emmintrin.h>
#include <mmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#if defined(__AVX2__)
#define IMAGING_SIMD_AVX2
#include <immintrin.h>
#if defined(__AVXVNNI__)
#define IMAGING_SIMD_AVXVNNI
#endif
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
#define IMAGING_SIMD_NEON
#include <arm_neon.h>
#endif

#endif /* the cases */

/* The kernel table this translation unit's tier defines (SIMDKernels.c)
   and the table SIMDDispatch.c selects: one per build when the build is
   compiled for one instruction set. Undefined for the scalar build. */
#if defined(IMAGING_SIMD_AVXVNNI)
#define IMAGING_SIMD_TABLE ImagingSIMDKernels_avxvnni
#define IMAGING_SIMD_TIER_NAME "avxvnni"
#elif defined(IMAGING_SIMD_AVX2)
#define IMAGING_SIMD_TABLE ImagingSIMDKernels_avx2
#define IMAGING_SIMD_TIER_NAME "avx2"
#elif defined(IMAGING_SIMD_SSE4)
#define IMAGING_SIMD_TABLE ImagingSIMDKernels_sse4
#define IMAGING_SIMD_TIER_NAME "sse4"
#elif defined(IMAGING_SIMD_NEON)
#define IMAGING_SIMD_TABLE ImagingSIMDKernels_neon
#define IMAGING_SIMD_TIER_NAME "neon"
#endif

/* Inlined at every call, for kernels that should see an argument as a
   constant where a call site passes one (a shift count): plain inline
   leaves a large body that is called twice out of line. */
#if defined(__GNUC__) || defined(__clang__)
#define SIMD_FORCE_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define SIMD_FORCE_INLINE static __forceinline
#else
#define SIMD_FORCE_INLINE static inline
#endif

#if defined(IMAGING_SIMD_SSE4)
/* Unaligned 32-bit load. */
static inline INT32
mm_load_si32(const void *ptr) {
    INT32 v;
    memcpy(&v, ptr, sizeof(v));
    return v;
}

static inline __m128i
mm_cvtepu8_epi32(const void *ptr) {
    INT32 v;
    memcpy(&v, ptr, sizeof(v));
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(v));
}
#endif

#if defined(IMAGING_SIMD_SSE4)
/* acc + a[0] * b[0] + a[1] * b[1] for each pair of signed 16-bit lanes,
   modulo 2^32: one vpdpwssd with AVX-VNNI, pmaddwd and paddd otherwise. */
#if defined(IMAGING_SIMD_AVXVNNI)
#define mm_dpwssd(acc, a, b) _mm_dpwssd_avx_epi32(acc, a, b)
#define mm256_dpwssd(acc, a, b) _mm256_dpwssd_avx_epi32(acc, a, b)
#else
#define mm_dpwssd(acc, a, b) _mm_add_epi32(acc, _mm_madd_epi16(a, b))
#define mm256_dpwssd(acc, a, b) _mm256_add_epi32(acc, _mm256_madd_epi16(a, b))
#endif
/* The same without vpdpwssd, for sums that are one long dependency chain:
   there pmaddwd stays out of the chain, and only the 1-cycle paddd is in it. */
#define mm_madd_add(acc, a, b) _mm_add_epi32(acc, _mm_madd_epi16(a, b))
#define mm256_madd_add(acc, a, b) _mm256_add_epi32(acc, _mm256_madd_epi16(a, b))
#endif

#if defined(IMAGING_SIMD_SSE4)
static inline __m128i
mm_cvtepu8_epi16(const void *ptr) {
    return _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)ptr));
}
#endif
#if defined(IMAGING_SIMD_AVX2)
static inline __m256i
mm256_cvtepu8_epi32(const void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr));
}
static inline __m256i
mm256_cvtepu8_epi16(const void *ptr) {
    return _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)ptr));
}
#endif

/* The table's pack->permute4 (PackSIMD.c): out[i * 4 + k] = in[i * 4 + pk],
   or 0 for PERMUTE_00 and 255 for PERMUTE_FF. */
#define PERMUTE_00 -1
#define PERMUTE_FF -2

#endif

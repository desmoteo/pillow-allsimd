#ifndef __IMAGING_SIMD_H__
#define __IMAGING_SIMD_H__

#include <string.h>

/* Instruction set the SIMD kernels are compiled for. At most one of
   IMAGING_SIMD_SSE4 and IMAGING_SIMD_NEON is defined; IMAGING_SIMD_AVX2
   implies IMAGING_SIMD_SSE4. With none defined, the portable scalar code
   is used. Define IMAGING_SIMD_DISABLE to force the scalar code. */

#if !defined(IMAGING_SIMD_DISABLE)

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
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
#define IMAGING_SIMD_NEON
#include <arm_neon.h>
#endif

#endif /* IMAGING_SIMD_DISABLE */

#if defined(IMAGING_SIMD_SSE4)
/* Unaligned 32-bit load. */
static INT32 inline
mm_load_si32(const void *ptr) {
    INT32 v;
    memcpy(&v, ptr, sizeof(v));
    return v;
}

static __m128i inline
mm_cvtepu8_epi32(const void *ptr) {
    INT32 v;
    memcpy(&v, ptr, sizeof(v));
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(v));
}
#endif

#if defined(IMAGING_SIMD_AVX2)
static __m256i inline
mm256_cvtepu8_epi32(const void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr));
}
#endif

#endif

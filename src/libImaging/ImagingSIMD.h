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
#if defined(__AVXVNNI__)
#define IMAGING_SIMD_AVXVNNI
#endif
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
#define IMAGING_SIMD_NEON
#include <arm_neon.h>
#endif

#endif /* IMAGING_SIMD_DISABLE */

#if defined(IMAGING_SIMD_SSE4)
/* Unaligned 32-bit load. */
static INT32 inline mm_load_si32(const void *ptr) {
    INT32 v;
    memcpy(&v, ptr, sizeof(v));
    return v;
}

static __m128i inline mm_cvtepu8_epi32(const void *ptr) {
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
static __m128i inline mm_cvtepu8_epi16(const void *ptr) {
    return _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)ptr));
}
#endif
#if defined(IMAGING_SIMD_AVX2)
static __m256i inline mm256_cvtepu8_epi32(const void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr));
}
static __m256i inline mm256_cvtepu8_epi16(const void *ptr) {
    return _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)ptr));
}
#endif

/* For pixels of 4 bytes: out[i * 4 + k] = in[i * 4 + pk], or 0 for
   PERMUTE_00 and 255 for PERMUTE_FF. Does the pixels it can in vectors
   and returns how many; 0 without SIMD. The indices are constants at the
   call sites, so are the shuffles once inlined. */
#define PERMUTE_00 -1
#define PERMUTE_FF -2

static inline int
permute4(UINT8 *out, const UINT8 *in, int pixels, int p0, int p1, int p2, int p3) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    int p[4] = {p0, p1, p2, p3}, k;
    char shuffle[16], fill[16];
    __m128i vshuffle, vfill;
    for (k = 0; k < 16; k++) {
        shuffle[k] = p[k % 4] < 0 ? -1 : (char)(p[k % 4] + k / 4 * 4);
        fill[k] = p[k % 4] == PERMUTE_FF ? -1 : 0;
    }
    vshuffle = _mm_loadu_si128((const __m128i *)shuffle);
    vfill = _mm_loadu_si128((const __m128i *)fill);
    for (; i < pixels - 3; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(in + i * 4));
        v = _mm_or_si128(_mm_shuffle_epi8(v, vshuffle), vfill);
        _mm_storeu_si128((__m128i *)(out + i * 4), v);
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; i < pixels - 15; i += 16) {
        uint8x16x4_t v = vld4q_u8(in + i * 4);
        uint8x16_t c[6] = {
            v.val[0], v.val[1], v.val[2], v.val[3], vdupq_n_u8(0), vdupq_n_u8(255)
        };
        uint8x16x4_t o = {
            {c[p0 >= 0 ? p0 : 3 - p0],
             c[p1 >= 0 ? p1 : 3 - p1],
             c[p2 >= 0 ? p2 : 3 - p2],
             c[p3 >= 0 ? p3 : 3 - p3]}
        };
        vst4q_u8(out + i * 4, o);
    }
#else
    (void)out, (void)in, (void)pixels, (void)p0, (void)p1, (void)p2, (void)p3;
#endif
    return i;
}

#endif

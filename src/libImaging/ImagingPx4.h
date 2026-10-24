#ifndef __IMAGING_PX4_H__
#define __IMAGING_PX4_H__

/* One 4-byte pixel as four 32-bit lanes, one per channel. All arithmetic
   is modulo 2^32, so the SSE4 and portable versions give identical
   results. Used by kernels that work on one pixel at a time.

   arm64 uses the portable version: on Cortex-A78AE with GCC 11 the
   compiler-vectorized code was faster than NEON intrinsics. */

#if defined(IMAGING_SIMD_SSE4)
typedef __m128i px4;

static inline px4
px4_load(const void *pixel) {
    return mm_cvtepu8_epi32(pixel);
}
static inline px4
px4_set1(UINT32 v) {
    return _mm_set1_epi32(v);
}
static inline px4
px4_add(px4 a, px4 b) {
    return _mm_add_epi32(a, b);
}
static inline px4
px4_sub(px4 a, px4 b) {
    return _mm_sub_epi32(a, b);
}
static inline px4
px4_mul(px4 a, px4 b) {
    return _mm_mullo_epi32(a, b);
}
static inline px4
px4_shr(px4 a, int n) {
    return _mm_srl_epi32(a, _mm_cvtsi32_si128(n));
}
/* Low byte of each lane, packed back into a pixel. */
static inline UINT32
px4_pack(px4 v) {
    return _mm_cvtsi128_si32(_mm_shuffle_epi8(
        v, _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0)
    ));
}

#else
typedef struct {
    UINT32 v[4];
} px4;

static inline px4
px4_load(const void *pixel) {
    const UINT8 *p = (const UINT8 *)pixel;
    px4 r = {{p[0], p[1], p[2], p[3]}};
    return r;
}
static inline px4
px4_set1(UINT32 v) {
    px4 r = {{v, v, v, v}};
    return r;
}
static inline px4
px4_add(px4 a, px4 b) {
    px4 r = {{a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3]}};
    return r;
}
static inline px4
px4_sub(px4 a, px4 b) {
    px4 r = {{a.v[0] - b.v[0], a.v[1] - b.v[1], a.v[2] - b.v[2], a.v[3] - b.v[3]}};
    return r;
}
static inline px4
px4_mul(px4 a, px4 b) {
    px4 r = {{a.v[0] * b.v[0], a.v[1] * b.v[1], a.v[2] * b.v[2], a.v[3] * b.v[3]}};
    return r;
}
static inline px4
px4_shr(px4 a, int n) {
    px4 r = {{a.v[0] >> n, a.v[1] >> n, a.v[2] >> n, a.v[3] >> n}};
    return r;
}
static inline UINT32
px4_pack(px4 v) {
    UINT8 p[4] = {(UINT8)v.v[0], (UINT8)v.v[1], (UINT8)v.v[2], (UINT8)v.v[3]};
    UINT32 r;
    memcpy(&r, p, sizeof(r));
    return r;
}
#endif

/* (v + (1 << 23)) >> 24 for each channel, packed back into a pixel. */
static inline UINT32
px4_round24(px4 v) {
    return px4_pack(px4_shr(px4_add(v, px4_set1(1 << 23)), 24));
}

#endif

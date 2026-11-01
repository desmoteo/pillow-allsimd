/* The PNG encoder's row filters (ZipEncode.c), compiled into SIMDKernels.c.
   Each filters the bytes its vector loop can, from byte i on for the
   filters that depend on the pixel to the left, adds their cost to *s and
   returns how many it did; ZipEncode.c finishes the row. */

#include "ZipEncode.h"

#if defined(IMAGING_SIMD_SSE4)
/* min(v, -v) as unsigned bytes is PNG_COST(v); psadbw adds them up. */
static inline __m128i
png_cost16(__m128i v) {
    __m128i c = _mm_min_epu8(v, _mm_sub_epi8(_mm_setzero_si128(), v));
    return _mm_sad_epu8(c, _mm_setzero_si128());
}

/* A row's cost fits 32 bits, and so does each 64-bit half; this also
   builds for 32-bit x86. */
static inline int
png_sum64(__m128i acc) {
    return _mm_cvtsi128_si32(acc) + _mm_cvtsi128_si32(_mm_srli_si128(acc, 8));
}
#elif defined(IMAGING_SIMD_NEON)
static inline uint32x4_t
png_cost16(uint32x4_t acc, uint8x16_t v) {
    uint8x16_t c = vminq_u8(v, vsubq_u8(vdupq_n_u8(0), v));
    return vpadalq_u16(acc, vpaddlq_u8(c));
}
#endif

static int
png_cost_none_simd(const UINT8 *cur, int n, int *s) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    __m128i acc = _mm_setzero_si128();
    for (; i < n - 15; i += 16) {
        acc = _mm_add_epi64(acc, png_cost16(_mm_loadu_si128((__m128i *)(cur + i))));
    }
    *s = png_sum64(acc);
#elif defined(IMAGING_SIMD_NEON)
    uint32x4_t acc = vdupq_n_u32(0);
    for (; i < n - 15; i += 16) {
        acc = png_cost16(acc, vld1q_u8(cur + i));
    }
    *s = (int)vaddvq_u32(acc);
#endif
    return i;
}

static int
png_filter_diff_simd(UINT8 *out, const UINT8 *cur, const UINT8 *ref, int n, int *s) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    __m128i acc = _mm_setzero_si128();
    for (; i < n - 15; i += 16) {
        __m128i v = _mm_sub_epi8(
            _mm_loadu_si128((__m128i *)(cur + i)), _mm_loadu_si128((__m128i *)(ref + i))
        );
        _mm_storeu_si128((__m128i *)(out + i), v);
        acc = _mm_add_epi64(acc, png_cost16(v));
    }
    *s = png_sum64(acc);
#elif defined(IMAGING_SIMD_NEON)
    uint32x4_t acc = vdupq_n_u32(0);
    for (; i < n - 15; i += 16) {
        uint8x16_t v = vsubq_u8(vld1q_u8(cur + i), vld1q_u8(ref + i));
        vst1q_u8(out + i, v);
        acc = png_cost16(acc, v);
    }
    *s = (int)vaddvq_u32(acc);
#endif
    return i;
}

static int
png_filter_average_simd(
    UINT8 *out, const UINT8 *cur, const UINT8 *prev, int n, int bpp, int i0, int *s
) {
    int i = i0;
#if defined(IMAGING_SIMD_SSE4)
    {
        __m128i acc = _mm_setzero_si128(), one = _mm_set1_epi8(1);
        for (; i < n - 15; i += 16) {
            __m128i a = _mm_loadu_si128((__m128i *)(cur + i - bpp));
            __m128i b = _mm_loadu_si128((__m128i *)(prev + i));
            // (a + b) / 2 rounded down: pavgb rounds up
            __m128i avg = _mm_sub_epi8(
                _mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), one)
            );
            __m128i v = _mm_sub_epi8(_mm_loadu_si128((__m128i *)(cur + i)), avg);
            _mm_storeu_si128((__m128i *)(out + i), v);
            acc = _mm_add_epi64(acc, png_cost16(v));
        }
        *s += png_sum64(acc);
    }
#elif defined(IMAGING_SIMD_NEON)
    {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i < n - 15; i += 16) {
            // vhaddq_u8 is (a + b) >> 1
            uint8x16_t avg = vhaddq_u8(vld1q_u8(cur + i - bpp), vld1q_u8(prev + i));
            uint8x16_t v = vsubq_u8(vld1q_u8(cur + i), avg);
            vst1q_u8(out + i, v);
            acc = png_cost16(acc, v);
        }
        *s += (int)vaddvq_u32(acc);
    }
#endif
    return i - i0;
}

static int
png_filter_paeth_simd(
    UINT8 *out, const UINT8 *cur, const UINT8 *prev, int n, int bpp, int i0, int *s
) {
    int i = i0;
#if defined(IMAGING_SIMD_SSE4)
    {
        __m128i acc = _mm_setzero_si128(), zero = _mm_setzero_si128();
        for (; i < n - 15; i += 16) {
            __m128i a8 = _mm_loadu_si128((__m128i *)(cur + i - bpp));
            __m128i b8 = _mm_loadu_si128((__m128i *)(prev + i));
            __m128i c8 = _mm_loadu_si128((__m128i *)(prev + i - bpp));
            __m128i pred[2];
            int h;
            for (h = 0; h < 2; h++) {
                __m128i a =
                    h ? _mm_unpackhi_epi8(a8, zero) : _mm_unpacklo_epi8(a8, zero);
                __m128i b =
                    h ? _mm_unpackhi_epi8(b8, zero) : _mm_unpacklo_epi8(b8, zero);
                __m128i c =
                    h ? _mm_unpackhi_epi8(c8, zero) : _mm_unpacklo_epi8(c8, zero);
                __m128i pa = _mm_abs_epi16(_mm_sub_epi16(b, c));
                __m128i pb = _mm_abs_epi16(_mm_sub_epi16(a, c));
                __m128i pc = _mm_abs_epi16(
                    _mm_sub_epi16(_mm_add_epi16(a, b), _mm_add_epi16(c, c))
                );
                // a if pa <= pb && pa <= pc, else b if pb <= pc, else c
                __m128i use_a = _mm_andnot_si128(
                    _mm_or_si128(_mm_cmpgt_epi16(pa, pb), _mm_cmpgt_epi16(pa, pc)),
                    _mm_cmpeq_epi16(a, a)
                );
                __m128i use_b =
                    _mm_andnot_si128(_mm_cmpgt_epi16(pb, pc), _mm_cmpeq_epi16(a, a));
                __m128i p = _mm_blendv_epi8(c, b, use_b);
                pred[h] = _mm_blendv_epi8(p, a, use_a);
            }
            __m128i v = _mm_sub_epi8(
                _mm_loadu_si128((__m128i *)(cur + i)),
                _mm_packus_epi16(pred[0], pred[1])
            );
            _mm_storeu_si128((__m128i *)(out + i), v);
            acc = _mm_add_epi64(acc, png_cost16(v));
        }
        *s += png_sum64(acc);
    }
#elif defined(IMAGING_SIMD_NEON)
    {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i < n - 15; i += 16) {
            uint8x16_t a = vld1q_u8(cur + i - bpp), b = vld1q_u8(prev + i);
            uint8x16_t c = vld1q_u8(prev + i - bpp);
            // |b - c| and |a - c| fit bytes; |a + b - 2c| needs 16 bits
            uint8x16_t pa8 = vabdq_u8(b, c), pb8 = vabdq_u8(a, c);
            int16x8_t pa_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pa8)));
            int16x8_t pa_hi = vreinterpretq_s16_u16(vmovl_high_u8(pa8));
            int16x8_t pb_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pb8)));
            int16x8_t pb_hi = vreinterpretq_s16_u16(vmovl_high_u8(pb8));
            // a + b - 2c = (a - c) + (b - c)
            int16x8_t ac_lo =
                vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(a), vget_low_u8(c)));
            int16x8_t ac_hi = vreinterpretq_s16_u16(vsubl_high_u8(a, c));
            int16x8_t bc_lo =
                vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(b), vget_low_u8(c)));
            int16x8_t bc_hi = vreinterpretq_s16_u16(vsubl_high_u8(b, c));
            int16x8_t pc_lo = vabsq_s16(vaddq_s16(ac_lo, bc_lo));
            int16x8_t pc_hi = vabsq_s16(vaddq_s16(ac_hi, bc_hi));
            uint8x16_t pc_ge_pa = vcombine_u8(
                vmovn_u16(vcgeq_s16(pc_lo, pa_lo)), vmovn_u16(vcgeq_s16(pc_hi, pa_hi))
            );
            uint8x16_t pc_ge_pb = vcombine_u8(
                vmovn_u16(vcgeq_s16(pc_lo, pb_lo)), vmovn_u16(vcgeq_s16(pc_hi, pb_hi))
            );
            uint8x16_t use_a = vandq_u8(vcleq_u8(pa8, pb8), pc_ge_pa);
            uint8x16_t use_b = pc_ge_pb;
            uint8x16_t pred = vbslq_u8(use_a, a, vbslq_u8(use_b, b, c));
            uint8x16_t v = vsubq_u8(vld1q_u8(cur + i), pred);
            vst1q_u8(out + i, v);
            acc = png_cost16(acc, v);
        }
        *s += (int)vaddvq_u32(acc);
    }
#endif
    return i - i0;
}

/* The PNG decoder's "up" unfilter (ZipDecode.c), a whole row. */
/* Undoes the PNG "up" filter 16 bytes at a time. The other filters depend
   on the pixel to the left: one pixel per vector step (libpng's way) was
   slower than the scalar loops, whose 3 or 4 byte positions run as
   independent chains (2560x1600 RGB, Paeth 80 -> 152 ms), so they stay. */
static void
png_unfilter_up_simd(UINT8 *row, const UINT8 *prev, int len) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    for (; i + 16 <= len; i += 16) {
        _mm_storeu_si128(
            (__m128i *)(row + i),
            _mm_add_epi8(
                _mm_loadu_si128((const __m128i *)(row + i)),
                _mm_loadu_si128((const __m128i *)(prev + i))
            )
        );
    }
#else
    for (; i + 16 <= len; i += 16) {
        vst1q_u8(row + i, vaddq_u8(vld1q_u8(row + i), vld1q_u8(prev + i)));
    }
#endif
    for (; i < len; i++) {
        row[i] += prev[i];
    }
}

static const ImagingSIMDZipKernels zip_simd_kernels = {
    .cost_none = png_cost_none_simd,
    .filter_diff = png_filter_diff_simd,
    .filter_average = png_filter_average_simd,
    .filter_paeth = png_filter_paeth_simd,
    .unfilter_up = png_unfilter_up_simd,
};

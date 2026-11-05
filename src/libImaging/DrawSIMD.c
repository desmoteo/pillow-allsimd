/* Drawing (Draw.c), compiled into SIMDKernels.c: the blend of an RGBA ink
   into a row of pixels, for the pixels its vector loop can; returns how
   many. (The plain fill of a span is a loop in Draw.c: a kernel call per
   span cost more than it gave.) */

static int
draw_hline_rgba_simd(UINT8 *out, int n, const UINT8 *in) {
    int x = 0;
    /* BLEND with the ink's terms computed once:
       DIV255(out * (255 - a) + in * a). Every term fits 16
       bits, so vectors of 16-bit lanes give the same bytes.
       The loop below does the last pixels. */
    unsigned int a = in[3], na = 255 - a;
    unsigned int c0 = in[0] * a, c1 = in[1] * a, c2 = in[2] * a;
#if defined(IMAGING_SIMD_SSE4)
    __m128i vna = _mm_set1_epi16((short)na);
    __m128i vc = _mm_set_epi16(
        0, (short)c2, (short)c1, (short)c0, 0, (short)c2, (short)c1, (short)c0
    );
    __m128i r128 = _mm_set1_epi16(128), zero = _mm_setzero_si128();
    __m128i alpha = _mm_set1_epi32((int)0xff000000);
    for (; x + 4 <= n; x += 4, out += 16) {
        __m128i px = _mm_loadu_si128((__m128i *)out);
        __m128i lo = _mm_unpacklo_epi8(px, zero);
        __m128i hi = _mm_unpackhi_epi8(px, zero);
        lo = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(lo, vna), vc), r128);
        hi = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(hi, vna), vc), r128);
        lo = _mm_srli_epi16(_mm_add_epi16(lo, _mm_srli_epi16(lo, 8)), 8);
        hi = _mm_srli_epi16(_mm_add_epi16(hi, _mm_srli_epi16(hi, 8)), 8);
        _mm_storeu_si128(
            (__m128i *)out, _mm_blendv_epi8(_mm_packus_epi16(lo, hi), px, alpha)
        );
    }
#elif defined(IMAGING_SIMD_NEON)
    uint16x8_t vna = vdupq_n_u16(na);
    const uint16_t cs[8] = {c0, c1, c2, 0, c0, c1, c2, 0};
    uint16x8_t vc = vld1q_u16(cs);
    uint8x16_t alpha = vreinterpretq_u8_u32(vdupq_n_u32(0xff000000));
    for (; x + 4 <= n; x += 4, out += 16) {
        uint8x16_t px = vld1q_u8(out);
        uint16x8_t lo = vmlaq_u16(vc, vmovl_u8(vget_low_u8(px)), vna);
        uint16x8_t hi = vmlaq_u16(vc, vmovl_high_u8(px), vna);
        lo = vaddq_u16(lo, vdupq_n_u16(128));
        hi = vaddq_u16(hi, vdupq_n_u16(128));
        // (x + (x >> 8)) >> 8
        uint8x16_t res = vcombine_u8(
            vshrn_n_u16(vsraq_n_u16(lo, lo, 8), 8),
            vshrn_n_u16(vsraq_n_u16(hi, hi, 8), 8)
        );
        vst1q_u8(out, vbslq_u8(alpha, px, res));
    }
#endif
    return x;
}

static const ImagingSIMDDrawKernels draw_simd_kernels = {
    .hline_rgba = draw_hline_rgba_simd,
};

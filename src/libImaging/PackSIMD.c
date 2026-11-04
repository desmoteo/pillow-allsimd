/* The packers of RGB and BGR triplets from 4-byte pixels (Pack.c), compiled
   into SIMDKernels.c: each packs the pixels its vector loop can and returns
   how many; Pack.c does the rest. */

static int
pack_rgb_simd(UINT8 *out, const UINT8 *in, int pixels) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    /* 4 pixels per step; the 16-byte store writes 4 bytes past them, so
       stop while at least 6 pixels (18 bytes) are left. */
    __m128i to_rgb =
        _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    for (; i < pixels - 5; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(in + i * 4));
        _mm_storeu_si128((__m128i *)out, _mm_shuffle_epi8(v, to_rgb));
        out += 12;
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; i < pixels - 15; i += 16) {
        uint8x16x4_t rgbx = vld4q_u8(in + i * 4);
        uint8x16x3_t rgb = {{rgbx.val[0], rgbx.val[1], rgbx.val[2]}};
        vst3q_u8(out, rgb);
        out += 48;
    }
#endif
    return i;
}

static int
pack_bgr_simd(UINT8 *out, const UINT8 *in, int pixels) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    /* As ImagingPackRGB, with the bytes of each pixel reversed. */
    __m128i to_bgr =
        _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
    for (; i < pixels - 5; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(in + i * 4));
        _mm_storeu_si128((__m128i *)out, _mm_shuffle_epi8(v, to_bgr));
        out += 12;
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; i < pixels - 15; i += 16) {
        uint8x16x4_t rgbx = vld4q_u8(in + i * 4);
        uint8x16x3_t bgr = {{rgbx.val[2], rgbx.val[1], rgbx.val[0]}};
        vst3q_u8(out, bgr);
        out += 48;
    }
#endif
    return i;
}

static const ImagingSIMDPackKernels pack_simd_kernels = {
    .rgb = pack_rgb_simd,
    .bgr = pack_bgr_simd,
};

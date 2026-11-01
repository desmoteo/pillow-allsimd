/* The unpackers of RGB and BGR triplets into 4-byte pixels (Unpack.c),
   compiled into SIMDKernels.c: each unpacks the pixels its vector loop can
   and returns how many; Unpack.c does the rest. */

static int
unpack_rgb_simd(UINT8 *_out, const UINT8 *in, int pixels) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    /* 4 pixels per step; a 16-byte load reads 4 bytes past them, so stop
       while at least 6 pixels (18 bytes) are left. */
    __m128i to_rgbx =
        _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    __m128i alpha = _mm_set1_epi32((int)MASK_UINT32_CHANNEL_3);
    for (; i < pixels - 5; i += 4) {
        __m128i v = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)in), to_rgbx);
        _mm_storeu_si128((__m128i *)_out, _mm_or_si128(v, alpha));
        in += 12;
        _out += 16;
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; i < pixels - 15; i += 16) {
        uint8x16x3_t rgb = vld3q_u8(in);
        uint8x16x4_t rgbx = {{rgb.val[0], rgb.val[1], rgb.val[2], vdupq_n_u8(255)}};
        vst4q_u8(_out, rgbx);
        in += 48;
        _out += 64;
    }
#endif
    return i;
}

static int
unpack_bgr_simd(UINT8 *_out, const UINT8 *in, int pixels) {
    int i = 0;
#if defined(IMAGING_SIMD_SSE4)
    /* As ImagingUnpackRGB, with the bytes of each pixel reversed. */
    __m128i to_rgbx =
        _mm_setr_epi8(2, 1, 0, -1, 5, 4, 3, -1, 8, 7, 6, -1, 11, 10, 9, -1);
    __m128i alpha = _mm_set1_epi32((int)MASK_UINT32_CHANNEL_3);
    for (; i < pixels - 5; i += 4) {
        __m128i v = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)in), to_rgbx);
        _mm_storeu_si128((__m128i *)_out, _mm_or_si128(v, alpha));
        in += 12;
        _out += 16;
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; i < pixels - 15; i += 16) {
        uint8x16x3_t bgr = vld3q_u8(in);
        uint8x16x4_t rgbx = {{bgr.val[2], bgr.val[1], bgr.val[0], vdupq_n_u8(255)}};
        vst4q_u8(_out, rgbx);
        in += 48;
        _out += 64;
    }
#endif
    return i;
}

static const ImagingSIMDUnpackKernels unpack_simd_kernels = {
    .rgb = unpack_rgb_simd,
    .bgr = unpack_bgr_simd,
};

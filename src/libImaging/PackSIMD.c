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

/* For pixels of 4 bytes: out[i * 4 + k] = in[i * 4 + pk], or 0 for
   PERMUTE_00 and 255 for PERMUTE_FF. Does the pixels it can in vectors and
   returns how many. The indices are constants at Pack.c's and Unpack.c's
   call sites; the shuffle is built per row here. */
static int
pack_permute4_simd(
    UINT8 *out, const UINT8 *in, int pixels, int p0, int p1, int p2, int p3
) {
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
#endif
    return i;
}

static const ImagingSIMDPackKernels pack_simd_kernels = {
    .permute4 = pack_permute4_simd,
    .rgb = pack_rgb_simd,
    .bgr = pack_bgr_simd,
};

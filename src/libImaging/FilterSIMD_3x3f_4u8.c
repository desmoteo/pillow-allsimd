/* SSE4 on every x86 tier: an AVX2 version that summed two columns in the
   two 128-bit halves and added them at the end was 20% slower than this on
   a Core Ultra 9 285H, and its rounding differed from this kernel's, which
   would have made a dispatch build's pixels depend on the CPU. */
static void
ImagingFilter3x3f_4u8(Imaging imOut, Imaging im, const float *kernel, float offset) {
#define MM_KERNEL1x3_LOAD(row, x)                           \
    pix0##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in1[x])); \
    pix1##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in0[x])); \
    pix2##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in_1[x]));

#define MM_KERNEL1x3_SUM(row, kindex)                                            \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix0##row, _mm_set1_ps(kernel[0 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, _mm_set1_ps(kernel[3 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix2##row, _mm_set1_ps(kernel[6 + kindex])));

    int x = 0, y = 0;

    memcpy(imOut->image32[0], im->image32[0], im->linesize);

    for (y = 1; y < im->ysize - 1; y++) {
        INT32 *in_1 = im->image32[y - 1];
        INT32 *in0 = im->image32[y];
        INT32 *in1 = im->image32[y + 1];
        INT32 *out = imOut->image32[y];

        __m128 pix00, pix10, pix20;
        __m128 pix01, pix11, pix21;
        __m128 pix02, pix12, pix22;
        MM_KERNEL1x3_LOAD(0, 0);
        MM_KERNEL1x3_LOAD(1, 1);

        out[0] = in0[0];
        x = 1;
        for (; x < im->xsize - 1 - 2; x += 3) {
            __m128 ss;
            __m128i ssi0, ssi1, ssi2;

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x3_SUM(0, 0);
            MM_KERNEL1x3_SUM(1, 1);
            MM_KERNEL1x3_LOAD(2, x + 1);
            MM_KERNEL1x3_SUM(2, 2);
            ssi0 = _mm_cvtps_epi32(ss);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x3_SUM(1, 0);
            MM_KERNEL1x3_SUM(2, 1);
            MM_KERNEL1x3_LOAD(0, x + 2);
            MM_KERNEL1x3_SUM(0, 2);
            ssi1 = _mm_cvtps_epi32(ss);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x3_SUM(2, 0);
            MM_KERNEL1x3_SUM(0, 1);
            MM_KERNEL1x3_LOAD(1, x + 3);
            MM_KERNEL1x3_SUM(1, 2);
            ssi2 = _mm_cvtps_epi32(ss);

            ssi0 = _mm_packs_epi32(ssi0, ssi1);
            ssi1 = _mm_packs_epi32(ssi2, ssi2);
            ssi0 = _mm_packus_epi16(ssi0, ssi1);
            _mm_storeu_si128((__m128i *)&out[x], ssi0);
        }
        for (; x < im->xsize - 1; x++) {
            __m128 ss = _mm_set1_ps(offset);
            __m128i ssi0;
            MM_KERNEL1x3_LOAD(0, x - 1);
            MM_KERNEL1x3_SUM(0, 0);
            MM_KERNEL1x3_LOAD(0, x + 0);
            MM_KERNEL1x3_SUM(0, 1);
            MM_KERNEL1x3_LOAD(0, x + 1);
            MM_KERNEL1x3_SUM(0, 2);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        out[x] = in0[x];
    }
    memcpy(imOut->image32[y], im->image32[y], im->linesize);

#undef MM_KERNEL1x3_LOAD
#undef MM_KERNEL1x3_SUM
}

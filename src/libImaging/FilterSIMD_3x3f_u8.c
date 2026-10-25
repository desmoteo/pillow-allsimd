void
ImagingFilter3x3f_u8(Imaging imOut, Imaging im, const float *kernel, float offset) {
#define MM_KERNEL1x3_SUM1(ss, row, kernel)                       \
    ss = _mm_mul_ps(pix0##row, kernel0##kernel);                 \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, kernel1##kernel)); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix2##row, kernel2##kernel));

#define MM_KERNEL1x3_SUM1_2(ss, row, kernel)                     \
    ss = _mm_mul_ps(pix3##row, kernel0##kernel);                 \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix0##row, kernel1##kernel)); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, kernel2##kernel));

#define MM_KERNEL1x3_LOAD(row, x)                           \
    pix0##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in1[x])); \
    pix1##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in0[x])); \
    pix2##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in_1[x]));

    int x, y;
    __m128 kernel00 = _mm_set_ps(0, kernel[2], kernel[1], kernel[0]);
    __m128 kernel10 = _mm_set_ps(0, kernel[5], kernel[4], kernel[3]);
    __m128 kernel20 = _mm_set_ps(0, kernel[8], kernel[7], kernel[6]);
    __m128 kernel01 = _mm_set_ps(kernel[2], kernel[1], kernel[0], 0);
    __m128 kernel11 = _mm_set_ps(kernel[5], kernel[4], kernel[3], 0);
    __m128 kernel21 = _mm_set_ps(kernel[8], kernel[7], kernel[6], 0);
    __m128 mm_offset = _mm_set1_ps(offset);

    memcpy(imOut->image8[0], im->image8[0], im->linesize);
    y = 1;
    for (; y < im->ysize - 1 - 1; y += 2) {
        UINT8 *in_1 = im->image8[y - 1];
        UINT8 *in0 = im->image8[y];
        UINT8 *in1 = im->image8[y + 1];
        UINT8 *in2 = im->image8[y + 2];
        UINT8 *out0 = imOut->image8[y];
        UINT8 *out1 = imOut->image8[y + 1];

        out0[0] = in0[0];
        out1[0] = in1[0];
        x = 1;
#if defined(IMAGING_SIMD_AVX2)
        /* 8 pixels per step: each 128-bit lane repeats the SSE4 step below
           for 4 pixels, x to x + 3 and x + 4 to x + 7, with the same
           operations in the same order. */
        {
            __m256 k00 = _mm256_broadcast_ps(&kernel00),
                   k01 = _mm256_broadcast_ps(&kernel01);
            __m256 k10 = _mm256_broadcast_ps(&kernel10),
                   k11 = _mm256_broadcast_ps(&kernel11);
            __m256 k20 = _mm256_broadcast_ps(&kernel20),
                   k21 = _mm256_broadcast_ps(&kernel21);
            __m256 off8 = _mm256_set1_ps(offset);
#define MM256_LOAD(p, x) _mm256_cvtepi32_ps(mm256_cvtepu8_epi32(&(p)[x]))
#define MM256_SUM3(a, b, c, ka, kb, kc)                            \
    _mm256_add_ps(                                                 \
        _mm256_add_ps(_mm256_mul_ps(a, ka), _mm256_mul_ps(b, kb)), \
        _mm256_mul_ps(c, kc)                                       \
    )
            for (; x < im->xsize - 8; x += 8) {
                __m256 p0, p1, p2, p3, s0, s1, s2, s3, s4, s5;
                __m256i si;
                __m128i packed;

                p0 = MM256_LOAD(in1, x - 1);
                p1 = MM256_LOAD(in0, x - 1);
                p2 = MM256_LOAD(in_1, x - 1);
                s0 = MM256_SUM3(p0, p1, p2, k00, k10, k20);
                s1 = MM256_SUM3(p0, p1, p2, k01, k11, k21);
                s0 = _mm256_hadd_ps(s0, s1);
                p3 = MM256_LOAD(in2, x - 1);
                s3 = MM256_SUM3(p3, p0, p1, k00, k10, k20);
                s4 = MM256_SUM3(p3, p0, p1, k01, k11, k21);
                s3 = _mm256_hadd_ps(s3, s4);

                p0 = MM256_LOAD(in1, x + 1);
                p1 = MM256_LOAD(in0, x + 1);
                p2 = MM256_LOAD(in_1, x + 1);
                s1 = MM256_SUM3(p0, p1, p2, k00, k10, k20);
                s2 = MM256_SUM3(p0, p1, p2, k01, k11, k21);
                s1 = _mm256_hadd_ps(s1, s2);
                p3 = MM256_LOAD(in2, x + 1);
                s4 = MM256_SUM3(p3, p0, p1, k00, k10, k20);
                s5 = MM256_SUM3(p3, p0, p1, k01, k11, k21);
                s4 = _mm256_hadd_ps(s4, s5);

                si = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_hadd_ps(s0, s1), off8));
                packed = _mm_packs_epi32(
                    _mm256_castsi256_si128(si), _mm256_extracti128_si256(si, 1)
                );
                _mm_storel_epi64((__m128i *)&out0[x], _mm_packus_epi16(packed, packed));

                si = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_hadd_ps(s3, s4), off8));
                packed = _mm_packs_epi32(
                    _mm256_castsi256_si128(si), _mm256_extracti128_si256(si, 1)
                );
                _mm_storel_epi64((__m128i *)&out1[x], _mm_packus_epi16(packed, packed));
            }
#undef MM256_LOAD
#undef MM256_SUM3
        }
#endif
        for (; x < im->xsize - 1 - 3; x += 4) {
            __m128 ss0, ss1, ss2, ss3, ss4, ss5;
            __m128 pix00, pix10, pix20, pix30;
            __m128i ssi0;

            MM_KERNEL1x3_LOAD(0, x - 1);
            MM_KERNEL1x3_SUM1(ss0, 0, 0);
            MM_KERNEL1x3_SUM1(ss1, 0, 1);
            ss0 = _mm_hadd_ps(ss0, ss1);
            pix30 = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in2[x - 1]));
            MM_KERNEL1x3_SUM1_2(ss3, 0, 0);
            MM_KERNEL1x3_SUM1_2(ss4, 0, 1);
            ss3 = _mm_hadd_ps(ss3, ss4);

            MM_KERNEL1x3_LOAD(0, x + 1);
            MM_KERNEL1x3_SUM1(ss1, 0, 0);
            MM_KERNEL1x3_SUM1(ss2, 0, 1);
            ss1 = _mm_hadd_ps(ss1, ss2);
            pix30 = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in2[x + 1]));
            MM_KERNEL1x3_SUM1_2(ss4, 0, 0);
            MM_KERNEL1x3_SUM1_2(ss5, 0, 1);
            ss4 = _mm_hadd_ps(ss4, ss5);

            ss0 = _mm_hadd_ps(ss0, ss1);
            ss0 = _mm_add_ps(ss0, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss0);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            *((UINT32 *)&out0[x]) = _mm_cvtsi128_si32(ssi0);

            ss3 = _mm_hadd_ps(ss3, ss4);
            ss3 = _mm_add_ps(ss3, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss3);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            *((UINT32 *)&out1[x]) = _mm_cvtsi128_si32(ssi0);
        }
        for (; x < im->xsize - 1; x++) {
            __m128 ss0, ss1;
            __m128 pix00, pix10, pix20, pix30;
            __m128i ssi0;

            pix00 = _mm_set_ps(0, in1[x + 1], in1[x], in1[x - 1]);
            pix10 = _mm_set_ps(0, in0[x + 1], in0[x], in0[x - 1]);
            pix20 = _mm_set_ps(0, in_1[x + 1], in_1[x], in_1[x - 1]);
            pix30 = _mm_set_ps(0, in2[x + 1], in2[x], in2[x - 1]);
            MM_KERNEL1x3_SUM1(ss0, 0, 0);
            MM_KERNEL1x3_SUM1_2(ss1, 0, 0);

            ss0 = _mm_hadd_ps(ss0, ss0);
            ss0 = _mm_hadd_ps(ss0, ss0);
            ss0 = _mm_add_ps(ss0, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss0);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out0[x] = _mm_cvtsi128_si32(ssi0);

            ss1 = _mm_hadd_ps(ss1, ss1);
            ss1 = _mm_hadd_ps(ss1, ss1);
            ss1 = _mm_add_ps(ss1, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss1);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out1[x] = _mm_cvtsi128_si32(ssi0);
        }
        out0[x] = in0[x];
        out1[x] = in1[x];
    }
    for (; y < im->ysize - 1; y++) {
        UINT8 *in_1 = im->image8[y - 1];
        UINT8 *in0 = im->image8[y];
        UINT8 *in1 = im->image8[y + 1];
        UINT8 *out = imOut->image8[y];

        out[0] = in0[0];
        x = 1;
        for (; x < im->xsize - 2; x++) {
            __m128 ss;
            __m128 pix00, pix10, pix20;
            __m128i ssi0;

            MM_KERNEL1x3_LOAD(0, x - 1);
            MM_KERNEL1x3_SUM1(ss, 0, 0);

            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_add_ps(ss, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        for (; x < im->xsize - 1; x++) {
            __m128 ss;
            __m128 pix00, pix10, pix20;
            __m128i ssi0;

            pix00 = _mm_set_ps(0, in1[x + 1], in1[x], in1[x - 1]);
            pix10 = _mm_set_ps(0, in0[x + 1], in0[x], in0[x - 1]);
            pix20 = _mm_set_ps(0, in_1[x + 1], in_1[x], in_1[x - 1]);
            MM_KERNEL1x3_SUM1(ss, 0, 0);

            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_add_ps(ss, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        out[x] = in0[x];
    }
    memcpy(imOut->image8[y], im->image8[y], im->linesize);

#undef MM_KERNEL1x3_SUM1
#undef MM_KERNEL1x3_SUM1_2
#undef MM_KERNEL1x3_LOAD
}

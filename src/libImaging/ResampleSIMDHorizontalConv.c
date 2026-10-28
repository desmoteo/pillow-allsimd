#if defined(IMAGING_SIMD_SSE4)

void
ImagingResampleHorizontalConvolution8u4x(
    UINT32 *restrict lineOut0,
    UINT32 *restrict lineOut1,
    UINT32 *restrict lineOut2,
    UINT32 *restrict lineOut3,
    UINT32 *restrict lineIn0,
    UINT32 *restrict lineIn1,
    UINT32 *restrict lineIn2,
    UINT32 *restrict lineIn3,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    int xmin, xmax, xx, x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        x = 0;

#if defined(IMAGING_SIMD_AVX2)
        {
            __m256i sss0, sss1;
            __m256i zero = _mm256_setzero_si256();
            __m256i initial = _mm256_set1_epi32(1 << (coefs_precision - 1));
            sss0 = initial;
            sss1 = initial;

            for (; x < xmax - 3; x += 4) {
                __m256i pix, mmk0, mmk1, source;

                mmk0 = _mm256_set1_epi32(*(INT32 *)&k[x]);
                mmk1 = _mm256_set1_epi32(*(INT32 *)&k[x + 2]);

                source = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        _mm_loadu_si128((__m128i *)&lineIn0[x + xmin])
                    ),
                    _mm_loadu_si128((__m128i *)&lineIn1[x + xmin]),
                    1
                );
                pix = _mm256_shuffle_epi8(
                    source,
                    _mm256_set_epi8(
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0,
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0
                    )
                );
                sss0 = mm256_dpwssd(sss0, pix, mmk0);
                pix = _mm256_shuffle_epi8(
                    source,
                    _mm256_set_epi8(
                        -1,
                        15,
                        -1,
                        11,
                        -1,
                        14,
                        -1,
                        10,
                        -1,
                        13,
                        -1,
                        9,
                        -1,
                        12,
                        -1,
                        8,
                        -1,
                        15,
                        -1,
                        11,
                        -1,
                        14,
                        -1,
                        10,
                        -1,
                        13,
                        -1,
                        9,
                        -1,
                        12,
                        -1,
                        8
                    )
                );
                sss0 = mm256_dpwssd(sss0, pix, mmk1);

                source = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        _mm_loadu_si128((__m128i *)&lineIn2[x + xmin])
                    ),
                    _mm_loadu_si128((__m128i *)&lineIn3[x + xmin]),
                    1
                );
                pix = _mm256_shuffle_epi8(
                    source,
                    _mm256_set_epi8(
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0,
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0
                    )
                );
                sss1 = mm256_dpwssd(sss1, pix, mmk0);
                pix = _mm256_shuffle_epi8(
                    source,
                    _mm256_set_epi8(
                        -1,
                        15,
                        -1,
                        11,
                        -1,
                        14,
                        -1,
                        10,
                        -1,
                        13,
                        -1,
                        9,
                        -1,
                        12,
                        -1,
                        8,
                        -1,
                        15,
                        -1,
                        11,
                        -1,
                        14,
                        -1,
                        10,
                        -1,
                        13,
                        -1,
                        9,
                        -1,
                        12,
                        -1,
                        8
                    )
                );
                sss1 = mm256_dpwssd(sss1, pix, mmk1);
            }

            for (; x < xmax - 1; x += 2) {
                __m256i pix, mmk;

                mmk = _mm256_set1_epi32(*(INT32 *)&k[x]);

                pix = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        _mm_loadl_epi64((__m128i *)&lineIn0[x + xmin])
                    ),
                    _mm_loadl_epi64((__m128i *)&lineIn1[x + xmin]),
                    1
                );
                pix = _mm256_shuffle_epi8(
                    pix,
                    _mm256_set_epi8(
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0,
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0
                    )
                );
                sss0 = mm256_dpwssd(sss0, pix, mmk);

                pix = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        _mm_loadl_epi64((__m128i *)&lineIn2[x + xmin])
                    ),
                    _mm_loadl_epi64((__m128i *)&lineIn3[x + xmin]),
                    1
                );
                pix = _mm256_shuffle_epi8(
                    pix,
                    _mm256_set_epi8(
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0,
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0
                    )
                );
                sss1 = mm256_dpwssd(sss1, pix, mmk);
            }

            for (; x < xmax; x++) {
                __m256i pix, mmk;

                // [16] xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0
                mmk = _mm256_set1_epi32(k[x]);

                // [16] xx a0 xx b0 xx g0 xx r0 xx a0 xx b0 xx g0 xx r0
                pix = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(mm_cvtepu8_epi32(&lineIn0[x + xmin])),
                    mm_cvtepu8_epi32(&lineIn1[x + xmin]),
                    1
                );
                sss0 = mm256_dpwssd(sss0, pix, mmk);

                pix = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(mm_cvtepu8_epi32(&lineIn2[x + xmin])),
                    mm_cvtepu8_epi32(&lineIn3[x + xmin]),
                    1
                );
                sss1 = mm256_dpwssd(sss1, pix, mmk);
            }

            sss0 = _mm256_srai_epi32(sss0, coefs_precision);
            sss1 = _mm256_srai_epi32(sss1, coefs_precision);
            sss0 = _mm256_packs_epi32(sss0, zero);
            sss1 = _mm256_packs_epi32(sss1, zero);
            sss0 = _mm256_packus_epi16(sss0, zero);
            sss1 = _mm256_packus_epi16(sss1, zero);
            lineOut0[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 0));
            lineOut1[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
            lineOut2[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 0));
            lineOut3[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));
        }
#else
        {
            __m128i sss0, sss1, sss2, sss3;
            __m128i initial = _mm_set1_epi32(1 << (coefs_precision - 1));
            sss0 = initial;
            sss1 = initial;
            sss2 = initial;
            sss3 = initial;

            for (; x < xmax - 3; x += 4) {
                __m128i pix, mmk_lo, mmk_hi, source;
                __m128i mask_lo = _mm_set_epi8(
                    -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
                );
                __m128i mask_hi = _mm_set_epi8(
                    -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8
                );

                mmk_lo = _mm_set1_epi32(*(INT32 *)&k[x]);
                mmk_hi = _mm_set1_epi32(*(INT32 *)&k[x + 2]);

                // [8] a3 b3 g3 r3 a2 b2 g2 r2 a1 b1 g1 r1 a0 b0 g0 r0
                source = _mm_loadu_si128((__m128i *)&lineIn0[x + xmin]);
                // [16] a1 a0 b1 b0 g1 g0 r1 r0
                pix = _mm_shuffle_epi8(source, mask_lo);
                sss0 = mm_dpwssd(sss0, pix, mmk_lo);
                // [16] a3 a2 b3 b2 g3 g2 r3 r2
                pix = _mm_shuffle_epi8(source, mask_hi);
                sss0 = mm_dpwssd(sss0, pix, mmk_hi);

                source = _mm_loadu_si128((__m128i *)&lineIn1[x + xmin]);
                pix = _mm_shuffle_epi8(source, mask_lo);
                sss1 = mm_dpwssd(sss1, pix, mmk_lo);
                pix = _mm_shuffle_epi8(source, mask_hi);
                sss1 = mm_dpwssd(sss1, pix, mmk_hi);

                source = _mm_loadu_si128((__m128i *)&lineIn2[x + xmin]);
                pix = _mm_shuffle_epi8(source, mask_lo);
                sss2 = mm_dpwssd(sss2, pix, mmk_lo);
                pix = _mm_shuffle_epi8(source, mask_hi);
                sss2 = mm_dpwssd(sss2, pix, mmk_hi);

                source = _mm_loadu_si128((__m128i *)&lineIn3[x + xmin]);
                pix = _mm_shuffle_epi8(source, mask_lo);
                sss3 = mm_dpwssd(sss3, pix, mmk_lo);
                pix = _mm_shuffle_epi8(source, mask_hi);
                sss3 = mm_dpwssd(sss3, pix, mmk_hi);
            }

            for (; x < xmax - 1; x += 2) {
                __m128i pix, mmk;
                __m128i mask = _mm_set_epi8(
                    -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
                );

                // [16] k1 k0 k1 k0 k1 k0 k1 k0
                mmk = _mm_set1_epi32(*(INT32 *)&k[x]);

                // [8] x x x x x x x x a1 b1 g1 r1 a0 b0 g0 r0
                pix = _mm_loadl_epi64((__m128i *)&lineIn0[x + xmin]);
                // [16] a1 a0 b1 b0 g1 g0 r1 r0
                pix = _mm_shuffle_epi8(pix, mask);
                sss0 = mm_dpwssd(sss0, pix, mmk);

                pix = _mm_loadl_epi64((__m128i *)&lineIn1[x + xmin]);
                pix = _mm_shuffle_epi8(pix, mask);
                sss1 = mm_dpwssd(sss1, pix, mmk);

                pix = _mm_loadl_epi64((__m128i *)&lineIn2[x + xmin]);
                pix = _mm_shuffle_epi8(pix, mask);
                sss2 = mm_dpwssd(sss2, pix, mmk);

                pix = _mm_loadl_epi64((__m128i *)&lineIn3[x + xmin]);
                pix = _mm_shuffle_epi8(pix, mask);
                sss3 = mm_dpwssd(sss3, pix, mmk);
            }

            for (; x < xmax; x++) {
                __m128i pix, mmk;
                // [16] xx k0 xx k0 xx k0 xx k0
                mmk = _mm_set1_epi32(k[x]);
                // [16] xx a0 xx b0 xx g0 xx r0
                pix = mm_cvtepu8_epi32(&lineIn0[x + xmin]);
                sss0 = mm_dpwssd(sss0, pix, mmk);

                pix = mm_cvtepu8_epi32(&lineIn1[x + xmin]);
                sss1 = mm_dpwssd(sss1, pix, mmk);

                pix = mm_cvtepu8_epi32(&lineIn2[x + xmin]);
                sss2 = mm_dpwssd(sss2, pix, mmk);

                pix = mm_cvtepu8_epi32(&lineIn3[x + xmin]);
                sss3 = mm_dpwssd(sss3, pix, mmk);
            }

            sss0 = _mm_srai_epi32(sss0, coefs_precision);
            sss1 = _mm_srai_epi32(sss1, coefs_precision);
            sss2 = _mm_srai_epi32(sss2, coefs_precision);
            sss3 = _mm_srai_epi32(sss3, coefs_precision);
            sss0 = _mm_packs_epi32(sss0, sss0);
            sss1 = _mm_packs_epi32(sss1, sss1);
            sss2 = _mm_packs_epi32(sss2, sss2);
            sss3 = _mm_packs_epi32(sss3, sss3);
            lineOut0[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss0, sss0));
            lineOut1[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss1, sss1));
            lineOut2[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss2, sss2));
            lineOut3[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss3, sss3));
        }
#endif
    }
}

void
ImagingResampleHorizontalConvolution8u(
    UINT32 *restrict lineOut,
    UINT32 *restrict lineIn,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    int xmin, xmax, xx, x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        __m128i sss;
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        x = 0;

#if defined(IMAGING_SIMD_AVX2)

        if (xmax < 8) {
            sss = _mm_set1_epi32(1 << (coefs_precision - 1));
        } else {
            // Lower part will be added to higher, use only half of the error
            __m256i sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

            for (; x < xmax - 7; x += 8) {
                __m256i pix, mmk, source;
                __m128i tmp = _mm_loadu_si128((__m128i *)&k[x]);
                __m256i ksource =
                    _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

                source = _mm256_loadu_si256((__m256i *)&lineIn[x + xmin]);

                pix = _mm256_shuffle_epi8(
                    source,
                    _mm256_set_epi8(
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0,
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0
                    )
                );
                mmk = _mm256_shuffle_epi8(
                    ksource,
                    _mm256_set_epi8(
                        11,
                        10,
                        9,
                        8,
                        11,
                        10,
                        9,
                        8,
                        11,
                        10,
                        9,
                        8,
                        11,
                        10,
                        9,
                        8,
                        3,
                        2,
                        1,
                        0,
                        3,
                        2,
                        1,
                        0,
                        3,
                        2,
                        1,
                        0,
                        3,
                        2,
                        1,
                        0
                    )
                );
                sss256 = mm256_dpwssd(sss256, pix, mmk);

                pix = _mm256_shuffle_epi8(
                    source,
                    _mm256_set_epi8(
                        -1,
                        15,
                        -1,
                        11,
                        -1,
                        14,
                        -1,
                        10,
                        -1,
                        13,
                        -1,
                        9,
                        -1,
                        12,
                        -1,
                        8,
                        -1,
                        15,
                        -1,
                        11,
                        -1,
                        14,
                        -1,
                        10,
                        -1,
                        13,
                        -1,
                        9,
                        -1,
                        12,
                        -1,
                        8
                    )
                );
                mmk = _mm256_shuffle_epi8(
                    ksource,
                    _mm256_set_epi8(
                        15,
                        14,
                        13,
                        12,
                        15,
                        14,
                        13,
                        12,
                        15,
                        14,
                        13,
                        12,
                        15,
                        14,
                        13,
                        12,
                        7,
                        6,
                        5,
                        4,
                        7,
                        6,
                        5,
                        4,
                        7,
                        6,
                        5,
                        4,
                        7,
                        6,
                        5,
                        4
                    )
                );
                sss256 = mm256_dpwssd(sss256, pix, mmk);
            }

            for (; x < xmax - 3; x += 4) {
                __m256i pix, mmk, source;
                __m128i tmp = _mm_loadl_epi64((__m128i *)&k[x]);
                __m256i ksource =
                    _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

                tmp = _mm_loadu_si128((__m128i *)&lineIn[x + xmin]);
                source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

                pix = _mm256_shuffle_epi8(
                    source,
                    _mm256_set_epi8(
                        -1,
                        15,
                        -1,
                        11,
                        -1,
                        14,
                        -1,
                        10,
                        -1,
                        13,
                        -1,
                        9,
                        -1,
                        12,
                        -1,
                        8,
                        -1,
                        7,
                        -1,
                        3,
                        -1,
                        6,
                        -1,
                        2,
                        -1,
                        5,
                        -1,
                        1,
                        -1,
                        4,
                        -1,
                        0
                    )
                );
                mmk = _mm256_shuffle_epi8(
                    ksource,
                    _mm256_set_epi8(
                        7,
                        6,
                        5,
                        4,
                        7,
                        6,
                        5,
                        4,
                        7,
                        6,
                        5,
                        4,
                        7,
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        0,
                        3,
                        2,
                        1,
                        0,
                        3,
                        2,
                        1,
                        0,
                        3,
                        2,
                        1,
                        0
                    )
                );
                sss256 = mm256_dpwssd(sss256, pix, mmk);
            }

            sss = _mm_add_epi32(
                _mm256_extracti128_si256(sss256, 0), _mm256_extracti128_si256(sss256, 1)
            );
        }

#else

        sss = _mm_set1_epi32(1 << (coefs_precision - 1));

        for (; x < xmax - 7; x += 8) {
            __m128i pix, mmk, source;
            __m128i ksource = _mm_loadu_si128((__m128i *)&k[x]);

            source = _mm_loadu_si128((__m128i *)&lineIn[x + xmin]);

            pix = _mm_shuffle_epi8(
                source,
                _mm_set_epi8(-1, 11, -1, 3, -1, 10, -1, 2, -1, 9, -1, 1, -1, 8, -1, 0)
            );
            mmk = _mm_shuffle_epi8(
                ksource, _mm_set_epi8(5, 4, 1, 0, 5, 4, 1, 0, 5, 4, 1, 0, 5, 4, 1, 0)
            );
            sss = mm_dpwssd(sss, pix, mmk);

            pix = _mm_shuffle_epi8(
                source,
                _mm_set_epi8(-1, 15, -1, 7, -1, 14, -1, 6, -1, 13, -1, 5, -1, 12, -1, 4)
            );
            mmk = _mm_shuffle_epi8(
                ksource, _mm_set_epi8(7, 6, 3, 2, 7, 6, 3, 2, 7, 6, 3, 2, 7, 6, 3, 2)
            );
            sss = mm_dpwssd(sss, pix, mmk);

            source = _mm_loadu_si128((__m128i *)&lineIn[x + 4 + xmin]);

            pix = _mm_shuffle_epi8(
                source,
                _mm_set_epi8(-1, 11, -1, 3, -1, 10, -1, 2, -1, 9, -1, 1, -1, 8, -1, 0)
            );
            mmk = _mm_shuffle_epi8(
                ksource,
                _mm_set_epi8(13, 12, 9, 8, 13, 12, 9, 8, 13, 12, 9, 8, 13, 12, 9, 8)
            );
            sss = mm_dpwssd(sss, pix, mmk);

            pix = _mm_shuffle_epi8(
                source,
                _mm_set_epi8(-1, 15, -1, 7, -1, 14, -1, 6, -1, 13, -1, 5, -1, 12, -1, 4)
            );
            mmk = _mm_shuffle_epi8(
                ksource,
                _mm_set_epi8(
                    15, 14, 11, 10, 15, 14, 11, 10, 15, 14, 11, 10, 15, 14, 11, 10
                )
            );
            sss = mm_dpwssd(sss, pix, mmk);
        }

        for (; x < xmax - 3; x += 4) {
            __m128i pix, mmk;
            __m128i source = _mm_loadu_si128((__m128i *)&lineIn[x + xmin]);
            __m128i ksource = _mm_loadl_epi64((__m128i *)&k[x]);

            pix = _mm_shuffle_epi8(
                source,
                _mm_set_epi8(-1, 11, -1, 3, -1, 10, -1, 2, -1, 9, -1, 1, -1, 8, -1, 0)
            );
            mmk = _mm_shuffle_epi8(
                ksource, _mm_set_epi8(5, 4, 1, 0, 5, 4, 1, 0, 5, 4, 1, 0, 5, 4, 1, 0)
            );
            sss = mm_dpwssd(sss, pix, mmk);

            pix = _mm_shuffle_epi8(
                source,
                _mm_set_epi8(-1, 15, -1, 7, -1, 14, -1, 6, -1, 13, -1, 5, -1, 12, -1, 4)
            );
            mmk = _mm_shuffle_epi8(
                ksource, _mm_set_epi8(7, 6, 3, 2, 7, 6, 3, 2, 7, 6, 3, 2, 7, 6, 3, 2)
            );
            sss = mm_dpwssd(sss, pix, mmk);
        }

#endif

        for (; x < xmax - 1; x += 2) {
            __m128i mmk = _mm_set1_epi32(*(INT32 *)&k[x]);
            __m128i source = _mm_loadl_epi64((__m128i *)&lineIn[x + xmin]);
            __m128i pix = _mm_shuffle_epi8(
                source,
                _mm_set_epi8(-1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0)
            );
            sss = mm_dpwssd(sss, pix, mmk);
        }

        for (; x < xmax; x++) {
            __m128i pix = mm_cvtepu8_epi32(&lineIn[x + xmin]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = mm_dpwssd(sss, pix, mmk);
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }
}

#else

/* Portable version of the kernels above, with the same arithmetic:
   16-bit coefficients, 32-bit accumulators, rounding by adding half. */

void
ImagingResampleHorizontalConvolution8u(
    UINT32 *restrict lineOut,
    UINT32 *restrict lineIn,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    int xmin, xmax, xx, x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        int ss0, ss1, ss2, ss3;
        UINT8 *in, *out;
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        in = (UINT8 *)&lineIn[xmin];
        ss0 = ss1 = ss2 = ss3 = 1 << (coefs_precision - 1);
        for (x = 0; x < xmax; x++) {
            ss0 += in[x * 4 + 0] * k[x];
            ss1 += in[x * 4 + 1] * k[x];
            ss2 += in[x * 4 + 2] * k[x];
            ss3 += in[x * 4 + 3] * k[x];
        }
        out = (UINT8 *)&lineOut[xx];
        out[0] = clip8(ss0, coefs_precision);
        out[1] = clip8(ss1, coefs_precision);
        out[2] = clip8(ss2, coefs_precision);
        out[3] = clip8(ss3, coefs_precision);
    }
}

void
ImagingResampleHorizontalConvolution8u4x(
    UINT32 *restrict lineOut0,
    UINT32 *restrict lineOut1,
    UINT32 *restrict lineOut2,
    UINT32 *restrict lineOut3,
    UINT32 *restrict lineIn0,
    UINT32 *restrict lineIn1,
    UINT32 *restrict lineIn2,
    UINT32 *restrict lineIn3,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    ImagingResampleHorizontalConvolution8u(
        lineOut0, lineIn0, xsize, xbounds, kk, kmax, coefs_precision
    );
    ImagingResampleHorizontalConvolution8u(
        lineOut1, lineIn1, xsize, xbounds, kk, kmax, coefs_precision
    );
    ImagingResampleHorizontalConvolution8u(
        lineOut2, lineIn2, xsize, xbounds, kk, kmax, coefs_precision
    );
    ImagingResampleHorizontalConvolution8u(
        lineOut3, lineIn3, xsize, xbounds, kk, kmax, coefs_precision
    );
}

#endif

/* 1-byte pixels. Taps go 8, 4 and 2 at a time as 16-bit products summed
   into 32 bits, then one at a time. Integer sums don't depend on the order,
   so the results match the scalar code. */
#if defined(IMAGING_SIMD_SSE4)
#define RESAMPLE_HORIZONTAL_1BAND_SIMD

/* Four rows at once: they share the coefficient loads and one set of
   horizontal sums per output pixel. The output rows may be the same row,
   with the same input row for each. */
static void
ImagingResampleHorizontalConvolution1u4x(
    UINT8 *lineOut0,
    UINT8 *lineOut1,
    UINT8 *lineOut2,
    UINT8 *lineOut3,
    const UINT8 *lineIn0,
    const UINT8 *lineIn1,
    const UINT8 *lineIn2,
    const UINT8 *lineIn3,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    int xmin, xmax, xx, x;
    const INT16 *k;
    __m128i initial = _mm_set1_epi32(1 << (coefs_precision - 1));

    for (xx = 0; xx < xsize; xx++) {
        const UINT8 *in0, *in1, *in2, *in3;
        __m128i sss0 = _mm_setzero_si128(), sss1 = sss0, sss2 = sss0, sss3 = sss0;
        __m128i mmk, sss;

        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        in0 = lineIn0 + xmin;
        in1 = lineIn1 + xmin;
        in2 = lineIn2 + xmin;
        in3 = lineIn3 + xmin;
        x = 0;

#define MM_LOAD8(p) _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(p)))
#define MM_LOAD4(p) _mm_cvtepu8_epi16(_mm_cvtsi32_si128(mm_load_si32(p)))
#define MM_LOAD2(p) _mm_cvtsi32_si128((p)[0] | ((p)[1] << 16))
#define MM_LOAD1(p) _mm_cvtsi32_si128((p)[0])
#define MM_TAPS(load, n)                        \
    sss0 = mm_dpwssd(sss0, load(in0 + x), mmk); \
    sss1 = mm_dpwssd(sss1, load(in1 + x), mmk); \
    sss2 = mm_dpwssd(sss2, load(in2 + x), mmk); \
    sss3 = mm_dpwssd(sss3, load(in3 + x), mmk); \
    x += n;

        while (x < xmax - 7) {
            mmk = _mm_loadu_si128((const __m128i *)&k[x]);
            MM_TAPS(MM_LOAD8, 8)
        }
        if (x < xmax - 3) {
            mmk = _mm_loadl_epi64((const __m128i *)&k[x]);
            MM_TAPS(MM_LOAD4, 4)
        }
        if (x < xmax - 1) {
            mmk = _mm_cvtsi32_si128(mm_load_si32(&k[x]));
            MM_TAPS(MM_LOAD2, 2)
        }
        if (x < xmax) {
            mmk = _mm_cvtsi32_si128((UINT16)k[x]);
            MM_TAPS(MM_LOAD1, 1)
        }
#undef MM_TAPS
#undef MM_LOAD2
#undef MM_LOAD1
#undef MM_LOAD4
#undef MM_LOAD8

        // [32] sum3 sum2 sum1 sum0
        sss = _mm_hadd_epi32(_mm_hadd_epi32(sss0, sss1), _mm_hadd_epi32(sss2, sss3));
        sss = _mm_srai_epi32(_mm_add_epi32(sss, initial), coefs_precision);
        sss = _mm_packus_epi16(_mm_packs_epi32(sss, sss), sss);
        lineOut0[xx] = (UINT8)_mm_extract_epi8(sss, 0);
        lineOut1[xx] = (UINT8)_mm_extract_epi8(sss, 1);
        lineOut2[xx] = (UINT8)_mm_extract_epi8(sss, 2);
        lineOut3[xx] = (UINT8)_mm_extract_epi8(sss, 3);
    }
}

#endif

#if defined(RESAMPLE_HORIZONTAL_1BAND_SIMD)
static void
ImagingResampleHorizontalConvolution1u(
    UINT8 *restrict lineOut,
    const UINT8 *lineIn,
    int xsize,
    int *xbounds,
    INT16 *kk,
    int kmax,
    int coefs_precision
) {
    ImagingResampleHorizontalConvolution1u4x(
        lineOut,
        lineOut,
        lineOut,
        lineOut,
        lineIn,
        lineIn,
        lineIn,
        lineIn,
        xsize,
        xbounds,
        kk,
        kmax,
        coefs_precision
    );
}
#endif

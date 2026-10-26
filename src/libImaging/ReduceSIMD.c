/* Kernels for images with 4-byte pixels, included from Reduce.c.
   They use the same arithmetic as the scalar code in Reduce.c. */

#if defined(IMAGING_SIMD_SSE4)
#define REDUCE_SIMD

static void
ImagingReduceNxN_4u8(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
    /* The most general implementation for any xscale and yscale
     */
    int x, y, xx, yy;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_multiplier = _mm_set1_epi32(multiplier);
    __m128i mm_amend = _mm_set1_epi32(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy_from = box[1] + y * yscale;
        for (x = 0; x < box[2] / xscale; x++) {
            int xx_from = box[0] + x * xscale;
            __m128i mm_ss0 = mm_amend;
            for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                UINT8 *line0 = (UINT8 *)imIn->image[yy];
                UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
                for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                    __m128i pix0 = _mm_unpacklo_epi8(
                        _mm_loadl_epi64((__m128i *)&line0[xx * 4]), _mm_setzero_si128()
                    );
                    __m128i pix1 = _mm_unpacklo_epi8(
                        _mm_loadl_epi64((__m128i *)&line1[xx * 4]), _mm_setzero_si128()
                    );
                    pix0 = _mm_add_epi16(pix0, pix1);
                    pix0 = _mm_add_epi32(
                        _mm_unpackhi_epi16(pix0, _mm_setzero_si128()),
                        _mm_unpacklo_epi16(pix0, _mm_setzero_si128())
                    );
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                }
                if (xscale & 0x01) {
                    __m128i pix0 = mm_cvtepu8_epi32(&line0[xx * 4]);
                    __m128i pix1 = mm_cvtepu8_epi32(&line1[xx * 4]);
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix1);
                }
            }
            if (yscale & 0x01) {
                UINT8 *line = (UINT8 *)imIn->image[yy];
                for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                    __m128i pix0 = _mm_unpacklo_epi8(
                        _mm_loadl_epi64((__m128i *)&line[xx * 4]), _mm_setzero_si128()
                    );
                    pix0 = _mm_add_epi32(
                        _mm_unpackhi_epi16(pix0, _mm_setzero_si128()),
                        _mm_unpacklo_epi16(pix0, _mm_setzero_si128())
                    );
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                }
                if (xscale & 0x01) {
                    __m128i pix0 = mm_cvtepu8_epi32(&line[xx * 4]);
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                }
            }

            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduce1xN_4u8(Imaging imOut, Imaging imIn, int box[4], int yscale) {
    /* Optimized implementation for xscale = 1.
     */
    int x, y, yy;
    int xscale = 1;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_multiplier = _mm_set1_epi32(multiplier);
    __m128i mm_amend = _mm_set1_epi32(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy_from = box[1] + y * yscale;
        for (x = 0; x < box[2] / xscale - 1; x += 2) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0 = mm_amend;
            __m128i mm_ss1 = mm_amend;
            for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
                UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *)&line0[xx * 4]), _mm_setzero_si128()
                );
                __m128i pix1 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *)&line1[xx * 4]), _mm_setzero_si128()
                );
                pix0 = _mm_add_epi16(pix0, pix1);
                mm_ss0 = _mm_add_epi32(
                    mm_ss0, _mm_unpacklo_epi16(pix0, _mm_setzero_si128())
                );
                mm_ss1 = _mm_add_epi32(
                    mm_ss1, _mm_unpackhi_epi16(pix0, _mm_setzero_si128())
                );
            }
            if (yscale & 0x01) {
                UINT8 *line = (UINT8 *)imIn->image[yy];
                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *)&line[xx * 4]), _mm_setzero_si128()
                );
                mm_ss0 = _mm_add_epi32(
                    mm_ss0, _mm_unpacklo_epi16(pix0, _mm_setzero_si128())
                );
                mm_ss1 = _mm_add_epi32(
                    mm_ss1, _mm_unpackhi_epi16(pix0, _mm_setzero_si128())
                );
            }

            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss1 = _mm_mullo_epi32(mm_ss1, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss1 = _mm_srli_epi32(mm_ss1, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss1);
            _mm_storel_epi64(
                (__m128i *)(imOut->image32[y] + x), _mm_packus_epi16(mm_ss0, mm_ss0)
            );
        }
        for (; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0 = mm_amend;
            for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
                UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
                __m128i pix0 = mm_cvtepu8_epi32(&line0[xx * 4]);
                __m128i pix1 = mm_cvtepu8_epi32(&line1[xx * 4]);
                mm_ss0 = _mm_add_epi32(mm_ss0, _mm_add_epi32(pix0, pix1));
            }
            if (yscale & 0x01) {
                UINT8 *line = (UINT8 *)imIn->image[yy];
                __m128i pix0 = mm_cvtepu8_epi32(&line[xx * 4]);
                mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
            }

            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduceNx1_4u8(Imaging imOut, Imaging imIn, int box[4], int xscale) {
    /* Optimized implementation for yscale = 1.
     */
    int x, y, xx;
    int yscale = 1;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_multiplier = _mm_set1_epi32(multiplier);
    __m128i mm_amend = _mm_set1_epi32(amend);

    for (y = 0; y < box[3] / yscale - 1; y += 2) {
        int yy = box[1] + y * yscale;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
        for (x = 0; x < box[2] / xscale; x++) {
            int xx_from = box[0] + x * xscale;
            __m128i mm_ss0 = mm_amend;
            __m128i mm_ss1 = mm_amend;
            for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *)&line0[xx * 4]), _mm_setzero_si128()
                );
                __m128i pix1 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *)&line1[xx * 4]), _mm_setzero_si128()
                );
                pix0 = _mm_add_epi32(
                    _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                    _mm_unpackhi_epi16(pix0, _mm_setzero_si128())
                );
                pix1 = _mm_add_epi32(
                    _mm_unpacklo_epi16(pix1, _mm_setzero_si128()),
                    _mm_unpackhi_epi16(pix1, _mm_setzero_si128())
                );
                mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                mm_ss1 = _mm_add_epi32(mm_ss1, pix1);
            }
            if (xscale & 0x01) {
                __m128i pix0 = mm_cvtepu8_epi32(&line0[xx * 4]);
                __m128i pix1 = mm_cvtepu8_epi32(&line1[xx * 4]);
                mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                mm_ss1 = _mm_add_epi32(mm_ss1, pix1);
            }

            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss1 = _mm_mullo_epi32(mm_ss1, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss1 = _mm_srli_epi32(mm_ss1, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            mm_ss1 = _mm_packs_epi32(mm_ss1, mm_ss1);
            ((UINT32 *)imOut->image[y + 0])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            ((UINT32 *)imOut->image[y + 1])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss1, mm_ss1));
        }
    }
    for (; y < box[3] / yscale; y++) {
        int yy = box[1] + y * yscale;
        UINT8 *line = (UINT8 *)imIn->image[yy];
        for (x = 0; x < box[2] / xscale; x++) {
            int xx_from = box[0] + x * xscale;
            __m128i mm_ss0 = mm_amend;
            for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *)&line[xx * 4]), _mm_setzero_si128()
                );
                pix0 = _mm_add_epi32(
                    _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                    _mm_unpackhi_epi16(pix0, _mm_setzero_si128())
                );
                mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
            }
            if (xscale & 0x01) {
                __m128i pix0 = mm_cvtepu8_epi32(&line[xx * 4]);
                mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
            }

            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduce1x2_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 1 and yscale = 2.
     */
    int xscale = 1, yscale = 2;
    int x, y;
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_amend = _mm_set1_epi32(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy = box[1] + y * yscale;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
        // (a + b + 1) >> 1 for every byte, four pixels at a time.
        for (x = 0; x < box[2] / xscale - 3; x += 4) {
            int xx = box[0] + x * xscale;
            _mm_storeu_si128(
                (__m128i *)&((UINT32 *)imOut->image[y])[x],
                _mm_avg_epu8(
                    _mm_loadu_si128((__m128i *)&line0[xx * 4]),
                    _mm_loadu_si128((__m128i *)&line1[xx * 4])
                )
            );
        }
        for (; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0;

            __m128i pix0 = mm_cvtepu8_epi32(&line0[xx * 4]);
            __m128i pix1 = mm_cvtepu8_epi32(&line1[xx * 4]);
            mm_ss0 = _mm_add_epi32(pix0, pix1);
            mm_ss0 = _mm_add_epi32(mm_ss0, mm_amend);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 1);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduce2x1_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 2 and yscale = 1.
     */
    int xscale = 2, yscale = 1;
    int x, y;
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_amend = _mm_set1_epi16(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy = box[1] + y * yscale;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        // Channels of neighbouring pixels side by side, then summed in pairs.
        for (x = 0; x < box[2] / xscale - 3; x += 4) {
            int xx = box[0] + x * xscale;
            __m128i shuffle =
                _mm_set_epi8(15, 11, 14, 10, 13, 9, 12, 8, 7, 3, 6, 2, 5, 1, 4, 0);
            __m128i ones = _mm_set1_epi8(1);
            __m128i ss0 = _mm_maddubs_epi16(
                _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&line0[xx * 4]), shuffle),
                ones
            );
            __m128i ss1 = _mm_maddubs_epi16(
                _mm_shuffle_epi8(
                    _mm_loadu_si128((__m128i *)&line0[xx * 4 + 16]), shuffle
                ),
                ones
            );
            ss0 = _mm_srli_epi16(_mm_add_epi16(ss0, mm_amend), 1);
            ss1 = _mm_srli_epi16(_mm_add_epi16(ss1, mm_amend), 1);
            _mm_storeu_si128(
                (__m128i *)&((UINT32 *)imOut->image[y])[x], _mm_packus_epi16(ss0, ss1)
            );
        }
        for (; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0;

            __m128i pix0 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((__m128i *)&line0[xx * 4]), _mm_setzero_si128()
            );
            pix0 = _mm_add_epi16(_mm_srli_si128(pix0, 8), pix0);
            mm_ss0 = _mm_add_epi16(pix0, mm_amend);
            mm_ss0 = _mm_srli_epi16(mm_ss0, 1);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduce2x2_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 2 and yscale = 2.
     */
    int xscale = 2, yscale = 2;
    int x, y;
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_amend = _mm_set1_epi16(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy = box[1] + y * yscale;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
        // Channels of neighbouring pixels side by side, then summed in pairs.
        for (x = 0; x < box[2] / xscale - 3; x += 4) {
            int xx = box[0] + x * xscale;
            __m128i shuffle =
                _mm_set_epi8(15, 11, 14, 10, 13, 9, 12, 8, 7, 3, 6, 2, 5, 1, 4, 0);
            __m128i ones = _mm_set1_epi8(1);
#define PAIRS(line, off)                                                            \
    _mm_maddubs_epi16(                                                              \
        _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&line[xx * 4 + off]), shuffle), \
        ones                                                                        \
    )
            __m128i ss0 = _mm_add_epi16(PAIRS(line0, 0), PAIRS(line1, 0));
            __m128i ss1 = _mm_add_epi16(PAIRS(line0, 16), PAIRS(line1, 16));
#undef PAIRS
            ss0 = _mm_srli_epi16(_mm_add_epi16(ss0, mm_amend), 2);
            ss1 = _mm_srli_epi16(_mm_add_epi16(ss1, mm_amend), 2);
            _mm_storeu_si128(
                (__m128i *)&((UINT32 *)imOut->image[y])[x], _mm_packus_epi16(ss0, ss1)
            );
        }
        for (; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0;

            __m128i pix0 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((__m128i *)&line0[xx * 4]), _mm_setzero_si128()
            );
            __m128i pix1 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((__m128i *)&line1[xx * 4]), _mm_setzero_si128()
            );
            pix0 = _mm_add_epi16(pix0, pix1);
            pix0 = _mm_add_epi16(_mm_srli_si128(pix0, 8), pix0);
            mm_ss0 = _mm_add_epi16(pix0, mm_amend);
            mm_ss0 = _mm_srli_epi16(mm_ss0, 2);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduce1x3_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 1 and yscale = 3.
     */
    int xscale = 1, yscale = 3;
    int x, y;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_multiplier = _mm_set1_epi32(multiplier);
    __m128i mm_amend = _mm_set1_epi32(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy = box[1] + y * yscale;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
        UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
        for (x = 0; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0 = _mm_add_epi32(
                _mm_add_epi32(
                    mm_cvtepu8_epi32(&line0[xx * 4]), mm_cvtepu8_epi32(&line1[xx * 4])
                ),
                _mm_add_epi32(mm_cvtepu8_epi32(&line2[xx * 4]), mm_amend)
            );

            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduce3x1_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 3 and yscale = 1.
     */
    int xscale = 3, yscale = 1;
    int x, y;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_multiplier = _mm_set1_epi32(multiplier);
    __m128i mm_amend = _mm_set1_epi32(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy = box[1] + y * yscale;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        for (x = 0; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0;

            __m128i pix0 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((__m128i *)&line0[xx * 4]), _mm_setzero_si128()
            );
            pix0 = _mm_add_epi32(
                _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                _mm_unpackhi_epi16(pix0, _mm_setzero_si128())
            );

            mm_ss0 = _mm_add_epi32(mm_cvtepu8_epi32(&line0[xx * 4 + 8]), pix0);

            mm_ss0 = _mm_add_epi32(mm_ss0, mm_amend);
            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

static void
ImagingReduce3x3_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 3 and yscale = 3.
     */
    int xscale = 3, yscale = 3;
    int x, y;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    __m128i mm_multiplier = _mm_set1_epi32(multiplier);
    __m128i mm_amend = _mm_set1_epi32(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy = box[1] + y * yscale;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
        UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
        for (x = 0; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            __m128i mm_ss0;

            __m128i pix0 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((__m128i *)&line0[xx * 4]), _mm_setzero_si128()
            );
            __m128i pix1 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((__m128i *)&line1[xx * 4]), _mm_setzero_si128()
            );
            __m128i pix2 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((__m128i *)&line2[xx * 4]), _mm_setzero_si128()
            );

            pix0 = _mm_add_epi16(_mm_add_epi16(pix0, pix1), pix2);
            pix0 = _mm_add_epi32(
                _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                _mm_unpackhi_epi16(pix0, _mm_setzero_si128())
            );

            mm_ss0 = _mm_add_epi32(
                _mm_add_epi32(
                    mm_cvtepu8_epi32(&line0[xx * 4 + 8]),
                    mm_cvtepu8_epi32(&line1[xx * 4 + 8])
                ),
                _mm_add_epi32(mm_cvtepu8_epi32(&line2[xx * 4 + 8]), pix0)
            );

            mm_ss0 = _mm_add_epi32(mm_ss0, mm_amend);
            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] =
                _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
        }
    }
}

#elif defined(IMAGING_SIMD_NEON)
#define REDUCE_SIMD

/* 16 input pixels per row are de-interleaved, pairs of neighbours summed
   and the sums rounded with the scalar code's (sum + N/2) / N, so the
   scalar loops can finish the row. */

static void
ImagingReduce2x2_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    int x, y, c;
    int xsize = box[2] / 2;

    for (y = 0; y < box[3] / 2; y++) {
        int yy = box[1] + y * 2;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
        UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
        UINT8 *out = (UINT8 *)imOut->image[y];
        for (x = 0; x < xsize - 7; x += 8) {
            int xx = box[0] + x * 2;
            uint8x16x4_t p0 = vld4q_u8(&line0[xx * 4]);
            uint8x16x4_t p1 = vld4q_u8(&line1[xx * 4]);
            uint8x8x4_t o;
            for (c = 0; c < 4; c++) {
                o.val[c] =
                    vrshrn_n_u16(vpadalq_u8(vpaddlq_u8(p0.val[c]), p1.val[c]), 2);
            }
            vst4_u8(&out[x * 4], o);
        }
        for (; x < xsize; x++) {
            int xx = box[0] + x * 2;
            for (c = 0; c < 4; c++) {
                UINT32 ss = line0[xx * 4 + c] + line0[xx * 4 + 4 + c] +
                            line1[xx * 4 + c] + line1[xx * 4 + 4 + c];
                out[x * 4 + c] = (ss + 2) >> 2;
            }
        }
    }
}

static void
ImagingReduce2x1_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    int x, y, c;
    int xsize = box[2] / 2;

    for (y = 0; y < box[3]; y++) {
        UINT8 *line = (UINT8 *)imIn->image[box[1] + y];
        UINT8 *out = (UINT8 *)imOut->image[y];
        for (x = 0; x < xsize - 7; x += 8) {
            int xx = box[0] + x * 2;
            uint8x16x4_t p = vld4q_u8(&line[xx * 4]);
            uint8x8x4_t o;
            for (c = 0; c < 4; c++) {
                o.val[c] = vrshrn_n_u16(vpaddlq_u8(p.val[c]), 1);
            }
            vst4_u8(&out[x * 4], o);
        }
        for (; x < xsize; x++) {
            int xx = box[0] + x * 2;
            for (c = 0; c < 4; c++) {
                out[x * 4 + c] = (line[xx * 4 + c] + line[xx * 4 + 4 + c] + 1) >> 1;
            }
        }
    }
}

/* Adds the sums of xscale neighbouring pixels, for 4 output pixels of
   one row, to lo (output pixels 0 and 1) and hi (2 and 3). vldN_u32 puts
   pixel N*i + k in lane i of vector k, so adding the vectors byte by byte
   sums each group of N pixels channel by channel. */
static inline void
neon_reduce_row_4u8(const UINT8 *p, int xscale, uint16x8_t *lo, uint16x8_t *hi) {
    switch (xscale) {
        case 1: {
            uint8x16_t v = vld1q_u8(p);
            *lo = vaddw_u8(*lo, vget_low_u8(v));
            *hi = vaddw_high_u8(*hi, v);
            break;
        }
        case 2: {
            uint32x4x2_t v = vld2q_u32((const uint32_t *)(const void *)p);
            uint8x16_t v0 = vreinterpretq_u8_u32(v.val[0]);
            uint8x16_t v1 = vreinterpretq_u8_u32(v.val[1]);
            *lo = vaddq_u16(*lo, vaddl_u8(vget_low_u8(v0), vget_low_u8(v1)));
            *hi = vaddq_u16(*hi, vaddl_high_u8(v0, v1));
            break;
        }
        case 3: {
            uint32x4x3_t v = vld3q_u32((const uint32_t *)(const void *)p);
            uint8x16_t v0 = vreinterpretq_u8_u32(v.val[0]);
            uint8x16_t v1 = vreinterpretq_u8_u32(v.val[1]);
            uint8x16_t v2 = vreinterpretq_u8_u32(v.val[2]);
            *lo = vaddw_u8(
                vaddq_u16(*lo, vaddl_u8(vget_low_u8(v0), vget_low_u8(v1))),
                vget_low_u8(v2)
            );
            *hi = vaddw_high_u8(vaddq_u16(*hi, vaddl_high_u8(v0, v1)), v2);
            break;
        }
        default: {
            uint32x4x4_t v = vld4q_u32((const uint32_t *)(const void *)p);
            uint8x16_t v0 = vreinterpretq_u8_u32(v.val[0]);
            uint8x16_t v1 = vreinterpretq_u8_u32(v.val[1]);
            uint8x16_t v2 = vreinterpretq_u8_u32(v.val[2]);
            uint8x16_t v3 = vreinterpretq_u8_u32(v.val[3]);
            *lo = vaddq_u16(
                vaddq_u16(*lo, vaddl_u8(vget_low_u8(v0), vget_low_u8(v1))),
                vaddl_u8(vget_low_u8(v2), vget_low_u8(v3))
            );
            *hi =
                vaddq_u16(vaddq_u16(*hi, vaddl_high_u8(v0, v1)), vaddl_high_u8(v2, v3));
            break;
        }
    }
}

/* ((sum + amend) * multiplier) >> 24 for the 4 channels of one pixel. */
static inline uint16x4_t
neon_reduce_round(uint16x4_t sum, uint32x4_t amend, uint32x4_t multiplier) {
    uint32x4_t v = vmulq_u32(vaddq_u32(vmovl_u16(sum), amend), multiplier);
    return vmovn_u32(vshrq_n_u32(v, 24));
}

/* Any xscale from 1 to 4 and yscale up to 16, so that the sums fit in 16
   bits: 4 output pixels per step, rounded as in the scalar code. */
static void
ImagingReduceNxN_4u8_neon(
    Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale
) {
    int x, y, yy, c;
    int xsize = box[2] / xscale;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    uint32x4_t mm_multiplier = vdupq_n_u32(multiplier);
    uint32x4_t mm_amend = vdupq_n_u32(amend);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy_from = box[1] + y * yscale;
        UINT8 *out = (UINT8 *)imOut->image[y];
        for (x = 0; x < xsize - 3; x += 4) {
            int xx = box[0] + x * xscale;
            uint16x8_t lo = vdupq_n_u16(0), hi = lo;
            for (yy = yy_from; yy < yy_from + yscale; yy++) {
                neon_reduce_row_4u8(
                    (UINT8 *)imIn->image[yy] + xx * 4, xscale, &lo, &hi
                );
            }
            uint16x8_t o01 = vcombine_u16(
                neon_reduce_round(vget_low_u16(lo), mm_amend, mm_multiplier),
                neon_reduce_round(vget_high_u16(lo), mm_amend, mm_multiplier)
            );
            uint16x8_t o23 = vcombine_u16(
                neon_reduce_round(vget_low_u16(hi), mm_amend, mm_multiplier),
                neon_reduce_round(vget_high_u16(hi), mm_amend, mm_multiplier)
            );
            vst1q_u8(&out[x * 4], vcombine_u8(vmovn_u16(o01), vmovn_u16(o23)));
        }
        for (; x < xsize; x++) {
            int xx = box[0] + x * xscale;
            for (c = 0; c < 4; c++) {
                UINT32 ss = amend;
                int i;
                for (yy = yy_from; yy < yy_from + yscale; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (i = 0; i < xscale; i++) {
                        ss += line[(xx + i) * 4 + c];
                    }
                }
                out[x * 4 + c] = (ss * multiplier) >> 24;
            }
        }
    }
}

/* Any scale with xscale up to 512: one output pixel at a time. Each row
   of its block is summed 4, 2 and 1 pixels at a time in 16-bit lanes; the
   pixels alternate between the two halves, so a lane holds at most 256
   bytes. The rows are summed in 32 bits. */
static void
ImagingReduceNxN_4u8_neon_wide(
    Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale
) {
    int x, y, yy, k;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;
    uint32x4_t mm_multiplier = vdupq_n_u32(multiplier);

    for (y = 0; y < box[3] / yscale; y++) {
        int yy_from = box[1] + y * yscale;
        UINT32 *out = (UINT32 *)imOut->image[y];
        for (x = 0; x < box[2] / xscale; x++) {
            int xx = box[0] + x * xscale;
            uint32x4_t ss = vdupq_n_u32(amend);
            for (yy = yy_from; yy < yy_from + yscale; yy++) {
                const UINT8 *p = (UINT8 *)imIn->image[yy] + xx * 4;
                uint16x8_t row = vdupq_n_u16(0);
                for (k = 0; k < xscale - 3; k += 4) {
                    uint8x16_t v = vld1q_u8(p + k * 4);
                    row = vaddw_high_u8(vaddw_u8(row, vget_low_u8(v)), v);
                }
                if (k < xscale - 1) {
                    row = vaddw_u8(row, vld1_u8(p + k * 4));
                    k += 2;
                }
                if (k < xscale) {
                    UINT32 v;
                    memcpy(&v, p + k * 4, sizeof(v));
                    row = vaddw_u8(row, vcreate_u8(v));
                }
                ss = vaddq_u32(ss, vaddl_u16(vget_low_u16(row), vget_high_u16(row)));
            }
            ss = vshrq_n_u32(vmulq_u32(ss, mm_multiplier), 24);
            uint8x8_t px = vmovn_u16(vcombine_u16(vmovn_u32(ss), vdup_n_u16(0)));
            out[x] = vget_lane_u32(vreinterpret_u32_u8(px), 0);
        }
    }
}

static void
ImagingReduce1x2_4u8(Imaging imOut, Imaging imIn, int box[4]) {
    int x, y;
    int size = box[2] * 4;

    for (y = 0; y < box[3] / 2; y++) {
        int yy = box[1] + y * 2;
        UINT8 *line0 = (UINT8 *)imIn->image[yy + 0] + box[0] * 4;
        UINT8 *line1 = (UINT8 *)imIn->image[yy + 1] + box[0] * 4;
        UINT8 *out = (UINT8 *)imOut->image[y];
        // Every byte is averaged with the one below: (a + b + 1) >> 1.
        for (x = 0; x < size - 15; x += 16) {
            vst1q_u8(&out[x], vrhaddq_u8(vld1q_u8(&line0[x]), vld1q_u8(&line1[x])));
        }
        for (; x < size; x++) {
            out[x] = (line0[x] + line1[x] + 1) >> 1;
        }
    }
}

#endif

#if defined(REDUCE_SIMD)
/* Returns 0 if there is no kernel for these scales; the caller then uses
   the scalar code. */
static int
ImagingReduce_4u8(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
#if defined(IMAGING_SIMD_SSE4)
    if (xscale == 1) {
        if (yscale == 2) {
            ImagingReduce1x2_4u8(imOut, imIn, box);
        } else if (yscale == 3) {
            ImagingReduce1x3_4u8(imOut, imIn, box);
        } else {
            ImagingReduce1xN_4u8(imOut, imIn, box, yscale);
        }
    } else if (yscale == 1) {
        if (xscale == 2) {
            ImagingReduce2x1_4u8(imOut, imIn, box);
        } else if (xscale == 3) {
            ImagingReduce3x1_4u8(imOut, imIn, box);
        } else {
            ImagingReduceNx1_4u8(imOut, imIn, box, xscale);
        }
    } else if (xscale == 2 && yscale == 2) {
        ImagingReduce2x2_4u8(imOut, imIn, box);
    } else if (xscale == 3 && yscale == 3) {
        ImagingReduce3x3_4u8(imOut, imIn, box);
    } else {
        ImagingReduceNxN_4u8(imOut, imIn, box, xscale, yscale);
    }
    return 1;
#else
    if (xscale == 1 && yscale == 2) {
        ImagingReduce1x2_4u8(imOut, imIn, box);
    } else if (xscale == 2 && yscale == 1) {
        ImagingReduce2x1_4u8(imOut, imIn, box);
    } else if (xscale == 2 && yscale == 2) {
        ImagingReduce2x2_4u8(imOut, imIn, box);
    } else if (xscale <= 4 && yscale <= 16) {
        ImagingReduceNxN_4u8_neon(imOut, imIn, box, xscale, yscale);
    } else if (xscale <= 512) {
        ImagingReduceNxN_4u8_neon_wide(imOut, imIn, box, xscale, yscale);
    } else {
        return 0;
    }
    return 1;
#endif
}
#endif

/* 2x2 and 4x4 for images with 1-byte pixels. Bytes are summed in pairs,
   rows added, and the sums rounded with the scalar code's
   (sum + N/2) >> log2(N). The scalar loop finishes each row. Returns 0 for
   other scales; the caller then uses the scalar code. */
#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
#define REDUCE_SIMD_1U8

static void
ImagingReduce2x2_1u8(Imaging imOut, Imaging imIn, int box[4]) {
    int x, y;
    int xsize = box[2] / 2;

    for (y = 0; y < box[3] / 2; y++) {
        int yy = box[1] + y * 2;
        const UINT8 *line0 = (UINT8 *)imIn->image8[yy + 0] + box[0];
        const UINT8 *line1 = (UINT8 *)imIn->image8[yy + 1] + box[0];
        UINT8 *out = imOut->image8[y];
        x = 0;
#if defined(IMAGING_SIMD_SSE4)
        {
            __m128i ones = _mm_set1_epi8(1);
            __m128i two = _mm_set1_epi16(2);
            for (; x < xsize - 15; x += 16) {
                const __m128i *p0 = (const __m128i *)&line0[x * 2];
                const __m128i *p1 = (const __m128i *)&line1[x * 2];
                // [16] sums of 2x2 blocks
                __m128i lo = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_loadu_si128(p0), ones),
                    _mm_maddubs_epi16(_mm_loadu_si128(p1), ones)
                );
                __m128i hi = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_loadu_si128(p0 + 1), ones),
                    _mm_maddubs_epi16(_mm_loadu_si128(p1 + 1), ones)
                );
                lo = _mm_srli_epi16(_mm_add_epi16(lo, two), 2);
                hi = _mm_srli_epi16(_mm_add_epi16(hi, two), 2);
                _mm_storeu_si128((__m128i *)&out[x], _mm_packus_epi16(lo, hi));
            }
        }
#else
        for (; x < xsize - 15; x += 16) {
            uint16x8_t lo = vpadalq_u8(
                vpaddlq_u8(vld1q_u8(&line0[x * 2])), vld1q_u8(&line1[x * 2])
            );
            uint16x8_t hi = vpadalq_u8(
                vpaddlq_u8(vld1q_u8(&line0[x * 2 + 16])), vld1q_u8(&line1[x * 2 + 16])
            );
            vst1q_u8(&out[x], vcombine_u8(vrshrn_n_u16(lo, 2), vrshrn_n_u16(hi, 2)));
        }
#endif
        for (; x < xsize; x++) {
            UINT32 ss =
                line0[x * 2] + line0[x * 2 + 1] + line1[x * 2] + line1[x * 2 + 1];
            out[x] = (ss + 2) >> 2;
        }
    }
}

static void
ImagingReduce4x4_1u8(Imaging imOut, Imaging imIn, int box[4]) {
    int x, y, r;
    int xsize = box[2] / 4;

    for (y = 0; y < box[3] / 4; y++) {
        int yy = box[1] + y * 4;
        const UINT8 *lines[4];
        UINT8 *out = imOut->image8[y];
        for (r = 0; r < 4; r++) {
            lines[r] = (UINT8 *)imIn->image8[yy + r] + box[0];
        }
        x = 0;
#if defined(IMAGING_SIMD_SSE4)
        {
            __m128i ones = _mm_set1_epi8(1);
            __m128i eight = _mm_set1_epi16(8);
            for (; x < xsize - 7; x += 8) {
                // [16] sums of 2x4 blocks, then of 4x4 blocks
                __m128i lo = _mm_setzero_si128(), hi = lo;
                for (r = 0; r < 4; r++) {
                    const __m128i *p = (const __m128i *)&lines[r][x * 4];
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(_mm_loadu_si128(p), ones));
                    hi = _mm_add_epi16(
                        hi, _mm_maddubs_epi16(_mm_loadu_si128(p + 1), ones)
                    );
                }
                lo = _mm_srli_epi16(_mm_add_epi16(_mm_hadd_epi16(lo, hi), eight), 4);
                _mm_storel_epi64((__m128i *)&out[x], _mm_packus_epi16(lo, lo));
            }
        }
#else
        for (; x < xsize - 7; x += 8) {
            uint16x8_t lo = vpaddlq_u8(vld1q_u8(&lines[0][x * 4]));
            uint16x8_t hi = vpaddlq_u8(vld1q_u8(&lines[0][x * 4 + 16]));
            for (r = 1; r < 4; r++) {
                lo = vpadalq_u8(lo, vld1q_u8(&lines[r][x * 4]));
                hi = vpadalq_u8(hi, vld1q_u8(&lines[r][x * 4 + 16]));
            }
            vst1_u8(&out[x], vrshrn_n_u16(vpaddq_u16(lo, hi), 4));
        }
#endif
        for (; x < xsize; x++) {
            UINT32 ss = 0;
            for (r = 0; r < 4; r++) {
                ss += lines[r][x * 4] + lines[r][x * 4 + 1] + lines[r][x * 4 + 2] +
                      lines[r][x * 4 + 3];
            }
            out[x] = (ss + 8) >> 4;
        }
    }
}

static int
ImagingReduce_1u8(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
    if (xscale == 2 && yscale == 2) {
        ImagingReduce2x2_1u8(imOut, imIn, box);
    } else if (xscale == 4 && yscale == 4) {
        ImagingReduce4x4_1u8(imOut, imIn, box);
    } else {
        return 0;
    }
    return 1;
}
#endif

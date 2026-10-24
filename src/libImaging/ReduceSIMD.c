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
                        _mm_loadl_epi64((__m128i *)&line0[xx * 4]),
                        _mm_setzero_si128()
                    );
                    __m128i pix1 = _mm_unpacklo_epi8(
                        _mm_loadl_epi64((__m128i *)&line1[xx * 4]),
                        _mm_setzero_si128()
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
                        _mm_loadl_epi64((__m128i *)&line[xx * 4]),
                        _mm_setzero_si128()
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
        for (x = 0; x < box[2] / xscale; x++) {
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
        for (x = 0; x < box[2] / xscale; x++) {
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
        for (x = 0; x < box[2] / xscale; x++) {
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
                    mm_cvtepu8_epi32(&line0[xx * 4]),
                    mm_cvtepu8_epi32(&line1[xx * 4])
                ),
                _mm_add_epi32(mm_cvtepu8_epi32(&line2[xx * 4]), mm_amend)
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

#endif

#if defined(REDUCE_SIMD)
static void
ImagingReduce_4u8(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
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
}
#endif

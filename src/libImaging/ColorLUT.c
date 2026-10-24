#include "Imaging.h"
#include <math.h>

#if defined(IMAGING_SIMD_SSE4)

/* 8 bits for result. Table can overflow [0, 1.0] range,
   so we need extra bits for overflow and negative values.
   NOTE: This value should be the same as in _imaging/_prepare_lut_table() */
#define PRECISION_BITS (16 - 8 - 2)
#define PRECISION_ROUNDING (1 << (PRECISION_BITS - 1))

/* 16 - UINT16 capacity
   6 - max index in the table
       (max size is 65, but index 64 is not reachable)
   7 - that much bits we can save if divide the value by 255. */
#define SCALE_BITS (16 - 6 + 7)
#define SCALE_MASK ((1 << SCALE_BITS) - 1)

/* Number of bits in the index tail which is used as a shift
   between two table values. Should be >= 9, <= 14. */
#define SHIFT_BITS (16 - 7)

__m128i static inline mm_lut_pixel(
    INT16 *table,
    int size2D_shift,
    int size3D_shift,
    int idx,
    __m128i shift,
    __m128i shuffle
) {
    __m128i leftleft, leftright, rightleft, rightright;
    __m128i source, left, right;
    __m128i shift1D = _mm_shuffle_epi32(shift, 0x00);
    __m128i shift2D = _mm_shuffle_epi32(shift, 0x55);
    __m128i shift3D = _mm_shuffle_epi32(shift, 0xaa);

    source = _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&table[idx + 0]), shuffle);
    leftleft = _mm_srai_epi32(_mm_madd_epi16(source, shift1D), SHIFT_BITS);

    source = _mm_shuffle_epi8(
        _mm_loadu_si128((__m128i *)&table[idx + size2D_shift]), shuffle
    );
    leftright = _mm_slli_epi32(_mm_madd_epi16(source, shift1D), 16 - SHIFT_BITS);

    source = _mm_shuffle_epi8(
        _mm_loadu_si128((__m128i *)&table[idx + size3D_shift]), shuffle
    );
    rightleft = _mm_srai_epi32(_mm_madd_epi16(source, shift1D), SHIFT_BITS);

    source = _mm_shuffle_epi8(
        _mm_loadu_si128((__m128i *)&table[idx + size3D_shift + size2D_shift]), shuffle
    );
    rightright = _mm_slli_epi32(_mm_madd_epi16(source, shift1D), 16 - SHIFT_BITS);

    left = _mm_srai_epi32(
        _mm_madd_epi16(_mm_blend_epi16(leftleft, leftright, 0xaa), shift2D), SHIFT_BITS
    );

    right = _mm_slli_epi32(
        _mm_madd_epi16(_mm_blend_epi16(rightleft, rightright, 0xaa), shift2D),
        16 - SHIFT_BITS
    );

    return _mm_srai_epi32(
        _mm_madd_epi16(_mm_blend_epi16(left, right, 0xaa), shift3D), SHIFT_BITS
    );
}

#if defined(IMAGING_SIMD_AVX2)
__m256i static inline mm256_lut_pixel(
    INT16 *table,
    int size2D_shift,
    int size3D_shift,
    int idx1,
    int idx2,
    __m256i shift,
    __m256i shuffle
) {
    __m256i leftleft, leftright, rightleft, rightright;
    __m256i source, left, right;
    __m256i shift1D = _mm256_shuffle_epi32(shift, 0x00);
    __m256i shift2D = _mm256_shuffle_epi32(shift, 0x55);
    __m256i shift3D = _mm256_shuffle_epi32(shift, 0xaa);

    source = _mm256_shuffle_epi8(
        _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128((__m128i *)&table[idx1 + 0])),
            _mm_loadu_si128((__m128i *)&table[idx2 + 0]),
            1
        ),
        shuffle
    );
    leftleft = _mm256_srai_epi32(_mm256_madd_epi16(source, shift1D), SHIFT_BITS);

    source = _mm256_shuffle_epi8(
        _mm256_inserti128_si256(
            _mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *)&table[idx1 + size2D_shift])
            ),
            _mm_loadu_si128((__m128i *)&table[idx2 + size2D_shift]),
            1
        ),
        shuffle
    );
    leftright = _mm256_slli_epi32(_mm256_madd_epi16(source, shift1D), 16 - SHIFT_BITS);

    source = _mm256_shuffle_epi8(
        _mm256_inserti128_si256(
            _mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *)&table[idx1 + size3D_shift])
            ),
            _mm_loadu_si128((__m128i *)&table[idx2 + size3D_shift]),
            1
        ),
        shuffle
    );
    rightleft = _mm256_srai_epi32(_mm256_madd_epi16(source, shift1D), SHIFT_BITS);

    source = _mm256_shuffle_epi8(
        _mm256_inserti128_si256(
            _mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *)&table[idx1 + size3D_shift + size2D_shift])
            ),
            _mm_loadu_si128((__m128i *)&table[idx2 + size3D_shift + size2D_shift]),
            1
        ),
        shuffle
    );
    rightright = _mm256_slli_epi32(_mm256_madd_epi16(source, shift1D), 16 - SHIFT_BITS);

    left = _mm256_srai_epi32(
        _mm256_madd_epi16(_mm256_blend_epi16(leftleft, leftright, 0xaa), shift2D),
        SHIFT_BITS
    );

    right = _mm256_slli_epi32(
        _mm256_madd_epi16(_mm256_blend_epi16(rightleft, rightright, 0xaa), shift2D),
        16 - SHIFT_BITS
    );

    return _mm256_srai_epi32(
        _mm256_madd_epi16(_mm256_blend_epi16(left, right, 0xaa), shift3D), SHIFT_BITS
    );
}
#endif

/*
 Transforms colors of imIn using provided 3D lookup table
 and puts the result in imOut. Returns imOut on success or 0 on error.

 imOut, imIn - images, should be the same size and may be the same image.
    Should have 3 or 4 channels.
 table_channels - number of channels in the lookup table, 3 or 4.
    Should be less or equal than number of channels in imOut image;
 size1D, size_2D and size3D - dimensions of provided table;
 table - flat table,
    array with table_channels * size1D * size2D * size3D elements,
    where channels are changed first, then 1D, then 2D, then 3D.
    Each element is signed 16-bit int where 0 is lowest output value
    and 255 << PRECISION_BITS (16320) is highest value.
*/
Imaging
ImagingColorLUT3D_linear(
    Imaging imOut,
    Imaging imIn,
    int table_channels,
    int size1D,
    int size2D,
    int size3D,
    INT16 *table
) {
    /* This float to int conversion doesn't have rounding
       error compensation (+0.5) for two reasons:
       1. As we don't hit the highest value,
          we can use one extra bit for precision.
       2. For every pixel, we interpolate 8 elements from the table:
          current and +1 for every dimension and their combinations.
          If we hit the upper cells from the table,
          +1 cells will be outside of the table.
          With this compensation we never hit the upper cells
          but this also doesn't introduce any noticeable difference. */
    int y, size1D_2D = size1D * size2D;
    int size2D_shift = size1D * table_channels;
    int size3D_shift = size1D_2D * table_channels;
    ImagingSectionCookie cookie;
    __m128i scale = _mm_set_epi16(
        0,
        (int)((size3D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size2D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size1D - 1) / 255.0 * (1 << SCALE_BITS)),
        0,
        (int)((size3D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size2D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size1D - 1) / 255.0 * (1 << SCALE_BITS))
    );
    __m128i scale_mask = _mm_set1_epi16(SCALE_MASK >> 8);
    __m128i index_mul = _mm_set_epi16(
        0,
        size1D_2D * table_channels,
        size1D * table_channels,
        table_channels,
        0,
        size1D_2D * table_channels,
        size1D * table_channels,
        table_channels
    );
    __m128i shuffle3 =
        _mm_set_epi8(-1, -1, -1, -1, 11, 10, 5, 4, 9, 8, 3, 2, 7, 6, 1, 0);
    __m128i shuffle4 =
        _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
#if defined(IMAGING_SIMD_AVX2)
    __m256i scale256 = _mm256_set_epi16(
        0,
        (int)((size3D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size2D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size1D - 1) / 255.0 * (1 << SCALE_BITS)),
        0,
        (int)((size3D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size2D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size1D - 1) / 255.0 * (1 << SCALE_BITS)),
        0,
        (int)((size3D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size2D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size1D - 1) / 255.0 * (1 << SCALE_BITS)),
        0,
        (int)((size3D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size2D - 1) / 255.0 * (1 << SCALE_BITS)),
        (int)((size1D - 1) / 255.0 * (1 << SCALE_BITS))
    );
    __m256i scale_mask256 = _mm256_set1_epi16(SCALE_MASK >> 8);
    // clang-format off
    __m256i index_mul256 = _mm256_set_epi16(
        0, size1D_2D * table_channels, size1D * table_channels, table_channels,
        0, size1D_2D * table_channels, size1D * table_channels, table_channels,
        0, size1D_2D * table_channels, size1D * table_channels, table_channels,
        0, size1D_2D * table_channels, size1D * table_channels, table_channels
    );
    __m256i shuffle3_256 = _mm256_set_epi8(
        -1,-1, -1,-1, 11,10, 5,4, 9,8, 3,2, 7,6, 1,0,
        -1,-1, -1,-1, 11,10, 5,4, 9,8, 3,2, 7,6, 1,0
    );
    __m256i shuffle4_256 = _mm256_set_epi8(
        15,14, 7,6, 13,12, 5,4, 11,10, 3,2, 9,8, 1,0,
        15,14, 7,6, 13,12, 5,4, 11,10, 3,2, 9,8, 1,0
    );
    // clang-format on
#endif

    if (table_channels < 3 || table_channels > 4) {
        PyErr_SetString(PyExc_ValueError, "table_channels could be 3 or 4");
        return NULL;
    }

    if (imIn->type != IMAGING_TYPE_UINT8 || imOut->type != IMAGING_TYPE_UINT8 ||
        imIn->bands < 3 || imOut->bands < table_channels) {
        return (Imaging)ImagingError_ModeError();
    }

    /* In case we have one extra band in imOut and don't have in imIn.*/
    if (imOut->bands > table_channels && imOut->bands > imIn->bands) {
        return (Imaging)ImagingError_ModeError();
    }

    ImagingSectionEnter(&cookie);
    for (y = 0; y < imOut->ysize; y++) {
        UINT32 *rowIn = (UINT32 *)imIn->image[y];
        UINT32 *rowOut = (UINT32 *)imOut->image[y];
        int x = 0;

#if defined(IMAGING_SIMD_AVX2)
        // This makes sense if 12 or more pixels remain (two loops)
        if (x < imOut->xsize - 11) {
            __m128i source = _mm_loadu_si128((__m128i *)&rowIn[x]);
            // scale up to 16 bits, but scale * 255 * 256 up to 31 bits
            // bi, gi and ri - 6 bits index
            // bs, gs and rs - 9 bits shift
            // 00 bi3.bs3 gi3.gs3 ri3.rs3 00 bi2.bs2 gi2.gs2 ri2.rs2
            // 00 bi1.bs1 gi1.gs1 ri1.rs1 00 bi0.bs0 gi0.gs0 ri0.rs0
            __m256i index = _mm256_mulhi_epu16(
                scale256, _mm256_slli_epi16(_mm256_cvtepu8_epi16(source), 8)
            );
            // 0000 0000 idx3 idx2
            // 0000 0000 idx1 idx0
            __m256i index_src = _mm256_hadd_epi32(
                _mm256_madd_epi16(index_mul256, _mm256_srli_epi16(index, SHIFT_BITS)),
                _mm256_setzero_si256()
            );

            for (; x < imOut->xsize - 7; x += 4) {
                __m128i next_source = _mm_loadu_si128((__m128i *)&rowIn[x + 4]);
                // scale up to 16 bits, but scale * 255 * 256 up to 31 bits
                // bi, gi and ri - 6 bits index
                // bs, gs and rs - 9 bits shift
                // 00 bi3.bs3 gi3.gs3 ri3.rs3 00 bi2.bs2 gi2.gs2 ri2.rs2
                // 00 bi1.bs1 gi1.gs1 ri1.rs1 00 bi0.bs0 gi0.gs0 ri0.rs0
                __m256i next_index = _mm256_mulhi_epu16(
                    scale256, _mm256_slli_epi16(_mm256_cvtepu8_epi16(next_source), 8)
                );
                // 0000 0000 idx3 idx2
                // 0000 0000 idx1 idx0
                __m256i next_index_src = _mm256_hadd_epi32(
                    _mm256_madd_epi16(
                        index_mul256, _mm256_srli_epi16(next_index, SHIFT_BITS)
                    ),
                    _mm256_setzero_si256()
                );

                int idx0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(index_src));
                int idx1 = _mm256_extract_epi32(index_src, 1);
                int idx2 = _mm256_extract_epi32(index_src, 4);
                int idx3 = _mm256_extract_epi32(index_src, 5);

                // 00 0.bs3 0.gs3 0.rs3 00 0.bs2 0.gs2 0.rs2
                // 00 0.bs1 0.gs1 0.rs1 00 0.bs0 0.gs0 0.rs0
                __m256i shift = _mm256_and_si256(index, scale_mask256);
                // 11 1-bs3 1-gs3 1-rs3 11 1-bs2 1-gs2 1-rs2
                // 11 1-bs1 1-gs1 1-rs1 11 1-bs0 1-gs0 1-rs0
                __m256i inv_shift =
                    _mm256_sub_epi16(_mm256_set1_epi16((1 << SHIFT_BITS)), shift);

                // 00 11 bs2 1-bs2 gs2 1-gs2 rs2 1-rs2
                // 00 11 bs0 1-bs0 gs0 1-gs0 rs0 1-rs0
                __m256i shift0 = _mm256_unpacklo_epi16(inv_shift, shift);

                // 00 11 bs3 1-bs3 gs3 1-gs3 rs3 1-rs3
                // 00 11 bs1 1-bs1 gs1 1-gs1 rs1 1-rs1
                __m256i shift1 = _mm256_unpackhi_epi16(inv_shift, shift);

                if (table_channels == 3) {
                    __m128i dest;
                    // a2 b2 g2 r2 a0 b0 g0 r0
                    __m256i result0 = mm256_lut_pixel(
                        table,
                        size2D_shift,
                        size3D_shift,
                        idx0,
                        idx2,
                        shift0,
                        shuffle3_256
                    );
                    // a3 b3 g3 r3 a1 b1 g1 r1
                    __m256i result1 = mm256_lut_pixel(
                        table,
                        size2D_shift,
                        size3D_shift,
                        idx1,
                        idx3,
                        shift1,
                        shuffle3_256
                    );

                    // a3 b3 g3 r3 a2 b2 g2 r2
                    // a1 b1 g1 r1 a0 b0 g0 r0
                    result0 = _mm256_packs_epi32(result0, result1);
                    result0 = _mm256_srai_epi16(
                        _mm256_add_epi16(
                            result0, _mm256_set1_epi16(PRECISION_ROUNDING)
                        ),
                        PRECISION_BITS
                    );

                    // a3 b3 g3 r3 a2 b2 g2 r2 a1 b1 g1 r1 a0 b0 g0 r0
                    dest = _mm_packus_epi16(
                        _mm256_castsi256_si128(result0),
                        _mm256_extracti128_si256(result0, 1)
                    );
                    dest = _mm_blendv_epi8(
                        dest,
                        source,
                        _mm_set_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0)
                    );
                    _mm_storeu_si128((__m128i *)&rowOut[x], dest);
                }

                if (table_channels == 4) {
                    __m128i dest;
                    __m256i result0 = mm256_lut_pixel(
                        table,
                        size2D_shift,
                        size3D_shift,
                        idx0,
                        idx2,
                        shift0,
                        shuffle4_256
                    );
                    __m256i result1 = mm256_lut_pixel(
                        table,
                        size2D_shift,
                        size3D_shift,
                        idx1,
                        idx3,
                        shift1,
                        shuffle4_256
                    );

                    result0 = _mm256_packs_epi32(result0, result1);
                    result0 = _mm256_srai_epi16(
                        _mm256_add_epi16(
                            result0, _mm256_set1_epi16(PRECISION_ROUNDING)
                        ),
                        PRECISION_BITS
                    );

                    dest = _mm_packus_epi16(
                        _mm256_castsi256_si128(result0),
                        _mm256_extracti128_si256(result0, 1)
                    );
                    _mm_storeu_si128((__m128i *)&rowOut[x], dest);
                }
                source = next_source;
                index = next_index;
                index_src = next_index_src;
            }
        }
#endif
        // This makes sense if 6 or more pixels remain (two loops)
        if (x < imOut->xsize - 5) {
            __m128i source = _mm_loadl_epi64((__m128i *)&rowIn[x]);
            // scale up to 16 bits, but scale * 255 * 256 up to 31 bits
            // bi, gi and ri - 6 bits index
            // bs, gs and rs - 9 bits shift
            // 00 bi1.bs1 gi1.gs1 ri1.rs1 00 bi0.bs0 gi0.gs0 ri0.rs0
            __m128i index =
                _mm_mulhi_epu16(scale, _mm_unpacklo_epi8(_mm_setzero_si128(), source));
            // 0000 0000 idx1 idx0
            __m128i index_src = _mm_hadd_epi32(
                _mm_madd_epi16(index_mul, _mm_srli_epi16(index, SHIFT_BITS)),
                _mm_setzero_si128()
            );

            for (; x < imOut->xsize - 3; x += 2) {
                __m128i next_source = _mm_loadl_epi64((__m128i *)&rowIn[x + 2]);
                __m128i next_index = _mm_mulhi_epu16(
                    scale, _mm_unpacklo_epi8(_mm_setzero_si128(), next_source)
                );
                __m128i next_index_src = _mm_hadd_epi32(
                    _mm_madd_epi16(index_mul, _mm_srli_epi16(next_index, SHIFT_BITS)),
                    _mm_setzero_si128()
                );

                int idx0 = _mm_cvtsi128_si32(index_src);
                int idx1 = _mm_cvtsi128_si32(_mm_srli_si128(index_src, 4));

                // 00 0.bs1 0.gs1 0.rs1 00 0.bs0 0.gs0 0.rs0
                __m128i shift = _mm_and_si128(index, scale_mask);
                // 11 1-bs1 1-gs1 1-rs1 11 1-bs0 1-gs0 1-rs0
                __m128i inv_shift =
                    _mm_sub_epi16(_mm_set1_epi16((1 << SHIFT_BITS)), shift);

                // 00 11 bs0 1-bs0 gs0 1-gs0 rs0 1-rs0
                __m128i shift0 = _mm_unpacklo_epi16(inv_shift, shift);

                // 00 11 bs1 1-bs1 gs1 1-gs1 rs1 1-rs1
                __m128i shift1 = _mm_unpackhi_epi16(inv_shift, shift);

                if (table_channels == 3) {
                    __m128i result0 = mm_lut_pixel(
                        table, size2D_shift, size3D_shift, idx0, shift0, shuffle3
                    );
                    __m128i result1 = mm_lut_pixel(
                        table, size2D_shift, size3D_shift, idx1, shift1, shuffle3
                    );

                    result0 = _mm_packs_epi32(result0, result1);
                    result0 = _mm_srai_epi16(
                        _mm_add_epi16(result0, _mm_set1_epi16(PRECISION_ROUNDING)),
                        PRECISION_BITS
                    );

                    result0 = _mm_packus_epi16(result0, result0);
                    result0 = _mm_blendv_epi8(
                        result0,
                        source,
                        _mm_set_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0)
                    );
                    _mm_storel_epi64((__m128i *)&rowOut[x], result0);
                }

                if (table_channels == 4) {
                    __m128i result0 = mm_lut_pixel(
                        table, size2D_shift, size3D_shift, idx0, shift0, shuffle4
                    );
                    __m128i result1 = mm_lut_pixel(
                        table, size2D_shift, size3D_shift, idx1, shift1, shuffle4
                    );

                    result0 = _mm_packs_epi32(result0, result1);
                    result0 = _mm_srai_epi16(
                        _mm_add_epi16(result0, _mm_set1_epi16(PRECISION_ROUNDING)),
                        PRECISION_BITS
                    );
                    _mm_storel_epi64(
                        (__m128i *)&rowOut[x], _mm_packus_epi16(result0, result0)
                    );
                }
                source = next_source;
                index = next_index;
                index_src = next_index_src;
            }
        }

        for (; x < imOut->xsize; x++) {
            __m128i source = _mm_cvtsi32_si128(rowIn[x]);
            // scale up to 16 bits, but scale * 255 * 256 up to 31 bits
            // bi, gi and ri - 6 bits index
            // bs, gs and rs - 9 bits shift
            // 00 00 00 00 00 bi.bs gi.gs ri.rs
            __m128i index =
                _mm_mulhi_epu16(scale, _mm_unpacklo_epi8(_mm_setzero_si128(), source));

            int idx0 = _mm_cvtsi128_si32(_mm_hadd_epi32(
                _mm_madd_epi16(index_mul, _mm_srli_epi16(index, SHIFT_BITS)),
                _mm_setzero_si128()
            ));

            // 00 00 00 00 00 0.bs 0.gs 0.rs
            __m128i shift = _mm_and_si128(index, scale_mask);
            // 11 11 11 11 11 1-bs 1-gs 1-rs
            __m128i inv_shift = _mm_sub_epi16(_mm_set1_epi16((1 << SHIFT_BITS)), shift);

            // 00 11 bs 1-bs gs 1-gs rs 1-rs
            __m128i shift0 = _mm_unpacklo_epi16(inv_shift, shift);

            if (table_channels == 3) {
                __m128i result = mm_lut_pixel(
                    table, size2D_shift, size3D_shift, idx0, shift0, shuffle3
                );

                result = _mm_packs_epi32(result, result);
                result = _mm_srai_epi16(
                    _mm_add_epi16(result, _mm_set1_epi16(PRECISION_ROUNDING)),
                    PRECISION_BITS
                );

                result = _mm_packus_epi16(result, result);
                result = _mm_blendv_epi8(
                    result,
                    source,
                    _mm_set_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0)
                );
                rowOut[x] = _mm_cvtsi128_si32(result);
            }

            if (table_channels == 4) {
                __m128i result = mm_lut_pixel(
                    table, size2D_shift, size3D_shift, idx0, shift0, shuffle4
                );

                result = _mm_packs_epi32(result, result);
                result = _mm_srai_epi16(
                    _mm_add_epi16(result, _mm_set1_epi16(PRECISION_ROUNDING)),
                    PRECISION_BITS
                );
                rowOut[x] = _mm_cvtsi128_si32(_mm_packus_epi16(result, result));
            }
        }
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}

#else

/* Portable version, as in upstream Pillow. */

/* 8 bits for result. Table can overflow [0, 1.0] range,
   so we need extra bits for overflow and negative values.
   NOTE: This value should be the same as in _imaging/_prepare_lut_table() */
#define PRECISION_BITS (16 - 8 - 2)
#define PRECISION_ROUNDING (1 << (PRECISION_BITS - 1))

/* 8 - scales are multiplied on byte.
   6 - max index in the table
       (max size is 65, but index 64 is not reachable) */
#define SCALE_BITS (32 - 8 - 6)
#define SCALE_MASK ((1 << SCALE_BITS) - 1)

#define SHIFT_BITS (16 - 1)

static inline UINT8
clip8(int in) {
    return clip8_lookups[(in + PRECISION_ROUNDING) >> PRECISION_BITS];
}

static inline void
interpolate3(INT16 out[3], const INT16 a[3], const INT16 b[3], INT16 shift) {
    out[0] = (a[0] * ((1 << SHIFT_BITS) - shift) + b[0] * shift) >> SHIFT_BITS;
    out[1] = (a[1] * ((1 << SHIFT_BITS) - shift) + b[1] * shift) >> SHIFT_BITS;
    out[2] = (a[2] * ((1 << SHIFT_BITS) - shift) + b[2] * shift) >> SHIFT_BITS;
}

static inline void
interpolate4(INT16 out[4], const INT16 a[4], const INT16 b[4], INT16 shift) {
    out[0] = (a[0] * ((1 << SHIFT_BITS) - shift) + b[0] * shift) >> SHIFT_BITS;
    out[1] = (a[1] * ((1 << SHIFT_BITS) - shift) + b[1] * shift) >> SHIFT_BITS;
    out[2] = (a[2] * ((1 << SHIFT_BITS) - shift) + b[2] * shift) >> SHIFT_BITS;
    out[3] = (a[3] * ((1 << SHIFT_BITS) - shift) + b[3] * shift) >> SHIFT_BITS;
}

static inline int
table_index3D(int index1D, int index2D, int index3D, int size1D, int size1D_2D) {
    return index1D + index2D * size1D + index3D * size1D_2D;
}

#if defined(IMAGING_SIMD_NEON)
/* interpolate3/4 for the channels in the lanes:
   (a * (2^15 - s) + b * s) >> 15 == a + (((b - a) * s) >> 15). */
static inline int32x4_t
neon_lut_interpolate(int32x4_t a, int32x4_t b, INT32 shift) {
    return vaddq_s32(a, vshrq_n_s32(vmulq_n_s32(vsubq_s32(b, a), shift), SHIFT_BITS));
}

/* Four table values from p. For a 3-channel table the fourth is the next
   entry's first value, or at the end the table's padding; it is unused. */
static inline int32x4_t
neon_lut_load(const INT16 *p) {
    return vmovl_s16(vld1_s16(p));
}

/* One pixel's lookup with the same arithmetic as the scalar code below,
   rounded and clipped to 0..255 in each lane. */
static inline uint8x8_t
neon_lut_pixel(
    const INT16 *table,
    int idx,
    int step1D,
    int step2D,
    int step3D,
    INT32 shift1D,
    INT32 shift2D,
    INT32 shift3D
) {
    const INT16 *t = &table[idx];
    int32x4_t left = neon_lut_interpolate(
        neon_lut_interpolate(neon_lut_load(t), neon_lut_load(t + step1D), shift1D),
        neon_lut_interpolate(
            neon_lut_load(t + step2D), neon_lut_load(t + step2D + step1D), shift1D
        ),
        shift2D
    );
    t += step3D;
    int32x4_t right = neon_lut_interpolate(
        neon_lut_interpolate(neon_lut_load(t), neon_lut_load(t + step1D), shift1D),
        neon_lut_interpolate(
            neon_lut_load(t + step2D), neon_lut_load(t + step2D + step1D), shift1D
        ),
        shift2D
    );
    int32x4_t result = neon_lut_interpolate(left, right, shift3D);
    // clip8: (result + PRECISION_ROUNDING) >> PRECISION_BITS, then 0..255
    result = vshrq_n_s32(vaddq_s32(result, vdupq_n_s32(PRECISION_ROUNDING)), PRECISION_BITS);
    return vqmovn_u16(vcombine_u16(vqmovun_s32(result), vdup_n_u16(0)));
}
#endif

/*
 Transforms colors of imIn using provided 3D lookup table
 and puts the result in imOut. Returns imOut on success or 0 on error.

 imOut, imIn - images, should be the same size and may be the same image.
    Should have 3 or 4 channels.
 table_channels - number of channels in the lookup table, 3 or 4.
    Should be less or equal than number of channels in imOut image;
 size1D, size_2D and size3D - dimensions of provided table;
 table - flat table,
    array with table_channels * size1D * size2D * size3D elements,
    where channels are changed first, then 1D, then 2D, then 3D.
    Each element is signed 16-bit int where 0 is lowest output value
    and 255 << PRECISION_BITS (16320) is highest value.
*/
Imaging
ImagingColorLUT3D_linear(
    Imaging imOut,
    Imaging imIn,
    int table_channels,
    int size1D,
    int size2D,
    int size3D,
    INT16 *table
) {
    /* This float to int conversion doesn't have rounding
       error compensation (+0.5) for two reasons:
       1. As we don't hit the highest value,
          we can use one extra bit for precision.
       2. For every pixel, we interpolate 8 elements from the table:
          current and +1 for every dimension and their combinations.
          If we hit the upper cells from the table,
          +1 cells will be outside of the table.
          With this compensation we never hit the upper cells
          but this also doesn't introduce any noticeable difference. */
    UINT32 scale1D = (size1D - 1) / 255.0 * (1 << SCALE_BITS);
    UINT32 scale2D = (size2D - 1) / 255.0 * (1 << SCALE_BITS);
    UINT32 scale3D = (size3D - 1) / 255.0 * (1 << SCALE_BITS);
    int size1D_2D = size1D * size2D;
    int x, y;
    ImagingSectionCookie cookie;

    if (table_channels < 3 || table_channels > 4) {
        PyErr_SetString(PyExc_ValueError, "table_channels could be 3 or 4");
        return NULL;
    }

    if (imIn->type != IMAGING_TYPE_UINT8 || imOut->type != IMAGING_TYPE_UINT8 ||
        imIn->bands < 3 || imOut->bands < table_channels) {
        return (Imaging)ImagingError_ModeError();
    }

    /* In case we have one extra band in imOut and don't have in imIn.*/
    if (imOut->bands > table_channels && imOut->bands > imIn->bands) {
        return (Imaging)ImagingError_ModeError();
    }

    ImagingSectionEnter(&cookie);
    for (y = 0; y < imOut->ysize; y++) {
        UINT8 *rowIn = (UINT8 *)imIn->image[y];
        char *rowOut = (char *)imOut->image[y];
        for (x = 0; x < imOut->xsize; x++) {
            UINT32 index1D = rowIn[x * 4 + 0] * scale1D;
            UINT32 index2D = rowIn[x * 4 + 1] * scale2D;
            UINT32 index3D = rowIn[x * 4 + 2] * scale3D;
            INT16 shift1D = (SCALE_MASK & index1D) >> (SCALE_BITS - SHIFT_BITS);
            INT16 shift2D = (SCALE_MASK & index2D) >> (SCALE_BITS - SHIFT_BITS);
            INT16 shift3D = (SCALE_MASK & index3D) >> (SCALE_BITS - SHIFT_BITS);
            int idx = table_channels * table_index3D(
                                           index1D >> SCALE_BITS,
                                           index2D >> SCALE_BITS,
                                           index3D >> SCALE_BITS,
                                           size1D,
                                           size1D_2D
                                       );
#if defined(IMAGING_SIMD_NEON)
            {
                uint8x8_t px = neon_lut_pixel(
                    table,
                    idx,
                    table_channels,
                    size1D * table_channels,
                    size1D_2D * table_channels,
                    shift1D,
                    shift2D,
                    shift3D
                );
                if (table_channels == 3) {
                    px = vset_lane_u8(rowIn[x * 4 + 3], px, 3);
                }
                vst1_lane_u32(
                    (uint32_t *)(void *)(rowOut + x * 4), vreinterpret_u32_u8(px), 0
                );
            }
#else
            INT16 result[4], left[4], right[4];
            INT16 leftleft[4], leftright[4], rightleft[4], rightright[4];

            if (table_channels == 3) {
                UINT32 v;
                interpolate3(leftleft, &table[idx + 0], &table[idx + 3], shift1D);
                interpolate3(
                    leftright,
                    &table[idx + size1D * 3],
                    &table[idx + size1D * 3 + 3],
                    shift1D
                );
                interpolate3(left, leftleft, leftright, shift2D);

                interpolate3(
                    rightleft,
                    &table[idx + size1D_2D * 3],
                    &table[idx + size1D_2D * 3 + 3],
                    shift1D
                );
                interpolate3(
                    rightright,
                    &table[idx + size1D_2D * 3 + size1D * 3],
                    &table[idx + size1D_2D * 3 + size1D * 3 + 3],
                    shift1D
                );
                interpolate3(right, rightleft, rightright, shift2D);

                interpolate3(result, left, right, shift3D);

                v = MAKE_UINT32(
                    clip8(result[0]),
                    clip8(result[1]),
                    clip8(result[2]),
                    rowIn[x * 4 + 3]
                );
                memcpy(rowOut + x * sizeof(v), &v, sizeof(v));
            }

            if (table_channels == 4) {
                UINT32 v;
                interpolate4(leftleft, &table[idx + 0], &table[idx + 4], shift1D);
                interpolate4(
                    leftright,
                    &table[idx + size1D * 4],
                    &table[idx + size1D * 4 + 4],
                    shift1D
                );
                interpolate4(left, leftleft, leftright, shift2D);

                interpolate4(
                    rightleft,
                    &table[idx + size1D_2D * 4],
                    &table[idx + size1D_2D * 4 + 4],
                    shift1D
                );
                interpolate4(
                    rightright,
                    &table[idx + size1D_2D * 4 + size1D * 4],
                    &table[idx + size1D_2D * 4 + size1D * 4 + 4],
                    shift1D
                );
                interpolate4(right, rightleft, rightright, shift2D);

                interpolate4(result, left, right, shift3D);

                v = MAKE_UINT32(
                    clip8(result[0]),
                    clip8(result[1]),
                    clip8(result[2]),
                    clip8(result[3])
                );
                memcpy(rowOut + x * sizeof(v), &v, sizeof(v));
            }
#endif
        }
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}

#endif

#if defined(IMAGING_SIMD_SSE4)


void
ImagingResampleVerticalConvolution8u(UINT32 *restrict lineOut, Imaging imIn,
    int xmin, int xmax, INT16 *k, int coefs_precision, int xsize)
{
    int x;
    int xx = 0;

    __m128i initial = _mm_set1_epi32(1 << (coefs_precision-1));

#if defined(IMAGING_SIMD_AVX2)

    __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision-1));

    for (; xx < xsize - 7; xx += 8) {
        __m256i sss0 = initial_256;
        __m256i sss1 = initial_256;
        __m256i sss2 = initial_256;
        __m256i sss3 = initial_256;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m256i source, source1, source2;
            __m256i pix, mmk;

            // Load two coefficients at once
            mmk = _mm256_set1_epi32(*(INT32 *) &k[x]);
            
            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &((UINT32 *)imIn->image[x + xmin])[xx]);
            source2 = _mm256_loadu_si256(  // bottom line
                (__m256i *) &((UINT32 *)imIn->image[x + 1 + xmin])[xx]);

            source = _mm256_unpacklo_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = mm256_dpwssd(sss0, pix, mmk);
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = mm256_dpwssd(sss1, pix, mmk);

            source = _mm256_unpackhi_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = mm256_dpwssd(sss2, pix, mmk);
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = mm256_dpwssd(sss3, pix, mmk);
        }
        for (; x < xmax; x += 1) {
            __m256i source, source1, pix, mmk;
            mmk = _mm256_set1_epi32(k[x]);
            
            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &((UINT32 *)imIn->image[x + xmin])[xx]);
            
            source = _mm256_unpacklo_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = mm256_dpwssd(sss0, pix, mmk);
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = mm256_dpwssd(sss1, pix, mmk);

            source = _mm256_unpackhi_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = mm256_dpwssd(sss2, pix, mmk);
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = mm256_dpwssd(sss3, pix, mmk);
        }
        sss0 = _mm256_srai_epi32(sss0, coefs_precision);
        sss1 = _mm256_srai_epi32(sss1, coefs_precision);
        sss2 = _mm256_srai_epi32(sss2, coefs_precision);
        sss3 = _mm256_srai_epi32(sss3, coefs_precision);

        sss0 = _mm256_packs_epi32(sss0, sss1);
        sss2 = _mm256_packs_epi32(sss2, sss3);
        sss0 = _mm256_packus_epi16(sss0, sss2);
        _mm256_storeu_si256((__m256i *) &lineOut[xx], sss0);
    }

#else

    for (; xx < xsize - 7; xx += 8) {
        __m128i sss0 = initial;
        __m128i sss1 = initial;
        __m128i sss2 = initial;
        __m128i sss3 = initial;
        __m128i sss4 = initial;
        __m128i sss5 = initial;
        __m128i sss6 = initial;
        __m128i sss7 = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &((UINT32 *)imIn->image[x + xmin])[xx]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &((UINT32 *)imIn->image[x + 1 + xmin])[xx]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = mm_dpwssd(sss0, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = mm_dpwssd(sss1, pix, mmk);

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = mm_dpwssd(sss2, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = mm_dpwssd(sss3, pix, mmk);
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &((UINT32 *)imIn->image[x + xmin])[xx + 4]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &((UINT32 *)imIn->image[x + 1 + xmin])[xx + 4]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = mm_dpwssd(sss4, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = mm_dpwssd(sss5, pix, mmk);

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = mm_dpwssd(sss6, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = mm_dpwssd(sss7, pix, mmk);
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &((UINT32 *)imIn->image[x + xmin])[xx]);
            
            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = mm_dpwssd(sss0, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = mm_dpwssd(sss1, pix, mmk);

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = mm_dpwssd(sss2, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = mm_dpwssd(sss3, pix, mmk);

            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &((UINT32 *)imIn->image[x + xmin])[xx + 4]);

            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = mm_dpwssd(sss4, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = mm_dpwssd(sss5, pix, mmk);

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = mm_dpwssd(sss6, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = mm_dpwssd(sss7, pix, mmk);
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);
        sss2 = _mm_srai_epi32(sss2, coefs_precision);
        sss3 = _mm_srai_epi32(sss3, coefs_precision);
        sss4 = _mm_srai_epi32(sss4, coefs_precision);
        sss5 = _mm_srai_epi32(sss5, coefs_precision);
        sss6 = _mm_srai_epi32(sss6, coefs_precision);
        sss7 = _mm_srai_epi32(sss7, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss2 = _mm_packs_epi32(sss2, sss3);
        sss0 = _mm_packus_epi16(sss0, sss2);
        _mm_storeu_si128((__m128i *) &lineOut[xx], sss0);
        sss4 = _mm_packs_epi32(sss4, sss5);
        sss6 = _mm_packs_epi32(sss6, sss7);
        sss4 = _mm_packus_epi16(sss4, sss6);
        _mm_storeu_si128((__m128i *) &lineOut[xx + 4], sss4);
    }

#endif

    for (; xx < xsize - 1; xx += 2) {
        __m128i sss0 = initial;  // left row
        __m128i sss1 = initial;  // right row
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm_loadl_epi64(  // top line
                (__m128i *) &((UINT32 *)imIn->image[x + xmin])[xx]);
            source2 = _mm_loadl_epi64(  // bottom line
                (__m128i *) &((UINT32 *)imIn->image[x + 1 + xmin])[xx]);
            
            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = mm_dpwssd(sss0, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = mm_dpwssd(sss1, pix, mmk);
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);
            
            source1 = _mm_loadl_epi64(  // top line
                (__m128i *) &((UINT32 *)imIn->image[x + xmin])[xx]);
            
            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = mm_dpwssd(sss0, pix, mmk);
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = mm_dpwssd(sss1, pix, mmk);
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss0 = _mm_packus_epi16(sss0, sss0);
        _mm_storel_epi64((__m128i *) &lineOut[xx], sss0);
    }

    for (; xx < xsize; xx++) {
        __m128i sss = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;
            
            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm_cvtsi32_si128(  // top line
                mm_load_si32(&((UINT32 *)imIn->image[x + xmin])[xx]));
            source2 = _mm_cvtsi32_si128(  // bottom line
                mm_load_si32(&((UINT32 *)imIn->image[x + 1 + xmin])[xx]));
            
            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss = mm_dpwssd(sss, pix, mmk);
        }
        for (; x < xmax; x++) {
            __m128i pix = mm_cvtepu8_epi32(&((UINT32 *)imIn->image[x + xmin])[xx]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = mm_dpwssd(sss, pix, mmk);
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        UINT32 v = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
        memcpy(&lineOut[xx], &v, sizeof(v));
    }
}

#elif defined(IMAGING_SIMD_NEON)

/* Accumulates 16 bytes of one row into four 32-bit sums. */
#define NEON_VERT_MLA(s0, s1, s2, s3, src, kx)                                    \
    {                                                                             \
        int16x8_t lo_ = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src)));        \
        int16x8_t hi_ = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src)));       \
        s0 = vmlal_n_s16(s0, vget_low_s16(lo_), kx);                              \
        s1 = vmlal_n_s16(s1, vget_high_s16(lo_), kx);                             \
        s2 = vmlal_n_s16(s2, vget_low_s16(hi_), kx);                              \
        s3 = vmlal_n_s16(s3, vget_high_s16(hi_), kx);                             \
    }

/* Shifts and saturates 16 sums to bytes, as srai, packs and packus do. */
static inline uint8x16_t
neon_resample_pack16(
    int32x4_t s0, int32x4_t s1, int32x4_t s2, int32x4_t s3, int32x4_t shift
) {
    int16x8_t r0 = vcombine_s16(
        vqmovn_s32(vshlq_s32(s0, shift)), vqmovn_s32(vshlq_s32(s1, shift)));
    int16x8_t r1 = vcombine_s16(
        vqmovn_s32(vshlq_s32(s2, shift)), vqmovn_s32(vshlq_s32(s3, shift)));
    return vcombine_u8(vqmovun_s16(r0), vqmovun_s16(r1));
}

void
ImagingResampleVerticalConvolution8u(UINT32 *restrict lineOut, Imaging imIn,
    int xmin, int xmax, INT16 *k, int coefs_precision, int xsize)
{
    int x, xx = 0;
    int size = xsize * 4;
    UINT8 *out = (UINT8 *)lineOut;
    int32x4_t initial = vdupq_n_s32(1 << (coefs_precision - 1));
    int32x4_t shift = vdupq_n_s32(-coefs_precision);

    for (; xx < size - 31; xx += 32) {
        int32x4_t s0 = initial, s1 = initial, s2 = initial, s3 = initial;
        int32x4_t s4 = initial, s5 = initial, s6 = initial, s7 = initial;
        for (x = 0; x < xmax; x++) {
            const UINT8 *in = (UINT8 *)imIn->image[x + xmin] + xx;
            uint8x16_t src0 = vld1q_u8(in);
            uint8x16_t src1 = vld1q_u8(in + 16);
            NEON_VERT_MLA(s0, s1, s2, s3, src0, k[x]);
            NEON_VERT_MLA(s4, s5, s6, s7, src1, k[x]);
        }
        vst1q_u8(out + xx, neon_resample_pack16(s0, s1, s2, s3, shift));
        vst1q_u8(out + xx + 16, neon_resample_pack16(s4, s5, s6, s7, shift));
    }
    for (; xx < size - 15; xx += 16) {
        int32x4_t s0 = initial, s1 = initial, s2 = initial, s3 = initial;
        for (x = 0; x < xmax; x++) {
            uint8x16_t src = vld1q_u8((UINT8 *)imIn->image[x + xmin] + xx);
            NEON_VERT_MLA(s0, s1, s2, s3, src, k[x]);
        }
        vst1q_u8(out + xx, neon_resample_pack16(s0, s1, s2, s3, shift));
    }
    for (; xx < size; xx += 4) {
        int32x4_t sss = initial;
        for (x = 0; x < xmax; x++) {
            UINT32 v;
            memcpy(&v, (UINT8 *)imIn->image[x + xmin] + xx, sizeof(v));
            int16x4_t pix = vreinterpret_s16_u16(
                vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(v)))));
            sss = vmlal_n_s16(sss, pix, k[x]);
        }
        int16x4_t s16 = vqmovn_s32(vshlq_s32(sss, shift));
        UINT32 r = vget_lane_u32(
            vreinterpret_u32_u8(vqmovun_s16(vcombine_s16(s16, s16))), 0);
        memcpy(out + xx, &r, sizeof(r));
    }
}

#undef NEON_VERT_MLA

#else

/* Portable version of the kernel above, with the same arithmetic.
   Works on chunks of bytes so rows are read sequentially. */
void
ImagingResampleVerticalConvolution8u(UINT32 *restrict lineOut, Imaging imIn,
    int xmin, int xmax, INT16 *k, int coefs_precision, int xsize)
{
    int x, xx, i, n;
    int size = xsize * 4;
    UINT8 *out = (UINT8 *)lineOut;
    int ss[64];

    for (xx = 0; xx < size; xx += n) {
        n = size - xx < 64 ? size - xx : 64;
        for (i = 0; i < n; i++) {
            ss[i] = 1 << (coefs_precision - 1);
        }
        for (x = 0; x < xmax; x++) {
            UINT8 *in = (UINT8 *)imIn->image[x + xmin] + xx;
            int kx = k[x];
            for (i = 0; i < n; i++) {
                ss[i] += in[i] * kx;
            }
        }
        for (i = 0; i < n; i++) {
            out[xx + i] = clip8(ss[i], coefs_precision);
        }
    }
}

#endif

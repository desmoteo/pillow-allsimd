/* NEON 3x3 and 5x5 kernels for 8-bit images, included from Filter.c.

   Same arithmetic as the SSE4 integer kernels: 16-bit coefficients scaled
   by 1 << PRECISION_BITS, 32-bit sums, an arithmetic shift and saturation
   to 0..255. The pixels left over at the end of a row use the same integer
   formula in C, so results don't depend on position.

   Kernel row 0 applies to image row y + R, the last kernel row to y - R,
   and the R border rows and columns are copied, as in the generic code. */

static inline UINT8
filter_clip(INT32 ss) {
    ss >>= PRECISION_BITS;
    return ss < 0 ? 0 : (ss > 255 ? 255 : ss);
}

/* Shifts and saturates 16 sums to bytes. */
static inline uint8x16_t
filter_pack16(int32x4_t s0, int32x4_t s1, int32x4_t s2, int32x4_t s3) {
    int32x4_t shift = vdupq_n_s32(-PRECISION_BITS);
    int16x8_t lo = vcombine_s16(
        vqmovn_s32(vshlq_s32(s0, shift)), vqmovn_s32(vshlq_s32(s1, shift)));
    int16x8_t hi = vcombine_s16(
        vqmovn_s32(vshlq_s32(s2, shift)), vqmovn_s32(vshlq_s32(s3, shift)));
    return vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
}

/* Adds 16 bytes times k to four 32-bit sums. */
#define FILTER_MLA16(s0, s1, s2, s3, ptr, k)                                  \
    {                                                                         \
        uint8x16_t src_ = vld1q_u8(ptr);                                      \
        int16x8_t lo_ = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_)));   \
        int16x8_t hi_ = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src_)));  \
        s0 = vmlal_n_s16(s0, vget_low_s16(lo_), k);                           \
        s1 = vmlal_n_s16(s1, vget_high_s16(lo_), k);                          \
        s2 = vmlal_n_s16(s2, vget_low_s16(hi_), k);                           \
        s3 = vmlal_n_s16(s3, vget_high_s16(hi_), k);                          \
    }

/* size is 3 or 5; bpp is 1 or 4 bytes per pixel. */
static inline void
ImagingFilterNxN_neon(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int size, int bpp
) {
    int x, y, r, d, c;
    int R = size / 2;
    int xsize = im->xsize, ysize = im->ysize;
    /* Each step makes 16 bytes: 16 1-byte pixels or 4 4-byte pixels. */
    int step = 16 / bpp;
    const UINT8 *rows[5];

    offset += 1 << (PRECISION_BITS - 1);

    for (y = 0; y < R; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
    for (y = R; y < ysize - R; y++) {
        UINT8 *out = (UINT8 *)imOut->image[y];
        const UINT8 *in0 = (UINT8 *)im->image[y];
        for (r = 0; r < size; r++) {
            rows[r] = (UINT8 *)im->image[y + R - r];
        }

        memcpy(out, in0, R * bpp);
        x = R;
        for (; x < xsize - R - (step - 1); x += step) {
            int32x4_t s0 = vdupq_n_s32(offset), s1 = s0, s2 = s0, s3 = s0;
            for (r = 0; r < size; r++) {
                for (d = 0; d < size; d++) {
                    FILTER_MLA16(
                        s0, s1, s2, s3, rows[r] + (x + d - R) * bpp, kernel[r * size + d]
                    );
                }
            }
            vst1q_u8(out + x * bpp, filter_pack16(s0, s1, s2, s3));
        }
        for (; x < xsize - R; x++) {
            for (c = 0; c < bpp; c++) {
                INT32 ss = offset;
                for (r = 0; r < size; r++) {
                    for (d = 0; d < size; d++) {
                        ss += rows[r][(x + d - R) * bpp + c] * kernel[r * size + d];
                    }
                }
                out[x * bpp + c] = filter_clip(ss);
            }
        }
        memcpy(out + x * bpp, in0 + x * bpp, R * bpp);
    }
    for (; y < ysize; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
}

#undef FILTER_MLA16

/* Separate entry points so the compiler unrolls each size fully. */
static void
ImagingFilter3x3i_u8_neon(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_neon(imOut, im, kernel, offset, 3, 1);
}

static void
ImagingFilter3x3i_4u8_neon(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_neon(imOut, im, kernel, offset, 3, 4);
}

static void
ImagingFilter5x5i_u8_neon(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_neon(imOut, im, kernel, offset, 5, 1);
}

static void
ImagingFilter5x5i_4u8_neon(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_neon(imOut, im, kernel, offset, 5, 4);
}

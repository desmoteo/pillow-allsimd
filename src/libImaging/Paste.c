/*
 * The Python Imaging Library
 * $Id$
 *
 * paste image on another image
 *
 * history:
 * 96-03-27 fl  Created
 * 96-07-16 fl  Support "1", "L" and "RGBA" masks
 * 96-08-16 fl  Merged with opaque paste
 * 97-01-17 fl  Faster blending, added support for RGBa images
 * 97-08-27 fl  Faster masking for 32-bit images
 * 98-02-02 fl  Fixed MULDIV255 macro for gcc
 * 99-02-02 fl  Added "RGBa" mask support
 * 99-02-06 fl  Rewritten.  Added support for masked fill operations.
 * 99-12-08 fl  Fixed matte fill.
 *
 * Copyright (c) Fredrik Lundh 1996-97.
 * Copyright (c) Secret Labs AB 1997-99.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

#define PREPARE_PASTE_LOOP()        \
    int y, y_end, offset;           \
    if (imOut == imIn && dy > sy) { \
        y = ysize - 1;              \
        y_end = -1;                 \
        offset = -1;                \
    } else {                        \
        y = 0;                      \
        y_end = ysize;              \
        offset = 1;                 \
    }

static inline void
paste(
    Imaging imOut,
    Imaging imIn,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* paste opaque region */

    dx *= pixelsize;
    sx *= pixelsize;

    xsize *= pixelsize;

    PREPARE_PASTE_LOOP();
    for (; y != y_end; y += offset) {
        memcpy(imOut->image[y + dy] + dx, imIn->image[y + sy] + sx, xsize);
    }
}

static inline void
paste_mask_1(
    Imaging imOut,
    Imaging imIn,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* paste with mode "1" mask */

    int x;

    PREPARE_PASTE_LOOP();
    if (imOut->image8) {
        int in_i16 = isModeI16(imIn->mode);
        int out_i16 = isModeI16(imOut->mode);
        for (; y != y_end; y += offset) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            if (out_i16) {
                out += dx;
            }
            UINT8 *in = imIn->image8[y + sy] + sx;
            if (in_i16) {
                in += sx;
            }
            UINT8 *mask = imMask->image8[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                if (*mask) {
                    *out = *in;
                }
                if (in_i16) {
                    in++;
                }
                if (out_i16) {
                    out++;
                    if (*mask) {
                        *out = *in;
                    }
                }
                out++, in++, mask++;
            }
        }

    } else {
        for (; y != y_end; y += offset) {
            INT32 *out = imOut->image32[y + dy] + dx;
            INT32 *in = imIn->image32[y + sy] + sx;
            UINT8 *mask = imMask->image8[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                if (*mask++) {
                    *out = *in;
                }
                out++, in++;
            }
        }
    }
}

static inline void
paste_mask_L(
    Imaging imOut,
    Imaging imIn,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* paste with mode "L" matte */

    int x;
    unsigned int tmp1;

    PREPARE_PASTE_LOOP();
    if (imOut->image8) {
        for (; y != y_end; y += offset) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            UINT8 *in = imIn->image8[y + sy] + sx;
            UINT8 *mask = imMask->image8[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                *out = BLEND(*mask, *out, *in, tmp1);
                out++, in++, mask++;
            }
        }

    } else {
        for (; y != y_end; y += offset) {
            UINT8 *out = (UINT8 *)(imOut->image32[y + dy] + dx);
            UINT8 *in = (UINT8 *)(imIn->image32[y + sy] + sx);
            UINT8 *mask = (UINT8 *)(imMask->image8[y + sy] + sx);
            for (x = 0; x < xsize; x++) {
                UINT8 a = mask[0];
                out[0] = BLEND(a, out[0], in[0], tmp1);
                out[1] = BLEND(a, out[1], in[1], tmp1);
                out[2] = BLEND(a, out[2], in[2], tmp1);
                out[3] = BLEND(a, out[3], in[3], tmp1);
                out += 4;
                in += 4;
                mask++;
            }
        }
    }
}

static inline void
paste_mask_RGBA(
    Imaging imOut,
    Imaging imIn,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* paste with mode "RGBA" matte */

    int x;
    unsigned int tmp1;

    PREPARE_PASTE_LOOP();
    if (imOut->image8) {
        for (; y != y_end; y += offset) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            UINT8 *in = imIn->image8[y + sy] + sx;
            UINT8 *mask = (UINT8 *)imMask->image[y + sy] + sx * 4 + 3;
            for (x = 0; x < xsize; x++) {
                *out = BLEND(*mask, *out, *in, tmp1);
                out++, in++, mask += 4;
            }
        }

    } else {
        for (; y != y_end; y += offset) {
            UINT8 *out = (UINT8 *)(imOut->image32[y + dy] + dx);
            UINT8 *in = (UINT8 *)(imIn->image32[y + sy] + sx);
            UINT8 *mask = (UINT8 *)(imMask->image32[y + sy] + sx);
            for (x = 0; x < xsize; x++) {
                UINT8 a = mask[3];
                out[0] = BLEND(a, out[0], in[0], tmp1);
                out[1] = BLEND(a, out[1], in[1], tmp1);
                out[2] = BLEND(a, out[2], in[2], tmp1);
                out[3] = BLEND(a, out[3], in[3], tmp1);
                out += 4;
                in += 4;
                mask += 4;
            }
        }
    }
}

static inline void
paste_mask_RGBa(
    Imaging imOut,
    Imaging imIn,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* paste with mode "RGBa" matte */

    int x;
    unsigned int tmp1;

    PREPARE_PASTE_LOOP();
    if (imOut->image8) {
        for (; y != y_end; y += offset) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            UINT8 *in = imIn->image8[y + sy] + sx;
            UINT8 *mask = (UINT8 *)imMask->image[y + sy] + sx * 4 + 3;
            for (x = 0; x < xsize; x++) {
                *out = PREBLEND(*mask, *out, *in, tmp1);
                out++, in++, mask += 4;
            }
        }

    } else {
        for (; y != y_end; y += offset) {
            UINT8 *out = (UINT8 *)(imOut->image32[y + dy] + dx);
            UINT8 *in = (UINT8 *)(imIn->image32[y + sy] + sx);
            UINT8 *mask = (UINT8 *)(imMask->image32[y + sy] + sx);
            for (x = 0; x < xsize; x++) {
                UINT8 a = mask[3];
                out[0] = PREBLEND(a, out[0], in[0], tmp1);
                out[1] = PREBLEND(a, out[1], in[1], tmp1);
                out[2] = PREBLEND(a, out[2], in[2], tmp1);
                out[3] = PREBLEND(a, out[3], in[3], tmp1);
                out += 4;
                in += 4;
                mask += 4;
            }
        }
    }
}

int
ImagingPaste(
    Imaging imOut, Imaging imIn, Imaging imMask, int dx0, int dy0, int dx1, int dy1
) {
    int64_t xsize, ysize;
    int pixelsize;
    int sx0, sy0;
    ImagingSectionCookie cookie;

    if (!imOut || !imIn) {
        (void)ImagingError_ModeError();
        return -1;
    }

    pixelsize = imOut->pixelsize;

    xsize = (int64_t)dx1 - dx0;
    ysize = (int64_t)dy1 - dy0;

    if (xsize != imIn->xsize || ysize != imIn->ysize || pixelsize != imIn->pixelsize) {
        (void)ImagingError_Mismatch();
        return -1;
    }

    if (imMask && (xsize != imMask->xsize || ysize != imMask->ysize)) {
        (void)ImagingError_Mismatch();
        return -1;
    }

    /* Determine which region to copy */
    sx0 = sy0 = 0;
    if (dx0 < 0) {
        xsize += dx0, sx0 = -dx0, dx0 = 0;
    }
    if (dx0 + xsize > imOut->xsize) {
        xsize = imOut->xsize - dx0;
    }
    if (dy0 < 0) {
        ysize += dy0, sy0 = -dy0, dy0 = 0;
    }
    if (dy0 + ysize > imOut->ysize) {
        ysize = imOut->ysize - dy0;
    }

    if (xsize <= 0 || ysize <= 0) {
        return 0;
    }

    if (!imMask) {
        ImagingSectionEnter(&cookie);
        paste(imOut, imIn, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_1) {
        ImagingSectionEnter(&cookie);
        paste_mask_1(imOut, imIn, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_L) {
        ImagingSectionEnter(&cookie);
        paste_mask_L(imOut, imIn, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_LA || imMask->mode == IMAGING_MODE_RGBA) {
        ImagingSectionEnter(&cookie);
        paste_mask_RGBA(
            imOut, imIn, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize
        );
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_RGBa) {
        ImagingSectionEnter(&cookie);
        paste_mask_RGBa(
            imOut, imIn, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize
        );
        ImagingSectionLeave(&cookie);
    } else {
        (void)ImagingError_ValueError("bad transparency mask");
        return -1;
    }

    return 0;
}

static inline void
fill(
    Imaging imOut, const UINT8 *ink, int dx, int dy, int xsize, int ysize, int pixelsize
) {
    /* fill opaque region */

    int x, y;
    UINT8 ink8 = 0;
    INT32 ink32 = 0L;

    memcpy(&ink32, ink, pixelsize);
    memcpy(&ink8, ink, sizeof(ink8));

    if (imOut->image8 || ink32 == 0L) {
        dx *= pixelsize;
        xsize *= pixelsize;
        for (y = 0; y < ysize; y++) {
            memset(imOut->image[y + dy] + dx, ink8, xsize);
        }

    } else {
#if defined _WIN32 && !defined _WIN64
        int i;
        dx *= pixelsize;
        for (y = 0; y < ysize; y++) {
            UINT8 *out = (UINT8 *)imOut->image[y + dy] + dx;
            for (x = 0; x < xsize; x++) {
                for (i = 0; i < pixelsize; i++) {
                    *out++ = ink[i];
                }
            }
        }
#else
        for (y = 0; y < ysize; y++) {
            INT32 *out = imOut->image32[y + dy] + dx;
            for (x = 0; x < xsize; x++) {
                out[x] = ink32;
            }
        }
#endif
    }
}

static inline void
fill_mask_1(
    Imaging imOut,
    const void *ink_,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* fill with mode "1" mask */

    int x, y;
    UINT8 ink8 = 0;
    INT32 ink32 = 0L;

    memcpy(&ink32, ink_, pixelsize);
    memcpy(&ink8, ink_, sizeof(ink8));

    if (imOut->image8) {
        for (y = 0; y < ysize; y++) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            UINT8 *mask = imMask->image8[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                if (*mask++) {
                    *out = ink8;
                }
                out++;
            }
        }

    } else {
        for (y = 0; y < ysize; y++) {
            INT32 *out = imOut->image32[y + dy] + dx;
            UINT8 *mask = imMask->image8[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                if (*mask++) {
                    *out = ink32;
                }
                out++;
            }
        }
    }
}

/* BLEND(m, out, ink) = DIV255(out * (255 - m) + ink * m) for vectors of
   bytes: every term fits 16 bits, so the result is the scalar one. */
#if defined(IMAGING_SIMD_SSE4)
static inline __m128i
fill_blend16(__m128i out, __m128i m, __m128i ink) {
    __m128i zero = _mm_setzero_si128(), c255 = _mm_set1_epi16(255);
    __m128i r128 = _mm_set1_epi16(128), res[2];
    int h;
    for (h = 0; h < 2; h++) {
        __m128i o = h ? _mm_unpackhi_epi8(out, zero) : _mm_unpacklo_epi8(out, zero);
        __m128i k = h ? _mm_unpackhi_epi8(m, zero) : _mm_unpacklo_epi8(m, zero);
        __m128i c = h ? _mm_unpackhi_epi8(ink, zero) : _mm_unpacklo_epi8(ink, zero);
        __m128i t = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(o, _mm_sub_epi16(c255, k)), _mm_mullo_epi16(c, k)
            ),
            r128
        );
        res[h] = _mm_srli_epi16(_mm_add_epi16(t, _mm_srli_epi16(t, 8)), 8);
    }
    return _mm_packus_epi16(res[0], res[1]);
}
#elif defined(IMAGING_SIMD_NEON)
static inline uint8x16_t
fill_blend16(uint8x16_t out, uint8x16_t m, uint8x16_t ink) {
    uint8x16_t nm = vsubq_u8(vdupq_n_u8(255), m);
    uint16x8_t lo = vmlal_u8(
        vmull_u8(vget_low_u8(out), vget_low_u8(nm)), vget_low_u8(ink), vget_low_u8(m)
    );
    uint16x8_t hi = vmlal_high_u8(vmull_high_u8(out, nm), ink, m);
    lo = vaddq_u16(lo, vdupq_n_u16(128));
    hi = vaddq_u16(hi, vdupq_n_u16(128));
    // (t + (t >> 8)) >> 8
    return vcombine_u8(
        vshrn_n_u16(vsraq_n_u16(lo, lo, 8), 8), vshrn_n_u16(vsraq_n_u16(hi, hi, 8), 8)
    );
}
#endif

/* Blends ink into n pixels of out through mask, as fill_mask_L's loops
   below: 1-byte pixels, or 4-byte ones, with alpha_channel as there.
   Returns the number of pixels done; the caller does the rest. */
static int
fill_mask_L_simd(
    UINT8 *out, const UINT8 *mask, const UINT8 *ink, int n, int pixelsize, int alpha
) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
    if (pixelsize == 1) {
        __m128i k = _mm_set1_epi8((char)ink[0]);
        for (; x < n - 15; x += 16) {
            __m128i o = _mm_loadu_si128((__m128i *)(out + x));
            __m128i m = _mm_loadu_si128((const __m128i *)(mask + x));
            _mm_storeu_si128((__m128i *)(out + x), fill_blend16(o, m, k));
        }
    } else if (pixelsize == 4) {
        INT32 ink4;
        memcpy(&ink4, ink, 4);
        __m128i k = _mm_set1_epi32(ink4);
        __m128i spread = _mm_set_epi8(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
        __m128i alpha_bytes =
            _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3);
        __m128i colour = _mm_set1_epi32(0x00ffffff), zero = _mm_setzero_si128();
        for (; x < n - 3; x += 4) {
            __m128i o = _mm_loadu_si128((__m128i *)(out + x * 4));
            __m128i m =
                _mm_shuffle_epi8(_mm_cvtsi32_si128(mm_load_si32(mask + x)), spread);
            if (alpha) {
                // colour channels of transparent pixels take a mask of 255
                __m128i transparent =
                    _mm_cmpeq_epi8(_mm_shuffle_epi8(o, alpha_bytes), zero);
                __m128i set = _mm_andnot_si128(
                    _mm_cmpeq_epi8(m, zero), _mm_and_si128(transparent, colour)
                );
                m = _mm_or_si128(m, set);
            }
            _mm_storeu_si128((__m128i *)(out + x * 4), fill_blend16(o, m, k));
        }
    }
#elif defined(IMAGING_SIMD_NEON)
    if (pixelsize == 1) {
        uint8x16_t k = vdupq_n_u8(ink[0]);
        for (; x < n - 15; x += 16) {
            vst1q_u8(out + x, fill_blend16(vld1q_u8(out + x), vld1q_u8(mask + x), k));
        }
    } else if (pixelsize == 4) {
        uint32_t ink4;
        memcpy(&ink4, ink, 4);
        uint8x16_t k = vreinterpretq_u8_u32(vdupq_n_u32(ink4));
        const uint8_t spread_idx[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        const uint8_t alpha_idx[16] = {
            3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15
        };
        uint8x16_t spread = vld1q_u8(spread_idx), alpha_bytes = vld1q_u8(alpha_idx);
        uint8x16_t colour = vreinterpretq_u8_u32(vdupq_n_u32(0x00ffffff));
        for (; x < n - 3; x += 4) {
            uint32_t m4;
            memcpy(&m4, mask + x, 4);
            uint8x16_t o = vld1q_u8(out + x * 4);
            uint8x16_t m = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(m4)), spread);
            if (alpha) {
                uint8x16_t transparent = vceqzq_u8(vqtbl1q_u8(o, alpha_bytes));
                m = vorrq_u8(
                    m, vandq_u8(vandq_u8(transparent, colour), vtstq_u8(m, m))
                );
            }
            vst1q_u8(out + x * 4, fill_blend16(o, m, k));
        }
    }
#endif
    return x;
}

static inline void
fill_mask_L(
    Imaging imOut,
    const UINT8 *ink,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* fill with mode "L" matte */

    int x, y, i;
    unsigned int tmp1;

    if (imOut->image8) {
        int i16 = isModeI16(imOut->mode);
        for (y = 0; y < ysize; y++) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            if (i16) {
                out += dx;
            }
            UINT8 *mask = imMask->image8[y + sy] + sx;
            x = 0;
            if (!i16) {
                x = fill_mask_L_simd(out, mask, ink, xsize, 1, 0);
                out += x, mask += x;
            }
            for (; x < xsize; x++) {
                *out = BLEND(*mask, *out, ink[0], tmp1);
                if (i16) {
                    out++;
                    *out = BLEND(*mask, *out, ink[1], tmp1);
                }
                out++, mask++;
            }
        }

    } else {
        int alpha_channel =
            imOut->mode == IMAGING_MODE_RGBa || imOut->mode == IMAGING_MODE_RGBA ||
            imOut->mode == IMAGING_MODE_La || imOut->mode == IMAGING_MODE_LA ||
            imOut->mode == IMAGING_MODE_PA;
        for (y = 0; y < ysize; y++) {
            UINT8 *out = (UINT8 *)imOut->image[y + dy] + dx * pixelsize;
            UINT8 *mask = (UINT8 *)imMask->image[y + sy] + sx;
            x = fill_mask_L_simd(out, mask, ink, xsize, pixelsize, alpha_channel);
            out += x * pixelsize, mask += x;
            for (; x < xsize; x++) {
                for (i = 0; i < pixelsize; i++) {
                    UINT8 channel_mask = *mask;
                    if (alpha_channel && i != 3 && channel_mask != 0) {
                        channel_mask =
                            255 - (255 - channel_mask) * (1 - (255 - out[3]) / 255);
                    }
                    out[i] = BLEND(channel_mask, out[i], ink[i], tmp1);
                }
                out += pixelsize;
                mask++;
            }
        }
    }
}

static inline void
fill_mask_RGBA(
    Imaging imOut,
    const UINT8 *ink,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* fill with mode "RGBA" matte */

    int x, y, i;
    unsigned int tmp1;

    if (imOut->image8) {
        sx = sx * 4 + 3;
        for (y = 0; y < ysize; y++) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            UINT8 *mask = (UINT8 *)imMask->image[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                *out = BLEND(*mask, *out, ink[0], tmp1);
                out++, mask += 4;
            }
        }

    } else {
        dx *= pixelsize;
        sx = sx * 4 + 3;
        for (y = 0; y < ysize; y++) {
            UINT8 *out = (UINT8 *)imOut->image[y + dy] + dx;
            UINT8 *mask = (UINT8 *)imMask->image[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                for (i = 0; i < pixelsize; i++) {
                    *out = BLEND(*mask, *out, ink[i], tmp1);
                    out++;
                }
                mask += 4;
            }
        }
    }
}

static inline void
fill_mask_RGBa(
    Imaging imOut,
    const UINT8 *ink,
    Imaging imMask,
    int dx,
    int dy,
    int sx,
    int sy,
    int xsize,
    int ysize,
    int pixelsize
) {
    /* fill with mode "RGBa" matte */

    int x, y, i;
    unsigned int tmp1;

    if (imOut->image8) {
        sx = sx * 4 + 3;
        for (y = 0; y < ysize; y++) {
            UINT8 *out = imOut->image8[y + dy] + dx;
            UINT8 *mask = (UINT8 *)imMask->image[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                *out = PREBLEND(*mask, *out, ink[0], tmp1);
                out++, mask += 4;
            }
        }

    } else {
        dx *= pixelsize;
        sx = sx * 4 + 3;
        for (y = 0; y < ysize; y++) {
            UINT8 *out = (UINT8 *)imOut->image[y + dy] + dx;
            UINT8 *mask = (UINT8 *)imMask->image[y + sy] + sx;
            for (x = 0; x < xsize; x++) {
                for (i = 0; i < pixelsize; i++) {
                    *out = PREBLEND(*mask, *out, ink[i], tmp1);
                    out++;
                }
                mask += 4;
            }
        }
    }
}

int
ImagingFill2(
    Imaging imOut, const void *ink, Imaging imMask, int dx0, int dy0, int dx1, int dy1
) {
    ImagingSectionCookie cookie;
    int64_t xsize, ysize;
    int pixelsize;
    int sx0, sy0;

    if (!imOut || !ink) {
        (void)ImagingError_ModeError();
        return -1;
    }

    pixelsize = imOut->pixelsize;

    xsize = (int64_t)dx1 - dx0;
    ysize = (int64_t)dy1 - dy0;

    if (imMask && (xsize != imMask->xsize || ysize != imMask->ysize)) {
        (void)ImagingError_Mismatch();
        return -1;
    }

    /* Determine which region to fill */
    sx0 = sy0 = 0;
    if (dx0 < 0) {
        xsize += dx0;
        if (xsize <= 0) {
            return 0;
        }
        sx0 = -dx0, dx0 = 0;
    }
    if (dx0 + xsize > imOut->xsize) {
        xsize = imOut->xsize - dx0;
    }
    if (dy0 < 0) {
        ysize += dy0;
        if (ysize <= 0) {
            return 0;
        }
        sy0 = -dy0, dy0 = 0;
    }
    if (dy0 + ysize > imOut->ysize) {
        ysize = imOut->ysize - dy0;
    }

    if (xsize <= 0 || ysize <= 0) {
        return 0;
    }

    if (!imMask) {
        ImagingSectionEnter(&cookie);
        fill(imOut, ink, dx0, dy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_1) {
        ImagingSectionEnter(&cookie);
        fill_mask_1(imOut, ink, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_L) {
        ImagingSectionEnter(&cookie);
        fill_mask_L(imOut, ink, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_RGBA) {
        ImagingSectionEnter(&cookie);
        fill_mask_RGBA(imOut, ink, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else if (imMask->mode == IMAGING_MODE_RGBa) {
        ImagingSectionEnter(&cookie);
        fill_mask_RGBa(imOut, ink, imMask, dx0, dy0, sx0, sy0, xsize, ysize, pixelsize);
        ImagingSectionLeave(&cookie);
    } else {
        (void)ImagingError_ValueError("bad transparency mask");
        return -1;
    }

    return 0;
}

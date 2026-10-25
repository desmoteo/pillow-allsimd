/*
 * The Python Imaging Library.
 * $Id$
 *
 * coder for ZIP (deflated) image data
 *
 * History:
 * 96-12-29 fl  created
 * 96-12-30 fl  adaptive filter selection, encoder tuning
 *
 * Copyright (c) Fredrik Lundh 1996.
 * Copyright (c) Secret Labs AB 1997.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

#ifdef HAVE_LIBZ

#include "ZipCodecs.h"

/* PNG filters for one row of n bytes, out[i] = cur[i] - predictor, with
   prev the row above and bpp the bytes per pixel. Each returns the
   filter's cost, the sum over the row of |out[i]| taken as signed bytes,
   min(v, 256 - v), as libpng chooses filters. When encoding, a predictor
   only reads the original rows, so whole vectors of bytes are filtered at
   once; the results and costs are those of the scalar loops. */

#define PNG_COST(v) ((v) < 128 ? (v) : 256 - (v))

#if defined(IMAGING_SIMD_SSE4)
/* min(v, -v) as unsigned bytes is PNG_COST(v); psadbw adds them up. */
static inline __m128i
png_cost16(__m128i v) {
    __m128i c = _mm_min_epu8(v, _mm_sub_epi8(_mm_setzero_si128(), v));
    return _mm_sad_epu8(c, _mm_setzero_si128());
}

/* A row's cost fits 32 bits, and so does each 64-bit half; this also
   builds for 32-bit x86. */
static inline int
png_sum64(__m128i acc) {
    return _mm_cvtsi128_si32(acc) + _mm_cvtsi128_si32(_mm_srli_si128(acc, 8));
}
#elif defined(IMAGING_SIMD_NEON)
static inline uint32x4_t
png_cost16(uint32x4_t acc, uint8x16_t v) {
    uint8x16_t c = vminq_u8(v, vsubq_u8(vdupq_n_u8(0), v));
    return vpadalq_u16(acc, vpaddlq_u8(c));
}
#endif

static int
png_cost_none(const UINT8 *cur, int n) {
    int i = 0, s = 0;
#if defined(IMAGING_SIMD_SSE4)
    __m128i acc = _mm_setzero_si128();
    for (; i < n - 15; i += 16) {
        acc = _mm_add_epi64(acc, png_cost16(_mm_loadu_si128((__m128i *)(cur + i))));
    }
    s = png_sum64(acc);
#elif defined(IMAGING_SIMD_NEON)
    uint32x4_t acc = vdupq_n_u32(0);
    for (; i < n - 15; i += 16) {
        acc = png_cost16(acc, vld1q_u8(cur + i));
    }
    s = (int)vaddvq_u32(acc);
#endif
    for (; i < n; i++) {
        s += PNG_COST(cur[i]);
    }
    return s;
}

/* out[i] = cur[i] - ref[i]: Up with ref the row above, Sub with ref the
   same row bpp bytes back. */
static int
png_filter_diff(UINT8 *out, const UINT8 *cur, const UINT8 *ref, int n) {
    int i = 0, s = 0;
#if defined(IMAGING_SIMD_SSE4)
    __m128i acc = _mm_setzero_si128();
    for (; i < n - 15; i += 16) {
        __m128i v = _mm_sub_epi8(
            _mm_loadu_si128((__m128i *)(cur + i)), _mm_loadu_si128((__m128i *)(ref + i))
        );
        _mm_storeu_si128((__m128i *)(out + i), v);
        acc = _mm_add_epi64(acc, png_cost16(v));
    }
    s = png_sum64(acc);
#elif defined(IMAGING_SIMD_NEON)
    uint32x4_t acc = vdupq_n_u32(0);
    for (; i < n - 15; i += 16) {
        uint8x16_t v = vsubq_u8(vld1q_u8(cur + i), vld1q_u8(ref + i));
        vst1q_u8(out + i, v);
        acc = png_cost16(acc, v);
    }
    s = (int)vaddvq_u32(acc);
#endif
    for (; i < n; i++) {
        UINT8 v = cur[i] - ref[i];
        out[i] = v;
        s += PNG_COST(v);
    }
    return s;
}

/* Average: cur[i] - (left + up) / 2, left being 0 for the first pixel. */
static int
png_filter_average(UINT8 *out, const UINT8 *cur, const UINT8 *prev, int n, int bpp) {
    int i, s = 0;
    for (i = 0; i < bpp && i < n; i++) {
        UINT8 v = cur[i] - prev[i] / 2;
        out[i] = v;
        s += PNG_COST(v);
    }
#if defined(IMAGING_SIMD_SSE4)
    {
        __m128i acc = _mm_setzero_si128(), one = _mm_set1_epi8(1);
        for (; i < n - 15; i += 16) {
            __m128i a = _mm_loadu_si128((__m128i *)(cur + i - bpp));
            __m128i b = _mm_loadu_si128((__m128i *)(prev + i));
            // (a + b) / 2 rounded down: pavgb rounds up
            __m128i avg = _mm_sub_epi8(
                _mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), one)
            );
            __m128i v = _mm_sub_epi8(_mm_loadu_si128((__m128i *)(cur + i)), avg);
            _mm_storeu_si128((__m128i *)(out + i), v);
            acc = _mm_add_epi64(acc, png_cost16(v));
        }
        s += png_sum64(acc);
    }
#elif defined(IMAGING_SIMD_NEON)
    {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i < n - 15; i += 16) {
            // vhaddq_u8 is (a + b) >> 1
            uint8x16_t avg = vhaddq_u8(vld1q_u8(cur + i - bpp), vld1q_u8(prev + i));
            uint8x16_t v = vsubq_u8(vld1q_u8(cur + i), avg);
            vst1q_u8(out + i, v);
            acc = png_cost16(acc, v);
        }
        s += (int)vaddvq_u32(acc);
    }
#endif
    for (; i < n; i++) {
        UINT8 v = cur[i] - (cur[i - bpp] + prev[i]) / 2;
        out[i] = v;
        s += PNG_COST(v);
    }
    return s;
}

/* Paeth: of a (left), b (up) and c (up left), the one closest to
   a + b - c, preferring a, then b. */
static int
png_filter_paeth(UINT8 *out, const UINT8 *cur, const UINT8 *prev, int n, int bpp) {
    int i, s = 0;
    for (i = 0; i < bpp && i < n; i++) {
        UINT8 v = cur[i] - prev[i];
        out[i] = v;
        s += PNG_COST(v);
    }
#if defined(IMAGING_SIMD_SSE4)
    {
        __m128i acc = _mm_setzero_si128(), zero = _mm_setzero_si128();
        for (; i < n - 15; i += 16) {
            __m128i a8 = _mm_loadu_si128((__m128i *)(cur + i - bpp));
            __m128i b8 = _mm_loadu_si128((__m128i *)(prev + i));
            __m128i c8 = _mm_loadu_si128((__m128i *)(prev + i - bpp));
            __m128i pred[2];
            int h;
            for (h = 0; h < 2; h++) {
                __m128i a =
                    h ? _mm_unpackhi_epi8(a8, zero) : _mm_unpacklo_epi8(a8, zero);
                __m128i b =
                    h ? _mm_unpackhi_epi8(b8, zero) : _mm_unpacklo_epi8(b8, zero);
                __m128i c =
                    h ? _mm_unpackhi_epi8(c8, zero) : _mm_unpacklo_epi8(c8, zero);
                __m128i pa = _mm_abs_epi16(_mm_sub_epi16(b, c));
                __m128i pb = _mm_abs_epi16(_mm_sub_epi16(a, c));
                __m128i pc = _mm_abs_epi16(
                    _mm_sub_epi16(_mm_add_epi16(a, b), _mm_add_epi16(c, c))
                );
                // a if pa <= pb && pa <= pc, else b if pb <= pc, else c
                __m128i use_a = _mm_andnot_si128(
                    _mm_or_si128(_mm_cmpgt_epi16(pa, pb), _mm_cmpgt_epi16(pa, pc)),
                    _mm_cmpeq_epi16(a, a)
                );
                __m128i use_b =
                    _mm_andnot_si128(_mm_cmpgt_epi16(pb, pc), _mm_cmpeq_epi16(a, a));
                __m128i p = _mm_blendv_epi8(c, b, use_b);
                pred[h] = _mm_blendv_epi8(p, a, use_a);
            }
            __m128i v = _mm_sub_epi8(
                _mm_loadu_si128((__m128i *)(cur + i)),
                _mm_packus_epi16(pred[0], pred[1])
            );
            _mm_storeu_si128((__m128i *)(out + i), v);
            acc = _mm_add_epi64(acc, png_cost16(v));
        }
        s += png_sum64(acc);
    }
#elif defined(IMAGING_SIMD_NEON)
    {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i < n - 15; i += 16) {
            uint8x16_t a = vld1q_u8(cur + i - bpp), b = vld1q_u8(prev + i);
            uint8x16_t c = vld1q_u8(prev + i - bpp);
            // |b - c| and |a - c| fit bytes; |a + b - 2c| needs 16 bits
            uint8x16_t pa8 = vabdq_u8(b, c), pb8 = vabdq_u8(a, c);
            int16x8_t pa_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pa8)));
            int16x8_t pa_hi = vreinterpretq_s16_u16(vmovl_high_u8(pa8));
            int16x8_t pb_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pb8)));
            int16x8_t pb_hi = vreinterpretq_s16_u16(vmovl_high_u8(pb8));
            // a + b - 2c = (a - c) + (b - c)
            int16x8_t ac_lo =
                vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(a), vget_low_u8(c)));
            int16x8_t ac_hi = vreinterpretq_s16_u16(vsubl_high_u8(a, c));
            int16x8_t bc_lo =
                vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(b), vget_low_u8(c)));
            int16x8_t bc_hi = vreinterpretq_s16_u16(vsubl_high_u8(b, c));
            int16x8_t pc_lo = vabsq_s16(vaddq_s16(ac_lo, bc_lo));
            int16x8_t pc_hi = vabsq_s16(vaddq_s16(ac_hi, bc_hi));
            uint8x16_t pc_ge_pa = vcombine_u8(
                vmovn_u16(vcgeq_s16(pc_lo, pa_lo)), vmovn_u16(vcgeq_s16(pc_hi, pa_hi))
            );
            uint8x16_t pc_ge_pb = vcombine_u8(
                vmovn_u16(vcgeq_s16(pc_lo, pb_lo)), vmovn_u16(vcgeq_s16(pc_hi, pb_hi))
            );
            uint8x16_t use_a = vandq_u8(vcleq_u8(pa8, pb8), pc_ge_pa);
            uint8x16_t use_b = pc_ge_pb;
            uint8x16_t pred = vbslq_u8(use_a, a, vbslq_u8(use_b, b, c));
            uint8x16_t v = vsubq_u8(vld1q_u8(cur + i), pred);
            vst1q_u8(out + i, v);
            acc = png_cost16(acc, v);
        }
        s += (int)vaddvq_u32(acc);
    }
#endif
    for (; i < n; i++) {
        int a = cur[i - bpp], b = prev[i], c = prev[i - bpp];
        int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
        UINT8 v = cur[i] - ((pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c);
        out[i] = v;
        s += PNG_COST(v);
    }
    return s;
}

#undef PNG_COST

int
ImagingZipEncode(Imaging im, ImagingCodecState state, UINT8 *buf, int bytes) {
    ZIPSTATE *context = (ZIPSTATE *)state->context;
    int err;
    int compress_level, compress_type;
    UINT8 *ptr;
    int i, bpp, s, sum;
    ImagingSectionCookie cookie;

    if (!state->state) {
        /* Initialization */

        /* Valid modes are ZIP_PNG, ZIP_PNG_PALETTE, and ZIP_TIFF */

        /* overflow check for malloc */
        if (state->bytes > INT_MAX - 1) {
            state->errcode = IMAGING_CODEC_MEMORY;
            return -1;
        }

        /* Expand standard buffer to make room for the filter selector,
           and allocate filter buffers */
        free(state->buffer);
        /* malloc check ok, overflow checked above */
        state->buffer = (UINT8 *)malloc(state->bytes + 1);
        context->previous = (UINT8 *)malloc(state->bytes + 1);
        context->prior = (UINT8 *)malloc(state->bytes + 1);
        context->up = (UINT8 *)malloc(state->bytes + 1);
        context->average = (UINT8 *)malloc(state->bytes + 1);
        context->paeth = (UINT8 *)malloc(state->bytes + 1);
        if (!state->buffer || !context->previous || !context->prior || !context->up ||
            !context->average || !context->paeth) {
            free(context->paeth);
            free(context->average);
            free(context->up);
            free(context->prior);
            free(context->previous);
            state->errcode = IMAGING_CODEC_MEMORY;
            return -1;
        }

        /* Initialise filter buffers */
        state->buffer[0] = 0;
        context->prior[0] = 1;
        context->up[0] = 2;
        context->average[0] = 3;
        context->paeth[0] = 4;

        /* Initialise previous buffer to black */
        memset(context->previous, 0, state->bytes + 1);

        /* Setup compression context */
        context->z_stream.zalloc = (alloc_func)0;
        context->z_stream.zfree = (free_func)0;
        context->z_stream.opaque = (voidpf)0;
        context->z_stream.next_in = 0;
        context->z_stream.avail_in = 0;

        compress_level =
            (context->optimize) ? Z_BEST_COMPRESSION : context->compress_level;

        if (context->compress_type == -1) {
            compress_type =
                (context->mode == ZIP_PNG) ? Z_FILTERED : Z_DEFAULT_STRATEGY;
        } else {
            compress_type = context->compress_type;
        }

        err = deflateInit2(
            &context->z_stream,
            /* compression level */
            compress_level,
            /* compression method */
            Z_DEFLATED,
            /* compression memory resources */
            15,
            9,
            /* compression strategy (image data are filtered)*/
            compress_type
        );
        if (err < 0) {
            state->errcode = IMAGING_CODEC_CONFIG;
            return -1;
        }

        if (context->dictionary && context->dictionary_size > 0) {
            err = deflateSetDictionary(
                &context->z_stream,
                (unsigned char *)context->dictionary,
                context->dictionary_size
            );
            if (err < 0) {
                state->errcode = IMAGING_CODEC_CONFIG;
                return -1;
            }
        }

        /* Ready to decode */
        state->state = 1;
    }

    /* Setup the destination buffer */
    context->z_stream.next_out = buf;
    context->z_stream.avail_out = bytes;
    if (context->z_stream.next_in && context->z_stream.avail_in > 0) {
        /* We have some data from previous round, deflate it first */
        err = deflate(&context->z_stream, Z_NO_FLUSH);

        if (err < 0) {
            /* Something went wrong inside the compression library */
            if (err == Z_DATA_ERROR) {
                state->errcode = IMAGING_CODEC_BROKEN;
            } else if (err == Z_MEM_ERROR) {
                state->errcode = IMAGING_CODEC_MEMORY;
            } else {
                state->errcode = IMAGING_CODEC_CONFIG;
            }
            free(context->paeth);
            free(context->average);
            free(context->up);
            free(context->prior);
            free(context->previous);
            deflateEnd(&context->z_stream);
            return -1;
        }
    }

    ImagingSectionEnter(&cookie);
    for (;;) {
        switch (state->state) {
            case 1:

                /* Compress image data */
                while (context->z_stream.avail_out > 0) {
                    if (state->y >= state->ysize) {
                        /* End of image; now flush compressor buffers */
                        state->state = 2;
                        break;
                    }

                    /* Stuff image data into the compressor */
                    state->shuffle(
                        state->buffer + 1,
                        (UINT8 *)im->image[state->y + state->yoff] +
                            state->xoff * im->pixelsize,
                        state->xsize
                    );

                    state->y++;

                    context->output = state->buffer;

                    if (context->mode == ZIP_PNG) {
                        /* Filter the image data.  For each line, select
                           the filter that gives the least total distance
                           from zero for the filtered data (taken from
                           LIBPNG) */

                        bpp = (state->bits + 7) / 8;

                        /* 0. No filter */
                        sum = png_cost_none(state->buffer + 1, state->bytes);

                        /* 2. Up.  We'll test this first to save time when
                           an image line is identical to the one above. */
                        if (sum > 0) {
                            s = png_filter_diff(
                                context->up + 1,
                                state->buffer + 1,
                                context->previous + 1,
                                state->bytes
                            );
                            if (s < sum) {
                                context->output = context->up;
                                sum = s; /* 0 if line was duplicated */
                            }
                        }

                        /* 1. Prior */
                        if (sum > 0) {
                            s = 0;
                            for (i = 1; i <= bpp && i <= state->bytes; i++) {
                                UINT8 v = state->buffer[i];
                                context->prior[i] = v;
                                s += (v < 128) ? v : 256 - v;
                            }
                            if (state->bytes > bpp) {
                                s += png_filter_diff(
                                    context->prior + 1 + bpp,
                                    state->buffer + 1 + bpp,
                                    state->buffer + 1,
                                    state->bytes - bpp
                                );
                            }
                            if (s < sum) {
                                context->output = context->prior;
                                sum = s; /* 0 if line is solid */
                            }
                        }

                        /* 3. Average (not very common in real-life images,
                           so its only used with the optimize option) */
                        if (context->optimize && sum > 0) {
                            s = png_filter_average(
                                context->average + 1,
                                state->buffer + 1,
                                context->previous + 1,
                                state->bytes,
                                bpp
                            );
                            if (s < sum) {
                                context->output = context->average;
                                sum = s;
                            }
                        }

                        /* 4. Paeth */
                        if (sum > 0) {
                            s = png_filter_paeth(
                                context->paeth + 1,
                                state->buffer + 1,
                                context->previous + 1,
                                state->bytes,
                                bpp
                            );
                            if (s < sum) {
                                context->output = context->paeth;
                                sum = s;
                            }
                        }
                    }

                    /* Compress this line */
                    context->z_stream.next_in = context->output;
                    context->z_stream.avail_in = state->bytes + 1;

                    err = deflate(&context->z_stream, Z_NO_FLUSH);

                    if (err < 0) {
                        /* Something went wrong inside the compression library */
                        if (err == Z_DATA_ERROR) {
                            state->errcode = IMAGING_CODEC_BROKEN;
                        } else if (err == Z_MEM_ERROR) {
                            state->errcode = IMAGING_CODEC_MEMORY;
                        } else {
                            state->errcode = IMAGING_CODEC_CONFIG;
                        }
                        free(context->paeth);
                        free(context->average);
                        free(context->up);
                        free(context->prior);
                        free(context->previous);
                        deflateEnd(&context->z_stream);
                        ImagingSectionLeave(&cookie);
                        return -1;
                    }

                    /* Swap buffer pointers */
                    ptr = state->buffer;
                    state->buffer = context->previous;
                    context->previous = ptr;
                }

                if (context->z_stream.avail_out == 0) {
                    break; /* Buffer full */
                }

            case 2:

                /* End of image data; flush compressor buffers */

                while (context->z_stream.avail_out > 0) {
                    err = deflate(&context->z_stream, Z_FINISH);

                    if (err == Z_STREAM_END) {
                        free(context->paeth);
                        free(context->average);
                        free(context->up);
                        free(context->prior);
                        free(context->previous);

                        deflateEnd(&context->z_stream);

                        state->errcode = IMAGING_CODEC_END;

                        break;
                    }

                    if (context->z_stream.avail_out == 0) {
                        break; /* Buffer full */
                    }
                }
        }
        ImagingSectionLeave(&cookie);
        return bytes - context->z_stream.avail_out;
    }

    /* Should never ever arrive here... */
    state->errcode = IMAGING_CODEC_CONFIG;
    ImagingSectionLeave(&cookie);
    return -1;
}

/* -------------------------------------------------------------------- */
/* Cleanup                                                              */
/* -------------------------------------------------------------------- */

void
ImagingZipEncodeCleanup(ImagingCodecState state) {
    ZIPSTATE *context = (ZIPSTATE *)state->context;

    if (context->dictionary) {
        free(context->dictionary);
        context->dictionary = NULL;
    }
}

const char *
ImagingZipVersion(void) {
    return zlibVersion();
}

#endif

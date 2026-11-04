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
#include "ZipEncode.h"

#ifdef HAVE_LIBZ

#include "ZipCodecs.h"

/* PNG filters for one row of n bytes, out[i] = cur[i] - predictor, with
   prev the row above and bpp the bytes per pixel. Each returns the
   filter's cost, the sum over the row of |out[i]| taken as signed bytes,
   min(v, 256 - v), as libpng chooses filters. When encoding, a predictor
   only reads the original rows, so whole vectors of bytes are filtered at
   once; the results and costs are those of the scalar loops. */

static int
png_cost_none(const UINT8 *cur, int n) {
    int s = 0;
    /* the active tier's kernel does the bytes it can (ZipEncodeSIMD.c) */
    int i = IMAGING_SIMD_CALL(zip->cost_none, cur, n, &s);
    for (; i < n; i++) {
        s += PNG_COST(cur[i]);
    }
    return s;
}

/* out[i] = cur[i] - ref[i]: Up with ref the row above, Sub with ref the
   same row bpp bytes back. */
static int
png_filter_diff(UINT8 *out, const UINT8 *cur, const UINT8 *ref, int n) {
    int s = 0;
    int i = IMAGING_SIMD_CALL(zip->filter_diff, out, cur, ref, n, &s);
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
    i += IMAGING_SIMD_CALL(zip->filter_average, out, cur, prev, n, bpp, i, &s);
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
    i += IMAGING_SIMD_CALL(zip->filter_paeth, out, cur, prev, n, bpp, i, &s);
    for (; i < n; i++) {
        int a = cur[i - bpp], b = prev[i], c = prev[i - bpp];
        int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
        UINT8 v = cur[i] - ((pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c);
        out[i] = v;
        s += PNG_COST(v);
    }
    return s;
}

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

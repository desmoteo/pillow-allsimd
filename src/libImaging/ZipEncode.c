/*
 * The Python Imaging Library.
 * $Id$
 *
 * coder for ZIP (deflated) image data
 *
 * History:
 * 96-12-29 fl  created
 * 96-12-30 fl  adaptive filter selection, encoder tuning
 * 2026-11-04   Pillow-AllSIMD: the filtered rows compressed at once with
 *              libdeflate when it is available
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

#ifdef HAVE_LIBDEFLATE
#include <libdeflate.h>

/* libdeflate compresses a whole buffer at once, several times faster than
   zlib's stream at every level and a little smaller, but the encoder then
   holds the filtered image and the output bound in memory where the zlib
   stream holds a few rows. Above this many filtered bytes the stream is
   used. PILLOW_LIBDEFLATE_MAX overrides it (bytes; 0 means zlib always),
   for tests and tuning. */
#define ZIP_LIBDEFLATE_MAX ((size_t)256 << 20)
#endif

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

/* Frees the encoder's own buffers; safe to call more than once. */
static void
zip_free_buffers(ZIPSTATE *context) {
    free(context->paeth);
    free(context->average);
    free(context->up);
    free(context->prior);
    free(context->previous);
    context->paeth = context->average = context->up = context->prior = NULL;
    context->previous = NULL;
#ifdef HAVE_LIBDEFLATE
    free(context->whole);
    free(context->out);
    context->whole = context->out = NULL;
#endif
}

/* Packs the next row into state->buffer + 1 and, for PNG data, chooses its
   filter: for each line, the filter that gives the least total distance
   from zero for the filtered data (taken from LIBPNG). Leaves the row to
   compress, state->bytes + 1 bytes with the filter type first, in
   context->output. */
static void
zip_filter_row(Imaging im, ImagingCodecState state, ZIPSTATE *context) {
    int i, bpp, s, sum;

    state->shuffle(
        state->buffer + 1,
        (UINT8 *)im->image[state->y + state->yoff] + state->xoff * im->pixelsize,
        state->xsize
    );

    state->y++;

    context->output = state->buffer;

    if (context->mode != ZIP_PNG) {
        return;
    }

    bpp = (state->bits + 7) / 8;

    /* 0. No filter */
    sum = png_cost_none(state->buffer + 1, state->bytes);

    /* 2. Up.  We'll test this first to save time when
       an image line is identical to the one above. */
    if (sum > 0) {
        s = png_filter_diff(
            context->up + 1, state->buffer + 1, context->previous + 1, state->bytes
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

/* After a row: the row just filtered becomes the previous one. */
static void
zip_swap_rows(ImagingCodecState state, ZIPSTATE *context) {
    UINT8 *ptr = state->buffer;
    state->buffer = context->previous;
    context->previous = ptr;
}

#ifdef HAVE_LIBDEFLATE

/* The libdeflate path: on the first call, filters every row into
   context->whole and compresses it; each call then copies the next piece
   of the stream into buf, and the last one ends the codec. The GIL is
   released while the rows are filtered and compressed. */
static int
zip_encode_libdeflate(Imaging im, ImagingCodecState state, UINT8 *buf, int bytes) {
    ZIPSTATE *context = (ZIPSTATE *)state->context;
    ImagingSectionCookie cookie;
    size_t n;

    if (state->state == 1) {
        struct libdeflate_compressor *compressor;
        /* compress_level as zlib's 0 to 9; optimize is libdeflate's 10,
           whose files are as small as 12's at 60% of its time, 3 to 9%
           smaller than zlib's 9 and, for RGB and RGBA, 2 to 9x faster */
        int level = context->optimize ? 10 : context->compress_level;
        if (level < 0) {
            level = 6; /* zlib's default */
        } else if (level > 12) {
            level = 12;
        }
        ImagingSectionEnter(&cookie);
        while (state->y < state->ysize) {
            zip_filter_row(im, state, context);
            memcpy(
                context->whole + context->whole_used,
                context->output,
                (size_t)state->bytes + 1
            );
            context->whole_used += (size_t)state->bytes + 1;
            zip_swap_rows(state, context);
        }
        compressor = libdeflate_alloc_compressor(level);
        if (compressor) {
            context->out_size =
                libdeflate_zlib_compress_bound(compressor, context->whole_used);
            context->out = malloc(context->out_size);
        }
        if (!compressor || !context->out) {
            libdeflate_free_compressor(compressor);
            ImagingSectionLeave(&cookie);
            zip_free_buffers(context);
            state->errcode = IMAGING_CODEC_MEMORY;
            return -1;
        }
        n = libdeflate_zlib_compress(
            compressor,
            context->whole,
            context->whole_used,
            context->out,
            context->out_size
        );
        libdeflate_free_compressor(compressor);
        ImagingSectionLeave(&cookie);
        if (n == 0) {
            zip_free_buffers(context);
            state->errcode = IMAGING_CODEC_CONFIG;
            return -1;
        }
        context->out_size = n;
        context->out_pos = 0;
        free(context->whole);
        context->whole = NULL;
        state->state = 2;
    }

    /* Hand the stream out, bytes at a time */
    n = context->out_size - context->out_pos;
    if (n > (size_t)bytes) {
        n = (size_t)bytes;
    }
    memcpy(buf, context->out + context->out_pos, n);
    context->out_pos += n;
    if (context->out_pos == context->out_size) {
        zip_free_buffers(context);
        state->errcode = IMAGING_CODEC_END;
    }
    return (int)n;
}

/* Whether this image goes through libdeflate: not with a dictionary or
   an explicit zlib strategy, which libdeflate has no equivalent of, and
   not above the size cap. Allocates the whole-image buffer. */
static int
zip_libdeflate_setup(ImagingCodecState state, ZIPSTATE *context) {
    size_t max = ZIP_LIBDEFLATE_MAX;
    size_t rowbytes = (size_t)state->bytes + 1;
    const char *env = getenv("PILLOW_LIBDEFLATE_MAX");

    if ((context->dictionary && context->dictionary_size > 0) ||
        context->compress_type != -1 || state->ysize <= 0) {
        return 0;
    }
    if (env && *env) {
        max = (size_t)strtoull(env, NULL, 10);
    }
    if (max == 0 || (size_t)state->ysize > max / rowbytes) {
        return 0;
    }
    context->whole_size = rowbytes * (size_t)state->ysize;
    context->whole_used = 0;
    context->whole = malloc(context->whole_size);
    return context->whole != NULL;
}

const char *
ImagingZipLibdeflateVersion(void) {
    return LIBDEFLATE_VERSION_STRING;
}

#endif

int
ImagingZipEncode(Imaging im, ImagingCodecState state, UINT8 *buf, int bytes) {
    ZIPSTATE *context = (ZIPSTATE *)state->context;
    int err;
    int compress_level, compress_type;
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
            zip_free_buffers(context);
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

#ifdef HAVE_LIBDEFLATE
        context->use_libdeflate = zip_libdeflate_setup(state, context);
        if (context->use_libdeflate) {
            state->state = 1;
            return zip_encode_libdeflate(im, state, buf, bytes);
        }
#endif

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

#ifdef HAVE_LIBDEFLATE
    if (context->use_libdeflate) {
        return zip_encode_libdeflate(im, state, buf, bytes);
    }
#endif

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
            zip_free_buffers(context);
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

                    /* Stuff image data into the compressor, filtered */
                    zip_filter_row(im, state, context);

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
                        zip_free_buffers(context);
                        deflateEnd(&context->z_stream);
                        ImagingSectionLeave(&cookie);
                        return -1;
                    }

                    zip_swap_rows(state, context);
                }

                if (context->z_stream.avail_out == 0) {
                    break; /* Buffer full */
                }

            case 2:

                /* End of image data; flush compressor buffers */

                while (context->z_stream.avail_out > 0) {
                    err = deflate(&context->z_stream, Z_FINISH);

                    if (err == Z_STREAM_END) {
                        zip_free_buffers(context);

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
    /* the buffers of an encoder that was not run to its end */
    zip_free_buffers(context);
}

const char *
ImagingZipVersion(void) {
    return zlibVersion();
}

#endif

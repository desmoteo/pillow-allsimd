/*
 * The Python Imaging Library.
 * $Id$
 *
 * decoder for ZIP (deflated) image data.
 *
 * history:
 * 1996-12-14 fl   Created (for PNG)
 * 1997-01-15 fl   Prepared to read TIFF/ZIP
 * 2001-11-19 fl   PNG incomplete read patch (from Bernhard Herzog)
 *
 * Copyright (c) Fredrik Lundh 1996.
 * Copyright (c) Secret Labs AB 1997-2001.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

#ifdef HAVE_LIBZ

#include "ZipCodecs.h"

#ifdef HAVE_LIBDEFLATE
#include <libdeflate.h>
/* As in ZipEncode.c: the one-piece path holds the compressed stream and
   the decompressed rows in memory; above this many bytes of either, or
   with PILLOW_LIBDEFLATE_MAX below them, zlib's stream is used. */
#define ZIP_LIBDEFLATE_MAX ((size_t)256 << 20)

/* The cap in force. Never above INT_MAX: the gathered stream may have to
   go through zlib, whose counts are 32-bit. */
static size_t
zip_libdeflate_max(void) {
    size_t max = ZIP_LIBDEFLATE_MAX;
    const char *env = getenv("PILLOW_LIBDEFLATE_MAX");
    if (env && *env) {
        max = (size_t)strtoull(env, NULL, 10);
    }
    return max > INT_MAX ? INT_MAX : max;
}
#endif

static const int OFFSET[] = {7, 3, 3, 1, 1, 0, 0};
static const int STARTING_COL[] = {0, 4, 0, 2, 0, 1, 0};
static const int STARTING_ROW[] = {0, 0, 4, 0, 2, 0, 1};
static const int COL_INCREMENT[] = {8, 8, 4, 4, 2, 2, 1};
static const int ROW_INCREMENT[] = {8, 8, 8, 4, 4, 2, 2};

/* Get the length in bytes of a scanline in the pass specified,
 * for interlaced images */
static int
get_row_len(ImagingCodecState state, int pass) {
    int row_len = (state->xsize + OFFSET[pass]) / COL_INCREMENT[pass];
    return ((row_len * state->bits) + 7) / 8;
}

/* -------------------------------------------------------------------- */
/* Decoder                                                              */
/* -------------------------------------------------------------------- */

/* One row, complete in state->buffer (with its filter byte in front for
   PNG): undoes the row's filter with the previous row, stores the pixels
   in the image and moves on to the next row, of the next pass when the
   image is interlaced. *row_len is the row's length and becomes the next
   row's. Returns 1 when the image is complete, 0 when more rows follow, -1
   for an unknown filter (errcode set). The caller swaps the row buffers. */
static int
zip_decode_row(Imaging im, ImagingCodecState state, ZIPSTATE *context, int *row_len_p) {
    int i, bpp;
    int row_len = *row_len_p;

    /* Apply predictor */
    switch (context->mode) {
        case ZIP_PNG:
            switch (state->buffer[0]) {
                case 0:
                    break;
                case 1:
                    /* prior */
                    bpp = (state->bits + 7) / 8;
                    for (i = bpp + 1; i <= row_len; i++) {
                        state->buffer[i] += state->buffer[i - bpp];
                    }
                    break;
                case 2:
                    /* up */
                    /* the active tier's kernel (ZipEncodeSIMD.c) */
                    if (ImagingSIMD->zip->unfilter_up) {
                        ImagingSIMD->zip->unfilter_up(
                            state->buffer + 1, context->previous + 1, row_len
                        );
                    } else {
                        for (i = 1; i <= row_len; i++) {
                            state->buffer[i] += context->previous[i];
                        }
                    }
                    break;
                case 3:
                    /* average */
                    bpp = (state->bits + 7) / 8;
                    for (i = 1; i <= bpp; i++) {
                        state->buffer[i] += context->previous[i] / 2;
                    }
                    for (; i <= row_len; i++) {
                        state->buffer[i] +=
                            (state->buffer[i - bpp] + context->previous[i]) / 2;
                    }
                    break;
                case 4:
                    /* paeth filtering */
                    bpp = (state->bits + 7) / 8;
                    for (i = 1; i <= bpp; i++) {
                        state->buffer[i] += context->previous[i];
                    }
                    for (; i <= row_len; i++) {
                        int a, b, c;
                        int pa, pb, pc;

                        /* fetch pixels */
                        a = state->buffer[i - bpp];
                        b = context->previous[i];
                        c = context->previous[i - bpp];

                        /* distances to surrounding pixels */
                        pa = abs(b - c);
                        pb = abs(a - c);
                        pc = abs(a + b - 2 * c);

                        /* pick predictor with the shortest distance */
                        state->buffer[i] += (pa <= pb && pa <= pc) ? a
                                            : (pb <= pc)           ? b
                                                                   : c;
                    }
                    break;
                default:
                    state->errcode = IMAGING_CODEC_UNKNOWN;
                    return -1;
            }
            break;
        case ZIP_TIFF_PREDICTOR:
            bpp = (state->bits + 7) / 8;
            for (i = bpp + 1; i <= row_len; i++) {
                state->buffer[i] += state->buffer[i - bpp];
            }
            break;
    }

    /* Stuff data into the image */
    if (context->interlaced) {
        int col = STARTING_COL[context->pass];
        if (state->bits >= 8) {
            /* Stuff pixels in their correct location, one by one */
            for (i = 0; i < row_len; i += ((state->bits + 7) / 8)) {
                state->shuffle(
                    (UINT8 *)im->image[state->y] + col * im->pixelsize,
                    state->buffer + context->prefix + i,
                    1
                );
                col += COL_INCREMENT[context->pass];
            }
        } else {
            /* Handle case with more than a pixel in each byte */
            int row_bits = ((state->xsize + OFFSET[context->pass]) /
                            COL_INCREMENT[context->pass]) *
                           state->bits;
            for (i = 0; i < row_bits; i += state->bits) {
                UINT8 byte = *(state->buffer + context->prefix + (i / 8));
                byte <<= (i % 8);
                state->shuffle(
                    (UINT8 *)im->image[state->y] + col * im->pixelsize, &byte, 1
                );
                col += COL_INCREMENT[context->pass];
            }
        }
        /* Find next valid scanline */
        state->y += ROW_INCREMENT[context->pass];
        while (state->y >= state->ysize || row_len <= 0) {
            context->pass++;
            if (context->pass == 7) {
                /* Force exit below */
                state->y = state->ysize;
                break;
            }
            state->y = STARTING_ROW[context->pass];
            row_len = get_row_len(state, context->pass);
            /* Since we're moving to the "first" line, the previous line
             * should be black to make filters work correctly */
            memset(state->buffer, 0, state->bytes + 1);
        }
    } else {
        state->shuffle(
            (UINT8 *)im->image[state->y + state->yoff] + state->xoff * im->pixelsize,
            state->buffer + context->prefix,
            state->xsize
        );
        state->y++;
    }

    *row_len_p = row_len;
    return state->y >= state->ysize;
}

/* The stream decoder: inflates what has arrived a row at a time and
   decodes each row. Returns bytes when all of it was consumed and more is
   needed, -1 at the end of the image (errcode 0) or on an error. */
static int
zip_decode_stream(Imaging im, ImagingCodecState state, UINT8 *buf, Py_ssize_t bytes) {
    ZIPSTATE *context = (ZIPSTATE *)state->context;
    int err;
    int n;
    UINT8 *ptr;
    int row_len;

    if (context->interlaced) {
        row_len = get_row_len(state, context->pass);
    } else {
        row_len = state->bytes;
    }

    /* Setup the source buffer */
    context->z_stream.next_in = buf;
    context->z_stream.avail_in = bytes;

    /* Decompress what we've got this far */
    while (context->z_stream.avail_in > 0) {
        context->z_stream.next_out = state->buffer + context->last_output;
        context->z_stream.avail_out = row_len + context->prefix - context->last_output;

        err = inflate(&context->z_stream, Z_NO_FLUSH);

        if (err < 0) {
            /* Something went wrong inside the compression library */
            if (err == Z_DATA_ERROR) {
                state->errcode = IMAGING_CODEC_BROKEN;
            } else if (err == Z_MEM_ERROR) {
                state->errcode = IMAGING_CODEC_MEMORY;
            } else {
                state->errcode = IMAGING_CODEC_CONFIG;
            }
            free(context->previous);
            context->previous = NULL;
            inflateEnd(&context->z_stream);
            return -1;
        }

        n = row_len + context->prefix - context->z_stream.avail_out;

        if (n < row_len + context->prefix) {
            context->last_output = n;
            break; /* need more input data */
        }

        n = zip_decode_row(im, state, context, &row_len);
        if (n < 0) {
            free(context->previous);
            context->previous = NULL;
            inflateEnd(&context->z_stream);
            return -1;
        }

        /* all inflate output has been consumed */
        context->last_output = 0;

        if (n > 0 || err == Z_STREAM_END) {
            /* The image and the data should end simultaneously */
            /* if (state->y < state->ysize || err != Z_STREAM_END)
                state->errcode = IMAGING_CODEC_BROKEN; */

            free(context->previous);
            context->previous = NULL;
            inflateEnd(&context->z_stream);
            return -1; /* end of file (errcode=0) */
        }

        /* Swap buffer pointers */
        ptr = state->buffer;
        state->buffer = context->previous;
        context->previous = ptr;
    }

    return bytes; /* consumed all of it */
}

#ifdef HAVE_LIBDEFLATE

/* The bytes the rows of the image decompress to, filter bytes included,
   walking the passes of an interlaced image as zip_decode_row() does. */
static size_t
zip_expected_size(ImagingCodecState state, ZIPSTATE *context) {
    size_t total = 0;
    int pass = 0, y, row_len;
    if (!context->interlaced) {
        return ((size_t)state->bytes + context->prefix) * (size_t)state->ysize;
    }
    y = STARTING_ROW[0];
    row_len = get_row_len(state, 0);
    for (;;) {
        total += (size_t)row_len + context->prefix;
        y += ROW_INCREMENT[pass];
        while (y >= state->ysize || row_len <= 0) {
            pass++;
            if (pass == 7) {
                return total;
            }
            y = STARTING_ROW[pass];
            row_len = get_row_len(state, pass);
        }
    }
}

/* The whole compressed stream at once (context->gathered): libdeflate
   inflates it two to three times faster than zlib's stream does, but takes
   and gives whole buffers. Decodes the image and returns 1 only if the
   stream is well formed and holds exactly the rows of the image; returns 0
   having touched nothing otherwise, and the stream decoder then gives the
   file the errors and the partial image it always gave. */
static int
zip_decode_whole(Imaging im, ImagingCodecState state, ZIPSTATE *context) {
    size_t expected, actual_in, actual_out, pos = 0;
    struct libdeflate_decompressor *decompressor;
    enum libdeflate_result result;
    UINT8 *out, *ptr;
    int row_len, n;

    if (state->ysize <= 0 || state->xsize <= 0) {
        return 0;
    }
    expected = zip_expected_size(state, context);
    if (expected == 0 || expected > zip_libdeflate_max()) {
        return 0;
    }
    out = malloc(expected);
    decompressor = libdeflate_alloc_decompressor();
    if (!out || !decompressor) {
        free(out);
        libdeflate_free_decompressor(decompressor);
        return 0;
    }
    /* mapped afresh at this size: fewer page faults (Storage.c) */
    ImagingMemoryAdviseHugePages(out, expected);
    result = libdeflate_zlib_decompress_ex(
        decompressor,
        context->gathered,
        context->gathered_len,
        out,
        expected,
        &actual_in,
        &actual_out
    );
    libdeflate_free_decompressor(decompressor);
    if (result != LIBDEFLATE_SUCCESS || actual_out != expected) {
        free(out);
        return 0;
    }

    if (!context->interlaced) {
        /* The rows are unfiltered where libdeflate left them, each with
           the one before it as its previous row: no copy, and the previous
           row is still in the cache. The first row's is the black one. */
        UINT8 *buffer = state->buffer, *previous = context->previous;
        size_t need = (size_t)state->bytes + context->prefix;
        row_len = state->bytes;
        for (pos = 0; pos + need <= expected; pos += need) {
            state->buffer = out + pos;
            n = zip_decode_row(im, state, context, &row_len);
            if (n) {
                break; /* the image is complete, or an unknown filter */
            }
            context->previous = out + pos;
        }
        state->buffer = buffer;
        context->previous = previous;
        free(out);
        return 1;
    }

    row_len = get_row_len(state, context->pass);
    for (;;) {
        size_t need = (size_t)row_len + context->prefix;
        if (need > expected - pos) {
            /* cannot happen: zip_expected_size() walks the same rows */
            state->errcode = IMAGING_CODEC_BROKEN;
            break;
        }
        memcpy(state->buffer, out + pos, need);
        pos += need;
        n = zip_decode_row(im, state, context, &row_len);
        if (n) {
            break; /* the image is complete, or a row had an unknown filter */
        }
        ptr = state->buffer;
        state->buffer = context->previous;
        context->previous = ptr;
    }
    free(out);
    return 1;
}

#endif

int
ImagingZipDecode(Imaging im, ImagingCodecState state, UINT8 *buf, Py_ssize_t bytes) {
    ZIPSTATE *context = (ZIPSTATE *)state->context;
    int err;

    if (!state->state) {
        /* Initialization */
        if (context->mode == ZIP_PNG || context->mode == ZIP_PNG_PALETTE) {
            context->prefix = 1; /* PNG */
        }

        /* overflow check for malloc */
        if (state->bytes > INT_MAX - 1) {
            state->errcode = IMAGING_CODEC_MEMORY;
            return -1;
        }
        /* Expand standard buffer to make room for the (optional) filter
           prefix, and allocate a buffer to hold the previous line */
        free(state->buffer);
        /* malloc check ok, overflow checked above */
        state->buffer = (UINT8 *)malloc(state->bytes + 1);
        context->previous = (UINT8 *)malloc(state->bytes + 1);
        if (!state->buffer || !context->previous) {
            state->errcode = IMAGING_CODEC_MEMORY;
            return -1;
        }

        context->last_output = 0;

        /* Initialize to black */
        memset(context->previous, 0, state->bytes + 1);

        /* Setup decompression context */
        context->z_stream.zalloc = (alloc_func)NULL;
        context->z_stream.zfree = (free_func)NULL;
        context->z_stream.opaque = (voidpf)NULL;

        err = inflateInit(&context->z_stream);
        if (err < 0) {
            state->errcode = IMAGING_CODEC_CONFIG;
            free(context->previous);
            context->previous = NULL;
            return -1;
        }

        if (context->interlaced) {
            context->pass = 0;
            state->y = STARTING_ROW[context->pass];
        }

#ifdef HAVE_LIBDEFLATE
        /* The caller knows how long the compressed stream is: gather it,
           for zip_decode_whole(). Not above the cap that the encoder has,
           and not without the memory; the stream decoder otherwise. */
        context->gathered = NULL;
        context->gathered_len = 0;
        if (context->gather_size > 0) {
            if (context->gather_size <= zip_libdeflate_max()) {
                context->gathered = malloc(context->gather_size);
            }
            if (!context->gathered) {
                context->gather_size = 0;
            } else {
                ImagingMemoryAdviseHugePages(context->gathered, context->gather_size);
            }
        }
#endif

        /* Ready to decode */
        state->state = 1;
    }

#ifdef HAVE_LIBDEFLATE
    if (context->gathered) {
        UINT8 *gathered = context->gathered;
        size_t have = context->gathered_len;
        int n;
        if ((size_t)bytes <= context->gather_size - have) {
            memcpy(gathered + have, buf, bytes);
            have += bytes;
            context->gathered_len = have;
            if (have < context->gather_size) {
                return bytes; /* consumed all of it, more to come */
            }
            /* the whole stream is here */
            if (zip_decode_whole(im, state, context)) {
                context->gathered = NULL;
                free(gathered);
                free(context->previous);
                context->previous = NULL;
                inflateEnd(&context->z_stream);
                return -1; /* end of file (errcode=0), or a bad row */
            }
            /* not a stream for the fast path: all of it through the stream
               decoder, as if it had arrived in one piece */
            context->gathered = NULL;
            n = zip_decode_stream(im, state, gathered, (Py_ssize_t)have);
            free(gathered);
            return n < 0 ? -1 : bytes;
        }
        /* more data than announced: what was gathered, then this, through
           the stream decoder */
        context->gathered = NULL;
        n = have ? zip_decode_stream(im, state, gathered, (Py_ssize_t)have) : 0;
        free(gathered);
        if (n < 0) {
            return -1;
        }
    }
#endif

    return zip_decode_stream(im, state, buf, bytes);
}

void
ImagingZipDecodeCleanup(ImagingCodecState state) {
    /* called to free the decompression engine when the decode terminates
       due to a corrupt or truncated image
    */
    ZIPSTATE *context = (ZIPSTATE *)state->context;

    /* Clean up */
    if (context->previous) {
        inflateEnd(&context->z_stream);
        free(context->previous);
        context->previous = NULL;
    }
#ifdef HAVE_LIBDEFLATE
    free(context->gathered);
    context->gathered = NULL;
#endif
}

#endif

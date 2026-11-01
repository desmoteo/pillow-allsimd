/*
 * The Python Imaging Library.
 * $Id$
 *
 * declarations for the ZIP codecs
 *
 * Copyright (c) Fredrik Lundh 1996.
 */

#include <zlib.h>

/* modes */
#define ZIP_PNG 0            /* continuous, filtered image data */
#define ZIP_PNG_PALETTE 1    /* non-continuous data, disable filtering */
#define ZIP_TIFF_PREDICTOR 2 /* TIFF, with predictor */
#define ZIP_TIFF 3           /* TIFF, without predictor */

typedef struct {
    /* CONFIGURATION */

    /* Codec mode */
    int mode;

    /* Optimize (max compression) SLOW!!! */
    int optimize;

    /* 0 no compression, 9 best compression, -1 default compression */
    int compress_level;
    /* compression strategy Z_XXX */
    int compress_type;

    /* Predefined dictionary (experimental) */
    char *dictionary;
    int dictionary_size;

    /* PRIVATE CONTEXT (set by decoder/encoder) */

    z_stream z_stream; /* (de)compression stream */

    UINT8 *previous; /* previous line (allocated) */

    int last_output; /* # bytes last output by inflate */

    /* Compressor specific stuff */
    UINT8 *prior; /* filter storage (allocated) */
    UINT8 *up;
    UINT8 *average;
    UINT8 *paeth;

    UINT8 *output; /* output data */

    int prefix; /* size of filter prefix (0 for TIFF data) */

    int interlaced; /* is the image interlaced? (PNG) */

    int pass; /* current pass of the interlaced image (PNG) */

    /* Pillow-AllSIMD: with libdeflate, the encoder keeps the filtered rows
       of the whole image and compresses them at once after the last one,
       then hands the result out (ZipEncode.c) */
    int use_libdeflate;
    UINT8 *whole;      /* the filtered rows (allocated) */
    size_t whole_size; /* its capacity, rows times bytes + 1 */
    size_t whole_used; /* bytes filled so far */
    UINT8 *out;        /* the compressed stream (allocated) */
    size_t out_size;   /* its length */
    size_t out_pos;    /* bytes handed out so far */

    /* and the decoder, told the length of the compressed stream, gathers
       it and decompresses the image at once (ZipDecode.c) */
    size_t gather_size;  /* the stream's length, 0 if not known */
    UINT8 *gathered;     /* the stream so far (allocated) */
    size_t gathered_len; /* bytes gathered */

} ZIPSTATE;

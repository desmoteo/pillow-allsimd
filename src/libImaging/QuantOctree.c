/* Copyright (c) 2010 Oliver Tonnhofer <olt@bogosoft.com>, Omniscale
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
*/

/*
// This file implements a variation of the octree color quantization algorithm.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "Imaging.h"
#include "ImagingUtils.h"
#include "PlainLoops.h"
#include "QuantOctree.h"

/* a bucket's count is its palette index when the cube is used for look up */
typedef ImagingOctreeBucket *ColorBucket;

typedef struct _ColorCube {
    unsigned int rBits, gBits, bBits, aBits;
    unsigned int rWidth, gWidth, bWidth, aWidth;
    unsigned int rOffset, gOffset, bOffset, aOffset;

    unsigned long size;
    ColorBucket buckets;
} *ColorCube;

static ColorCube
new_color_cube(int r, int g, int b, int a) {
    ColorCube cube;

    /* malloc check ok, small constant allocation */
    cube = malloc(sizeof(struct _ColorCube));
    if (!cube) {
        return NULL;
    }

    cube->rBits = MAX(r, 0);
    cube->gBits = MAX(g, 0);
    cube->bBits = MAX(b, 0);
    cube->aBits = MAX(a, 0);

    /* overflow check for size multiplication below */
    if (cube->rBits + cube->gBits + cube->bBits + cube->aBits > 31) {
        free(cube);
        return NULL;
    }

    /* the width of the cube for each dimension */
    cube->rWidth = 1 << cube->rBits;
    cube->gWidth = 1 << cube->gBits;
    cube->bWidth = 1 << cube->bBits;
    cube->aWidth = 1 << cube->aBits;

    /* the offsets of each color */

    cube->rOffset = cube->gBits + cube->bBits + cube->aBits;
    cube->gOffset = cube->bBits + cube->aBits;
    cube->bOffset = cube->aBits;
    cube->aOffset = 0;

    /* the number of color buckets */
    cube->size = (unsigned long)cube->rWidth * cube->gWidth *
                 (unsigned long)cube->bWidth * cube->aWidth;
    /* malloc check ok, overflow checked above */
    cube->buckets = calloc(cube->size, sizeof(ImagingOctreeBucket));

    if (!cube->buckets) {
        free(cube);
        return NULL;
    }
    return cube;
}

static void
free_color_cube(ColorCube cube) {
    if (cube != NULL) {
        free(cube->buckets);
        free(cube);
    }
}

static long
color_bucket_offset_pos(
    const ColorCube cube, unsigned int r, unsigned int g, unsigned int b, unsigned int a
) {
    return r << cube->rOffset | g << cube->gOffset | b << cube->bOffset |
           a << cube->aOffset;
}

static long
color_bucket_offset(const ColorCube cube, const Pixel *p) {
    unsigned int r = p->c.r >> (8 - cube->rBits);
    unsigned int g = p->c.g >> (8 - cube->gBits);
    unsigned int b = p->c.b >> (8 - cube->bBits);
    unsigned int a = p->c.a >> (8 - cube->aBits);
    return color_bucket_offset_pos(cube, r, g, b, a);
}

static ColorBucket
color_bucket_from_cube(const ColorCube cube, const Pixel *p) {
    unsigned int offset = color_bucket_offset(cube, p);
    return &cube->buckets[offset];
}

/* Pillow-AllSIMD: the cube as the loops over the pixels take it
   (PlainLoops.h): the shifts and offsets, without the pointer through
   which the compiler would reload them for every pixel. */
static ImagingOctreeIndex
cube_index(const ColorCube cube) {
    ImagingOctreeIndex index = {
        {8 - cube->rBits, 8 - cube->gBits, 8 - cube->bBits, 8 - cube->aBits},
        {cube->rOffset, cube->gOffset, cube->bOffset, cube->aOffset},
        cube->buckets,
        NULL
    };
    return index;
}

/* Moves the sums that plain_octree_count_rgb() keeps, r g b and the count
   of each bucket, to the cube, and clears them. The fourth byte of an RGB
   pixel counts as 255, so its sum is 255 times the count. */
static void
add_sums_to_color_cube(const ColorCube cube, uint32_t *sums) {
    unsigned long i;
    for (i = 0; i < cube->size; i++) {
        ColorBucket bucket = &cube->buckets[i];
        bucket->count += sums[i * 4 + 3];
        bucket->r += sums[i * 4];
        bucket->g += sums[i * 4 + 1];
        bucket->b += sums[i * 4 + 2];
        bucket->a += (uint64_t)255 * sums[i * 4 + 3];
    }
    memset(sums, 0, cube->size * 4 * sizeof(uint32_t));
}

/* Adds n pixels of four bytes, r g b a, to the cube: the active tier's
   kernel, which is the loop of PlainLoops.h compiled for it (QuantSIMD.c),
   or that loop as compiled here. */
static void
add_colors_to_color_cube(
    const ColorCube cube, const UINT8 *pixels, int n, unsigned int alpha_or
) {
    ImagingOctreeIndex index = cube_index(cube);
    if (!IMAGING_SIMD_CALL(quant->octree_count, &index, pixels, n, alpha_or)) {
        plain_octree_count(&index, pixels, n, alpha_or);
    }
}

static unsigned long
count_used_color_buckets(const ColorCube cube) {
    unsigned long usedBuckets = 0;
    unsigned long i;
    for (i = 0; i < cube->size; i++) {
        if (cube->buckets[i].count > 0) {
            usedBuckets += 1;
        }
    }
    return usedBuckets;
}

static void
avg_color_from_color_bucket(const ColorBucket bucket, Pixel *dst) {
    float count = bucket->count;
    if (count != 0) {
        dst->c.r = CLIP8((int)(bucket->r / count));
        dst->c.g = CLIP8((int)(bucket->g / count));
        dst->c.b = CLIP8((int)(bucket->b / count));
        dst->c.a = CLIP8((int)(bucket->a / count));
    } else {
        dst->c.r = 0;
        dst->c.g = 0;
        dst->c.b = 0;
        dst->c.a = 0;
    }
}

static int
compare_bucket_count(const ColorBucket a, const ColorBucket b) {
    return b->count - a->count;
}

static ColorBucket
create_sorted_color_palette(const ColorCube cube) {
    ColorBucket buckets;
    if (cube->size > LONG_MAX / sizeof(ImagingOctreeBucket)) {
        return NULL;
    }
    /* malloc check ok, calloc + overflow check above for memcpy */
    buckets = calloc(cube->size, sizeof(ImagingOctreeBucket));
    if (!buckets) {
        return NULL;
    }
    memcpy(buckets, cube->buckets, sizeof(ImagingOctreeBucket) * cube->size);

    qsort(
        buckets,
        cube->size,
        sizeof(ImagingOctreeBucket),
        (int (*)(void const *, void const *))&compare_bucket_count
    );

    return buckets;
}

void
add_bucket_values(ColorBucket src, ColorBucket dst) {
    dst->count += src->count;
    dst->r += src->r;
    dst->g += src->g;
    dst->b += src->b;
    dst->a += src->a;
}

/* expand or shrink a given cube to level */
static ColorCube
copy_color_cube(
    const ColorCube cube,
    unsigned int rBits,
    unsigned int gBits,
    unsigned int bBits,
    unsigned int aBits
) {
    unsigned int r, g, b, a;
    long src_pos, dst_pos;
    unsigned int src_reduce[4] = {0}, dst_reduce[4] = {0};
    unsigned int width[4];
    ColorCube result;

    result = new_color_cube(rBits, gBits, bBits, aBits);
    if (!result) {
        return NULL;
    }

    if (cube->rBits > rBits) {
        dst_reduce[0] = cube->rBits - result->rBits;
        width[0] = cube->rWidth;
    } else {
        src_reduce[0] = result->rBits - cube->rBits;
        width[0] = result->rWidth;
    }
    if (cube->gBits > gBits) {
        dst_reduce[1] = cube->gBits - result->gBits;
        width[1] = cube->gWidth;
    } else {
        src_reduce[1] = result->gBits - cube->gBits;
        width[1] = result->gWidth;
    }
    if (cube->bBits > bBits) {
        dst_reduce[2] = cube->bBits - result->bBits;
        width[2] = cube->bWidth;
    } else {
        src_reduce[2] = result->bBits - cube->bBits;
        width[2] = result->bWidth;
    }
    if (cube->aBits > aBits) {
        dst_reduce[3] = cube->aBits - result->aBits;
        width[3] = cube->aWidth;
    } else {
        src_reduce[3] = result->aBits - cube->aBits;
        width[3] = result->aWidth;
    }

    for (r = 0; r < width[0]; r++) {
        for (g = 0; g < width[1]; g++) {
            for (b = 0; b < width[2]; b++) {
                for (a = 0; a < width[3]; a++) {
                    src_pos = color_bucket_offset_pos(
                        cube,
                        r >> src_reduce[0],
                        g >> src_reduce[1],
                        b >> src_reduce[2],
                        a >> src_reduce[3]
                    );
                    dst_pos = color_bucket_offset_pos(
                        result,
                        r >> dst_reduce[0],
                        g >> dst_reduce[1],
                        b >> dst_reduce[2],
                        a >> dst_reduce[3]
                    );
                    add_bucket_values(
                        &cube->buckets[src_pos], &result->buckets[dst_pos]
                    );
                }
            }
        }
    }
    return result;
}

void
subtract_color_buckets(ColorCube cube, ColorBucket buckets, long nBuckets) {
    ColorBucket minuend, subtrahend;
    long i;
    Pixel p;
    for (i = 0; i < nBuckets; i++) {
        subtrahend = &buckets[i];

        // If the subtrahend contains no buckets, there is nothing to subtract.
        if (subtrahend->count == 0) {
            continue;
        }

        avg_color_from_color_bucket(subtrahend, &p);
        minuend = color_bucket_from_cube(cube, &p);
        minuend->count -= subtrahend->count;
        minuend->r -= subtrahend->r;
        minuend->g -= subtrahend->g;
        minuend->b -= subtrahend->b;
        minuend->a -= subtrahend->a;
    }
}

static void
set_lookup_value(const ColorCube cube, const Pixel *p, long value) {
    ColorBucket bucket = color_bucket_from_cube(cube, p);
    bucket->count = value;
}

uint64_t
lookup_color(const ColorCube cube, const Pixel *p) {
    ColorBucket bucket = color_bucket_from_cube(cube, p);
    return bucket->count;
}

void
add_lookup_buckets(ColorCube cube, ColorBucket palette, long nColors, long offset) {
    long i;
    Pixel p;
    for (i = offset + nColors - 1; i >= offset; i--) {
        avg_color_from_color_bucket(&palette[i], &p);
        set_lookup_value(cube, &p, i);
    }
}

ColorBucket
combined_palette(
    ColorBucket bucketsA,
    unsigned long nBucketsA,
    ColorBucket bucketsB,
    unsigned long nBucketsB
) {
    ColorBucket result;
    if (nBucketsA > LONG_MAX - nBucketsB ||
        (nBucketsA + nBucketsB) > LONG_MAX / sizeof(ImagingOctreeBucket)) {
        return NULL;
    }
    /* malloc check ok, overflow check above */
    result = calloc(nBucketsA + nBucketsB, sizeof(ImagingOctreeBucket));
    if (!result) {
        return NULL;
    }
    memcpy(result, bucketsA, sizeof(ImagingOctreeBucket) * nBucketsA);
    memcpy(&result[nBucketsA], bucketsB, sizeof(ImagingOctreeBucket) * nBucketsB);
    return result;
}

static Pixel *
create_palette_array(const ColorBucket palette, unsigned int paletteLength) {
    Pixel *paletteArray;
    unsigned int i;

    /* malloc check ok, calloc for overflow */
    paletteArray = calloc(paletteLength, sizeof(Pixel));
    if (!paletteArray) {
        return NULL;
    }

    for (i = 0; i < paletteLength; i++) {
        avg_color_from_color_bucket(&palette[i], &paletteArray[i]);
    }
    return paletteArray;
}

static void
map_image_pixels(
    const Pixel *pixelData,
    uint32_t nPixels,
    const ColorCube lookupCube,
    uint32_t *pixelArray
) {
    const ImagingOctreeIndex index = cube_index(lookupCube);
    PLAIN_OCTREE_LOCALS(&index);
    uint32_t i;
    for (i = 0; i < nPixels; i++) {
        const unsigned int r = pixelData[i].c.r, g = pixelData[i].c.g;
        const unsigned int b = pixelData[i].c.b, a = pixelData[i].c.a;
        pixelArray[i] = buckets[PLAIN_OCTREE_BUCKET(r, g, b, a)].count;
    }
}

/* PlainLoops.h has an RGB image's fine cube, four bits a channel, as
   constants (PLAIN_OCTREE_RGB_BUCKET) */
const unsigned int CUBE_LEVELS[8] = {
    PLAIN_OCTREE_RGB_BITS, PLAIN_OCTREE_RGB_BITS, PLAIN_OCTREE_RGB_BITS, 0, 2, 2, 2, 0
};
const unsigned int CUBE_LEVELS_ALPHA[8] = {3, 4, 3, 3, 2, 2, 2, 2};

/* From the fine cube with every pixel counted to the palette's buckets
   (nQuantPixels of them) and the cube that gives a pixel its palette
   index: everything between the two passes over the pixels. Returns 0 if
   out of memory. */
static int
octree_build(
    ColorCube fineCube,
    uint32_t nQuantPixels,
    const unsigned int *cubeBits,
    ColorBucket *paletteBucketsOut,
    ColorCube *lookupCubeOut
) {
    ColorCube coarseCube = NULL;
    ColorCube lookupCube = NULL;
    ColorCube coarseLookupCube = NULL;
    ColorBucket paletteBucketsCoarse = NULL;
    ColorBucket paletteBucketsFine = NULL;
    ColorBucket paletteBuckets = NULL;
    unsigned long nCoarseColors, nFineColors, nAlreadySubtracted;

    /*
    Two color cubes, one fine grained with 8x16x8=1024
    colors buckets and a coarse with 4x4x4=64 color buckets.
    The coarse one guarantees that there are color buckets available for
    the whole color range (assuming nQuantPixels > 64).

    For a quantization to 256 colors all 64 coarse colors will be used
    plus the 192 most used color buckets from the fine color cube.
    The average of all colors within one bucket is used as the actual
    color for that bucket.

     For images with alpha the cubes gets a forth dimension,
     8x16x8x8 and 4x4x4x4.
    */

    /* create coarse cube */
    coarseCube =
        copy_color_cube(fineCube, cubeBits[4], cubeBits[5], cubeBits[6], cubeBits[7]);
    if (!coarseCube) {
        goto error;
    }
    nCoarseColors = count_used_color_buckets(coarseCube);

    /* limit to nQuantPixels */
    if (nCoarseColors > nQuantPixels) {
        nCoarseColors = nQuantPixels;
    }

    /* how many space do we have in our palette for fine colors? */
    nFineColors = nQuantPixels - nCoarseColors;

    /* create fine color palette */
    paletteBucketsFine = create_sorted_color_palette(fineCube);
    if (!paletteBucketsFine) {
        goto error;
    }

    /* remove the used fine colors from the coarse cube */
    subtract_color_buckets(coarseCube, paletteBucketsFine, nFineColors);

    /* did the subtraction cleared one or more coarse bucket? */
    while (nCoarseColors > count_used_color_buckets(coarseCube)) {
        /* then we can use the free buckets for fine colors */
        nAlreadySubtracted = nFineColors;
        nCoarseColors = count_used_color_buckets(coarseCube);
        nFineColors = nQuantPixels - nCoarseColors;
        subtract_color_buckets(
            coarseCube,
            &paletteBucketsFine[nAlreadySubtracted],
            nFineColors - nAlreadySubtracted
        );
    }

    /* create our palette buckets with fine and coarse combined */
    paletteBucketsCoarse = create_sorted_color_palette(coarseCube);
    if (!paletteBucketsCoarse) {
        goto error;
    }
    paletteBuckets = combined_palette(
        paletteBucketsCoarse, nCoarseColors, paletteBucketsFine, nFineColors
    );

    free(paletteBucketsFine);
    paletteBucketsFine = NULL;
    free(paletteBucketsCoarse);
    paletteBucketsCoarse = NULL;
    if (!paletteBuckets) {
        goto error;
    }

    /* add all coarse colors to our coarse lookup cube. */
    coarseLookupCube =
        new_color_cube(cubeBits[4], cubeBits[5], cubeBits[6], cubeBits[7]);
    if (!coarseLookupCube) {
        goto error;
    }
    add_lookup_buckets(coarseLookupCube, paletteBuckets, nCoarseColors, 0);

    /* expand coarse cube (64) to larger fine cube (4k). the value of each
       coarse bucket is then present in the according 64 fine buckets. */
    lookupCube = copy_color_cube(
        coarseLookupCube, cubeBits[0], cubeBits[1], cubeBits[2], cubeBits[3]
    );
    if (!lookupCube) {
        goto error;
    }

    /* add fine colors to the lookup cube */
    add_lookup_buckets(lookupCube, paletteBuckets, nFineColors, nCoarseColors);

    free_color_cube(coarseCube);
    free_color_cube(coarseLookupCube);
    *paletteBucketsOut = paletteBuckets;
    *lookupCubeOut = lookupCube;
    return 1;

error:
    /* everything is initialized to NULL
       so we are safe to call free */
    free_color_cube(lookupCube);
    free_color_cube(coarseLookupCube);
    free(paletteBuckets);
    free(paletteBucketsCoarse);
    free(paletteBucketsFine);
    free_color_cube(coarseCube);
    return 0;
}

int
quantize_octree(
    Pixel *pixelData,
    uint32_t nPixels,
    uint32_t nQuantPixels,
    Pixel **palette,
    uint32_t *paletteLength,
    uint32_t **quantizedPixels,
    int withAlpha
) {
    ColorCube fineCube = NULL;
    ColorCube lookupCube = NULL;
    ColorBucket paletteBuckets = NULL;
    uint32_t *qp = NULL;
    const unsigned int *cubeBits = withAlpha ? CUBE_LEVELS_ALPHA : CUBE_LEVELS;

    /* create fine cube */
    fineCube = new_color_cube(cubeBits[0], cubeBits[1], cubeBits[2], cubeBits[3]);
    if (!fineCube) {
        goto error;
    }
    /* a Pixel is its four bytes, r g b a; the caller checked that the
       count fits an int */
    add_colors_to_color_cube(fineCube, (const UINT8 *)pixelData, (int)nPixels, 0);

    if (!octree_build(fineCube, nQuantPixels, cubeBits, &paletteBuckets, &lookupCube)) {
        goto error;
    }

    /* create result pixels and map palette indices */
    /* malloc check ok, calloc for overflow */
    qp = calloc(nPixels, sizeof(Pixel));
    if (!qp) {
        goto error;
    }
    ImagingMemoryAdviseHugePages(qp, (size_t)nPixels * sizeof(Pixel));
    map_image_pixels(pixelData, nPixels, lookupCube, qp);

    /* convert palette buckets to RGB pixel palette */
    *palette = create_palette_array(paletteBuckets, nQuantPixels);
    if (!(*palette)) {
        goto error;
    }

    *quantizedPixels = qp;
    *paletteLength = nQuantPixels;

    free_color_cube(fineCube);
    free_color_cube(lookupCube);
    free(paletteBuckets);
    return 1;

error:
    /* everything is initialized to NULL
       so we are safe to call free */
    free(qp);
    free_color_cube(lookupCube);
    free(paletteBuckets);
    free_color_cube(fineCube);
    return 0;
}

/* Pillow-AllSIMD: the same quantization from the rows of an RGB or RGBA
   image (ysize rows of xsize pixels of four bytes) into the rows of a P
   image, a byte per pixel, without the copy of the pixels and the array
   of indices that the other quantizers' interface needs: 16 MB each for
   a 2560x1600 image, allocated, filled and freed on every call. The
   palette and the indices are those of quantize_octree() for the same
   pixels.

   That function is given the pixels with every fully transparent one
   recoloured as the first of them (Quant.c); here the rows are read as
   they are, so both passes do the same: the first pass remembers the
   colour, and the second uses it. */
int
quantize_octree_rows(
    UINT8 **rows,
    int xsize,
    int ysize,
    uint32_t nQuantPixels,
    Pixel **palette,
    uint32_t *paletteLength,
    UINT8 **outRows,
    int withAlpha
) {
    ColorCube fineCube = NULL;
    ColorCube lookupCube = NULL;
    ColorBucket paletteBuckets = NULL;
    uint32_t *sums = NULL;
    uint32_t *lut = NULL;
    const unsigned int *cubeBits = withAlpha ? CUBE_LEVELS_ALPHA : CUBE_LEVELS;
    int transparency = 0;
    unsigned int tr = 0, tg = 0, tb = 0;
    int x, y;
    unsigned long i;

    fineCube = new_color_cube(cubeBits[0], cubeBits[1], cubeBits[2], cubeBits[3]);
    if (!fineCube) {
        goto error;
    }
    if (!withAlpha) {
        /* An RGB image: each bucket's sums of r, g and b and its count in
           four 32-bit numbers, which one vector addition per pixel keeps
           (PlainLoops.h, QuantSIMD.c), moved to the cube's buckets before
           they could overflow. */
        int pending = 0;
        /* malloc check ok, the cube's size is a small constant */
        sums = calloc(fineCube->size, 4 * sizeof(uint32_t));
        if (!sums) {
            goto error;
        }
        for (y = 0; y < ysize; y++) {
            int done = 0;
            while (done < xsize) {
                const UINT8 *in = rows[y] + (size_t)done * 4;
                int n = xsize - done;
                if (n > PLAIN_OCTREE_FLUSH - pending) {
                    n = PLAIN_OCTREE_FLUSH - pending;
                }
                if (!IMAGING_SIMD_CALL(quant->octree_count_rgb, in, n, sums)) {
                    plain_octree_count_rgb(in, n, sums);
                }
                done += n;
                pending += n;
                if (pending == PLAIN_OCTREE_FLUSH) {
                    add_sums_to_color_cube(fineCube, sums);
                    pending = 0;
                }
            }
        }
        add_sums_to_color_cube(fineCube, sums);
        free(sums);
        sums = NULL;
    }
    for (y = 0; withAlpha && y < ysize; y++) {
        /* An RGBA image: a row of opaque and translucent pixels goes as it
           is; the pixels from the first transparent one on, one by one. */
        const UINT8 *in = rows[y];
        int first = 0;
        while (first < xsize && in[first * 4 + 3] != 0) {
            first++;
        }
        add_colors_to_color_cube(fineCube, in, first, 0);
        if (first < xsize) {
            const ImagingOctreeIndex index = cube_index(fineCube);
            PLAIN_OCTREE_LOCALS(&index);
            for (x = first; x < xsize; x++) {
                unsigned int r = in[x * 4], g = in[x * 4 + 1], b = in[x * 4 + 2];
                const unsigned int a = in[x * 4 + 3];
                ColorBucket bucket;
                if (a == 0) {
                    if (!transparency) {
                        transparency = 1;
                        tr = r, tg = g, tb = b;
                    } else {
                        r = tr, g = tg, b = tb;
                    }
                }
                bucket = &buckets[PLAIN_OCTREE_BUCKET(r, g, b, a)];
                bucket->count += 1;
                bucket->r += r;
                bucket->g += g;
                bucket->b += b;
                bucket->a += a;
            }
        }
    }

    if (!octree_build(fineCube, nQuantPixels, cubeBits, &paletteBuckets, &lookupCube)) {
        goto error;
    }

    {
        ImagingOctreeIndex index = cube_index(lookupCube);
        /* the palette index of every bucket, a table of the first-level
           cache's size, where the buckets themselves are ten times that */
        /* malloc check ok, the cube's size is a small constant */
        lut = malloc(lookupCube->size * sizeof(uint32_t));
        if (!lut) {
            goto error;
        }
        for (i = 0; i < lookupCube->size; i++) {
            lut[i] = lookupCube->buckets[i].count;
        }
        index.lut = lut;
        /* a transparent pixel's index is that of the colour they share */
        const UINT8 transparent = (UINT8)
            lut[(tr >> index.shift[0]) << index.offset[0] |
                (tg >> index.shift[1]) << index.offset[1] |
                (tb >> index.shift[2]) << index.offset[2]];
        for (y = 0; y < ysize; y++) {
            const UINT8 *in = rows[y];
            UINT8 *out = outRows[y];
            if (!withAlpha) {
                if (!IMAGING_SIMD_CALL(quant->octree_map_rgb, lut, in, xsize, out)) {
                    plain_octree_map_rgb(lut, in, xsize, out);
                }
            } else if (!IMAGING_SIMD_CALL(quant->octree_map, &index, in, xsize, out)) {
                plain_octree_map(&index, in, xsize, out);
            }
            if (transparency) {
                for (x = 0; x < xsize; x++) {
                    if (in[x * 4 + 3] == 0) {
                        out[x] = transparent;
                    }
                }
            }
        }
    }

    /* convert palette buckets to RGB pixel palette */
    *palette = create_palette_array(paletteBuckets, nQuantPixels);
    if (!(*palette)) {
        goto error;
    }
    *paletteLength = nQuantPixels;

    free(lut);
    free_color_cube(fineCube);
    free_color_cube(lookupCube);
    free(paletteBuckets);
    return 1;

error:
    free(sums);
    free(lut);
    free_color_cube(lookupCube);
    free(paletteBuckets);
    free_color_cube(fineCube);
    return 0;
}

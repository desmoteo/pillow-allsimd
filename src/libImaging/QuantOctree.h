#ifndef __QUANT_OCTREE_H__
#define __QUANT_OCTREE_H__

#include "QuantTypes.h"

int
quantize_octree(Pixel *, uint32_t, uint32_t, Pixel **, uint32_t *, uint32_t **, int);

/* from the rows of an RGB or RGBA image to the rows of a P image */
int
quantize_octree_rows(
    unsigned char **rows,
    int xsize,
    int ysize,
    uint32_t nQuantPixels,
    Pixel **palette,
    uint32_t *paletteLength,
    unsigned char **outRows,
    int withAlpha
);

#endif

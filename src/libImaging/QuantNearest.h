/* The palette of the quantizer's vector search for the nearest entry
   (Quant.c builds it, QuantSIMD.c searches it): the entries padded to a
   multiple of the widest vector's lanes, 8 with AVX2, which the 4-lane
   kernels take as well. */
#ifndef __QUANT_NEAREST_H__
#define __QUANT_NEAREST_H__

#include <stdint.h>

#define NEAREST_LANES 8

typedef struct ImagingNearestPalette {
    int32_t r[256 + NEAREST_LANES], g[256 + NEAREST_LANES], b[256 + NEAREST_LANES];
    uint32_t rank[256 + NEAREST_LANES]; /* or all ones for padding */
    uint32_t entry[257];                /* palette entry of each rank */
    uint32_t n;                         /* entries, padded */
} NearestPalette;

#endif

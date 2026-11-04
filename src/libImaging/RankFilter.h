/* The rank filters' edge helper (RankFilter.c, RankFilterSIMD.c). */
#ifndef __RANK_FILTER_H__
#define __RANK_FILTER_H__

#include <string.h>

/* n bytes of the ps-byte pixel at pixel, repeated: an edge of a row. */
static inline void
RankFillPixels(UINT8 *dst, const UINT8 *pixel, int ps, int n) {
    int k;
    if (ps == 1) {
        memset(dst, pixel[0], n);
        return;
    }
    for (k = 0; k < n; k++) {
        dst[k] = pixel[k % ps];
    }
}

#endif

/* The box blur's shared pieces (BoxBlur.c, BoxBlurSIMD.c): the running-sum
   row blur for any vector type and its px4 instance, and the direct passes'
   parameters. */
#ifndef __BOX_BLUR_H__
#define __BOX_BLUR_H__

#include "ImagingPx4.h"

/* The running-sum box blur of a row of 4-byte pixels, for any vector
   type: the caller defines V (the type), V_LOAD(i) (pixel i as four
   32-bit channels), V_SET1, V_ADD, V_SUB, V_MUL and V_SAVE(x, bulk). */
#define BOX_LINE_BLUR                                                                 \
    {                                                                                 \
        int x;                                                                        \
        V acc, bulk;                                                                  \
        V wws = V_SET1(ww);                                                           \
        V fws = V_SET1(fw);                                                           \
        /* Compute acc for -1 pixel (outside of image): from "-radius-1" to */        \
        /* "-1" get first pixel, then from "0" to "radius-1". */                      \
        acc = V_MUL(V_LOAD(0), V_SET1(radius + 1));                                   \
        /* As radius can be bigger than xsize, iterate to edgeA -1. */                \
        for (x = 0; x < edgeA - 1; x++) {                                             \
            acc = V_ADD(acc, V_LOAD(x));                                              \
        }                                                                             \
        /* Then multiply remainder to last x. */                                      \
        acc = V_ADD(acc, V_MUL(V_LOAD(x), V_SET1(radius - edgeA + 1)));               \
        if (edgeA <= edgeB) {                                                         \
            /* Subtract pixel from left ("0"). Add pixels from radius. */             \
            for (x = 0; x < edgeA; x++) {                                             \
                BOX_STEP(0, x + radius, 0, x + radius + 1);                           \
            }                                                                         \
            /* Subtract previous pixel from "-radius". Add from radius. */            \
            for (x = edgeA; x < edgeB; x++) {                                         \
                BOX_STEP(x - radius - 1, x + radius, x - radius - 1, x + radius + 1); \
            }                                                                         \
            /* Subtract previous pixel from "-radius". Add last pixel. */             \
            for (x = edgeB; x <= lastx; x++) {                                        \
                BOX_STEP(x - radius - 1, lastx, x - radius - 1, lastx);               \
            }                                                                         \
        } else {                                                                      \
            for (x = 0; x < edgeB; x++) {                                             \
                BOX_STEP(0, x + radius, 0, x + radius + 1);                           \
            }                                                                         \
            for (x = edgeB; x < edgeA; x++) {                                         \
                BOX_STEP(0, lastx, 0, lastx);                                         \
            }                                                                         \
            for (x = edgeA; x <= lastx; x++) {                                        \
                BOX_STEP(x - radius - 1, lastx, x - radius - 1, lastx);               \
            }                                                                         \
        }                                                                             \
    }

/* acc[0] += lineIn[add][0] - lineIn[subtract][0];
   bulk[0] = (acc[0] * ww) + (lineIn[left][0] + lineIn[right][0]) * fw;
   lineOut[x][0] = (UINT8)((bulk[0] + (1 << 23)) >> 24); */
#define BOX_STEP(subtract, add, left, right)                                       \
    acc = V_SUB(V_ADD(acc, V_LOAD(add)), V_LOAD(subtract));                        \
    bulk = V_ADD(V_MUL(V_ADD(V_LOAD(left), V_LOAD(right)), fws), V_MUL(acc, wws)); \
    V_SAVE(x, bulk);

/* The running-sum blur of a row of 4-byte pixels, a pixel at a time, with
   px4: the portable lanes in BoxBlur.c, the SIMD ones in BoxBlurSIMD.c. */
static inline void
box_line_blur32(
    UINT32 *lineOut,
    UINT32 *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
){
#define V px4
#define V_LOAD(i) px4_load(&lineIn[i])
#define V_SET1 px4_set1
#define V_ADD px4_add
#define V_SUB px4_sub
#define V_MUL px4_mul
#define V_SAVE(x, bulk) lineOut[x] = px4_round24(bulk)
    BOX_LINE_BLUR
#undef V
#undef V_LOAD
#undef V_SET1
#undef V_ADD
#undef V_SUB
#undef V_MUL
#undef V_SAVE
}

/* Up to this radius a pass may use ImagingBoxBlurDirect instead of running
   sums: always for radius 0, and with the active tier's direct kernel for
   radius 1, and to BOX_DIRECT_MAX where the fast form below applies. */
#define BOX_DIRECT_MAX 4

/* How ImagingBoxBlurDirect computes a pass: from count = 2 * radius + 3
   sources with the weights ww and fw, or, if fast, from the sum c of the
   box alone, out = (((c << s) + k) * m >> 16) >> t in 16 bits. That needs
   fw = 0, which integer radii from 1 to 4 have, and the result equals the
   one of the full arithmetic for every sum (checked in BoxDirectInit). */
typedef struct ImagingBoxDirect {
    int direct, count;
    UINT32 ww, fw;
    int fast, s, t, m, k;
} BoxDirect;

#endif

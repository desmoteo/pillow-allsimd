/* Alpha compositing (AlphaComposite.c, AlphaCompositeSIMD.c): a pixel and
   the coefficients' precision. */
#ifndef __ALPHA_COMPOSITE_H__
#define __ALPHA_COMPOSITE_H__

/* the extra bits of the coefficients */
#define AC_PRECISION_BITS 7

typedef struct {
    UINT8 r;
    UINT8 g;
    UINT8 b;
    UINT8 a;
} rgba8;

#endif

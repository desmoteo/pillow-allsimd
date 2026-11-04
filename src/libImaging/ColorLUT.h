/* The 3D colour lookup table's fixed-point layout (ColorLUT.c), for its
   NEON kernel too (ColorLUTSIMD.c); the x86 kernels use a layout of their
   own. */
#ifndef __COLOR_LUT_H__
#define __COLOR_LUT_H__

/* 8 bits for result. Table can overflow [0, 1.0] range,
   so we need extra bits for overflow and negative values.
   NOTE: This value should be the same as in _imaging/_prepare_lut_table() */
#define LUT_PRECISION_BITS (16 - 8 - 2)
#define LUT_PRECISION_ROUNDING (1 << (LUT_PRECISION_BITS - 1))

/* 8 - scales are multiplied on byte.
   6 - max index in the table
       (max size is 65, but index 64 is not reachable) */
#define LUT_SCALE_BITS (32 - 8 - 6)
#define LUT_SCALE_MASK ((1 << LUT_SCALE_BITS) - 1)

#define LUT_SHIFT_BITS (16 - 1)

static inline int
table_index3D(int index1D, int index2D, int index3D, int size1D, int size1D_2D) {
    return index1D + index2D * size1D + index3D * size1D_2D;
}

#endif

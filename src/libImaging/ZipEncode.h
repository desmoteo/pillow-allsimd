/* The PNG encoder's filter cost (ZipEncode.c, ZipEncodeSIMD.c): the sum
   over a row of |v| taken as signed bytes, as libpng chooses filters. */
#ifndef __ZIP_ENCODE_H__
#define __ZIP_ENCODE_H__

#define PNG_COST(v) ((v) < 128 ? (v) : 256 - (v))

#endif

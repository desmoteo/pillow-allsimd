/* Fast mode's transforms with the vector loops of this tier: the
   instance of TransformFast.h with TF_SIMD, compiled into SIMDKernels.c. */

#define TF_SIMD
#include "TransformFast.h"
#undef TF_SIMD

static const ImagingSIMDTransformFastKernels transform_fast_simd_kernels = {
    .tiles = tf_transform_tiles,
};

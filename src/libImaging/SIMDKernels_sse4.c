/* The SSE4.1 tier of a dispatch build: SIMDKernels.c compiled for it
   (setup.py adds -msse4.1 for this file). */
#define IMAGING_SIMD_TIER IMAGING_SIMD_TIER_SSE4
#include "SIMDKernels.c"

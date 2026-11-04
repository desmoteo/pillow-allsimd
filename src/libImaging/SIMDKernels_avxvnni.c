/* The AVX-VNNI tier of a dispatch build: SIMDKernels.c compiled for it
   (setup.py adds -msse4.1 -mavx2 -mavxvnni for this file). */
#define IMAGING_SIMD_TIER IMAGING_SIMD_TIER_AVXVNNI
#include "SIMDKernels.c"

/*
 * The Imaging library
 *
 * Fast mode: whether operations may use kernels that are faster than the
 * exact ones and whose results may differ from Pillow's by a documented
 * amount. On by default; ImagingSetFast(0) selects the exact kernels.
 * See design/APPROXIMATE_KERNELS_PLAN.md.
 *
 * The flags are a bit set (IMAGING_FAST_ALL is the only bit so far). An
 * operation reads them once, at its entry point, so one call never mixes
 * kernels; writes and reads are atomic, for free-threaded Python.
 */

#include "Imaging.h"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

static long fast_flags = IMAGING_FAST_ALL;

void
ImagingSetFast(int flags) {
#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(&fast_flags, (long)flags, __ATOMIC_RELEASE);
#elif defined(_MSC_VER)
    _InterlockedExchange(&fast_flags, (long)flags);
#else
    fast_flags = flags;
#endif
}

int
ImagingGetFast(void) {
#if defined(__GNUC__) || defined(__clang__)
    return (int)__atomic_load_n(&fast_flags, __ATOMIC_ACQUIRE);
#elif defined(_MSC_VER)
    return (int)_InterlockedOr(&fast_flags, 0);
#else
    return (int)fast_flags;
#endif
}

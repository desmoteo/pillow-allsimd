/*
 * The Imaging library
 *
 * Fast mode: whether operations may use kernels that are faster than the
 * exact ones and whose results may differ from Pillow's by a documented
 * amount. On by default; ImagingSetFast(0) selects the exact kernels.
 * See design/APPROXIMATE_KERNELS_PLAN.md.
 *
 * The flags are a bit set (IMAGING_FAST_ALL is the only bit so far). There
 * is a process-wide default, set by ImagingSetFast() and read atomically,
 * for free-threaded Python, and a per-thread override, set by
 * ImagingSetFastThread(), for Image.fast_mode() blocks: a block changes
 * the thread it runs in and no other, and nested blocks restore in order
 * whatever other threads do (design/REVIEW_2026-11.md, finding 1). An
 * operation reads the flags once, at its entry point, so one call never
 * mixes kernels.
 */

#include "Imaging.h"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define IMAGING_THREAD_LOCAL __declspec(thread)
#else
#define IMAGING_THREAD_LOCAL __thread
#endif

static long fast_flags = IMAGING_FAST_ALL;

/* This thread's flags, or -1 for the process default */
static IMAGING_THREAD_LOCAL int fast_thread_flags = -1;

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
ImagingSetFastThread(int flags) {
    int previous = fast_thread_flags;
    fast_thread_flags = flags < 0 ? -1 : flags;
    return previous;
}

int
ImagingGetFast(void) {
    if (fast_thread_flags >= 0) {
        return fast_thread_flags;
    }
#if defined(__GNUC__) || defined(__clang__)
    return (int)__atomic_load_n(&fast_flags, __ATOMIC_ACQUIRE);
#elif defined(_MSC_VER)
    return (int)_InterlockedOr(&fast_flags, 0);
#else
    return (int)fast_flags;
#endif
}

from __future__ import annotations

import platform
import sys

from PIL import features


def test_wheel_modules() -> None:
    expected_modules = {"pil", "tkinter", "freetype2", "littlecms2", "webp", "avif"}

    if sys.platform == "win32":
        # tkinter is not available in cibuildwheel installed CPython on Windows
        try:
            import tkinter

            assert tkinter
        except ImportError:
            expected_modules.remove("tkinter")

        # libavif is not available on Windows for ARM64 architectures
        if platform.machine() == "ARM64":
            expected_modules.remove("avif")

    elif sys.platform == "ios":
        # tkinter is not available on iOS
        expected_modules.remove("tkinter")

    assert set(features.get_supported_modules()) == expected_modules


def test_wheel_codecs() -> None:
    expected_codecs = {"jpg", "jpg_2000", "zlib", "libtiff"}

    assert set(features.get_supported_codecs()) == expected_codecs


def test_wheel_features() -> None:
    expected_features = {
        "raqm",
        "fribidi",
        "harfbuzz",
        "libjpeg_turbo",
        "zlib_ng",
        "xcb",
        # Pillow-AllSIMD's fast mode and SIMD kernels, in every wheel, and
        # libdeflate for PNG saving
        "fast",
        "simd",
        "libdeflate",
    }

    if sys.platform == "win32":
        expected_features.remove("xcb")
    elif sys.platform == "ios":
        # Can't distribute raqm due to licensing, and there's no system version;
        # fribidi and harfbuzz won't be available if raqm isn't available.
        expected_features -= {"raqm", "fribidi", "harfbuzz"}

    assert set(features.get_supported_features()) == expected_features


def test_wheel_simd_tiers() -> None:
    # An x86-64 wheel holds every x86 tier and chooses at run time; an
    # arm64 wheel has NEON (design/SIMD_DISPATCH_PLAN.md)
    from PIL import Image

    machine = platform.machine().lower()
    expected: tuple[str, ...]
    if machine in ("x86_64", "amd64"):
        expected = ("scalar", "sse4", "avx2", "avxvnni")
        if sys.platform == "win32":  # MSVC has no AVX-VNNI tier
            expected = expected[:3]
    elif machine in ("aarch64", "arm64"):
        expected = ("scalar", "neon")
    else:
        expected = ("scalar",)
    assert Image.core.simd_tiers == expected
    assert Image.core.simd_kernels in expected

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from PIL import Image, ImageFilter, features

from .helper import hopper

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def restore_fast_mode() -> Generator[None]:
    previous = Image.get_fast()
    yield
    Image.set_fast(previous)


def test_feature() -> None:
    assert features.check_feature("fast") is True
    assert features.version_feature("fast") is None


def test_off_by_default() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PILLOW_SIMD_FAST"}
    out = subprocess.check_output(
        [sys.executable, "-c", "from PIL import Image; print(Image.get_fast())"],
        env=env,
    )
    assert out.strip() == b"False"


def test_set_get() -> None:
    Image.set_fast(True)
    assert Image.get_fast() is True
    Image.set_fast(False)
    assert Image.get_fast() is False
    Image.set_fast()
    assert Image.get_fast() is True


@pytest.mark.parametrize(
    "value, expected",
    (
        ("1", True),
        ("true", True),
        ("ON", True),
        (" yes ", True),
        ("0", False),
        ("false", False),
        ("Off", False),
        ("no", False),
        ("", False),
    ),
)
def test_env(value: str, expected: bool) -> None:
    Image.set_fast(not expected)
    Image._apply_env_variables({"PILLOW_SIMD_FAST": value})
    assert Image.get_fast() is expected


def test_env_invalid() -> None:
    Image.set_fast(True)
    with pytest.warns(UserWarning, match="PILLOW_SIMD_FAST"):
        Image._apply_env_variables({"PILLOW_SIMD_FAST": "maybe"})
    assert Image.get_fast() is True


def test_env_unset() -> None:
    Image.set_fast(True)
    Image._apply_env_variables({})
    assert Image.get_fast() is True


def test_env_at_import() -> None:
    env = dict(os.environ, PILLOW_SIMD_FAST="1")
    out = subprocess.check_output(
        [sys.executable, "-c", "from PIL import Image; print(Image.get_fast())"],
        env=env,
    )
    assert out.strip() == b"True"


def test_context_manager() -> None:
    Image.set_fast(False)
    with Image.fast_mode():
        assert Image.get_fast() is True
        with Image.fast_mode(False):
            assert Image.get_fast() is False
        assert Image.get_fast() is True
    assert Image.get_fast() is False


def test_context_manager_restores_after_exception() -> None:
    Image.set_fast(False)
    with pytest.raises(ZeroDivisionError):
        with Image.fast_mode():
            1 / 0
    assert Image.get_fast() is False


def test_toggle_while_processing() -> None:
    # The flag is read once per operation; toggling it from another thread
    # while operations run must neither crash nor leave it in a bad state.
    im = hopper("RGB")
    stop = threading.Event()

    def toggle() -> None:
        while not stop.is_set():
            Image.set_fast(True)
            Image.set_fast(False)

    t = threading.Thread(target=toggle)
    t.start()
    try:
        for _ in range(20):
            im.filter(ImageFilter.GaussianBlur(2))
            im.convert("HSV")
            im.convert("YCbCr")
            im.rotate(30, Image.Resampling.BILINEAR)
    finally:
        stop.set()
        t.join()
    assert Image.get_fast() in (True, False)

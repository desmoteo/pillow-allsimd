from __future__ import annotations

import io
import sys
import sysconfig

import pytest

from PIL import Image

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Generator


FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

gil_enabled_at_start = True
if FREE_THREADED_BUILD:
    gil_enabled_at_start = sys._is_gil_enabled()  # type: ignore[attr-defined]


def pytest_report_header(config: pytest.Config) -> str:
    try:
        from PIL import features

        with io.StringIO() as out:
            features.pilinfo(out=out, supported_formats=False)
            return out.getvalue()
    except Exception as e:
        return f"pytest_report_header failed: {e}"


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if (
        FREE_THREADED_BUILD
        and not gil_enabled_at_start
        and sys._is_gil_enabled()  # type: ignore[attr-defined]
    ):
        tr = terminalreporter
        tr.ensure_newline()
        tr.section("GIL re-enabled", red=True, bold=True)
        tr.line("The GIL was re-enabled at runtime during the tests.")
        tr.line("This can happen with no test failures if the RuntimeWarning")
        tr.line("raised by Python when this happens is filtered by a test.")
        tr.line("")
        tr.line("Please ensure all new C modules declare support for running")
        tr.line("without the GIL. Any new tests that intentionally imports")
        tr.line("code that re-enables the GIL should do so in a subprocess.")
        pytest.exit("GIL re-enabled during tests", returncode=1)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "pil_noop_mark: A conditional mark where nothing special happens",
    )

    # We're marking some tests to ignore valgrind errors and XFAIL them.
    # Ensure that the mark is defined
    # even in cases where pytest-valgrind isn't installed
    try:
        config.addinivalue_line(
            "markers",
            "valgrind_known_error: Tests that have known issues with valgrind",
        )
    except Exception:
        # valgrind is already installed
        pass


@pytest.fixture
def enable_block_allocator() -> Generator[None]:
    """
    Enable the block allocator for image storage.
    """
    Image.core.set_use_block_allocator(1)
    try:
        yield
    finally:
        Image.core.set_use_block_allocator(0)


@pytest.fixture
def aligned_arena() -> Generator[None]:
    """
    Configure the global image allocator to use a 64-byte alignment.
    """
    alignment = Image.core.get_alignment()
    Image.core.set_alignment(64)
    try:
        yield
    finally:
        Image.core.set_alignment(alignment)


# Pillow-AllSIMD: fast mode is on by default. These upstream tests check
# values that Pillow's exact arithmetic gives (YCbCr histograms and colour
# counts, rotated reference images, blend rounding), or compare a resize
# without reducing_gap with ones with it, so they run with the exact kernels.
EXACT_KERNEL_TESTS = {
    "test_image_entropy.py::test_entropy",
    "test_image_getcolors.py::test_getcolors",
    "test_image_getdata.py::test_mode",
    "test_image_histogram.py::test_histogram",
    "test_image_histogram.py::test_histogram_masked",
    "test_image_resize.py::TestReducingGapResize::test_reducing_gap_2",
    "test_image_resize.py::TestReducingGapResize::test_reducing_gap_3",
    "test_image_resize.py::TestReducingGapResize::test_reducing_gap_8",
    "test_image_rotate.py::test_resample",
    "test_image_rotate.py::test_translate",
    "test_imagechops.py::test_blend",
}


@pytest.fixture(autouse=True)
def exact_kernels_for_pillow_values(request: pytest.FixtureRequest) -> Generator[None]:
    names = request.node.nodeid.split("[")[0].split("::")[1:]
    if "::".join((request.node.path.name, *names)) not in EXACT_KERNEL_TESTS:
        yield
        return
    with Image.fast_mode(False):
        yield

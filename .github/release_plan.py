#!/usr/bin/env python3
"""
The plan of a release run (.github/workflows/release.yml): which platforms
to build, from the tag's suffix or the manual run's input, as outputs for
the workflow's other jobs:

    sdist     true to build the source package
    wheels    the wheel matrix, a JSON list of cibuildwheel platforms
    expected  how many files the run should upload

Environment: EVENT (push or workflow_dispatch), TAG (the tag of a push), and
PLATFORMS (all, linux or macos: a manual run's input; a tag chooses with
its suffix, -linux or -macos, or none for all). A push also checks that the
tag's version is the package's.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

WHEELS_PER_PLATFORM = 7  # pyproject.toml's [tool.cibuildwheel]

# Each wheel's build log must end its configuration summary with
# "--- SIMD kernels: <kernels>": the dispatch build on x86-64 (SSE4.1, AVX2
# and AVX-VNNI kernels, chosen when _imaging loads), NEON on arm64.
X86_64_KERNELS = "dispatch (SSE4.1, AVX2, AVX-VNNI)"

PLATFORMS = {
    "linux": [
        {
            "name": "manylinux_2_28 x86_64",
            "platform": "linux",
            "os": "ubuntu-latest",
            "cibw_arch": "x86_64",
            "build": "*manylinux*",
            "macosx_deployment_target": "",
            "kernels": X86_64_KERNELS,
        },
        {
            "name": "musllinux x86_64",
            "platform": "linux",
            "os": "ubuntu-latest",
            "cibw_arch": "x86_64",
            "build": "*musllinux*",
            "macosx_deployment_target": "",
            "kernels": X86_64_KERNELS,
        },
        {
            "name": "manylinux_2_28 aarch64",
            "platform": "linux",
            "os": "ubuntu-24.04-arm",
            "cibw_arch": "aarch64",
            "build": "*manylinux*",
            "macosx_deployment_target": "",
            "kernels": "NEON",
        },
        {
            "name": "musllinux aarch64",
            "platform": "linux",
            "os": "ubuntu-24.04-arm",
            "cibw_arch": "aarch64",
            "build": "*musllinux*",
            "macosx_deployment_target": "",
            "kernels": "NEON",
        },
    ],
    "macos": [
        {
            "name": "macOS arm64",
            "platform": "macos",
            "os": "macos-latest",
            "cibw_arch": "arm64",
            "build": "",
            "macosx_deployment_target": "11.0",
            "kernels": "NEON",
        },
    ],
}


def version() -> str:
    source = (Path(__file__).parent.parent / "src/PIL/_version.py").read_text()
    match = re.search(r'__version__ = "([^"]+)"', source)
    assert match
    return match.group(1)


def plan(event: str, tag: str, platforms: str) -> dict[str, object]:
    if event == "push":
        match = re.fullmatch(r"allsimd-(.+?)(?:-(linux|macos))?", tag)
        if not match or match.group(1) != version():
            sys.exit(f"tag {tag}, version {version()}")
        platforms = match.group(2) or "all"
    elif platforms not in ("all", "linux", "macos"):
        sys.exit(f"unknown platforms {platforms!r}")
    wheels = [
        p for name in PLATFORMS if platforms in ("all", name) for p in PLATFORMS[name]
    ]
    sdist = platforms != "macos"
    return {
        "sdist": sdist,
        "wheels": wheels,
        "expected": int(sdist) + WHEELS_PER_PLATFORM * len(wheels),
    }


def main() -> None:
    env = os.environ
    out = plan(env.get("EVENT", ""), env.get("TAG", ""), env.get("PLATFORMS") or "all")
    print(f"sdist={str(out['sdist']).lower()}")
    print(f"wheels={json.dumps(out['wheels'])}")
    print(f"expected={out['expected']}")


if __name__ == "__main__":
    main()

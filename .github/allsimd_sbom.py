#!/usr/bin/env python3
"""
The CycloneDX SBOM of Pillow-AllSIMD's wheels: Pillow's (generate-sbom.py,
which describes Pillow's C extensions and the native libraries the wheels
bundle, with the versions of dependencies.json) renamed for this package,
with Pillow as its ancestor. The extensions and libraries are the same;
only the package differs. Kept apart from generate-sbom.py so that merging
a Pillow release doesn't conflict.

Usage:
    python3 .github/allsimd_sbom.py [output-file]

The output defaults to pillow_allsimd-{version}.cdx.json in the current
directory. The release workflow embeds it in every wheel (embed-sbom.py,
PEP 770).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

TYPE_CHECKING = False
if TYPE_CHECKING:
    from types import ModuleType

HERE = Path(__file__).parent
REPOSITORY = "https://github.com/desmoteo/pillow-allsimd"


def pillow_sbom() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_sbom", HERE / "generate-sbom.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(version: str) -> dict[str, Any]:
    pillow = pillow_sbom()
    pillow_version = re.sub(r"\.post\d+$", "", version)
    pillow_purl = f"pkg:pypi/pillow@{version}"
    purl = f"pkg:pypi/pillow-allsimd@{version}"

    # Pillow's SBOM, its package references renamed
    text = json.dumps(pillow.generate(version))
    sbom: dict[str, Any] = json.loads(text.replace(pillow_purl, purl))

    sbom["metadata"]["tools"]["components"].append(
        {"type": "application", "name": "allsimd_sbom.py", "group": "pillow-allsimd"}
    )
    package = sbom["metadata"]["component"]
    package.update(
        {
            "name": "pillow-allsimd",
            "description": (
                "Python Imaging Library (Pillow fork with SIMD kernels for x86 and "
                "arm64)"
            ),
            "externalReferences": [
                {"type": "website", "url": REPOSITORY},
                {"type": "vcs", "url": REPOSITORY},
                {
                    "type": "documentation",
                    "url": "https://desmoteo.github.io/pillow-allsimd/",
                },
                {"type": "issue-tracker", "url": f"{REPOSITORY}/issues"},
                {"type": "security-contact", "url": f"{REPOSITORY}/security/policy"},
            ],
            "pedigree": {
                "ancestors": [
                    {
                        "bom-ref": f"pkg:pypi/pillow@{pillow_version}#upstream",
                        "type": "library",
                        "name": "Pillow",
                        "version": pillow_version,
                        "purl": f"pkg:pypi/pillow@{pillow_version}",
                        "externalReferences": [
                            {
                                "type": "vcs",
                                "url": "https://github.com/python-pillow/Pillow",
                            },
                        ],
                    }
                ],
                "notes": (
                    f"Pillow {pillow_version} with SIMD kernels (SSE4.1, AVX2, "
                    "AVX-VNNI, NEON) and a fast mode added to the C extensions; "
                    "forked from Pillow-SIMD (https://github.com/uploadcare/"
                    "pillow-simd). The C extensions and the bundled native "
                    "libraries are those of the Pillow release, plus libdeflate, "
                    "which compresses the PNGs it saves."
                ),
            },
        }
    )
    # libdeflate, which the wheels bundle for PNG saving (ZipEncode.c)
    versions = pillow.load_dep_versions()
    sbom["components"].append(
        {
            "bom-ref": "pkg:generic/libdeflate",
            "type": "library",
            "name": "libdeflate",
            "version": versions["libdeflate"],
            "scope": "optional",
            "description": (
                "DEFLATE compression of the PNGs Pillow-AllSIMD saves (optional; "
                "disable with -C libdeflate=disable)."
            ),
            "licenses": [{"license": {"id": "MIT"}}],
            "externalReferences": [
                {"type": "website", "url": "https://github.com/ebiggers/libdeflate"},
                {
                    "type": "distribution",
                    "url": "https://github.com/ebiggers/libdeflate/releases",
                },
            ],
        }
    )
    for dependency in sbom["dependencies"]:
        if dependency["ref"] == f"{purl}#c-ext/PIL._imaging":
            dependency["dependsOn"].append("pkg:generic/libdeflate")
    return sbom


def main() -> None:
    version = pillow_sbom().get_version()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(f"pillow_allsimd-{version}.cdx.json"),
        help="output file",
    )
    args = parser.parse_args()

    sbom = generate(version)
    args.output.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output} (pillow-allsimd {version}, "
        f"{len(sbom['components'])} components)"
    )


if __name__ == "__main__":
    main()

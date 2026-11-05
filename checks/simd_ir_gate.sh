#!/usr/bin/env bash
# Runs checks/simd_ir_gate.py the way CI does: a dispatch build of this
# checkout in python:3.12-trixie (GCC 14.2, Valgrind 3.24), the image
# checks/simd_ir_baseline.json was made in, since instruction counts belong
# to a compiler.
#
#   checks/simd_ir_gate.sh             compare with the baseline (as CI)
#   checks/simd_ir_gate.sh --update    rewrite the baseline; commit it with
#                                      the change that moved the counts
#
# Other arguments go to simd_ir_gate.py (--jobs, --tolerance, --strict).
# Needs Docker and a CPU with AVX2; uncommitted changes are included.
# With --inside, the steps themselves, in whatever environment this is:
# the workflow .github/workflows/simd-instructions.yml runs that in the
# image.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-python:3.12-trixie}

if [ "${1:-}" = "--inside" ]; then
    shift
    export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PIP_ROOT_USER_ACTION=ignore
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends valgrind \
        libjpeg62-turbo-dev zlib1g-dev libdeflate-dev libpng-dev libtiff-dev \
        libwebp-dev libfreetype-dev liblcms2-dev libopenjp2-7-dev > /dev/null
    python3 -m pip install -q setuptools wheel pybind11
    python3 -m pip install -q --no-deps --no-build-isolation -C simd=dispatch \
        --target /tmp/simd-ir-build .
    cc --version | head -1
    exec python3 checks/simd_ir_gate.py --build /tmp/simd-ir-build \
        --out /tmp/simd-ir "$@"
fi

# A copy of the tree without the build products and local results, so that
# the checkout itself can be mounted read-only.
EXCLUDE="--exclude=./.git --exclude=./build --exclude=./bench-results"
COPY="mkdir /tmp/src && tar -C /src $EXCLUDE --exclude=*.so -cf - ."
COPY="$COPY | tar -C /tmp/src -xf - && cd /tmp/src"
if [ "${1:-}" = "--update" ]; then
    shift
    docker run --rm -v "$PWD":/src:ro -v "$PWD/checks":/out "$IMAGE" bash -c "
        $COPY && checks/simd_ir_gate.sh --inside --update \
            --baseline /tmp/baseline.json $* &&
        cat /tmp/baseline.json > /out/simd_ir_baseline.json"
else
    docker run --rm -v "$PWD":/src:ro "$IMAGE" bash -c "
        $COPY && checks/simd_ir_gate.sh --inside $*"
fi

#!/usr/bin/env bash
# Benchmarks pillow-simd on a remote arm64 or x86-64 machine over SSH.
#
# Copies a git ref of this repository to the host and builds it several
# times: on arm64 NEON and scalar (-C simd=auto / disable); on x86-64 scalar,
# SSE4, AVX2, and AVX-VNNI if the host has it. Installs upstream Pillow from
# PyPI as a baseline, then runs checks/simd_bench.py with an output
# comparison and copies the results back.
#
# Usage: checks/simd_bench_remote.sh [options] HOST
#
#   HOST              host name or IP, optionally user@host
#   -u, --user USER   SSH user
#   -p, --port PORT   SSH port
#   -i, --identity F  SSH private key
#   -o OPTION         extra ssh -o option, e.g. StrictHostKeyChecking=accept-new
#                     (repeatable)
#   --python PY       Python on the host (default: python3, 3.10 or newer)
#   --ref REF         git ref to benchmark (default: HEAD; uncommitted
#                     changes are not included)
#   --workdir DIR     working directory on the host (default: pillow-simd-bench)
#   --rounds N        benchmark rounds (default: 5)
#   --size WxH        benchmark image size (default: 2560x1600)
#   --only NAME       only operations containing NAME (repeatable)
#   --set SET         operation set of simd_bench.py: simd (default), extra
#                     or all
#   --cpu N           pin the benchmark to CPU N (Linux hosts with taskset)
#   --no-upstream     skip the upstream Pillow baseline (no PyPI access needed
#                     for it; build tools must then already be installed)
#   --out DIR         local directory for results (default: bench-results)
#   --keep            keep the working directory on the host
#   --builds "N=OPT .."  builds to compare, each a name and a -C simd= value,
#                     e.g. "avx2=avx2 avxvnni=avxvnni" (default: per host)
#   --force           run even if the host is neither arm64 nor x86-64
#   --docker          build and run in a container instead of on the host:
#                     builds a small image (python:3.12-slim + gcc) on the
#                     host on the fly, with docker or podman
#   --docker-base IMG base image for --docker (default: python:3.12-slim)
#
# Requirements on the host: SSH access, Python 3.10+ with headers and venv,
# a C compiler, and PyPI access to install setuptools/pybind11 into a venv
# (and Pillow for the baseline). With --docker, only docker or podman and
# access to the base image, Debian packages and PyPI; HTTP(S)_PROXY and
# NO_PROXY set on the host are passed on. No image libraries are needed:
# the builds disable the jpeg and zlib codecs.
#
# Examples:
#   checks/simd_bench_remote.sh -u ubuntu --cpu 2 --rounds 7 192.0.2.10
#   checks/simd_bench_remote.sh -u root --cpu 2 --docker server.example.com

set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit "${1:-0}"; }
die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*" >&2; }

HOST="" USER_OPT="" PORT="" IDENTITY="" PY="python3" REF="HEAD"
WORKDIR="pillow-simd-bench" ROUNDS=5 SIZE="2560x1600" CPU="" UPSTREAM=1 SET="simd"
OUT="bench-results" KEEP=0 FORCE=0 BUILDS="" DOCKER=0 DOCKER_BASE="python:3.12-slim"
ONLY=()
SSH_EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        -u|--user) USER_OPT=$2; shift 2 ;;
        -p|--port) PORT=$2; shift 2 ;;
        -i|--identity) IDENTITY=$2; shift 2 ;;
        -o) SSH_EXTRA+=(-o "$2"); shift 2 ;;
        --python) PY=$2; shift 2 ;;
        --ref) REF=$2; shift 2 ;;
        --builds) BUILDS=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        --rounds) ROUNDS=$2; shift 2 ;;
        --size) SIZE=$2; shift 2 ;;
        --only) ONLY+=("$2"); shift 2 ;;
        --set) SET=$2; shift 2 ;;
        --cpu) CPU=$2; shift 2 ;;
        --no-upstream) UPSTREAM=0; shift ;;
        --out) OUT=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --force) FORCE=1; shift ;;
        --docker) DOCKER=1; shift ;;
        --docker-base) DOCKER_BASE=$2; DOCKER=1; shift 2 ;;
        -*) echo "unknown option: $1" >&2; usage 2 ;;
        *) [ -z "$HOST" ] || die "more than one host given"; HOST=$1; shift ;;
    esac
done
[ -n "$HOST" ] || usage 2
[ -n "$USER_OPT" ] && HOST="$USER_OPT@${HOST#*@}"

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
COMMIT=$(git -C "$REPO" rev-parse --short "$REF^{commit}") || die "unknown ref $REF"
if [ "$REF" = "HEAD" ] && [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    log "note: uncommitted changes are not benchmarked (using $COMMIT)"
fi

# One SSH connection for all commands.
CTL_DIR=$(mktemp -d)
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$CTL_DIR/%C" -o ControlPersist=600
          -o ServerAliveInterval=30)
[ -n "$PORT" ] && SSH_OPTS+=(-p "$PORT")
[ -n "$IDENTITY" ] && SSH_OPTS+=(-i "$IDENTITY")
SSH_OPTS+=("${SSH_EXTRA[@]+"${SSH_EXTRA[@]}"}")
cleanup() {
    ssh "${SSH_OPTS[@]}" -O exit "$HOST" 2>/dev/null || true
    rm -rf "$CTL_DIR"
}
trap cleanup EXIT
rsh() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
# Runs a script with bash on the host, whatever the login shell is.
rbash() { ssh "${SSH_OPTS[@]}" "$HOST" bash -s <<< "$1"; }

log "connecting to $HOST"
ARCH=$(rsh uname -m) || die "cannot connect to $HOST"
OS=$(rsh uname -s)
case "$ARCH" in
    aarch64|arm64) DEFAULT_BUILDS="scalar=disable neon=auto" ;;
    x86_64|amd64)
        DEFAULT_BUILDS="scalar=disable sse4=sse4 avx2=avx2"
        if rsh "grep -qw avx_vnni /proc/cpuinfo" 2>/dev/null; then
            DEFAULT_BUILDS="$DEFAULT_BUILDS avxvnni=avxvnni"
        fi ;;
    *) [ "$FORCE" = 1 ] || die "$HOST is $ARCH, not arm64 or x86-64 (use --force to run anyway)"
       DEFAULT_BUILDS="auto=auto scalar=disable" ;;
esac
BUILDS=${BUILDS:-$DEFAULT_BUILDS}
log "host: $OS $ARCH"

# Build commands run in $WORKDIR, on the host or in a container.
if [ "$DOCKER" = 1 ]; then
    ENGINE=$(rsh 'command -v docker || command -v podman' | head -1) \
        || die "--docker needs docker or podman on $HOST"
    ENGINE=${ENGINE##*/}
    IMAGE="pillow-simd-bench"
    PROXY_ARGS="-e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY -e http_proxy -e https_proxy -e no_proxy"
    log "building the $IMAGE image from $DOCKER_BASE with $ENGINE"
    rbash "set -e
        $ENGINE build -q -t $IMAGE \
            --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
            --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy - <<'DOCKERFILE'
FROM $DOCKER_BASE
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir setuptools wheel pybind11
DOCKERFILE
    " >/dev/null || die "could not build the image (no access to $DOCKER_BASE, Debian or PyPI?)"
    BPY="python3" BUILD_PY="python3"
    renv() {
        ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $WORKDIR && cd $WORKDIR && $ENGINE run --rm -i \
            -v \"\$PWD:/work:Z\" -w /work $PROXY_ARGS $IMAGE bash -s" <<< "set -e; $1"
    }
else
    BPY="venv/bin/python" BUILD_PY="../venv/bin/python"
    renv() { rbash "set -e; mkdir -p $WORKDIR && cd $WORKDIR; $1"; }
fi

if [ "$DOCKER" = 1 ]; then
    PY=python3
    log "checking python3, headers and compiler in the container"
else
    log "checking $PY, headers and compiler on the host"
fi
renv "
    command -v $PY >/dev/null || { echo 'missing: $PY'; exit 1; }
    $PY -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
        || { echo 'Python 3.10 or newer required'; exit 1; }
    inc=\$($PY -c 'import sysconfig; print(sysconfig.get_paths()[\"include\"])')
    [ -f \"\$inc/Python.h\" ] || { echo \"missing Python headers in \$inc (install python3-dev)\"; exit 1; }
    $PY -c 'import venv' 2>/dev/null || { echo 'missing venv module (install python3-venv)'; exit 1; }
    cc=\$($PY -c 'import sysconfig; print((sysconfig.get_config_var(\"CC\") or \"cc\").split()[0])')
    command -v \$cc >/dev/null || command -v cc >/dev/null \
        || { echo 'missing C compiler (install gcc or Xcode command line tools)'; exit 1; }
" || die "host is missing build requirements (see above)"

log "copying $COMMIT to $HOST:$WORKDIR"
rsh "rm -rf $WORKDIR && mkdir -p $WORKDIR/src"
git -C "$REPO" archive --format=tar "$COMMIT" | rsh "tar -x -C $WORKDIR/src"
# The harness is taken from the working tree, so older refs can be benchmarked.
rsh "cat > $WORKDIR/simd_bench.py" < "$REPO/checks/simd_bench.py"

VERSION=$(git -C "$REPO" show "$COMMIT:src/PIL/_version.py" \
    | sed -n 's/^__version__ = "\([0-9.]*\).*"/\1/p' | sed 's/\.$//')

if [ "$DOCKER" = 0 ]; then
    log "setting up a venv with build tools"
    renv "$PY -m venv venv
        venv/bin/python -m pip install -q --upgrade pip setuptools wheel pybind11
    " || die "could not install build tools into the venv (no PyPI access?)"
fi

build() {  # name simd-option
    log "building $1 (-C simd=$2)"
    renv "rm -rf build-$1 && cp -r src build-$1
        cd build-$1
        $BUILD_PY -m pip install -v --no-deps --no-build-isolation --no-cache-dir \
            -C simd=$2 -C jpeg=disable -C zlib=disable --target ../$1 . > ../build-$1.log 2>&1 \
            || { tail -40 ../build-$1.log; exit 1; }
        grep 'SIMD kernels' ../build-$1.log | sed 's/^ *//'
    " || die "build of $1 failed (log above, full log in $WORKDIR/build-$1.log)"
}
for b in $BUILDS; do
    build "${b%%=*}" "${b#*=}"
done

VARIANTS=()
if [ "$UPSTREAM" = 1 ]; then
    log "installing upstream Pillow ${VERSION:-latest} as the baseline"
    if renv "$BPY -m pip install -q --no-deps --only-binary :all: \
            --target upstream 'pillow==$VERSION'" 2>/dev/null \
       || renv "rm -rf upstream && $BPY -m pip install -q --no-deps \
            --only-binary :all: --target upstream pillow"; then
        VARIANTS+=(--variant "upstream=upstream")
    else
        log "warning: could not install upstream Pillow, continuing without it"
    fi
fi
for b in $BUILDS; do
    VARIANTS+=(--variant "${b%%=*}=${b%%=*}")
done

ARGS=("${VARIANTS[@]}" --set "$SET" --rounds "$ROUNDS" --size "$SIZE" --check --json results.json)
[ -n "$CPU" ] && ARGS+=(--cpu "$CPU")
for o in "${ONLY[@]+"${ONLY[@]}"}"; do ARGS+=(--only "$o"); done
QARGS=$(printf ' %q' "${ARGS[@]}")

log "running the benchmark (${ROUNDS} rounds, $SIZE); this takes a few minutes"
STAMP=$(date +%Y%m%d-%H%M%S)
DEST="$OUT/${HOST#*@}-$COMMIT-$STAMP"
mkdir -p "$DEST"
renv "$BPY -I simd_bench.py$QARGS" | tee "$DEST/report.txt"

log "copying results to $DEST"
rsh "cd $WORKDIR && cat results.json" > "$DEST/results.json"
for b in $BUILDS; do
    rsh "cd $WORKDIR && cat build-${b%%=*}.log" > "$DEST/build-${b%%=*}.log"
done
echo "commit $COMMIT, host $HOST ($OS $ARCH)" > "$DEST/info.txt"

if [ "$KEEP" = 0 ]; then
    if [ "$DOCKER" = 1 ]; then
        # Files written in the container may belong to its root user.
        renv "rm -rf /work/* /work/.[!.]*" || true
    fi
    rsh "rm -rf $WORKDIR"
else
    log "kept $HOST:$WORKDIR"
fi
log "done: $DEST/report.txt"

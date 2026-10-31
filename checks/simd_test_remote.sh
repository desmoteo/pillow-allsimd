#!/usr/bin/env bash
# Runs the pillow-simd test suite in a container, on a remote machine over
# SSH or on this one, for each SIMD build.
#
# The container brings its own Python and image libraries, so the host's
# Python version does not matter: a host with Python 3.10 can test a
# release that needs 3.11. Copies a git ref of this repository to the host,
# builds an image (python:<version>-slim with the codec libraries) with
# docker or podman, builds pillow-simd once per SIMD option, runs pytest on
# each build and compares the outputs of the builds with
# checks/simd_bench.py --check. Logs and JUnit XML are copied back.
#
# Usage: checks/simd_test_remote.sh [options] HOST
#
#   HOST              host name or IP, optionally user@host, or "local" to
#                     run on this machine
#   -u, --user USER   SSH user
#   -p, --port PORT   SSH port
#   -i, --identity F  SSH private key
#   -o OPTION         extra ssh -o option (repeatable)
#   --ref REF         git ref to test (default: HEAD; uncommitted changes
#                     are not included)
#   --python-version V  Python in the container (default: 3.13)
#   --docker-base IMG base image (default: python:<version>-slim-trixie)
#   --platform P      container platform, e.g. linux/arm64 to test arm64
#                     under qemu on an x86-64 host (slow)
#   --builds "N=OPT .."  builds to test, each a name and a -C simd= value
#                     (default: per architecture, as simd_bench_remote.sh)
#   -n, --jobs N      pytest-xdist workers per build (default: the host's
#                     CPU count, at most 8; 0 runs without xdist)
#   --pytest-args A   extra pytest arguments, e.g. "-k filter" or test paths
#                     (default: Tests -m "not mem")
#   --no-compare      skip the output comparison between the builds
#   --workdir DIR     working directory below the home directory on the
#                     host (default: pillow-simd-test)
#   --out DIR         local directory for results (default:
#                     bench-results/test-results)
#   --keep            keep the working directory on the host
#
# The working directory is created by this script and marked with a
# .pillow-simd-test file; an existing directory without that file is never
# deleted. The container runs as the SSH user, so everything it writes can
# be removed without root.
#
# Requirements on the host: SSH access (unless HOST is "local"), docker or
# podman usable by that user, and access to the base image, Debian packages
# and PyPI; HTTP(S)_PROXY and NO_PROXY set on the host are passed on.
#
# The image has the optional test dependencies (Tk, Qt, Arrow, psutil,
# IPython, ImageMagick, djpeg, a Polish locale, a virtual display), and the
# extra test images of this checkout, if installed with
# depends/install_extra_test_images.sh, are copied with the ref.
#
# Examples:
#   checks/simd_test_remote.sh -u ubuntu 192.0.2.10
#   checks/simd_test_remote.sh --python-version 3.14 -n 4 192.0.2.10
#   checks/simd_test_remote.sh --platform linux/arm64 \
#       --pytest-args "Tests/test_image_filter.py Tests/test_image_resample.py" local

set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit "${1:-0}"; }
die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*" >&2; }

HOST="" USER_OPT="" PORT="" IDENTITY="" REF="HEAD" PYV="3.13" DOCKER_BASE=""
PLATFORM="" BUILDS="" JOBS="" PYTEST_ARGS='Tests -m "not mem"' COMPARE=1
WORKDIR="pillow-simd-test" OUT="bench-results/test-results" KEEP=0
SSH_EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        -u|--user) USER_OPT=$2; shift 2 ;;
        -p|--port) PORT=$2; shift 2 ;;
        -i|--identity) IDENTITY=$2; shift 2 ;;
        -o) SSH_EXTRA+=(-o "$2"); shift 2 ;;
        --ref) REF=$2; shift 2 ;;
        --python-version) PYV=$2; shift 2 ;;
        --docker-base) DOCKER_BASE=$2; shift 2 ;;
        --platform) PLATFORM=$2; shift 2 ;;
        --builds) BUILDS=$2; shift 2 ;;
        -n|--jobs) JOBS=$2; shift 2 ;;
        --pytest-args) PYTEST_ARGS=$2; shift 2 ;;
        --no-compare) COMPARE=0; shift ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -*) echo "unknown option: $1" >&2; usage 2 ;;
        *) [ -z "$HOST" ] || die "more than one host given"; HOST=$1; shift ;;
    esac
done
[ -n "$HOST" ] || usage 2
[[ $PYV =~ ^3\.[0-9]+t?$ ]] || die "--python-version must look like 3.13"
[ -z "$JOBS" ] || [[ $JOBS =~ ^[0-9]+$ ]] || die "--jobs must be a number"
DOCKER_BASE=${DOCKER_BASE:-python:$PYV-slim-trixie}
# WORKDIR goes unquoted into remote commands and is mounted into the
# container: only allow a relative path below the home directory whose
# components start with a letter, digit or '_'.
if ! [[ $WORKDIR =~ ^[A-Za-z0-9_][A-Za-z0-9._-]*(/[A-Za-z0-9_][A-Za-z0-9._-]*)*/?$ ]]; then
    die "--workdir must be a relative path whose parts start with a letter, digit or '_'"
fi
WORKDIR=${WORKDIR%/}
MARKER=".pillow-simd-test"

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
COMMIT=$(git -C "$REPO" rev-parse --short "$REF^{commit}") || die "unknown ref $REF"
if [ "$REF" = "HEAD" ] && [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    log "note: uncommitted changes are not tested (using $COMMIT)"
fi

if [ "$HOST" = "local" ]; then
    CTL_DIR=""
    rsh() { (cd ~ && bash -c "$*"); }
    rbash() { (cd ~ && bash -s <<< "$1"); }
else
    [ -n "$USER_OPT" ] && HOST="$USER_OPT@${HOST#*@}"
    # One SSH connection for all commands.
    CTL_DIR=$(mktemp -d)
    SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$CTL_DIR/%C" -o ControlPersist=600
              -o ServerAliveInterval=30)
    [ -n "$PORT" ] && SSH_OPTS+=(-p "$PORT")
    [ -n "$IDENTITY" ] && SSH_OPTS+=(-i "$IDENTITY")
    SSH_OPTS+=("${SSH_EXTRA[@]+"${SSH_EXTRA[@]}"}")
    rsh() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
    # Runs a script with bash on the host, whatever the login shell is.
    rbash() { ssh "${SSH_OPTS[@]}" "$HOST" bash -s <<< "$1"; }
fi
cleanup() {
    if [ -n "$CTL_DIR" ]; then
        ssh "${SSH_OPTS[@]}" -O exit "$HOST" 2>/dev/null || true
        rm -rf "$CTL_DIR"
    fi
}
trap cleanup EXIT

log "connecting to $HOST"
ARCH=$(rsh uname -m) || die "cannot connect to $HOST"
ENGINE=$(rsh 'command -v docker || command -v podman' | head -1) \
    || die "$HOST needs docker or podman"
ENGINE=${ENGINE##*/}
PLATFORM_ARG=""
if [ -n "$PLATFORM" ]; then
    PLATFORM_ARG="--platform $PLATFORM"
    case "$PLATFORM" in
        linux/arm64*) ARCH=aarch64 ;;
        linux/amd64) ARCH=x86_64 ;;
        *) die "unsupported --platform $PLATFORM (linux/arm64 or linux/amd64)" ;;
    esac
fi
case "$ARCH" in
    aarch64|arm64) DEFAULT_BUILDS="scalar=disable neon=auto" ;;
    x86_64|amd64)
        # SIMD builds only where the CPU can run them (none under qemu).
        DEFAULT_BUILDS="scalar=disable"
        if [ -z "$PLATFORM" ]; then
            for f in sse4_1:sse4=sse4 avx2:avx2=avx2 avx_vnni:avxvnni=avxvnni; do
                if rsh "grep -qw ${f%%:*} /proc/cpuinfo" 2>/dev/null; then
                    DEFAULT_BUILDS="$DEFAULT_BUILDS ${f#*:}"
                fi
            done
        fi ;;
    *) DEFAULT_BUILDS="scalar=disable auto=auto" ;;
esac
BUILDS=${BUILDS:-$DEFAULT_BUILDS}
if [ -z "$JOBS" ]; then
    JOBS=$(rsh nproc 2>/dev/null || echo 2)
    [ "$JOBS" -le 8 ] || JOBS=8
fi
log "host: $ARCH, $ENGINE; builds: $BUILDS; Python $PYV"

# The workdir must be new or one this script made.
rbash "if [ -e $WORKDIR ] && [ ! -e $WORKDIR/$MARKER ]; then
        echo '$WORKDIR exists and was not made by simd_test_remote.sh'; exit 1
    fi" || die "refusing to use $HOST:$WORKDIR (remove it or pass another --workdir)"

IMAGE="pillow-simd-test:py$PYV${PLATFORM:+-${PLATFORM##*/}}"
BUILD_ARGS="--build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy"
log "building the $IMAGE image from $DOCKER_BASE (cached after the first run)"
rbash "set -e
    $ENGINE build -q $PLATFORM_ARG -t $IMAGE $BUILD_ARGS - <<'DOCKERFILE'
FROM $DOCKER_BASE
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev pkg-config ghostscript \
        libjpeg62-turbo-dev zlib1g-dev libpng-dev libtiff-dev libwebp-dev \
        libfreetype-dev liblcms2-dev libopenjp2-7-dev libharfbuzz-dev \
        libfribidi-dev libraqm-dev libimagequant-dev libavif-dev libxcb1-dev \
        libtk8.6 xvfb xauth libjpeg-turbo-progs imagemagick locales \
        libgl1 libegl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'pl_PL ISO-8859-2' >> /etc/locale.gen && locale-gen \
    && pip install --no-cache-dir setuptools wheel pybind11 \
        pytest pytest-timeout pytest-xdist packaging defusedxml olefile \
        psutil ipython markdown2 \
    && (pip install --no-cache-dir --only-binary :all: numpy || true) \
    && (pip install --no-cache-dir --only-binary :all: \
        pyarrow nanoarrow arro3-core arro3-compute PyQt6 || true)
DOCKERFILE
" >/dev/null || die "could not build the image (no access to $DOCKER_BASE, Debian or PyPI?)"

# Runs a script in the container, in $WORKDIR, as the SSH user.
PROXY_ARGS="-e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY -e http_proxy -e https_proxy -e no_proxy"
renv() {
    rsh "mkdir -p $WORKDIR && touch $WORKDIR/$MARKER && cd $WORKDIR && \
        $ENGINE run --rm -i $PLATFORM_ARG --user \$(id -u):\$(id -g) -e HOME=/tmp \
        -v \"\$PWD:/work:Z\" -w /work $PROXY_ARGS $IMAGE bash -s" <<< "set -e; $1"
}

log "copying $COMMIT to $HOST:$WORKDIR"
rsh "rm -rf $WORKDIR && mkdir -p $WORKDIR/src && touch $WORKDIR/$MARKER"
git -C "$REPO" archive --format=tar "$COMMIT" | rsh "tar -x -C $WORKDIR/src"
# The extra test images (depends/install_extra_test_images.sh), if this
# checkout has them: they are ignored by git, so not in the archive.
EXTRA=$(git -C "$REPO" ls-files --others --ignored --exclude-standard Tests/images)
if [ -n "$EXTRA" ]; then
    log "copying the extra test images"
    (cd "$REPO" && printf '%s\n' "$EXTRA" | tar -cf - -T -) | rsh "tar -x -C $WORKDIR/src"
fi
# The harness is taken from the working tree, so older refs can be compared.
rsh "cat > $WORKDIR/simd_bench.py" < "$REPO/checks/simd_bench.py"

STAMP=$(date +%Y%m%d-%H%M%S)
DEST="$OUT/${HOST#*@}-$COMMIT-py$PYV-$STAMP"
mkdir -p "$DEST"
renv "python3 -VV; uname -m" | tee "$DEST/info.txt"
echo "commit $COMMIT, host $HOST ($ARCH), image $DOCKER_BASE${PLATFORM:+ $PLATFORM}" >> "$DEST/info.txt"

XDIST=""
[ "$JOBS" -gt 0 ] && XDIST="-n $JOBS"
FAILED=()
for b in $BUILDS; do
    name=${b%%=*} opt=${b#*=}
    log "building $name (-C simd=$opt)"
    if ! renv "rm -rf build-$name $name && cp -r src build-$name
        cd build-$name
        python3 -m pip install -v --no-deps --no-build-isolation --no-cache-dir \
            -C simd=$opt --target ../$name . > ../build-$name.log 2>&1 \
            || { tail -40 ../build-$name.log; exit 1; }
        grep 'SIMD kernels' ../build-$name.log | sed 's/^ *//'"; then
        log "build of $name failed (log above)"
        FAILED+=("$name (build)")
        continue
    fi
    log "testing $name"
    # Tests run from the source tree for Tests/images, against the build.
    if renv "cd build-$name
        PYTHONPATH=/work/$name python3 -c 'from PIL import Image; print(\"Pillow\", Image.__version__)'
        # A virtual display for the Tk and X11 screen grab tests; Qt draws
        # off screen
        export QT_QPA_PLATFORM=offscreen
        XVFB=\$(command -v xvfb-run >/dev/null && echo 'xvfb-run -a' || true)
        PYTHONPATH=/work/$name \$XVFB python3 -m pytest -p no:cacheprovider -q -rfE $XDIST \
            --junitxml=../junit-$name.xml $PYTEST_ARGS" > "$DEST/test-$name.log" 2>&1; then
        log "$name: $(tail -1 "$DEST/test-$name.log")"
    else
        log "$name FAILED: $(tail -1 "$DEST/test-$name.log")"
        grep -E '^(FAILED|ERROR) ' "$DEST/test-$name.log" | head -20 >&2 || true
        FAILED+=("$name")
    fi
done

BUILT=()
for b in $BUILDS; do
    name=${b%%=*}
    rsh "cd $WORKDIR && cat build-$name.log" > "$DEST/build-$name.log" 2>/dev/null || continue
    rsh "cd $WORKDIR && cat junit-$name.xml" > "$DEST/junit-$name.xml" 2>/dev/null || true
    rsh "test -d $WORKDIR/$name/PIL" && BUILT+=("$name")
done

if [ "$COMPARE" = 1 ] && [ "${#BUILT[@]}" -gt 1 ]; then
    log "comparing the outputs of ${BUILT[*]} (baseline ${BUILT[0]})"
    ARGS=(--baseline "${BUILT[0]}" --set full --rounds 1 --size 67x23 --check)
    for name in "${BUILT[@]}"; do ARGS+=(--variant "$name=$name"); done
    renv "python3 -I simd_bench.py$(printf ' %q' "${ARGS[@]}")" > "$DEST/compare.txt" 2>&1 \
        || log "warning: the comparison did not run (see $DEST/compare.txt)"
    sed -n '/^Output differences/,$p' "$DEST/compare.txt" >&2
fi

if [ "$KEEP" = 0 ]; then
    rsh "rm -rf $WORKDIR"
else
    log "kept $HOST:$WORKDIR"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    log "failed: ${FAILED[*]}; logs in $DEST"
    exit 1
fi
log "all builds passed; logs in $DEST"

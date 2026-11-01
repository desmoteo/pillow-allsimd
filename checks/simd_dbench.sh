#!/usr/bin/env bash
# Builds and runs this checkout in the benchmark image, with the same
# Python, compiler and default -O3 as the published wheels and benchmark
# results, so a local A/B measures the code the users get (a local
# Debian, Ubuntu or conda Python builds at -O2, where GCC unrolls and
# vectorizes less; see design/OPTIMIZATION_METHOD.md).
#
#   checks/simd_dbench.sh build NAME SIMD [OPT]
#       the working tree (HEAD plus uncommitted changes) with -C simd=SIMD
#       into $DBUILDS/NAME; OPT, e.g. -O2, replaces the image's -O3 (a
#       CFLAGS variable replaces Python's flags, so the whole set is given)
#   checks/simd_dbench.sh run NAME SCRIPT [ARGS...]
#       runs SCRIPT with that build on PYTHONPATH, pinned to CPU $CPU
#       (default 2); NAME "upstream" is the Pillow 13.0.0 wheel of the image
#
# The image, pillow-simd-perf, is built from the Dockerfile of the
# benchmark repository (https://github.com/desmoteo/pillow-allsimd-perf).
# DBUILDS defaults to bench-results/dbuilds, which git ignores. The
# script's directory is mounted so that it can import neighbours.
set -euo pipefail
cd "$(dirname "$0")/.."
DBUILDS=${DBUILDS:-$(pwd)/bench-results/dbuilds}
CPU=${CPU:-2}
IMAGE=${IMAGE:-pillow-simd-perf}
E="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $DBUILDS:/dbuilds"
cmd=$1 NAME=$2
shift 2
case "$cmd" in
build)
    SIMD=$1 OPT=${2:-}
    SRC=$DBUILDS/src-$NAME
    rm -rf "$SRC" "$DBUILDS/$NAME"
    mkdir -p "$SRC"
    git archive HEAD | tar -x -C "$SRC"
    { git diff HEAD --name-only --diff-filter=AM; git ls-files --others --exclude-standard src Tests checks; } \
        | tar -cf - -T - | tar -x -C "$SRC"
    FLAGS=()
    [ -n "$OPT" ] && FLAGS=(-e "CFLAGS=-fno-strict-overflow -Wsign-compare -DNDEBUG -g $OPT -Wall")
    $E "${FLAGS[@]}" -w /dbuilds/src-$NAME "$IMAGE" python3 -m pip install -q --no-deps \
        --no-build-isolation -C simd="$SIMD" --target /dbuilds/$NAME . > "$DBUILDS/build-$NAME.log" 2>&1 \
        || { tail -30 "$DBUILDS/build-$NAME.log"; exit 1; }
    rm -rf "$SRC"
    echo "built $DBUILDS/$NAME"
    ;;
run)
    SCRIPT=$(realpath "$1")
    shift
    P=/dbuilds/$NAME
    [ "$NAME" = upstream ] && P=/opt/variants/pillow-13.0.0
    $E --cpuset-cpus="$CPU" -v "$(dirname "$SCRIPT"):/s" -e PYTHONPATH=$P "$IMAGE" \
        python3 "/s/$(basename "$SCRIPT")" "$@"
    ;;
*)
    echo "usage: $0 build NAME SIMD [OPT] | run NAME SCRIPT [ARGS...]" >&2
    exit 2
    ;;
esac

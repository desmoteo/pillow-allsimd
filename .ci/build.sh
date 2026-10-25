#!/bin/bash

set -e

make clean
make install-coverage SIMD="$SIMD"

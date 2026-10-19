#!/bin/bash
# Builds the encode-score binary: GPU parts in the CUDA container (build-gpu.sh), then the host program with the system
# compiler against the system FFmpeg. Usage: native/build.sh BUILD_DIR
set -euo pipefail
B=$(realpath "$1"); HERE=$(realpath "$(dirname "$0")/..")
"$HERE/scripts/build-gpu.sh" "$B"
VMAF=$B/prefix
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$HERE/pyproject.toml")
g++ -std=c++20 -O2 -DENCODE_SCORE_VERSION="\"$VERSION\"" -Wall -Wno-missing-field-initializers -I"$B" -I/usr/local/cuda-13.1/include -I"$VMAF/include" $(pkg-config --cflags libavformat) \
    "$HERE"/src/*.cpp -o "$B/encode-score" \
    "$VMAF/lib/x86_64-linux-gnu/libvmaf.a" $(pkg-config --libs libavformat libavcodec libavutil) -lcuda -ldl -lpthread -lm
echo "built $B/encode-score"

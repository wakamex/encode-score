#!/bin/bash
# Builds the encode-score binary: GPU parts in the CUDA container (build-gpu.sh), then the host program with the system
# compiler against the system FFmpeg. Usage: native/build.sh BUILD_DIR
set -euo pipefail
B=$(realpath "$1"); HERE=$(realpath "$(dirname "$0")/..")
"$HERE/scripts/build-gpu.sh" "$B"
VMAF=$B/prefix
g++ -std=c++26 -O2 -Wall -Wno-missing-field-initializers --embed-dir="$B" -I"$B" -I/usr/local/cuda-13.1/include -I"$VMAF/include" $(pkg-config --cflags libavformat) \
    "$HERE"/src/*.cpp -o "$B/encode-score" \
    "$VMAF/lib/x86_64-linux-gnu/libvmaf.a" $(pkg-config --libs libavformat libavcodec libavutil) -lcuda -ldl -lpthread -lm
echo "built $B/encode-score"

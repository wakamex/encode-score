#!/bin/bash
# Builds the GPU parts inside the CUDA 13.1 container, because the host's compiler and C library are newer than CUDA 13.1
# accepts. Usage: native/build-gpu.sh BUILD_DIR. native/build.sh runs it and then links the program on the host.
set -euo pipefail
B=$(realpath "$1"); mkdir -p "$B"; HERE=$(realpath "$(dirname "$0")/..")
podman run --rm -v "$B:/b:Z" -v "$HERE:/n:Z,ro" docker.io/nvidia/cuda:13.1.1-devel-ubuntu24.04 bash /n/scripts/container-build.sh 2>&1 | grep -E "error|built|warning" || true
[ -s "$B/kernels.fatbin" ]

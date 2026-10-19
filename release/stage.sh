#!/bin/bash
# Builds the release assets of the checked-out commit into OUT_DIR (see build-linux.sh), in the CUDA 13.1 container on
# Ubuntu 22.04, so the program needs glibc 2.35 or newer. Uses podman, or docker where podman is absent. WORK_DIR keeps
# the downloads and builds between runs (default: a temporary folder).
#   release/stage.sh OUT_DIR [WORK_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(mkdir -p "${1:?output directory}" && cd "$1" && pwd)
work=${2:-$(mktemp -d)}
mkdir -p "$work"
engine=$(command -v podman || command -v docker)
"$engine" run --rm -v "$work:/b:Z" -v "$here:/n:Z,ro" -v "$out:/out:Z" docker.io/nvidia/cuda:13.1.1-devel-ubuntu22.04 \
    bash /n/release/build-linux.sh
ls -l "$out"

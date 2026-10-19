#!/bin/bash
# Builds the release assets of the checked-out commit into OUT_DIR (see build-linux.sh), in the CUDA 13.1 container on
# Ubuntu 22.04, pinned by the digest of its x86-64 image, so the program needs glibc 2.35 or newer. Uses podman, or
# docker where podman is absent. WORK_DIR keeps the downloads and builds between runs (default: a temporary folder).
#   release/stage.sh OUT_DIR [WORK_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(mkdir -p "${1:?output directory}" && cd "$1" && pwd)
work=${2:-$(mktemp -d)}
mkdir -p "$work"
engine=$(command -v podman || command -v docker)
image=docker.io/nvidia/cuda:13.1.1-devel-ubuntu22.04@sha256:f3e8766fa1eee14fe197d900812aef31c36402b58dbcfcadeed11e0d72fe4374
"$engine" run --rm -v "$work:/b:Z" -v "$here:/n:Z,ro" -v "$out:/out:Z" "$image" \
    bash /n/release/build-linux.sh
(cd "$out" && sha256sum -- *)

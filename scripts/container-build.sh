#!/bin/bash
# Runs inside the CUDA 13.1 container (see build-gpu.sh): builds libvmaf with its CUDA backend as a static library, and
# the kernels as a fatbin, into /b.
set -euo pipefail
apt-get update -qq >/dev/null && apt-get install -y -qq meson ninja-build nasm pkg-config git xxd >/dev/null 2>&1
# the libvmaf and NVIDIA codec header commits the program was tested with. libvmaf.patch makes libvmaf's motion kernel
# for 10-bit pictures round with an integer shift instead of a double-precision pow() per pixel, which made it take a
# quarter of all GPU time; the result is the same.
VMAF_COMMIT=229f2478ab53ae0fd6b07328802294ba15fd0cdd
NVH_COMMIT=71acc268a541f7d7f33cf0387229e3015cfb4977
[ -d /b/nvh ] || git clone -q https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /b/nvh
git -C /b/nvh checkout -q "$NVH_COMMIT"
make -s -C /b/nvh install PREFIX=/usr >/dev/null
[ -d /b/vmaf ] || git clone -q https://github.com/Netflix/vmaf.git /b/vmaf
# applied once, so that a rebuild leaves the patched files alone and ninja does not recompile them
[ "$(git -C /b/vmaf rev-parse HEAD)" = "$VMAF_COMMIT" ] && git -C /b/vmaf apply --reverse --check /n/scripts/libvmaf.patch 2>/dev/null || {
  git -C /b/vmaf checkout -q -f "$VMAF_COMMIT" && git -C /b/vmaf apply /n/scripts/libvmaf.patch; }
cd /b/vmaf/libvmaf
[ -f build/build.ninja ] || meson setup build --buildtype=release -Denable_cuda=true -Denable_tests=false -Denable_docs=false \
  -Dbuilt_in_models=true -Denable_float=false -Ddefault_library=static --prefix=/b/prefix >/dev/null
ninja -C build >/dev/null && ninja -C build install >/dev/null
# CAMBI's reciprocal table, copied exactly from libvmaf so the GPU port computes the same c-values
awk '/static const float reciprocal_lut\[CAMBI_RECIPROCAL_LUT_SIZE\]/{on=1; sub(/static const float/, "__constant__ float")} on{print} on&&/};/{exit}' \
  /b/vmaf/libvmaf/src/feature/cambi.h | sed 's/CAMBI_RECIPROCAL_LUT_SIZE/4226/' > /b/cambi_lut.h
# native code for the GPU generations CUDA 13 supports, and PTX for newer ones
ARCHS=""
for a in 75 80 86 89 90 100 120; do ARCHS="$ARCHS -gencode arch=compute_$a,code=sm_$a"; done
nvcc -I/b -fatbin -O3 -std=c++17 $ARCHS -gencode arch=compute_75,code=compute_75 /n/src/kernels.cu -o /b/kernels.fatbin
echo gpu parts built

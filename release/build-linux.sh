#!/bin/bash
# Runs inside docker.io/nvidia/cuda:13.1.1-devel-ubuntu22.04 (see stage.sh), with the source at /n (read only) and a
# work folder at /b: builds libvmaf and the kernels as scripts/container-build.sh does, then dav1d and an LGPL build of
# FFmpeg as static libraries, and links encode-score against them all, with only glibc and the NVIDIA driver's
# libcuda.so.1 left to load at run time. Packs into /out the program with the license texts of everything in it,
# encode-score-vVERSION-x86_64-linux.tar.gz, and what the LGPL asks to accompany it, FFmpeg's release tarball and the
# scripts that build it, encode-score-vVERSION-ffmpeg-source.tar.gz.
set -euo pipefail
bash /n/scripts/container-build.sh
apt-get install -y -qq curl xz-utils >/dev/null 2>&1

FFMPEG_VERSION=9.0.2
FFMPEG_SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
DAV1D_VERSION=1.5.4
DAV1D_SHA256=686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5
P=/b/static  # where dav1d and FFmpeg are installed
mkdir -p /b/sources /b/deps
fetch() {  # URL SHA256: downloads into /b/sources unless already there, and checks it
    local file=/b/sources/$(basename "$1")
    [ -f "$file" ] || curl -sSfL -o "$file" "$1"
    echo "$2  $file" | sha256sum -c --quiet
}

# dav1d decodes AV1 on the processor (FFmpeg's own AV1 decoder only drives the GPU's decoder)
fetch "https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz" "$DAV1D_SHA256"
if [ ! -f "$P/lib/pkgconfig/dav1d.pc" ]; then
    rm -rf "/b/deps/dav1d-$DAV1D_VERSION" && tar -xJf "/b/sources/dav1d-$DAV1D_VERSION.tar.xz" -C /b/deps
    meson setup "/b/deps/dav1d-$DAV1D_VERSION/build" "/b/deps/dav1d-$DAV1D_VERSION" --buildtype=release --default-library=static \
        --prefix="$P" --libdir=lib -Denable_tools=false -Denable_tests=false >/dev/null
    ninja -C "/b/deps/dav1d-$DAV1D_VERSION/build" install >/dev/null
fi

# FFmpeg with only what encode-score reads: local files in the usual containers, the decoders of the formats encodes
# come in, NVDEC for those the GPU decodes, and no GPL or nonfree parts
fetch "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz" "$FFMPEG_SHA256"
if [ ! -f "$P/lib/libavformat.a" ]; then
    rm -rf "/b/deps/ffmpeg-$FFMPEG_VERSION" && tar -xJf "/b/sources/ffmpeg-$FFMPEG_VERSION.tar.xz" -C /b/deps
    cd "/b/deps/ffmpeg-$FFMPEG_VERSION"
    PKG_CONFIG_PATH="$P/lib/pkgconfig" ./configure --prefix="$P" --enable-static --disable-shared --enable-pic \
        --disable-autodetect --disable-everything --disable-programs --disable-doc --disable-network \
        --disable-avdevice --disable-avfilter --disable-swscale --disable-swresample \
        --enable-protocol=file \
        --enable-demuxer=matroska,mov,mpegts,ivf,avi,obu,h264,hevc,m4v,yuv4mpegpipe \
        --enable-parser=h264,hevc,av1,vp9,mpeg4video,mpegvideo \
        --enable-decoder=h264,hevc,av1,libdav1d,vp9,mpeg2video,mpeg4,ffv1,rawvideo \
        --enable-libdav1d --enable-cuda --enable-ffnvcodec --enable-nvdec \
        --enable-hwaccel=h264_nvdec,hevc_nvdec,av1_nvdec,vp9_nvdec,mpeg2_nvdec,mpeg4_nvdec >/b/deps/ffmpeg-configure.log
    make -s -j"$(nproc)" >/b/deps/ffmpeg-build.log 2>&1 && make -s install >/dev/null
fi

# the program, with the C++ runtime linked in too
cd /b
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' /n/pyproject.toml)
g++ -std=c++20 -O2 -DENCODE_SCORE_VERSION="\"$VERSION\"" -Wall -Wno-missing-field-initializers -I/b -I/usr/local/cuda/include \
    -I/b/prefix/include -I"$P/include" /n/src/*.cpp -o /b/encode-score -static-libstdc++ -static-libgcc \
    /b/prefix/lib/x86_64-linux-gnu/libvmaf.a $(PKG_CONFIG_PATH="$P/lib/pkgconfig" pkg-config --static --libs libavformat libavcodec libavutil) \
    -L/usr/local/cuda/lib64/stubs -lcuda -ldl -lpthread -lm

# it loads nothing but glibc and the driver, and starts (with the toolkit's stub standing in for the driver)
needed=$(readelf -d /b/encode-score | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | sort | tr '\n' ' ')
for lib in $needed; do
    case $lib in libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|libcuda.so.1|ld-linux-x86-64.so.2) ;; *) echo "unexpected library $lib" >&2; exit 1 ;; esac
done
mkdir -p /b/stub && cp /usr/local/cuda/lib64/stubs/libcuda.so /b/stub/libcuda.so.1
LD_LIBRARY_PATH=/b/stub /b/encode-score --version | grep -qx "encode-score $VERSION"
LD_LIBRARY_PATH=/b/stub /b/encode-score --help | grep -q -- --offset
echo "built encode-score $VERSION, loading $needed"

# the release assets
name=encode-score-v$VERSION-x86_64-linux
rm -rf "/b/$name" && mkdir -p "/b/$name/licenses"
cp /b/encode-score /n/README.md /n/LICENSE "/b/$name/"
cp /b/vmaf/LICENSE "/b/$name/licenses/libvmaf-LICENSE"
cp "/b/deps/ffmpeg-$FFMPEG_VERSION/COPYING.LGPLv2.1" "/b/$name/licenses/FFmpeg-COPYING.LGPLv2.1"
cp "/b/deps/dav1d-$DAV1D_VERSION/COPYING" "/b/$name/licenses/dav1d-COPYING"
sed -n '1,/\*\//p' /b/nvh/include/ffnvcodec/dynlink_loader.h > "/b/$name/licenses/nv-codec-headers-LICENSE"
mkdir -p /out && tar -czf "/out/$name.tar.gz" -C /b "$name"
src=encode-score-v$VERSION-ffmpeg-source
rm -rf "/b/$src" && mkdir -p "/b/$src/scripts"
cp "/b/sources/ffmpeg-$FFMPEG_VERSION.tar.xz" "/b/$src/"
cp /n/release/build-linux.sh /n/scripts/container-build.sh "/b/$src/scripts/"
cat > "/b/$src/README.txt" <<TXT
encode-score $VERSION links FFmpeg $FFMPEG_VERSION's libavformat, libavcodec and libavutil statically, under the GNU
Lesser General Public License version 2.1 or later (COPYING.LGPLv2.1 in the tarball).

ffmpeg-$FFMPEG_VERSION.tar.xz is FFmpeg's unmodified release tarball. scripts/build-linux.sh configures and builds it
(see its ./configure line) and links encode-score against it, inside docker.io/nvidia/cuda:13.1.1-devel-ubuntu22.04;
scripts/container-build.sh, which it runs first, builds libvmaf and encode-score's GPU kernels.

To relink encode-score with a modified FFmpeg, check out https://github.com/wakamex/encode-score at v$VERSION, change
the FFmpeg tarball and checksum at the top of release/build-linux.sh to your own, and run release/stage.sh.
TXT
tar -czf "/out/$src.tar.gz" -C /b "$src"
echo "packed $name.tar.gz and $src.tar.gz"

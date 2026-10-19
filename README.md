# encode-score

encode-score compares video encodes with their source and says what a viewer would notice: lost texture, banding and lost grain, as well as the overall fidelity that [Video Multi-Method Assessment Fusion (VMAF)](https://github.com/Netflix/vmaf) measures. It scores every frame of a whole film on the GPU in one pass over the files, gives each minute and the whole run a score from 0 to 100, and prints the parts that make it up, so encodes can be ranked and the reason for the ranking is visible.

VMAF alone is a poor judge of small encodes of grainy or dark material. It scores brightness only, it rates a soft, grain-free picture as nearly perfect, and it does not see banding. In the film trial below, a 9.3 GB public encode and our own encode scored within 0.1 VMAF of each other on the hardest scene, while one of them banded visibly and had lost three quarters of the film's grain.

## Installing

encode-score runs on an NVIDIA GPU (Turing or newer) and Linux. It decodes each file with the GPU's video decoder or on the processor, whichever makes the run faster (see Speed), computes VMAF with libvmaf's CUDA backend, and runs its own measures as CUDA kernels.

Each [release](https://github.com/wakamex/encode-score/releases) has a Linux x86-64 build, `encode-score-vX.Y.Z-x86_64-linux.tar.gz`, which needs glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or later) and an NVIDIA driver for CUDA 13, release 580 or newer; it was tested with 615. [FFmpeg](https://ffmpeg.org/) and libvmaf are built into it, so nothing else needs installing:

```
tar -xzf encode-score-vX.Y.Z-x86_64-linux.tar.gz
encode-score-vX.Y.Z-x86_64-linux/encode-score --help
```

The archive's `licenses/` folder holds the license texts of what is built in. FFmpeg is linked statically under the GNU Lesser General Public License (LGPL), so each release also has `encode-score-vX.Y.Z-ffmpeg-source.tar.gz`, FFmpeg's release tarball with the scripts that build it and link it in. Check a downloaded archive with:

```
gh attestation verify ASSET --repo wakamex/encode-score --signer-workflow wakamex/release-actions/.github/workflows/binary-release.yml
gh release verify-asset vX.Y.Z ASSET --repo wakamex/encode-score
```

## Building

Building needs:

- the FFmpeg development libraries (libavformat, libavcodec, libavutil; developed against FFmpeg 8.1 and released with 9.0) and a C++20 compiler;
- the CUDA driver API headers, from the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 13;
- [Podman](https://podman.io/) and the `nvidia/cuda:13.1.1-devel-ubuntu24.04` image, in which the GPU code (libvmaf with CUDA, and the kernels) is compiled, because CUDA 13.1 does not accept the newer host compilers and C library of current Linux distributions.

```
scripts/build.sh build
build/encode-score --help
```

The build fetches libvmaf and the NVIDIA codec headers into the build folder the first time. libvmaf comes from [our fork](https://github.com/wakamex/vmaf), at a commit that adds two fixes to Netflix's: libvmaf's motion kernel for 10-bit pictures rounds with an integer shift instead of a double-precision `pow()` per pixel, which takes it from 0.61 to 0.08 ms per 1080p frame on an RTX 3090 with the same results, and VIF and motion clear their sums on the stream that adds to them. libvmaf cleared them on another stream, so the clearing could land after some of a frame's sums were added, and VMAF varied from run to run by up to 0.4 on some frames.

## Usage

```
encode-score SOURCE ENCODE [ENCODE ...] [--start SECONDS] [--seconds N] [--offset auto|SECONDS] [--segment 60] [--json FILE]
```

The source is decoded once and every encode is scored against it in the same pass. Without `--seconds` the run goes to the end of the shortest file. The program prints, for each encode, the whole run and its five worst segments (60 seconds each by default), and `--json FILE` writes every figure for the run and each segment, and the per-frame VMAF, CAMBI and PSNR.

The source's CAMBI depends on the source alone, so it is kept per frame in `~/.cache/encode-score` (or `$XDG_CACHE_HOME/encode-score`), and a later run against the same source reads the frames it already has, from any start point, instead of computing them again. A source is recognised by its size and its first and last mebibyte, together with the crop and bit depth of the run, so a renamed or moved file keeps its cache. Deleting the folder clears it.

The files are lined up automatically: by default the program matches thumbnails of the first 8 seconds of each encode against the source within 10 seconds either side, to the frame, and refuses to score an encode that does not match. `--offset auto=SECONDS` searches around a guess, such as the point of a film a clip was cut from, and `--offset SECONDS` sets the offset directly. When the pictures differ in size, for example because one has its black bars cropped, all files are cropped to their common area. Frames are paired by position, so the files must have the same frame rate.

The VMAF model is the 4K model for pictures 3840 pixels wide or more and the standard model, which assumes a 1080p screen, below that (`--model` picks another). Input must be 4:2:0 at 8 or 10 bits. When every file is 8-bit, VMAF, CAMBI and PSNR are computed at 8 bits, as libvmaf does; otherwise at 10 bits, with 8-bit files shifted up.

## What the score measures

The score is the weighted mean of four parts, each from 0 to 100, where 100 means no visible difference from the source.

| Part | Weight | What it measures | 100 means |
|---|---|---|---|
| Fidelity | 0.30 | VMAF, mean over the frames | identical brightness detail |
| Texture | 0.30 | Share of the source's lasting texture that the encode reproduces: half the median window, half the worst tenth of windows | all lasting texture kept, everywhere |
| Banding | 0.25 | [Contrast Aware Multiscale Banding Index (CAMBI)](https://github.com/Netflix/vmaf/blob/master/resource/doc/papers/CAMBI_PCS2021.pdf) of the encode above the source's, mean over the frames | no more banding than the source; 0 at 5 points more |
| Grain | 0.15 | Amount of fine detail (1 to 2 pixels) over the whole picture against the source's | the same amount; a point lost per percent above or below |

The weights are a judgement, not fitted to viewer ratings, so read the parts beside the score. A difference of about 1 VMAF is hard to see, while a CAMBI rise of 2 or more is visible as steps in gradients.

Lasting texture is the measure that separates real detail from grain. Grain changes every frame and texture does not, so for each frame the program averages the source's 6 frames before and 6 after, leaving the frame itself out, which keeps the texture and averages the grain down. It counts only the 20% of 16 by 16 pixel windows with the most directional detail (edges, fabric, faces), and of those only the windows where two interleaved halves of the neighbouring frames agree. That choice uses neither the measured frame of the source nor the encode, so a chance resemblance between two frames of grain cannot pass for texture. Windows that move during those frames are left out, and the first and last 6 frames of a run have no texture figure. The figure is the share of that texture the encode's frame reproduces, from a projection of one onto the other, so new grain or artifacts cannot add to it.

The detailed output also gives peak signal-to-noise ratio (PSNR) per color plane, the amount and the matching share of fine and mid detail (3 to 8 pixels) over the whole picture and over the structured windows, and the mean color values of both files. Medians and tenth percentiles of per-window figures are read from 0.05-point bins.

## Accuracy

The program computes the same figures as the reference implementations:

| Figure | Checked against | Largest difference per frame |
|---|---|---|
| VMAF | libvmaf on the processor, 240 frames of a 4K film and generated 8-bit and 10-bit video | 0.0006 |
| CAMBI of the encode and of the source | libvmaf on the processor, including a scene that bands visibly (CAMBI 3.6) | 0.0000003 |
| PSNR | libvmaf on the processor | 0.001 dB |
| Detail, texture, grain and color | the numpy implementation in `reference/`, on every frame of 40 frames of a 4K film | totals 0.001 points, medians within one bin |

CAMBI is a port of libvmaf's `cambi.c`; it counts the same windows with a different algorithm and uses libvmaf's own luminance tables and reciprocal table, so its c-values are bit-identical. Its scores differ from libvmaf's only in the order floating-point sums are added.

## Speed

The program works on two frames at once: while the GPU computes the measures of one frame, the processor adds up the figures of the frame before. Decoding, the GPU's work and the processor's then overlap, and a run goes as fast as the slowest of them.

Decoding is often the slowest. The GPU's video decoder takes turns among all the files of a run, so for runs of 5 minutes or more the program measures how fast each file decodes on the GPU's decoder and on the processor, and gives files to the processor when that makes decoding faster. A 1080p [H.264](https://en.wikipedia.org/wiki/Advanced_Video_Coding) source, for example, decodes faster on a 16-core processor than on the GPU's decoder, which is then left to the encodes. If the run then turns out slower than decoding everything on the GPU's decoder would allow, because other programs come to load the processor or because the GPU's other work is the limit and the processor's decoding slows the program's feeding of it, those files go back to the GPU's decoder from their next frame, and the program says so.

On an RTX 3090 with a 16-core Ryzen 9 3950X, with the source's CAMBI cached (computing it as well takes the first run to 615-616):

| Run | Frames per second |
|---|---:|
| 1080p H.264 source (a streaming service's episode) against an HEVC encode, 5 minutes | 673-675 |
| 4K [High Efficiency Video Coding (HEVC)](https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding) remux of Dune: Part Two against an AV1 encode, 1 minute | 173-175 |
| The same remux against two AV1 encodes in one pass, 1 minute | 115-116 |
| The whole 44-minute episode against its encode, the source's CAMBI computed too | 579 (112 s) |
| 10 minutes of the film's remux against an AV1 encode of the whole film, the source's CAMBI computed too | 163 |

Before scoring, finding the offset decodes 28 seconds of the source and 8 seconds of each encode, all side by side, which takes about 5 seconds at 4K with two encodes (`--offset SECONDS` skips it), and measuring the decoding rates for a run of 5 minutes or more takes about 2 seconds at 4K. Scenes with much banding are slower, because CAMBI counts every banded pixel's window, and other work on the GPU slows a run in proportion. A run against a source scored before skips the source's CAMBI for the frames already in the cache. The same measures as 6 second clips in the earlier Python version would have taken about 27 hours for one film.

## Use case: choosing settings to re-encode a 4K film

The question was whether a 68 GB 4K Blu-ray remux of Dune: Part Two could be replaced by an AV1 encode under about 10 GB with no visible loss on a 65 inch OLED, and which encoder settings to use. The benchmark to beat was a widely shared 9.3 GB AV1 encode of the film (video at 6.6 Mbps). Candidates were encoded with [SVT-AV1](https://gitlab.com/AOMediaCodec/SVT-AV1) through FFmpeg and scored against the remux, on 6 second clips of the scenes that compress worst: dim, grainy close-ups of faces at 82.8 and 140.7 minutes, and bright amber smoke at 71 seconds.

The settings were chosen one at a time, each at matched bitrate: encode at two quality settings that bracket the benchmark's bitrate, score both, and compare at the same bytes. The figures settled each choice:

| Choice | Finding | Decision |
|---|---|---|
| Film grain synthesis | Strip the grain and let the decoder regenerate it: banding falls from 2 to 4 CAMBI to 0, at no cost in bytes | on, level 8, encoder denoise off |
| Encoder speed preset | Texture plateaus from preset 7 down; at 4K preset 5 beats 6 by about 4 points on the worst tenth of windows, and 4 adds nothing | preset 5 |
| Denoising the source first | Six denoisers, up to a motion-compensated one masked to protect texture, tied plain encoding on texture at best and lost grain and VMAF | no denoising |
| Quantization matrices | 4 to 7% fewer bits at the same quality setting in two of three scenes, with texture, banding and VMAF unchanged | on |
| Variance boost, sharpness | No gain at equal bitrate | off |
| Quality setting (CRF) | Across 20 random samples of the whole film, CRF 10 averages VMAF 96.3 against the benchmark's 94.9 on the same frames | CRF 10 |

The resulting recipe is preset 5, `crf=10:film-grain=8:film-grain-denoise=0:enable-cdef=0:enable-qm=1`, with [Dolby Vision](https://www.dolby.com/technologies/dolby-vision/) metadata carried over, which cost nothing measurable. The whole film came out at 11.9 GB of video (9.6 Mbps).

This is how the recipe scored against the benchmark encode on the 140.7 minute scene. Each part's contribution is its weight times the difference in that part, and the contributions add up to the difference in score:

| Score part | Weight | Recipe, CRF 10 | Benchmark encode | Result | Contribution to the recipe's lead (points) |
|---|---|---|---|---|---|
| Fidelity (VMAF) | 0.30 | 95.7 | 95.6 | tie | +0.03 |
| Texture | 0.30 | 56.7 | 55.8 | tie | +0.27 |
| Banding (CAMBI 0.00 against 3.83) | 0.25 | 100.0 | 24.0 | recipe wins | +18.99 |
| Grain (93% of the source's against 24%) | 0.15 | 92.5 | 24.0 | recipe wins | +10.28 |
| Score | | 84.6 | 55.0 | recipe wins | +29.56 |

The recipe used 2.4 Mbps in that clip against the benchmark's 3.1 Mbps. On the 82.8 minute scene it scored 78.8 against 52.1, and there it also kept more lasting texture in the median window (67% against 46%), at twice the benchmark's bitrate in that clip.

## Use case: choosing encoder settings for a Twitch stream

A live stream is a fixed-bitrate problem. [Twitch](https://www.twitch.tv/) takes H.264 at a capped bitrate with a keyframe every 2 seconds (check the current limits in Twitch's [broadcasting guidelines](https://help.twitch.tv/s/article/broadcasting-guidelines)), and the encoder must keep up in real time beside the game. So the bitrate is set, and the question is which encoder, preset, resolution and frame rate look best at it. encode-score answers that from a recording of your own game, since what compresses badly differs from game to game.

1. Record a master. In [OBS Studio](https://obsproject.com/), set the recording to the stream's resolution and frame rate with recording quality "Lossless", and record a few minutes that include the hard parts of your game: fast camera turns, foliage or particles, dark scenes with gradients, and a static menu.

2. Encode the candidates from the master the way the stream would, at the stream bitrate in constant-bitrate mode with 2 second keyframes. For example, at 6 Mbps and 60 frames per second:

   ```
   for c in "x264-veryfast|-c:v libx264 -preset veryfast" "x264-medium|-c:v libx264 -preset medium" \
            "nvenc-p5|-c:v h264_nvenc -preset p5 -tune hq" "nvenc-p7|-c:v h264_nvenc -preset p7 -tune hq"; do
     ffmpeg -i master.mkv ${c#*|} -b:v 6M -maxrate 6M -bufsize 6M -g 120 -pix_fmt yuv420p "${c%%|*}.mp4"
   done
   ```

3. Score them all against the master in one pass:

   ```
   encode-score master.mkv x264-veryfast.mp4 x264-medium.mp4 nvenc-p5.mp4 nvenc-p7.mp4 --segment 10 --json stream.json
   ```

4. Rank by the score of the whole run, then read the parts and the worst segments. In game footage, banding in dark scenes and skies and lost texture in foliage and fast motion are the faults that differ most between encoders. Check each encode's real bitrate too (`ffprobe -show_entries format=bit_rate`): encoders do not all fill a constant-bitrate budget equally (in a synthetic test, x264 delivered 5.1 Mbps of a 6 Mbps budget and NVENC 5.9), so a candidate that undershoots is giving away quality you are paying for in bandwidth.

5. Keep only candidates that are fast enough. FFmpeg prints `speed=` while it encodes. A software encoder needs well above 1x on its own to keep up while the game uses the same processor, so measure it with the game running. Hardware encoders such as [NVENC](https://developer.nvidia.com/video-codec-sdk) barely load the processor.

To compare resolutions or frame rates, put each candidate back into the master's format before scoring, as a viewer's player would, because the program compares pictures of the same size frame by frame. Scale a 936p or 720p encode up to the master's size with `-vf scale=1920:1080:flags=bicubic`, and repeat frames of a 30 frames per second encode up to 60 with `-vf fps=60`, writing each to a lossless file such as `-c:v libx264 -qp 0`. Then score that file. Lower resolution trades sharpness for fewer artifacts at a fixed bitrate, and this shows which way the trade goes for your game.

Two parts of the score mean something slightly different for game footage. The grain part assumes the source's finest detail is film grain; rendered games usually have none, so there it measures how much fine detail and sharpness survive. The weights were chosen for film, so for a stream, weigh the parts yourself. Banding and texture usually matter most to viewers.

## Limitations

- The score has not been fitted to viewer ratings. It ranks encodes by measured faults, and the weights decide how those faults trade against each other.
- VMAF is trained on standard dynamic range pictures. For HDR it is a relative measure of drift from the source, not an absolute rating.
- Lasting texture needs 13 neighbouring frames with little motion; a segment with fewer than 100 qualifying windows is scored without the texture part, and the worst tenth of windows is noisy in very dark scenes.
- Encodes scored in one pass must start at the same point of the source; score encodes cut from different points in separate runs.

## Development

```
uv run --locked python -m unittest discover -s tests
```

The tests generate their own video with FFmpeg. They check the program frame by frame against FFmpeg's libvmaf filter for VMAF, CAMBI and PSNR at 8 and 10 bits; check the detail, texture, grain and color figures against the numpy implementation in `reference/`; check that the offset of an encode starting a second later is found to the frame and that a different picture is refused; and check that a source decoded on the processor scores exactly as when decoded on the GPU, and that two encodes in one pass score as they do alone. They keep their CAMBI cache in a temporary folder and check that a second run reads the source's CAMBI from it unchanged. The tests of the program are skipped without an NVIDIA GPU and the built program (`build/encode-score`, or the path in `ENCODE_SCORE_BIN`).

`reference/encode_score` is the numpy implementation of the measures, kept as the specification the GPU kernels are tested against; its own tests are the `test_reference_*` files.

## License

encode-score is MIT; see [LICENSE](LICENSE). The build fetches and links [libvmaf](https://github.com/Netflix/vmaf) (BSD+Patent), from our fork, and uses NVIDIA's [codec headers](https://github.com/FFmpeg/nv-codec-headers) (MIT); CAMBI's reciprocal table is copied from libvmaf at build time. The release builds also link FFmpeg (LGPL 2.1 or later, built without its GPL and nonfree parts) and [dav1d](https://code.videolan.org/videolan/dav1d) (BSD-2-Clause) statically, and carry each of these licenses, and those of the third-party code libvmaf compiles in, in their `licenses/` folder.

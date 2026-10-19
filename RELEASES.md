# Releases

encode-score is released as a prebuilt Linux archive on GitHub Releases, built and published by `.github/workflows/publish.yml` from a `v*` tag. The version source is `version` in `pyproject.toml`, which the build compiles in (`encode-score --version`).

## Assets

| Asset | File |
|---|---|
| Linux x86-64, glibc 2.35 or newer | `encode-score-vX.Y.Z-x86_64-linux.tar.gz` |
| FFmpeg's source and the scripts that build it | `encode-score-vX.Y.Z-ffmpeg-source.tar.gz` |

`release/stage.sh OUT_DIR [WORK_DIR]` builds both in `docker.io/nvidia/cuda:13.1.1-devel-ubuntu22.04` with Podman, or Docker where Podman is absent; Ubuntu 22.04 sets the glibc floor. Inside it, `release/build-linux.sh` runs `scripts/container-build.sh` for libvmaf (our fork, at the commit that script pins) and the kernels, then builds dav1d 1.5.4 and FFmpeg 9.0.2 as static libraries from their checksummed release tarballs, FFmpeg configured with only the demuxers, decoders and NVDEC hardware decoders the program uses and no GPL or nonfree parts, and links the program with the C++ runtime statically. It checks that the program loads nothing but glibc and `libcuda.so.1` and that it starts, with the CUDA toolkit's stub standing in for the driver. Each archive holds the executable, `README.md`, `LICENSE`, and `licenses/` with the texts for libvmaf (BSD+Patent), the third-party code compiled into libvmaf (`libvmaf-third-party.txt`: libsvm, Xiph.Org's PSNR-HVS, Tom Distler's IQA, dav1d's x86 CPU detection, x264's x86inc.asm, which FFmpeg and dav1d also assemble, and mkdirp), FFmpeg (LGPL 2.1 or later), dav1d (BSD-2-Clause) and NVIDIA's codec headers (MIT). The third-party notices are copied from the headers of libvmaf's sources at build time, so check them against `ar t` of `libvmaf.a` when the libvmaf commit changes. FFmpeg is linked statically, so every release also attaches FFmpeg's unmodified tarball with the two build scripts, which together let a user relink the program with a modified FFmpeg.

## Checks

`validation.yml` runs the tests on a runner, where the frame-by-frame tests against libvmaf skip themselves for want of a GPU and the numpy reference's tests run, and builds both assets. Run the full suite on a machine with an NVIDIA GPU against the release build before releasing:

```sh
release/stage.sh dist work
tar -xzf dist/encode-score-vX.Y.Z-x86_64-linux.tar.gz -C work
ENCODE_SCORE_BIN=work/encode-score-vX.Y.Z-x86_64-linux/encode-score uv --no-config run --locked python -m unittest discover -s tests
```

Its gate job, `validate`, passes only when the build job succeeded (`wakamex/release-actions/validate-gate`); it is the required check `release-eligible / validate`, produced on every push to `main` by `release-eligibility.yml`.

## Releasing vX.Y.Z

1. Set `version` in `pyproject.toml` to `X.Y.Z`, run `uv --no-config lock`, and write `release-notes/vX.Y.Z.md`.
2. Commit them as `Release vX.Y.Z` and push `main` without tags.
3. Wait for `release-eligible / validate` to pass on that commit, and check that remote `main` still points to it.
4. Tag it with an annotated `vX.Y.Z` and push only the tag.

`publish.yml` then reruns validation, checks that the tag is annotated, matches `pyproject.toml` and has release notes (`wakamex/release-actions/verify-release-tag`), and rebuilds the assets from the tag. The shared binary release workflow writes `SHA256SUMS`, attests every asset and verifies the attestations, and creates the Release from the notes file. Rerunning it on a published tag verifies the existing Release instead of replacing it. A broken release is fixed by a new version; tags and published assets are never moved or replaced.

## One-time setup

- Turn on immutable releases in the repository settings.
- Once `release-eligible / validate` has passed on `main`, add the tag ruleset `Validated release tags` for `refs/tags/v*` that requires that check, with no bypass actors, deletion restricted and non-fast-forward updates blocked, restricted to the GitHub Actions app that produced the check.

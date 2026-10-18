"""Compare a re-encode with its original on what a viewer notices: lost texture, banding and color, not only brightness.

Video Multi-Method Assessment Fusion (VMAF) scores brightness only and rates a soft, grain-free picture as nearly
perfect, so it misses the texture that a small encode smooths away. For each clip this tool reports:

- VMAF (with the 4K model for pictures 3840 pixels wide or more and the standard model otherwise, unless `--model`
  names one), and the peak signal-to-noise ratio (PSNR) of brightness and of both color channels, from one libvmaf run.
- Banding, as the Contrast Aware Multiscale Banding Index (CAMBI) of each file; a higher score is more visible banding.
- The amount of detail in the re-encode, as a percentage of the original's, measured on the original 10-bit picture
  in single frames. Fine detail is 1 to 2 pixels (mostly grain) and mid detail is 3 to 8 pixels (texture and grain).
  The amount says how much texture there is, not whether it is the original's: grain that the decoder synthesizes
  counts in full.
- Matching detail: how much of the original's own fine and mid detail is still in the re-encode, as the share of the
  original's detail that the re-encode's detail reproduces pixel for pixel (the projection of one onto the other).
  New grain, ringing and blocking add nothing to it, so a blurred picture with fresh grain scores near zero.
- The amount of oriented detail: grain has no preferred direction and a wrinkle, an edge or a fold of fabric does, so the
  tool measures how much of the picture's gradient energy points one way inside each 16 by 16 pixel window (the
  difference of the two eigenvalues of the structure tensor). Grain cancels out in this measure without needing
  neighbouring frames, so it holds in moving shots. A cross-hatched weave points two ways and partly cancels too, so
  read the mid-detail figure beside it.
- The same figures for the structured windows only: the 20% of windows with the most oriented detail in the original.
  These are the edges, fabric and faces that a viewer looks at, so grain on a flat wall does not dilute them. A total
  over them is dominated by the few windows with the strongest edges, which every encoder keeps, so the median
  window and the worst tenth of the windows are reported beside it.
- Persistent texture: grain changes every frame and texture does not, so the mean of the original's 6 frames before
  and 6 frames after a frame, leaving that frame out, holds the texture with the grain averaged down. The tool
  reports the share of that texture that the re-encode's frame reproduces. Only windows where the texture is strong
  enough to tell from grain are counted, and they are chosen by whether two separate halves of those neighbouring
  frames agree, which uses neither the measured frame of the original nor the re-encode, so a chance resemblance
  between two frames of grain cannot pass for texture. A window that moves during those frames is not counted.
- Color: the mean blue and red color values and the overall color strength of both files.
- The video bitrate of the re-encode inside the clip.
- A score from 0 to 100 for each clip and for the run, the weighted mean of four parts that are each 0 to 100:
  fidelity (VMAF), texture (persistent texture, half the median window and half the worst tenth), banding (100 when
  the re-encode bands no more than the original, 0 when its CAMBI is 5 higher) and grain (100 when the amount of
  fine detail over the whole picture equals the original's, losing a point per percent above or below). The weights
  are in WEIGHTS; they are a judgement and are not fitted to viewer ratings, so read the parts beside the score.

It also writes a picture of the three structured windows that lost the most oriented detail, enlarged: original,
re-encode, and their difference amplified 8 times, with the same brightness stretch on each.

The two files are aligned by `video_vmaf.find_offset` unless `--offset` is given, and are cropped to their common
picture area when one has its black bars trimmed. Each clip is decoded once: one ffmpeg process feeds libvmaf and
hands back the evenly spaced single frames that the detail figures use.

Usage: encode-score ORIGINAL NEW --start SECONDS [SECONDS ...] [--seconds 6] [--samples 12] [--offset 0] [--model auto] [--out DIR] [--json FILE]
"""

import argparse
import json
import os
import struct
import subprocess
import tempfile
import threading
import zlib
from pathlib import Path

import numpy as np

from . import align as vv

WINDOW = 16  # pixels; the unit for choosing the structured part of the picture
STRUCTURE_SHARE = 0.20
PANEL = 192  # pixels cut from each picture around a window for the montage
ZOOM = 3
KEYS = ("fine", "mid", "oriented")
BANDS = ("fine", "mid")
RADIUS = 6  # frames before and after a frame whose mean, without the frame itself, is taken as its texture
HALVES = ("1 0 1 0 1 0 0 0 1 0 1 0 1", "0 1 0 1 0 1 0 1 0 1 0 1 0")  # the two interleaved halves of those 2 * RADIUS frames, as tmix weights
PERSISTS = 0.5  # correlation between the two halves' mid detail in a window, above which the window holds texture
MIN_WINDOWS = 100  # fewer persistent windows than this in a clip and its texture is not scored
WEIGHTS = {"fidelity": 0.30, "texture": 0.30, "banding": 0.25, "grain": 0.15}
BANDING_FLOOR = 5.0  # CAMBI above the original's at which the banding part reaches 0


def box(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over a k by k window (k odd) at every pixel, with the edges repeated."""
    r = k // 2
    c = np.pad(a, r, mode="edge").cumsum(0, dtype=np.float64).cumsum(1, dtype=np.float64)
    c = np.pad(c, ((1, 0), (1, 0)))
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)).astype(np.float32)


def window_sums(values: np.ndarray) -> np.ndarray:
    """Sum of an array inside each WINDOW by WINDOW window, as a grid."""
    h, w = values.shape[0] // WINDOW * WINDOW, values.shape[1] // WINDOW * WINDOW
    return values[:h, :w].astype(np.float64).reshape(h // WINDOW, WINDOW, w // WINDOW, WINDOW).sum(axis=(1, 3))


def bands(picture: np.ndarray) -> dict[str, np.ndarray]:
    """The fine (1 to 2 pixel) and mid (3 to 8 pixel) detail of a brightness picture, and the picture smoothed over 3 pixels."""
    b3 = box(picture, 3)
    return {"fine": picture - b3, "mid": b3 - box(picture, 9), "smooth": b3}


def window_energies(picture: np.ndarray, detail: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Per window: fine detail energy, mid detail energy and oriented detail energy of a brightness picture.

    Fine and mid are the energy of the 1 to 2 pixel and 3 to 8 pixel bands. Oriented is the difference of the
    eigenvalues of the summed structure tensor, which is near zero for grain and large for a dominant direction.
    """
    detail = detail or bands(picture)
    b3 = detail["smooth"]
    gx, gy = np.zeros_like(b3), np.zeros_like(b3)
    gx[:, 1:-1], gy[1:-1, :] = b3[:, 2:] - b3[:, :-2], b3[2:, :] - b3[:-2, :]
    jxx, jyy, jxy = window_sums(gx * gx), window_sums(gy * gy), window_sums(gx * gy)
    return {"fine": window_sums(detail["fine"] ** 2), "mid": window_sums(detail["mid"] ** 2), "oriented": np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)}


def structure_mask(oriented: np.ndarray, share: float = STRUCTURE_SHARE) -> np.ndarray:
    """The windows with the most oriented detail: a boolean grid with about `share` of the windows set."""
    return (oriented >= np.quantile(oriented, 1 - share)) & (oriented > 0)


def kept(dist: float, ref: float) -> float:
    """Percentage of the original's detail amplitude (root of the energy) that the re-encode keeps."""
    return 100.0 * float(np.sqrt(dist / ref)) if ref > 0 else float("nan")


def spread(ratios: list[np.ndarray]) -> dict[str, float]:
    """The median window and the 10th percentile window of per-window ratios, in percent."""
    values = np.concatenate(ratios) if ratios else np.array([])
    if not values.size:
        return {"median": float("nan"), "worst_tenth": float("nan")}
    return {"median": 100.0 * float(np.median(values)), "worst_tenth": 100.0 * float(np.quantile(values, 0.1))}


def compare_frames(ref: list[dict], dist: list[dict], steady: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """Amount of detail, matching detail, color and the windows that lost the most oriented detail, from matching single frames.

    Each frame is a dict of `y` (height, width) and `u`, `v` (height / 2, width / 2) arrays of 10-bit values.
    `regions` holds the amount of each kind of detail and `matching` the share of the original's own detail that
    is reproduced, each as a total over the whole picture, a total over the structured windows, and the median and
    worst tenth of the structured windows. `steady` gives, for each frame, the mean brightness of two interleaved
    halves of the original's neighbouring frames (on any linear scale). With it, `persistent` holds the share of the
    original's persistent texture that the re-encode reproduces: over the structured windows where the two halves'
    mid detail agree, the re-encode's mid detail projected onto the neighbours' mean, against the original's.
    """
    totals = {region: {key: [0.0, 0.0] for key in KEYS} for region in ("all", "structure")}
    shared = {region: {key: [0.0, 0.0] for key in BANDS} for region in ("all", "structure")}
    amounts, matches = {key: [] for key in KEYS}, {key: [] for key in BANDS}
    candidates, color = [], {"ref": [], "dist": []}
    lasting, lasting_totals = [], [0.0, 0.0]
    for index, (r, d) in enumerate(zip(ref, dist)):
        detail = {"ref": bands(r["y"].astype(np.float32)), "dist": bands(d["y"].astype(np.float32))}
        energy = {name: window_energies(detail[name]["smooth"], detail[name]) for name in detail}
        for name, frame in (("ref", r), ("dist", d)):
            u, v = frame["u"].astype(np.float64), frame["v"].astype(np.float64)
            color[name].append((u.mean(), v.mean(), np.sqrt((u - 512) ** 2 + (v - 512) ** 2).mean()))
        mask = structure_mask(energy["ref"]["oriented"])
        for key in KEYS:
            totals["all"][key][0] += energy["ref"][key].sum()
            totals["all"][key][1] += energy["dist"][key].sum()
            totals["structure"][key][0] += energy["ref"][key][mask].sum()
            totals["structure"][key][1] += energy["dist"][key][mask].sum()
            amounts[key].append(np.sqrt(energy["dist"][key][mask] / np.maximum(energy["ref"][key][mask], 1e-9)))
        for key in BANDS:
            cross = window_sums(detail["ref"][key] * detail["dist"][key])
            shared["all"][key][0] += energy["ref"][key].sum()
            shared["all"][key][1] += cross.sum()
            shared["structure"][key][0] += energy["ref"][key][mask].sum()
            shared["structure"][key][1] += cross[mask].sum()
            matches[key].append(cross[mask] / np.maximum(energy["ref"][key][mask], 1e-9))
        if steady:
            half = [bands(picture.astype(np.float32))["mid"] for picture in steady[index]]
            texture = (half[0] + half[1]) / 2
            own = window_sums(detail["ref"]["mid"] * texture)
            agree = window_sums(half[0] * half[1]) / np.sqrt(np.maximum(window_sums(half[0] ** 2) * window_sums(half[1] ** 2), 1e-9))
            keep = mask & (agree > PERSISTS) & (own > 0)
            mine = window_sums(detail["dist"]["mid"] * texture)
            lasting.append(mine[keep] / own[keep])
            lasting_totals[0] += own[keep].sum()
            lasting_totals[1] += mine[keep].sum()
        ratio = np.sqrt(energy["dist"]["oriented"] / np.maximum(energy["ref"]["oriented"], 1e-9))
        candidates += [(float(ratio[y, x]), index, int(y), int(x)) for y, x in zip(*np.nonzero(mask))]
    candidates.sort()
    picked = []
    for entry in candidates:  # neighbouring windows show the same patch, so keep the chosen ones apart
        if all(entry[1] != o[1] or abs(entry[2] - o[2]) > PANEL // WINDOW or abs(entry[3] - o[3]) > PANEL // WINDOW for o in picked):
            picked.append(entry)
        if len(picked) == 3:
            break
    result = {"regions": {region: {key: kept(v[1], v[0]) for key, v in values.items()} for region, values in totals.items()}, "worst": picked}
    result["matching"] = {region: {key: 100.0 * v[1] / v[0] if v[0] > 0 else float("nan") for key, v in values.items()} for region, values in shared.items()}
    for name, key in (("median", "median"), ("worst_tenth", "worst_tenth")):
        result["regions"][name] = {k: spread(amounts[k])[key] for k in KEYS}
        result["matching"][name] = {k: spread(matches[k])[key] for k in BANDS}
    if steady:
        windows = int(sum(a.size for a in lasting))
        result["persistent"] = dict(spread(lasting), windows=windows, total=float(100.0 * lasting_totals[1] / lasting_totals[0]) if lasting_totals[0] > 0 else float("nan"))
    for name, rows in color.items():
        result[f"color_{name}"] = {"cb": float(np.mean([c[0] for c in rows])), "cr": float(np.mean([c[1] for c in rows])), "strength": float(np.mean([c[2] for c in rows]))}
    return result


def clamp(value: float) -> float:
    return float(min(max(value, 0.0), 100.0))


def score(vmaf: float, banding: float, original_banding: float, found: dict) -> dict:
    """The four parts of the score, each 0 to 100, and their weighted mean; a part that could not be measured is left out."""
    texture = found.get("persistent", {"windows": 0})
    parts = {"fidelity": clamp(vmaf),
             "texture": clamp(texture["median"]) / 2 + clamp(texture["worst_tenth"]) / 2 if texture["windows"] >= MIN_WINDOWS else float("nan"),
             "banding": clamp(100.0 * (1 - (banding - original_banding) / BANDING_FLOOR)),
             "grain": clamp(100.0 - abs(found["regions"]["all"]["fine"] - 100.0))}
    measured = {key: value for key, value in parts.items() if not np.isnan(value)}
    return dict(parts, score=sum(WEIGHTS[key] * value for key, value in measured.items()) / sum(WEIGHTS[key] for key in measured))


def write_png(path: Path, picture: np.ndarray) -> None:
    """Write an 8-bit grayscale picture as a PNG."""
    height, width = picture.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in picture.astype(np.uint8))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def montage(ref: list[dict], dist: list[dict], windows: list[tuple]) -> np.ndarray:
    """Rows of original | re-encode | amplified difference, enlarged, for each (ratio, frame, window row, window column)."""
    rows = []
    for _, index, wy, wx in windows:
        r, d = ref[index]["y"], dist[index]["y"]
        top = int(np.clip(wy * WINDOW + WINDOW // 2 - PANEL // 2, 0, r.shape[0] - PANEL))
        left = int(np.clip(wx * WINDOW + WINDOW // 2 - PANEL // 2, 0, r.shape[1] - PANEL))
        rc, dc = r[top:top + PANEL, left:left + PANEL].astype(np.float32), d[top:top + PANEL, left:left + PANEL].astype(np.float32)
        low = np.percentile(rc, 1)
        scale = 255.0 / max(np.percentile(rc, 99.5) - low, 8.0)
        panels = [np.clip((rc - low) * scale, 0, 255), np.clip((dc - low) * scale, 0, 255), np.clip(np.abs(rc - dc) * scale * 8, 0, 255)]
        panels = [np.repeat(np.repeat(p, ZOOM, axis=0), ZOOM, axis=1) for p in panels]
        gap = np.full((panels[0].shape[0], 6), 255.0)
        rows.append(np.hstack([panels[0], gap, panels[1], gap, panels[2]]))
    if not rows:
        return np.zeros((8, 8))
    out = rows[0]
    for row in rows[1:]:
        out = np.vstack([out, np.full((6, out.shape[1]), 255.0), row])
    return out


def decode(path: str, start: float, crop: str, width: int, height: int) -> dict | None:
    """One frame at `start` as 10-bit planes, or None when the file ends first."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", path, "-an", "-frames:v", "1",
                          "-vf", f"{crop},format=yuv420p10le", "-f", "rawvideo", "-"], capture_output=True, check=True, timeout=900).stdout
    luma, chroma = width * height, (width // 2) * (height // 2)
    if len(raw) < (luma + 2 * chroma) * 2:
        return None
    values = np.frombuffer(raw[:(luma + 2 * chroma) * 2], dtype="<u2")
    return {"y": values[:luma].reshape(height, width), "u": values[luma:luma + chroma].reshape(height // 2, width // 2),
            "v": values[luma + chroma:].reshape(height // 2, width // 2)}


def run_libvmaf(dist: str, dist_start: float, dist_crop: str, ref: str, ref_start: float, ref_crop: str, seconds: float, threads: int, log: Path) -> dict:
    """VMAF with the 4K model, per-plane PSNR and CAMBI over `seconds`; returns the pooled mean of each metric."""
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-ss", str(dist_start), "-t", str(seconds), "-i", dist, "-ss", str(ref_start), "-t", str(seconds), "-i", ref,
                    "-lavfi", f"[0:v:0]setpts=N/FRAME_RATE/TB,setparams=colorspace=unknown,{dist_crop}[d];[1:v:0]setpts=N/FRAME_RATE/TB,setparams=colorspace=unknown,{ref_crop}[r];"
                              f"[d][r]libvmaf=model=version=vmaf_4k_v0.6.1:feature=name=psnr|name=cambi:n_threads={threads}:log_fmt=json:log_path={log}",
                    "-f", "null", "-"], check=True, timeout=3600)
    data = json.loads(log.read_text())
    metrics = data["pooled_metrics"]
    return {"vmaf": vv.pooled(data), "psnr_y": metrics["psnr_y"]["mean"], "psnr_cb": metrics["psnr_cb"]["mean"], "psnr_cr": metrics["psnr_cr"]["mean"], "cambi": metrics["cambi"]["mean"]}


def frame_rate(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path], capture_output=True, text=True, timeout=120).stdout
    top, _, bottom = out.strip().strip(",").partition("/")
    return float(top) / float(bottom or 1)


def bitrate(path: str, start: float, seconds: float) -> float:
    """Video bitrate in Mbps of the packets between `start` and `start + seconds`."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{start}%{start + seconds}", "-show_entries", "packet=pts_time,size", "-of", "csv=p=0", path],
                         capture_output=True, text=True, timeout=600).stdout
    rows = [line.split(",") for line in out.split() if line.count(",") >= 1 and line[0].isdigit()]
    return sum(int(size) for pts, size, *_ in rows if start <= float(pts) < start + seconds) * 8 / seconds / 1e6


def vmaf_model(width: int) -> str:
    """The libvmaf model for a picture `width` pixels wide: the 4K model for 4K, the standard 1080p model below it."""
    return "vmaf_4k_v0.6.1" if width >= 3840 else "vmaf_v0.6.1"


def measure(new: str, new_start: float, new_crop: str, original: str, original_start: float, original_crop: str, seconds: float, size: tuple[int, int], samples: int, threads: int, model: str = "vmaf_4k_v0.6.1") -> tuple[dict, list[dict], list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    """Decode a clip of both files once: the libvmaf scores, `samples` evenly spaced frames of each file, and for each the mean brightness of two interleaved halves of the original's RADIUS frames before and after it.

    One ffmpeg process pairs the frames by position, runs libvmaf (VMAF with the 4K model, PSNR per plane, and CAMBI of
    both files) on every pair, and writes the chosen frames to pipes.
    """
    width, height = size
    count = int(seconds * frame_rate(original)) - 2 * RADIUS - 1
    picks = sorted({RADIUS + max(int(count * (i + 0.5) / samples), 0) for i in range(samples)})
    chosen = "+".join(f"eq(n,{n})" for n in picks)
    ahead = "+".join(f"eq(n,{n + RADIUS})" for n in picks)  # tmix gives the mix of the frames up to and including n
    pipes = [os.pipe() for _ in range(4)]
    with tempfile.TemporaryDirectory(prefix="video-compare-") as tmp:
        log = Path(tmp) / "libvmaf.json"
        graph = (f"[0:v:0]setpts=N/FRAME_RATE/TB,setparams=colorspace=unknown,{new_crop},split[d][dp];[1:v:0]setpts=N/FRAME_RATE/TB,setparams=colorspace=unknown,{original_crop},split=3[r][rp][rn];"
                 f"[d][r]libvmaf=model=version={model}:feature=name=psnr|name=cambi\\\\:full_ref=true:n_threads={threads}:log_fmt=json:log_path={log}[scored];"
                 f"[dp]select='{chosen}',format=yuv420p10le[do];[rp]select='{chosen}',format=yuv420p10le[ro];"
                 f"[rn]format=yuv420p10le,extractplanes=y,format=gray16le,split[na][nb];" + ";".join(
                     f"[n{name}]tmix=frames={2 * RADIUS + 1}:weights='{weights}',select='{ahead}'[{name}o]" for name, weights in zip("ab", HALVES)))
        command = ["nice", "-n", "10", "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-ss", str(new_start), "-t", str(seconds), "-i", new, "-ss", str(original_start), "-t", str(seconds), "-i", original,
                   "-filter_complex", graph, "-map", "[scored]", "-f", "null", "-"]
        for label, (_, write) in zip(("do", "ro", "ao", "bo"), pipes):
            command += ["-map", f"[{label}]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{write}"]
        process = subprocess.Popen(command, pass_fds=[write for _, write in pipes])
        raw = [b""] * 4

        def drain(index: int) -> None:
            with os.fdopen(pipes[index][0], "rb") as stream:
                raw[index] = stream.read()

        readers = [threading.Thread(target=drain, args=(i,)) for i in range(4)]
        for _, write in pipes:
            os.close(write)
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        data = json.loads(log.read_text())
    metrics = data["pooled_metrics"]
    scores = {"vmaf": vv.pooled(data), "psnr_y": metrics["psnr_y"]["mean"], "psnr_cb": metrics["psnr_cb"]["mean"], "psnr_cr": metrics["psnr_cr"]["mean"],
              "cambi": metrics["cambi"]["mean"], "cambi_original": metrics["cambi_source"]["mean"]}
    luma, chroma = width * height, (width // 2) * (height // 2)

    def planes(buffer: bytes) -> list[dict]:
        values = np.frombuffer(buffer, dtype="<u2")
        frames = values[:values.size // (luma + 2 * chroma) * (luma + 2 * chroma)].reshape(-1, luma + 2 * chroma)
        return [{"y": f[:luma].reshape(height, width), "u": f[luma:luma + chroma].reshape(height // 2, width // 2), "v": f[luma + chroma:].reshape(height // 2, width // 2)} for f in frames]

    dist, ref = planes(raw[0]), planes(raw[1])
    halves = [np.frombuffer(buffer, dtype="<u2")[:len(buffer) // 2 // luma * luma].reshape(-1, height, width) for buffer in raw[2:]]
    steady = list(zip(*halves))
    shortest = min(len(dist), len(ref), len(steady))
    return scores, ref[:shortest], dist[:shortest], steady[:shortest]


def compare_clip(original: str, new: str, start: float, seconds: float, samples: int, offset: float | None, threads: int, out: Path, model: str = "auto") -> dict | None:
    """Measure one clip, print its figures and return them; None when the clip could not be aligned or read."""
    sizes = vv.size_of(original), vv.size_of(new)
    common = (min(sizes[0][0], sizes[1][0]), min(sizes[0][1], sizes[1][1]))
    crops = vv.crop_filter(sizes[0], common), vv.crop_filter(sizes[1], common)
    if offset is None:
        offset, residual = vv.find_offset(original, new, start, crops, frame_rate(original), window=max(seconds, 6.0), search=min(10.0, start))
        if residual > vv.MAX_RESIDUAL:
            print(f"clip at {start:.1f} s: could not align the two files (best offset {offset:+.2f} s, residual {residual:.1f}), not scored", flush=True)
            return None
    model = vmaf_model(common[0]) if model == "auto" else model
    scored, ref, dist, steady = measure(new, start, crops[1], original, start + offset, crops[0], seconds, common, samples, threads, model)
    if not ref:
        print(f"clip at {start:.1f} s: no frames could be read", flush=True)
        return None
    found = compare_frames(ref, dist, steady)
    picture = out / f"compare_{start:.0f}s.png"
    write_png(picture, montage(ref, dist, found.pop("worst")))
    vmaf, regions, texture = scored["vmaf"], found["regions"], found["persistent"]
    parts = score(vmaf["mean"], scored["cambi"], scored["cambi_original"], found)
    mbps = bitrate(new, start, seconds)
    print(f"\nclip at {start:.1f} s for {seconds:.1f} s (offset {offset:+.3f} s, {len(ref)} frames measured, re-encode video {mbps:.1f} Mbps)")
    print(f"  Score {parts['score']:.1f} of 100: fidelity {parts['fidelity']:.0f}, texture {parts['texture']:.0f}, banding {parts['banding']:.0f}, grain {parts['grain']:.0f}")
    print(f"  VMAF ({model}, brightness only): mean {vmaf['mean']:.2f}, 1st percentile {vmaf['p1']:.2f}, worst frame {vmaf['min']:.2f}")
    print(f"  PSNR in dB: brightness {scored['psnr_y']:.1f}, blue difference {scored['psnr_cb']:.1f}, red difference {scored['psnr_cr']:.1f}")
    print(f"  Banding score (CAMBI, higher is more banding): original {scored['cambi_original']:.2f}, re-encode {scored['cambi']:.2f}")
    print(f"  Persistent texture reproduced (the original's detail that stays put over the {RADIUS} frames either side, so grain cannot count), over {texture['windows']} windows: "
          f"median window {texture['median']:.0f}%, worst tenth {texture['worst_tenth']:.0f}%, total {texture['total']:.0f}%")
    rows = (("whole picture", "all"), ("structured windows (top 20%)", "structure"), ("median structured window", "median"), ("worst tenth of structured windows", "worst_tenth"))
    print("  Amount of detail in the re-encode, as a percentage of the original's (synthesized grain counts)")
    print(f"    {'':36}{'fine':>8}{'mid':>8}{'oriented':>10}")
    for label, region in rows:
        r = regions[region]
        print(f"    {label:36}{r['fine']:7.0f}%{r['mid']:7.0f}%{r['oriented']:9.0f}%")
    print("  Share of the original's own detail that the re-encode reproduces (new grain and artifacts do not count)")
    print(f"    {'':36}{'fine':>8}{'mid':>8}")
    for label, region in rows:
        r = found["matching"][region]
        print(f"    {label:36}{r['fine']:7.0f}%{r['mid']:7.0f}%")
    c0, c1 = found["color_ref"], found["color_dist"]
    print(f"  Color (10-bit values, 512 is neutral), original to re-encode: blue difference {c0['cb']:.1f} to {c1['cb']:.1f}, red difference {c0['cr']:.1f} to {c1['cr']:.1f}, strength {c0['strength']:.1f} to {c1['strength']:.1f}")
    print(f"  Enlarged windows that lost the most oriented detail (original, re-encode, difference x8): {picture}", flush=True)
    return dict(found, start=start, seconds=seconds, offset=offset, mbps=mbps, model=model, score=parts, **scored)


def main() -> None:
    parser = argparse.ArgumentParser(prog="encode-score", description=__doc__.split("\n")[0])
    parser.add_argument("original")
    parser.add_argument("new")
    parser.add_argument("--start", type=float, nargs="+", required=True, help="clip start times in seconds")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=12, help="single frames measured per clip")
    parser.add_argument("--offset", type=float, help="seconds to add to the start in the original; found automatically when omitted")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model", default="auto", help="libvmaf model; auto picks vmaf_4k_v0.6.1 for 4K pictures and vmaf_v0.6.1 below")
    parser.add_argument("--out", type=Path, default=Path("encode-score-out"))
    parser.add_argument("--json", type=Path, help="write every figure of every clip to this file")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    clips = [compare_clip(args.original, args.new, start, args.seconds, args.samples, args.offset, args.threads, args.out, args.model) for start in args.start]
    clips = [clip for clip in clips if clip]
    if not clips:
        raise SystemExit("no clip could be measured")
    overall = {key: float(np.nanmean([clip["score"][key] for clip in clips])) for key in ("score", *WEIGHTS)}
    worst = min(clips, key=lambda clip: clip["score"]["score"])
    print(f"\noverall score {overall['score']:.1f} of 100 over {len(clips)} clips (fidelity {overall['fidelity']:.0f}, texture {overall['texture']:.0f}, banding {overall['banding']:.0f}, "
          f"grain {overall['grain']:.0f}); worst clip {worst['score']['score']:.1f} at {worst['start']:.1f} s; mean video bitrate in the clips {np.mean([clip['mbps'] for clip in clips]):.1f} Mbps")
    if args.json:
        args.json.write_text(json.dumps({"original": args.original, "new": args.new, "weights": WEIGHTS, "overall": overall, "clips": clips}, indent=1))


if __name__ == "__main__":
    main()

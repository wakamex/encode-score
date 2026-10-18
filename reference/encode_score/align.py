"""Line two video files up in time and space: their common picture area, and the offset that makes their frames match."""

import math
import subprocess


def pooled(log: dict) -> dict:
    """Mean, minimum and 1st percentile VMAF over the frames in a libvmaf JSON log."""
    scores = sorted(f["metrics"]["vmaf"] for f in log["frames"])
    return {"mean": log["pooled_metrics"]["vmaf"]["mean"], "min": scores[0], "p1": scores[math.ceil(len(scores) * 0.01) - 1], "frames": len(scores)}


def duration_of(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True, timeout=120).stdout
    return float(out.strip())


def size_of(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, timeout=120).stdout
    width, height = out.strip().split(",")[:2]
    return int(width), int(height)


def crop_filter(size: tuple[int, int], common: tuple[int, int]) -> str:
    """A centred crop of a picture of `size` down to the `common` size, or nothing when they already match."""
    if size == common:
        return "null"
    return f"crop={common[0]}:{common[1]}:{(size[0] - common[0]) // 2}:{(size[1] - common[1]) // 2}"


FPS, TINY = 2, (24, 14)


def best_lag(reference: list[bytes], probe: list[bytes], max_lag: int) -> tuple[int, float]:
    """The shift (in frames) of `probe` against `reference` with the smallest mean absolute pixel difference, and that difference.

    reference[i + lag] is compared with probe[i]; reference must be at least len(probe) + max_lag frames long
    when lag runs from 0 to max_lag, so callers pass a reference that starts max_lag frames early.
    """
    best = (0, float("inf"))
    for lag in range(0, 2 * max_lag + 1):
        total = count = 0
        for i, frame in enumerate(probe):
            if i + lag >= len(reference):
                break
            ref = reference[i + lag]
            total += sum(abs(a - b) for a, b in zip(ref, frame))
            count += len(frame)
        if count and total / count < best[1]:
            best = (lag - max_lag, total / count)
    return best


def tiny_frames(path: str, start: float, seconds: float, crop: str, fps: float) -> list[bytes]:
    """Small grayscale frames, `fps` per second, from a window of the file."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(max(start, 0)), "-t", str(seconds), "-i", path, "-an", "-vf",
                          f"{crop},fps={fps},scale={TINY[0]}:{TINY[1]}:flags=area,format=gray", "-f", "rawvideo", "-"], capture_output=True, timeout=900).stdout
    size = TINY[0] * TINY[1]
    return [out[i:i + size] for i in range(0, len(out) - size + 1, size)]


MAX_RESIDUAL = 4.0  # mean absolute gray difference (0 to 255) above which two windows are not the same picture


def find_offset(original: str, new: str, start: float, crops: tuple[str, str], fps: float, window: float = 40.0, search: float = 240.0) -> tuple[float, float]:
    """Seconds to add to `start` in the original so its frames line up with the new file's frames at `start`, and the residual difference.

    A coarse search at half a frame per second over +/- `search` seconds finds the rough shift, a finer one at
    FPS frames per second over +/- 3 seconds sharpens it, and a last one at the film's frame rate over +/- 1.5 seconds
    makes it frame accurate, which VMAF needs. `fps` is the original's frame rate.
    """
    coarse_fps = 0.5
    probe = tiny_frames(new, start, window, crops[1], coarse_fps)
    reference = tiny_frames(original, start - search, window + 2 * search, crops[0], coarse_fps)
    lag, _ = best_lag(reference, probe, int(search * coarse_fps))
    rough = lag / coarse_fps
    fine_probe = tiny_frames(new, start, window, crops[1], FPS)
    fine_reference = tiny_frames(original, start + rough - 3, window + 6, crops[0], FPS)
    fine_lag, residual = best_lag(fine_reference, fine_probe, 3 * FPS)
    fine = rough + fine_lag / FPS
    exact_fps = fps
    exact_probe = tiny_frames(new, start, 8.0, crops[1], exact_fps)
    exact_reference = tiny_frames(original, start + fine - 1.5, 8.0 + 3.0, crops[0], exact_fps)
    exact_lag, residual = best_lag(exact_reference, exact_probe, round(1.5 * exact_fps))
    return fine + exact_lag / exact_fps, residual

"""The native program against authoritative and reference implementations, on generated test videos.

VMAF, CAMBI and PSNR are checked frame by frame against FFmpeg's libvmaf filter on the processor; the detail, texture,
grain and color figures against the numpy reference in reference/encode_score. The tests need an NVIDIA GPU and the built
program (build/encode-score, or the path in ENCODE_SCORE_BIN), and are skipped without them.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from encode_score import compare as reference  # noqa: E402

BINARY = Path(os.environ.get("ENCODE_SCORE_BIN", ROOT / "build" / "encode-score"))
W, H, FPS, SECONDS = 1280, 720, 24, 3
# A dark, slowly changing gradient with fine noise and moving shapes: enough texture for the detail measures, smooth
# areas that a coarse encode bands, and motion for VMAF.
PATTERN = (f"gradients=s={W}x{H}:c0=0x101418:c1=0x30343a:speed=0.02:r={FPS}:d={SECONDS + 2}[g];"
           f"testsrc2=s={W}x{H}:r={FPS}:d={SECONDS + 2},format=yuv420p,colorchannelmixer=.15:0:0:0:0:.15:0:0:0:0:.15[t];"
           f"[g]format=yuv420p[gg];[gg][t]blend=all_mode=addition,noise=alls=3:allf=t,format=yuv420p10le")


def available():
    return BINARY.exists() and shutil.which("nvidia-smi") and shutil.which("ffmpeg")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


@unittest.skipUnless(available(), "needs an NVIDIA GPU, ffmpeg and the built program")
class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="encode-score-test-")
        d = Path(cls.tmp.name)
        cls.source, cls.encode, cls.late, cls.other = d / "source.mkv", d / "encode.mkv", d / "late.mkv", d / "other.mkv"
        x265 = ["-c:v", "libx265", "-pix_fmt", "yuv420p10le"]
        run(["ffmpeg", "-v", "error", "-y", "-filter_complex", PATTERN, *x265, "-x265-params", "lossless=1:log-level=error", str(cls.source)])
        run(["ffmpeg", "-v", "error", "-y", "-i", str(cls.source), *x265, "-x265-params", "crf=38:log-level=error", str(cls.encode)])
        # the same encode starting one second later, and a different picture
        run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(cls.source), *x265, "-x265-params", "crf=38:log-level=error", str(cls.late)])
        run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"mandelbrot=s={W}x{H}:r={FPS}", "-t", str(SECONDS + 2), *x265,
             "-x265-params", "crf=30:log-level=error", str(cls.other)])
        cls.result = cls.score(cls.encode, "--offset", "0", "--seconds", str(SECONDS))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def score(cls, *args):
        out = Path(cls.tmp.name) / "score.json"
        run([str(BINARY), str(cls.source), *map(str, args), "--json", str(out)])
        return json.loads(out.read_text())

    def libvmaf(self, features):
        log = Path(self.tmp.name) / "libvmaf.json"
        run(["ffmpeg", "-v", "error", "-i", str(self.encode), "-i", str(self.source), "-lavfi",
             f"[0:v]setpts=N/FRAME_RATE/TB[d];[1:v]setpts=N/FRAME_RATE/TB[r];[d][r]libvmaf=model=version=vmaf_v0.6.1:"
             f"feature={features}:n_threads=8:log_fmt=json:log_path={log}", "-frames:v", str(SECONDS * FPS), "-f", "null", "-"])
        return json.loads(log.read_text())["frames"]

    def test_vmaf_cambi_and_psnr_match_libvmaf_frame_by_frame(self):
        cpu = self.libvmaf("name=psnr|name=cambi\\\\:full_ref=true")
        mine = self.result["encodes"][0]["per_frame"]
        self.assertEqual(len(mine["vmaf"]), SECONDS * FPS)
        for i, f in enumerate(cpu[:SECONDS * FPS]):
            m = f["metrics"]
            self.assertAlmostEqual(mine["vmaf"][i], m["vmaf"], delta=1e-3)
            self.assertAlmostEqual(mine["cambi"][i], m["cambi"], delta=1e-5)
            self.assertAlmostEqual(mine["cambi_source"][i], m["cambi_source"], delta=1e-5)
            self.assertAlmostEqual(mine["psnr_y"][i], m["psnr_y"], delta=1e-3)
        self.assertGreater(max(mine["cambi"]), 0.05, "the test video should band, or CAMBI is not exercised")

    def frames(self, path):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(SECONDS * FPS), "-pix_fmt", "yuv420p10le",
                              "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
        size = W * H * 3 // 2
        out = []
        for i in range(len(raw) // (size * 2)):
            v = np.frombuffer(raw[i * size * 2:(i + 1) * size * 2], dtype="<u2")
            out.append({"y": v[:W * H].reshape(H, W), "u": v[W * H:W * H * 5 // 4].reshape(H // 2, W // 2), "v": v[W * H * 5 // 4:].reshape(H // 2, W // 2)})
        return out

    def test_detail_texture_and_color_match_the_reference_implementation(self):
        ref, dist = self.frames(self.source), self.frames(self.encode)
        n = len(ref)
        found = reference.compare_frames(ref, dist)
        centres = range(6, n - 6)
        steady = [(np.mean([ref[c + o]["y"].astype(np.float64) for o in (-6, -4, -2, 2, 4, 6)], axis=0),
                   np.mean([ref[c + o]["y"].astype(np.float64) for o in (-5, -3, -1, 1, 3, 5)], axis=0)) for c in centres]
        lasting = reference.compare_frames([ref[c] for c in centres], [dist[c] for c in centres], steady)["persistent"]
        mine = self.result["encodes"][0]["whole"]
        for region in ("all", "structure"):
            for i, key in enumerate(reference.KEYS):
                self.assertAlmostEqual(mine["amount"][region][i], found["regions"][region][key], delta=0.01, msg=f"amount {region} {key}")
            for i, key in enumerate(reference.BANDS):
                self.assertAlmostEqual(mine["matching"][region][i], found["matching"][region][key], delta=0.01, msg=f"matching {region} {key}")
        # medians and tenths come from 0.05-point bins
        for region in ("median", "worst_tenth"):
            for i, key in enumerate(reference.KEYS):
                self.assertAlmostEqual(mine["amount"][region][i], found["regions"][region][key], delta=0.1, msg=f"amount {region} {key}")
            for i, key in enumerate(reference.BANDS):
                self.assertAlmostEqual(mine["matching"][region][i], found["matching"][region][key], delta=0.1, msg=f"matching {region} {key}")
        self.assertEqual(mine["lasting"]["windows"], lasting["windows"])
        self.assertAlmostEqual(mine["lasting"]["total"], lasting["total"], delta=0.01)
        self.assertAlmostEqual(mine["lasting"]["median"], lasting["median"], delta=0.1)
        for i, key in enumerate(("cb", "cr", "strength")):
            self.assertAlmostEqual(mine["color_source"][i], found["color_ref"][key], delta=1e-3)
            self.assertAlmostEqual(mine["color_encode"][i], found["color_dist"][key], delta=1e-3)

    def test_the_offset_of_a_later_start_is_found_to_the_frame(self):
        result = self.score(self.late, "--seconds", "1")
        self.assertAlmostEqual(result["offset"], 1.0, delta=0.5 / FPS)
        self.assertGreater(result["encodes"][0]["whole"]["psnr_y"], 30)

    def test_a_different_picture_is_refused_rather_than_scored(self):
        with self.assertRaises(subprocess.CalledProcessError) as failed:
            self.score(self.other, "--seconds", "1")
        self.assertIn("does not match the source", failed.exception.stderr)

    def test_scoring_two_encodes_in_one_pass_gives_the_same_figures_as_one_at_a_time(self):
        both = self.score(self.encode, self.encode, "--offset", "0", "--seconds", str(SECONDS))
        one = self.result["encodes"][0]["whole"]
        for e in both["encodes"]:
            for key in ("score", "vmaf", "cambi", "psnr_y"):
                self.assertAlmostEqual(e["whole"][key], one[key], places=6)


@unittest.skipUnless(available(), "needs an NVIDIA GPU, ffmpeg and the built program")
class OtherInputs(unittest.TestCase):
    """8-bit video, and files the GPU cannot decode, which are decoded on the processor and uploaded."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="encode-score-test-")
        d = Path(cls.tmp.name)
        pattern = PATTERN.replace("format=yuv420p10le", "format=yuv420p")
        cls.source8, cls.encode8 = d / "source8.mkv", d / "encode8.mkv"
        # FFV1 has no GPU decoder, so the 8-bit source also takes the processor path
        run(["ffmpeg", "-v", "error", "-y", "-filter_complex", pattern, "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(cls.source8)])
        run(["ffmpeg", "-v", "error", "-y", "-i", str(cls.source8), "-c:v", "libx264", "-crf", "34", "-pix_fmt", "yuv420p", str(cls.encode8)])
        cls.hevc, cls.ffv1, cls.encode10 = d / "source.mkv", d / "source_ffv1.mkv", d / "encode10.mkv"
        run(["ffmpeg", "-v", "error", "-y", "-filter_complex", PATTERN, "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
             "-x265-params", "lossless=1:log-level=error", str(cls.hevc)])
        run(["ffmpeg", "-v", "error", "-y", "-i", str(cls.hevc), "-c:v", "ffv1", str(cls.ffv1)])
        run(["ffmpeg", "-v", "error", "-y", "-i", str(cls.hevc), "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
             "-x265-params", "crf=38:log-level=error", str(cls.encode10)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def score(self, source, encode):
        out = Path(self.tmp.name) / "score.json"
        result = run([str(BINARY), str(source), str(encode), "--offset", "0", "--seconds", str(SECONDS), "--json", str(out)])
        return json.loads(out.read_text()), result.stderr

    def test_8_bit_video_matches_libvmaf_at_8_bits(self):
        mine, stderr = self.score(self.source8, self.encode8)
        self.assertIn("decodes on the processor", stderr)
        log = Path(self.tmp.name) / "libvmaf8.json"
        run(["ffmpeg", "-v", "error", "-i", str(self.encode8), "-i", str(self.source8), "-lavfi",
             f"[0:v]setpts=N/FRAME_RATE/TB[d];[1:v]setpts=N/FRAME_RATE/TB[r];[d][r]libvmaf=model=version=vmaf_v0.6.1:"
             f"feature=name=psnr|name=cambi\\\\:full_ref=true:n_threads=8:log_fmt=json:log_path={log}", "-frames:v", str(SECONDS * FPS), "-f", "null", "-"])
        cpu = json.loads(log.read_text())["frames"][:SECONDS * FPS]
        per = mine["encodes"][0]["per_frame"]
        for i, f in enumerate(cpu):
            m = f["metrics"]
            self.assertAlmostEqual(per["vmaf"][i], m["vmaf"], delta=1e-3)
            self.assertAlmostEqual(per["cambi"][i], m["cambi"], delta=1e-5)
            self.assertAlmostEqual(per["cambi_source"][i], m["cambi_source"], delta=1e-5)
            self.assertAlmostEqual(per["psnr_y"][i], m["psnr_y"], delta=1e-3)

    def test_a_source_decoded_on_the_processor_scores_the_same_as_on_the_gpu(self):
        gpu, _ = self.score(self.hevc, self.encode10)
        cpu, stderr = self.score(self.ffv1, self.encode10)
        self.assertIn("decodes on the processor", stderr)
        self.assertEqual(gpu["encodes"][0]["whole"], cpu["encodes"][0]["whole"])


if __name__ == "__main__":
    unittest.main()

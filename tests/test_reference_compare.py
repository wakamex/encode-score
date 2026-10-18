
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from encode_score import compare

RNG = np.random.default_rng(7)


def texture(height=128, width=128, period=8):
    """A picture with oriented structure: diagonal stripes `period` pixels apart."""
    y, x = np.mgrid[0:height, 0:width]
    return (512 + 120 * np.sin((x + y) * 2 * np.pi / period)).astype(np.float32)


def frame(luma):
    h, w = luma.shape
    flat = np.full((h // 2, w // 2), 512, dtype=np.uint16)
    return {"y": luma.astype(np.uint16), "u": flat, "v": flat}


class Box(unittest.TestCase):
    def test_box_matches_a_direct_mean_and_keeps_the_shape(self):
        a = RNG.random((20, 30)).astype(np.float32)
        out = compare.box(a, 3)
        self.assertEqual(out.shape, a.shape)
        self.assertAlmostEqual(float(out[5, 7]), float(a[4:7, 6:9].mean()), places=5)
        self.assertAlmostEqual(float(out[0, 0]), float(np.pad(a, 1, mode="edge")[0:3, 0:3].mean()), places=5)


class WindowEnergies(unittest.TestCase):
    def test_grain_has_no_oriented_energy_and_stripes_have_a_lot(self):
        noise = (512 + RNG.normal(0, 20, (128, 128))).astype(np.float32)
        grain, stripes = compare.window_energies(noise), compare.window_energies(texture())
        self.assertGreater(stripes["oriented"].mean(), 20 * grain["oriented"].mean())
        self.assertGreater(grain["fine"].mean(), 0)

    def test_all_bands_share_one_window_grid(self):
        energies = compare.window_energies(texture(96, 160))
        self.assertEqual({tuple(v.shape) for v in energies.values()}, {(6, 10)})

    def test_blurring_removes_fine_and_oriented_detail(self):
        sharp = texture() + RNG.normal(0, 10, (128, 128)).astype(np.float32)
        soft = compare.box(sharp, 5)
        a, b = compare.window_energies(sharp), compare.window_energies(soft)
        self.assertLess(b["fine"].sum(), 0.5 * a["fine"].sum())


class Structure(unittest.TestCase):
    def test_structure_windows_are_the_ones_with_the_most_oriented_energy(self):
        grid = np.zeros((10, 10))
        grid[2:4, 3:6] = 5.0
        mask = compare.structure_mask(grid, 0.06)
        self.assertEqual(int(mask.sum()), 6)
        self.assertTrue(mask[2:4, 3:6].all())

    def test_flat_picture_has_no_structure_windows(self):
        self.assertFalse(compare.structure_mask(np.zeros((5, 5))).any())

    def test_kept_is_the_amplitude_ratio_in_percent(self):
        self.assertAlmostEqual(compare.kept(25.0, 100.0), 50.0)
        self.assertTrue(np.isnan(compare.kept(1.0, 0.0)))


class CompareFrames(unittest.TestCase):
    def setUp(self):
        base = texture(400, 400)
        self.ref = [frame(base + RNG.normal(0, 8, base.shape))]
        self.smooth = [frame(compare.box(self.ref[0]["y"].astype(np.float32), 7))]

    def test_identical_frames_keep_everything(self):
        result = compare.compare_frames(self.ref, self.ref)
        for region in result["regions"].values():
            for value in region.values():
                self.assertAlmostEqual(value, 100.0, places=3)
        self.assertEqual(result["color_ref"], result["color_dist"])

    def test_smoothing_loses_detail_and_names_the_windows_to_show(self):
        result = compare.compare_frames(self.ref, self.smooth)
        self.assertLess(result["regions"]["structure"]["fine"], 60)
        self.assertLess(result["regions"]["structure"]["oriented"], 90)
        self.assertEqual(len(result["worst"]), 3)
        ratios = [w[0] for w in result["worst"]]
        self.assertEqual(ratios, sorted(ratios))

    def test_removing_only_the_grain_keeps_the_oriented_detail(self):
        clean = texture(400, 400, period=40)
        grainy = [frame(clean + RNG.normal(0, 12, clean.shape))]
        result = compare.compare_frames(grainy, [frame(clean)])
        region = result["regions"]["all"]
        self.assertLess(region["fine"], 50)
        self.assertGreater(region["oriented"], 85)

    def test_identical_frames_match_completely(self):
        result = compare.compare_frames(self.ref, self.ref)
        for region in result["matching"].values():
            for value in region.values():
                self.assertAlmostEqual(value, 100.0, places=3)

    def test_fresh_grain_on_a_blurred_picture_has_the_amount_but_does_not_match(self):
        original = self.ref[0]["y"].astype(np.float32)
        lost = np.sqrt(np.mean(compare.bands(original)["fine"] ** 2))
        fake = [frame(compare.box(original, 9) + RNG.normal(0, lost * 1.06, original.shape))]
        result = compare.compare_frames(self.ref, fake)
        self.assertGreater(result["regions"]["all"]["fine"], 85)
        self.assertLess(result["matching"]["all"]["fine"], 20)
        self.assertLess(result["matching"]["median"]["fine"], 20)

    def test_a_loss_in_part_of_the_picture_shows_in_the_worst_tenth_of_windows(self):
        original = self.ref[0]["y"].astype(np.float32)
        patched = original.copy()
        patched[:, :80] = compare.box(original, 9)[:, :80]
        result = compare.compare_frames(self.ref, [frame(patched)])
        self.assertGreater(result["matching"]["median"]["mid"], 95)
        self.assertLess(result["matching"]["worst_tenth"]["mid"], 60)
        self.assertLess(result["regions"]["worst_tenth"]["mid"], result["regions"]["median"]["mid"])

    def test_color_strength_is_reported_for_both_files(self):
        dull = [{"y": self.ref[0]["y"], "u": np.full((200, 200), 520, np.uint16), "v": np.full((200, 200), 505, np.uint16)}]
        result = compare.compare_frames(self.ref, dull)
        self.assertAlmostEqual(result["color_ref"]["strength"], 0.0)
        self.assertGreater(result["color_dist"]["strength"], 0)


class Persistent(unittest.TestCase):
    def setUp(self):
        self.clean = texture(400, 400)
        self.ref = [frame(self.clean + RNG.normal(0, 8, self.clean.shape))]
        self.steady = [tuple(64 * (self.clean + RNG.normal(0, 3, self.clean.shape)) for _ in range(2))]

    def test_a_clean_copy_keeps_the_texture_although_its_grain_is_gone(self):
        result = compare.compare_frames(self.ref, [frame(self.clean)], self.steady)
        self.assertGreater(result["persistent"]["windows"], 50)
        self.assertGreater(result["persistent"]["median"], 95)

    def test_fresh_grain_on_a_flattened_picture_keeps_none_of_it(self):
        fake = [frame(512 + RNG.normal(0, 60, self.clean.shape))]
        result = compare.compare_frames(self.ref, fake, self.steady)
        self.assertLess(abs(result["persistent"]["median"]), 15)

    def test_a_picture_that_is_only_grain_has_no_persistent_windows(self):
        noise = lambda: (512 + RNG.normal(0, 20, (400, 400))).astype(np.uint16)
        grain = [frame(noise())]
        result = compare.compare_frames(grain, grain, [(noise(), noise())])
        self.assertLess(result["persistent"]["windows"], 5)

    def test_grain_that_happens_to_resemble_the_measured_frame_is_not_texture(self):
        noise = lambda: (512 + RNG.normal(0, 20, (400, 400))).astype(np.float32)
        now, other = noise(), noise()
        result = compare.compare_frames([frame(now)], [frame(now)], [((now + other) / 2, noise())])
        self.assertLess(result["persistent"]["windows"], 5)

    def test_without_later_frames_it_is_not_reported(self):
        self.assertNotIn("persistent", compare.compare_frames(self.ref, self.ref))


class Score(unittest.TestCase):
    def found(self, fine=100.0, median=100.0, worst=100.0, windows=500):
        return {"regions": {"all": {"fine": fine}}, "persistent": {"median": median, "worst_tenth": worst, "windows": windows}}

    def test_a_perfect_copy_scores_100(self):
        parts = compare.score(100.0, 0.5, 0.5, self.found())
        self.assertEqual({key: parts[key] for key in compare.WEIGHTS}, {key: 100.0 for key in compare.WEIGHTS})
        self.assertAlmostEqual(parts["score"], 100.0)

    def test_each_fault_lowers_only_its_own_part(self):
        banded = compare.score(100.0, 3.0, 0.5, self.found())
        self.assertAlmostEqual(banded["banding"], 50.0)
        self.assertAlmostEqual(banded["score"], 100 - 50 * compare.WEIGHTS["banding"])
        self.assertAlmostEqual(compare.score(100.0, 0.0, 0.5, self.found())["banding"], 100.0)
        self.assertAlmostEqual(compare.score(100.0, 0, 0, self.found(fine=40.0))["grain"], 40.0)
        self.assertAlmostEqual(compare.score(100.0, 0, 0, self.found(fine=130.0))["grain"], 70.0)
        self.assertAlmostEqual(compare.score(100.0, 0, 0, self.found(median=80.0, worst=-20.0))["texture"], 40.0)

    def test_texture_is_left_out_when_too_few_windows_persist(self):
        parts = compare.score(90.0, 0, 0, self.found(median=0.0, worst=0.0, windows=3))
        self.assertTrue(np.isnan(parts["texture"]))
        rest = 1 - compare.WEIGHTS["texture"]
        self.assertAlmostEqual(parts["score"], (90 * compare.WEIGHTS["fidelity"] + 100 * (rest - compare.WEIGHTS["fidelity"])) / rest)


class Montage(unittest.TestCase):
    def test_rows_hold_three_panels_and_png_round_trips(self):
        ref = [frame(texture(400, 400))]
        dist = [frame(compare.box(ref[0]["y"].astype(np.float32), 5))]
        picture = compare.montage(ref, dist, [(0.5, 0, 6, 6), (0.6, 0, 12, 14)])
        self.assertEqual(picture.shape, (2 * compare.PANEL * compare.ZOOM + 6, 3 * compare.PANEL * compare.ZOOM + 12))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.png"
            compare.write_png(path, picture)
            data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        self.assertEqual((height, width), picture.shape)
        idat = data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8]
        self.assertEqual(len(zlib.decompress(idat)), height * (width + 1))

    def test_no_windows_gives_a_small_placeholder(self):
        self.assertEqual(compare.montage([], [], []).shape, (8, 8))


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class EndToEnd(unittest.TestCase):
    def make(self, path, vf):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=2", "-vf", f"{vf}format=yuv420p10le",
                        "-c:v", "ffv1", str(path)], check=True)

    def test_a_blurred_copy_scores_below_an_identical_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, blurred = Path(tmp) / "a.mkv", Path(tmp) / "b.mkv"
            self.make(original, "")
            self.make(blurred, "gblur=sigma=2,")
            frames = []
            for path in (original, blurred):
                frames.append(compare.decode(str(path), 0.5, "null", 640, 360))
            self.assertEqual(frames[0]["y"].shape, (360, 640))
            same = compare.compare_frames([frames[0]], [frames[0]])
            soft = compare.compare_frames([frames[0]], [frames[1]])
            self.assertAlmostEqual(same["regions"]["all"]["mid"], 100.0, places=3)
            self.assertLess(soft["regions"]["structure"]["oriented"], 90)
            log = Path(tmp) / "v.json"
            scores = compare.run_libvmaf(str(blurred), 0, "null", str(original), 0, "null", 1.0, 2, log)
            self.assertLess(scores["vmaf"]["mean"], 99)
            self.assertIn("cambi", scores)
            self.assertGreater(scores["psnr_y"], 10)
            self.assertIn("pooled_metrics", json.loads(log.read_text()))

    def test_one_pass_returns_the_scores_and_the_frames_a_separate_decode_gives(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, blurred = Path(tmp) / "a.mkv", Path(tmp) / "b.mkv"
            self.make(original, "")
            self.make(blurred, "gblur=sigma=2,setparams=colorspace=bt709,")
            scores, ref, dist, steady = compare.measure(str(blurred), 0, "null", str(original), 0, "null", 2.0, (640, 360), 4, 2)
            self.assertEqual((len(ref), len(dist), len(steady)), (4, 4, 4))
            self.assertLess(scores["vmaf"]["mean"], 99)
            self.assertIn("cambi_original", scores)
            count = int(2.0 * 24) - 2 * compare.RADIUS - 1
            for index, pick in enumerate(compare.RADIUS + int(count * (i + 0.5) / 4) for i in range(4)):
                for frames, path, n in ((ref, original, pick), (dist, blurred, pick)):
                    alone = compare.decode(str(path), (n - 0.5) / 24, "null", 640, 360)
                    for plane in "yuv":
                        self.assertTrue(np.array_equal(frames[index][plane], alone[plane]))
                near = [compare.decode(str(original), (pick + k - 0.5) / 24, "null", 640, 360)["y"].astype(np.float64) for k in range(-compare.RADIUS, compare.RADIUS + 1)]
                for half, offsets in zip(steady[index], ((0, 2, 4, 8, 10, 12), (1, 3, 5, 7, 9, 11))):
                    self.assertLess(np.abs(half.astype(np.float64) * 1023 / 65535 - np.mean([near[k] for k in offsets], axis=0)).max(), 0.05)
            self.assertGreater(compare.bitrate(str(blurred), 0, 2.0), 0)


if __name__ == "__main__":
    unittest.main()

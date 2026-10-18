import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from encode_score import align as vmaf


class Cropping(unittest.TestCase):
    def test_matching_sizes_need_no_crop_and_larger_ones_are_cropped_centrally(self):
        self.assertEqual(vmaf.crop_filter((3840, 2160), (3840, 2160)), "null")
        self.assertEqual(vmaf.crop_filter((3840, 2160), (3840, 2076)), "crop=3840:2076:0:42")
        self.assertEqual(vmaf.crop_filter((3840, 2076), (3840, 2076)), "null")
        self.assertEqual(vmaf.crop_filter((3840, 2160), (3800, 2000)), "crop=3800:2000:20:80")


class Alignment(unittest.TestCase):
    @staticmethod
    def frames(values):
        return [bytes([v] * 4) for v in values]

    def test_best_lag_finds_the_shift_of_a_probe_inside_a_reference(self):
        reference = self.frames([0, 0, 0, 10, 50, 90, 30, 70, 20, 0, 0, 0, 0, 0])
        probe = self.frames([10, 50, 90, 30, 70])
        lag, residual = vmaf.best_lag(reference, probe, 5)
        self.assertEqual((lag, residual), (3 - 5, 0.0))

    def test_no_shift_gives_zero_and_a_mismatch_leaves_a_large_residual(self):
        reference = self.frames([5, 40, 80, 20, 60, 5, 5, 5, 5, 5, 5])
        self.assertEqual(vmaf.best_lag(reference, self.frames([5, 40, 80]), 0)[0], 0)
        lag, residual = vmaf.best_lag(self.frames([0] * 20), self.frames([200] * 5), 3)
        self.assertGreater(residual, 100)


class Pooling(unittest.TestCase):
    def test_pooled_reads_mean_and_the_tail_of_the_frame_scores(self):
        frames = [{"metrics": {"vmaf": 90.0 + i % 10}} for i in range(200)]
        result = vmaf.pooled({"frames": frames, "pooled_metrics": {"vmaf": {"mean": 94.5}}})
        self.assertEqual((result["mean"], result["min"], result["frames"]), (94.5, 90.0, 200))
        self.assertEqual(result["p1"], 90.0)

    def test_first_percentile_is_not_the_minimum_once_there_are_over_a_hundred_frames(self):
        frames = [{"metrics": {"vmaf": float(i)}} for i in range(152)]
        result = vmaf.pooled({"frames": frames, "pooled_metrics": {"vmaf": {"mean": 75.5}}})
        self.assertEqual((result["min"], result["p1"]), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()

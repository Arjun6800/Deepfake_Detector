"""Phase 4.5 numerical and experimental isolation regressions."""

import unittest

import cv2
import numpy as np

from deepfake_rppg.landmarks import measure_masks
from deepfake_rppg.quality_algorithms import ALGORITHMS, project
from deepfake_rppg.quality_capture import bounded_dilate, rigid_align, select
from deepfake_rppg.quality_experiment import score
from deepfake_rppg.signal import pos, resample_observations
from deepfake_rppg.window_features import rppg_features, trace_features


class QualityExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fps = 30
        self.time = np.arange(240) / self.fps
        pulse = np.sin(2 * np.pi * 1.25 * self.time)
        rng = np.random.default_rng(0)
        self.rgb = np.array([155., 111., 84.]) + pulse[:, None] * [0.45, 1.2, 0.25]
        self.rgb += rng.normal(0, .05, self.rgb.shape)
        self.groups = np.ones(len(self.time), dtype=int)

    def test_all_methods_recover_clean_frequency(self):
        for method in ALGORITHMS:
            with self.subTest(method=method):
                rows, _, _ = score(method, self.rgb[:, None], self.groups[:, None], self.fps)
                self.assertAlmostEqual(rows[0]["pulse_bpm_estimate"], 75)

    def test_native_scoring_matches_phase4(self):
        rgb = np.repeat(self.rgb[:, None], 3, axis=1)
        groups = np.repeat(self.groups[:, None], 3, axis=1)
        ref = rppg_features(rgb, rgb, groups, self.fps)
        for method in ("GREEN", "POS"):
            rows, _, _ = score(method, rgb, groups, self.fps)
            for metric in ("spectral_concentration", "processed_coverage", "quality_pass"):
                self.assertEqual(rows[0][metric], ref[f"rppg_forehead_{method.lower()}_{metric}"])

    def test_pos_projection_matches_single_published_window(self):
        rgb = self.rgb[:48]
        r, g, b = (rgb / rgb.mean(axis=0)).T
        s1, s2 = g - b, -2 * r + g + b
        expected = s1 + s2 * s1.std() / s2.std()
        np.testing.assert_allclose(pos(rgb, self.fps), expected - expected.mean())

    def test_gate_accepts_a_periodic_nonphysiological_artifact(self):
        artifact = np.sin(2 * np.pi * 1.625 * self.time)
        metrics, _ = trace_features(artifact, self.fps, self.groups)
        self.assertEqual(metrics["quality_pass"], 1)
        self.assertAlmostEqual(metrics["pulse_bpm_estimate"], 97.5)

    def test_algorithms_do_not_bridge_long_gaps_or_group_changes(self):
        rgb = np.tile(self.rgb, (3, 1))
        rgb[240:285] = np.nan
        groups = np.r_[np.ones(480), np.full(240, 2)]
        for method in ALGORITHMS:
            with self.subTest(method=method):
                pulse = project(method, rgb, self.fps, groups)
                self.assertTrue(np.isnan(pulse[240:285]).all())
                np.testing.assert_equal(pulse[480:], project(method, self.rgb, self.fps, self.groups))
        grid, values, codes = resample_observations(
            np.arange(len(rgb)) / self.fps, rgb[:, None], self.fps, groups[:, None])
        self.assertEqual(len(grid), len(rgb))
        self.assertTrue(np.isnan(values[240:285]).all())
        self.assertNotEqual(codes[479, 0], codes[480, 0])

    def test_dilation_matches_baseline_interior_and_edges(self):
        for y, x in ((0, 0), (25, 34), (47, 55)):
            mask = np.zeros((54, 64), dtype=np.uint8)
            mask[y:y + 7, x:x + 9] = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            np.testing.assert_array_equal(bounded_dilate(mask, kernel, 7), cv2.dilate(mask, kernel))

    def test_native_bgr_pixels_are_exported_as_rgb(self):
        frame = np.full((5, 5, 3), [38, 91, 169], dtype=np.uint8)
        rgb, *_ = measure_masks(frame, np.ones((3, 5, 5), dtype=bool), np.zeros(2))
        np.testing.assert_array_equal(rgb, [[169, 91, 38]] * 3)

    def test_similarity_transform_recovers_rotation_scale_translation(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(468, 2))
        new = 1.4 * points @ np.array([[.6, .8], [-.8, .6]]) + [2.5, -4.]
        np.testing.assert_allclose(rigid_align(points, new), new, atol=1e-12)

    def test_selection_is_complete_and_train_only(self):
        from deepfake_rppg.dataset import METHODS

        rows = []
        for partition in ("train", "val", "test"):
            for method in ("original", *METHODS):
                for i in range(2):
                    rows.append(dict(partition=partition, group_id=partition, duration=31,
                                     video_id=f"{partition}/{method}/{i}", manipulation_method=method,
                                     validation_status="audited", label=int(method != "original"),
                                     source_video_ids=[str(i)] if method == "original" else ["0", "1"]))
        selected = select(rows, 451, 1, include_long=True)
        self.assertEqual(len(selected), 12)
        self.assertEqual({row["partition"] for row in selected}, {"train"})
        self.assertEqual(selected, select(list(reversed(rows)), 451, 1, include_long=True))


if __name__ == "__main__":
    unittest.main()

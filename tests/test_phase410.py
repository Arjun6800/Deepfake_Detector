"""Regression checks for support, provenance, features and evaluation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from deepfake_rppg.phase410_features import (VARIANTS, align, cross_modal, spectral, synthetic_control,
                                             temporal, waveform)
from deepfake_rppg.phase410_run import checkpoint, nested_fold


class Features(unittest.TestCase):
    fps = 30

    def test_alignment_and_collisions(self):
        grid = np.arange(5) / 30
        times = np.array([0, .9, 1.1, 3.3, 4]) / 30
        actual = align(times, np.arange(5.)[:, None], grid, 30)[:, 0]
        np.testing.assert_equal(actual, [0, np.nan, np.nan, np.nan, 4])
        with self.assertRaises(ValueError):
            align(times[::-1], np.arange(5.), grid, 30)

    def test_features_on_known_period(self):
        time = np.arange(240) / self.fps
        wave = np.sin(2 * np.pi * 1.25 * time)
        temporal_values = temporal(wave, self.fps)
        self.assertAlmostEqual(temporal_values["acf_peak_lag_s"], .8, places=2)
        self.assertAlmostEqual(temporal_values["hjorth_activity"], .5)
        self.assertAlmostEqual(spectral(wave, self.fps)["centroid_hz"], 1.25)
        self.assertTrue(np.isnan(spectral(wave[:95], self.fps)["centroid_hz"]))

    def test_no_fabricated_observations(self):
        t = np.arange(360) / self.fps
        rgb = np.array([100., 85., 75.]) + np.sin(t * 8)[:, None] * [.3, .7, .2]
        colors = np.repeat(rgb[:, None, :], 3, axis=1)
        colors[140:150] = np.nan
        ids = np.zeros((360, 3), int)
        ids[300:] = 1
        for variant in VARIANTS:
            regional, aggregate, provenance = waveform(colors, ids, np.zeros((360, 4)), self.fps, variant)
            self.assertTrue(np.isnan(regional[140:150]).all())
            self.assertTrue(np.isnan(aggregate[140:150]).all())
            self.assertTrue(np.isfinite(aggregate).any())
            self.assertNotEqual(provenance[299], provenance[300])

    def test_synthetic_signal_and_nuisance(self):
        reference = synthetic_control("pos_reference")
        combined = synthetic_control("pos_combined")
        self.assertGreater(combined["clean_pulse_abs_r"], .95)
        self.assertGreater(combined["contaminated_pulse_abs_r"], reference["contaminated_pulse_abs_r"])
        self.assertLess(combined["contamination_motion_abs_r"], reference["contamination_motion_abs_r"])

    def test_cross_modal_alignment_and_insufficient_joint_support(self):
        t = np.arange(240) / self.fps
        wave = np.sin(t * 8)
        scores = abs(wave[:, None])
        groups, speed = np.zeros(len(t)), t.copy()
        values = cross_modal(wave, groups, scores, speed, self.fps)
        self.assertAlmostEqual(values["amplitude_activity_r"], 1., places=2)
        self.assertEqual(values["lag_s"], 0)
        scores[::10] = np.nan
        self.assertTrue(all(np.isnan(v) for v in cross_modal(wave, groups, scores, speed, self.fps).values()))


class Evaluation(unittest.TestCase):
    def test_checkpoint_resume_and_invalidate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "nested" / "checkpoint.json"
            with patch("deepfake_rppg.phase410_run.space"):
                self.assertEqual(checkpoint(path, "key", lambda: {"value": 1}), {"value": 1})
                self.assertEqual(checkpoint(path, "key", lambda: self.fail("recomputed")), {"value": 1})
                self.assertEqual(checkpoint(path, "new", lambda: {"value": 2}), {"value": 2})
                path.write_text("{invalid")
                self.assertEqual(checkpoint(path, "new", lambda: {"value": 3}), {"value": 3})

    def test_nested_isolation_and_determinism(self):
        items, arrays, outer = [], [], []
        rng = np.random.default_rng(41)
        for family in range(10):
            for label in (0, 1):
                items.append(dict(video_id=f"{family}-{label}", group_id=str(family), phase48_cohort="new",
                                  label=label, manipulation_method="fake" if label else "original"))
                outer.append(family % 5 + 1)
                arrays.append(np.array([[label + rng.normal(), np.nan if family % 2 else rng.normal()]]))
        outer = np.array(outer)
        fields = [("obs", "x"), ("obs", "z")]
        schema = dict(expression=[], rppg=dict(example=["x", "z"]))
        with threadpool_limits(limits=1):
            result = nested_fold(items, arrays, fields, schema, outer, 1, "logistic")
            self.assertEqual(result, nested_fold(items, arrays, fields, schema, outer, 1, "logistic"))
            changed = [a + 999 if fold == 1 else a for a, fold in zip(arrays, outer, strict=True)]
            perturbed = nested_fold(items, changed, fields, schema, outer, 1, "logistic")
            self.assertEqual(result["setting"], perturbed["setting"])
            self.assertEqual(result["threshold"], perturbed["threshold"])
            self.assertNotEqual(result["rows"], perturbed["rows"])


if __name__ == "__main__":
    unittest.main()


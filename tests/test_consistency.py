"""Regression tests for the BPM-independent feature path."""
import unittest

import numpy as np

from deepfake_rppg.consistency import aggregate, describe, one_trace, pair_features, reduce_nuisance, robust_trace
from deepfake_rppg.consistency_experiment import choose_regions
from deepfake_rppg.window_features import trace_features


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.fps = 30
        self.time = np.arange(240) / self.fps
        self.pulse = np.sin(2 * np.pi * 1.25 * self.time)
        self.ids = np.ones(len(self.time), dtype=int)

    def test_subthreshold_trace_keeps_waveform_and_features(self):
        rng = np.random.default_rng(45)
        raw = rng.normal(size=len(self.time))
        wave = robust_trace(raw, self.ids, self.fps)
        result, hz, psd = one_trace(raw, wave, self.ids, self.fps)
        baseline, _ = trace_features(raw, self.fps, self.ids)
        self.assertEqual(result["reliability"]["legacy_gate"], baseline["quality_pass"])
        self.assertEqual(result["reliability"]["legacy_gate"], 0)
        self.assertTrue(np.isnan(baseline["pulse_bpm_estimate"]))
        self.assertTrue(np.isfinite(wave).all())
        self.assertTrue(np.isfinite(result["spectral"]["entropy"]))
        self.assertEqual(len(hz), len(psd))
        self.assertAlmostEqual(sum(result["spectral"][f"relative_power_{i}"] for i in range(5)), 1.)

    def test_joint_phase_features_require_support(self):
        paired_ids = np.tile(self.ids[:, None], (1, 2))
        result, _ = pair_features(self.pulse, -2 * self.pulse, paired_ids, self.fps)
        self.assertAlmostEqual(result["correlation"], -1.)
        self.assertAlmostEqual(result["coherence"], 1.)
        self.assertAlmostEqual(result["phase_consistency"], 1.)
        self.assertAlmostEqual(result["spectral_disagreement"], 0.)
        paired_ids[90:160, 1] = 2
        unavailable, reliability = pair_features(self.pulse, self.pulse, paired_ids, self.fps)
        self.assertTrue(np.isnan(unavailable["phase_consistency"]))
        self.assertEqual(reliability["joint_coverage"], 1)

    def test_aggregation_keeps_gaps_and_tracks_membership(self):
        regional = np.tile(self.pulse[:, None], (1, 3))
        groups = np.tile(self.ids[:, None], (1, 3))
        regional[40:80, 2] = np.nan
        regional[80:120, 1:] = np.nan
        wave, provenance = aggregate(regional, groups, np.array([1., .5, 1.]))
        self.assertTrue(np.isfinite(wave[40:80]).all())
        self.assertTrue(np.isnan(wave[80:120]).all())
        self.assertNotEqual(provenance[39], provenance[40])
        self.assertNotEqual(provenance[119], provenance[120])

    def test_measured_nuisance_is_attenuated_without_filling_gaps(self):
        artifact = np.sin(2 * np.pi * 1.75 * self.time)
        raw = self.pulse + 2 * artifact
        raw[200:205] = np.nan
        corrected = reduce_nuisance(raw, artifact[:, None], self.ids, self.fps)
        self.assertTrue(np.isnan(corrected[200:205]).all())
        self.assertLess(abs(np.dot(corrected[:200], artifact[:200])), .6 * abs(np.dot(raw[:200], artifact[:200])))

    def test_schema_stable_with_missing_regions(self):
        raw = np.tile(self.pulse[:, None], (1, 4))
        ids = np.tile(self.ids[:, None], (1, 4))
        valid, _ = describe(raw, raw, ids, self.fps)
        empty, _ = describe(np.full_like(raw, np.nan), np.full_like(raw, np.nan), ids, self.fps)
        self.assertEqual(list(valid), list(empty))
        self.assertEqual(len(valid), 89)
        self.assertTrue(np.isnan(empty["spatial__forehead--cheek_image_left__coherence"]))
        self.assertEqual(empty["reliability__forehead__waveform_coverage"], 0)

    def test_adaptive_mask_requires_coverage_and_respects_provenance(self):
        base = np.tile(self.pulse[:, None, None], (1, 3, 3)) + 100
        alternative = (base - 100) * .5 + 100
        alternative[40:80, 1] = np.nan
        ids = np.tile(self.ids[:, None], (1, 3))
        rgb, provenance, chosen = choose_regions(base, alternative, ids, ids + 2, [True, True, False])
        self.assertEqual(chosen, [0])
        np.testing.assert_array_equal(rgb[:, 0], alternative[:, 0])
        np.testing.assert_array_equal(rgb[:, 1:], base[:, 1:])
        np.testing.assert_array_equal(provenance[:, 0], ids[:, 0] + 2)


if __name__ == "__main__":
    unittest.main()

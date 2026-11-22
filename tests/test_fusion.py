"""Window alignment, missingness, feature math, and preprocessing boundaries."""

import unittest

import numpy as np

from deepfake_rppg.expressions import SELECTED
from deepfake_rppg.fusion import FRAME_KEYS, build_windows, require_alignment, time_slice, window_bounds
from deepfake_rppg.window_features import expression_features, rppg_features, snap_rgb, trace_features


class FusionTests(unittest.TestCase):
    def test_alignment_rejects_shift_reordered_frames_and_bad_time(self):
        a = np.zeros(4, dtype=[(key, float) for key in FRAME_KEYS])
        a["frame_index"] = np.arange(4)
        a["analysis_timestamp_s"] = np.arange(4) / 24
        a["source_timestamp_s"] = a["analysis_timestamp_s"] + 3
        require_alignment(a, a.copy())
        bad = a.copy()
        bad["source_timestamp_s"] += 1 / 24
        with self.assertRaisesRegex(ValueError, "source_timestamp_s"):
            require_alignment(a, bad)
        bad = a.copy()
        bad["analysis_timestamp_s"] *= 1.001
        with self.assertRaisesRegex(ValueError, "analysis_timestamp_s"):
            require_alignment(a, bad)
        with self.assertRaisesRegex(ValueError, "Frame indices"):
            require_alignment(a, a[::-1])
        with self.assertRaises(ValueError):
            require_alignment(a, a[:-1])

    def test_half_open_windows_include_exact_end_and_drop_partial_tail(self):
        self.assertEqual(list(window_bounds(10.5, 4, 2)), [(0, 4), (2, 6), (4, 8), (6, 10)])
        self.assertEqual(list(window_bounds(4, 4, 0)), [(0, 4)])
        self.assertEqual(list(window_bounds(3.99, 4, 0)), [])
        times = np.array([0, 1, 2, 3.999999999999, 5])
        self.assertEqual(time_slice(times, 2, 4), slice(2, 3))
        with self.assertRaises(ValueError):
            list(window_bounds(10, 4, 4))

    def test_grid_mapping_drops_collisions_and_never_fills_gaps(self):
        fps = 10
        times = np.array([0, 0.09, 0.11, 0.20, 0.34, 0.40])
        rgb = np.broadcast_to(np.arange(6.)[:, None, None], (6, 3, 3))
        groups = np.broadcast_to(np.arange(6)[:, None], (6, 3))
        sampled, source = snap_rgb(times, rgb, groups, np.arange(5) / fps, fps)
        np.testing.assert_allclose(sampled[:, 0, 0], [0, np.nan, 3, np.nan, 5])
        np.testing.assert_array_equal(source[:, 0], [0, 0, 3, 0, 5])

    def test_periodic_trace_features_and_signal_gaps(self):
        fps = 24
        times = np.arange(8 * fps) / fps
        pulse = np.sin(2 * np.pi * 1.25 * times)
        groups = np.zeros(len(times))
        metrics, processed = trace_features(pulse, fps, groups)
        self.assertEqual(metrics["quality_pass"], 1)
        self.assertAlmostEqual(metrics["dominant_hz"], 1.25)
        self.assertAlmostEqual(metrics["pulse_bpm_estimate"], 75)
        self.assertAlmostEqual(metrics["prefilter_variance"], 0.5)
        self.assertGreater(metrics["spectral_concentration"], 0.9)
        self.assertGreater(metrics["inband_to_outband_db"], 10)
        self.assertTrue(np.isfinite(processed).all())
        groups[4 * fps:] = 1
        split, _ = trace_features(pulse, fps, groups)
        self.assertEqual(split["processed_coverage"], 1)
        self.assertTrue(np.isnan(split["pulse_bpm_estimate"]))
        pulse[3 * fps:5 * fps] = np.nan
        missing, processed = trace_features(pulse, fps, groups)
        self.assertEqual(missing["raw_coverage"], 0.75)
        self.assertEqual(missing["quality_pass"], 0)
        self.assertTrue(np.isnan(missing["pulse_bpm_estimate"]))
        self.assertTrue(np.isnan(processed).all())

    def test_interregion_support_and_expression_asymmetry(self):
        fps = 24
        times = np.arange(8 * fps) / fps
        pulse = np.sin(2 * np.pi * 1.25 * times)
        colors = [100 + pulse * 0.3, 80 + pulse * 1.3, 60 + pulse * 0.5]
        rgb = np.tile(np.column_stack(colors)[:, None, :], (1, 3, 1))
        rgb[:, 2] = np.nan
        features = rppg_features(rgb, rgb, np.zeros(rgb.shape[:2]), fps)
        pair = "rppg_pair_forehead__cheek_image_left_"
        self.assertAlmostEqual(features[pair + "pos_peak_delta_hz"], 0)
        self.assertAlmostEqual(features[pair + "pos_correlation"], 1)
        absent = "rppg_pair_forehead__cheek_image_right_"
        self.assertEqual(features[absent + "joint_processed_coverage"], 0)
        self.assertTrue(np.isnan(features[absent + "pos_correlation"]))
        times = np.arange(5) / 10
        scores = np.full((len(times), len(SELECTED)), np.nan)
        scores[:, SELECTED.index("eyeBlinkLeft")] = [0, 0.1, np.nan, 0.9, 1]
        scores[:, SELECTED.index("eyeBlinkRight")] = [0, 0, 0, 0.2, np.nan]
        expr = expression_features(times, scores, 10)
        self.assertAlmostEqual(expr["expr_asym_eyeBlink_paired_coverage"], 0.6)
        self.assertAlmostEqual(expr["expr_asym_eyeBlink_mean_left_minus_right"], (0 + 0.1 + 0.7) / 3)
        self.assertAlmostEqual(expr["expr_eyeBlinkLeft_mean_abs_velocity_per_s"], 1)
        self.assertTrue(np.isnan(expr["expr_eyeBlinkLeft_mean_rolling_std_0p5s"]))
        self.assertTrue(np.isnan(expr["expr_asym_cheekSquint_mean_abs_difference"]))

    def test_fused_keys_and_window_local_preprocessing(self):
        fps = 24
        times = np.arange(16 * fps) / fps
        rgb = np.full((len(times), 3, 3), 100.0)
        groups = np.zeros(rgb.shape[:2], dtype=int)
        scores = np.full((len(times), len(SELECTED)), 0.2)
        before = build_windows(times, times, rgb, groups, scores, fps, 8, 0, "clip", "source")
        # A large change after the first window may affect only the following window.
        rgb[8 * fps:] += np.sin(2 * np.pi * times[8 * fps:])[:, None, None]
        scores[8 * fps:] = 0.8
        after = build_windows(times, times, rgb, groups, scores, fps, 8, 0, "clip", "source")
        for table1, table2 in zip(before, after, strict=True):
            for key, value in table1[0].items():
                other = table2[0][key]
                if isinstance(value, (float, np.floating)) and np.isnan(value):
                    self.assertTrue(np.isnan(other), key)
                else:
                    self.assertEqual(value, other, key)
        first, second = after[2]
        self.assertEqual(first, {**after[0][0], **after[1][0]})
        self.assertEqual(second["split_group"], first["split_group"])
        self.assertEqual(second["first_frame_index"], first["last_frame_index"] + 1)

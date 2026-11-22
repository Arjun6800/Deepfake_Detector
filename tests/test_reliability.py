"""Regression tests for ROI support, stabilization, and hybrid signal seams."""

import unittest
from unittest.mock import patch

import numpy as np

from deepfake_rppg.landmarks import measure_masks
from deepfake_rppg.reliability import (
    LANDMARK_UPDATE_WEIGHT, SAMPLING_VERTICES, LandmarkStabilizer, RegionMeasurements,
    choose_fallback, reliable_masks, rectangular_masks,
)
from deepfake_rppg.signal import finite_runs, pos, process_signal, resample_observations


class ReliabilityTests(unittest.TestCase):
    def test_smoothing_follows_translation_and_scale_and_resets_for_large_changes(self):
        smoother = LandmarkStabilizer()
        box = np.array([50, 100, 120, 160])
        points = np.tile([0.3, 0.6], (478, 1))
        smoother.update(points * box[2:] + box[:2], box)
        moved_box = np.array([62, 110, 160, 180])
        moved = points.copy()
        moved[SAMPLING_VERTICES, 0] += 0.01
        observed = moved * moved_box[2:] + moved_box[:2]
        expected = (points + LANDMARK_UPDATE_WEIGHT * (moved - points)) * moved_box[2:] + moved_box[:2]
        np.testing.assert_allclose(smoother.update(observed, moved_box), expected)
        observed += [16, 0]
        np.testing.assert_allclose(smoother.update(observed, moved_box), observed)
        smoother.reset()
        observed += [1, 0]
        np.testing.assert_allclose(smoother.update(observed, moved_box), observed)

    def test_current_allowed_area_constrains_stabilized_shape_and_fallback(self):
        points = np.array([[2, 2], [20, 2], [20, 20], [2, 20]])
        allowed = np.zeros((32, 32), np.uint8)
        allowed[5:26, 5:26] = 1
        allowed[10:14, 10:14] = 0
        regions = dict.fromkeys(("forehead", "cheek_image_left", "cheek_image_right"), range(4))
        rectangles = np.ones((3, *allowed.shape), bool)
        with patch("deepfake_rppg.reliability.RELIABLE_POLYGONS", regions), patch(
            "deepfake_rppg.reliability.allowed_skin", return_value=allowed,
        ):
            primary, fallback = reliable_masks(points, points + [2, 0], allowed.shape, 32, rectangles)
        for masks in (primary, fallback):
            self.assertFalse(masks[:, allowed == 0].any())
            self.assertTrue(masks[:, 8, 8].all())
            self.assertFalse(masks[:, 25, 25].any())
        clipped = rectangular_masks((10, 10), np.array([12, 12]), [0, 0, 20, 20])
        self.assertFalse(clipped[0].any())

    def test_pixel_failure_categories_and_fallback_are_per_region(self):
        crop = np.full((8, 8, 3), 100)
        crop[:4, :4] = 0
        masks = np.zeros((3, 8, 8), dtype=bool)
        masks[1, 4:6, 4:6] = True
        masks[2, :4, :4] = True
        result = RegionMeasurements(*measure_masks(crop, masks, [0, 0]))
        np.testing.assert_array_equal(result.failures("accepted"), [
            "empty_geometry", "insufficient_mask_pixels", "insufficient_exposure_valid_pixels",
        ])
        self.assertEqual(result.failures("face_tracking"), ["face_tracking"] * 3)
        fallback_masks = masks.copy()
        fallback_masks[1, 4:8, 4:8] = True
        fallback = RegionMeasurements.from_masks(crop, fallback_masks, [0, 0])
        merged, sources = choose_fallback(result, fallback)
        np.testing.assert_array_equal(sources, [0, 2, 0])
        np.testing.assert_array_equal(merged.rgb[1], [100, 100, 100])
        self.assertTrue(np.isnan(merged.rgb[[0, 2]]).all())
        _, sources = choose_fallback(fallback, fallback)
        np.testing.assert_array_equal(sources, [0, 1, 0])

    def test_source_switch_and_long_gap_do_not_create_rgb_or_pos(self):
        fps = 24
        times = np.arange(150) / fps
        rgb = np.full((len(times), 1, 3), 100.0)
        rgb[20:22] = rgb[75:90] = np.nan
        groups = np.ones((len(times), 1), int)
        groups[22:42] = 2
        _, colors, sampled_sources = resample_observations(times, rgb, fps, groups)
        self.assertTrue(np.isnan(colors[20:22]).all())
        self.assertTrue(np.isnan(colors[75:90]).all())
        self.assertTrue(np.isfinite(colors[22:42]).all())
        # The short fallback run has too few samples for a POS window or filtering.
        source = sampled_sources[:, 0]
        self.assertTrue(np.isnan(pos(colors[:, 0], fps, source)[22:42]).all())
        waveform = np.sin(2 * np.pi * times)
        signal_source = np.where(np.arange(len(times)) < 80, 1, 2)
        processed, quality = process_signal(waveform, fps, groups=signal_source)
        self.assertTrue(np.isnan(processed).all())
        self.assertEqual(quality["processed_coverage"], 0)

    def test_off_grid_fallback_still_splits_the_landmark_runs(self):
        times = np.array([0, 0.10, 0.14, 0.20, 0.30])
        rgb = np.full((len(times), 1, 3), 100.0)
        sources = np.array([1, 1, 2, 1, 1])[:, None]
        _, colors, run_ids = resample_observations(times, rgb, 10, sources)
        runs = list(finite_runs(np.isfinite(colors[:, 0, 0]), run_ids[:, 0]))
        self.assertEqual(runs, [slice(0, 2), slice(2, 4)])

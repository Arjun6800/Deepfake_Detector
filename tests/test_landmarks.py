"""Landmark mask geometry and missing observations without a model dependency."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from deepfake_rppg.landmarks import FACE_OVAL, LandmarkSampler, skin_masks


class LandmarkTests(unittest.TestCase):
    def test_masks_clip_to_face_and_exclude_eyes_brows_and_lips(self):
        def square(x, y, size):
            return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]

        points = np.array(
            square(8, 8, 44) + square(0, 0, 60) + square(8, 25, 22)
            + square(30, 25, 22) + square(16, 23, 4) + square(39, 23, 4)
            + square(16, 17, 4) + square(39, 17, 4) + square(27, 36, 6)
        )
        polygons = dict(zip(
            ("forehead", "cheek_image_left", "cheek_image_right"),
            (range(4, 8), range(8, 12), range(12, 16)),
        ))
        with patch.multiple(
            "deepfake_rppg.landmarks",
            FACE_OVAL=range(4), SKIN_POLYGONS=polygons,
            FEATURES=tuple(range(i, i + 4) for i in range(16, 36, 4)),
        ):
            masks = skin_masks(points, (64, 64))
        for x, y in [(4, 4), (8, 30), (18, 25), (41, 25),
                     (18, 19), (41, 19), (30, 39), (30, 35)]:
            self.assertFalse(masks[:, y, x].any(), (x, y))
        self.assertTrue(masks[0, 12, 30])
        self.assertTrue(masks[1, 35, 15])
        self.assertTrue(masks[2, 35, 47])

    def test_misses_and_ambiguous_faces_leave_no_samples_and_preserve_track(self):
        sampler = LandmarkSampler.__new__(LandmarkSampler)
        sampler.previous = None
        angles = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
        points = np.full((478, 2), 45.0)
        points[list(FACE_OVAL)] = 45 + 20 * np.column_stack((np.cos(angles), np.sin(angles)))
        sampler.detect = Mock(side_effect=[
            [points], [], [points, points], [points[:8]], [points + 120], [points + 1],
        ])
        frame = np.full((90, 90, 3), [70, 110, 150], dtype=np.uint8)
        sample = sampler.sample_with_geometry
        with patch("deepfake_rppg.landmarks.skin_masks") as masks:
            masks.side_effect = lambda points, shape: np.ones((3, *shape), dtype=bool)
            rgb, coverage, detected, _ = sample(frame)
            self.assertTrue(detected)
            np.testing.assert_array_equal(rgb, np.tile([150, 110, 70], (3, 1)))
            np.testing.assert_array_equal(coverage, 1)
            for missing in (None, frame, frame, frame, frame):
                rgb, coverage, detected, geometry = sample(missing)
                self.assertFalse(detected)
                self.assertTrue(np.isnan(rgb).all())
                self.assertTrue(np.isnan(geometry).all())
                np.testing.assert_array_equal(coverage, 0)
            self.assertTrue(sample(frame)[2])
            self.assertEqual(masks.call_count, 2)

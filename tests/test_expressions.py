"""Expression features preserve frame time, observed zeroes, and detection gaps."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from deepfake_rppg.expressions import (
    SELECTED, ExpressionSampler, check_alignment, export_features, temporal_features, write_columns,
)


class ExpressionTests(unittest.TestCase):
    def test_missing_track_and_missing_channels_are_never_filled(self):
        sampler = ExpressionSampler.__new__(ExpressionSampler)
        sampler.face_status = "face_tracking"
        sampler.blendshapes = [
            SimpleNamespace(category_name="eyeBlinkLeft", score=0),
            SimpleNamespace(category_name="eyeBlinkRight", score=np.nan),
        ]
        sampler.select_face = Mock(side_effect=[None, object(), object()])
        self.assertEqual(sampler.sample("missing"), ("face_tracking", {}))
        self.assertEqual(sampler.sample("partial"), ("partial_blendshapes", {"eyeBlinkLeft": 0.0}))
        sampler.blendshapes = []
        self.assertEqual(sampler.sample("no scores"), ("blendshapes_unavailable", {}))

    def test_temporal_calculation_uses_actual_time_and_restarts_at_gaps(self):
        times = np.array([0, 0.10, 0.20, 0.31, 0.41, 0.51, 0.62, 0.90, 1.00, 1.10])
        values = np.column_stack((2 * times, times))
        values[3, 1] = np.nan
        means, std, velocity = temporal_features(times, values, 10)
        np.testing.assert_allclose(velocity[[1, 2, 3, 4, 5, 6, 8, 9], 0], 2)
        self.assertTrue(np.isnan(velocity[[0, 7], 0]).all())
        self.assertTrue(np.isnan(velocity[[3, 4, 7], 1]).all())
        self.assertTrue(np.isnan(means[[3, 4, 5, 7, 8], 1]).all())
        self.assertAlmostEqual(means[6, 0], values[2:7, 0].mean())
        self.assertAlmostEqual(std[6, 0], values[2:7, 0].std())
        self.assertAlmostEqual(means[9, 1], times[7:10].mean())
        self.assertTrue(all(np.isnan(a).all() for a in temporal_features(times, values, None)))

    def test_export_retains_indices_blank_scores_and_reference_alignment(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            reference = output / "frame_traces.csv"
            source = 3 + np.arange(5) / 10
            analysis = source - source[0]
            write_columns(reference, {
                "frame_index": np.arange(5), "source_timestamp_s": source,
                "analysis_timestamp_s": analysis, "face_detected": [1, 1, 0, 1, 1],
            })
            complete = dict.fromkeys(SELECTED, 0.0)
            complete["extra"] = 0.7
            observations = [
                ("accepted", complete), ("accepted", complete), ("face_tracking", {}),
                ("accepted", complete), ("accepted", complete),
            ]
            metadata = export_features(
                Path("video.mp4"), Path("model.task"), output, (source, observations, 10, "mock"),
                0, 0, alignment_reference=reference,
            )
            self.assertEqual(metadata["coverage"], 0.8)
            self.assertEqual(metadata["frames_with_all_selected_scores"], 4)
            self.assertTrue(metadata["alignment"]["frame_and_timestamp_match"])
            self.assertEqual(metadata["alignment"]["shared_expression_and_rppg_face_frames"], 4)
            mismatch = check_alignment(reference, source + 0.001, analysis, np.ones(5, bool))
            self.assertFalse(mismatch["frame_and_timestamp_match"])
            with (output / "blendshapes.csv").open(newline="", encoding="utf-8") as stream:
                scores = list(csv.DictReader(stream))
            self.assertEqual(scores[0]["eyeBlinkLeft"], "0.0")
            self.assertEqual(scores[0]["extra"], "0.7")
            self.assertEqual(scores[2]["frame_index"], "2")
            self.assertEqual(scores[2]["has_blendshapes"], "0")
            self.assertEqual(scores[2]["eyeBlinkLeft"], "")
            with (output / "temporal_features.csv").open(newline="", encoding="utf-8") as stream:
                dynamics = list(csv.DictReader(stream))
            self.assertEqual(dynamics[3]["eyeBlinkLeft_velocity_per_s"], "")
            self.assertEqual(dynamics[4]["eyeBlinkLeft_velocity_per_s"], "0.0")

"""Fixture-only checks for Phase 4.8. No real video decoding."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from deepfake_rppg import phase48_extract as extraction

try:
    import sklearn
except ImportError:
    sklearn = None


class ExtractionTests(unittest.TestCase):
    def test_center_and_regular_windows_deduplicate(self):
        times = np.arange(480) / 30
        result = extraction.windows(times, 30)
        self.assertEqual([int(grid[0]) for _, grid in result], [0, 4, 8])
        self.assertEqual([center for center, _ in result], [False, True, False])
        self.assertEqual(len(extraction.windows(np.arange(300) / 30, 30)), 1)

    def test_disk_guard_blocks_writes_below_reserve(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(extraction, "OUTPUT", output), patch.object(extraction, "disk_usage") as usage:
                usage.return_value.free = extraction.MIN_FREE
                with self.assertRaises(extraction.DiskPause):
                    extraction.save(output / "blocked.json", {"n": 1})
                self.assertFalse(list(output.iterdir()))
                usage.return_value.free += extraction.BUFFER
                extraction.save(output / "complete.json", {"n": 1})
                self.assertEqual(json.loads((output / "complete.json").read_text()), {"n": 1})

    def test_failed_checkpoint_write_preserves_previous_commit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_bytes(b"original")

            def fail(stream):
                stream.write(b"incomplete")
                raise RuntimeError("fixture interruption")

            with self.assertRaises(RuntimeError):
                extraction.atomic(path, fail, reserve=False)
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_schema_compatible_synthetic_features_and_missing_values(self):
        schema = json.loads(Path("outputs/test10/feature_schema.json").read_text(encoding="utf-8"))
        t = np.arange(360) / 30
        pulse = np.sin(2 * np.pi * t)
        rgb = 120 + pulse[:, None, None] * np.array([1, 2, 3])[None, None, :]
        rgb = np.repeat(rgb, 3, axis=1)
        data = dict(times=t, fps=30, baseline=rgb, base_codes=np.ones((len(t), 3)),
                    scores=np.full((len(t), 30), np.nan), geometry=np.full((len(t), 4), np.nan),
                    light=np.zeros(len(t)))
        with patch.object(extraction, "ensure_space"):
            result = extraction.feature_rows(dict(file_size=100, width=640, height=480), data, schema)
        self.assertEqual(len(result), 3)
        self.assertEqual(sum(row["center"] for row in result), 1)
        self.assertEqual({(len(row["legacy"]), len(row["green"]), len(row["pos"])) for row in result}, {(69, 89, 276)})
        self.assertIsNone(result[0]["audit"]["motion"])


@unittest.skipIf(sklearn is None, "Phase 4.8 requires the project scikit-learn environment")
class ClassificationTests(unittest.TestCase):
    def test_source_family_assignment_does_not_split_related_clips(self):
        from deepfake_rppg.phase48_classify import fold_groups
        items = [dict(group_id=str(family), phase48_cohort="retained" if family < 8 else "new")
                 for family in range(50) for _ in range(2)]
        folds = list(fold_groups(items, list(range(len(items))), 5, 480))
        self.assertEqual([len(test) for _, test in folds], [20] * 5)
        for train, test in folds:
            self.assertTrue({items[i]["group_id"] for i in train}.isdisjoint(items[i]["group_id"] for i in test))

    def test_temporal_aggregation_and_missingness(self):
        from deepfake_rppg.phase48_classify import representation
        records = [dict(center=True, pos={"a": 1., "b": None}), dict(center=False, pos={"a": 3., "b": None})]
        np.testing.assert_allclose(representation(records, ["a", "b"], "fusion250", "mean_std"),
                                   [[2, np.nan, 1, np.nan]], equal_nan=True)
        np.testing.assert_array_equal(representation(records, ["a", "b"], "missingness_only", "center"), [[0, 1]])

    def test_video_scores_collapse_multiple_windows(self):
        from deepfake_rppg.phase48_classify import fit_predict
        labels = np.array([0, 0, 1, 1, 0, 1])
        arrays = [np.full((i + 1, 1), x, dtype=float) for i, x in enumerate(labels)]
        scores, *_, counts = fit_predict(arrays, labels, [0, 1, 2, 3], [4, 5], "logistic", 1.)
        np.testing.assert_array_equal(counts, [5, 6])
        self.assertEqual((scores >= .5).tolist(), [False, True])

    def test_family_interval_is_degenerate_for_equal_changes(self):
        from deepfake_rppg.phase48_evaluate import interval
        self.assertEqual(interval([.25] * 6), dict(mean=.25, ci_low=.25, ci_high=.25, families=6))

"""Regression coverage for aligned fusion and source-family-held-out evaluation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deepfake_rppg.phase47_capture import verify_timeline
from deepfake_rppg.phase47_fusion import index_rows, match, numeric, read_csv, verify_families

try:
    import sklearn
except ImportError:
    sklearn = None


class FusionIntegrityTests(unittest.TestCase):
    def test_decoder_positions_and_window_must_match_exactly(self):
        times = np.arange(12) / 4 + .5
        np.testing.assert_array_equal(verify_timeline(times, times, 2, 5, .5, 1.5), [.5, .75, 1, 1.25])
        changed = times.copy()
        changed[1] += 1e-9
        with self.assertRaisesRegex(ValueError, "Decoded positions"):
            verify_timeline(changed, times, 2, 5, .5, 1.5)
        with self.assertRaisesRegex(ValueError, "Frame indices"):
            verify_timeline(times, times, 2, 6, .5, 1.5)

    def test_identity_mismatch_and_duplicate_observations_fail(self):
        expected = {"clip_id": "one", "window_start_s": "1.25", "split_group": "family_a"}
        conflicting = dict(expected, window_start_s="1.25001")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            match(conflicting, expected, tuple(expected))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_rows([expected, conflicting])

    def test_invalid_names_and_infinities_fail_but_missing_survives(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dup.csv"
            path.write_text("clip_id,foo,foo\none,1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_csv(path)
        self.assertTrue(np.isnan(numeric({"missing": ""}, ["missing"])[0]))
        with self.assertRaisesRegex(ValueError, "Infinite"):
            numeric({"bad": "inf"}, ["bad"])

    def test_source_identifiers_cannot_cross_families(self):
        matches = [dict(group_id="A", source_video_ids=["original"]),
                   dict(group_id="B", source_video_ids=["original", "partner"])]
        with self.assertRaisesRegex(ValueError, "crosses source"):
            verify_families(matches)
        matches[1]["group_id"] = "A"
        verify_families(matches)


@unittest.skipIf(sklearn is None, "Phase 4.7 optional scikit-learn environment")
class GroupedClassificationTests(unittest.TestCase):
    def test_imputation_and_column_selection_use_training_rows_only(self):
        from deepfake_rppg.phase47_classification import FoldTransform

        train = np.array([[1., 7, np.nan, 1], [2, 7, np.nan, 2], [3, 7, np.nan, 3]])
        heldout = np.array([[np.nan, 6, 1000, 40]])
        fitted = FoldTransform().fit(train)
        np.testing.assert_array_equal(fitted.columns, [0])
        np.testing.assert_array_equal(fitted.medians, [2.])
        np.testing.assert_array_equal(fitted.transform(heldout), [[0.]])

    def test_related_observations_share_heldout_fold(self):
        from deepfake_rppg.phase47_classification import grouped_folds

        groups = np.repeat(["a", "b", "c"], 4)
        labels = np.tile([0, 0, 1, 1], 3)
        folds = grouped_folds(labels, groups)
        self.assertEqual(len(folds), 3)
        np.testing.assert_array_equal(np.sort(np.concatenate([test for _, test in folds])), np.arange(12))
        for train, test in folds:
            self.assertTrue(set(groups[train]).isdisjoint(groups[test]))
            self.assertEqual(len(set(groups[test])), 1)

    def test_video_aggregation_uses_mean_and_refuses_mixed_folds(self):
        from deepfake_rppg.phase47_classification import aggregate_scores

        first = dict(clip_id="a", split_group="family", label=0, manipulation="original", fold=1, score=.1)
        result, = aggregate_scores([first, dict(first, score=.7)])
        self.assertAlmostEqual(result["videoscore"], .4)
        self.assertEqual(result["prediction"], 0)
        with self.assertRaisesRegex(ValueError, "cross"):
            aggregate_scores([first, dict(first, fold=2)])


if __name__ == "__main__":
    unittest.main()

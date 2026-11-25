"""Regression coverage for the checkpoint-only Phase 4.11A pipeline."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits

from deepfake_rppg import phase411a as run
from deepfake_rppg import phase411a_report as report


def toy_cohort():
    rng = np.random.default_rng(42)
    items, outer, arrays = [], [], []
    for family in range(15):
        for label in (0, 1):
            clip = f"{family}-{label}"
            items.append(dict(video_id=clip, group_id=str(family), label=label,
                              phase48_cohort="new", manipulation_method="control"))
            outer.append(family % 5 + 1)
            arrays.append(rng.normal(size=(1, 3)) + label * .8)
    layout = [("expression", "a", "mean"), ("original_temporal", "b", "mean"),
              ("enhanced_temporal", "c", "mean")]
    return items, np.array(outer), arrays, layout


class Phase411ATests(unittest.TestCase):
    def test_feature_alignment_and_missing_window_aggregation(self):
        fields = [("enhanced", name) for name in run.TEMPORAL]
        records = [{"start_s": 0.}, {"start_s": 4.}]
        first = [None, 2.] + [None] * (len(run.WINDOW_FIELDS) - 2)
        second = [None, None] + [None] * (len(run.WINDOW_FIELDS) - 2)
        data = dict(clip_id="a", starts=[0., 4.], windows=[first, second],
                    summary=dict.fromkeys(run.TEMPORAL))
        run.validate_feature(data, "a", records)
        blocks = dict(enhanced_temporal=fields, original_spectral=[("pos", "freq")])
        rows = [dict(r, pos=dict(freq=value)) for r, value in zip(records, (2., 4.))]
        mode = "mean_std"
        arrays, layout = run.representation([rows], [data], blocks, "enhanced_temporal_original_spectral", mode)
        matrix = dict(zip(layout, arrays[0][0], strict=True))
        group, name = "enhanced_temporal", run.WINDOW_FIELDS[1]
        self.assertEqual(matrix[group, name, mode], 2.)
        self.assertTrue(np.isnan(matrix[group, name, "sd"]))
        self.assertEqual(matrix[group, name, "missing_fraction"], .5)
        self.assertEqual(matrix["original_spectral", "freq", "sd"], 1.)
        self.assertTrue(np.isnan(matrix[group, run.WINDOW_FIELDS[0], mode]))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            run.validate_feature(dict(data, starts=[4., 0.]), "a", records)

    def test_training_only_constant_redundancy_and_missingness(self):
        x = np.arange(250, dtype=float)
        partial = np.where(x < 50, np.nan, 2.)
        near = np.zeros(len(x))
        near[-1] = 1.
        values = np.column_stack((x, -x, partial, near, np.full(len(x), np.nan)))
        layout = [("expression", str(j), "mean") for j in range(values.shape[1])]
        transform = run.HybridTransform(layout, redundancy=True, expression_penalty=None).fit(values)
        np.testing.assert_array_equal(transform.columns, [0, 1])
        np.testing.assert_array_equal(transform.missing, [2])
        np.testing.assert_array_equal(transform.selected, [0, 2])
        held = np.array([[10000., 99., np.nan, 999., -99.]])
        transformed = transform.transform(held)
        self.assertEqual(transformed.shape, (1, 2))
        self.assertEqual(transformed[0, 1], 1.)
        self.assertEqual(transform.centers[0], 124.5)

    def test_all_missing_and_constant_inputs(self):
        values = np.array([[np.nan, 5.], [np.nan, 5.]])
        layout = [("expression", str(j), "mean") for j in range(2)]
        transform = run.HybridTransform(layout, redundancy=True).fit(values)
        self.assertEqual(transform.transform(values).shape, (2, 0))

    def test_modality_scaling_changes_relative_penalty(self):
        values = np.array([[0., 1., 4.], [1., 1., 5.], [2., 3., 7.]])
        layout = [(group, str(i), "mean") for i, group in enumerate(("expression", "expression", "original_temporal"))]
        transform = run.HybridTransform(layout, expression_penalty=2.).fit(values)
        np.testing.assert_allclose(transform.multipliers, [.5, .5, 1.])

    def test_determinism_and_outer_training_isolation(self):
        items, outer, arrays, layout = toy_cohort()
        config = ("hybrid_redundancy", "mean", "logistic_l2")
        with threadpool_limits(limits=1):
            first = run.evaluate_fold(items, outer, arrays, layout, config, 1)
            second = run.evaluate_fold(items, outer, arrays, layout, config, 1)
            shifted = [values + 1000 if fold == 1 else values for values, fold in zip(arrays, outer)]
            altered = run.evaluate_fold(items, outer, shifted, layout, config, 1)
        self.assertEqual({k: v for k, v in first.items() if k != "seconds"},
                         {k: v for k, v in second.items() if k != "seconds"})
        for field in ("setting", "threshold", "selected_columns"):
            self.assertEqual(first[field], altered[field])
        self.assertNotEqual(first["rows"], altered["rows"])

    def test_checkpoint_version_and_shape(self):
        with TemporaryDirectory(dir=os.environ.get("TEMP")) as folder:
            path = Path(folder) / "cached.json"
            path.write_text('{"key":"prior","result":{"value":5}}')
            self.assertIsNone(run.cached(path, "current"))
            self.assertEqual(run.cached(path, "prior"), {"value": 5})
            path.write_text("{")
            self.assertIsNone(run.cached(path, "prior"))

    def test_shortlisted_estimators_and_window_permutation(self):
        items, outer, arrays, layout = toy_cohort()
        labels = np.array([item["label"] for item in items])
        train = np.flatnonzero(outer != 1)
        held = np.flatnonzero(outer == 1)
        window_arrays = [np.concatenate((a, a + .1), axis=0) if i % 3 else a
                         for i, a in enumerate(arrays)]
        with threadpool_limits(limits=1):
            for arm in run.SHORTLIST:
                matrix = arrays if arm == "phase49_selective" else window_arrays
                values, counts = run.stack(matrix, held)
                for classifier, settings in run.MODEL_GRIDS.items():
                    with self.subTest(arm=arm, classifier=classifier):
                        predict, used = run.fit_model(matrix, layout, labels, train,
                                                      (arm, "mean", classifier), settings[0])
                        self.assertGreater(used, 0)
                        scores = predict(values, counts)
                        self.assertEqual(scores.shape, (len(held),))
                        self.assertTrue(np.isfinite(scores).all())
                        result = run.grouped_importance(predict, values, counts, labels[held], layout, scores, 1)
                        self.assertEqual(len(result), 3)

    def test_report_outputs_with_synthetic_folds(self):
        items, outer, arrays, layout = toy_cohort()
        config = ("hybrid", "mean", "logistic_l2")
        with threadpool_limits(limits=1):
            folds = [run.evaluate_fold(items, outer, arrays, layout, config, fold) for fold in range(1, 6)]
        baseline = [dict(row) for fold in folds for row in fold["rows"]]
        data = (items, [], outer, baseline)
        with TemporaryDirectory(dir=os.environ.get("TEMP")) as folder, patch.object(run, "ROOT", Path(folder)):
            for arm, aggregation, classifier in run.plan():
                for fold in folds:
                    key = (arm, aggregation, classifier)
                    path = run.checkpoint_path(key, fold["fold"])
                    run.save(path, dict(key="synthetic", result=dict(fold, config=key)))
            result = report.evaluate(data, "synthetic")
            report.report("synthetic")
            self.assertEqual(result["videos"], 30)
            for name in ("PHASE411A_REPORT.md", "hybrid_feature_comparison.csv", "classifier_comparison.csv",
                         "ablation_comparison.csv", "paired_family_comparison.csv", "feature_importance_summary.csv",
                         "video_predictions.csv", "best_model_config.json", "roc_curves.png",
                         "precision_recall_curves.png", "confusion_matrices.png", "benchmark_comparison.png"):
                self.assertGreater((Path(folder) / name).stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()

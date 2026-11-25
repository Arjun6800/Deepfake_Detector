"""Regression checks for Phase 4.9 fold isolation and checkpoint compatibility."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from deepfake_rppg.phase48_classify import representation
from deepfake_rppg.phase49_classify import (
    AGGREGATIONS, BASELINES, LATE, OUTPUT, PREVIOUS, arrays_for, audit, checkpoint_path,
    choose_fusion, evaluate_fold, inner_oof, plan, read_csv, read_result, save,
)
from deepfake_rppg import phase49_evaluate
from deepfake_rppg.phase49_evaluate import metrics, summarize


class Phase49Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = audit()

    def test_phase48_checkpoint_cohort_and_representations(self):
        items, records, schema, fields, outer, multi, centers, _, _, result = self.data
        self.assertEqual((result["videos"], result["families"], result["new_families"]), (600, 50, 42))
        self.assertEqual((len(multi), result["temporal_families"], result["temporal_new_families"]), (484, 47, 39))
        self.assertEqual(result["predictions"], 32480)
        for stage in (1, 3):
            eligible = sorted(multi) if stage == 1 else range(len(items))
            for fold in range(1, 6):
                self.assertEqual({items[i]["label"] for i in eligible if outer[i] == fold}, {0, 1})
        for arm in BASELINES:
            for mode in AGGREGATIONS:
                checked = sorted(multi)[:5]
                new = arrays_for([records[i] for i in checked], fields[arm],
                                 {j: centers[i] for j, i in enumerate(checked)}, mode)
                for j, i in enumerate(checked):
                    np.testing.assert_allclose(new[j], representation(records[i], schema["ablations"][arm], arm, mode))
        self.assertEqual(Counter(cfg[0] for cfg in plan()), {1: 27, 2: 13, 3: 6})

    def test_phase48_reported_metrics_against_predictions(self):
        keys = ("comparison", "arm", "temporal", "classifier")
        grouped = defaultdict(list)
        for row in read_csv(PREVIOUS / "video_predictions.csv"):
            key = tuple(row[k] for k in keys)
            grouped[key, "all"].append(row)
            grouped[key, row["cohort"]].append(row)
        for name in ("ablation_comparison", "temporal_comparison"):
            for summary in read_csv(PREVIOUS / f"{name}.csv"):
                rows = grouped[tuple(summary[k] for k in keys), summary["cohort"]]
                labels = [int(r["label"]) for r in rows]
                scores = [float(r["score"]) for r in rows]
                pred = [int(r["prediction"]) for r in rows]
                self.assertEqual(len(rows), int(summary["videos"]))
                self.assertAlmostEqual(balanced_accuracy_score(labels, pred), float(summary["balanced_accuracy"]))
                self.assertAlmostEqual(roc_auc_score(labels, scores), float(summary["roc_auc"]))

    def test_outer_holdout_perturbation_cannot_change_training_settings(self):
        rng = np.random.default_rng(49)
        items, rows = [], []
        for i in range(60):
            label = i % 2
            items.append(dict(label=label, video_id=str(i), group_id=str(i // 2),
                              phase48_cohort="new", manipulation_method="fake" if label else "original"))
            rows.append([{"pos": {"exp": float(label + rng.normal()), "pulse": float(label + rng.normal())}}
                         for _ in range(2)])
        schema = {"expression": ["exp"], "rppg": {"temporal": ["pulse"]}}
        fields = {"auto": [("pos", "exp"), ("pos", "pulse")], "expression187": [("pos", "exp")],
                  "pos_temporal": [("pos", "pulse")]}
        outer = np.repeat(np.arange(1, 6), 12)
        multi, coverage = set(range(60)), rng.uniform(size=60)
        data = (items, rows, schema, fields, outer, multi, {}, coverage, "synthetic", {})
        new_items = [dict(item, label=1 - item["label"]) if outer[i] == 1 else item for i, item in enumerate(items)]
        new_rows = [[{"pos": {"exp": 100., "pulse": -100.}}] * 2 if outer[i] == 1 else row for i, row in enumerate(rows)]
        changed = (new_items, new_rows, schema, fields, outer, multi, {},
                   np.where(outer == 1, 0, coverage), "synthetic", {})
        configs = [(1, "auto", "mean", model) for model in ("logistic", "random_forest", "svm_rbf", "hist_gradient_boosting")]
        configs += [(2, arm, "mean", "logistic") for arm in LATE]
        with threadpool_limits(limits=1):
            for cfg in configs:
                with self.subTest(cfg=cfg):
                    original = evaluate_fold(cfg, 1, data)
                    perturbed = evaluate_fold(cfg, 1, changed)
                    self.assertEqual(original["setting"], perturbed["setting"])
                    self.assertEqual(original["threshold"], perturbed["threshold"])
                    self.assertEqual([r["clip_id"] for r in original["rows"]], list(map(str, range(12))))
                    self.assertTrue(original["importance"])

    def test_oof_rejects_outer_held_index(self):
        with self.assertRaisesRegex(ValueError, "Invalid inner OOF partition"):
            inner_oof([], np.array([0, 1, 0]), [0, 1], [([0], [2])], "logistic", .1)

    def test_fusion_can_ignore_rppg_and_metrics_use_decisions(self):
        labels = np.array([0, 1, 0, 1])
        oof = np.column_stack(([.1, .8, .2, .7], [.9, .2, .8, .3]))
        for arm in LATE:
            scores, settings, _ = choose_fusion(arm, labels, oof, np.ones(4))
            # The expression scores already maximize both training objectives.
            self.assertEqual(settings["weight"], 0.)
            np.testing.assert_array_equal(scores, oof[:, 0])
        rows = [dict(label=int(label), prediction=int(label), score=score, cohort="new",
                     family="g", manipulation="fake" if label else "original", stage=1,
                     arm=arm, temporal="mean", classifier="logistic")
                for arm in ("expression187", "temporal")
                for label, score in zip(labels, [.1, .3, .2, .4], strict=True)]
        result = metrics(rows[:4])
        self.assertEqual((result["balanced_accuracy"], result["roc_auc"]), (1., 1.))
        _, _, paired, _ = summarize(rows)
        self.assertTrue(paired)
        self.assertTrue(all(row["mean"] == 0 for row in paired))

    def test_resume_and_evaluation_artifacts(self):
        items, rows, outer = [], [], []
        for i in range(20):
            label, fold = i % 2, i // 4 + 1
            items.append(dict(video_id=str(i), label=label, group_id=str(i // 2), phase48_cohort="new",
                              manipulation_method="fake" if label else "original"))
            rows.append(dict(clip_id=str(i), label=label, family=str(i // 2), cohort="new",
                             manipulation="fake" if label else "original", available_windows=2, model_rows=1,
                             score=.2 + .2 * label, prediction=label))
            outer.append(fold)
        cfg = (1, "expression187", "mean", "logistic")
        data = (items, [[{}, {}]] * 20, {}, {}, outer, set(range(20)), {}, [], "fixture", {})
        with TemporaryDirectory(dir=OUTPUT) as directory:
            output = Path(directory)
            (output / "classification").mkdir()
            with (patch("deepfake_rppg.phase49_classify.OUTPUT", output),
                  patch.object(phase49_evaluate, "OUTPUT", output),
                  patch.object(phase49_evaluate, "audit", return_value=data),
                  patch.object(phase49_evaluate, "plan", return_value=[cfg]),
                  patch("sys.argv", ["phase49_evaluate"])):
                self.assertIsNone(read_result(cfg, 1, "fixture"))
                for fold in range(1, 6):
                    save(checkpoint_path(cfg, fold),
                         dict(revision="fixture", config=list(cfg), fold=fold,
                              rows=[r for r, f in zip(rows, outer, strict=True) if f == fold],
                              threshold=.3, seconds=1., setting={"C": .1}, importance={"expression": .2}))
                self.assertEqual(read_result(cfg, 1, "fixture")["fold"], 1)
                with self.assertRaisesRegex(ValueError, "Checkpoint input or implementation changed"):
                    read_result(cfg, 1, "stale")
                phase49_evaluate.main()
            for name in ("PHASE49_REPORT.md", "baseline_comparison.csv", "late_fusion_comparison.csv",
                         "video_predictions.csv", "best_model_config.json", "runtime_summary.json",
                         "roc_curves.png", "precision_recall_curves.png", "confusion_matrices.png"):
                self.assertGreater((output / name).stat().st_size, 0)
            self.assertEqual(len(read_csv(output / "video_predictions.csv")), 20)

"""Generate Phase 4.8 reports from completed out-of-fold video predictions."""
from collections import defaultdict
import json

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .phase47_classification import METRICS, MODELS, evaluate
from .phase47_fusion import read_csv
from .phase48_extract import OUTPUT, ensure_space, signature
from .quality_experiment import write_table

KEYS = ("comparison", "arm", "temporal", "classifier")
BOOTSTRAPS = 4000


def metrics(rows):
    labels = np.array([int(row["label"]) for row in rows])
    scores = np.array([float(row["score"]) for row in rows])
    return evaluate(labels, scores)


def grouped(rows, key):
    buckets = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return buckets


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(480)
    samples = rng.integers(0, len(values), (BOOTSTRAPS, len(values)))
    low, high = np.quantile(values[samples].mean(axis=1), (.025, .975))
    return dict(mean=values.mean(), ci_low=low, ci_high=high, families=len(values))


def review(predictions):
    summaries, family_stats, folds, pairs, subgroup = [], [], [], [], []
    outcomes = {}
    for key, rows in grouped(predictions, lambda row: tuple(row[k] for k in KEYS)).items():
        if len({row["clip_id"] for row in rows}) != len(rows):
            raise ValueError("Expected one out-of-fold score per video per run")
        info = dict(zip(KEYS, key, strict=True))
        for cohort in ("all", "new", "retained"):
            subset = [row for row in rows if cohort == "all" or row["cohort"] == cohort]
            if {row["label"] for row in subset} != {"0", "1"}:
                continue
            metadata = dict(**info, cohort=cohort)
            by_family = {}
            for family, members in grouped(subset, lambda row: row["family"]).items():
                if {row["label"] for row in members} == {"0", "1"}:
                    by_family[family] = metrics(members)
                    family_stats.append(dict(**metadata, family=family, videos=len(members), **by_family[family]))
            outcomes[key, cohort] = by_family
            summary = dict(**metadata, videos=len(subset), **metrics(subset))
            if by_family:
                for metric in METRICS:
                    values = [row[metric] for row in by_family.values()]
                    summary.update({f"family_{metric}_{k}": v for k, v in interval(values).items()})
                    summary[f"family_{metric}_sd"] = np.std(values, ddof=1) if len(values) > 1 else np.nan
            summaries.append(summary)
            for fold, group in grouped(subset, lambda row: row["fold"]).items():
                if {row["label"] for row in group} == {"0", "1"}:
                    folds.append(dict(**metadata, fold=fold, videos=len(group), **metrics(group)))
            for method in sorted({row["manipulation"] for row in subset if row["label"] == "1"}):
                members = [row for row in subset if row["label"] == "0" or row["manipulation"] == method]
                subgroup.append(dict(**metadata, manipulation=method, videos=len(members), **metrics(members)))
    for classifier in MODELS:
        contrasts = [(("replication", arm, "center", classifier),
                      ("replication", "expression187", "center", classifier))
                     for arm in ("fusion256", "fusion276", "fusion250")]
        contrasts.extend((("temporal", "fusion250", temporal, classifier),
                          ("temporal", "fusion250", "center", classifier))
                         for temporal in ("mean", "mean_std", "median", "window_scores"))
        for trial, control in contrasts:
            for cohort in ("all", "new", "retained"):
                trial_rows, control_rows = outcomes.get((trial, cohort), {}), outcomes.get((control, cohort), {})
                shared = sorted(trial_rows.keys() & control_rows.keys())
                if not shared:
                    continue
                for metric in METRICS:
                    deltas = [trial_rows[family][metric] - control_rows[family][metric] for family in shared]
                    pairs.append(dict(comparison=trial[0], classifier=classifier, cohort=cohort,
                                      trial=f"{trial[1]}/{trial[2]}", control=f"{control[1]}/{control[2]}",
                                      metric=metric, improved_families=sum(delta > 1e-12 for delta in deltas),
                                      degraded_families=sum(delta < -1e-12 for delta in deltas), **interval(deltas)))
    return summaries, family_stats, folds, pairs, subgroup


def render(predictions):
    import matplotlib.pyplot as plt

    logistic = [row for row in predictions if row["classifier"] == "logistic" and row["comparison"] == "replication"]
    arms = grouped(logistic, lambda row: row["arm"])
    for name in ("roc", "precision_recall"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for cohort, ax in zip(("all", "new"), axes, strict=True):
            for arm, rows in arms.items():
                subset = [row for row in rows if cohort == "all" or row["cohort"] == cohort]
                labels = np.array([int(row["label"]) for row in subset])
                scores = np.array([float(row["score"]) for row in subset])
                if name == "roc":
                    x, y, _ = roc_curve(labels, scores)
                else:
                    y, x, _ = precision_recall_curve(labels, scores)
                ax.plot(x, y, label=arm)
            if name == "roc":
                ax.plot([0, 1], [0, 1], "k:")
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
            else:
                ax.axhline(5 / 6, color="k", linestyle=":")
                ax.set(xlabel="Recall", ylabel="Precision")
            ax.set_title("All 50 families" if cohort == "all" else "42 newly added families")
            ax.legend(fontsize=8)
        fig.savefig(OUTPUT / f"{name}_curves.png", dpi=140)
        plt.close(fig)


def main():
    ensure_space()
    stamp = json.loads((OUTPUT / "classification_complete.json").read_text(encoding="utf-8"))
    predictions = read_csv(OUTPUT / "video_predictions.csv")
    if len(predictions) != stamp["predictions"] or stamp["revision"] != signature():
        raise ValueError("Predictions do not match the completed Phase 4.8 run")
    summary, families, folds, pairs, subgroup = review(predictions)
    write_table(OUTPUT, "ablation_comparison", [row for row in summary if row["comparison"] == "replication"])
    write_table(OUTPUT, "temporal_comparison", [row for row in summary if row["comparison"] == "temporal"])
    write_table(OUTPUT, "family_metrics", families)
    write_table(OUTPUT, "fold_metrics", folds)
    write_table(OUTPUT, "paired_family_deltas", pairs)
    write_table(OUTPUT, "per_manipulation", subgroup)
    write_table(OUTPUT, "confusion_matrices", [
        {key: row[key] for key in (*KEYS, "cohort", "tn", "fp", "fn", "tp")} for row in summary])
    importance = read_csv(OUTPUT / "feature_importance.csv")
    importance_summary = []
    for key, group in grouped(importance, lambda row: (row["arm"], row["classifier"], row["feature_group"])).items():
        importance_summary.append(dict(zip(("arm", "classifier", "feature_group"), key, strict=True),
                                       **interval([float(row["mean_auc_drop"]) for row in group])))
    write_table(OUTPUT, "feature_importance_summary", importance_summary)
    render(predictions)
    primary = [row for row in pairs if row["comparison"] == "replication" and row["classifier"] == "logistic"
               and row["metric"] == "balanced_accuracy" and row["cohort"] in ("all", "new")]
    article = [
        "# Phase 4.8 training-cohort replication results", "",
        f"The study evaluated {stamp['videos']} videos using five outer source-family folds and four inner grouped folds.",
        f"The temporal comparison included {stamp['multiwindow']} videos with at least two complete windows.", "",
        "## Paired Logistic Regression results", "",
        "The table reports mean held-out family balanced-accuracy differences against expression-only.",
        "The percentile intervals resample families. Overlapping training folds limit their inferential interpretation.", "",
        "| Comparison | Cohort | Mean change (points) | 95% descriptive interval (points) | Families |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in primary:
        article.append(f"| {row['trial'].split('/')[0]} | {row['cohort']} | {100 * row['mean']:+.2f} | "
                       f"{100 * row['ci_low']:+.2f} to {100 * row['ci_high']:+.2f} | {row['families']} |")
    article.extend(["", "## Interpretation", "",
                    "Use the new-family comparison as the replication check. Read paired intervals alongside pooled video metrics and the reliability, missingness and nuisance probes. The evaluation supports no claim about physiological accuracy. Select a final architecture only after reviewing the paired results and feature-group evidence.", "",
                    "[Ablations](ablation_comparison.csv) · [Temporal results](temporal_comparison.csv) · "
                    "[Predictions](video_predictions.csv) · [Importance](feature_importance_summary.csv) · "
                    "[ROC](roc_curves.png) · [Precision and recall](precision_recall_curves.png)"])
    brief = OUTPUT / "PHASE48_APPROVAL_BRIEF.md"
    report = OUTPUT / "PHASE48_REPORT.md"
    if not brief.exists() and report.exists():
        brief.write_bytes(report.read_bytes())
    report.write_text("\n".join(article) + "\n", encoding="utf-8")
    print(f"Evaluated {len(predictions)} out-of-fold video scores. See outputs/test11/PHASE48_REPORT.md")


if __name__ == "__main__":
    main()

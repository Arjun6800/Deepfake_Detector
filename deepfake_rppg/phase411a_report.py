"""Pooled predictions, paired family estimates and figures for Phase 4.11A."""
from collections import defaultdict
import csv
import io

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from . import phase411a as run
from .phase48_evaluate import interval
from .phase49_evaluate import SUMMARY_METRICS, both_classes, metrics

HISTORICAL = ("phase49_historical", "mean", "logistic_l2")
SELECTIVE = ("phase49_selective", "mean", "logistic_l2")
ABLATIONS = ("expression_only", "original_rppg_only", "enhanced_temporal_original_spectral", "phase49_selective")
TARGETS = dict(accuracy=.90, balanced_accuracy=.85, precision=.95, recall=.90, f1=.92, roc_auc=.90)


def table(name, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows(rows)
    run.write(run.ROOT / name, buffer.getvalue().encode("utf-8"))


def identity(config):
    return dict(zip(("arm", "aggregation", "classifier"), config, strict=True))


def collect(data, key):
    items, _, outer, historical, *_ = data
    predictions = {HISTORICAL: historical}
    details = {}
    for config in run.plan():
        predictions[config], details[config] = [], []
        for fold in range(1, 6):
            result = run.cached(run.checkpoint_path(config, fold), key)
            if result is None:
                raise ValueError(f"Run classify first. Missing current fold for {config}, fold {fold}")
            expected = {item["video_id"]: (item["group_id"], item["label"], fold)
                        for item, member in zip(items, outer, strict=True) if member == fold}
            actual = {row["clip_id"]: (row["family"], row["label"], row["fold"]) for row in result["rows"]}
            if (result["config"] != list(config) or result["fold"] != fold or actual != expected
                    or len(result["rows"]) != len(expected)):
                raise ValueError(f"Fold prediction alignment differs for {config}, fold {fold}")
            predictions[config].extend(result["rows"])
            details[config].append(result)
    return predictions, details


def pair(candidate, reference, predictions):
    candidate_rows = predictions[candidate]
    reference_by_id = {r["clip_id"]: r for r in predictions[reference]}
    if set(reference_by_id) != {r["clip_id"] for r in candidate_rows}:
        raise ValueError("Paired comparisons require identical clip IDs")
    groups = defaultdict(list)
    for row in candidate_rows:
        other = reference_by_id[row["clip_id"]]
        if any(row[field] != other[field] for field in ("family", "label", "fold")):
            raise ValueError("Paired family metadata differs")
        groups[row["family"]].append((row, other))
    differences = defaultdict(list)
    per_family = []
    tag = dict(**identity(candidate), **{f"reference_{k}": v for k, v in identity(reference).items()})
    for family, pairs in sorted(groups.items()):
        left, right = map(list, zip(*pairs))
        if not both_classes(left):
            continue
        a, b = metrics(left), metrics(right)
        delta = {name: float(a[name] - b[name]) for name in SUMMARY_METRICS}
        per_family.append(dict(**tag, family=family, videos=len(pairs), **delta))
        for name, value in delta.items():
            differences[name].append(value)
    return [dict(**tag, metric=name, **interval(values)) for name, values in differences.items()], per_family


def evaluate(data, key):
    predictions, details = collect(data, key)
    summaries = {c: dict(**identity(c), videos=len(rows), families=len({r["family"] for r in rows}), **metrics(rows))
                 for c, rows in predictions.items()}
    rank = lambda c: (summaries[c]["balanced_accuracy"], summaries[c]["roc_auc"])
    best = max(details, key=rank)
    hybrid = max((c for c in details if c[0] in ("hybrid", "hybrid_redundancy", "hybrid_modality_regularization")), key=rank)
    ordered = sorted(details, key=rank, reverse=True)
    table("hybrid_feature_comparison.csv", [summaries[c] for c in ordered if c[2] == "logistic_l2"])
    table("classifier_comparison.csv", [summaries[c] for c in ordered if c[0] in run.SHORTLIST and c[1] == "mean"])
    table("ablation_comparison.csv", [summaries[c] for c in (HISTORICAL, *[
        (arm, "mean", "logistic_l2") for arm in ABLATIONS], ("hybrid", "mean", "logistic_l2"))])
    table("confusion_matrices.csv", [{k: row[k] for k in (*identity(HISTORICAL), "tn", "fp", "fn", "tp")}
                                    for row in summaries.values()])
    table("video_predictions.csv", [dict(**identity(c), **row) for c, rows in predictions.items() for row in rows])
    pairs = [(c, HISTORICAL) for c in ordered]
    pairs.extend((c, (arm, "mean", "logistic_l2")) for c in dict.fromkeys((best, hybrid))
                 for arm in ABLATIONS[:2] if c != (arm, "mean", "logistic_l2"))
    pairs.extend((("hybrid", "mean", "logistic_l2"), (arm, "mean", "logistic_l2"))
                 for arm in ABLATIONS)
    pair_rows, raw_pairs = [], []
    for candidate, reference in dict.fromkeys(pairs):
        summary, raw = pair(candidate, reference, predictions)
        pair_rows.extend(summary)
        raw_pairs.extend(raw)
    table("paired_family_comparison.csv", pair_rows)
    table("paired_family_deltas.csv", raw_pairs)
    importance_rows = []
    for config, folds in details.items():
        drops = defaultdict(list)
        for fold in folds:
            for row in fold["importance"]:
                drops[row["feature_group"]].append(np.mean(row["drops"]))
        for group, means in drops.items():
            importance_rows.append(dict(**identity(config), feature_group=group, folds=len(means),
                                        repeats_per_fold=5, mean_auc_drop=np.mean(means),
                                        sd_fold_mean_auc_drop=np.std(means, ddof=1)))
    table("feature_importance_summary.csv", importance_rows)
    table("classification_runtime.csv", [dict(**identity(c), fold=f["fold"], seconds=f["seconds"],
                                              selected_columns=f["selected_columns"])
                                        for c, folds in details.items() for f in folds])
    replay_delta = {name: float(summaries[SELECTIVE][name] - summaries[HISTORICAL][name]) for name in SUMMARY_METRICS}
    run.save(run.ROOT / "best_model_config.json",
             dict(**identity(best), selection="Post hoc pooled balanced accuracy, then ROC-AUC",
                  metrics=summaries[best], historical=summaries[HISTORICAL],
                  historical_replay_delta=replay_delta, best_full_hybrid=summaries[hybrid],
                  folds=[{k: f[k] for k in ("fold", "setting", "threshold", "selected_columns")} for f in details[best]]))
    run.save(run.ROOT / "evaluation.json", dict(key=key, best=best, hybrid=hybrid, summaries=list(summaries.values())))
    return dict(configurations=len(details), videos=len(predictions[best]),
                total_fold_compute_seconds=sum(f["seconds"] for folds in details.values() for f in folds))


def save_figure(fig, name):
    import matplotlib.pyplot as plt
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    run.write(run.ROOT / name, buffer.getvalue())


def report(key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = run.read(run.ROOT / "evaluation.json")
    if result["key"] != key:
        raise ValueError("Evaluation is stale. Run evaluate.")
    best = tuple(result["best"])
    hybrid = tuple(result["hybrid"])
    keys = tuple(identity(best))
    summaries = {tuple(row[k] for k in keys): row for row in result["summaries"]}
    configurations = list(dict.fromkeys((HISTORICAL, best, hybrid, *[
        (arm, "mean", "logistic_l2") for arm in ABLATIONS[:2]])))
    predictions = defaultdict(list)
    for row in run.read_csv(run.ROOT / "video_predictions.csv"):
        config = tuple(row[k] for k in keys)
        if config in configurations:
            predictions[config].append(row)
    labels = [" / ".join(c) for c in configurations]
    for name, curve, xlabel, ylabel in (
        ("roc_curves.png", roc_curve, "False positive rate", "True positive rate"),
        ("precision_recall_curves.png", precision_recall_curve, "Recall", "Precision"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for c, label in zip(configurations, labels, strict=True):
            rows = predictions[c]
            a, b, _ = curve([int(r["label"]) for r in rows], [float(r["score"]) for r in rows])
            x, y = (a, b) if curve is roc_curve else (b, a)
            ax.plot(x, y, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="lower left" if curve is roc_curve else "lower right")
        save_figure(fig, name)
    matrices = list(dict.fromkeys((HISTORICAL, best, hybrid)))
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices), 4), squeeze=False)
    for c, ax in zip(matrices, axes[0], strict=True):
        row = summaries[c]
        counts = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(max(r) for r in counts))
        for (i, j), value in np.ndenumerate(counts):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > 180 else "black")
        ax.set(title="\n".join(c), xticks=(0, 1), yticks=(0, 1), xlabel="Predicted", ylabel="Actual")
    save_figure(fig, "confusion_matrices.png")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for (name, target), ax in zip(TARGETS.items(), axes.flat, strict=True):
        ax.bar(("Phase 4.9", "Winner", "Hybrid"), [summaries[c][name] for c in (HISTORICAL, best, hybrid)])
        ax.axhline(target, color="black", linestyle="--", linewidth=1)
        ax.set(title=name.replace("_", " ").title(), ylim=(0, 1))
    save_figure(fig, "benchmark_comparison.png")
    report_text(best, hybrid, summaries)


def report_text(best, hybrid, summaries):
    configs = [HISTORICAL, SELECTIVE, best, hybrid, *[(arm, "mean", "logistic_l2") for arm in ABLATIONS[:2]]]
    rows = ["| Configuration | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | AP |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for config in dict.fromkeys(configs):
        values = summaries[config]
        rows.append("| " + " / ".join(config) + " | " + " | ".join(f"{values[k]:.5f}" for k in SUMMARY_METRICS) + " |")
    current, historical = summaries[best], summaries[HISTORICAL]
    targets = "\n".join(f"- {name.replace('_', ' ').title()} {current[name]:.2%} against {target:.0%} target"
                        for name, target in TARGETS.items())
    comparisons = [r for r in run.read_csv(run.ROOT / "paired_family_comparison.csv")
                   if tuple(r[k] for k in identity(best)) == hybrid and r["metric"] in ("balanced_accuracy", "roc_auc")]
    deltas = "\n".join(f"- {r['reference_arm']}, {r['metric'].replace('_', ' ')} "
                       f"{float(r['mean']):+.5f} (95% family bootstrap interval "
                       f"{float(r['ci_low']):+.5f} to {float(r['ci_high']):+.5f}, {r['families']} families)"
                       for r in comparisons)
    report = f"""# Phase 4.11A targeted hybrid feature optimization

These are exploratory development estimates. Earlier experiments informed the feature choices.
The run used 484 matched training videos, the five established source-family outer folds and
four inner grouped folds. Official validation and test partitions were untouched.
Configurations were ranked after evaluation by pooled balanced accuracy, then ROC-AUC.

{chr(10).join(rows)}

The top configuration changed balanced accuracy by {current['balanced_accuracy'] - historical['balanced_accuracy']:+.5f}
and ROC-AUC by {current['roc_auc'] - historical['roc_auc']:+.5f} relative to the historical Phase 4.9 predictions.
The replay row uses the original Phase 4.9 fitting recipe. AP denotes average precision.

## Aspirational targets

{targets}

## Paired full-hybrid comparisons

Deltas use equally weighted source families with both classes. Bootstrap intervals describe
the observed family deltas and do not adjust for searching across configurations.

{deltas}

## Interpretation

Permutation importance measures held-fold ROC-AUC changes after jointly shuffling each block.
Window models permute videos within window-count strata to preserve within-video structure.
Correlated blocks can share predictive signal. Importance and ablation results are exploratory
and do not identify independent feature effects. Window models omit the two video-only enhanced
spread features. The classifier shortlist was fixed before fitting Phase 4.11A.

See the comparison CSVs for all configurations, the confusion matrix CSV for class counts,
and the PNG figures for ROC, precision–recall, confusion matrices and benchmark comparisons.
"""
    run.write(run.ROOT / "PHASE411A_REPORT.md", report.encode("utf-8"))

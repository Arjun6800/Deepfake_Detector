"""Summarize completed Phase 4.9 fold checkpoints and render diagnostic plots."""
import argparse
from collections import defaultdict
import csv
import io
import json
from time import perf_counter

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .phase48_evaluate import grouped, interval
from .phase48_extract import atomic
from .phase49_classify import BASELINES, LATE, OUTPUT, SELECTIVE, audit, ensure_space, plan, read_result, save

KEYS = ("stage", "arm", "temporal", "classifier")
SUMMARY_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_ap")


def metrics(rows):
    y = [r["label"] for r in rows]
    tn, fp, fn, tp = map(int, confusion_matrix(y, [r["prediction"] for r in rows], labels=(0, 1)).ravel())
    scores = [r["score"] for r in rows]
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    return dict(accuracy=(tn + tp) / len(rows), balanced_accuracy=(specificity + recall) / 2,
                precision=tp / (tp + fp) if tp + fp else 0., recall=recall,
                f1=2 * tp / (2 * tp + fp + fn), roc_auc=roc_auc_score(y, scores),
                pr_auc_ap=average_precision_score(y, scores), tn=tn, fp=fp, fn=fn, tp=tp)


def both_classes(rows):
    return {r["label"] for r in rows} == {0, 1}


def write_table(name, rows, empty_fields=()):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    with io.StringIO(newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
        payload = stream.getvalue().encode("utf-8")
    write_bytes(f"{name}.csv", payload)


def write_bytes(name, payload):
    ensure_space()
    atomic(OUTPUT / name, lambda stream: stream.write(payload), reserve=False)


def summarize(predictions):
    summaries, subgroups, paired, family_deltas = [], [], [], []
    family_summary = {}
    keys = grouped(predictions, lambda r: tuple(r[k] for k in KEYS))
    for cfg, rows in keys.items():
        info = dict(zip(KEYS, cfg, strict=True))
        for cohort in ("all", "new", "retained"):
            subset = [r for r in rows if cohort == "all" or r["cohort"] == cohort]
            if not both_classes(subset):
                continue
            families = grouped(subset, lambda r: r["family"])
            by_family = {f: metrics(r) for f, r in families.items() if both_classes(r)}
            family_summary[cfg, cohort] = by_family
            meta = dict(**info, cohort=cohort)
            result = dict(**meta, videos=len(subset), families=len(families),
                          families_with_both_classes=len(by_family), **metrics(subset))
            for name in SUMMARY_METRICS:
                if by_family:
                    result.update({f"family_{name}_{k}": v
                                   for k, v in interval([r[name] for r in by_family.values()]).items()})
            summaries.append(result)
            for method in sorted({r["manipulation"] for r in subset if r["label"] == 1}):
                same_method = [r for r in subset if r["manipulation"] == method]
                source_families = {r["family"] for r in same_method}
                comparison = same_method + [r for r in subset if r["label"] == 0 and r["family"] in source_families]
                if both_classes(comparison):
                    subgroups.append(dict(**meta, manipulation=method, videos=len(comparison), **metrics(comparison)))
    for (cfg, cohort), trial in family_summary.items():
        stage, arm, mode, classifier = cfg
        if arm in ("expression187", "rppg69"):
            continue
        control_key = (1 if stage != 3 and classifier == "logistic" else stage, "expression187", mode, classifier)
        control = family_summary.get((control_key, cohort), {})
        families = sorted(trial.keys() & control.keys())
        if not families:
            continue
        info = dict(zip(KEYS, cfg, strict=True), cohort=cohort, control_stage=control_key[0],
                    control_arm="expression187")
        for name in SUMMARY_METRICS:
            deltas = [trial[f][name] - control[f][name] for f in families]
            paired.append(dict(**info, metric=name, improved_families=sum(d > 1e-12 for d in deltas),
                               degraded_families=sum(d < -1e-12 for d in deltas), **interval(deltas)))
            if name in ("balanced_accuracy", "roc_auc"):
                family_deltas.extend(dict(**info, family=f, metric=name, delta=d)
                                     for f, d in zip(families, deltas, strict=True))
    return summaries, subgroups, paired, family_deltas


def render(predictions, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = tuple(chosen[k] for k in KEYS)
    stage, arm, mode, classifier = cfg
    options = [(cfg, "Top observed configuration"),
               ((1, "expression187", mode, "logistic"), "Expression-only"),
               ((1, "fusion250", mode, "logistic"), "Fusion250"),
               ((1, "rppg69", mode, "logistic"), "Legacy rPPG-only"),
               ((2, "expression187", mode, classifier), "Expression-only, matched classifier")]
    series = []
    seen = set()
    for key, name in options:
        rows = [r for r in predictions if tuple(r[k] for k in KEYS) == key]
        if rows and key not in seen:
            series.append((rows, f"{name} ({key[1]}, {key[3]})"))
            seen.add(key)
    for kind in ("roc", "precision_recall", "confusion_matrices"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for cohort, ax in zip(("all", "new"), axes, strict=True):
            for rows, name in (series[:1] if kind == "confusion_matrices" else series):
                subset = [r for r in rows if cohort == "all" or r["cohort"] == cohort]
                y, scores = [r["label"] for r in subset], [r["score"] for r in subset]
                if kind == "roc":
                    x, tpr, _ = roc_curve(y, scores)
                    ax.plot(x, tpr, label=name)
                elif kind == "precision_recall":
                    precision, recall, _ = precision_recall_curve(y, scores)
                    ax.plot(recall, precision, label=name)
                else:
                    matrix = confusion_matrix(y, [r["prediction"] for r in subset], labels=(0, 1))
                    ax.imshow(matrix, cmap="Blues")
                    for (i, j), count in np.ndenumerate(matrix):
                        ax.text(j, i, str(count), ha="center", va="center",
                                color="white" if count > matrix.max() / 2 else "black")
                    ax.set(xticks=(0, 1), yticks=(0, 1), xlabel="Predicted label", ylabel="True label")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], "k:")
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
            elif kind == "precision_recall":
                ax.axhline(np.mean(y), color="k", linestyle=":")
                ax.set(xlabel="Recall", ylabel="Precision")
            if kind != "confusion_matrices":
                ax.legend(fontsize=8)
            ax.set_title("All temporal videos" if cohort == "all" else "New-family temporal videos")
        with io.BytesIO() as stream:
            fig.savefig(stream, format="png", dpi=140)
            write_bytes(f"{kind}_curves.png" if kind != "confusion_matrices" else f"{kind}.png", stream.getvalue())
        plt.close(fig)


def collect(configs, data):
    items, records, _, _, outer, multi, _, _, revision, _ = data
    predictions, timings, importance = [], [], []
    for cfg in configs:
        eligible = range(len(items)) if cfg[0] == 3 else sorted(multi)
        info = dict(zip(KEYS, cfg, strict=True))
        for fold in range(1, 6):
            result = read_result(cfg, fold, revision)
            if result is None:
                raise ValueError(f"Missing fold {fold} for {cfg}. Resume classification before evaluation.")
            expected = [i for i in eligible if outer[i] == fold]
            for i, row in zip(expected, result["rows"], strict=True):
                item = items[i]
                if ((row["clip_id"], row["label"], row["family"], row["cohort"], row["manipulation"],
                     row["available_windows"]) !=
                    (item["video_id"], item["label"], item["group_id"], item["phase48_cohort"],
                     item["manipulation_method"], len(records[i]))
                        or not np.isfinite(row["score"]) or row["prediction"] != int(row["score"] >= result["threshold"])):
                    raise ValueError(f"Invalid held-out video in {cfg}, fold {fold}")
                predictions.append(dict(**info, fold=fold, threshold=result["threshold"], **row))
            timings.append(dict(**info, fold=fold, seconds=result["seconds"], threshold=result["threshold"],
                                setting=result["setting"]))
            importance.extend(dict(**info, fold=fold, feature_group=group, auc_drop=drop)
                              for group, drop in result["importance"].items())
    return predictions, timings, importance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "all"), default="all",
                        help="Stage 1 produces interim tables; all requires every planned fold")
    args = parser.parse_args()
    started = perf_counter()
    ensure_space()
    data = audit()
    configs = plan(args.stage)
    predictions, timings, importance = collect(configs, data)
    summary, methods, paired, family_deltas = summarize(predictions)
    chosen = max((r for r in summary if r["stage"] != 3 and r["cohort"] == "all"),
                 key=lambda r: (r["balanced_accuracy"], r["roc_auc"]))
    identity = {k: chosen[k] for k in KEYS}
    best_settings = [r for r in timings if all(r[k] == v for k, v in identity.items())]
    write_table("baseline_comparison", [r for r in summary if r["stage"] == 1 and r["arm"] in BASELINES])
    write_table("selective_fusion_comparison", [r for r in summary if
                (r["stage"] == 1 and r["arm"] in SELECTIVE) or (r["stage"] == 2 and r["arm"] not in LATE)])
    write_table("late_fusion_comparison", [r for r in summary if r["arm"] in LATE], (*KEYS, "cohort"))
    write_table("temporal_comparison", [r for r in summary if r["stage"] == 1])
    write_table("development_robustness_comparison", [r for r in summary if r["stage"] == 3], (*KEYS, "cohort"))
    write_table("paired_family_comparison", paired)
    write_table("paired_family_deltas", family_deltas)
    write_table("per_manipulation", methods)
    write_table("video_predictions", predictions)
    write_table("confusion_matrices", [{k: r[k] for k in (*KEYS, "cohort", "tn", "fp", "fn", "tp")} for r in summary])
    write_table("feature_importance_by_fold", importance)
    import_summary = []
    for key, rows in grouped(importance, lambda r: tuple(r[k] for k in (*KEYS, "feature_group"))).items():
        drops = [r["auc_drop"] for r in rows]
        import_summary.append(dict(zip((*KEYS, "feature_group"), key, strict=True),
                                   folds=len(drops), mean_auc_drop=float(np.mean(drops))))
    write_table("feature_importance_summary", import_summary)
    write_table("classification_runtime", [dict(row, setting=json.dumps(row["setting"])) for row in timings])
    save(OUTPUT / "best_model_config.json", dict(
        **identity, selection="Post hoc ranking of outer OOF balanced accuracy, then ROC-AUC",
        scope=args.stage, metrics={k: chosen[k] for k in SUMMARY_METRICS}, folds=best_settings))
    render(predictions, chosen)
    stamps = {}
    for path in OUTPUT.glob("stage_*_complete.json"):
        stamp = json.loads(path.read_text())
        if stamp["revision"] == data[-2]:
            stamps[path.stem] = stamp
    save(OUTPUT / "runtime_summary.json", dict(scope=args.stage, experiments=len(configs), folds=len(timings),
         predictions=len(predictions), summed_worker_seconds=sum(r["seconds"] for r in timings),
         window_model_rows=sum(r["model_rows"] for r in predictions),
         classifier_seconds={k: sum(r["seconds"] for r in rows)
                             for k, rows in grouped(timings, lambda r: r["classifier"]).items()},
         latest_classification_invocations=stamps, evaluation_seconds=perf_counter() - started))
    new = next(r for r in summary if r["cohort"] == "new" and all(r[k] == v for k, v in identity.items()))
    robust = [r for r in summary if r["stage"] == 3 and r["cohort"] == "new"]
    report = [
        "# Phase 4.9 selective rPPG and facial expression fusion", "",
        f"Evaluation scope **{args.stage}**. The completed run contains {len(configs)} experiments and "
        f"{len(predictions):,} out-of-fold video predictions.",
        f"Across the shared {chosen['videos']}-video temporal cohort, the top observed arm was **{chosen['arm']}** "
        f"with **{chosen['classifier']}** and **{chosen['temporal']}** aggregation. Balanced accuracy was "
        f"**{chosen['balanced_accuracy']:.3f}** and ROC-AUC was **{chosen['roc_auc']:.3f}**. "
        f"On the {new['videos']} eligible videos from {new['families']} newly studied families, these were "
        f"**{new['balanced_accuracy']:.3f}** and **{new['roc_auc']:.3f}**.",
        "The top configuration is a post hoc development ranking. Its scores can reflect selection across experiments.",
        "",
        "The outer splits match Phase 4.8. Four inner source-family folds provide out-of-fold scores for threshold "
        "and parameter selection. Late fusion uses fixed base learner settings and learns its fusion rule from inner "
        "out-of-fold scores. The adaptive gate uses waveform coverage as a signal-availability proxy.",
        "PR-AUC is average precision. Manipulation comparisons pair each manipulation with available originals "
        "from its source families. Family intervals are descriptive percentile bootstrap intervals over families "
        "containing both classes. Group permutation importance is the average held-out ROC-AUC drop across folds. "
        "Dependent and redundant features can affect this measure.",
        "Phase 4.9 was informed by Phase 4.8. The newly studied families are a development robustness subset. "
        "All videos come from the approved training partition. Physiological accuracy was outside the evaluation.",
        "",
    ]
    matched = [row for row in paired if row["metric"] == "balanced_accuracy"
               and row["cohort"] in ("all", "new") and all(row[k] == v for k, v in identity.items())]
    if matched:
        report.extend(["## Paired balanced accuracy against expression-only", "",
                       "| Cohort | Mean change (points) | 95% descriptive interval (points) | Families |",
                       "| --- | ---: | ---: | ---: |"])
        report.extend(f"| {r['cohort']} | {100 * r['mean']:+.2f} | {100 * r['ci_low']:+.2f} to "
                      f"{100 * r['ci_high']:+.2f} | {r['families']} |" for r in matched)
        report.append("")
    if robust:
        report.extend(["## Center-window development check", "",
                       "All 42 newly studied families contribute 504 videos to this check.", "",
                       "| Arm | Balanced accuracy | ROC-AUC |", "| --- | ---: | ---: |"])
        report.extend(f"| {r['arm']} | {r['balanced_accuracy']:.3f} | {r['roc_auc']:.3f} |" for r in robust)
    report.extend(["", "[Baselines](baseline_comparison.csv) · [Selective fusion](selective_fusion_comparison.csv) · "
                   "[Late fusion](late_fusion_comparison.csv) · [Paired families](paired_family_comparison.csv) · "
                   "[Runtime](runtime_summary.json) · [ROC](roc_curves.png) · [Precision–recall](precision_recall_curves.png)"])
    write_bytes("PHASE49_REPORT.md", ("\n".join(report) + "\n").encode())
    print(f"Evaluated {len(predictions):,} predictions from {len(configs)} experiments.")


if __name__ == "__main__":
    main()

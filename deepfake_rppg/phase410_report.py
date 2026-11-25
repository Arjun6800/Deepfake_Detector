"""Phase 4.10 CSV summaries and plots, written only to the new experiment directory."""
from collections import defaultdict
import io

import numpy as np
from scipy import signal
from sklearn.metrics import precision_recall_curve, roc_curve

from .phase48_evaluate import interval
from .phase48_extract import windows
from .phase49_evaluate import metrics
from .phase410_features import observations, spectrum, waveform


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(root, fig, name):
    from .phase410_run import write
    with io.BytesIO() as stream:
        fig.savefig(stream, format="png", dpi=130)
        write(root / f"{name}.png", stream.getvalue())
    figures().close(fig)


def diagnostics(root, data, indices, variants):
    from .phase410_run import archive_data
    plt = figures()
    examples = [next(i for i in indices if data["items"][i]["label"] == label) for label in (0, 1)] if (
        {data["items"][i]["label"] for i in indices} == {0, 1}) else indices[:2]
    traces, ax1 = plt.subplots(len(examples), 1, figsize=(12, 3.4 * len(examples)), squeeze=False,
                              layout="constrained")
    spectra, ax2 = plt.subplots(len(examples), 1, figsize=(9, 3.4 * len(examples)), squeeze=False,
                               layout="constrained")
    for i, a, b in zip(examples, ax1[:, 0], ax2[:, 0], strict=True):
        item = data["items"][i]
        obs, _ = archive_data(item, data["phase48_revision"])
        fps = round(float(obs["fps"]), 12)
        grid = next(grid for center, grid in windows(obs["times"], fps) if center)
        colors, ids, nuisance, _, _ = observations(obs, grid, fps)
        from .window_features import longest_run
        for variant in variants:
            _, wave, groups = waveform(colors, ids, nuisance, fps, variant)
            a.plot(grid, wave, label=variant, alpha=.8)
            hz, psd = spectrum(wave[longest_run(np.isfinite(wave), groups)], fps)
            if len(hz):
                b.plot(hz, 10 * np.log10(np.maximum(psd, 1e-18)), label=variant)
        a.set(title=item["video_id"], xlabel="Time (s)", ylabel="Standardized aggregate")
        b.set(title=item["video_id"], xlabel="Frequency (Hz)", ylabel="Power density (dB)")
        for ax in (a, b):
            if ax.lines:
                ax.legend(fontsize=7, ncol=2)
    save_figure(root, traces, "waveform_comparison")
    save_figure(root, spectra, "psd_comparison")


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def both(rows):
    return {row["label"] for row in rows} == {0, 1}


def evaluation_figures(root, comparisons, chosen):
    plt = figures()
    keys = list(dict.fromkeys([chosen, ("expression187", "original", "logistic"),
                              ("rppg69", "original", "logistic"), ("temporal_spectral", "original", "logistic")]))
    curves, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    matrices, panels = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.7), squeeze=False, layout="constrained")
    for key, panel in zip(keys, panels[0], strict=True):
        rows = comparisons[key]
        label = f"{key[0]} ({key[2]})"
        y, score = [r["label"] for r in rows], [r["score"] for r in rows]
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=label)
        axes[1].plot(recall, precision, label=label)
        result = metrics(rows)
        matrix = [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]
        panel.imshow(matrix, cmap="Blues")
        for (i, j), count in np.ndenumerate(matrix):
            panel.text(j, i, str(count), ha="center", va="center",
                       color="white" if count > np.max(matrix) / 2 else "black")
        panel.set(xticks=(0, 1), yticks=(0, 1), title=key[0], xlabel="Predicted", ylabel="Observed")
    axes[0].plot([0, 1], [0, 1], "k:")
    axes[1].axhline(np.mean(y), linestyle=":", color="black")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=7)
    curves.canvas.draw()
    renderer = curves.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    for ax, name in zip(axes, ("roc_curves", "precision_recall_curves"), strict=True):
        box = Bbox.from_extents(*ax.get_tightbbox(renderer).extents).expanded(1.12, 1.12)
        from .phase410_run import write
        with io.BytesIO() as stream:
            curves.savefig(stream, format="png", dpi=130,
                           bbox_inches=box.transformed(curves.dpi_scale_trans.inverted()))
            write(root / f"{name}.png", stream.getvalue())
    plt.close(curves)
    save_figure(root, matrices, "confusion_matrices")


def report(root, folds, features, prior):
    from .phase410_run import table, save, write
    def config(row):
        return row["arm"], row["variant"], row["classifier"]

    def identity(key):
        return dict(zip(("arm", "variant", "classifier"), key, strict=True))

    predictions = [dict(**identity(config(f)), **r) for f in folds for r in f["rows"]]
    comparisons = grouped(predictions, config)
    means, per_method, family_metrics, deltas, paired, drops = [], [], {}, [], [], []
    for key, rows in comparisons.items():
        meta = identity(key)
        arms_families = grouped(rows, lambda r: r["family"])
        family_metrics[key] = {family: metrics(rs) for family, rs in arms_families.items() if both(rs)}
        summary = dict(**meta, videos=len(rows), families=len(arms_families), **metrics(rows))
        for metric in ("balanced_accuracy", "roc_auc"):
            summary.update({f"family_{metric}_{k}": v
                            for k, v in interval([r[metric] for r in family_metrics[key].values()]).items()})
        means.append(summary)
        for method in sorted({r["manipulation"] for r in rows if r["label"]}):
            manipulated = [r for r in rows if r["label"] and r["manipulation"] == method]
            families = {r["family"] for r in manipulated}
            matching = manipulated + [r for r in rows if not r["label"] and r["family"] in families]
            if both(matching):
                per_method.append(dict(**meta, manipulation=method, videos=len(matching), **metrics(matching)))
        by_group = defaultdict(list)
        for f in folds:
            if config(f) == key:
                for name, value in f["importance"].items():
                    by_group[name].append(value)
        drops.extend(dict(**meta, feature_group=name, folds=len(values), mean_auc_drop=float(np.mean(values)))
                     for name, values in by_group.items())

    for key, trial in family_metrics.items():
        for baseline in ("expression187", "temporal_spectral"):
            control_key = (baseline, "original", "logistic")
            if key == control_key:
                continue
            control = family_metrics[control_key]
            families = sorted(trial.keys() & control.keys())
            for metric in ("balanced_accuracy", "roc_auc"):
                values = [trial[f][metric] - control[f][metric] for f in families]
                meta = dict(**identity(key), baseline=baseline, metric=metric)
                paired.append(dict(**meta, **interval(values)))
                deltas.extend(dict(**meta, family=f, delta=d) for f, d in zip(families, values, strict=True))

    options = [r for r in means if r["arm"] not in ("nuisance", "motion_only", "missingness")]
    best = max(options, key=lambda row: (row["balanced_accuracy"], row["roc_auc"]))
    chosen = config(best)
    tables = {
        "ablation_comparison": means, "paired_family_comparison": paired, "paired_family_deltas": deltas,
        "feature_importance_summary": drops, "video_predictions": predictions, "per_manipulation": per_method,
        "confusion_matrices": [{k: r[k] for k in ("arm", "variant", "classifier", "tn", "fp", "fn", "tp")} for r in means],
        "classification_runtime": [dict(**identity(config(f)), fold=f["fold"], seconds=f["seconds"]) for f in folds],
        "temporal_feature_comparison": [r for r in means if r["arm"] in
                                        ("expression187", "temporal_spectral", "expression_temporal", "expression_enhanced")],
        "spectral_feature_comparison": [r for r in means if r["arm"] in
                                        ("expression187", "temporal_spectral", "expression_spectral", "expression_enhanced")],
        "cross_modal_comparison": [r for r in means if r["arm"] in
                                     ("expression187", "expression_cross", "expression_enhanced", "expression_enhanced_cross")],
        "feature_availability": [dict(group=b, feature=k, available=sum(r["features"][b][k] is not None
                                                                       for r in features.values()),
                                      videos=len(features))
                                 for b, fields in next(iter(features.values()))["features"].items() for k in fields],
    }
    for name, rows in tables.items():
        table(root / f"{name}.csv", rows)
    settings = [{k: f[k] for k in ("fold", "threshold", "setting")} for f in folds if config(f) == chosen]
    save(root / "best_model_config.json", dict(**best, selection="Post hoc development ranking", folds=settings))
    evaluation_figures(root, comparisons, chosen)
    lines = [
        "# Phase 4.10 rPPG and expression enhancement", "",
        f"Completed nested evaluation of {len(means)} configurations on {best['videos']} matched training videos.",
        "Pilot selection was exploratory and used overlapping source families. Final development estimates are",
        "conditional on that screening. The official validation and test partitions were untouched.", "",
        "| Arm | Waveform | Classifier | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | AP |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in means:
        values = " | ".join(f"{row[k]:.4f}" for k in ("accuracy", "balanced_accuracy", "precision", "recall",
                                                       "f1", "roc_auc", "pr_auc_ap"))
        lines.append(f"| {row['arm']} | {row['variant']} | {row['classifier']} | {values} |")
    lines.extend(["", f"Top observed configuration was {chosen[0]} with {chosen[1]} and {chosen[2]}.",
                  f"Historical Phase 4.9 winner achieved {float(prior['winner']['balanced_accuracy']):.4f} "
                  f"balanced accuracy and {float(prior['winner']['roc_auc']):.4f} ROC-AUC.", "",
                  "Thresholds and model settings came from four inner grouped folds. Family intervals are descriptive",
                  "bootstrap intervals over complete label pairs. AU proxies come from MediaPipe blendshapes.",
                  "The waveform and cross-modal features are experimental discriminators. Reference pulse measurements",
                  "were unavailable.", "",
                  "[Pilot](signal_enhancement_comparison.csv) | [Paired families](paired_family_comparison.csv) | "
                  "[Importance](feature_importance_summary.csv) | [Runtime](runtime_summary.json) | "
                  "[ROC](roc_curves.png) | [Precision-recall](precision_recall_curves.png)"])
    write(root / "PHASE410_REPORT.md", ("\n".join(lines) + "\n").encode())

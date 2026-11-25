"""Fixed, source-family-held-out classification feasibility experiments."""
from collections import defaultdict
import json
from time import perf_counter

import numpy as np
from scipy.special import expit
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .phase47_capture import OUTPUT
from .phase47_fusion import load_fusion
from .quality_experiment import write_table

SEED = 470
MODELS = ("logistic", "random_forest", "svm_rbf", "hist_gradient_boosting")
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_ap")


class FoldTransform:
    """Fit median imputation, nonconstant unique columns and scaling on training rows."""

    def fit(self, values):
        active = np.flatnonzero(np.isfinite(values).any(axis=0))
        observed = values[:, active]
        medians = np.nanmedian(observed, axis=0) if len(active) else np.empty(0)
        filled = np.where(np.isfinite(observed), observed, medians)
        nonconstant = np.flatnonzero(np.ptp(filled, axis=0) > 0)
        _, unique = np.unique(filled[:, nonconstant], axis=1, return_index=True)
        chosen = nonconstant[np.sort(unique)]
        self.columns = active[chosen]
        self.medians = medians[chosen]
        self.centers = filled[:, chosen].mean(axis=0)
        self.scales = filled[:, chosen].std(axis=0)
        return self

    def transform(self, values):
        selected = values[:, self.columns]
        return (np.where(np.isfinite(selected), selected, self.medians) - self.centers) / self.scales


def estimator(name):
    if name == "logistic":
        return LogisticRegression(C=1, max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=3,
                                      random_state=SEED, n_jobs=1)
    if name == "svm_rbf":
        return SVC(C=1, kernel="rbf", gamma="scale")
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=8,
                                              learning_rate=.08, l2_regularization=1, early_stopping=False,
                                              random_state=SEED)
    raise ValueError(f"Unknown classifier {name}")


def score(model, values):
    if isinstance(model, SVC):
        return expit(model.decision_function(values))
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def grouped_folds(labels, families):
    folds = list(LeaveOneGroupOut().split(np.zeros(len(labels)), labels, families))
    for train, heldout in folds:
        if set(families[train]) & set(families[heldout]):
            raise ValueError("Source family leakage")
        if len(set(labels[train])) != 2 or len(set(labels[heldout])) != 2:
            raise ValueError("Each split must support both classes")
    return folds


def aggregate_scores(rows):
    """Mean score across windows, with one prediction per video and no label mixing."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["clip_id"]].append(row)
    result = []
    for bucket in buckets.values():
        first = bucket[0]
        if any(any(row[k] != first[k] for k in ("fold", "split_group", "label", "manipulation")) for row in bucket):
            raise ValueError("A video's windows cross video identity or held-out folds")
        numeric_score = float(np.mean([row["score"] for row in bucket]))
        result.append(dict(**first, videoscore=numeric_score, prediction=int(numeric_score >= .5), windows=len(bucket)))
    return result


def evaluate(labels, scores):
    decisions = scores >= .5
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(labels, decisions), balanced_accuracy=balanced_accuracy_score(labels, decisions),
                precision=precision_score(labels, decisions, zero_division=0), recall=recall_score(labels, decisions),
                f1=f1_score(labels, decisions, zero_division=0), roc_auc=roc_auc_score(labels, scores),
                pr_auc_ap=average_precision_score(labels, scores), tn=tn, fp=fp, fn=fn, tp=tp)


def importance_blocks(model, transformed, y, names, schema, fold, arm, classifier):
    if arm not in ("fusion276", "fusion250"):
        return []
    original_auc = roc_auc_score(y, score(model, transformed))
    groups = {"expression": schema["expression"], **schema["rppg"]}
    rng = np.random.default_rng(SEED + fold)
    rows = []
    for name, columns in groups.items():
        selected = [i for i, key in enumerate(names) if key in columns]
        if not selected:
            continue
        drops = []
        for _ in range(12):
            shuffled = transformed.copy()
            shuffled[:, selected] = transformed[rng.permutation(len(y))][:, selected]
            drops.append(original_auc - roc_auc_score(y, score(model, shuffled)))
        rows.append(dict(arm=arm, classifier=classifier, fold=fold, feature_group=name,
                         columns=len(selected), mean_auc_drop=float(np.mean(drops))))
    return rows


def run_arm(rows, matrix, fields, labels, folds, arm, classifier, schema):
    windows, timings, importance, coefficients = [], [], [], []
    for fold, (train, heldout) in enumerate(folds, 1):
        started = perf_counter()
        transform = FoldTransform().fit(matrix[train])
        x_train, x_test = transform.transform(matrix[train]), transform.transform(matrix[heldout])
        names = [fields[i] for i in transform.columns]
        if classifier == "majority" or not names:
            model = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), labels[train])
            x_test = np.zeros((len(heldout), 1))
        else:
            model = estimator(classifier).fit(x_train, labels[train],
                                             sample_weight=compute_sample_weight("balanced", labels[train]))
        fit_seconds = perf_counter() - started
        started = perf_counter()
        predictions = score(model, x_test)
        predict_seconds = perf_counter() - started
        for i, value in zip(heldout, predictions, strict=True):
            identity = {key: rows[i][key] for key in
                        ("clip_id", "split_group", "label", "manipulation", "window_index",
                         "window_start_s", "window_end_s_exclusive")}
            windows.append(dict(**identity, fold=fold, score=float(value)))
        timings.append(dict(fold=fold, fit_and_preprocess_s=fit_seconds, predict_s=predict_seconds,
                            selected_features=len(names), train_videos=len(train), heldout_videos=len(heldout)))
        if names:
            importance.extend(importance_blocks(model, x_test, labels[heldout], names, schema, fold, arm, classifier))
        if isinstance(model, LogisticRegression):
            coefficients.extend(dict(arm=arm, fold=fold, feature=key, coefficient=float(value))
                                for key, value in zip(names, model.coef_[0], strict=True))
    return aggregate_scores(windows), timings, importance, coefficients


def paired_changes(folds):
    """Compare paired held-out family metrics under a prespecified Logistic Regression."""
    values = {(row["arm"], row["fold"]): row for row in folds if row["classifier"] == "logistic"}
    comparisons = (("fusion276", "expression187"), ("fusion250", "expression187"),
                   ("fusion276", "reliability26"), ("rppg_signal63", "reliability26"),
                   ("fusion276", "fusion256"), ("fusion276", "fusion250"))
    groups = sorted({fold for _, fold in values})
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(0, len(groups), (4000, len(groups)))
    results = []
    for trial, control in comparisons:
        for metric in ("balanced_accuracy", "roc_auc", "pr_auc_ap"):
            delta = np.array([values[trial, group][metric] - values[control, group][metric] for group in groups])
            low, high = np.quantile(delta[resamples].mean(axis=1), [.025, .975])
            results.append(dict(trial=trial, control=control, metric=metric, families=len(groups),
                                mean_delta=delta.mean(), bootstrap_low=low, bootstrap_high=high,
                                improved_folds=int((delta > 1e-12).sum()), degraded_folds=int((delta < -1e-12).sum())))
    return results


def render(predictions):
    import matplotlib.pyplot as plt

    for curve in ("roc", "precision_recall"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        for classifier, ax in zip(MODELS, axes.flat, strict=True):
            for (arm, name), rows in predictions.items():
                if name != classifier or arm == "missingness_probe":
                    continue
                labels, scores = prediction_arrays(rows)
                if curve == "roc":
                    x, y, _ = roc_curve(labels, scores)
                    value = roc_auc_score(labels, scores)
                else:
                    y, x, _ = precision_recall_curve(labels, scores)
                    value = average_precision_score(labels, scores)
                ax.plot(x, y, label=f"{arm} ({value:.3f})")
            if curve == "roc":
                ax.plot([0, 1], [0, 1], "k:", label="Majority (0.500)")
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
            else:
                ax.axhline(80 / 96, color="k", linestyle=":", label="Majority (0.833)")
                ax.set(xlabel="Recall", ylabel="Precision")
            ax.set_title(classifier.replace("_", " "))
            ax.legend(fontsize=8)
        fig.savefig(OUTPUT / f"{curve}_curves.png", dpi=140)
        plt.close(fig)

    arms = [arm for arm, classifier in predictions if classifier == "logistic" and arm != "missingness_probe"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    for arm, ax in zip(arms, axes.flat, strict=True):
        labels, scores = prediction_arrays(predictions[arm, "logistic"])
        matrix = confusion_matrix(labels, scores >= .5, labels=[0, 1])
        ax.imshow(matrix, vmin=0, vmax=80, cmap="Blues")
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > 40 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=("Real", "Fake"), yticklabels=("Real", "Fake"),
               xlabel="Predicted", ylabel="Observed", title=arm)
    fig.savefig(OUTPUT / "confusion_matrices_logistic.png", dpi=140)
    plt.close(fig)


def prediction_arrays(rows):
    return np.array([row["label"] for row in rows]), np.array([row["videoscore"] for row in rows])


def main():
    rows, matrices, schema = load_fusion()
    labels, families = (np.array([row[key] for row in rows]) for key in ("label", "split_group"))
    folds = grouped_folds(labels, families)
    if len(rows) != 96 or len(folds) != 8:
        raise ValueError("Expected 96 videos across eight independent families")
    fields = dict(schema["ablations"])
    matrices["missingness_probe"] = (~np.isfinite(matrices["fusion276"])).astype(float)
    fields["missingness_probe"] = fields["fusion276"]
    matrices["majority"] = np.zeros((len(rows), 0))
    fields["majority"] = []
    plans = [(arm, model) for arm in schema["ablations"] for model in MODELS]
    plans.extend([("missingness_probe", "logistic"), ("majority", "majority")])
    predictions, summaries = {}, []
    measurements, fold_outputs, subgroup, blocks, weights = [], [], [], [], []
    with threadpool_limits(limits=1):
        for arm, classifier in plans:
            result, timing, importance, coefficients = run_arm(rows, matrices[arm], fields[arm], labels, folds,
                                                              arm, classifier, schema)
            predictions[arm, classifier] = result
            prefix = dict(arm=arm, classifier=classifier)
            labels_oof, scores_oof = prediction_arrays(result)
            fold_results = []
            for measured in timing:
                sub = [row for row in result if row["fold"] == measured["fold"]]
                fold_results.append(dict(**prefix, **measured, heldout_family=sub[0]["split_group"],
                                         **evaluate(*prediction_arrays(sub))))
            fold_outputs.extend(fold_results)
            summaries.append(dict(**prefix, videos=len(result), requested_features=len(fields[arm]),
                                  **evaluate(labels_oof, scores_oof),
                                  **{f"{metric}_fold_sd": float(np.std([r[metric] for r in fold_results], ddof=1))
                                     for metric in METRICS},
                                  **{f"{metric}_fold_mean": float(np.mean([r[metric] for r in fold_results]))
                                     for metric in METRICS},
                                  fit_and_preprocess_s=sum(r["fit_and_preprocess_s"] for r in timing),
                                  predict_s=sum(r["predict_s"] for r in timing)))
            measurements.extend(dict(**prefix, **row) for row in result)
            for method in sorted({row["manipulation"] for row in result if row["label"] == 1}):
                pair = [row for row in result if row["label"] == 0 or row["manipulation"] == method]
                subgroup.append(dict(**prefix, manipulation=method, videos=len(pair),
                                     **evaluate(*prediction_arrays(pair))))
            blocks.extend(importance)
            weights.extend(coefficients)
    write_table(OUTPUT, "ablation_comparison", summaries)
    write_table(OUTPUT, "fold_metrics", fold_outputs)
    write_table(OUTPUT, "video_predictions", measurements)
    write_table(OUTPUT, "per_manipulation", subgroup)
    write_table(OUTPUT, "confusion_matrices", [{key: row[key] for key in ("arm", "classifier", "tn", "fp", "fn", "tp")}
                                                for row in summaries])
    write_table(OUTPUT, "paired_family_deltas", paired_changes(fold_outputs))
    write_table(OUTPUT, "feature_group_importance", blocks)
    write_table(OUTPUT, "logistic_feature_coefficients", weights)
    write_table(OUTPUT, "fold_assignment", [dict(clip_id=rows[i]["clip_id"], split_group=rows[i]["split_group"], fold=k)
                                             for k, (_, test) in enumerate(folds, 1) for i in test])
    (OUTPUT / "evaluation.json").write_text(json.dumps(dict(
        sklearn=sklearn.__version__, videos=len(rows), folds=len(folds), completed_runs=len(plans),
        seed=SEED, decision_threshold=.5, video_aggregation="mean of window scores",
        primary_classifier="logistic", primary_metric="family-mean balanced accuracy",
    ), indent=2) + "\n", encoding="utf-8")
    render(predictions)
    print(f"{len(rows)} videos, {len(folds)} held-out source families, {len(plans)} evaluation runs")


if __name__ == "__main__":
    main()

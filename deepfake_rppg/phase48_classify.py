"""Nested grouped feasibility evaluation on completed Phase 4.8 checkpoints."""
import json
from time import perf_counter
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from .phase47_classification import MODELS, FoldTransform, estimator, importance_blocks, score
from .phase48_extract import CHECKPOINTS, OUTPUT, PHASE47, approved_selection, ensure_space, read_checkpoint, save, signature, token
from .quality_experiment import write_table

SEED = 480
TEMPORAL = ("center", "mean", "mean_std", "median", "window_scores")
GRID = {"logistic": ("C", (.1, 1.0)), "random_forest": ("max_depth", (3, 5)),
        "svm_rbf": ("C", (.3, 1.0)), "hist_gradient_boosting": ("max_leaf_nodes", (3, 7))}


def fold_groups(items, indices, count, seed):
    """Assign families as blocks and distribute the retained cohort across folds."""
    family_cohort = {items[i]["group_id"]: items[i]["phase48_cohort"] for i in indices}
    if len(family_cohort) < count:
        raise ValueError("Insufficient independent source families")
    rng = np.random.default_rng(seed)
    families = []
    for cohort in ("retained", "new"):
        names = sorted(group for group, kind in family_cohort.items() if kind == cohort)
        families.extend(rng.permutation(names))
    for fold in range(count):
        held = set(families[fold::count])
        yield ([i for i in indices if items[i]["group_id"] not in held],
               [i for i in indices if items[i]["group_id"] in held])


def load():
    items = approved_selection()
    schema = json.loads((PHASE47 / "feature_schema.json").read_text(encoding="utf-8"))
    revision = signature()
    windows = []
    missing = []
    for item in items:
        clip = item["video_id"]
        checkpoint = read_checkpoint(CHECKPOINTS / f"{token(clip)}.json", token(revision + item["sha256"]), schema)
        if checkpoint is None or checkpoint["clip_id"] != clip or checkpoint["cohort"] != item["phase48_cohort"]:
            missing.append(clip)
        else:
            windows.append(checkpoint["windows"])
    if missing:
        raise ValueError(f"{len(missing)} required checkpoints incomplete. Resume extraction. First: {missing[:5]}")
    return items, windows, schema, revision


def representation(records, fields, arm, temporal):
    if temporal == "center":
        records = [next(row for row in records if row["center"])]
    source = ("legacy" if arm == "rppg69" else "green" if arm == "green89"
              else "audit" if arm == "nuisance_probe" else "pos")
    if arm == "fusion256":
        values = np.array([[(row["legacy"] if key in row["legacy"] else row["pos"])[key] for key in fields]
                           for row in records], dtype=float)
    else:
        values = np.array([[row[source][key] for key in fields] for row in records], dtype=float)
    if arm == "missingness_only":
        values = np.isnan(values).astype(float)
    if np.isinf(values).any():
        raise ValueError("Infinite model input")
    if temporal in ("center", "window_scores"):
        return values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-missing features remain NaN.
        if temporal == "median":
            return np.nanmedian(values, axis=0, keepdims=True)
        mean = np.nanmean(values, axis=0, keepdims=True)
        if temporal == "mean_std":
            return np.column_stack((mean, np.nanstd(values, axis=0, keepdims=True)))
    return mean


def stack(arrays, indices):
    return np.concatenate([arrays[i] for i in indices], axis=0), np.array([len(arrays[i]) for i in indices])


def fit_predict(arrays, labels, train, heldout, classifier, setting):
    started = perf_counter()
    x, train_counts = stack(arrays, train)
    test, held_counts = stack(arrays, heldout)
    transform = FoldTransform().fit(x)
    train_label = labels[train]
    y = np.repeat(train_label, train_counts)
    class_counts = np.bincount(train_label, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("A fold lacks both classes")
    # Mean weight is one, with equal total video weights within each class.
    weights = np.repeat(len(y) / (2 * class_counts[train_label] * train_counts), train_counts)
    if classifier == "majority" or not len(transform.columns):
        model = DummyClassifier(strategy="prior").fit(np.zeros((len(y), 1)), y, sample_weight=(
            None if classifier == "majority" else weights))
        test_x = np.zeros((len(test), 1))
        indices = []
    else:
        parameter = GRID[classifier][0]
        model = estimator(classifier).set_params(**{parameter: setting})
        model.fit(transform.transform(x), y, sample_weight=weights)
        test_x = transform.transform(test)
        indices = transform.columns
    fit_s = perf_counter() - started
    started = perf_counter()
    raw_scores = score(model, test_x)
    result = np.add.reduceat(raw_scores, np.r_[0, held_counts.cumsum()[:-1]]) / held_counts
    return result, model, test_x, indices, fit_s, perf_counter() - started, held_counts


def choose(arrays, items, labels, train, classifier, number):
    if classifier == "majority":
        return None, None
    inner = list(fold_groups(items, train, 4, SEED + number))
    candidates = []
    for value in GRID[classifier][1]:
        metrics = []
        for fit, test in inner:
            predictions, *_ = fit_predict(arrays, labels, fit, test, classifier, value)
            metrics.append(balanced_accuracy_score(labels[test], predictions >= .5))
        candidates.append((value, float(np.mean(metrics))))
    return max(candidates, key=lambda pair: pair[1])


def main():
    ensure_space()
    items, records, schema, revision = load()
    labels = np.array([item["label"] for item in items])
    indices = list(range(len(items)))
    multi = {i for i, rows in enumerate(records) if len(rows) > 1}
    if len({items[i]["group_id"] for i in multi}) < 5:
        raise ValueError("Insufficient multi-window families")
    assignment = list(fold_groups(items, indices, 5, SEED))
    save(OUTPUT / "expanded_feature_schema.json", schema)
    arms = dict(schema["ablations"])
    arms.update(missingness_only=arms["fusion276"], nuisance_probe=["motion", "light_cv", "bits_per_pixel"],
                majority=[])
    configs = [(arm, "replication", "center", classifier) for arm in schema["ablations"]
               for classifier in MODELS]
    configs.extend(("missingness_only", "replication", "center", classifier) for classifier in MODELS)
    configs.extend([("nuisance_probe", "replication", "center", "logistic"),
                    ("majority", "replication", "center", "majority")])
    configs.extend(("fusion250", "temporal", temporal, classifier)
                   for temporal in TEMPORAL for classifier in MODELS)
    predictions, timings, importance = [], [], []
    with threadpool_limits(limits=1):
        for arm, comparison, temporal, classifier in configs:
            arrays = [representation(row, arms[arm], arm, temporal) for row in records]
            prefix = dict(arm=arm, comparison=comparison, temporal=temporal, classifier=classifier)
            for fold, (train, heldout) in enumerate(assignment, 1):
                if comparison == "temporal":
                    train, heldout = [[i for i in part if i in multi] for part in (train, heldout)]
                if set(labels[train]) != {0, 1} or set(labels[heldout]) != {0, 1}:
                    raise ValueError("Temporal subset lacks both classes in an outer fold")
                started = perf_counter()
                setting, inner_ba = choose(arrays, items, labels, train, classifier, fold)
                nested_s = perf_counter() - started
                result, model, x_test, columns, fit_s, predict_s, counts = fit_predict(
                    arrays, labels, train, heldout, classifier, setting)
                timings.append(dict(**prefix, fold=fold, selected_setting=setting, inner_balanced_accuracy=inner_ba,
                                    nested_s=nested_s, fit_s=fit_s, predict_s=predict_s, train=len(train),
                                    heldout=len(heldout), features_retained=len(columns)))
                for i, value, count in zip(heldout, result, counts, strict=True):
                    item = items[i]
                    predictions.append(dict(**prefix, fold=fold, clip_id=item["video_id"], label=item["label"],
                                            family=item["group_id"], cohort=item["phase48_cohort"],
                                            manipulation=item["manipulation_method"], score=float(value),
                                            prediction=int(value >= .5), model_rows=int(count),
                                            available_windows=len(records[i])))
                if comparison == "replication" and arm in ("fusion276", "fusion250") and len(columns):
                    names = [arms[arm][i] for i in columns]
                    for family in {items[i]["group_id"] for i in heldout}:
                        mask = np.array([items[i]["group_id"] == family for i in heldout])
                        group_rows = importance_blocks(model, x_test[mask], labels[heldout][mask], names,
                                                       schema, fold, arm, classifier)
                        importance.extend(dict(row, family=family) for row in group_rows)
            print(f"{comparison} {arm} {temporal} {classifier} complete", flush=True)
    ensure_space()
    write_table(OUTPUT, "video_predictions", predictions)
    write_table(OUTPUT, "classification_runtime", timings)
    write_table(OUTPUT, "feature_importance", importance)
    write_table(OUTPUT, "fold_assignment", [
        dict(clip_id=items[i]["video_id"], family=items[i]["group_id"], cohort=items[i]["phase48_cohort"], fold=k)
        for k, (_, heldout) in enumerate(assignment, 1) for i in heldout])
    save(OUTPUT / "classification_complete.json", dict(revision=revision, runs=len(configs), videos=len(items),
                                                       multiwindow=len(multi), predictions=len(predictions),
                                                       outer_folds=5, inner_folds=4))


if __name__ == "__main__":
    main()

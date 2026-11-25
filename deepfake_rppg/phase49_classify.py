"""Phase 4.9 checkpoint audit and resumable, grouped classification."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from hashlib import sha256
import json
from pathlib import Path
from shutil import disk_usage
from time import perf_counter
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from .phase47_classification import FoldTransform, estimator, score
from .phase48_classify import GRID, SEED, fold_groups, load, stack
from .phase48_extract import BUFFER, CHECKPOINTS, MIN_FREE, OUTPUT as PREVIOUS, atomic, token, windows

OUTPUT = Path("outputs/test12")
BASELINES = ("expression187", "rppg69", "fusion256", "fusion276", "fusion250")
AGGREGATIONS = ("center", "mean", "mean_std", "window_scores")
SELECTIVE = ("temporal", "spectral", "temporal_spectral", "spatial", "pos", "green", "auto")
LATE = ("stack", "weighted", "adaptive")


def ensure_space():
    if disk_usage(OUTPUT).free < MIN_FREE + BUFFER:
        raise RuntimeError("Phase 4.9 requires 2 GiB free plus a 64 MiB write buffer")


def save(path, data):
    ensure_space()
    atomic(path, lambda stream: stream.write(json.dumps(data, allow_nan=False).encode()), reserve=False)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plan(stage="all"):
    configs = [(1, arm, mode, "logistic") for arm in BASELINES for mode in AGGREGATIONS]
    configs += [(1, arm, "mean", "logistic") for arm in SELECTIVE]
    configs += [(2, "regularized", "mean", "logistic")]
    configs += [(2, arm, "mean", model) for arm in ("expression187", "temporal", "fusion250")
                for model in GRID if model != "logistic"]
    configs += [(2, arm, "mean", "logistic") for arm in LATE]
    configs += [(3, arm, "center", "logistic") for arm in (*BASELINES, "temporal")]
    return [cfg for cfg in configs if stage == "all" or cfg[0] == int(stage)]


def schema_fields(schema):
    ablations = schema["ablations"]
    expression = [("pos", name) for name in schema["expression"]]
    groups = schema["rppg"]
    fields = {
        arm: [("legacy" if name in ablations["rppg69"] else "pos", name) for name in ablations[arm]]
        for arm in BASELINES
    }
    for arm, blocks in {
        "temporal": ("temporal",), "spectral": ("spectral",),
        "temporal_spectral": ("temporal", "spectral"), "spatial": ("spatial",),
        "pos": ("temporal", "spectral", "spatial"), "green": ("temporal", "spectral", "spatial"),
        "auto": ("signal", "temporal", "spectral", "spatial"),
        "regularized": ("signal", "temporal", "spectral", "spatial"),
    }.items():
        source = "green" if arm == "green" else "pos"
        source_fields = ablations["green89" if arm == "green" else "fusion276"]
        extra = [(source, name) for block in blocks for name in groups[block]]
        if any(name not in source_fields for _, name in extra):
            raise ValueError(f"Missing checkpoint schema fields for {arm}")
        fields[arm] = expression + extra
    fields["pos_temporal"] = [("pos", name) for name in groups["temporal"]]
    return fields


def audit():
    items, records, schema, revision = load()
    stamp = json.loads((PREVIOUS / "classification_complete.json").read_text())
    if stamp["revision"] != revision or schema != json.loads((PREVIOUS / "expanded_feature_schema.json").read_text()):
        raise ValueError("Phase 4.8 revision or schema differs from the completed classification")
    fields = schema_fields(schema)
    old_folds = read_csv(PREVIOUS / "fold_assignment.csv")
    mapping = {row["clip_id"]: row for row in old_folds}
    if len(mapping) != len(items) or len(old_folds) != len(items):
        raise ValueError("Phase 4.8 fold assignment has missing or duplicate IDs")
    ids = {}
    membership = {}
    fold_ids = []
    centers = {}
    fingerprint = sha256()
    for i, (item, rows) in enumerate(zip(items, records, strict=True)):
        clip = item["video_id"]
        family, cohort = item["group_id"], item["phase48_cohort"]
        meta = mapping[clip]
        fold = int(meta["fold"])
        if (family, cohort, fold) != (meta["family"], meta["cohort"], membership.setdefault(family, fold)):
            raise ValueError(f"Inconsistent fold metadata for {clip}")
        ids[clip] = i
        fold_ids.append(fold)
        checkpoint_id = token(clip)
        fingerprint.update((CHECKPOINTS / f"{checkpoint_id}.json").read_bytes())
        with np.load(CHECKPOINTS / f"{checkpoint_id}.npz", allow_pickle=False) as archive:
            if str(archive["key"]) != token(revision + item["sha256"]):
                raise ValueError(f"Observation checkpoint key differs for {clip}")
            times = archive["times"]
            fps = round(float(archive["fps"]), 12)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid observation timestamps for {clip}")
        starts = np.array([row["start_s"] for row in rows])
        if not np.isfinite(starts).all() or starts[0] < 0 or np.any(np.diff(starts) <= 0):
            raise ValueError(f"Invalid window ordering for {clip}")
        for j, (row, (center, grid)) in enumerate(zip(rows, windows(times, fps), strict=True)):
            support = np.flatnonzero((times >= grid[0] - 1e-8) & (times < grid[-1] + 1 / fps - 1e-8))
            if (row["window"] != j or not np.isclose(row["end_s"] - row["start_s"], 8, rtol=0, atol=1e-8)
                    or row["center"] != center or not np.isclose(row["start_s"], grid[0], rtol=0, atol=1e-8)
                    or not len(support) or row["first_frame"] != support[0] or row["last_frame"] != support[-1]
                    or row["frames"] != len(support) or row["grid_samples"] != len(grid)):
                raise ValueError(f"Invalid checkpoint window support for {clip}")
            if row["center"]:
                centers[i] = j
            for source in ("pos", "green", "legacy"):
                if np.isinf(np.array(list(row[source].values()), dtype=float)).any():
                    raise ValueError(f"Infinite feature value for {clip}")
    fold_ids = np.array(fold_ids)
    if set(fold_ids) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected five established outer folds")
    expected_folds = fold_groups(items, range(len(items)), 5, SEED)
    for fold, (_, held) in enumerate(expected_folds, 1):
        if set(held) != set(np.flatnonzero(fold_ids == fold)):
            raise ValueError("Phase 4.8 fold CSV differs from deterministic assignment")
    multi = {i for i, rows in enumerate(records) if len(rows) >= 2}
    predictions = read_csv(PREVIOUS / "video_predictions.csv")
    experiments = defaultdict(set)
    for row in predictions:
        i = ids[row["clip_id"]]
        item = items[i]
        key = tuple(row[k] for k in ("comparison", "arm", "temporal", "classifier"))
        if (i in experiments[key] or int(row["label"]) != item["label"] or int(row["fold"]) != fold_ids[i]
                or row["family"] != item["group_id"] or row["cohort"] != item["phase48_cohort"]
                or row["manipulation"] != item["manipulation_method"]
                or int(row["available_windows"]) != len(records[i])):
            raise ValueError("Phase 4.8 prediction metadata does not align with the checkpoints")
        experiments[key].add(i)
    if (len(predictions), len(experiments), len(multi)) != (stamp["predictions"], stamp["runs"], stamp["multiwindow"]):
        raise ValueError("Phase 4.8 cohort counts differ from completion metadata")
    if any(seen != (multi if key[0] == "temporal" else set(ids.values())) for key, seen in experiments.items()):
        raise ValueError("Phase 4.8 experiment cohort differs from checkpoint eligibility")
    for source in (Path(__file__), Path(__file__).with_name("phase47_classification.py"),
                   Path(__file__).with_name("phase48_classify.py"), PREVIOUS / "fold_assignment.csv"):
        fingerprint.update(source.read_bytes())
    import sklearn
    fingerprint.update(f"{revision}:{np.__version__}:{sklearn.__version__}".encode())

    reliability = [name for name in schema["rppg"]["reliability"] if name.endswith("__waveform_coverage")]
    if not reliability:
        raise ValueError("No compatible waveform coverage reliability fields")
    coverage = np.array([np.mean([float(row["pos"][name] or 0) for row in rows for name in reliability])
                         for rows in records])
    if not np.all((coverage >= 0) & (coverage <= 1)):
        raise ValueError("Waveform coverage outside [0, 1]")
    audit_result = dict(
        phase48_revision=revision, videos=len(items), predictions=len(predictions), experiments=len(experiments),
        families=len(membership), new_families=len({row["group_id"] for row in items if row["phase48_cohort"] == "new"}),
        temporal_videos=len(multi), temporal_families=len({items[i]["group_id"] for i in multi}),
        temporal_new_families=len({items[i]["group_id"] for i in multi if items[i]["phase48_cohort"] == "new"}),
        temporal_new_videos=sum(items[i]["phase48_cohort"] == "new" for i in multi),
        fields={arm: len(names) for arm, names in fields.items()}, reliability_fields=reliability,
        coverage_min=float(coverage.min()), coverage_max=float(coverage.max()),
    )
    return items, records, schema, fields, fold_ids, multi, centers, coverage, fingerprint.hexdigest(), audit_result


def arrays_for(records, fields, centers, mode):
    arrays = [np.array([[row[source][name] for source, name in fields] for row in rows], dtype=float)
              for rows in records]
    if mode == "center":
        return [x[centers[i]:centers[i] + 1] for i, x in enumerate(arrays)]
    if mode == "window_scores":
        return arrays
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = [np.nanmean(x, axis=0, keepdims=True) for x in arrays]
        if mode == "mean_std":
            return [np.column_stack((mean, np.nanstd(x, axis=0, keepdims=True)))
                    for x, mean in zip(arrays, means, strict=True)]
    return means


def fit(arrays, labels, train, classifier, setting, selectable=()):
    values, counts = stack(arrays, train)
    transform = FoldTransform().fit(values)
    x = transform.transform(values)
    y = np.repeat(labels[train], counts)
    weights = np.repeat(len(y) / (2 * np.bincount(labels[train], minlength=2)[labels[train]] * counts), counts)
    candidates = np.flatnonzero(np.isin(transform.columns, selectable))
    selected = np.arange(x.shape[1])
    if len(candidates):
        # Absolute class-conditional mean difference after fold-local scaling.
        separation = np.abs(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))
        ranked = candidates[np.argsort(-separation[candidates], kind="stable")]
        selected = np.sort(np.r_[np.setdiff1d(selected, candidates), ranked[:12]])
    if not len(selected):
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="prior").fit(x, y, sample_weight=weights)
    else:
        model = estimator(classifier).set_params(**{GRID[classifier][0]: setting})
        if classifier == "hist_gradient_boosting":
            model.set_params(max_bins=63)
        model.fit(x[:, selected], y, sample_weight=weights)
    return transform, selected, model


def predict(fitted, values, counts):
    transform, selected, model = fitted
    raw = score(model, transform.transform(values)[:, selected])
    return np.add.reduceat(raw, np.r_[0, counts.cumsum()[:-1]]) / counts


def threshold(labels, scores):
    fpr, tpr, cuts = roc_curve(labels, scores, drop_intermediate=False)
    # Use a finite all-negative threshold for serialization.
    cuts[0] = np.nextafter(scores.max(), np.inf)
    candidates = np.flatnonzero(np.isclose(tpr - fpr, (tpr - fpr).max(), rtol=0, atol=1e-12))
    return float(cuts[candidates[np.argmin(np.abs(cuts[candidates] - .5))]])


def objective(labels, scores):
    cut = threshold(labels, scores)
    decisions = scores >= cut
    ba = .5 * (decisions[labels == 1].mean() + (~decisions[labels == 0]).mean())
    return float(ba), float(roc_auc_score(labels, scores))


def inner_oof(arrays, labels, outer_train, splits, classifier, setting, selectable=()):
    predicted = np.full(len(labels), np.nan)
    allowed = set(outer_train)
    for train, held in splits:
        if set(train) & set(held) or set(train) | set(held) != allowed or np.isfinite(predicted[held]).any():
            raise ValueError("Invalid inner OOF partition")
        model = fit(arrays, labels, train, classifier, setting, selectable)
        predicted[held] = predict(model, *stack(arrays, held))
    if not np.isfinite(predicted[outer_train]).all():
        raise ValueError("Incomplete inner OOF predictions")
    return predicted[outer_train]


def blend(scores, reliability, weight, exponent):
    gate = weight * reliability ** exponent
    return scores[:, 0] + gate * (scores[:, 1] - scores[:, 0])


def choose_fusion(arm, labels, oof, reliability):
    if arm == "stack":
        meta = LogisticRegression(C=.3, class_weight="balanced").fit(oof, labels)
        stacked = meta.predict_proba(oof)[:, 1]
        if objective(labels, stacked) > objective(labels, oof[:, 0]):
            return stacked, dict(coefficients=meta.coef_[0].tolist(), intercept=float(meta.intercept_[0])), meta
        return oof[:, 0], dict(weight=0., exponent=0), None
    exponents = (1, 2) if arm == "adaptive" else (0,)
    candidates = [(weight, power) for weight in (0., .25, .5, .75, 1.) for power in exponents]
    weight, power = max(candidates, key=lambda pair: objective(labels, blend(oof, reliability, *pair)))
    return blend(oof, reliability, weight, power), dict(weight=weight, exponent=power), None


def fit_late(arm, arrays, labels, train, held, splits, coverage):
    setting = .1
    oof = np.column_stack([inner_oof(x, labels, train, splits, "logistic", setting) for x in arrays])
    train_scores, fusion, meta = choose_fusion(arm, labels[train], oof, coverage[train])
    held_scores = np.column_stack([predict(fit(x, labels, train, "logistic", setting), *stack(x, held))
                                   for x in arrays])
    if meta is None:
        scores = blend(held_scores, coverage[held], fusion["weight"], fusion["exponent"])
    else:
        scores = meta.predict_proba(held_scores)[:, 1]
    drops = {}
    rng = np.random.default_rng(SEED)
    for j, group in enumerate(("expression", "pos_temporal", "reliability")):
        if j == 2 and arm != "adaptive":
            continue
        shuffled = held_scores.copy()
        reliability = coverage[held].copy()
        if j == 2:
            rng.shuffle(reliability)
        else:
            rng.shuffle(shuffled[:, j])
        changed = (meta.predict_proba(shuffled)[:, 1] if meta is not None
                   else blend(shuffled, reliability, fusion["weight"], fusion["exponent"]))
        drops[group] = float(roc_auc_score(labels[held], scores) - roc_auc_score(labels[held], changed))
    return scores, train_scores, dict(C=setting, **fusion), drops


def importance(fitted, values, counts, labels, fields, schema, original, fold):
    groups = {name: group for group, names in {"expression": schema["expression"], **schema["rppg"]}.items()
              for name in names}
    active = fitted[0].columns[fitted[1]]
    columns = defaultdict(list)
    for index in active:
        source, name = fields[index % len(fields)]
        group = groups.get(name, "legacy")
        columns[group if group == "expression" else f"{source}_{group}"].append(index)
    rng = np.random.default_rng(SEED + fold)
    start = np.r_[0, counts.cumsum()[:-1]]
    permutation = np.arange(len(values))
    for count in np.unique(counts):
        members = start[counts == count]
        order = rng.permutation(members)
        permutation[(members[:, None] + np.arange(count)).ravel()] = (order[:, None] + np.arange(count)).ravel()
    auc = roc_auc_score(labels, original)
    drops = {}
    for group, indices in columns.items():
        changed = values.copy()
        changed[:, indices] = values[permutation][:, indices]
        drops[group] = float(auc - roc_auc_score(labels, predict(fitted, changed, counts)))
    return drops


def evaluate_fold(cfg, fold, data):
    stage, arm, mode, classifier = cfg
    items, records, schema, fields, outer, multi, centers, coverage, revision, _ = data
    eligible = np.arange(len(items)) if stage == 3 else np.array(sorted(multi))
    train, held = eligible[outer[eligible] != fold], eligible[outer[eligible] == fold]
    labels = np.array([row["label"] for row in items])
    families = np.array([row["group_id"] for row in items])
    splits = list(fold_groups(items, train, 4, SEED + fold))
    for a, b in [(train, held), *splits]:
        if set(families[a]) & set(families[b]) or set(labels[a]) != {0, 1} or set(labels[b]) != {0, 1}:
            raise ValueError("Invalid grouped split")
    started = perf_counter()
    if arm in LATE:
        arrays = [arrays_for(records, fields[name], centers, mode) for name in ("expression187", "pos_temporal")]
        scores, oof, setting, drops = fit_late(arm, arrays, labels, train, held, splits, coverage)
        model_rows = np.ones(len(held), dtype=int)
    else:
        arrays = arrays_for(records, fields[arm], centers, mode)
        selectable = [i for i, (_, name) in enumerate(fields[arm]) if name not in schema["expression"]] if arm == "auto" else ()
        grid = (.01, .03) if arm == "regularized" else GRID[classifier][1]
        candidates = [(value, inner_oof(arrays, labels, train, splits, classifier, value, selectable)) for value in grid]
        value, oof = max(candidates, key=lambda candidate: objective(labels[train], candidate[1]))
        setting = {GRID[classifier][0]: value}
        fitted = fit(arrays, labels, train, classifier, value, selectable)
        if arm == "auto":
            setting["selected_rppg"] = [fields[arm][i][1] for i in fitted[0].columns[fitted[1]] if i in selectable]
        values, model_rows = stack(arrays, held)
        scores = predict(fitted, values, model_rows)
        drops = importance(fitted, values, model_rows, labels[held], fields[arm], schema, scores, fold)
    cut = threshold(labels[train], oof)
    rows = []
    for i, value, count in zip(held, scores, model_rows, strict=True):
        item = items[i]
        rows.append(dict(clip_id=item["video_id"], label=item["label"], family=item["group_id"],
                         cohort=item["phase48_cohort"], manipulation=item["manipulation_method"],
                         score=float(value), prediction=int(value >= cut), model_rows=int(count),
                         available_windows=len(records[i])))
    return dict(config=list(cfg), fold=fold, revision=revision, rows=rows, threshold=cut, setting=setting,
                importance=drops, seconds=perf_counter() - started)


def checkpoint_path(cfg, fold):
    return OUTPUT / "classification" / ("_".join(map(str, (*cfg, fold))) + ".json")


def read_result(cfg, fold, revision):
    path = checkpoint_path(cfg, fold)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (result["revision"], result["config"], result["fold"]) != (revision, list(cfg), fold):
        raise ValueError(f"Checkpoint input or implementation changed. Archive the Phase 4.9 classification directory before restarting: {path}")
    return result


WORKER_DATA = None


def initialize(data):
    global WORKER_DATA
    WORKER_DATA = data


def run_fold(cfg, fold):
    ensure_space()
    with threadpool_limits(limits=1):
        result = evaluate_fold(cfg, fold, WORKER_DATA)
    save(checkpoint_path(cfg, fold), result)
    return cfg, fold


def workers_for(requested):
    import ctypes
    import os

    if os.name != "nt":
        return requested
    # MEMORYSTATUSEX has two 32-bit fields followed by seven 64-bit fields.
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("load", ctypes.c_uint32),
                    ("physical_total", ctypes.c_uint64), ("physical_free", ctypes.c_uint64),
                    ("remaining", ctypes.c_uint64 * 5)]

    memory = MemoryStatus()
    memory.size = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        return 1
    free_gib = memory.physical_free / 1024**3
    return min(requested, 2 if free_gib >= 3 else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--audit", action="store_true", help="Validate inputs without fitting any model")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    ensure_space()
    data = audit()
    revision, audit_result = data[-2:]
    save(OUTPUT / "checkpoint_audit.json", dict(**audit_result, revision=revision))
    configs = plan(args.stage)
    if args.audit:
        print(json.dumps(audit_result, indent=2))
        print(f"{len(configs)} planned experiments, no models fitted")
        return

    import msvcrt
    with (OUTPUT / "classification.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        checkpoint_dir = OUTPUT / "classification"
        checkpoint_dir.mkdir(exist_ok=True)
        tasks = [(cfg, fold) for cfg in configs for fold in range(1, 6)]
        pending = [(cfg, fold) for cfg, fold in tasks if read_result(cfg, fold, revision) is None]
        done = set(tasks) - set(pending)
        started = perf_counter()
        workers = workers_for(args.workers)
        print(f"{workers} worker(s), {len(done)}/{len(tasks)} folds already checkpointed", flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(data,)) as pool:
                jobs = [pool.submit(run_fold, *task) for task in pending]
                for completed, job in enumerate(as_completed(jobs), 1):
                    done.add(job.result())
                    elapsed = perf_counter() - started
                    finished = sum(all((cfg, fold) in done for fold in range(1, 6)) for cfg in configs)
                    eta = elapsed / completed * (len(pending) - completed)
                    print(f"{finished}/{len(configs)} experiments, {len(done)}/{len(tasks)} folds, "
                          f"elapsed {elapsed:.0f}s, estimated remaining {eta:.0f}s", flush=True)
        save(OUTPUT / f"stage_{args.stage}_complete.json",
             dict(revision=revision, experiments=len(configs), folds=len(tasks), workers=workers,
                  computed_folds=len(pending), elapsed_seconds=perf_counter() - started))


if __name__ == "__main__":
    main()

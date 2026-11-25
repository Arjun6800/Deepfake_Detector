"""Checkpoint-only Phase 4.11A feature preparation and nested classification."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import disk_usage
from time import perf_counter
import warnings

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .phase47_classification import FoldTransform, estimator, score
from .phase48_classify import GRID, SEED, fold_groups, stack
from .phase48_extract import BUFFER, MIN_FREE, atomic, token, windows
from .phase49_classify import audit, fit as historical_fit, objective, predict as historical_predict, read_csv, threshold
from .phase410_features import TEMPORAL, describe, observations, waveform
from .phase410_run import OUTPUT as PHASE410, archive_data, cached, code_revision, progress, summarize_windows

ROOT = Path("outputs/test14")
PHASE49 = Path("outputs/test12")
WINDOW_FIELDS = TEMPORAL[:-2]
HYBRID = ("expression", "original_temporal", "enhanced_temporal", "original_spectral")
ARMS = {
    "expression_only": ("expression",),
    "original_rppg_only": ("original_rppg",),
    "phase49_selective": ("expression", "original_temporal", "original_spectral"),
    "expression_enhanced_temporal": ("expression", "enhanced_temporal"),
    "expression_enhanced_original_spectral": ("expression", "enhanced_temporal", "original_spectral"),
    "expression_original_enhanced_temporal": ("expression", "original_temporal", "enhanced_temporal"),
    "hybrid": HYBRID,
    "enhanced_temporal_original_spectral": ("enhanced_temporal", "original_spectral"),
    "hybrid_redundancy": HYBRID,
    "hybrid_modality_regularization": HYBRID,
}
BASELINES = tuple(ARMS)[:3]
MODES = ("mean", "mean_std", "window_scores")
SHORTLIST = ("phase49_selective", "hybrid")
MODEL_GRIDS = {
    "logistic_l2": [{"C": c} for c in (.1, 1.)],
    "logistic_l1": [{"C": c} for c in (.1, 1.)],
    "elastic_net": [{"C": c, "l1_ratio": ratio} for c, ratio in ((.1, .25), (1., .75))],
    **{name: [{parameter: v} for v in values]
       for name, (parameter, values) in GRID.items() if name != "logistic"},
}


def plan():
    primary = [(arm, mode, "logistic_l2") for arm in ARMS
               for mode in (("mean",) if arm in BASELINES else MODES)]
    return primary + [(arm, "mean", model) for arm in SHORTLIST
                      for model in MODEL_GRIDS if model != "logistic_l2"]


def ensure_space(size=0):
    free = disk_usage(ROOT).free
    if free < MIN_FREE + max(BUFFER, size):
        raise RuntimeError("Phase 4.11A needs a 2 GiB reserve plus write headroom")


def write(path, payload):
    ensure_space(len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, lambda stream: stream.write(payload), reserve=False)


def save(path, value):
    write(path, json.dumps(value, allow_nan=False).encode("utf-8"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def historical_predictions(items, outer):
    identity = ("1", "temporal_spectral", "mean", "logistic")
    keys = ("stage", "arm", "temporal", "classifier")
    rows = [r for r in read_csv(PHASE49 / "video_predictions.csv") if tuple(r[k] for k in keys) == identity]
    expected = {item["video_id"]: (item["group_id"], int(fold), item["label"])
                for item, fold in zip(items, outer, strict=True)}
    actual = {r["clip_id"]: (r["family"], int(r["fold"]), int(r["label"])) for r in rows}
    if len(rows) != 484 or expected != actual or set(outer) != set(range(1, 6)):
        raise ValueError("Phase 4.9 matched development cohort or fold assignments differ")
    return [dict(clip_id=r["clip_id"], family=r["family"], fold=int(r["fold"]), label=int(r["label"]),
                 cohort=r["cohort"], manipulation=r["manipulation"], available_windows=int(r["available_windows"]),
                 score=float(r["score"]), prediction=int(r["prediction"])) for r in rows]


def source_data():
    items, records, schema, fields, outer, multi, _, _, revision, summary = audit()
    indices = sorted(multi)
    items, records, outer = [items[i] for i in indices], [records[i] for i in indices], outer[indices]
    historical = historical_predictions(items, outer)
    prior = read(PHASE410 / "pilot" / "shortlist.json")
    if prior["revision"] != code_revision(revision):
        raise ValueError("Phase 4.10 shortlist is incompatible with audited checkpoint provenance")
    blocks = {
        "expression": fields["expression187"],
        "original_rppg": fields["rppg69"],
        "original_temporal": fields["pos_temporal"],
        "original_spectral": [("pos", name) for name in schema["rppg"]["spectral"]],
        "enhanced_temporal": [("enhanced", name) for name in TEMPORAL],
    }
    if blocks["expression"] + blocks["original_temporal"] + blocks["original_spectral"] != fields["temporal_spectral"]:
        raise ValueError("Selective fusion schema differs from Phase 4.9")
    run_revision = token(prior["revision"] + Path(__file__).read_bytes().hex()
                         + sklearn.__version__ + sha256(json.dumps(historical).encode()).hexdigest())
    return items, records, outer, historical, blocks, prior, summary["phase48_revision"], run_revision


def enhanced_windows(obs, records, variant):
    fps = round(float(obs["fps"]), 12)
    output = []
    for index, (record, (center, grid)) in enumerate(zip(records, windows(obs["times"], fps), strict=True)):
        if (record["window"] != index or record["center"] != center
                or not np.isclose(record["start_s"], grid[0], rtol=0, atol=1e-8)):
            raise ValueError("Phase 4.8 and enhanced window grids are misaligned")
        colors, ids, nuisance, scores, motion = observations(obs, grid, fps)
        regional, wave, provenance = waveform(colors, ids, nuisance, fps, variant)
        output.append(describe(regional, wave, ids, provenance, scores, motion, fps))
    return output


def prepare(data, smoke=False):
    items, records, outer, historical, blocks, prior, phase48_revision, revision = data
    root = ROOT / "smoke" if smoke else ROOT
    indices = [0, next(i for i in range(len(items)) if outer[i] != outer[0])] if smoke else list(range(len(items)))
    started = perf_counter()
    prepared = []
    for done, i in enumerate(indices, 1):
        item = items[i]
        clip = item["video_id"]
        obs, digest = archive_data(item, phase48_revision)
        variant = prior["variant"]
        previous_key = token(prior["revision"] + digest + variant)
        previous = cached(PHASE410 / "features" / variant / f"{token(clip)}.json", previous_key)
        if previous is None:
            raise ValueError(f"Missing compatible Phase 4.10 feature checkpoint for {clip}")
        previous_summary = previous["features"]["temporal"]
        key = token(revision + previous_key + json.dumps(previous_summary, sort_keys=True))
        path = root / "features" / f"{token(clip)}.json"
        value = cached(path, key)
        if value is None:
            rows = enhanced_windows(obs, records[i], variant)
            summary = summarize_windows(rows)["temporal"]
            if not np.allclose(np.array([summary[n] for n in TEMPORAL], dtype=float),
                               np.array([previous_summary[n] for n in TEMPORAL], dtype=float),
                               equal_nan=True, rtol=1e-8, atol=1e-10):
                # Convert serialized nulls to numeric NaNs before comparison.
                raise ValueError(f"Enhanced temporal summary differs for {clip}")
            raw = [[float(r["temporal"][n]) if np.isfinite(r["temporal"][n]) else None
                    for n in WINDOW_FIELDS] for r in rows]
            value = dict(clip_id=clip, starts=[r["start_s"] for r in records[i]],
                         windows=raw, summary=previous_summary)
            save(path, dict(key=key, result=value))
        validate_feature(value, clip, records[i])
        prepared.append(dict(clip_id=clip, key=key, digest=sha256(path.read_bytes()).hexdigest()))
        if done % 10 == 0 or done == len(indices):
            progress("prepare", done, len(indices), started)
    save(root / "prepared.json", dict(revision=revision, clips=prepared, seconds=perf_counter() - started))
    save(root / "feature_schema.json", dict(blocks=blocks, arms=ARMS, aggregations=MODES, variant=prior["variant"],
         window_seconds=8, hop_seconds=4, enhanced_window_fields=WINDOW_FIELDS, missing_value=None,
         video_only_enhanced_fields=TEMPORAL[-2:], redundancy_correlation=.95, near_constant_dominance=.995,
         modality_penalty_multipliers={"expression": [.5, 1., 2.], "rppg": 1.},
         aggregation_details="Mean and population SD use finite windows. SD requires two observations. Optimized arms "
                             "retain partial missing fractions and training-supported missingness flags."))
    return root


def validate_feature(value, clip, records):
    if (value is None or value["clip_id"] != clip or value["starts"] != [r["start_s"] for r in records]
            or np.asarray(value["windows"], dtype=float).shape != (len(records), len(WINDOW_FIELDS))
            or set(value["summary"]) != set(TEMPORAL)
            or np.isinf(np.asarray(value["windows"], dtype=float)).any()
            or np.isinf(np.asarray(list(value["summary"].values()), dtype=float)).any()):
        raise ValueError(f"Incompatible prepared features for {clip}")


def load_prepared(data, root=ROOT, smoke=False):
    items, records, *_, revision = data
    manifest = read(root / "prepared.json")
    indices = {item["video_id"]: i for i, item in enumerate(items)}
    names = [row["clip_id"] for row in manifest["clips"]]
    if (manifest["revision"] != revision or len(names) != len(set(names))
            or (len(names) != 2 if smoke else names != list(indices))):
        raise ValueError("Prepared cohort or revision differs. Run prepare.")
    result = []
    for row in manifest["clips"]:
        clip = row["clip_id"]
        path = root / "features" / f"{token(clip)}.json"
        if sha256(path.read_bytes()).hexdigest() != row["digest"]:
            raise ValueError(f"Prepared checkpoint digest differs for {clip}")
        value = cached(path, row["key"])
        validate_feature(value, clip, records[indices[clip]])
        result.append(value)
    return result, token(json.dumps([r["digest"] for r in manifest["clips"]]) + revision)


def representation(records, enhanced, blocks, arm, mode, historical=False):
    arrays = []
    layout = []
    for block in ARMS[arm]:
        fields = blocks[block]
        enhanced_block = block == "enhanced_temporal"
        raw_fields = fields[:len(WINDOW_FIELDS)] if enhanced_block else fields
        layout.extend((block, name, mode) for _, name in raw_fields)
        if mode == "mean_std":
            layout.extend((block, name, "sd") for _, name in raw_fields)
        if mode != "window_scores" and not historical:
            layout.extend((block, name, "missing_fraction") for _, name in raw_fields)
        if enhanced_block and mode != "window_scores":
            layout.extend((block, name, "video_summary") for name in TEMPORAL[-2:])
    for rows, derived in zip(records, enhanced, strict=True):
        parts = []
        for block in ARMS[arm]:
            enhanced_block = block == "enhanced_temporal"
            raw = np.asarray(derived["windows"] if enhanced_block else
                             [[row[source][name] for source, name in blocks[block]] for row in rows], dtype=float)
            if mode == "window_scores":
                parts.append(raw)
                continue
            observed = np.isfinite(raw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                parts.append(np.nanmean(raw, axis=0, keepdims=True))
                if mode == "mean_std":
                    sd = np.nanstd(raw, axis=0, keepdims=True)
                    sd[:, observed.sum(axis=0) < 2] = np.nan
                    parts.append(sd)
            if not historical:
                parts.append(1 - observed.mean(axis=0, keepdims=True))
            if enhanced_block:
                parts.append(np.array([[derived["summary"][n] for n in TEMPORAL[-2:]]], dtype=float))
        arrays.append(np.hstack(parts))
    return arrays, layout



class HybridTransform:
    """Learn value scaling, missingness indicators and a column mask from fitting rows."""

    def __init__(self, layout, redundancy=False, expression_penalty=1.):
        self.layout = layout
        self.redundancy = redundancy
        self.expression_penalty = expression_penalty

    def fit(self, values):
        valid = np.isfinite(values)
        active = np.flatnonzero(valid.any(axis=0))
        self.missing = np.flatnonzero(valid.any(axis=0) & ~valid.all(axis=0))
        observed = values[:, active]
        medians = np.nanmedian(observed, axis=0)
        filled = np.where(valid[:, active], observed, medians)
        self.centers = filled.mean(axis=0)
        self.scales = filled.std(axis=0)
        spans = np.ptp(filled, axis=0)
        tolerance = 1e-12 * np.maximum(np.max(np.abs(filled), axis=0), 1e-12)
        candidates = np.flatnonzero(spans > tolerance)
        keep = [j for j in candidates if np.unique(filled[:, j], return_counts=True)[1].max()
                / len(filled) < .995]
        self.columns = active[keep]
        self.medians = medians[keep]
        self.centers = self.centers[keep]
        self.scales = self.scales[keep]
        transformed = self.unscaled_blocks(values)
        self.selected = np.arange(transformed.shape[1])
        if self.redundancy and transformed.shape[1] > 1:
            correlations = np.abs(np.corrcoef(transformed, rowvar=False))
            # Prefer better observed values and preserve stable schema order on ties.
            coverage = np.r_[valid[:, self.columns].mean(axis=0), np.ones(len(self.missing))]
            selected = []
            for col in np.argsort(-coverage, kind="stable"):
                if np.all(correlations[col, selected] < .95):
                    selected.append(col)
            self.selected = np.sort(selected)
        groups = [self.layout[j][0] for j in np.r_[self.columns, self.missing]]
        modality = np.array(["expression" if g == "expression" else "rppg" for g in groups], dtype=str)
        modality = modality[self.selected]
        self.multipliers = np.ones(len(modality))
        if self.expression_penalty is not None:
            for name, penalty in (("expression", self.expression_penalty), ("rppg", 1.)):
                mask = modality == name
                if mask.any():
                    # Normalize modality size and induce separate L2 penalties on z-score coefficients.
                    self.multipliers[mask] = 1 / np.sqrt(mask.sum() * penalty)
        return self

    def unscaled_blocks(self, values):
        selected = values[:, self.columns]
        filled = np.where(np.isfinite(selected), selected, self.medians)
        return np.column_stack(((filled - self.centers) / self.scales, ~np.isfinite(values[:, self.missing])))

    def transform(self, values):
        return self.unscaled_blocks(values)[:, self.selected] * self.multipliers


def settings_for(arm, model):
    if arm == "hybrid_modality_regularization":
        return [dict(C=c, expression_penalty=p) for c in (1., 10.) for p in (.5, 1., 2.)]
    return MODEL_GRIDS[model]


def fit_model(arrays, layout, labels, train, config, setting):
    arm, _, classifier = config
    if arm in BASELINES and classifier == "logistic_l2":
        model = historical_fit(arrays, labels, train, "logistic", setting["C"])
        return lambda values, counts: historical_predict(model, values, counts), len(model[1])
    values, counts = stack(arrays, train)
    if arm in BASELINES:
        transform = FoldTransform().fit(values)
    else:
        transform = HybridTransform(layout, arm == "hybrid_redundancy", setting.get("expression_penalty")).fit(values)
    x = transform.transform(values)
    y = np.repeat(labels[train], counts)
    weights = np.repeat(len(y) / (2 * np.bincount(labels[train], minlength=2)[labels[train]] * counts), counts)
    parameters = {k: v for k, v in setting.items() if k != "expression_penalty"}
    modern_penalty = tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 8)
    if classifier == "logistic_l1":
        penalty = {"l1_ratio": 1.} if modern_penalty else {"penalty": "l1"}
        model = LogisticRegression(solver="liblinear", **penalty, random_state=SEED, max_iter=2000)
    elif classifier == "elastic_net":
        penalty = {} if modern_penalty else {"penalty": "elasticnet"}
        model = LogisticRegression(solver="saga", **penalty, random_state=SEED, max_iter=3000, tol=.001)
    elif classifier == "logistic_l2":
        model = estimator("logistic")
    else:
        model = estimator(classifier)
        if classifier == "hist_gradient_boosting":
            model.set_params(max_bins=63)
    if x.shape[1]:
        model.set_params(**parameters).fit(x, y, sample_weight=weights)
    else:
        model = DummyClassifier(strategy="prior").fit(np.zeros((len(y), 1)), y, sample_weight=weights)

    def predict(values, counts):
        matrix = transform.transform(values)
        raw = score(model, matrix if matrix.shape[1] else np.zeros((len(matrix), 1)))
        return np.add.reduceat(raw, np.r_[0, counts.cumsum()[:-1]]) / counts

    return predict, x.shape[1]


def grouped_importance(predict, values, counts, labels, layout, baseline, fold):
    rng = np.random.default_rng(SEED + fold)
    starts = np.r_[0, counts.cumsum()[:-1]]
    observed_auc = roc_auc_score(labels, baseline)
    mapping = []
    for _ in range(5):
        donors = np.arange(len(counts))
        for count in np.unique(counts):
            members = np.flatnonzero(counts == count)
            donors[members] = rng.permutation(members)
        mapping.append(np.concatenate([starts[d] + np.arange(counts[d]) for d in donors]))
    rows = []
    for block in dict.fromkeys(field[0] for field in layout):
        columns = np.array([i for i, field in enumerate(layout) if field[0] == block])
        drops = []
        for donors in mapping:
            permuted = values.copy()
            permuted[:, columns] = values[np.ix_(donors, columns)]
            drops.append(float(observed_auc - roc_auc_score(labels, predict(permuted, counts))))
        rows.append(dict(feature_group=block, drops=drops))
    return rows


def evaluate_fold(items, outer, arrays, layout, config, fold):
    begin = perf_counter()
    labels = np.array([item["label"] for item in items])
    families = np.array([item["group_id"] for item in items])
    train, held = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
    splits = list(fold_groups(items, train, 4, SEED + fold))
    if sorted(i for _, validation in splits for i in validation) != train.tolist():
        raise ValueError("Inner validation must cover outer training videos exactly once")
    for a, b in [(train, held), *splits]:
        if set(families[a]) & set(families[b]) or set(labels[a]) != {0, 1} or set(labels[b]) != {0, 1}:
            raise ValueError("Invalid grouped development split")
    if any(set(a) | set(b) != set(train) for a, b in splits):
        raise ValueError("Inner split differs from the outer training partition")
    best = None
    for setting in settings_for(config[0], config[2]):
        oof = np.full(len(items), np.nan)
        for a, b in splits:
            predict, _ = fit_model(arrays, layout, labels, np.asarray(a), config, setting)
            oof[b] = predict(*stack(arrays, b))
        rank = objective(labels[train], oof[train])
        if best is None or rank > best[0]:
            best = rank, setting, threshold(labels[train], oof[train])
    _, setting, cut = best
    predict, selected = fit_model(arrays, layout, labels, train, config, setting)
    values, counts = stack(arrays, held)
    scores = predict(values, counts)
    rows = [dict(clip_id=items[i]["video_id"], family=items[i]["group_id"],
                 cohort=items[i]["phase48_cohort"], manipulation=items[i]["manipulation_method"],
                 label=int(labels[i]), fold=fold, score=float(s), threshold=cut, prediction=int(s >= cut))
            for i, s in zip(held, scores, strict=True)]
    return dict(config=config, fold=fold, setting=setting, threshold=cut, selected_columns=selected, rows=rows,
                importance=grouped_importance(predict, values, counts, labels[held], layout, scores, fold),
                seconds=perf_counter() - begin)


def checkpoint_path(config, fold):
    return ROOT / "folds" / ("--".join((*config, str(fold))) + ".json")


def classify(data, enhanced, key):
    items, records, outer, _, blocks, *_ = data
    counts_by_clip = {item["video_id"]: len(rows) for item, rows in zip(items, records, strict=True)}
    started = perf_counter()
    done = reused = 0
    configs = plan()
    for config in configs:
        arm, mode, _ = config
        arrays, layout = representation(records, enhanced, blocks, arm, mode, arm in BASELINES)
        for fold in range(1, 6):
            path = checkpoint_path(config, fold)
            result = cached(path, key)
            hit = result is not None
            reused += hit
            if result is None:
                result = evaluate_fold(items, outer, arrays, layout, config, fold)
                for row in result["rows"]:
                    row["available_windows"] = counts_by_clip[row["clip_id"]]
                save(path, dict(key=key, result=result))
            done += 1
            status = "reused" if hit else "fit"
            progress("classify " + "/".join(config) + f" fold {fold} ({status})", done, len(configs) * 5, started)
    return dict(folds=done, reused=reused)


def main():
    from . import phase411a_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "classify", "evaluate", "report", "smoke"))
    stage = parser.parse_args().stage
    ROOT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (ROOT / "phase411a.lock").open("a+b") as lock, threadpool_limits(limits=1):
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        ensure_space()
        began = perf_counter()
        print(f"{stage}: auditing Phase 4.8–4.10 checkpoint provenance", flush=True)
        data = source_data()
        extra = {}
        if stage in ("prepare", "smoke"):
            smoke = stage == "smoke"
            root = prepare(data, smoke)
            features, key = load_prepared(data, root, smoke)
            extra = dict(videos=len(features), feature_revision=key)
        else:
            features, key = load_prepared(data)
            if stage == "classify":
                extra = classify(data, features, key)
            elif stage == "evaluate":
                extra = phase411a_report.evaluate(data, key)
            else:
                phase411a_report.report(key)
        runtime = ROOT / "runtime_summary.json"
        previous = read(runtime) if runtime.exists() else {}
        previous[stage] = dict(seconds=perf_counter() - began, free_bytes=disk_usage(ROOT).free, **extra)
        save(runtime, previous)


if __name__ == "__main__":
    main()



"""Standalone cached extraction and nested evaluation for Phase 4.10."""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import io
from itertools import combinations
import json
from pathlib import Path
from shutil import disk_usage
from time import perf_counter
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from .phase48_classify import GRID, SEED, fold_groups, stack
from .phase48_extract import BUFFER, CHECKPOINTS, MIN_FREE, atomic, token, windows
from .phase49_classify import arrays_for, audit, fit, importance, inner_oof, objective, predict, read_csv, threshold
from .phase49_evaluate import metrics
from .phase410_features import CROSS, SPECTRAL, TEMPORAL, VARIANTS, describe, observations, synthetic_control, waveform

OUTPUT = Path("outputs/test13")
PREVIOUS = Path("outputs/test12")
BASELINES = {"expression187": ("expression187",), "rppg69": ("rppg69",),
             "temporal_spectral": ("temporal_spectral",)}
ARMS = {
    **BASELINES,
    "enhanced_rppg": ("temporal", "spectral"),
    "expression_temporal": ("expression187", "temporal"),
    "expression_spectral": ("expression187", "spectral"),
    "expression_enhanced": ("expression187", "temporal", "spectral"),
    "expression_cross": ("expression187", "cross"),
    "expression_enhanced_cross": ("expression187", "temporal", "spectral", "cross"),
    "nuisance": ("nuisance",), "motion_only": ("motion_only",), "missingness": ("missingness",),
}
FIELDS = dict(temporal=TEMPORAL, spectral=SPECTRAL, cross=CROSS, missingness=("wave_coverage", "joint_coverage"),
              nuisance=("motion", "light_cv", "bits_per_pixel"), motion_only=("motion",))


def space():
    if disk_usage(OUTPUT).free < MIN_FREE + BUFFER:
        raise RuntimeError("Free workspace space must exceed 2 GiB plus the 64 MiB write buffer")


def write(path, payload):
    space()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, lambda handle: handle.write(payload), reserve=False)


def save(path, value):
    write(path, json.dumps(value, allow_nan=False).encode())


def table(path, rows):
    if not rows:
        return
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows(rows)
    write(path, out.getvalue().encode())


def cached(path, key):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["key"] == key:
            return data["result"]
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def checkpoint(path, key, compute):
    result = cached(path, key)
    if result is None:
        result = compute()
        save(path, dict(key=key, result=result))
    return result


def code_revision(previous):
    import scipy
    from . import phase410_features
    digest = sha256(previous.encode())
    for path in (Path(__file__), Path(phase410_features.__file__)):
        digest.update(path.read_bytes())
    digest.update(f"{np.__version__}:{scipy.__version__}".encode())
    return digest.hexdigest()


def verify_previous(fold_by_clip):
    def pooled(name):
        return [r for r in read_csv(PREVIOUS / name) if r["cohort"] == "all" and int(r["videos"]) == 484]
    temporal = pooled("temporal_comparison.csv")
    keys = ("balanced_accuracy", "roc_auc")
    best = max(temporal + pooled("late_fusion_comparison.csv"), key=lambda row: tuple(float(row[k]) for k in keys))
    identity = ("stage", "arm", "temporal", "classifier")
    if tuple(best[k] for k in identity) != ("1", "temporal_spectral", "mean", "logistic"):
        raise ValueError("Unexpected Phase 4.9 winning configuration")
    baseline = next(r for r in temporal if tuple(r[k] for k in identity) == ("1", "expression187", "mean", "logistic"))
    predictions = [r for r in read_csv(PREVIOUS / "video_predictions.csv")
                   if all(r[k] == best[k] for k in identity)]
    if len(predictions) != 484 or {r["clip_id"]: int(r["fold"]) for r in predictions} != fold_by_clip:
        raise ValueError("Phase 4.9 matched predictions differ from audited outer fold membership")
    drops = {r["feature_group"]: float(r["mean_auc_drop"])
             for r in read_csv(PREVIOUS / "feature_importance_summary.csv") if all(r[k] == best[k] for k in identity)}
    columns = (*keys, "accuracy", "precision", "recall", "f1", "pr_auc_ap")
    return dict(winner={k: best[k] for k in (*identity, *columns)}, baseline={k: baseline[k] for k in columns},
                importance=drops)


def pilot_indices(items, records, outer, multi):
    # Choose two or three matched families per outer fold using support only.
    chosen = []
    for fold in sorted(set(outer)):
        families = defaultdict(list)
        for i in sorted(multi):
            if outer[i] == fold:
                families[items[i]["group_id"]].append(i)
        valid = [ix for ix in families.values() if {items[i]["label"] for i in ix} == {0, 1}]
        target = 24 if fold in (1, 2) else 16
        candidates = [sides for size in (2, 3) for sides in combinations(valid, size)]
        best = min(candidates, key=lambda sides: (abs(sum(map(len, sides)) - target),
                   sum(len(records[i]) for side in sides for i in side),
                   tuple(items[side[0]]["group_id"] for side in sides)))
        chosen.extend(i for side in best for i in side)
    return sorted(chosen)


def average(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=0)


def summarize_windows(rows):
    def spread(block, name, logarithm=False):
        values = np.array([row[block][name] for row in rows])
        values = values[np.isfinite(values)]
        if logarithm:
            values = np.log(values + 1e-12)
        return values.std() if len(values) >= 2 else np.nan

    extra = dict(inter_window_acf_sd=spread("temporal", "acf_peak"),
                 inter_window_activity_log_sd=spread("temporal", "hjorth_activity", True),
                 inter_window_centroid_sd=spread("spectral", "centroid_hz"),
                 inter_window_activity_r_sd=spread("cross", "amplitude_activity_r"))
    result = {}
    for group in ("temporal", "spectral", "cross", "missingness"):
        result[group] = {}
        for name in FIELDS[group]:
            value = extra[name] if name in extra else average([row[group][name] for row in rows])
            result[group][name] = float(value) if np.isfinite(value) else None
    return result


def archive_data(item, phase48_revision):
    path = CHECKPOINTS / f"{token(item['video_id'])}.npz"
    payload = path.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        if str(source["key"]) != token(phase48_revision + item["sha256"]):
            raise ValueError(f"Observation provenance mismatch for {item['video_id']}")
        data = dict(source)
    if len(data["scores"]) != len(data["times"]):
        raise ValueError(f"Missing synchronized expression observations for {item['video_id']}")
    return data, sha256(payload).hexdigest()


def video_features(data, variant):
    fps = round(float(data["fps"]), 12)
    results = []
    for _, grid in windows(data["times"], fps):
        colors, ids, nuisance, scores, motion = observations(data, grid, fps)
        regional, wave, provenance = waveform(colors, ids, nuisance, fps, variant)
        results.append(describe(regional, wave, ids, provenance, scores, motion, fps))
    return summarize_windows(results)


def progress(stage, completed, total, started):
    elapsed = perf_counter() - started
    print(f"{stage} {completed}/{total} elapsed {elapsed:.1f}s ETA {elapsed * (total-completed) / completed:.1f}s",
          flush=True)


def extract(root, data, indices, variants):
    items, revision, phase48_revision = data["items"], data["revision"], data["phase48_revision"]
    started = perf_counter()
    result = {variant: {} for variant in variants}
    for done, i in enumerate(indices, 1):
        item = items[i]
        obs, digest = archive_data(item, phase48_revision)
        for variant in variants:
            path = root / "features" / variant / f"{token(item['video_id'])}.json"
            key = token(revision + digest + variant)
            def compute():
                if root == OUTPUT:
                    pilot_value = cached(root / "pilot" / "features" / variant / path.name, key)
                    if pilot_value is not None:
                        return pilot_value
                begin = perf_counter()
                features = video_features(obs, variant)
                return dict(features=features, seconds=perf_counter() - begin)
            result[variant][i] = checkpoint(path, key, compute)
        progress("features", done, len(indices), started)
    return result


def read_features(root, data, indices, variants):
    result = {}
    for variant in variants:
        result[variant] = {}
        for i in indices:
            item = data["items"][i]
            path = root / "features" / variant / f"{token(item['video_id'])}.json"
            archive = CHECKPOINTS / f"{token(item['video_id'])}.npz"
            key = token(data["revision"] + sha256(archive.read_bytes()).hexdigest() + variant)
            value = cached(path, key)
            if value is None:
                raise ValueError(f"Run extraction first. Missing current checkpoint {path}")
            result[variant][i] = value
    return result


def matrix(data, indices, extracted, blocks):
    arrays, fields = {}, []
    for block in dict.fromkeys(blocks):
        if block in BASELINES:
            old = data["old_fields"][block]
            arrays[block] = [x[0] for x in arrays_for([data["records"][i] for i in indices], old, {}, "mean")]
            fields.extend(old)
        elif block in ("nuisance", "motion_only"):
            arrays[block] = [average(np.array([[r["audit"][k] for k in FIELDS[block]] for r in data["records"][i]],
                                             dtype=float)) for i in indices]
            fields.extend(("control", f"{block}__{k}") for k in FIELDS[block])
        else:
            arrays[block] = [[extracted[i]["features"][block][k] for k in FIELDS[block]] for i in indices]
            fields.extend(("enhanced", f"{block}__{k}") for k in FIELDS[block])
    design = [np.hstack(row).astype(float)[None, :] for row in zip(*(arrays[b] for b in blocks), strict=True)]
    schema = dict(data["schema"], rppg={**data["schema"]["rppg"], **{
        f"phase410_{b}": [f"{b}__{k}" for k in keys] for b, keys in FIELDS.items()}})
    return design, fields, schema


def nested_fold(items, arrays, fields, schema, outer, fold, classifier):
    y = np.array([item["label"] for item in items])
    families = np.array([item["group_id"] for item in items])
    train, held = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
    splits = list(fold_groups(items, train, 4, SEED + fold))
    for a, b in [(train, held), *splits]:
        if set(families[a]) & set(families[b]) or set(y[a]) != {0, 1} or set(y[b]) != {0, 1}:
            raise ValueError("Invalid source-family split")
    parameter, grid = GRID[classifier]
    setting, oof = max(((v, inner_oof(arrays, y, train, splits, classifier, v)) for v in grid),
                       key=lambda pair: objective(y[train], pair[1]))
    model = fit(arrays, y, train, classifier, setting)
    cut = threshold(y[train], oof)
    values, counts = stack(arrays, held)
    scores = predict(model, values, counts)
    drops = importance(model, values, counts, y[held], fields, schema, scores, fold)
    rows = []
    for i, value in zip(held, scores, strict=True):
        item = items[i]
        rows.append(dict(clip_id=item["video_id"], family=item["group_id"], cohort=item["phase48_cohort"],
                         manipulation=item["manipulation_method"], label=item["label"], fold=fold,
                         threshold=cut, score=float(value), prediction=int(value >= cut)))
    return dict(rows=rows, importance=drops, setting={parameter: setting}, threshold=cut)


def classify(root, data, indices, configs, extracted):
    items = [data["items"][i] for i in indices]
    outer = data["outer"][indices]
    all_results = []
    started = perf_counter()
    for arm, variant, classifier in configs:
        design, fields, schema = matrix(data, indices, extracted.get(variant, {}), ARMS[arm])
        design_hash = sha256(np.vstack(design).tobytes()).hexdigest()
        key = token(data["revision"] + design_hash + repr(indices) + repr((arm, variant, classifier)))
        for fold in sorted(set(outer)):
            def compute():
                begin = perf_counter()
                result = nested_fold(items, design, fields, schema, outer, int(fold), classifier)
                return dict(**result, seconds=perf_counter() - begin)
            path = root / "folds" / f"{arm}--{variant}--{classifier}--{fold}.json"
            result = checkpoint(path, key, compute)
            all_results.append(dict(**result, arm=arm, variant=variant, classifier=classifier, fold=int(fold)))
            progress("nested folds", len(all_results), len(configs) * len(set(outer)), started)
    return all_results


def choose_variant(results, controls, extracted):
    rows = []
    for variant, control in controls.items():
        folds = [r for r in results if r["variant"] == variant]
        pooled = [row for fold in folds for row in fold["rows"]]
        features = list(extracted[variant].values())
        coverage = [record["features"]["missingness"]["wave_coverage"] for record in features]
        stats = metrics(pooled)
        rows.append(dict(variant=variant, **control, **stats,
                         median_wave_coverage=float(np.median(coverage)),
                         median_feature_seconds=float(np.median([r["seconds"] for r in features]))))
    reference = next(r for r in rows if r["variant"] == "pos_reference")
    for row in rows:
        row["eligible"] = row is reference or (
            row["clean_pulse_abs_r"] >= max(.7, reference["clean_pulse_abs_r"] - .15)
            and row["contaminated_pulse_abs_r"] >= reference["contaminated_pulse_abs_r"] - .10
            and row["contamination_motion_abs_r"] <= reference["contamination_motion_abs_r"] + .15
            and row["artifact_only_motion_abs_r"] <= reference["artifact_only_motion_abs_r"] + .05
            and row["artifact_only_pulse_abs_r"] <= .15
            and row["median_wave_coverage"] >= reference["median_wave_coverage"] - .15
            and row["median_feature_seconds"] <= max(1., 4 * reference["median_feature_seconds"]))
    selected = max((r for r in rows if r["eligible"]), key=lambda r: (r["balanced_accuracy"], r["roc_auc"]))
    return selected["variant"], rows


def resources():
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    *[(name, ctypes.c_uint64) for name in ("total", "available", "page", "page_free",
                                                          "virtual", "virtual_free", "extended")]]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ("peak", "working", "peak_pool", "pool",
                                                           "peak_nonpaged", "nonpaged", "page", "peak_page")]]
    memory, process = Memory(), ProcessMemory()
    memory.length, process.size = ctypes.sizeof(memory), ctypes.sizeof(process)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    memory_ok = kernel.GlobalMemoryStatusEx(ctypes.byref(memory))
    process_ok = kernel.K32GetProcessMemoryInfo(wintypes.HANDLE(kernel.GetCurrentProcess()),
                                               ctypes.byref(process), process.size)
    return dict(available_physical_bytes=memory.available if memory_ok else None,
                total_physical_bytes=memory.total if memory_ok else None,
                process_peak_working_set_bytes=process.peak if process_ok else None,
                free_disk_bytes=disk_usage(OUTPUT).free, workers=1, numerical_threads=1)


def full_configs(variant):
    primary = [(arm, "original" if arm in BASELINES else variant, "logistic") for arm in ARMS]
    secondary = [(arm, variant, model) for arm in ("expression_enhanced", "expression_enhanced_cross")
                 for model in GRID if model != "logistic"]
    return primary + secondary


def main():
    import msvcrt
    from . import phase410_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "pilot", "pilot-eval", "extract", "classify", "report", "smoke"))
    stage = parser.parse_args().stage
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "phase410.lock").open("a+b") as lock, threadpool_limits(limits=1):
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        begin = perf_counter()
        before = resources()
        space()
        items, records, schema, fields, outer, multi, _, _, revision, summary = audit()
        prior = verify_previous({items[i]["video_id"]: int(outer[i]) for i in multi})
        data = dict(items=items, records=records, schema=schema, old_fields=fields, outer=outer,
                    revision=code_revision(revision), phase48_revision=summary["phase48_revision"])
        indices = pilot_indices(items, records, outer, multi)
        pilot = OUTPUT / "pilot"
        controls = {v: synthetic_control(v) for v in VARIANTS}
        save(OUTPUT / "feature_schema.json", dict(version="phase410_v1", features=FIELDS,
             historical_fields=fields, arms=ARMS, variants=VARIANTS, missing_value=None,
             window_seconds=8, hop_seconds=4, temporal_min_seconds=2, spectral_min_seconds=4,
             cross_modal_min_seconds=3, units=dict(hjorth_activity="projection units squared",
                                                  hjorth_mobility="inverse seconds",
                                                  lag_s="seconds, positive when waveform follows expression"),
             support="Longest contiguous observed run with fixed ROI provenance per feature",
             expression="Frame-level MediaPipe blendshape proxies"))
        if stage == "audit":
            save(OUTPUT / "historical_verification.json", dict(summary=summary, results=prior,
                 pilot_clips=[items[i]["video_id"] for i in indices]))
            print(f"Verified {len(multi)} matched training videos and {len(indices)} pilot videos.")
        elif stage == "smoke":
            subset = [indices[0], next(i for i in indices if items[i]["phase48_cohort"] == "retained")]
            if subset[0] == subset[1]:
                subset[1] = next(i for i in indices if items[i]["phase48_cohort"] == "new")
            root = OUTPUT / "smoke"
            extracted = extract(root, data, subset, VARIANTS)
            assert read_features(root, data, subset, VARIANTS) == extracted
            phase410_report.diagnostics(root, data, subset, VARIANTS)
            save(root / "controls.json", controls)
        elif stage == "pilot":
            extract(pilot, data, indices, VARIANTS)
            phase410_report.diagnostics(OUTPUT, data, indices, VARIANTS)
        elif stage == "pilot-eval":
            extracted = read_features(pilot, data, indices, VARIANTS)
            configs = [("expression_enhanced", v, "logistic") for v in VARIANTS]
            results = classify(pilot, data, indices, configs, extracted)
            variant, rows = choose_variant(results, controls, extracted)
            table(OUTPUT / "signal_enhancement_comparison.csv", rows)
            save(pilot / "shortlist.json", dict(revision=data["revision"], variant=variant, videos=len(indices),
                                               selection="Exploratory overlapping development pilot"))
        else:
            shortlist = json.loads((pilot / "shortlist.json").read_text(encoding="utf-8"))
            if shortlist["revision"] != data["revision"]:
                raise ValueError("Pilot shortlist is stale. Rerun pilot and pilot-eval.")
            variant = shortlist["variant"]
            indices = sorted(multi)
            configs = full_configs(variant)
            if stage == "extract":
                extract(OUTPUT, data, indices, (variant,))
            else:
                extracted = read_features(OUTPUT, data, indices, (variant,))
                feature_revision = token(json.dumps(extracted[variant], sort_keys=True))
                if stage == "classify":
                    results = classify(OUTPUT, data, indices, configs, extracted)
                    save(OUTPUT / "classification.json", dict(revision=data["revision"], variant=variant, feature_revision=feature_revision,
                                                                  folds=results))
                else:
                    run = json.loads((OUTPUT / "classification.json").read_text(encoding="utf-8"))
                    if (run["revision"], run["variant"], run["feature_revision"]) != (
                            data["revision"], variant, feature_revision):
                        raise ValueError("Classification is stale. Rerun classify.")
                    phase410_report.report(OUTPUT, run["folds"], extracted[variant], prior)
        path = OUTPUT / "runtime_summary.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing[stage] = dict(seconds=perf_counter() - begin, initial=before, final=resources())
        save(path, existing)


if __name__ == "__main__":
    main()




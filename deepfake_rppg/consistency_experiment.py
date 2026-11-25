"""Evaluate an experimental rPPG schema on the cached C23 training cohort."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import signal

from .batch import FEATURE_COUNTS, roi_runs
from .consistency import PSD_EDGES, REGION_NAMES, SCHEMA, aggregate, correlation, describe, reduce_nuisance, robust_trace
from .quality_algorithms import ALGORITHMS, project
from .quality_experiment import write_table
from .window_features import rppg_features, snap_rgb

CAPTURE = Path("outputs/test8/training")
OUTPUT = Path("outputs/test9")
CATEGORIES = ("signal", "spatial", "temporal", "spectral", "reliability")


def choose_regions(base, alternative, groups, other_groups, eligible):
    """Choose a single mask per ROI on shared-frame discontinuity and support."""
    rgb, codes = base.copy(), groups.copy()
    picks = []
    for i in np.flatnonzero(eligible):
        a, b = base[:, i], alternative[:, i]
        va, vb = np.isfinite(a).all(axis=1), np.isfinite(b).all(axis=1)
        shared = va & vb
        adjacent = (shared[1:] & shared[:-1] & (np.diff(groups[:, i]) == 0)
                    & (np.diff(other_groups[:, i]) == 0))
        if vb.mean() < va.mean() - .02 or adjacent.sum() < 20:
            continue
        da, db = [np.linalg.norm(np.diff(x, axis=0)[adjacent], axis=1) for x in (a, b)]
        if np.quantile(db, .9) < .95 * np.quantile(da, .9):
            rgb[:, i], codes[:, i] = b, other_groups[:, i]
            picks.append(i)
    return rgb, codes, picks


def aligned_window(data):
    times, fps = data["times"], round(float(data["fps"]), 12)
    grid = np.arange(int(np.floor(times[-1] * fps + 1e-8)) + 1) / fps
    count = round(8 * fps)
    if len(grid) < count:
        raise ValueError("Expected eight seconds in every selected training clip")
    start = (len(grid) - count) // 2
    window = grid[start:start + count]
    frames = (times >= window[0] - 1e-8) & (times < window[-1] + 1 / fps - 1e-8)

    def snap(values, groups):
        return snap_rgb(times[frames], values[frames], groups[frames], window, fps)

    geometry = data["geometry"]
    widths = geometry[:, 2]
    scale = np.median(widths[np.isfinite(widths) & (widths > 0)]) if np.any(widths > 0) else np.nan
    center = geometry[:, :2] + geometry[:, 2:] / 2
    proxy = np.column_stack([center / scale, widths / scale, data["nose"], data["light"]])
    nuisance = snap(proxy[:, None], np.zeros((len(times), 1), dtype=int))[0][:, 0]
    areas = data["area"][frames].sum(axis=0)
    exposure, retention = [np.divide(data[key][frames].sum(axis=0), areas, out=np.zeros(3), where=areas > 0)
                           for key in ("valid_area", "skin_area")]
    candidates = {kind: snap(data[kind], roi_runs(data[codes])) for kind, codes in
                  (("baseline", "base_codes"), ("skin", "skin_codes"), ("rigid", "rigid_codes"))}
    return fps, window, frames, candidates, nuisance, exposure, retention


def extract(method, colors, groups, fps, nuisance, exposure, robust=False, motion=False, weighted=False):
    projected = np.column_stack([project(method, colors[:, i], fps, groups[:, i]) for i in range(3)])
    raw = projected
    if motion:
        raw = np.column_stack([reduce_nuisance(raw[:, i], nuisance, groups[:, i], fps) for i in range(3)])
    wave = np.column_stack([robust_trace(raw[:, i], groups[:, i], fps, robust) for i in range(3)])
    weights = np.isfinite(wave).mean(axis=0) * exposure if weighted else np.ones(3)
    combined, ids = aggregate(wave, groups, weights)
    raw = np.column_stack((raw, combined))
    wave = np.column_stack((wave, combined))
    groups = np.column_stack((groups, ids))
    return (*describe(raw, wave, groups, fps), raw, wave, groups, projected)


def compare(item, data, rows, baselines, archives):
    fps, grid, frames, channels, nuisance, exposure, retention = aligned_window(data)
    base, groups = channels["baseline"]
    skin, skin_ids, skin_picks = choose_regions(base, channels["skin"][0], groups, channels["skin"][1], retention >= .7)
    rigid, rigid_ids, rigid_picks = choose_regions(base, channels["rigid"][0], groups, channels["rigid"][1], np.ones(3))
    integrated, integrated_ids = rigid.copy(), rigid_ids.copy()
    integrated[:, skin_picks], integrated_ids[:, skin_picks] = skin[:, skin_picks], skin_ids[:, skin_picks]
    identity = dict(clip_id=item["video_id"], split_group=item["group_id"], partition=item["partition"],
                    window_index=0, window_start_s=float(grid[0]), window_end_s_exclusive=float(grid[0] + 8))
    native = np.flatnonzero(frames)
    keys = dict(**identity, first_frame_index=int(native[0]), last_frame_index=int(native[-1]),
                decoded_frames=len(native), rppg_grid_samples=len(grid))
    baseline = rppg_features(data["baseline"][frames], base, groups, fps)
    if len(baseline) != FEATURE_COUNTS["rppg"]:
        raise ValueError("The preserved baseline no longer has 69 features")
    baselines.append(dict(**keys, **baseline))
    variants = [(name, "reference", base, groups, {}) for name in ALGORITHMS] + [
        ("POS", "adaptive_skin", skin, skin_ids, {}),
        ("POS", "adaptive_tracking", rigid, rigid_ids, {}),
        ("POS", "motion_reduction", base, groups, {"motion": True}),
        ("POS", "robust_preprocessing", base, groups, {"robust": True}),
        ("POS", "weighted_aggregation", base, groups, {"weighted": True}),
        ("POS", "combined", integrated, integrated_ids, {"motion": True, "robust": True, "weighted": True}),
    ]
    arrays = {"time_s": grid}
    for method, variant, colors, ids, options in variants:
        start = perf_counter()
        features, spectra, raw, wave, provenance, projected = extract(
            method, colors, ids, fps, nuisance, exposure, **options)
        elapsed = perf_counter() - start
        rows.append(dict(**identity, category="real" if item["label"] == 0 else "manipulated",
                         manipulation=item["manipulation_method"], schema=SCHEMA, method=method, variant=variant,
                         seconds=elapsed, adaptive_skin_regions=len(skin_picks) if variant in ("adaptive_skin", "combined") else 0,
                         rigid_regions=len(set(rigid_picks) - set(skin_picks)) if variant == "combined"
                         else len(rigid_picks) if variant == "adaptive_tracking" else 0, **features))
        prefix = f"{method}__{variant}"
        arrays.update({f"{prefix}__projected": projected, f"{prefix}__raw": raw,
                       f"{prefix}__wave": wave, f"{prefix}__run_ids": provenance})
        for name, (hz, psd) in spectra.items():
            arrays.update({f"{prefix}__{name}__hz": hz, f"{prefix}__{name}__psd": psd})
    path = archives / f"{sha256(item['video_id'].encode()).hexdigest()}.npz"
    np.savez_compressed(path, **arrays)
    return dict(**keys, archive=str(path.relative_to(OUTPUT)))


def summarize(rows, keys):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["category"], row["method"], row["variant"]].append(row)
    summaries = []
    for (category, method, variant), subset in buckets.items():
        def mean(fields):
            values = np.array([row[key] for row in subset for key in fields])
            return np.nanmean(values) if np.isfinite(values).any() else np.nan

        regional = lambda metric: [f"reliability__{name}__{metric}" for name in REGION_NAMES[:3]]
        summary = dict(category=category, method=method, variant=variant, videos=len(subset),
                       regional_waveform_coverage=mean(regional("waveform_coverage")),
                       legacy_processed_coverage=mean(regional("legacy_processed_coverage")),
                       aggregate_waveform_coverage=mean(["reliability__aggregate__waveform_coverage"]),
                       regional_gate_pass_fraction=mean(regional("legacy_gate")),
                       retained_nonpassing_waveforms=sum(
                           row[gate] == 0 and row[coverage] > 0 for row in subset
                           for gate, coverage in zip(regional("legacy_gate"), regional("waveform_coverage"), strict=True)),
                       feature_availability=np.isfinite([[row[key] for key in keys] for row in subset]).mean(),
                       mean_ms=1000 * mean(["seconds"]), skin_selected=sum(row["adaptive_skin_regions"] for row in subset),
                       rigid_selected=sum(row["rigid_regions"] for row in subset))
        for metric in ("correlation", "coherence", "phase_consistency", "spectral_disagreement"):
            summary[metric] = mean([key for key in keys if key.startswith("spatial__") and key.endswith(f"__{metric}")])
        summaries.append(summary)
    return summaries


def synthetic():
    rows = []
    fps = 30
    t = np.arange(8 * fps) / fps
    pulse = np.sin(2 * np.pi * 1.25 * t)
    clean = signal.detrend(pulse)
    groups = np.ones(len(t), dtype=int)
    for seed in range(12):
        rng = np.random.default_rng(seed)
        for case, frequency, strength in (("clean", 1.75, 0), ("distinct_motion", 1.75, 2),
                                           ("overlapping_motion", 1.25, 2), ("artifact_only", 1.75, 2)):
            phase = 0 if case == "overlapping_motion" else rng.uniform(0, 2 * np.pi)
            motion = np.sin(2 * np.pi * frequency * t + phase)
            observed = (pulse if case != "artifact_only" else 0) + strength * motion + rng.normal(0, .15, len(t))
            for variant in ("reference", "motion_reduction"):
                reduced = reduce_nuisance(observed, motion[:, None], groups, fps) if variant != "reference" else observed
                wave = robust_trace(reduced, groups, fps)
                from .consistency import one_trace
                metrics, _, _ = one_trace(reduced, wave, groups, fps)
                rows.append(dict(seed=seed, case=case, variant=variant,
                                 pulse_correlation=correlation(wave, pulse) if case != "artifact_only" else np.nan,
                                 pulse_projection_amplitude=np.dot(wave, clean) / np.dot(clean, clean),
                                 relative_residual_rms=wave.std() / signal.detrend(observed).std(),
                                 legacy_gate=metrics["reliability"]["legacy_gate"]))
    return rows


def main():
    OUTPUT.mkdir(exist_ok=True)
    archives = OUTPUT / "waveforms"
    archives.mkdir(exist_ok=True)
    items = json.loads((CAPTURE / "selection.json").read_text(encoding="utf-8"))
    settings = json.loads((CAPTURE / "capture_settings.json").read_text(encoding="utf-8"))
    if len(items) != 96 or any(item["partition"] != "train" for item in items):
        raise ValueError("Expected the fixed 96-video training cohort")
    rows, baseline, index = [], [], []
    for item in items:
        token = sha256(item["video_id"].encode()).hexdigest()
        with np.load(CAPTURE / "observations" / f"{token}.npz", allow_pickle=False) as capture:
            if str(capture["key"]) != sha256((item["sha256"] + settings["revision"]).encode()).hexdigest():
                raise ValueError(f"Capture provenance mismatch for {item['video_id']}")
            index.append(compare(item, dict(capture), rows, baseline, archives))
    keys = [key for key in rows[0] if key.split("__")[0] in CATEGORIES]
    schema = dict(version=SCHEMA, baseline_counts=FEATURE_COUNTS, expanded_count=len(keys),
                  roi_order=REGION_NAMES, relative_psd_edges_hz=PSD_EDGES,
                  features={category: [key for key in keys if key.startswith(category + "__")] for category in CATEGORIES})
    (OUTPUT / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    write_table(OUTPUT, "baseline_rppg69", baseline)
    write_table(OUTPUT, SCHEMA, rows)
    write_table(OUTPUT, "cohort_comparison", summarize(rows, keys))
    write_table(OUTPUT, "waveform_index", index)
    write_table(OUTPUT, "synthetic_artifact_check", synthetic())
    print(f"{len(items)} videos, {len(rows)} experimental rows, {len(keys)} expanded features")


if __name__ == "__main__":
    main()

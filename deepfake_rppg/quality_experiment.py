"""Run the training-only Phase 4.5 comparisons on captured RGB observations."""

import argparse
import csv
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy import signal, stats

from .batch import roi_runs
from .face import REGIONS
from .quality_algorithms import ALGORITHMS, project
from .signal import BAND_HZ, MIN_QUALITY, resample_observations
from .window_features import longest_run, peak_and_power_ratio, snap_rgb, trace_features

OUTPUT = Path("outputs/test8")
DURATIONS = (8, 12, 16, 20, 30)
METRICS = ("spectral_concentration", "inband_to_outband_db", "processed_coverage",
           "quality_pass", "cross_roi_delta_hz", "cross_roi_abs_correlation",
           "pulse_available", "seconds", "motion_aware_pass", "raw_coverage")


def write_table(output, name, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def score(method, rgb, groups, fps):
    traces, waves, rows = [], [], []
    for index in range(rgb.shape[1]):
        start = perf_counter()
        raw = project(method, rgb[:, index], fps, groups[:, index])
        row, waveform = trace_features(raw, fps, groups[:, index])
        row["seconds"] = perf_counter() - start
        row["pulse_available"] = int(np.isfinite(row["pulse_bpm_estimate"]))
        rows.append(row)
        traces.append(raw)
        waves.append(waveform)
    for i, j in combinations(range(len(rows)), 2):
        joint = np.isfinite(waves[i]) & np.isfinite(waves[j])
        changes = np.any(np.diff(groups[:, [i, j]], axis=0), axis=1)
        run = longest_run(joint, np.r_[0, changes.cumsum()])
        a, b = [peak_and_power_ratio(traces[k][run], fps)[0] for k in (i, j)]
        if np.isfinite(a + b):
            rows[i].setdefault("pair_deltas", []).append(abs(a - b))
            rows[j].setdefault("pair_deltas", []).append(abs(a - b))
            x, y = waves[i][run], waves[j][run]
            if x.std() > 1e-8 and y.std() > 1e-8:
                corr = abs(np.corrcoef(x, y)[0, 1])
                for k in (i, j):
                    rows[k].setdefault("pair_correlations", []).append(corr)
    for row in rows:
        for key, destination in (("pair_deltas", "cross_roi_delta_hz"),
                                 ("pair_correlations", "cross_roi_abs_correlation")):
            values = row.pop(key, [])
            row[destination] = np.mean(values) if values else np.nan
    return rows, traces, waves


def synthetic(output):
    rows, examples = [], {}
    fps = 30
    t = np.arange(600) / fps
    for scenario in ("clean", "drift", "in_band_motion", "noise", "gaps", "no_pulse", "artifact_only"):
        for frequency in (0.9, 1.25, 2.1):
            for seed in range(5):
                rng = np.random.default_rng(seed)
                pulse = np.sin(2 * np.pi * frequency * t)
                tone = np.array([155., 111., 84.])
                rgb = tone + rng.normal(0, .035, (len(t), 3))
                has_pulse = scenario not in ("no_pulse", "artifact_only")
                if has_pulse:
                    rgb += pulse[:, None] * [0.45, 1.2, 0.25]
                if scenario in ("drift", "in_band_motion"):
                    rgb += .045 * np.sin(2 * np.pi * .22 * t[:, None]) * tone
                if scenario in ("in_band_motion", "artifact_only"):
                    rgb += np.sin(2 * np.pi * 1.65 * t[:, None]) * [2, 3.5, 2.7]
                if scenario == "noise":
                    rgb += rng.normal(0, 1.4, rgb.shape)
                if scenario == "gaps":
                    rgb[220:370] = np.nan
                groups = np.ones((len(t), 1), dtype=int)
                for method in ALGORITHMS:
                    scored, raw, filtered = score(method, rgb[:, None], groups, fps)
                    row = scored[0]
                    estimate = row["pulse_bpm_estimate"]
                    error = abs(estimate - 60 * frequency) if has_pulse else np.nan
                    # Known-frequency spectral SNR using a fixed physical bandwidth and full finite run.
                    run = longest_run(np.isfinite(raw[0]))
                    values = raw[0][run]
                    snr = np.nan
                    if len(values) >= 4 * fps and has_pulse:
                        hz, power = signal.periodogram(signal.detrend(values), fs=fps, window="hann")
                        near = (hz > 0) & (abs(hz - frequency) <= .25)
                        noise_power = power[(hz > 0) & ~near].sum()
                        if noise_power > 0 and power[near].sum() > 0:
                            snr = 10 * np.log10(power[near].sum() / noise_power)
                    rows.append(dict(scenario=scenario, seed=seed, method=method,
                                     reference_hz=frequency if has_pulse else np.nan,
                                     error_bpm=error, known_frequency_snr_db=snr, **row))
                    if scenario == "in_band_motion" and frequency == 1.25 and seed == 0:
                        examples[method] = (raw[0], filtered[0])
    write_table(output, "synthetic", rows)
    plots(output, "synthetic", t, examples, 1.25, "Synthetic 75 bpm pulse with a 99 bpm motion artifact")


def compression_phantom(output):
    """Compare identical generated RGB pixels before and after local video encoding."""
    import cv2

    fps, frequency, size = 30, 1.25, (128, 96)
    rng = np.random.default_rng(451)
    texture = rng.normal(0, 5, (*size[::-1], 3))
    y, x = np.indices(size[::-1])
    texture += ((x // 5 + y // 5) % 2)[:, :, None] * 14
    tone = np.array([152, 104, 80]) + texture
    t = np.arange(360) / fps
    frames = {amplitude: np.empty((len(t), *size[::-1], 3), dtype=np.uint8)
              for amplitude in (.35, 1.2)}
    for amplitude, pixels in frames.items():
        for i, time in enumerate(t):
            pulse = amplitude * np.sin(2 * np.pi * frequency * time)
            illumination = .8 * np.sin(2 * np.pi * .25 * time)
            rgb = tone + pulse * np.array([.35, 1, .2]) + illumination
            pixels[i] = np.clip(np.rint(rgb[:, :, ::-1]), 0, 255).astype(np.uint8)

    def rgb_trace(pixels):
        return pixels[:, 16:-16, 16:-16, ::-1].mean(axis=(1, 2))

    rows = []
    with TemporaryDirectory(dir=output) as directory:
        for amplitude, pixels in frames.items():
            versions = {"uint8_reference": (rgb_trace(pixels), None, None)}
            for fourcc, extension in (("FFV1", "avi"), ("mp4v", "mp4"), ("avc1", "mp4")):
                path = Path(directory) / f"{fourcc}.{extension}"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
                if fourcc == "avc1" and not writer.isOpened():
                    writer.release()
                    writer = cv2.VideoWriter(str(path), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*fourcc), fps, size)
                try:
                    if not writer.isOpened():
                        rows.append(dict(amplitude_rgb_counts=amplitude, codec=fourcc, status="encoder_unavailable"))
                        continue
                    backend = writer.getBackendName()
                    for frame in pixels:
                        writer.write(frame)
                finally:
                    writer.release()
                capture = cv2.VideoCapture(str(path))
                decoded = []
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        decoded.append(frame[16:-16, 16:-16, ::-1].mean(axis=(0, 1)))
                finally:
                    capture.release()
                if len(decoded) == len(t):
                    versions[fourcc] = (np.asarray(decoded), backend, path.stat().st_size)
                else:
                    rows.append(dict(amplitude_rgb_counts=amplitude, codec=fourcc, status="decode_count_mismatch"))
            for codec, (colors, backend, file_bytes) in versions.items():
                for method in ALGORITHMS:
                    scored, _, _ = score(method, colors[:, None], np.ones((len(t), 1), dtype=int), fps)
                    rows.append(dict(amplitude_rgb_counts=amplitude, codec=codec, status="scored", method=method,
                                     reference_hz=frequency, encoder_backend=backend, encoded_bytes=file_bytes,
                                     rgb_rmse=np.sqrt(np.mean((colors - versions["uint8_reference"][0]) ** 2)),
                                     **scored[0]))
    write_table(output, "compression_phantom", rows)


def plots(output, prefix, t, examples, reference=None, subject=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 2, figsize=(11, 1.8 * len(examples)), squeeze=False)
    if subject:
        fig.suptitle(subject)
    for (name, (raw, waveform)), (time_axis, psd_axis) in zip(examples.items(), axes, strict=True):
        time_axis.plot(t, waveform, linewidth=.8)
        time_axis.set_ylabel(name)
        run = longest_run(np.isfinite(raw))
        if run.stop > run.start + 10:
            fps = 1 / np.median(np.diff(t))
            hz, power = signal.welch(signal.detrend(raw[run]), fs=fps, nperseg=min(round(8 * fps), len(raw[run])))
            psd_axis.semilogy(hz, power)
        psd_axis.axvspan(*BAND_HZ, alpha=.15)
        if reference is not None:
            psd_axis.axvline(reference, color="k", linestyle=":")
        psd_axis.set_xlim(0, 5)
    axes[0, 0].set_title("Filtered waveform (standard deviations)")
    axes[0, 1].set_title("Pre-bandpass PSD (projection units squared per Hz)")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_waveforms_psd.png", dpi=140)
    plt.close(fig)


def nuisance(data, frames, fps):
    geometry = data["geometry"][frames]
    center = geometry[:, :2] + geometry[:, 2:] / 2
    motion = np.linalg.norm(np.diff(center, axis=0), axis=1)
    widths = geometry[:-1, 2]
    valid = np.isfinite(motion) & (widths > 0)
    speed = np.sqrt(np.mean((fps * motion[valid] / widths[valid]) ** 2)) if valid.any() else np.nan
    light = data["light"][frames]
    valid_light = light[np.isfinite(light)]
    light_cv = valid_light.std() / valid_light.mean() if len(valid_light) and valid_light.mean() > 0 else np.nan
    return speed, light_cv


def compare_video(data, item, windows, inventory):
    times, fps = data["times"], round(float(data["fps"]), 12)
    grid = np.arange(int(np.floor(times[-1] * fps + 1e-8)) + 1) / fps
    duration = len(grid) / fps
    identity = dict(video_id=item["video_id"], group_id=item["group_id"],
                    category="real" if item["label"] == 0 else "manipulated",
                    method_family=item["manipulation_method"] if item["label"] else "original")
    nominal = float(data["nominal"])
    base = data["baseline"]
    base_codes = data["base_codes"]
    base_runs = roi_runs(base_codes)
    centers = data["centers"]
    displacement = np.linalg.norm(np.diff(centers, axis=0), axis=2)
    area = data["area"]
    positive_area = area > 0
    exposure = np.divide(data["valid_area"], area, out=np.full(area.shape, np.nan), where=positive_area)
    inventory.append(dict(**identity, duration=duration, fps=fps, nominal=nominal,
                          manifest_duration=item["duration"], manifest_frames=item["frame_count"],
                          decoded_frames=len(times), resolution=item["resolution"], codec=item["codec_fourcc"],
                          bits_per_pixel_frame=8 * item["file_size"] /
                          (len(times) * item["width"] * item["height"]),
                          max_interval_deviation_ms=1000 * np.max(abs(np.diff(times) - 1 / fps)),
                          observed_rgb=np.isfinite(base).all(axis=2).mean(),
                          fallback_fraction=np.mean(base_codes == 2),
                          face_acceptance_fraction=np.mean(data["face_status"] == "accepted"),
                          median_centroid_step_px=np.median(displacement[np.isfinite(displacement)])
                          if np.isfinite(displacement).any() else np.nan,
                          mean_exposure_support=exposure[positive_area].mean() if positive_area.any() else np.nan,
                          skin_retained_area=data["skin_area"].sum() / area.sum() if area.sum() else np.nan,
                          capture_seconds=float(data["seconds"])))
    for length in DURATIONS:
        count = round(length * fps)
        if duration + 1e-8 < length or count < 1:
            continue
        start = (len(grid) - count) // 2
        local = grid[start:start + count]
        frames = (times >= local[0] - 1e-8) & (times < local[-1] + 1 / fps - 1e-8)
        speed, light = nuisance(data, frames, fps)
        metadata = dict(**identity, duration_s=length, matched_30=int(duration >= 30),
                        face_widths_per_s=speed, light_cv=light, available_duration_s=duration,
                        bits_per_pixel_frame=inventory[-1]["bits_per_pixel_frame"])
        rgb, groups = snap_rgb(times[frames], base[frames], base_runs[frames], local, fps)

        def append(variant, method, colors, codes, names=REGIONS):
            rows, raw, waves = score(method, colors, codes, fps)
            for region, row in zip(names, rows, strict=True):
                row["motion_aware_pass"] = (int(row["quality_pass"] and speed <= .5)
                                            if np.isfinite(speed) else np.nan)
                windows.append(dict(**metadata, variant=variant, method=method, region=region, **row))
            return raw, waves

        for method in ALGORITHMS:
            append("native_snap", method, rgb, groups)
        if length != 8:
            continue
        # Compare each alternate mask to its baseline on common observed frame slots.
        for kind in ("skin", "rigid"):
            other, other_groups = snap_rgb(times[frames], data[kind][frames],
                                            roi_runs(data[f"{kind}_codes"])[frames], local, fps)
            shared = np.isfinite(rgb).all(axis=2) & np.isfinite(other).all(axis=2)
            joint_groups = groups * (other_groups.max() + 1) + other_groups
            append(f"{kind}_control", "POS", np.where(shared[:, :, None], rgb, np.nan), joint_groups)
            append(kind, "POS", np.where(shared[:, :, None], other, np.nan), joint_groups)
            append(f"{kind}_all", "POS", other, other_groups)
        # Compare bounded interpolation without sampling across ROI source changes.
        rel = times[frames] - local[0]
        _, interp, codes = resample_observations(np.r_[rel, count / fps],
                                                np.concatenate([base[frames], np.full((1, 3, 3), np.nan)]),
                                                fps, np.vstack([base_runs[frames], np.zeros(3, dtype=int)]))
        append("bounded_interpolation", "POS", interp[:count], codes[:count])
        append("CHROM_bandpassed", "CHROM_FILTERED", rgb, groups)
        brightness = data["light"][frames]
        usable_light = (brightness > 10) & np.isfinite(brightness)
        scale = np.where(usable_light, brightness, np.nan)
        if usable_light.any():
            corrected = base[frames] * (np.median(brightness[usable_light]) / scale)[:, None, None]
            corrected, light_groups = snap_rgb(times[frames], corrected, base_runs[frames], local, fps)
            append("illumination_control", "POS", np.where(np.isfinite(corrected), rgb, np.nan), groups)
            append("illumination", "POS", corrected, light_groups)
        valid = np.isfinite(rgb).all(axis=2)
        common = valid.all(axis=1)
        shared_codes = np.r_[0, np.any(np.diff(groups, axis=0), axis=1).cumsum()]
        supported = np.where(common[:, None, None], rgb, np.nan)
        repeated = np.repeat(shared_codes[:, None], rgb.shape[1], axis=1)
        append("multi_roi_control", "POS", supported, repeated)
        selected = int(np.argmax([np.isfinite(rgb[:, i]).all(axis=1).sum() for i in range(rgb.shape[1])]))
        append("forehead_only", "POS", rgb[:, :1], groups[:, :1], (next(iter(REGIONS)),))
        append("adaptive_roi", "POS", rgb[:, selected:selected + 1], groups[:, selected:selected + 1],
               (tuple(REGIONS)[selected],))
        append("multi_roi_rgb_mean", "POS", supported.mean(axis=1, keepdims=True), shared_codes[:, None], ("mean",))


def summarize(rows, keys):
    buckets = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for labels, bucket in buckets.items():
        result = dict(zip(keys, labels, strict=True), traces=len(bucket),
                      videos=len({row["video_id"] for row in bucket}))
        for metric in METRICS:
            values = np.array([row.get(metric, np.nan) for row in bucket], dtype=float)
            finite = values[np.isfinite(values)]
            result[metric] = np.mean(finite) if len(finite) else np.nan
            result[f"{metric}_n"] = len(finite)
        measured = ("spectral_concentration", "dominant_hz", "inband_to_outband_db",
                    "prefilter_variance", "pulse_bpm_estimate")
        result["missing_feature_pct"] = 100 * np.mean(
            [not np.isfinite(float(row.get(key) if row.get(key) is not None else np.nan))
             for row in bucket for key in measured])
        output.append(result)
    return output


def nuisance_associations(rows):
    """Describe per-video rank associations without counting ROIs as independent clips."""
    buckets = {}
    for row in rows:
        if row["variant"] == "native_snap":
            key = row["category"], row["method"], row["video_id"]
            buckets.setdefault(key, []).append(row)
    by_method = {}
    for (category, method, _), region_rows in buckets.items():
        summaries = dict(region_rows[0])
        for name in ("spectral_concentration", "processed_coverage", "quality_pass"):
            values = np.array([row[name] for row in region_rows], dtype=float)
            summaries[name] = np.mean(values[np.isfinite(values)]) if np.isfinite(values).any() else np.nan
        by_method.setdefault((category, method), []).append(summaries)
    output = []
    for (category, method), clips in by_method.items():
        for proxy in ("face_widths_per_s", "light_cv", "bits_per_pixel_frame"):
            for metric in ("spectral_concentration", "processed_coverage", "quality_pass"):
                values = np.asarray([(row[proxy], row[metric]) for row in clips])
                valid = values[np.isfinite(values).all(axis=1)]
                varies = len(valid) >= 3 and np.all(np.ptp(valid, axis=0) > 0)
                rho = stats.spearmanr(valid[:, 0], valid[:, 1]).statistic if varies else np.nan
                output.append(dict(category=category, method=method, proxy=proxy, metric=metric,
                                   videos=len(valid), spearman_rho=rho))
    return output


def paired_ablations(rows):
    comparisons = (("skin_control", "skin"), ("rigid_control", "rigid"),
                   ("illumination_control", "illumination"), ("forehead_only", "adaptive_roi"),
                   ("native_snap", "bounded_interpolation"))
    result = []
    for control, trial in comparisons:
        for category in ("real", "manipulated"):
            keyed = {}
            for row in rows:
                if row["category"] != category or row["method"] != "POS" or row["variant"] not in (control, trial):
                    continue
                key = (row["video_id"], row["region"] if trial != "adaptive_roi" else "")
                keyed.setdefault(key, {})[row["variant"]] = row
            pairs = [(pair[control], pair[trial]) for pair in keyed.values() if len(pair) == 2]
            concentration_delta = [b["spectral_concentration"] - a["spectral_concentration"]
                                   for a, b in pairs if a["spectral_concentration"] is not None
                                   and b["spectral_concentration"] is not None]
            coverage_delta = [b["processed_coverage"] - a["processed_coverage"] for a, b in pairs]
            transitions = [b["quality_pass"] - a["quality_pass"] for a, b in pairs]
            result.append(dict(category=category, control=control, variant=trial, pairs=len(pairs),
                               concentration_pairs=len(concentration_delta),
                               mean_concentration_change=np.mean(concentration_delta) if concentration_delta else np.nan,
                               mean_coverage_change=np.mean(coverage_delta) if coverage_delta else np.nan,
                               gained_passes=transitions.count(1), lost_passes=transitions.count(-1)))
    return result


def video_plots(output, rows, examples):
    import matplotlib.pyplot as plt

    for category in ("real", "manipulated"):
        baseline = [row for row in rows if row["category"] == category and row["variant"] == "native_snap"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for method in ALGORITHMS:
            named = [row for row in baseline if row["method"] == method]
            quality = [row["spectral_concentration"] for row in named
                       if row["spectral_concentration"] is not None and row["duration_s"] == 8]
            if quality:
                axes[0].hist(quality, bins=np.linspace(0, 1, 21), histtype="step", label=method)
            matched = [[row["spectral_concentration"] for row in named if row["matched_30"]
                        and row["duration_s"] == length and row["spectral_concentration"] is not None]
                       for length in DURATIONS]
            axes[1].plot(DURATIONS, [np.mean(values) if values else np.nan for values in matched],
                         marker="o", label=method)
        axes[0].axvline(MIN_QUALITY, color="k", linestyle=":")
        axes[0].set(xlabel="Spectral concentration", ylabel="Trace count", title="Eight-second quality")
        axes[1].axhline(MIN_QUALITY, color="k", linestyle=":")
        axes[1].set(xlabel="Duration (s)", ylabel="Mean concentration", ylim=(0, 1),
                    title="Common 30-second video subset")
        labels = tuple(REGIONS)
        for index, (variant, label) in enumerate((("native_snap", "Baseline"), ("skin", "Skin mask"),
                                                  ("rigid", "Rigid alignment"), ("illumination", "Lighting"))):
            values = [[row["processed_coverage"] for row in rows if row["category"] == category
                       and row["duration_s"] == 8 and row["variant"] == variant and row["method"] == "POS"
                       and row["region"] == region] for region in labels]
            axes[2].bar(np.arange(len(labels)) + index * .19,
                        [np.mean(value) if value else np.nan for value in values], .18, label=label)
        axes[2].set(xticks=np.arange(len(labels)) + .285, xticklabels=("Forehead", "Left cheek", "Right cheek"),
                    ylabel="Processed fraction", title="POS ROI support")
        for axis in axes:
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{category}_quality_duration_roi.png", dpi=140)
        plt.close(fig)
    for category, (clip_id, t, waveforms) in examples.items():
        plots(output, category, t, waveforms, subject=f"{clip_id} · first eight seconds · forehead")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, category in enumerate(("real", "manipulated")):
        subset = [row for row in rows if row["category"] == category and row["variant"] == "native_snap"]
        counts = [len({row["video_id"] for row in subset if row["duration_s"] == length})
                  for length in DURATIONS]
        axes[0].plot(DURATIONS, counts, marker="o", label=category)
        runtimes = [[row["seconds"] for row in subset if row["duration_s"] == 8 and row["method"] == method]
                    for method in ALGORITHMS]
        axes[1].bar(np.arange(len(ALGORITHMS)) + .35 * index,
                    [1000 * np.mean(values) if values else np.nan for values in runtimes], .35, label=category)
    axes[0].set(xlabel="Window duration (s)", ylabel="Eligible videos", title="Available video coverage")
    axes[1].set(xticks=np.arange(len(ALGORITHMS)) + .175, xticklabels=ALGORITHMS,
                ylabel="Mean ms per trace", title="Eight-second projection and scoring cost")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "coverage_runtime.png", dpi=140)
    plt.close(fig)


def pilot_audit(output):
    directory = Path("outputs/test7/pilot")
    selected = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    training = {row["video_id"]: row for row in selected if row["partition"] == "train"}
    results = []
    for file in sorted((directory / "checkpoints").glob("*.zip")):
        with ZipFile(file) as archive:
            windows = json.loads(archive.read("windows.json"))
            if not windows or windows[0]["clip_id"] not in training:
                continue
            features = csv.DictReader(archive.read("rppg.csv").decode("utf-8").splitlines())
            item = training[windows[0]["clip_id"]]
            for window, row in zip(windows, features, strict=True):
                for region in REGIONS:
                    for method in ("GREEN", "POS"):
                        prefix = f"rppg_{region}_{method.lower()}_"
                        concentration = float(row[prefix + "spectral_concentration"] or "nan")
                        coverage = float(row[prefix + "processed_coverage"])
                        passes = np.isfinite(concentration) and concentration >= MIN_QUALITY and coverage >= .5
                        if passes != bool(float(row[prefix + "quality_pass"])):
                            raise ValueError("Stored baseline pass disagrees with the gate")
                        results.append(dict(video_id=item["video_id"], category="real" if not item["label"]
                                            else "manipulated", method=method, region=region,
                                            window_index=window["window_index"], processed_coverage=coverage,
                                            raw_coverage=float(row[prefix + "raw_coverage"]),
                                            longest_processed_run_s=float(row[prefix + "longest_processed_run_s"]),
                                            spectral_concentration=concentration, quality_pass=int(passes),
                                            pulse_available=int(bool(row[prefix + "pulse_bpm_estimate"])),
                                            cause="unscorable" if not np.isfinite(concentration) else
                                            "coverage" if coverage < .5 else
                                            "low_concentration" if concentration < MIN_QUALITY else "passed"))
    write_table(output, "phase4_pilot_gate_audit", results)


def timing_probe(output, selection):
    """Cross-check millisecond positions against FFmpeg PTS ticks on one clip per family."""
    import cv2

    rows, covered = [], set()
    for row in selection:
        method = row["manipulation_method"]
        if method in covered:
            continue
        covered.add(method)
        capture = cv2.VideoCapture(row["absolute_path"], cv2.CAP_FFMPEG)
        times, pts = [], []
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            while capture.grab():
                times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                pts.append(capture.get(cv2.CAP_PROP_PTS))
        finally:
            capture.release()
        disagreement = np.nan
        monotonic = len(pts) > 1 and np.all(np.diff(pts) > 0) and fps > 0
        if monotonic:
            elapsed = np.asarray(times) - times[0]
            relative_pts = (np.asarray(pts) - pts[0]) / fps
            disagreement = 1000 * np.max(np.abs(elapsed - relative_pts))
        rows.append(dict(video_id=row["video_id"], method_family=method, frames=len(pts), nominal_fps=fps,
                         monotonic_pts=int(monotonic), max_pts_difference_ms=disagreement))
    write_table(output, "timing_probe", rows)


def acquisition_sensitivity(output, primary_capture):
    """Compare full and downscaled face detection on the initial shared captures."""
    rows = []
    for cohort, originals, alternatives, metadata in (
        ("preliminary", output / "observations", output / "reduced" / "observations",
         output / "capture_progress.jsonl"),
        ("primary_overlap", output / "native_overlap", primary_capture / "observations",
         output / "native_overlap" / "capture_summary.json"),
    ):
        if not metadata.is_file():
            continue
        contents = metadata.read_text(encoding="utf-8")
        items = ([json.loads(line) for line in contents.splitlines()] if metadata.suffix == ".jsonl"
                 else json.loads(contents))
        for item in items:
            if "archive" not in item:
                continue
            name = item["archive"]
            if not (alternatives / name).is_file():
                continue
            rows.extend(compare_detectors(originals / name, alternatives / name, cohort, item["video_id"]))
    write_table(output, "detector_sensitivity", rows)


def compare_detectors(native, reduced, cohort, video_id):
    with np.load(native, allow_pickle=False) as a, np.load(reduced, allow_pickle=False) as b:
        if not np.array_equal(a["times"], b["times"]):
            return []
        first, second = a["baseline"], b["baseline"]
        shared = np.isfinite(first).all(axis=2) & np.isfinite(second).all(axis=2)
        fps = round(float(a["fps"]), 12)
        n = round(8 * fps)
        start = (len(first) - n) // 2
        window = slice(start, start + n)
        rows = []
        for variant, archive, colors in (("native_detector", a, first), ("384px_detector", b, second)):
            groups = roi_runs(archive["base_codes"])
            rgb = np.where(shared[:, :, None], colors, np.nan)
            for method in ("GREEN", "POS"):
                scored, _, _ = score(method, rgb[window], groups[window], fps)
                for j, (region, row) in enumerate(zip(REGIONS, scored, strict=True)):
                    common = shared[:, j]
                    difference = first[common, j] - second[common, j]
                    rows.append(dict(cohort=cohort, video_id=video_id, variant=variant, region=region, method=method,
                                     rgb_rmse=np.sqrt(np.mean(difference**2)) if len(difference) else np.nan,
                                     full_observed_coverage=np.isfinite(colors[:, j]).all(axis=1).mean(),
                                     shared_observed_coverage=common.mean(), **row))
        return rows


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=OUTPUT / "training")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    synthetic(OUTPUT)
    compression_phantom(OUTPUT)
    pilot_audit(OUTPUT)
    selection = json.loads((args.capture / "selection.json").read_text(encoding="utf-8"))
    settings = json.loads((args.capture / "capture_settings.json").read_text(encoding="utf-8"))
    if any(item["partition"] != "train" for item in selection):
        raise ValueError("Phase 4.5 may only use training videos")
    timing_probe(OUTPUT, selection)
    acquisition_sensitivity(OUTPUT, args.capture)
    rows, inventory, examples = [], [], {}
    for item in selection:
        path = args.capture / "observations" / f"{sha256(item['video_id'].encode()).hexdigest()}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["key"]) != sha256((item["sha256"] + settings["revision"]).encode()).hexdigest():
                continue
            data = dict(archive)
            compare_video(data, item, rows, inventory)
            category = "real" if item["label"] == 0 else "manipulated"
            if category not in examples:
                fps = float(data["fps"])
                n = min(round(8 * fps), len(data["times"]))
                rgb, groups = data["baseline"][:n, :1], roi_runs(data["base_codes"])[:n, :1]
                example = {}
                for method in ALGORITHMS:
                    _, raw, wave = score(method, rgb, groups, fps)
                    example[method] = (raw[0], wave[0])
                examples[category] = (item["video_id"], np.arange(n) / fps, example)
    if args.require_complete and len(inventory) != len(selection):
        raise ValueError(f"Captured {len(inventory)} of {len(selection)} selected videos")
    write_table(OUTPUT, "video_inventory", inventory)
    write_table(OUTPUT, "trace_metrics", rows)
    eight = [row for row in rows if row["duration_s"] == 8]
    write_table(OUTPUT, "algorithm_comparison", summarize(eight, ("category", "variant", "method")))
    write_table(OUTPUT, "method_family_comparison",
                summarize(eight, ("method_family", "variant", "method")))
    write_table(OUTPUT, "roi_comparison", summarize(eight, ("category", "variant", "method", "region")))
    write_table(OUTPUT, "paired_ablation_comparison", paired_ablations(eight))
    duration_rows = [row for row in rows if row["variant"] == "native_snap"]
    keys = ("category", "duration_s", "method")
    write_table(OUTPUT, "duration_comparison", summarize(duration_rows, keys))
    write_table(OUTPUT, "duration_matched_30", summarize([row for row in duration_rows if row["matched_30"]], keys))
    matched_rows = []
    for longer in DURATIONS[1:]:
        candidates = [row for row in duration_rows
                      if row["available_duration_s"] >= longer and row["duration_s"] in (8, longer)]
        reliable = {(row["video_id"], row["method"], row["region"], row["duration_s"])
                    for row in candidates if row["processed_coverage"] >= .95}
        for row in candidates:
            identity = (row["video_id"], row["method"], row["region"])
            matched_rows.append(dict(row, matched_against_s=longer, support="all"))
            if all((*identity, length) in reliable for length in (8, longer)):
                matched_rows.append(dict(row, matched_against_s=longer, support="both_at_least_95pct"))
    write_table(OUTPUT, "duration_paired_support",
                summarize(matched_rows, ("matched_against_s", "support", *keys)))
    nuisance_rows = []
    for row in eight:
        if row["variant"] != "native_snap":
            continue
        for proxy, cutoff in (("face_widths_per_s", .5), ("light_cv", .05), ("bits_per_pixel_frame", .1)):
            value = row[proxy]
            level = "unknown" if not np.isfinite(value) else "higher" if value > cutoff else "lower"
            nuisance_rows.append(dict(row, proxy=proxy, level=level))
    write_table(OUTPUT, "nuisance_sensitivity", summarize(nuisance_rows, ("category", "method", "proxy", "level")))
    write_table(OUTPUT, "nuisance_rank_associations", nuisance_associations(eight))
    write_table(OUTPUT, "runtime_comparison", [
        dict(category=row["category"], variant=row["variant"], method=row["method"],
             traces=row["traces"], mean_ms_per_trace=1000 * row["seconds"])
        for row in summarize(eight, ("category", "variant", "method"))
    ])
    if rows:
        video_plots(OUTPUT, rows, examples)
    (OUTPUT / "experiment_run.json").write_text(json.dumps(dict(
        capture=str(args.capture), selected=len(selection), scored=len(inventory), trace_windows=len(rows),
        detector_width=settings.get("detector_width", 0), minimum_concentration=MIN_QUALITY,
    ), indent=2) + "\n", encoding="utf-8")
    print(f"{len(inventory)}/{len(selection)} selected videos scored, {len(rows)} trace-window measurements")


if __name__ == "__main__":
    main()

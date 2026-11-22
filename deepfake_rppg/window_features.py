"""Signal and expression measurements computed independently within one window."""

from itertools import combinations

import numpy as np
from scipy import signal

from .expressions import SELECTED, temporal_features
from .face import REGIONS
from .signal import BAND_HZ, MIN_QUALITY, MIN_SEGMENT_SECONDS, finite_runs, pos, process_signal


MIN_PROCESSED_COVERAGE = 0.5
MIN_PULSE_SECONDS = 6.0
MIN_EXPRESSION_SAMPLES = 3
ASYMMETRY_PAIRS = tuple(name[:-4] for name in SELECTED if name.endswith("Left") and name[:-4] + "Right" in SELECTED)


def snap_rgb(times, rgb, groups, grid, fps):
    """Place observations in unique, nearby grid slots without filling samples."""
    sampled = np.full((len(grid), *rgb.shape[1:]), np.nan)
    sampled_groups = np.zeros(sampled.shape[:2], dtype=int)
    if not len(grid):
        return sampled, sampled_groups
    relative = (times - grid[0]) * fps
    indices = np.rint(relative).astype(int)
    candidates = (indices >= 0) & (indices < len(grid)) & (np.abs(relative - indices) <= 0.25 + 1e-9)
    unique, counts = np.unique(indices[candidates], return_counts=True)
    rows = candidates & np.isin(indices, unique[counts == 1])
    sampled[indices[rows]] = rgb[rows]
    sampled_groups[indices[rows]] = groups[rows]
    return sampled, sampled_groups


def longest_run(valid, groups=None):
    return max(finite_runs(valid, groups), key=lambda run: run.stop - run.start, default=slice(0, 0))


def peak_and_power_ratio(values, fps):
    """Use a pre-filter Welch spectrum and compare in-band to other non-DC power."""
    if len(values) < np.ceil(MIN_SEGMENT_SECONDS * fps - 1e-9):
        return np.nan, np.nan
    nperseg = min(len(values), round(8 * fps))
    hz, power = signal.welch(signal.detrend(values), fs=fps, nperseg=nperseg)
    in_band = (hz >= BAND_HZ[0]) & (hz <= BAND_HZ[1])
    in_power = power[in_band].sum()
    out_power = power[(hz > 0) & ~in_band].sum()
    if in_power <= 0:
        return np.nan, np.nan
    peak = float(hz[in_band][np.argmax(power[in_band])])
    ratio_db = float(10 * np.log10(in_power / out_power)) if out_power > 0 else np.nan
    return peak, ratio_db


def trace_features(raw, fps, groups):
    processed, quality = process_signal(raw, fps, groups=groups)
    valid = np.isfinite(processed)
    coverage = quality["processed_coverage"]
    concentration = quality["spectral_concentration"]
    run = longest_run(valid, groups)
    length = (run.stop - run.start) / fps
    peak, db = peak_and_power_ratio(raw[run], fps)
    passes = concentration is not None and concentration >= MIN_QUALITY and coverage >= MIN_PROCESSED_COVERAGE
    pulse_estimate = np.nan
    if passes and length + 1e-9 >= MIN_PULSE_SECONDS and np.isfinite(peak):
        _, run_quality = process_signal(raw[run], fps)
        run_score = run_quality["spectral_concentration"]
        if run_score is not None and run_score >= MIN_QUALITY:
            pulse_estimate = 60 * peak
    observed = raw[np.isfinite(raw)]
    return {
        "raw_coverage": len(observed) / len(raw),
        "processed_coverage": coverage,
        "longest_processed_run_s": length,
        "prefilter_variance": observed.var() if len(observed) >= 2 else np.nan,
        "spectral_concentration": concentration,
        "dominant_hz": peak,
        "inband_to_outband_db": db,
        "quality_pass": int(passes),
        "pulse_bpm_estimate": pulse_estimate,
    }, processed


def rppg_features(native_rgb, rgb, groups, fps):
    features = {}
    raw_pos, processed_pos = [], []
    for i, region in enumerate(REGIONS):
        prefix = f"rppg_{region}_"
        features[prefix + "observed_rgb_coverage"] = (
            np.isfinite(native_rgb[:, i]).all(axis=1).mean() if len(native_rgb) else 0.0
        )
        colors = rgb[:, i]
        projected = pos(colors, fps, groups[:, i])
        for method, raw in (("green", colors[:, 1]), ("pos", projected)):
            metrics, processed = trace_features(raw, fps, groups[:, i])
            features.update((prefix + f"{method}_{key}", value) for key, value in metrics.items())
        raw_pos.append(projected)
        processed_pos.append(processed)

    for i, j in combinations(range(len(REGIONS)), 2):
        regions = tuple(REGIONS)
        prefix = f"rppg_pair_{regions[i]}__{regions[j]}_"
        x, y = processed_pos[i], processed_pos[j]
        paired = np.isfinite(x) & np.isfinite(y)
        changed = np.any(np.diff(groups[:, [i, j]], axis=0), axis=1)
        run = longest_run(paired, np.r_[0, np.cumsum(changed)])
        samples = run.stop - run.start
        correlation, peak_delta = np.nan, np.nan
        if samples >= np.ceil(MIN_SEGMENT_SECONDS * fps - 1e-9):
            if x[run].std() > 1e-8 and y[run].std() > 1e-8:
                correlation = np.corrcoef(x[run], y[run])[0, 1]
            left, _ = peak_and_power_ratio(raw_pos[i][run], fps)
            right, _ = peak_and_power_ratio(raw_pos[j][run], fps)
            peak_delta = abs(left - right)
        features.update({
            prefix + "joint_processed_coverage": paired.mean(),
            prefix + "longest_joint_run_s": samples / fps,
            prefix + "pos_correlation": correlation,
            prefix + "pos_peak_delta_hz": peak_delta,
        })
    return features


def available_mean(values):
    observed = values[np.isfinite(values)]
    return observed.mean() if len(observed) else np.nan


def expression_features(times, scores, fps):
    _, local_std, velocity = temporal_features(times, scores, fps)
    features = {
        "expr_all_selected_coverage": np.isfinite(scores).all(axis=1).mean() if len(scores) else 0.0,
    }
    for i, name in enumerate(SELECTED):
        valid = scores[np.isfinite(scores[:, i]), i]
        enough = len(valid) >= MIN_EXPRESSION_SAMPLES
        prefix = f"expr_{name}_"
        features.update({
            prefix + "coverage": len(valid) / len(scores) if len(scores) else 0.0,
            prefix + "mean": valid.mean() if enough else np.nan,
            prefix + "std": valid.std() if enough else np.nan,
            prefix + "mean_rolling_std_0p5s": available_mean(local_std[:, i]),
            prefix + "mean_abs_velocity_per_s": available_mean(np.abs(velocity[:, i])),
        })
    for name in ASYMMETRY_PAIRS:
        i, j = (SELECTED.index(name + side) for side in ("Left", "Right"))
        difference = scores[:, i] - scores[:, j]
        paired = difference[np.isfinite(difference)]
        enough = len(paired) >= MIN_EXPRESSION_SAMPLES
        prefix = f"expr_asym_{name}_"
        features.update({
            prefix + "paired_coverage": len(paired) / len(scores) if len(scores) else 0.0,
            prefix + "mean_left_minus_right": paired.mean() if enough else np.nan,
            prefix + "mean_abs_difference": np.abs(paired).mean() if enough else np.nan,
        })
    return features

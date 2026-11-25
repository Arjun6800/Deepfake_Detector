"""Versioned, BPM-independent features of facial color projections."""

from itertools import combinations

import numpy as np
from scipy import signal

from .face import REGIONS
from .signal import BAND_HZ, MIN_QUALITY, finite_runs, process_signal
from .window_features import MIN_PROCESSED_COVERAGE, longest_run

SCHEMA = "rppg_consistency_v1"
REGION_NAMES = (*REGIONS, "aggregate")
PSD_EDGES = (0.25, 0.7, 1.2, 2.0, 3.0, 5.0)


def robust_trace(raw, groups, fps, robust=False):
    """Keep signed projection units and split at gaps and mask provenance changes."""
    output = np.full(len(raw), np.nan)
    for run in finite_runs(np.isfinite(raw), groups):
        if run.stop - run.start < round(fps):
            continue
        values = signal.detrend(raw[run])
        if robust:
            center = np.median(values)
            scale = 1.4826 * np.median(abs(values - center))
            if scale > 1e-10:
                values = np.clip(values, center - 6 * scale, center + 6 * scale)
        output[run] = values
    return output


def reduce_nuisance(raw, nuisance, groups, fps):
    """Subtract a shrunk fit of measured nuisance, without extrapolating across gaps."""
    output = raw.copy()
    valid = np.isfinite(raw) & np.isfinite(nuisance).all(axis=1)
    for run in finite_runs(valid, groups):
        if run.stop - run.start < round(4 * fps):
            continue
        proxies = signal.detrend(nuisance[run], axis=0)
        spread = proxies.std(axis=0)
        keep = spread > 1e-8
        if not keep.any():
            continue
        design = proxies[:, keep] / spread[keep]
        gram = design.T @ design + 0.25 * len(design) * np.eye(design.shape[1])
        beta = np.linalg.solve(gram, design.T @ signal.detrend(raw[run]))
        output[run] -= 0.75 * (design @ beta)
    return output


def aggregate(regional, groups, weights):
    """Combine standardized observations with measurement-based weights and provenance."""
    standardized = np.full_like(regional, np.nan)
    for i in range(regional.shape[1]):
        for run in finite_runs(np.isfinite(regional[:, i]), groups[:, i]):
            values = regional[run, i]
            scale = values.std()
            if scale > 1e-10:
                standardized[run, i] = (values - values.mean()) / scale
    active = np.isfinite(standardized) & (weights > 0)
    support = np.where(active, weights, 0)
    total = support.sum(axis=1)
    combined = np.divide(np.nansum(standardized * support, axis=1), total,
                         out=np.full(len(regional), np.nan), where=active.sum(axis=1) >= 2)
    signature = np.where(active, groups + 1, 0)
    changes = np.any(np.diff(signature, axis=0) != 0, axis=1)
    return combined, np.r_[0, changes.cumsum()]


def one_trace(raw, wave, groups, fps):
    run = longest_run(np.isfinite(wave), groups)
    values = wave[run]
    _, quality = process_signal(raw, fps, groups=groups)
    score = quality["spectral_concentration"]
    parts = {
        "signal": dict(rms=np.nan, amplitude_iqr=np.nan),
        "temporal": dict(acf_half_second=np.nan, segment_rms_cv=np.nan, derivative_rms=np.nan),
        "spectral": dict(entropy=np.nan, concentration=score if score is not None else np.nan,
                         **{f"relative_power_{i}": np.nan for i in range(len(PSD_EDGES) - 1)}),
        "reliability": dict(raw_coverage=np.isfinite(raw).mean(), waveform_coverage=np.isfinite(wave).mean(),
                            longest_run_s=len(values) / fps, legacy_processed_coverage=quality["processed_coverage"],
                            legacy_gate=int(score is not None and score >= MIN_QUALITY
                                            and quality["processed_coverage"] >= MIN_PROCESSED_COVERAGE)),
    }
    hz, psd = np.empty(0), np.empty(0)
    if len(values) >= round(fps):
        parts["signal"].update(rms=np.sqrt(np.mean(values**2)),
                               amplitude_iqr=np.diff(np.quantile(values, [.25, .75]))[0])
        parts["temporal"]["derivative_rms"] = fps * np.sqrt(np.mean(np.diff(values)**2))
        lag = round(.5 * fps)
        parts["temporal"]["acf_half_second"] = correlation(values[:-lag], values[lag:])
        blocks = len(values) // round(fps)
        if blocks >= 2:
            sections = values[:blocks * round(fps)].reshape(blocks, -1)
            levels = np.sqrt(np.mean(sections**2, axis=1))
            if levels.mean() > 1e-10:
                parts["temporal"]["segment_rms_cv"] = levels.std() / levels.mean()
    if len(values) >= round(4 * fps) and values.std() > 1e-10:
        hz, psd = signal.welch(values, fs=fps, nperseg=len(values))
        positive = (hz >= PSD_EDGES[0]) & (hz < PSD_EDGES[-1])
        total = psd[positive].sum()
        if total > 0:
            probabilities = psd[positive] / total
            nonzero = probabilities > 0
            parts["spectral"]["entropy"] = -np.dot(probabilities[nonzero],
                                                  np.log(probabilities[nonzero])) / np.log(len(probabilities))
            for i, (low, high) in enumerate(zip(PSD_EDGES[:-1], PSD_EDGES[1:], strict=True)):
                parts["spectral"][f"relative_power_{i}"] = psd[(hz >= low) & (hz < high)].sum() / total
    return parts, hz, psd


def correlation(x, y):
    if len(x) < 2 or min(x.std(), y.std()) <= 1e-10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pair_features(x, y, groups, fps):
    joint = np.isfinite(x) & np.isfinite(y)
    provenance = np.r_[0, np.any(np.diff(groups, axis=0) != 0, axis=1).cumsum()]
    run = longest_run(joint, provenance)
    x, y = x[run], y[run]
    values = dict(correlation=np.nan, coherence=np.nan, phase_consistency=np.nan,
                  spectral_disagreement=np.nan, relative_rms_disagreement=np.nan)
    reliability = dict(joint_coverage=joint.mean(), longest_joint_run_s=len(x) / fps)
    if len(x) < round(4 * fps) or min(x.std(), y.std()) <= 1e-10:
        return values, reliability
    values["correlation"] = correlation(x, y)
    values["relative_rms_disagreement"] = abs(x.std() - y.std()) / (x.std() + y.std())
    size = round(2 * fps)
    hz, _, ax = signal.stft(x, fs=fps, nperseg=size, noverlap=size // 2, boundary=None, padded=False)
    _, _, ay = signal.stft(y, fs=fps, nperseg=size, noverlap=size // 2, boundary=None, padded=False)
    band = (hz >= BAND_HZ[0]) & (hz <= BAND_HZ[1])
    cross = ax[band].conj() * ay[band]
    sx, sy = np.mean(abs(ax[band])**2, axis=1), np.mean(abs(ay[band])**2, axis=1)
    if sx.sum() > 0 and sy.sum() > 0:
        values["spectral_disagreement"] = 0.5 * abs(sx / sx.sum() - sy / sy.sum()).sum()
        weights = np.sqrt(sx * sy)
        valid = (weights > 0) & (np.count_nonzero(abs(cross) > 0, axis=1) >= 3)
        if valid.any():
            selected = cross[valid]
            coh = abs(selected.mean(axis=1))**2 / weights[valid]**2
            unit = np.divide(selected, abs(selected), out=np.zeros_like(selected), where=abs(selected) > 0)
            plv = abs(unit.sum(axis=1)) / np.count_nonzero(abs(selected) > 0, axis=1)
            values["coherence"] = np.clip(np.average(coh, weights=weights[valid]), 0, 1)
            values["phase_consistency"] = np.clip(np.average(plv, weights=weights[valid]), 0, 1)
    return values, reliability


def describe(raw, wave, groups, fps):
    """Return grouped scalar features and physical-frequency PSD arrays."""
    features, spectra = {}, {}
    for i, name in enumerate(REGION_NAMES):
        parts, hz, psd = one_trace(raw[:, i], wave[:, i], groups[:, i], fps)
        for category, metrics in parts.items():
            features.update((f"{category}__{name}__{key}", value) for key, value in metrics.items())
        spectra[name] = (hz, psd)
    for i, j in combinations(range(len(REGIONS)), 2):
        metrics, reliability = pair_features(wave[:, i], wave[:, j], groups[:, [i, j]], fps)
        name = f"{REGION_NAMES[i]}--{REGION_NAMES[j]}"
        for category, items in (("spatial", metrics), ("reliability", reliability)):
            features.update((f"{category}__{name}__{key}", value) for key, value in items.items())
    return features, spectra

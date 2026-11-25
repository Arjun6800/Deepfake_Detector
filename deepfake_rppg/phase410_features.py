"""Gap-aware waveform and synchronized expression features for Phase 4.10."""
from itertools import combinations

import numpy as np
from scipy import signal

from .batch import roi_runs
from .consistency import aggregate, correlation, pair_features, reduce_nuisance, robust_trace
from .quality_algorithms import project
from .signal import finite_runs
from .window_features import longest_run, snap_rgb

VARIANTS = {
    "pos_reference": ("POS", ()),
    "green": ("GREEN", ()),
    "chrom": ("CHROM", ()),
    "pos_robust": ("POS", ("illumination", "robust", "adaptive")),
    "pos_motion": ("POS", ("motion",)),
    "pos_combined": ("POS", ("illumination", "robust", "adaptive", "motion", "weighted")),
}
TEMPORAL = ("acf_peak", "acf_peak_lag_s", "acf_decay", "ordinal_entropy", "crossings_per_s",
            "crossing_interval_cv", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
            "derivative_abs_median", "derivative_abs_p90", "short_long_scale_ratio",
            "regional_correlation", "inter_window_acf_sd", "inter_window_activity_log_sd")
SPECTRAL = ("flatness", "centroid_hz", "bandwidth_hz", "peak_prominence_fraction", "harmonic_fraction",
            "peak_stability_hz_sd", "haar_detail_entropy", "regional_coherence", "inter_window_centroid_sd")
CROSS = ("amplitude_activity_r", "derivative_velocity_r", "lagged_activity_r", "lag_s",
         "high_low_activity_amplitude_log_ratio", "motion_peak_shift_hz", "inter_window_activity_r_sd")


def align(times, values, grid, fps):
    if (len(times) != len(values) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not len(grid) or not np.isfinite(grid).all() or not np.isfinite(fps) or fps <= 0
            or not np.allclose(np.diff(grid), 1 / fps, rtol=0, atol=1e-8)):
        raise ValueError("Invalid observation or grid timebase")
    return snap_rgb(times, values[:, None], np.zeros((len(times), 1), int), grid, fps)[0][:, 0]


def observations(data, grid, fps):
    times = data["times"]
    geometry = data["geometry"]
    width = geometry[:, 2]
    accepted = np.isfinite(geometry).all(axis=1) & (width > 0)
    scale = median(width[accepted])
    center = geometry[:, :2] + geometry[:, 2:] / 2
    proxies = np.column_stack((center / scale, width / scale, data["light"]))
    proxies[~accepted] = np.nan
    nuisance = align(times, proxies, grid, fps)
    colors, ids = snap_rgb(times, data["baseline"], roi_runs(data["base_codes"]), grid, fps)
    scores = align(times, data["scores"], grid, fps)
    motion = np.r_[np.nan, fps * np.linalg.norm(np.diff(nuisance[:, :2], axis=0), axis=1)]
    return colors, ids, nuisance, scores, motion


def median(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def normalize(colors, groups, fps):
    """Remove slow multiplicative brightness drift per region and provenance run."""
    result = colors.copy()
    for run in finite_runs(np.isfinite(colors).all(axis=1), groups):
        values = colors[run]
        if len(values) < round(3 * fps):
            continue
        brightness = np.median(values, axis=1)
        size = 2 * int(fps) + 1
        smooth = signal.medfilt(np.pad(brightness, size // 2, mode="edge"), size)[size // 2:-(size // 2)]
        level = np.median(smooth)
        if level > 0:
            result[run] /= np.clip(smooth / level, .5, 2)[:, None]
    return result


def adaptive(raw, groups, fps):
    """Use robust linear detrending on short runs, broad zero-phase filtering on long runs."""
    result = robust_trace(raw, groups, fps, robust=True)
    sos = signal.butter(2, (.35, min(5., .45 * fps)), btype="bandpass", fs=fps, output="sos")
    for run in finite_runs(np.isfinite(result), groups):
        if run.stop - run.start >= round(4 * fps):
            result[run] = signal.sosfiltfilt(sos, result[run])
    return result


def waveform(colors, ids, nuisance, fps, variant):
    method, flags = VARIANTS[variant]
    waves = []
    for i, channel in enumerate(colors.transpose(1, 0, 2)):
        groups = ids[:, i]
        if "illumination" in flags:
            channel = normalize(channel, groups, fps)
        raw = project(method, channel, fps, groups)
        if "motion" in flags:
            raw = reduce_nuisance(raw, nuisance, groups, fps)
        wave = adaptive(raw, groups, fps) if "adaptive" in flags else robust_trace(raw, groups, fps)
        waves.append(wave)
    regional = np.column_stack(waves)
    weights = np.ones(regional.shape[1])
    if "weighted" in flags:
        # Coverage and RGB discontinuity, independent of spectral concentration.
        shared = (np.diff(ids, axis=0) == 0) & np.isfinite(regional[:-1]) & np.isfinite(regional[1:])
        jump = np.linalg.norm(np.diff(colors, axis=0), axis=2)
        level = np.linalg.norm(colors[:-1], axis=2)
        relative = np.divide(jump, level, out=np.full_like(jump, np.nan), where=shared & (level > 0))
        penalty = [median(relative[:, i]) for i in range(len(weights))]
        weights = np.isfinite(regional).mean(axis=0) / (1 + 20 * np.nan_to_num(penalty))
    combined, provenance = aggregate(regional, ids, weights)
    return regional, combined, provenance


def acf(x):
    x = x - x.mean()
    values = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1:]
    return values / values[0]


def temporal(x, fps):
    result = dict.fromkeys(TEMPORAL[:-3], np.nan)
    if len(x) < round(2 * fps) or x.std() <= 1e-10:
        return result
    d = np.diff(x) * fps
    mobility = d.std() / x.std()
    a = acf(x)
    low, high = max(1, round(.25 * fps)), min(len(a) - 1, round(1.6 * fps))
    peaks = signal.find_peaks(a)[0]
    peaks = peaks[(peaks >= low) & (peaks <= high)]
    peak = peaks[np.argmax(a[peaks])] if len(peaks) else None
    decay = np.flatnonzero(np.abs(a) < np.exp(-1))
    centered = x - np.median(x)
    crossings = np.flatnonzero(np.diff(np.signbit(centered)))
    intervals = np.diff(crossings)
    delay = max(1, round(.1 * fps))
    triples = np.lib.stride_tricks.sliding_window_view(x, 2 * delay + 1)[:, ::delay]
    triples = triples[(np.diff(np.sort(triples, axis=1), axis=1) > 0).all(axis=1)]
    probabilities = np.unique(np.argsort(triples, axis=1), axis=0, return_counts=True)[1]
    probabilities = probabilities / len(triples) if len(triples) else probabilities
    width = round(fps)
    scales = x[:len(x) // width * width].reshape(-1, width).std(axis=1)
    result.update(acf_peak=a[peak] if peak is not None else np.nan,
                  acf_peak_lag_s=peak / fps if peak is not None else np.nan,
                  acf_decay=decay[0] / fps if len(decay) else np.nan,
                  ordinal_entropy=-np.dot(probabilities, np.log(probabilities)) / np.log(6) if len(triples) else np.nan,
                  crossings_per_s=len(crossings) * fps / (len(x) - 1),
                  crossing_interval_cv=intervals.std() / intervals.mean() if len(intervals) >= 2 else np.nan,
                  hjorth_activity=x.var(), hjorth_mobility=mobility,
                  hjorth_complexity=np.diff(d).std() * fps / d.std() / mobility if mobility > 0 else np.nan,
                  derivative_abs_median=np.median(abs(d)), derivative_abs_p90=np.quantile(abs(d), .9),
                  short_long_scale_ratio=np.median(scales) / x.std())
    return result


def spectrum(x, fps):
    if len(x) < round(4 * fps) or x.std() <= 1e-10:
        return np.empty(0), np.empty(0)
    hz, psd = signal.welch(x, fs=fps, nperseg=len(x))
    band = (hz >= .25) & (hz <= 5)
    return hz[band], psd[band]


def spectral(x, fps):
    result = dict.fromkeys(SPECTRAL[:-2], np.nan)
    hz, p = spectrum(x, fps)
    if not len(p) or not p.sum() > 0:
        return result
    q = p / p.sum()
    peak = int(q.argmax())
    centroid = q @ hz
    peaks = signal.find_peaks(q)[0]
    prominence = signal.peak_prominences(q, [peak])[0][0] if peak in peaks else np.nan
    harmonic = 2 * hz[peak]
    width = round(2 * fps)
    _, _, stft = signal.stft(x, fs=fps, nperseg=width, boundary=None, padded=False)
    frequencies = np.fft.rfftfreq(width, 1 / fps)
    band = (frequencies >= .5) & (frequencies <= 5)
    energy = abs(stft[band])**2
    valid_frames = energy.sum(axis=0) > 0
    dominant = frequencies[band][energy[:, valid_frames].argmax(axis=0)]
    details = []
    values = x
    for _ in range(5):
        a, b = values[:len(values) // 2 * 2:2], values[1::2]
        details.append(np.sum((a - b)**2) / 2)
        values = (a + b) / np.sqrt(2)
    weights = np.array(details) / sum(details) if sum(details) > 0 else np.zeros(5)
    nz = weights > 0
    result.update(flatness=np.exp(np.log(np.maximum(q, 1e-15)).mean()) / q.mean(),
                  centroid_hz=centroid, bandwidth_hz=np.sqrt(q @ (hz - centroid)**2),
                  peak_prominence_fraction=prominence,
                  harmonic_fraction=q[np.argmin(abs(hz - harmonic))] if harmonic <= hz[-1] else np.nan,
                  peak_stability_hz_sd=dominant.std() if len(dominant) >= 3 else np.nan,
                  haar_detail_entropy=-weights[nz] @ np.log(weights[nz]) / np.log(5) if nz.any() else np.nan)
    return result


def cross_modal(wave, groups, scores, motion, fps):
    result = dict.fromkeys(CROSS[:-1], np.nan)
    active = np.isfinite(scores).all(axis=1)
    run = longest_run(active & np.isfinite(wave), groups)
    x, scores, speed = wave[run], scores[run], motion[run]
    if len(x) < round(3 * fps):
        return result
    activity = scores.mean(axis=1)
    amplitude = abs(x - np.median(x))
    velocity = abs(np.diff(scores, axis=0)).mean(axis=1) * fps
    zero = correlation(amplitude, activity)
    step = max(1, round(.1 * fps))
    limit = int(.5 * fps) // step
    lags = range(-limit * step, (limit + 1) * step, step)
    pairs = [(correlation(amplitude[max(lag, 0):len(x) + min(lag, 0)],
                          activity[max(-lag, 0):len(x) - max(lag, 0)]), lag) for lag in lags]
    coefficient, lag = max(((r, lag) for r, lag in pairs if np.isfinite(r)),
                           key=lambda pair: (abs(pair[0]), -abs(pair[1])), default=(np.nan, np.nan))
    lo, hi = np.quantile(activity, [.25, .75])
    lower, upper = amplitude[activity <= lo], amplitude[activity >= hi]
    ratio = np.log((upper.std() + 1e-8) / (lower.std() + 1e-8)) if lo < hi else np.nan
    width = round(2 * fps)
    peaks = []
    for start in range(0, len(x) - width + 1, width):
        section = slice(start, start + width)
        m = speed[section]
        if np.isfinite(m).sum() >= width - 1 and np.std(x[section]) > 1e-10:
            hz, p = signal.periodogram(x[section], fs=fps, window="hann")
            band = (hz >= .5) & (hz <= 5)
            peaks.append((np.nanmedian(m), hz[band][p[band].argmax()]))
    shift = abs(max(peaks)[1] - min(peaks)[1]) if len(peaks) >= 2 and max(peaks)[0] > min(peaks)[0] else np.nan
    result.update(amplitude_activity_r=zero, derivative_velocity_r=correlation(abs(np.diff(x)) * fps, velocity),
                  lagged_activity_r=coefficient, lag_s=lag / fps,
                  high_low_activity_amplitude_log_ratio=ratio, motion_peak_shift_hz=shift)
    return result


def describe(regional, wave, ids, provenance, scores, motion, fps):
    run = longest_run(np.isfinite(wave), provenance)
    x = wave[run]
    time, freq = temporal(x, fps), spectral(x, fps)
    pairwise = [pair_features(regional[:, i], regional[:, j], ids[:, [i, j]], fps)[0]
                for i, j in combinations(range(regional.shape[1]), 2)]
    regional_activity = []
    for channel, groups in zip(regional.T, ids.T, strict=True):
        segment = channel[longest_run(np.isfinite(channel), groups)]
        if len(segment) >= round(2 * fps):
            regional_activity.append(segment.var())
    time["hjorth_activity"] = median(regional_activity)
    time["regional_correlation"] = median([p["correlation"] for p in pairwise])
    freq["regional_coherence"] = median([p["coherence"] for p in pairwise])
    return {"temporal": time, "spectral": freq, "cross": cross_modal(wave, provenance, scores, motion, fps),
            "missingness": dict(wave_coverage=np.isfinite(wave).mean(),
                                joint_coverage=np.mean(np.isfinite(wave) & np.isfinite(scores).all(axis=1)))}


def synthetic_control(variant, fps=30):
    t = np.arange(8 * fps) / fps
    pulse = np.sin(2 * np.pi * 1.25 * t)
    motion = np.sin(2 * np.pi * .55 * t)
    nuisance = np.column_stack((motion, np.zeros((len(t), 3))))
    ids = np.zeros((len(t), 3), int)
    base = np.array([110, 100, 90])
    clean = base + pulse[:, None] * [.25, 1.0, .15]
    contaminated = clean * (1 + .08 * np.sin(2 * np.pi * .18 * t[:, None]))
    contaminated += motion[:, None] * [1., 2.5, .4]
    contaminated[len(t) // 2] += [10, 17, 8]
    artifact_only = contaminated - pulse[:, None] * [.25, 1.0, .15] * (
        1 + .08 * np.sin(2 * np.pi * .18 * t[:, None]))
    waves = [waveform(np.repeat(x[:, None, :], 3, axis=1), ids, n, fps, variant)[1]
             for x, n in ((clean, np.zeros_like(nuisance)), (contaminated, nuisance), (artifact_only, nuisance))]
    def matched(w, reference):
        valid = np.isfinite(w)
        return abs(correlation(w[valid], reference[valid]))
    return dict(clean_pulse_abs_r=matched(waves[0], pulse), contaminated_pulse_abs_r=matched(waves[1], pulse),
                contamination_motion_abs_r=matched(waves[1], motion),
                artifact_only_motion_abs_r=matched(waves[2], motion),
                artifact_only_pulse_abs_r=matched(waves[2], pulse))



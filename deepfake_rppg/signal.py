"""Time-grid conversion, classical waveform extraction and quality measures."""

import numpy as np
from scipy import signal


BAND_HZ = (0.7, 3.0)
POS_WINDOW_SECONDS = 1.6
MIN_SEGMENT_SECONDS = 4.0
MAX_GAP_SECONDS = 0.25
MIN_QUALITY = 0.35


def finite_runs(valid, groups=None):
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for start, stop in edges.reshape(-1, 2):
        cuts = []
        if groups is not None:
            cuts = np.flatnonzero(np.diff(groups[start:stop])) + start + 1
        bounds = [start, *cuts, stop]
        for left, right in zip(bounds[:-1], bounds[1:], strict=True):
            yield slice(left, right)


def resample_rgb(times, rgb, fps, max_gap=MAX_GAP_SECONDS):
    """Interpolate within bounded gaps. Missing regions stay unobserved."""
    grid, result, _ = resample_observations(times, rgb, fps, max_gap=max_gap)
    return grid, result


def resample_observations(times, rgb, fps, groups=None, max_gap=MAX_GAP_SECONDS):
    """Return resampled colors and run IDs, with no interpolation across source changes."""
    grid = np.arange(int(np.floor(times[-1] * fps + 1e-8)) + 1) / fps
    result = np.full((len(grid), *rgb.shape[1:]), np.nan)
    grid_groups = np.zeros(result.shape[:2], dtype=int)
    for region in range(rgb.shape[1]):
        valid = np.isfinite(rgb[:, region]).all(axis=1)
        observed_times = times[valid]
        observed_rgb = rgb[valid, region]
        observed_groups = np.ones(len(observed_times), dtype=int) if groups is None else groups[valid, region]
        # Use the same timestamp tolerance as the grid endpoint comparisons.
        # For example, six 24 FPS intervals can round just above 0.25 s.
        breaks = np.flatnonzero(
            (np.diff(observed_times) > max_gap + 1e-9) | (np.diff(observed_groups) != 0)
        ) + 1
        for run_id, indices in enumerate(np.split(np.arange(len(observed_times)), breaks), start=1):
            if not len(indices):
                continue
            t = observed_times[indices]
            start = np.searchsorted(grid, t[0] - 1e-9)
            end = np.searchsorted(grid, t[-1] + 1e-9, side="right")
            grid_groups[start:end, region] = run_id
            for channel in range(3):
                result[start:end, region, channel] = np.interp(
                    grid[start:end], t, observed_rgb[indices, channel]
                )
    return grid, result, grid_groups


def pos(rgb, fps, groups=None):
    """Apply POS and average overlapping centered windows to normalize edge gain."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("POS expects N by 3 RGB samples")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("POS requires finite positive FPS")
    size = max(2, round(POS_WINDOW_SECONDS * fps))
    waveform = np.zeros(len(rgb))
    counts = np.zeros(len(rgb), dtype=int)
    for run in finite_runs(np.isfinite(rgb).all(axis=1), groups):
        for start in range(run.start, run.stop - size + 1):
            section = slice(start, start + size)
            window = rgb[section]
            mean = window.mean(axis=0)
            if np.any(mean <= 1e-8):
                continue
            red, green, blue = (window / mean).T
            first = green - blue
            second = green + blue - 2 * red
            spread = second.std()
            alpha = first.std() / spread if spread > 1e-8 else 0.0
            pulse = first + alpha * second
            waveform[section] += pulse - pulse.mean()
            counts[section] += 1
    return np.divide(waveform, counts, out=np.full(len(rgb), np.nan), where=counts > 0)


def process_signal(raw, fps, band=BAND_HZ, groups=None):
    """Filter valid runs and score peak concentration against pre-filter power."""
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(fps) or not 0 < band[0] < band[1] < fps / 2:
        raise ValueError("Bandpass requires finite FPS and 0 < low < high < Nyquist")
    sos = signal.butter(3, band, btype="bandpass", fs=fps, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    output = np.full_like(raw, np.nan)
    weighted_score = 0.0
    processed_count = 0
    for run in finite_runs(np.isfinite(raw), groups):
        segment = raw[run]
        if len(segment) < max(padlen + 1, np.ceil(MIN_SEGMENT_SECONDS * fps)):
            continue
        detrended = signal.detrend(segment, type="linear")
        filtered = signal.sosfiltfilt(sos, detrended, padlen=padlen)
        spread = filtered.std()
        if not np.isfinite(spread) or spread <= 1e-8:
            continue
        output[run] = (filtered - filtered.mean()) / spread
        nperseg = min(len(segment), round(8 * fps))
        frequencies, power = signal.welch(detrended, fs=fps, nperseg=nperseg)
        in_band = (frequencies >= band[0]) & (frequencies <= band[1])
        if np.any(in_band) and power[1:].sum() > 0:
            peak = frequencies[in_band][np.argmax(power[in_band])]
            neighborhood = in_band & (abs(frequencies - peak) <= fps / nperseg * 1.01)
            weighted_score += len(segment) * power[neighborhood].sum() / power[1:].sum()
        processed_count += len(segment)
    coverage = processed_count / len(raw) if len(raw) else 0.0
    score = float(np.clip(weighted_score / processed_count, 0, 1)) if processed_count else None
    return output, {"spectral_concentration": score, "processed_coverage": coverage}

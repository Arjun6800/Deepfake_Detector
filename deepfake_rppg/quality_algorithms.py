"""Classical rPPG challengers implemented from published mathematical formulations."""
from functools import lru_cache

import numpy as np
from scipy import signal

from .signal import BAND_HZ, MIN_SEGMENT_SECONDS, finite_runs, pos

ALGORITHMS = ("GREEN", "POS", "CHROM", "ICA", "LGI")


@lru_cache(maxsize=32)
def chrom_filter(fps):
    return signal.butter(3, BAND_HZ, btype="bandpass", fs=fps, output="sos")


def chrom(rgb, fps, filtered=False):
    """CHROM ratio from filtered projections, with a pre-filter counterpart for scoring."""
    size = round(1.6 * fps)
    if len(rgb) < size or size <= 21 or fps <= 2 * BAND_HZ[1]:
        return np.full(len(rgb), np.nan)
    last = len(rgb) - size
    starts = sorted({*range(0, last + 1, max(1, size // 2)), last})
    weights = np.hanning(size + 2)[1:-1]
    values, totals = np.zeros(len(rgb)), np.zeros(len(rgb))
    for start in starts:
        sl = slice(start, start + size)
        data = rgb[sl]
        mean = data.mean(axis=0)
        if np.any(mean <= 1e-8):
            continue
        red, green, blue = (data / mean - 1).T
        x, y = 3 * red - 2 * green, 1.5 * red + green - 1.5 * blue
        fx, fy = signal.sosfiltfilt(chrom_filter(fps), np.array([x, y]), axis=1)
        if fy.std() <= 1e-10:
            continue
        alpha = fx.std() / fy.std()
        component = fx - alpha * fy if filtered else x - alpha * y
        values[sl] += (component - component.mean()) * weights
        totals[sl] += weights
    return np.divide(values, totals, out=np.full(len(rgb), np.nan), where=totals > 0)


def lgi(rgb):
    """LGI's local first-singular-direction projection, using its green coordinate."""
    _, _, vt = np.linalg.svd(rgb, full_matrices=False)
    axis = vt[0]
    return (rgb - (rgb @ axis)[:, None] * axis)[:, 1]


def decorrelate(matrix):
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def independent_components(rgb):
    """Whiten and fit symmetric tanh FastICA with a fixed initialization."""
    centered = signal.detrend(rgb, axis=0)
    scale = centered.std(axis=0)
    if np.any(scale <= 1e-10):
        return np.empty((len(rgb), 0))
    u, singular, _ = np.linalg.svd(centered / scale, full_matrices=False)
    rank = np.count_nonzero(singular > 1e-7 * singular[0])
    if rank < 2:
        return np.empty((len(rgb), 0))
    whitened = u[:, :rank] * np.sqrt(len(rgb))
    weights = decorrelate(np.random.default_rng(17).normal(size=(rank, rank)))
    for _ in range(300):
        projected = whitened @ weights.T
        nonlinear = np.tanh(projected)
        updated = decorrelate(nonlinear.T @ whitened / len(rgb)
                              - (1 - nonlinear**2).mean(axis=0)[:, None] * weights)
        limit = np.max(np.abs(np.abs(np.diag(updated @ weights.T)) - 1))
        weights = updated
        if limit < 1e-6:
            return whitened @ weights.T
    return np.empty((len(rgb), 0))


def ica(rgb, fps):
    """Choose the component with maximal in-band peak power relative to all non-DC power."""
    components = independent_components(rgb)
    if not components.shape[1]:
        return np.full(len(rgb), np.nan)
    hz, power = signal.welch(components, fs=fps, axis=0, nperseg=min(len(rgb), round(8 * fps)))
    in_band = (hz >= BAND_HZ[0]) & (hz <= BAND_HZ[1])
    if not in_band.any():
        return np.full(len(rgb), np.nan)
    scores = power[in_band].max(axis=0) / np.maximum(power[1:].sum(axis=0), 1e-15)
    chosen = components[:, scores.argmax()]
    # Deterministic orientation against green, used only for waveform visualization.
    return chosen if np.dot(chosen, rgb[:, 1] - rgb[:, 1].mean()) >= 0 else -chosen


def project(method, rgb, fps, groups):
    if method == "GREEN":
        return rgb[:, 1].copy()
    if method == "POS":
        return pos(rgb, fps, groups)
    output = np.full(len(rgb), np.nan)
    for run in finite_runs(np.isfinite(rgb).all(axis=1), groups):
        if run.stop - run.start < np.ceil(MIN_SEGMENT_SECONDS * fps - 1e-9):
            continue
        section = rgb[run]
        if method.startswith("CHROM"):
            output[run] = chrom(section, fps, filtered=method == "CHROM_FILTERED")
        elif method == "ICA":
            output[run] = ica(section, fps)
        elif method == "LGI":
            output[run] = lgi(section)
        else:
            raise ValueError(f"Unknown method {method}")
    return output

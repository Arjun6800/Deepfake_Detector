"""OpenCV decoding and validation of the source timebase."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoTiming:
    timestamps: np.ndarray
    fps: float | None
    source: str


def choose_timing(position_seconds, nominal_fps, assume_cfr=False):
    """Keep decoder timing where usable and require opt-in for CFR fallback."""
    times = np.asarray(position_seconds, dtype=float)
    if (
        len(times) >= 2
        and np.isfinite(times).all()
        and times[0] >= 0
        and np.all(np.diff(times) > 0)
    ):
        fps = float(1 / np.median(np.diff(times)))
        if np.isfinite(fps):
            return VideoTiming(times - times[0], fps, "opencv_pos_msec")
    if assume_cfr and np.isfinite(nominal_fps) and nominal_fps > 0:
        return VideoTiming(np.arange(len(times)) / nominal_fps, nominal_fps, "assumed_cfr")
    return VideoTiming(np.full(len(times), np.nan), None, "unavailable")


def read_video(path: Path, extract):
    """Decode sequentially and retain only timestamps and extracted observations."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open {path}")
        nominal_fps = float(capture.get(cv2.CAP_PROP_FPS))
        backend = capture.getBackendName()
        positions, observations = [], []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            positions.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            observations.append(extract(frame))
        return np.asarray(positions), observations, nominal_fps, backend
    finally:
        capture.release()

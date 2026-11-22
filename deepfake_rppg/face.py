"""Single-subject tracking and geometric facial skin sampling."""

from pathlib import Path

import numpy as np


# Relative to the frontal face box. Keep away from eyes, mouth and jawline.
REGIONS = {
    "forehead": (0.29, 0.15, 0.71, 0.29),
    "cheek_image_left": (0.16, 0.51, 0.37, 0.68),
    "cheek_image_right": (0.63, 0.51, 0.84, 0.68),
}
BOX_UPDATE_WEIGHT = 0.35


def intersection_over_union(first, second):
    x, y, w, h = first
    a, b, u, v = second
    overlap = max(0, min(x + w, a + u) - max(x, a))
    overlap *= max(0, min(y + h, b + v) - max(y, b))
    return overlap / (w * h + u * v - overlap)


class FaceSampler:
    """Use the OpenCV distribution's bundled frontal-face cascade."""

    def __init__(self):
        import cv2

        self.cv2 = cv2
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(path))
        if self.detector.empty():
            raise ValueError(f"Could not load bundled face detector at {path}")
        self.previous = None
        self.detected_box = None
        self.sampling_box = None

    def locate(self, frame):
        """Update the target track without sampling any pixels."""
        previous_sampling_box = self.sampling_box
        self.detected_box = None
        self.sampling_box = None
        if frame is None or frame.size == 0:
            return None
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        if self.previous is None and len(boxes) == 1:
            self.previous = boxes[0]
            box = self.previous
        elif self.previous is not None:
            matches = [box for box in boxes if intersection_over_union(box, self.previous) >= 0.3]
            if len(matches) != 1:
                return None
            box = self.previous = matches[0]
        else:
            return None

        self.detected_box = box
        self.sampling_box = np.asarray(box, dtype=float)
        if previous_sampling_box is not None:
            self.sampling_box = (
                BOX_UPDATE_WEIGHT * self.sampling_box
                + (1 - BOX_UPDATE_WEIGHT) * previous_sampling_box
            )
        return box

    def __call__(self, frame):
        self.locate(frame)
        return self.sample_located(frame)

    def sample_located(self, frame):
        """Sample the current track without running detection again."""
        rgb = np.full((len(REGIONS), 3), np.nan)
        coverage = np.zeros(len(REGIONS))
        if self.sampling_box is None:
            return rgb, coverage, False
        x, y, width, height = self.sampling_box
        rows, cols = frame.shape[:2]
        for i, (left, top, right, bottom) in enumerate(REGIONS.values()):
            x0, x1 = round(x + left * width), round(x + right * width)
            y0, y1 = round(y + top * height), round(y + bottom * height)
            roi = frame[max(0, y0):min(rows, y1), max(0, x0):min(cols, x1)]
            if roi.size == 0:
                continue
            # Screen exposure extremes without imposing a fixed skin color range.
            valid = np.all((roi > 4) & (roi < 251) & np.isfinite(roi), axis=2)
            coverage[i] = valid.sum() / ((y1 - y0) * (x1 - x0))
            if coverage[i] >= 0.5 and valid.sum() >= 16:
                rgb[i] = roi[valid, ::-1].mean(axis=0)
        return rgb, coverage, True

    def sample_with_geometry(self, frame):
        """Include the detected box and the box used for RGB sampling."""
        self.locate(frame)
        return self.sample_located_with_geometry(frame)

    def sample_located_with_geometry(self, frame):
        rgb, coverage, detected = self.sample_located(frame)
        geometry = (
            np.concatenate((self.detected_box, self.sampling_box))
            if detected else np.full(8, np.nan)
        )
        return rgb, coverage, detected, geometry

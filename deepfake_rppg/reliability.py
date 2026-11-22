"""Pose-tolerant cheek sampling and per-frame ROI provenance."""

from dataclasses import dataclass

import numpy as np

from .face import REGIONS
from .landmarks import (
    SKIN_POLYGONS, LandmarkSampler, allowed_skin, mask_crop, measure_masks, polygon_mask, skin_masks,
)


RELIABLE_POLYGONS = {
    **SKIN_POLYGONS,
    "cheek_image_left": (116, 117, 118, 119, 100, 142, 203, 206, 216, 212, 214,
                         192, 215, 177, 137),
    "cheek_image_right": (345, 346, 347, 348, 329, 371, 423, 426, 436, 432, 434,
                          416, 435, 401, 366),
}
LANDMARK_UPDATE_WEIGHT = 0.55
MAX_NORMALIZED_STEP = 0.035
FALLBACK_MARGIN_FRACTION = 0.05
METHODS = ("rectangular", "landmark-legacy", "landmark", "hybrid")
SAMPLING_VERTICES = sorted({index for polygon in RELIABLE_POLYGONS.values() for index in polygon})


class LandmarkStabilizer:
    """Smooth small shape changes in box-relative coordinates."""

    def reset(self):
        self.previous_raw = self.previous = None

    def __init__(self):
        self.reset()

    def update(self, points, box):
        relative = (points - box[:2]) / box[2:]
        filtered = relative
        if self.previous_raw is not None:
            # Inspect the sampling vertices. Rigid translation and scale use the current box.
            step = np.linalg.norm(relative[SAMPLING_VERTICES] - self.previous_raw[SAMPLING_VERTICES], axis=1)
            if np.max(step) <= MAX_NORMALIZED_STEP:
                filtered = LANDMARK_UPDATE_WEIGHT * relative + (1 - LANDMARK_UPDATE_WEIGHT) * self.previous
        self.previous_raw, self.previous = relative, filtered
        return filtered * box[2:] + box[:2]


def reliable_masks(points, stabilized, shape, face_width, rectangle_masks=None):
    """Limit smoothed regions and fallback rectangles with the current landmarks."""
    import cv2

    allowed = allowed_skin(points, shape)
    margin = max(1, round(face_width * FALLBACK_MARGIN_FRACTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
    primary, fallback = [], []
    for i, indices in enumerate(RELIABLE_POLYGONS.values()):
        current = polygon_mask(points[list(indices)], shape)
        nearby = cv2.dilate(current, kernel) & allowed
        primary.append(polygon_mask(stabilized[list(indices)], shape) & nearby)
        if rectangle_masks is not None:
            fallback.append(rectangle_masks[i] & nearby)
    return np.asarray(primary, dtype=bool), np.asarray(fallback, dtype=bool)


def rectangular_masks(shape, origin, box):
    masks = np.zeros((len(REGIONS), *shape), dtype=bool)
    if box is not None:
        x, y, width, height = box
        rows, cols = shape
        for mask, (left, top, right, bottom) in zip(masks, REGIONS.values(), strict=True):
            x0, x1 = np.clip([round(x + left * width) - origin[0],
                              round(x + right * width) - origin[0]], 0, cols)
            y0, y1 = np.clip([round(y + top * height) - origin[1],
                              round(y + bottom * height) - origin[1]], 0, rows)
            mask[y0:y1, x0:x1] = True
    return masks


@dataclass
class RegionMeasurements:
    rgb: np.ndarray
    coverage: np.ndarray
    area: np.ndarray
    valid_area: np.ndarray
    centers: np.ndarray

    @classmethod
    def empty(cls):
        size = len(REGIONS)
        return cls(np.full((size, 3), np.nan), np.zeros(size), np.zeros(size, dtype=int),
                   np.zeros(size, dtype=int), np.full((size, 2), np.nan))

    @classmethod
    def from_masks(cls, crop, masks, origin):
        return cls(*measure_masks(crop, masks, origin))

    def failures(self, face_status):
        if face_status != "accepted":
            return [face_status] * len(REGIONS)
        return np.select(
            [self.area == 0, self.area < 16, ~np.isfinite(self.rgb[:, 0])],
            ["empty_geometry", "insufficient_mask_pixels", "insufficient_exposure_valid_pixels"],
            default="valid",
        )


def choose_fallback(primary, fallback):
    replace = ~np.isfinite(primary.rgb[:, 0]) & np.isfinite(fallback.rgb[:, 0])
    measured = RegionMeasurements(
        np.where(replace[:, None], fallback.rgb, primary.rgb),
        np.where(replace, fallback.coverage, primary.coverage),
        np.where(replace, fallback.area, primary.area),
        np.where(replace, fallback.valid_area, primary.valid_area),
        np.where(replace[:, None], fallback.centers, primary.centers),
    )
    source = np.where(replace, 2, np.isfinite(primary.rgb[:, 0]).astype(int))
    return measured, source


class ReliabilitySampler(LandmarkSampler):
    """Evaluate all modes with one decoded frame and one landmark inference."""

    def __init__(self, model_path):
        super().__init__(model_path)
        self.stabilizer = LandmarkStabilizer()

    def sample_methods(self, frame):
        if frame is None or frame.size == 0:
            self.target.locate(frame)
        face = self.select_face(frame)
        rectangular = self.target.sample_located_with_geometry(frame)
        legacy = primary = fallback = RegionMeasurements.empty()
        box = np.full(8, np.nan)
        nose_offset = np.nan
        if face is None:
            self.stabilizer.reset()
        else:
            points, oval_box = face
            stabilized = self.stabilizer.update(points, oval_box)
            box[:4] = oval_box
            crop, origin = mask_crop(frame, oval_box)
            if crop.size:
                shape = crop.shape[:2]
                rectangles = rectangular_masks(shape, origin, self.target.sampling_box)
                primary_masks, fallback_masks = reliable_masks(
                    points - origin, stabilized - origin, shape, oval_box[2], rectangles,
                )
                legacy = RegionMeasurements.from_masks(crop, skin_masks(points - origin, shape), origin)
                primary = RegionMeasurements.from_masks(crop, primary_masks, origin)
                fallback = RegionMeasurements.from_masks(crop, fallback_masks, origin)
                nose_offset = 2 * (points[1, 0] - oval_box[0]) / oval_box[2] - 1
        hybrid, sources = choose_fallback(primary, fallback)
        observed = {"rectangular": rectangular}
        diagnostics = {}
        for method, measurement in zip(METHODS[1:], (legacy, primary, hybrid), strict=True):
            codes = sources if method == "hybrid" else np.isfinite(measurement.rgb[:, 0]).astype(int)
            geometry = box.copy()
            if method == "hybrid" and (codes == 2).any():
                geometry[4:] = self.target.sampling_box
            observed[method] = (measurement.rgb, measurement.coverage, face is not None, geometry)
            diagnostics[method] = (self.face_status, measurement, codes, nose_offset)
        return observed, diagnostics

"""MediaPipe landmark polygons for forehead and cheek RGB sampling."""

from pathlib import Path

import numpy as np

from .face import REGIONS, FaceSampler, intersection_over_union


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
CROP_PADDING = 0.25
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
SKIN_POLYGONS = {
    "forehead": (103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 9, 108, 69, 104),
    "cheek_image_left": (116, 117, 118, 101, 205, 187, 123),
    "cheek_image_right": (345, 346, 347, 330, 425, 411, 352),
}
FEATURES = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
     267, 0, 37, 39, 40, 185),
)


def polygon_mask(points, shape, hull=False):
    import cv2

    vertices = np.rint(points).astype(np.int32)
    if hull:
        vertices = cv2.convexHull(vertices)
    mask = np.zeros(shape, dtype=np.uint8)
    if cv2.contourArea(vertices) > 0:
        cv2.fillPoly(mask, [vertices], 1)
    return mask


def allowed_skin(points, shape):
    import cv2

    oval = points[list(FACE_OVAL)]
    margin = max(1, round(np.ptp(oval[:, 0]) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
    allowed = cv2.erode(polygon_mask(oval, shape), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    excluded = np.zeros(shape, dtype=np.uint8)
    for feature in FEATURES:
        excluded |= polygon_mask(points[list(feature)], shape, hull=True)
    return allowed & (1 - cv2.dilate(excluded, kernel))


def skin_masks(points, shape):
    """Phase 1.3 polygons constrained to the current facial silhouette."""
    allowed = allowed_skin(points, shape)
    return np.asarray([polygon_mask(points[list(indices)], shape) & allowed
                       for indices in SKIN_POLYGONS.values()], dtype=bool)


def mask_crop(frame, box):
    height, width = frame.shape[:2]
    x0, y0 = np.clip(np.floor(box[:2]).astype(int) - 1, 0, [width, height])
    x1, y1 = np.clip(np.ceil(box[:2] + box[2:]).astype(int) + 2, 0, [width, height])
    return frame[y0:y1, x0:x1], np.array([x0, y0])


def measure_masks(crop, masks, origin):
    """Return RGB, exposure fraction, mask area, valid area and pixel centroids."""
    rgb = np.full((len(REGIONS), 3), np.nan)
    coverage = np.zeros(len(REGIONS))
    area = np.zeros(len(REGIONS), dtype=int)
    valid_area = area.copy()
    centers = np.full((len(REGIONS), 2), np.nan)
    for i, mask in enumerate(masks):
        y, x = np.nonzero(mask)
        area[i] = len(x)
        if area[i]:
            pixels = crop[y, x]
            valid = pixels[np.all((pixels > 4) & (pixels < 251) & np.isfinite(pixels), axis=1)]
            valid_area[i] = len(valid)
            coverage[i] = valid_area[i] / area[i]
            centers[i] = [x.mean() + origin[0], y.mean() + origin[1]]
            if coverage[i] >= 0.5 and valid_area[i] >= 16:
                rgb[i] = valid[:, ::-1].mean(axis=0)
    return rgb, coverage, area, valid_area, centers


class LandmarkSampler:
    """Run independent image inference and associate one face by oval-box overlap."""

    def __init__(self, model_path, *, output_blendshapes=False):
        import cv2
        import mediapipe as mp

        self.cv2 = cv2
        self.mp = mp
        self.previous = None
        self.blendshapes = ()
        self.target = FaceSampler()
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=2,
                output_face_blendshapes=output_blendshapes,
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detector.close()

    def detect(self, frame):
        box = self.target.locate(frame)
        if box is None:
            self.face_status = "face_tracking"
            return []
        x, y, width, height = box
        rows, cols = frame.shape[:2]
        pad_x, pad_y = round(width * CROP_PADDING), round(height * CROP_PADDING)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        crop = frame[y0:min(rows, y + height + pad_y), x0:min(cols, x + width + pad_x)]
        height, width = crop.shape[:2]
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2RGB),
        )
        result = self.detector.detect(image)
        if len(result.face_landmarks) == len(result.face_blendshapes) == 1:
            self.blendshapes = result.face_blendshapes[0]
        return [np.asarray([(x0 + point.x * width, y0 + point.y * height) for point in face])
                for face in result.face_landmarks]

    def select_face(self, frame):
        self.blendshapes = ()
        self.face_status = "face_tracking"
        if frame is None or frame.size == 0:
            return None
        self.face_status = "landmark_inference"
        faces = self.detect(frame)
        if len(faces) != 1:
            return None
        points = faces[0]
        if len(points) < 468 or not np.isfinite(points).all():
            return None
        oval = points[list(FACE_OVAL)]
        origin = oval.min(axis=0)
        size = oval.max(axis=0) - origin
        if np.any(size <= 0):
            return None
        box = np.r_[origin, size]
        if self.previous is not None and intersection_over_union(box, self.previous) < 0.3:
            self.face_status = "landmark_association"
            return None
        self.previous = box
        self.face_status = "accepted"
        return points, box

    def sample_with_geometry(self, frame):
        rgb = np.full((len(REGIONS), 3), np.nan)
        coverage = np.zeros(len(REGIONS))
        face = self.select_face(frame)
        if face is None:
            return rgb, coverage, False, np.full(8, np.nan)
        points, box = face
        crop, origin = mask_crop(frame, box)
        if crop.size:
            masks = skin_masks(points - origin, crop.shape[:2])
            rgb, coverage, _, _, _ = measure_masks(crop, masks, origin)
        return rgb, coverage, True, np.r_[box, np.full(4, np.nan)]

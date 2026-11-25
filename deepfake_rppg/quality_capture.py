"""Read-only, training-only capture of RGB and nuisance observations for Phase 4.5."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from .dataset import METHODS, save_json
from .face import FaceSampler
from .fusion import digest
from .landmarks import DEFAULT_MODEL, allowed_skin, mask_crop, polygon_mask
from .reliability import (
    FALLBACK_MARGIN_FRACTION, RELIABLE_POLYGONS, ReliabilitySampler, RegionMeasurements,
    choose_fallback, rectangular_masks,
)
from .video import choose_timing, read_video

# Stable nose bridge and outside eye corners. These anchors exclude lips and jaw.
ANCHORS = [33, 263, 6, 168]


def select(rows, seed, pairs, include_long=False):
    groups = defaultdict(list)
    for row in rows:
        if row["partition"] == "train" and row["validation_status"] in ("audited", "metadata_warning"):
            groups[row["group_id"]].append(row)
    eligible = []
    for group in sorted(groups):
        items = groups[group]
        originals = [row for row in items if row["label"] == 0]
        sources = {source for row in originals for source in row["source_video_ids"]}
        if len(originals) != 2 or len(sources) != 2:
            continue
        if (len(items) == 12 and all(sum(row["manipulation_method"] == method
                                        and set(row["source_video_ids"]) == sources for row in items) == 2
                                     for method in METHODS)):
            eligible.append(sorted(items, key=lambda row: (row["label"], row["video_id"])))
    if pairs < 1 or len(eligible) < pairs:
        raise ValueError("Insufficient complete training pairs")
    rng = random.Random(seed)
    reserved = []
    if include_long:
        candidates = [group for group in eligible if all((row["duration"] or 0) >= 30 for row in group)]
        if not candidates:
            raise ValueError("No complete 30-second training pair")
        reserved = [rng.choice(candidates)]
    chosen = reserved + rng.sample([group for group in eligible if group not in reserved], pairs - len(reserved))
    # Try one ordinary, complete pair first, then the duration stratum.
    if reserved and len(chosen) > 1:
        chosen[0], chosen[1] = chosen[1], chosen[0]
    return [row for pair in chosen for row in pair]


def skin_mask(bgr):
    """Conservative broad chroma mask, evaluated only inside geometric skin regions."""
    import cv2

    y, cr, cb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).transpose(2, 0, 1)
    return (y >= 24) & (cr >= 128) & (cr <= 180) & (cb >= 72) & (cb <= 138)


def rigid_align(template, points):
    """Fit a 2D similarity transform from a fixed landmark template to the current frame."""
    x, y = template[ANCHORS], points[ANCHORS]
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x0, y0 = x - x_mean, y - y_mean
    u, singular, vt = np.linalg.svd(x0.T @ y0)
    signs = np.array([1, np.linalg.det(u @ vt)])
    power = np.sum(x0**2)
    if power < 1e-8:
        return points
    matrix = ((u * signs) @ vt) * (singular @ signs / power)
    return (template - x_mean) @ matrix + y_mean


class ReducedDetector(FaceSampler):
    """Run target detection on a smaller image, then map its box to native pixels."""

    def __init__(self, width):
        super().__init__()
        self.width = width

    def locate(self, frame):
        if frame is None or frame.shape[1] <= self.width:
            return super().locate(frame)
        height, width = frame.shape[:2]
        size = (self.width, round(height * self.width / width))
        small = self.cv2.resize(frame, size, interpolation=self.cv2.INTER_AREA)
        # The tracker stores boxes in detector coordinates. Expose the sampling box
        # in native coordinates until the next call.
        scale = np.array([width / size[0], height / size[1]] * 2)
        if self.sampling_box is not None:
            self.sampling_box /= scale
        box = super().locate(small)
        if box is None:
            return None
        self.sampling_box *= scale
        self.detected_box = np.rint(box * scale).astype(int)
        return self.detected_box


class ResearchSampler(ReliabilitySampler):
    def __init__(self, model, detector_width=0):
        super().__init__(model)
        if detector_width:
            self.target = ReducedDetector(detector_width)
        self.template = None
    def observe(self, frame):
        import cv2

        face = self.select_face(frame)
        baseline = skin = rigid = RegionMeasurements.empty()
        source = skin_codes = rigid_codes = np.zeros(3, dtype=int)
        nose = np.nan
        geometry = np.full(4, np.nan)
        height, width = frame.shape[:2]
        bounds = self.target.sampling_box
        # Use surroundings as an illumination nuisance proxy, avoiding pixels in the tracked face box.
        step = max(2, min(height, width) // 48)
        lattice = frame[::step, ::step]
        outside = np.ones(lattice.shape[:2], dtype=bool)
        if bounds is not None:
            x, y, w, h = np.rint(bounds).astype(int)
            outside[max(0, y // step):max(0, (y + h + step - 1) // step),
                    max(0, x // step):max(0, (x + w + step - 1) // step)] = False
        background = lattice[outside]
        light = float(np.median(background @ np.array([.114, .587, .299]))) if len(background) else np.nan
        if face is None:
            self.stabilizer.reset()
        else:
            points, geometry = face
            stable = self.stabilizer.update(points, geometry)
            nose = 2 * (points[1, 0] - geometry[0]) / geometry[2] - 1
            crop, origin = mask_crop(frame, geometry)
            if crop.size:
                if self.template is None:
                    self.template = points.copy()
                shape = crop.shape[:2]
                rectangles = rectangular_masks(shape, origin, self.target.sampling_box)
                allowed = allowed_skin(points - origin, shape)
                margin = max(1, round(geometry[2] * FALLBACK_MARGIN_FRACTION))
                nearby = []
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
                for polygon in RELIABLE_POLYGONS.values():
                    current = polygon_mask(points[list(polygon)] - origin, shape)
                    nearby.append(bounded_dilate(current, kernel, margin) & allowed)
                main = constrained_masks(stable - origin, shape, nearby)
                fallback = rectangles & np.asarray(nearby, dtype=bool)
                fallback_measurement = RegionMeasurements.from_masks(crop, fallback, origin)
                baseline, source = choose_fallback(RegionMeasurements.from_masks(crop, main, origin),
                                                   fallback_measurement)
                pixels = skin_mask(crop)
                skin, skin_codes = choose_fallback(RegionMeasurements.from_masks(crop, main & pixels, origin),
                                                   RegionMeasurements.from_masks(crop, fallback & pixels, origin))
                aligned = rigid_align(self.template, points)
                rigid_main = constrained_masks(aligned - origin, shape, nearby)
                rigid, rigid_codes = choose_fallback(RegionMeasurements.from_masks(crop, rigid_main, origin),
                                                     fallback_measurement)
        return (baseline.rgb, skin.rgb, rigid.rgb, source, skin_codes, rigid_codes, geometry,
                nose, light, baseline.area, skin.area, baseline.centers, baseline.valid_area, self.face_status)


def bounded_dilate(mask, kernel, margin):
    """Restrict dilation to the only rectangle where the result can be nonzero."""
    import cv2

    x, y, w, h = cv2.boundingRect(mask)
    if not w or not h:
        return mask
    top, left = max(0, y - margin), max(0, x - margin)
    bottom, right = min(mask.shape[0], y + h + margin), min(mask.shape[1], x + w + margin)
    output = np.zeros_like(mask)
    output[top:bottom, left:right] = cv2.dilate(mask[top:bottom, left:right], kernel)
    return output


def constrained_masks(points, shape, nearby):
    return np.asarray([polygon_mask(points[list(polygon)], shape) & allowed
                       for polygon, allowed in zip(RELIABLE_POLYGONS.values(), nearby, strict=True)], dtype=bool)


def cached(path, key):
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["key"]) == key and len(data["positions"]) == len(data["baseline"])
    except (OSError, ValueError, KeyError):
        return False


def capture(row, directory, model, revision, detector_width):
    import cv2
    cv2.setNumThreads(1)
    start = perf_counter()
    source = Path(row["absolute_path"])
    version = sha256((row["sha256"] + revision).encode()).hexdigest()
    output = directory / f"{sha256(row['video_id'].encode()).hexdigest()}.npz"
    try:
        if digest(source) != row["sha256"]:
            raise ValueError("Video differs from the audited manifest hash")
        if cached(output, version):
            return dict(video_id=row["video_id"], status="cached", archive=output.name)
        with ResearchSampler(model, detector_width) as sampler:
            positions, observations, nominal, backend = read_video(source, sampler.observe)
        timing = choose_timing(positions, nominal)
        if not observations or timing.fps is None:
            raise ValueError("No usable decoded timestamps")
        columns = ("baseline", "skin", "rigid", "base_codes", "skin_codes", "rigid_codes", "geometry",
                   "nose", "light", "area", "skin_area", "centers", "valid_area", "face_status")
        arrays = {key: np.asarray(values) for key, values in zip(columns, zip(*observations, strict=True), strict=True)}
        with NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as stream:
            staged = Path(stream.name)
            np.savez_compressed(stream, **arrays, positions=positions, times=timing.timestamps, fps=timing.fps,
                                nominal=nominal, backend=backend, key=version, seconds=perf_counter() - start)
        try:
            staged.replace(output)
        finally:
            staged.unlink(missing_ok=True)
        return dict(video_id=row["video_id"], status="captured", archive=output.name, frames=len(positions),
                    seconds=perf_counter() - start)
    except Exception as error:
        return dict(video_id=row["video_id"], status="error", error=f"{type(error).__name__}: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/test7/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test8"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=451)
    parser.add_argument("--detector-width", type=int, default=0)
    parser.add_argument("--include-long", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("outputs/test8").resolve()):
        parser.error("Capture output must be within outputs/test8")
    rows = select(json.loads(args.manifest.read_text(encoding="utf-8")), args.seed, args.pairs, args.include_long)
    directory = output / "observations"
    directory.mkdir(parents=True, exist_ok=True)
    save_json(output / "selection.json", rows)
    revision = "".join(digest(Path(__file__).with_name(name)) for name in
                       ("quality_capture.py", "reliability.py", "landmarks.py", "face.py", "video.py")) + digest(args.model)
    revision += f":{args.detector_width}"
    save_json(output / "capture_settings.json", dict(seed=args.seed, pairs=args.pairs, workers=args.workers,
                                                       detector_width=args.detector_width, include_long=args.include_long,
                                                       model_sha256=digest(args.model), revision=revision))
    elapsed_start = perf_counter()
    results = []
    with (output / "capture_progress.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(capture, row, directory, str(args.model.resolve()), revision,
                                   args.detector_width): row for row in rows}
            for i, task in enumerate(as_completed(futures), 1):
                try:
                    item = task.result()
                except Exception as error:
                    item = dict(video_id=futures[task]["video_id"], status="error", error=str(error))
                results.append(item)
                print(json.dumps(item), file=log)
                elapsed = perf_counter() - elapsed_start
                print(f"{i}/{len(rows)} {item['video_id']} {item['status']} ETA {elapsed / i * (len(rows) - i):.0f}s",
                      flush=True)
    save_json(output / "capture_summary.json", dict(seconds=perf_counter() - elapsed_start,
                                                     results=sorted(results, key=lambda item: item["video_id"])))


if __name__ == "__main__":
    main()

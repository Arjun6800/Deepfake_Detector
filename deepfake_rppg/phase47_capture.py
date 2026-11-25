"""Re-decode the fixed training cohort for aligned action-unit features."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from .batch import WINDOW_KEYS
from .consistency_experiment import CAPTURE
from .expressions import ExpressionSampler, SELECTED
from .fusion import digest
from .landmarks import DEFAULT_MODEL
from .quality_capture import ReducedDetector
from .video import read_video
from .window_features import expression_features

OUTPUT = Path("outputs/test10")
PHASE46 = Path("outputs/test9")
CACHE = OUTPUT / "expression_checkpoints"
KEYS = (*WINDOW_KEYS, "partition")


def verify_timeline(actual, expected, first, last, start, end):
    if not np.array_equal(actual, expected):
        raise ValueError("Decoded positions differ from the rPPG capture")
    times = actual - actual[0]
    indices = np.flatnonzero((times >= start - 1e-8) & (times < end - 1e-8))
    if not np.array_equal(indices, np.arange(first, last + 1)):
        raise ValueError("Frame indices disagree with the archived time window")
    return times[indices]


def capture(task):
    import cv2

    item, window, settings, revision = task
    cv2.setNumThreads(1)
    started = perf_counter()
    token = sha256(item["video_id"].encode()).hexdigest()
    path = Path(item["absolute_path"])
    checkpoint = CACHE / f"{token}.json"
    keys = {key: window[key] for key in KEYS}
    if digest(path) != item["sha256"]:
        raise ValueError(f"Source hash mismatch for {item['video_id']}")
    if checkpoint.is_file():
        row = json.loads(checkpoint.read_text(encoding="utf-8"))
        if row.get("revision") == revision and row.get("source_sha256") == item["sha256"] and row.get("keys") == keys:
            return row
    with np.load(CAPTURE / "observations" / f"{token}.npz", allow_pickle=False) as archive:
        key = sha256((item["sha256"] + settings["revision"]).encode()).hexdigest()
        if str(archive["key"]) != key:
            raise ValueError("Wrong rPPG capture provenance")
        positions, fps = archive["positions"], round(float(archive["fps"]), 12)
        timing = archive["times"]
    first, last = int(window["first_frame_index"]), int(window["last_frame_index"])
    count = 0
    with ExpressionSampler(DEFAULT_MODEL) as sampler:
        sampler.target = ReducedDetector(settings["detector_width"])

        def observe(frame):
            nonlocal count
            index = count
            count += 1
            if index > last:
                return None
            _, scores = sampler.sample(frame)
            if index >= first:
                return [scores.get(name, np.nan) for name in SELECTED]
            return None

        actual, observations, _, backend = read_video(path, observe)
    start, end = float(window["window_start_s"]), float(window["window_end_s_exclusive"])
    times = verify_timeline(actual, positions, first, last, start, end)
    with np.load(PHASE46 / window["archive"], allow_pickle=False) as waveform:
        grid = waveform["time_s"]
        if (len(grid) != int(window["rppg_grid_samples"]) or start != grid[0]
                or not np.allclose(np.diff(grid), 1 / fps, rtol=0, atol=1e-12)):
            raise ValueError("Expanded waveform grid does not match recorded window")
    if not np.array_equal(times, timing[first:last + 1]) or len(times) != int(window["decoded_frames"]):
        raise ValueError("Expanded feature frame support does not match decoded time")
    scores = np.asarray(observations[first:last + 1], dtype=float)
    features = expression_features(times, scores, fps)
    if len(features) != 187 or any(np.isinf(value) for value in features.values()):
        raise ValueError("Invalid expression schema or values")
    data = dict(keys=keys, features={key: float(value) if np.isfinite(value) else None
                                   for key, value in features.items()},
                source_sha256=item["sha256"], revision=revision, aligned_frames=len(times), backend=backend,
                seconds=perf_counter() - started)
    with NamedTemporaryFile(dir=CACHE, mode="w", encoding="utf-8", delete=False) as stream:
        json.dump(data, stream, allow_nan=False)
        stage = Path(stream.name)
    stage.replace(checkpoint)
    return data


def load_inputs():
    import csv

    items = json.loads((CAPTURE / "selection.json").read_text(encoding="utf-8"))
    settings = json.loads((CAPTURE / "capture_settings.json").read_text(encoding="utf-8"))
    with (PHASE46 / "waveform_index.csv").open(newline="", encoding="utf-8") as stream:
        windows = {row["clip_id"]: row for row in csv.DictReader(stream)}
    if len(items) != 96 or len(windows) != 96 or {row["video_id"] for row in items} != windows.keys():
        raise ValueError("Expected the original 96 matched clips")
    if any(item["partition"] != "train" or windows[item["video_id"]]["partition"] != "train"
           or item["group_id"] != windows[item["video_id"]]["split_group"] for item in items):
        raise ValueError("Unexpected partition or source family")
    revision = "".join(digest(Path(__file__).with_name(name)) for name in
                       ("phase47_capture.py", "quality_capture.py", "landmarks.py", "face.py",
                        "video.py", "expressions.py", "window_features.py"))
    model_sha = digest(DEFAULT_MODEL)
    if model_sha != settings["model_sha256"]:
        raise ValueError("Expression model differs from Phase 4.6 capture")
    revision += model_sha + settings["revision"]
    return items, windows, settings, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True, parents=True)
    items, windows, settings, revision = load_inputs()
    tasks = [(item, windows[item["video_id"]], settings, revision) for item in items]
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(capture, tasks), 1):
            print(f"{index}/96 {row['keys']['clip_id']}", flush=True)
    (OUTPUT / "expression_capture_summary.json").write_text(
        json.dumps(dict(videos=len(tasks), seconds=perf_counter() - started, workers=args.workers), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()

"""Extract checkpointed Phase 4 feature tables from an audited manifest."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED

import numpy as np

from .dataset import DFD, METHODS, save_json
from .expressions import SELECTED, blendshape_scores
from .fusion import build_windows, digest
from .landmarks import DEFAULT_MODEL
from .signal import BAND_HZ
from .reliability import ReliabilitySampler
from .video import choose_timing, read_video

TABLES = ("rppg", "expression", "fused")
FEATURE_COUNTS = dict(zip(TABLES, (69, 187, 256), strict=True))
WINDOW_KEYS = ("clip_id", "split_group", "window_index", "window_start_s", "window_end_s_exclusive",
               "first_frame_index", "last_frame_index", "decoded_frames", "rppg_grid_samples")


def pilot(rows, seed):
    """Select five complete training pairs and four fakes per method."""
    groups = defaultdict(list)
    for row in rows:
        if row["partition"] == "train" and row["validation_status"] in ("audited", "metadata_warning"):
            groups[row["group_id"]].append(row)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    chosen = []
    for group_id in group_ids:
        group = groups[group_id]
        originals = [row for row in group if row["label"] == 0]
        if len(originals) != 2:
            continue
        ids = {source for row in originals for source in row["source_video_ids"]}
        target_methods = [METHODS[len(chosen) // 6], METHODS[(len(chosen) // 6 + 1) % len(METHODS)]]
        candidates = [[row for row in group if row["manipulation_method"] == method
                       and set(row["source_video_ids"]) == ids] for method in target_methods]
        if all(len(items) == 2 for items in candidates):
            chosen.extend(originals + candidates[0] + candidates[1])
        if len(chosen) == 30:
            return chosen
    raise ValueError("Could not find five complete training pairs for the requested pilot")


def sample(sampler, frame, roi):
    observed, diagnostics = sampler.sample_methods(frame)
    rgb = observed[roi][0]
    codes = diagnostics["hybrid"][2] if roi == "hybrid" else np.isfinite(rgb[:, 0]).astype(int)
    by_name = blendshape_scores(sampler.blendshapes)
    return rgb, codes, [by_name.get(name, np.nan) for name in SELECTED]


def roi_runs(codes):
    groups = np.zeros_like(codes)
    for i, column in enumerate(codes.T):
        observed = np.flatnonzero(column)
        if len(observed):
            groups[observed, i] = np.r_[0, np.cumsum(np.diff(column[observed]) != 0)]
    return groups


def table_csv(rows, columns):
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(row[key] if row[key] is not None and np.isfinite(row[key]) else "" for key in columns)
    return stream.getvalue()


def extract(row, settings):
    import cv2

    cv2.setNumThreads(1)
    video = Path(row["absolute_path"])
    if digest(video) != row["sha256"]:
        raise ValueError("Source SHA-256 changed since the dataset audit")
    with ReliabilitySampler(settings["model"], output_blendshapes=True) as sampler:
        positions, observations, nominal_fps, backend = read_video(
            video, lambda frame: sample(sampler, frame, settings["roi"]),
        )
    timing = choose_timing(positions, nominal_fps, settings["assume_cfr"])
    if len(observations) < 2 or timing.fps is None or timing.fps <= 2 * BAND_HZ[1]:
        raise ValueError("Video lacks a usable timestamp sequence")
    rgb, codes, scores = (np.asarray(values) for values in zip(*observations, strict=True))
    fps = round(timing.fps, 12)
    grid = np.arange(int(np.floor(timing.timestamps[-1] * fps + 1e-8)) + 1) / fps
    tables = build_windows(timing.timestamps, grid, rgb, roi_runs(codes), scores, fps,
                           settings["length"], settings["overlap"], row["video_id"], row["group_id"])
    frame_coverage = float(np.isfinite(scores).all(axis=1).mean())
    metrics = dict(status="complete" if tables[-1] else "no_windows", windows=len(tables[-1]),
                   decoded_frames=len(positions), timing=timing.source, backend=backend,
                   expression_frame_coverage=frame_coverage, flags=[])
    if len(positions) != row["frame_count"]:
        metrics["flags"].append("decoded_frame_count_differs")
    if frame_coverage < 1:
        metrics["flags"].append("incomplete_expression_coverage")
    files = {}
    if tables[-1]:
        columns = {name: [key for key in table[0] if key.startswith(("rppg_", "expr_")) and key not in WINDOW_KEYS]
                   for name, table in zip(TABLES, tables, strict=True)}
        if {name: len(fields) for name, fields in columns.items()} != FEATURE_COUNTS:
            raise ValueError("Extracted feature columns do not match the expected modality schemas")
        metrics["feature_columns"] = columns
        quality = [value for window in tables[0] for key, value in window.items()
                   if key.endswith("_spectral_concentration") and value is not None and np.isfinite(value)]
        passes = sum(value for window in tables[0] for key, value in window.items()
                     if key.endswith("_quality_pass"))
        metrics.update(quality_passes=passes, quality_checks=6 * len(tables[0]),
                       spectral_concentration=quality,
                       missing_features={name: sum(window[key] is None or not np.isfinite(window[key])
                                                   for window in table for key in columns[name])
                                         for name, table in zip(TABLES, tables, strict=True)})
        if passes < metrics["quality_checks"]:
            metrics["flags"].append("rppg_quality_gate_failed_on_some_traces")
        files = {f"{name}.csv": table_csv(table, columns[name]) for name, table in zip(TABLES, tables, strict=True)}
        keys = [{key: value for key, value in window.items() if key in WINDOW_KEYS}
                for window in tables[-1]]
        files["windows.json"] = json.dumps(keys, allow_nan=False)
    return metrics, files


def checkpoint_key(row, settings):
    path = Path(row["absolute_path"])
    try:
        stat = path.stat()
        file_state = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        file_state = None
    identity = (row, settings, file_state)
    return sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def restore(path, key, retry_failed):
    try:
        with ZipFile(path) as archive:
            metrics = json.loads(archive.read("metrics.json"))
            if archive.testzip() is not None or metrics["checkpoint_key"] != key:
                return None
            if metrics["status"] == "complete":
                expected = {"metrics.json", "windows.json", *(f"{name}.csv" for name in TABLES)}
                if set(archive.namelist()) != expected:
                    return None
            if metrics["status"] == "error" and retry_failed:
                return None
            return metrics
    except (OSError, BadZipFile, KeyError, ValueError, RuntimeError):
        return None


def commit(path, metrics, files):
    # A unique staging file and one atomic replace keep readers away from partial outputs.
    with NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as handle:
        staged = Path(handle.name)
    try:
        with ZipFile(staged, "w", compression=ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
            archive.writestr("metrics.json", json.dumps(metrics, allow_nan=False))
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def process(row, directory, settings, retry_failed):
    key = checkpoint_key(row, settings)
    path = directory / f"{sha256(row['video_id'].encode()).hexdigest()}.zip"
    previous = restore(path, key, retry_failed)
    if previous is not None:
        return {**previous, "resumed": True, "storage_bytes": path.stat().st_size}
    start = perf_counter()
    try:
        metrics, files = extract(row, settings)
    except Exception as error:
        metrics, files = {"status": "error", "error": f"{type(error).__name__}: {error}", "windows": 0}, {}
    metrics.update(checkpoint_key=key, video_id=row["video_id"], seconds=perf_counter() - start,
                   manipulation_method=row["manipulation_method"], partition=row["partition"])
    commit(path, metrics, files)
    return {**metrics, "resumed": False, "storage_bytes": path.stat().st_size}


def run(rows, output, settings, workers, retry_failed=False):
    output = output.resolve()
    for row in rows:
        source = Path(row["absolute_path"]).resolve()
        root = source.parents[len(Path(row["relative_path"]).parts) - 1]
        if output.is_relative_to(root):
            raise ValueError("Batch output must be outside dataset roots")
    directory = output / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(process, row, directory, settings, retry_failed): row for row in rows}
        for i, task in enumerate(as_completed(tasks), 1):
            try:
                result = task.result()
            except Exception as error:
                result = dict(video_id=tasks[task]["video_id"], status="worker_error", error=str(error), windows=0)
            results.append(result)
            elapsed = perf_counter() - start
            print(f"{i}/{len(rows)} {result['video_id']} {result['status']} "
                  f"windows={result['windows']} ETA={elapsed / i * (len(rows) - i):.0f}s", flush=True)
    summary = dict(wall_seconds=perf_counter() - start, workers=workers, settings=settings,
                   results=sorted(results, key=lambda row: row["video_id"]))
    save_json(output / "batch_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--partition", choices=("train", "val", "test"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--roi", choices=("hybrid", "landmark", "rectangular"), default="hybrid")
    parser.add_argument("--length", type=float, default=8)
    parser.add_argument("--overlap", type=float, default=4)
    parser.add_argument("--assume-cfr", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = pilot(rows, args.seed) if args.pilot else [
        row for row in rows if row["partition"] == args.partition and row["manipulation_method"] != DFD
        and row["validation_status"] in ("audited", "metadata_warning")
    ]
    if not rows:
        parser.error("No eligible manifest rows")
    if not np.isfinite([args.length, args.overlap]).all() or not 0 <= args.overlap < args.length or args.workers < 1:
        parser.error("Require positive workers and 0 <= overlap < length")
    settings = dict(model=str(args.model.resolve()), model_sha256=digest(args.model),
                    code_sha256={path.name: digest(path) for path in Path(__file__).parent.glob("*.py")},
                    roi=args.roi, length=args.length, overlap=args.overlap, assume_cfr=args.assume_cfr)
    run(rows, args.output, settings, args.workers, args.retry_failed)
    save_json(args.output / "selection.json", rows)


if __name__ == "__main__":
    main()

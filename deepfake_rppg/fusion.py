"""Build aligned, window-local rPPG and expression feature tables."""

import argparse
import csv
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from .expressions import MAX_STEP_PERIODS, MIN_WINDOW_SAMPLES, SELECTED, WINDOW_SECONDS, write_columns
from .face import REGIONS
from .signal import BAND_HZ, MIN_QUALITY, MIN_SEGMENT_SECONDS, POS_WINDOW_SECONDS
from .window_features import (
    MIN_EXPRESSION_SAMPLES, MIN_PROCESSED_COVERAGE, MIN_PULSE_SECONDS, expression_features,
    rppg_features, snap_rgb,
)


FRAME_KEYS = ("frame_index", "source_timestamp_s", "analysis_timestamp_s")
TIME_EPSILON = 1e-9


def numeric_csv(path, fields):
    return np.genfromtxt(path, delimiter=",", names=True, usecols=fields, ndmin=1, encoding="utf-8-sig")


def require_alignment(rppg, expression):
    n = len(rppg)
    if n < 2 or n != len(expression):
        raise ValueError("Frame exports must contain matching rows for at least two decoded frames")
    for rows in (rppg, expression):
        if not np.array_equal(rows["frame_index"], np.arange(n)):
            raise ValueError("Frame indices must be consecutive from zero")
        times = rows["analysis_timestamp_s"]
        if not np.isfinite(times).all() or abs(times[0]) > TIME_EPSILON or not np.all(np.diff(times) > 0):
            raise ValueError("Analysis timestamps must start at zero and increase strictly")
    for name in FRAME_KEYS[1:]:
        if not np.allclose(rppg[name], expression[name], rtol=0, atol=TIME_EPSILON, equal_nan=True):
            raise ValueError(f"Modality timestamp mismatch in {name}")


def window_bounds(end, length, overlap):
    if not np.isfinite([end, length, overlap]).all() or length <= 0 or not 0 <= overlap < length:
        raise ValueError("Require positive window length and 0 <= overlap < window length")
    step = length - overlap
    if step <= TIME_EPSILON:
        raise ValueError("Window step must be greater than timestamp tolerance")
    count = max(0, int(np.floor((end - length + TIME_EPSILON) / step)) + 1)
    for i in range(count):
        start = i * step
        yield start, start + length


def time_slice(times, start, end):
    left, right = np.searchsorted(times, [start - TIME_EPSILON, end - TIME_EPSILON])
    return slice(int(left), int(right))


def hybrid_sources(path, rgb):
    sources = np.full(rgb.shape[:2], -1, dtype=int)
    regions = tuple(REGIONS)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            index, region = int(row["frame_index"]), regions.index(row["region"])
            if not 0 <= index < len(rgb) or sources[index, region] != -1:
                raise ValueError("Invalid or duplicate hybrid provenance row")
            sources[index, region] = ("missing", "landmark", "fallback").index(row["source"])
    if np.any(sources == -1) or not np.array_equal(sources > 0, np.isfinite(rgb).all(axis=2)):
        raise ValueError("Hybrid source labels must match RGB observations")
    # Unique run IDs preserve a switch even if the switching frame misses the analysis grid.
    ids = np.zeros_like(sources)
    for i, column in enumerate(sources.T):
        observed = np.flatnonzero(column)
        ids[observed, i] = np.r_[0, np.cumsum(np.diff(column[observed]) != 0)] if len(observed) else []
    return ids


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_exports(video, rppg_dir, expression_dir):
    rppg_meta = rppg_dir / "metadata.json"
    expr_meta = expression_dir / "metadata.json"
    frame_file = rppg_dir / "frame_traces.csv"
    blendshape_file = expression_dir / "blendshapes.csv"
    raw_file = rppg_dir / "raw_waveform.csv"
    processed_file = rppg_dir / "processed_waveform.csv"
    paths = [rppg_meta, expr_meta, frame_file, blendshape_file, raw_file, processed_file]
    r_meta, e_meta = (json.loads(path.read_text(encoding="utf-8")) for path in (rppg_meta, expr_meta))
    for metadata in (r_meta, e_meta):
        input_name = metadata["input_video"].replace("\\", "/").rsplit("/", 1)[-1]
        if input_name != video.name:
            raise ValueError(f"Export identifies a different input video: {input_name}")
    fps = r_meta["fps"]
    if (fps is None or e_meta["fps"] is None or not np.isfinite([fps, e_meta["fps"]]).all()
            or fps <= 2 * BAND_HZ[1]
            or abs(fps - e_meta["fps"]) > TIME_EPSILON
            or r_meta["timestamp_source"] != e_meta["timestamp_source"]):
        raise ValueError("Exports must agree on a usable analysis timebase")
    fps = round(fps, 12)
    rgb_fields = tuple(f"{region}_{channel}" for region in REGIONS for channel in ("red", "green", "blue"))
    frames = numeric_csv(frame_file, (*FRAME_KEYS, *rgb_fields))
    expressions = numeric_csv(blendshape_file, (*FRAME_KEYS, *SELECTED))
    require_alignment(frames, expressions)
    if len(frames) != r_meta["decoded_frames"] or len(frames) != e_meta["decoded_frames"]:
        raise ValueError("Metadata frame counts must agree with both frame exports")
    times = frames["analysis_timestamp_s"]
    raw_grid, processed_grid = (numeric_csv(path, ("timestamp_s",))["timestamp_s"]
                                for path in (raw_file, processed_file))
    expected = np.arange(int(np.floor(times[-1] * fps + 1e-8)) + 1) / fps
    if raw_grid.shape != expected.shape or not np.allclose(raw_grid, expected, rtol=0, atol=TIME_EPSILON):
        raise ValueError("Raw rPPG analysis grid disagrees with frame timing")
    if processed_grid.shape != expected.shape or not np.allclose(processed_grid, expected, rtol=0, atol=TIME_EPSILON):
        raise ValueError("Processed rPPG analysis grid disagrees with frame timing")
    rgb = np.column_stack([frames[name] for name in rgb_fields]).reshape(-1, len(REGIONS), 3)
    scores = np.column_stack([expressions[name] for name in SELECTED])
    if r_meta["roi_method"] == "hybrid":
        paths.append(rppg_dir / "roi_diagnostics.csv")
        groups = hybrid_sources(paths[-1], rgb)
    else:
        groups = np.zeros(rgb.shape[:2], dtype=int)
    return times, expected, rgb, groups, scores, fps, paths, r_meta, e_meta


def build_windows(times, grid, rgb, groups, scores, fps, length, overlap, clip_id, split_group):
    rppg_rows, expression_rows, fused_rows = [], [], []
    if length * fps < 1:
        raise ValueError("Window duration must span at least one analysis sample")
    for i, (start, end) in enumerate(window_bounds(grid[-1] + 1 / fps, length, overlap)):
        frame = time_slice(times, start, end)
        grid_slice = time_slice(grid, start, end)
        native_times = times[frame]
        local_grid = grid[grid_slice]
        sampled, sampled_groups = snap_rgb(native_times, rgb[frame], groups[frame], local_grid, fps)
        keys = {
            "clip_id": clip_id, "split_group": split_group, "window_index": i,
            "window_start_s": start, "window_end_s_exclusive": end,
            "first_frame_index": frame.start if len(native_times) else None,
            "last_frame_index": frame.stop - 1 if len(native_times) else None,
            "decoded_frames": len(native_times), "rppg_grid_samples": len(local_grid),
        }
        rppg = rppg_features(rgb[frame], sampled, sampled_groups, fps)
        expression = expression_features(native_times, scores[frame], fps)
        rppg_rows.append({**keys, **rppg})
        expression_rows.append({**keys, **expression})
        fused_rows.append({**keys, **rppg, **expression})
    return rppg_rows, expression_rows, fused_rows


def write_table(path, rows):
    write_columns(path, {key: [row[key] for row in rows] for key in rows[0]})


def diagnostics(output, rppg_rows, expr_rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    traces = [f"rppg_{region}_{method}_" for region in REGIONS for method in ("green", "pos")]
    centers = np.array([(row["window_start_s"] + row["window_end_s_exclusive"]) / 2 for row in rppg_rows])
    labels = [name.removeprefix("rppg_").replace("_", " ").strip() for name in traces]
    figure = Figure(figsize=(11, 7), layout="constrained")
    axes = figure.subplots(2, sharex=True)
    for prefix, label in zip(traces, labels, strict=True):
        axes[0].plot(centers, [row[prefix + "processed_coverage"] for row in rppg_rows],
                     marker=".", label=label)
        values = [row[prefix + "spectral_concentration"] for row in rppg_rows]
        axes[1].plot(centers, np.asarray(values, dtype=float), marker=".", label=label)
    axes[0].axhline(MIN_PROCESSED_COVERAGE, ls=":", color="black")
    axes[1].axhline(MIN_QUALITY, ls=":", color="black")
    axes[0].set_ylim(-0.03, 1.07)
    axes[0].set_ylabel("Processed fraction")
    axes[1].set_ylabel("Spectral concentration")
    axes[1].set_xlabel("Window center (s)")
    axes[0].legend(loc="lower left", ncols=3, fontsize=8)
    FigureCanvasAgg(figure).print_png(output / "rppg_window_quality.png")

    figure = Figure(figsize=(11, 7), layout="constrained")
    axes = figure.subplots(3, sharex=True)
    for name in ("browInnerUp", "eyeBlinkLeft", "jawOpen", "mouthSmileLeft"):
        axes[0].plot(centers, [row[f"expr_{name}_mean"] for row in expr_rows], marker=".", label=name)
        axes[1].plot(centers, [row[f"expr_{name}_mean_abs_velocity_per_s"] for row in expr_rows],
                     marker=".", label=name)
    axes[2].plot(centers, [row["expr_all_selected_coverage"] for row in expr_rows], marker=".")
    axes[2].set_ylim(-0.03, 1.05)
    axes[0].legend(fontsize=8, ncols=4)
    axes[0].set_ylabel("Mean coefficient")
    axes[1].set_ylabel("Mean absolute velocity\n(score / s)")
    axes[2].set_ylabel("Complete score fraction")
    axes[2].set_xlabel("Window center (s)")
    FigureCanvasAgg(figure).print_png(output / "expression_windows.png")


def run(video, rppg_dir, expression_dir, output, length=8.0, overlap=4.0, split_group=None):
    times, grid, rgb, groups, scores, fps, paths, r_meta, e_meta = read_exports(video, rppg_dir, expression_dir)
    clip_id = digest(video)
    split_group = clip_id if split_group is None else split_group.strip()
    if not split_group:
        raise ValueError("Split group must be a nonempty source or subject identifier")
    tables = build_windows(times, grid, rgb, groups, scores, fps, length, overlap, clip_id, split_group)
    if not tables[-1]:
        raise ValueError("Input does not contain a complete window of the requested length")
    output.mkdir(parents=True, exist_ok=True)
    names = ("rppg_features", "expression_features", "fused_features")
    for name, rows in zip(names, tables, strict=True):
        write_table(output / f"{name}.csv", rows)
    keys = [name for name in tables[-1][0] if not name.startswith(("rppg_", "expr_")) or name == "rppg_grid_samples"]
    metadata = {
        "schema_version": 1, "input_video": str(video), "clip_id_sha256": clip_id, "split_group": split_group,
        "split_policy": "Group all windows by source family or subject before splitting. Exclude key columns from feature inputs.",
        "inputs": [{"path": str(path), "sha256": digest(path)} for path in paths],
        "alignment": {"frame_keys": FRAME_KEYS, "matched_rows": len(times), "fps": fps,
                      "timestamp_source": r_meta["timestamp_source"], "tolerance_s": TIME_EPSILON},
        "windowing": {"length_s": length, "overlap_s": overlap, "step_s": length - overlap,
                      "complete_windows_only": True, "interval": "[start,end)", "count": len(tables[-1]),
                      "uncovered_tail_s": max(0, grid[-1] + 1 / fps - tables[-1][-1]["window_end_s_exclusive"])},
        "rppg": {"roi_method": r_meta["roi_method"], "upstream_status": r_meta["status"],
                 "upstream_settings": r_meta["settings"], "band_hz": BAND_HZ,
                 "pos_window_s": POS_WINDOW_SECONDS, "maximum_snap_period_fraction": 0.25,
                 "minimum_segment_s": MIN_SEGMENT_SECONDS, "minimum_concentration": MIN_QUALITY,
                 "minimum_processed_coverage": MIN_PROCESSED_COVERAGE,
                 "minimum_contiguous_pulse_s": MIN_PULSE_SECONDS, "window_local_reprocessing": True,
                 "pulse_requires_peak_run_quality_pass": True,
                 "interpolation": False},
        "expression": {"upstream_model": e_meta["landmark_model"], "selected_channels": SELECTED,
                       "minimum_summary_samples": MIN_EXPRESSION_SAMPLES, "interpolation": False,
                       "rolling_window_s": WINDOW_SECONDS, "minimum_rolling_samples": MIN_WINDOW_SAMPLES,
                       "maximum_temporal_step_periods": MAX_STEP_PERIODS,
                       "provenance": "MediaPipe blendshape coefficients",
                       "window_local_temporal_features": True},
        "key_columns": keys,
        "feature_columns": {name: [key for key in rows[0] if key not in keys]
                            for name, rows in zip(names, tables, strict=True)},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    diagnostics(output, tables[0], tables[1])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("rppg_dir", type=Path)
    parser.add_argument("expression_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=4.0)
    parser.add_argument("--split-group", help="Shared source-family or subject ID for related clips")
    args = parser.parse_args()
    metadata = run(args.video, args.rppg_dir, args.expression_dir, args.output,
                   args.window_seconds, args.overlap_seconds, args.split_group)
    print(f"{metadata['windowing']['count']} synchronized windows | {args.output}")


if __name__ == "__main__":
    main()

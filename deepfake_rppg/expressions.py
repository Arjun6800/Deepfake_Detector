"""Extract timestamp-aligned MediaPipe blendshapes and temporal features."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .landmarks import DEFAULT_MODEL, LandmarkSampler
from .signal import finite_runs
from .video import choose_timing, read_video


FEATURE_GROUPS = {
    "Eyebrows": ("browInnerUp", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight"),
    "Eyes": ("eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight"),
    "Cheeks": ("cheekPuff", "cheekSquintLeft", "cheekSquintRight"),
    "Mouth": (
        "jawOpen", "mouthClose", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthStretchLeft", "mouthStretchRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
        "mouthFunnel", "mouthUpperUpLeft", "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    ),
}
SELECTED = tuple(name for names in FEATURE_GROUPS.values() for name in names)
WINDOW_SECONDS = 0.5
MIN_WINDOW_SAMPLES = 3
MAX_STEP_PERIODS = 1.5


def blendshape_scores(categories):
    return {
        category.category_name: float(category.score) for category in categories
        if category.category_name and np.isfinite(category.score)
    }


class ExpressionSampler(LandmarkSampler):
    def __init__(self, model_path):
        super().__init__(model_path, output_blendshapes=True)

    def sample(self, frame):
        if self.select_face(frame) is None:
            return self.face_status, {}
        scores = blendshape_scores(self.blendshapes)
        if not scores:
            return "blendshapes_unavailable", scores
        if any(name not in scores for name in SELECTED):
            return "partial_blendshapes", scores
        return "accepted", scores


def temporal_features(times, values, fps):
    """Calculate trailing statistics within contiguous observations and signed score/s."""
    means = np.full_like(values, np.nan, dtype=float)
    deviations = means.copy()
    velocity = means.copy()
    if fps is None:
        return means, deviations, velocity
    dt = np.diff(times)
    groups = np.r_[0, np.cumsum((dt <= 0) | (dt > MAX_STEP_PERIODS / fps))]
    for column, series in enumerate(values.T):
        for run in finite_runs(np.isfinite(series) & np.isfinite(times), groups):
            start, stop = run.start, run.stop
            velocity[start + 1:stop, column] = np.diff(series[run]) / dt[start:stop - 1]
            for i in range(start + MIN_WINDOW_SAMPLES - 1, stop):
                left = start + np.searchsorted(times[start:i + 1], times[i] - WINDOW_SECONDS - 1e-9)
                if i - left + 1 >= MIN_WINDOW_SAMPLES:
                    window = series[left:i + 1]
                    means[i, column] = window.mean()
                    deviations[i, column] = window.std()
    return means, deviations, velocity


def write_columns(path, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(
            ["" if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value for value in row]
            for row in zip(*columns.values(), strict=True)
        )


def check_alignment(path, source_times, times, has_scores):
    reference = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    matches = len(reference) == len(times) and (
        np.array_equal(reference["frame_index"], np.arange(len(times)))
        and np.allclose(reference["source_timestamp_s"], source_times, rtol=0, atol=1e-9, equal_nan=True)
        and np.allclose(reference["analysis_timestamp_s"], times, rtol=0, atol=1e-9, equal_nan=True)
    )
    return {
        "reference": str(path),
        "frame_and_timestamp_match": bool(matches),
        "shared_expression_and_rppg_face_frames": (
            int((has_scores & (reference["face_detected"] == 1)).sum()) if matches else None
        ),
    }


def plot_features(output, times, selected, deviations, velocity, has_scores):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    columns = dict(zip(SELECTED, selected.T, strict=True))
    choices = (
        ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
        ("eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft"),
        ("cheekSquintLeft", "cheekSquintRight", "cheekPuff"),
        ("jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker"),
    )
    figure = Figure(figsize=(12, 8), layout="constrained")
    axes = figure.subplots(4, sharex=True)
    for axis, (group, names) in zip(axes, zip(FEATURE_GROUPS, choices, strict=True), strict=True):
        for name in names:
            axis.plot(times, columns[name], lw=0.8, label=name)
        axis.set_ylabel(group + "\n(score)")
        axis.legend(loc="upper right", fontsize=8, ncols=2)
    axes[-1].set_xlabel("Time from first decoded frame (s)")
    FigureCanvasAgg(figure).print_png(output / "expression_scores.png")

    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(3, sharex=True)
    for name in ("browInnerUp", "eyeBlinkLeft", "cheekSquintLeft", "jawOpen"):
        i = SELECTED.index(name)
        axes[0].plot(times, velocity[:, i], lw=0.8, label=name)
        axes[1].plot(times, deviations[:, i], lw=0.8, label=name)
    axes[0].set_ylabel("Velocity\n(score / s)")
    axes[0].legend(fontsize=8, ncols=4)
    axes[1].set_ylabel("Trailing 0.5 s\nscore SD")
    axes[2].step(times, has_scores.astype(int), where="post")
    axes[2].set_ylabel("Scores available")
    axes[2].set_yticks([0, 1], ["Missing", "Available"])
    axes[2].set_ylim(-0.15, 1.15)
    axes[-1].set_xlabel("Time from first decoded frame (s)")
    FigureCanvasAgg(figure).print_png(output / "expression_dynamics.png")


def export_features(video, model, output, decoded, model_seconds, extraction_seconds,
                    assume_cfr=False, alignment_reference=None):
    started = perf_counter()
    source_times, observations, nominal, backend = decoded
    timing = choose_timing(source_times, nominal, assume_cfr)
    times = timing.timestamps
    statuses = [status for status, _ in observations]
    detections = [scores for _, scores in observations]
    names = sorted({name for scores in detections for name in scores})
    csv_names = (*SELECTED, *(name for name in names if name not in SELECTED))
    has_scores = np.asarray([bool(scores) for scores in detections], dtype=bool)
    selected = np.full((len(detections), len(SELECTED)), np.nan)
    for i, scores in enumerate(detections):
        selected[i] = [scores.get(name, np.nan) for name in SELECTED]
    rolling_means, rolling_std, velocity = temporal_features(times, selected, timing.fps)
    index_columns = {
        "frame_index": np.arange(len(observations)),
        "source_timestamp_s": source_times,
        "analysis_timestamp_s": times,
        "status": statuses,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_columns(output / "blendshapes.csv", {
        **index_columns, "has_blendshapes": has_scores.astype(int),
        **{name: [scores.get(name, np.nan) for scores in detections]
           for name in csv_names},
    })
    dynamics = {}
    for i, name in enumerate(SELECTED):
        dynamics[f"{name}_mean_0p5s"] = rolling_means[:, i]
        dynamics[f"{name}_std_0p5s"] = rolling_std[:, i]
        dynamics[f"{name}_velocity_per_s"] = velocity[:, i]
    write_columns(output / "temporal_features.csv", {**index_columns, **dynamics})
    rows = []
    for i, name in enumerate(SELECTED):
        valid = selected[np.isfinite(selected[:, i]), i]
        local_std = rolling_std[np.isfinite(rolling_std[:, i]), i]
        speeds = velocity[np.isfinite(velocity[:, i]), i]
        rows.append((
            name, len(valid), len(speeds), len(local_std), valid.mean() if len(valid) else np.nan,
            valid.std() if len(valid) else np.nan, np.abs(speeds).mean() if len(speeds) else np.nan,
            np.median(local_std) if len(local_std) else np.nan,
        ))
    fields = ("blendshape", "observations", "velocity_observations", "rolling_observations",
              "mean", "std", "mean_absolute_velocity_per_s", "median_std_0p5s")
    write_columns(output / "clip_summary.csv", dict(zip(fields, zip(*rows, strict=True), strict=True)))
    plot_features(output, times, selected, rolling_std, velocity, has_scores)
    metadata = {
        "input_video": str(video), "landmark_model": str(model), "decoder_backend": backend,
        "decoded_frames": len(observations), "fps": timing.fps, "timestamp_source": timing.source,
        "duration_seconds": float(times[-1]) if timing.fps is not None and len(times) else None,
        "status_counts": dict(Counter(statuses)), "frames_with_scores": int(has_scores.sum()),
        "frames_with_all_selected_scores": int(np.isfinite(selected).all(axis=1).sum()),
        "coverage": float(has_scores.mean()) if len(has_scores) else 0,
        "emitted_blendshapes": names, "selected_feature_groups": FEATURE_GROUPS,
        "temporal_settings": {"window_seconds": WINDOW_SECONDS, "minimum_window_samples": MIN_WINDOW_SAMPLES,
                              "maximum_step_periods": MAX_STEP_PERIODS, "interpolation": False},
        "alignment": (check_alignment(alignment_reference, source_times, times, has_scores)
                      if alignment_reference else None),
    }
    export_seconds = perf_counter() - started
    metadata["runtime_seconds"] = {
        "model_setup": model_seconds, "decode_track_and_infer": extraction_seconds,
        "feature_export_and_plots": export_seconds, "total": model_seconds + extraction_seconds + export_seconds,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return metadata


def run(video, output, model=DEFAULT_MODEL, assume_cfr=False, alignment_reference=None):
    if not video.is_file():
        raise ValueError(f"Input is not a file: {video}")
    if not model.is_file():
        raise ValueError(f"Landmark model is not a file: {model}")
    started = perf_counter()
    with ExpressionSampler(model) as sampler:
        setup = perf_counter() - started
        decoded = read_video(video, sampler.sample)
    extraction = perf_counter() - started - setup
    return export_features(video, model, output, decoded, setup, extraction, assume_cfr, alignment_reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--landmark-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--assume-cfr", action="store_true")
    parser.add_argument("--alignment-reference", type=Path, help="Optional rPPG frame_traces.csv")
    args = parser.parse_args()
    metadata = run(args.video, args.output, args.landmark_model, args.assume_cfr, args.alignment_reference)
    print(f"{metadata['frames_with_scores']}/{metadata['decoded_frames']} frames with blendshapes | {args.output}")
    alignment = metadata["alignment"]
    if alignment is not None and not alignment["frame_and_timestamp_match"]:
        print(f"Frame or timestamp mismatch against {alignment['reference']}")
        return 1
    return 0 if metadata["frames_with_scores"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for the local Phase 1 pipeline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .face import BOX_UPDATE_WEIGHT, REGIONS, FaceSampler
from .landmarks import CROP_PADDING, DEFAULT_MODEL, SKIN_POLYGONS
from .reliability import (
    FALLBACK_MARGIN_FRACTION, LANDMARK_UPDATE_WEIGHT, MAX_NORMALIZED_STEP, METHODS,
    RELIABLE_POLYGONS, ReliabilitySampler,
)
from .signal import (
    BAND_HZ,
    MAX_GAP_SECONDS,
    MIN_QUALITY,
    MIN_SEGMENT_SECONDS,
    POS_WINDOW_SECONDS,
    pos,
    process_signal,
    resample_observations,
)
from .video import choose_timing, read_video


def extract_waveforms(times, rgb, fps, groups=None, max_gap=MAX_GAP_SECONDS):
    grid, colors, grid_groups = resample_observations(times, rgb, fps, groups, max_gap)
    raw, processed, quality = {}, {}, {}
    for i, region in enumerate(REGIONS):
        trace = colors[:, i]
        raw[f"{region}_green"] = trace[:, 1]
        source = grid_groups[:, i]
        raw[f"{region}_pos"] = pos(trace, fps, source)
        for method in ("green", "pos"):
            name = f"{region}_{method}"
            waveform = raw[name]
            processed[name], metrics = process_signal(waveform, fps, groups=source)
            metrics["raw_coverage"] = float(np.isfinite(waveform).mean())
            score = metrics["spectral_concentration"]
            metrics["passes_quality_threshold"] = bool(
                score is not None and score >= MIN_QUALITY and metrics["processed_coverage"] >= 0.5
            )
            quality[name] = metrics
    return grid, raw, processed, quality


def write_csv(path, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(
            [value if np.isfinite(value) else "" for value in row]
            for row in zip(*columns.values(), strict=True)
        )


def write_plot(path, grid, processed, status):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 7), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(len(REGIONS), sharex=True)
    for axis, region in zip(axes, REGIONS, strict=True):
        for method in ("green", "pos"):
            waveform = processed[f"{region}_{method}"]
            if np.isfinite(waveform).any():
                axis.plot(grid, waveform, label=method.upper(), linewidth=0.9)
        if axis.lines:
            axis.legend(loc="upper right")
        else:
            axis.text(0.5, 0.5, "No processed samples", ha="center", transform=axis.transAxes)
        axis.set_ylabel(region.replace("_", " ") + "\n(z-score)")
    axes[-1].set_xlabel("Time from first decoded frame (s)")
    figure.suptitle(f"Processed rPPG waveforms | {status.replace('_', ' ')}")
    canvas.print_png(path)


def run(video, output, assume_cfr=False, roi_method="rectangular", landmark_model=DEFAULT_MODEL):
    try:
        if not video.is_file():
            raise ValueError(f"Input is not a file: {video}")
        if roi_method == "rectangular":
            decoded = read_video(video, FaceSampler().sample_with_geometry)
        elif roi_method in METHODS:
            if not Path(landmark_model).is_file():
                raise ValueError(f"Landmark model is not a file: {landmark_model}")
            with ReliabilitySampler(landmark_model) as sampler:
                captured = read_video(video, sampler.sample_methods)
            decoded = select_method(captured, roi_method)
        else:
            raise ValueError(f"Unknown ROI method: {roi_method}")
    except (ImportError, ValueError, RuntimeError) as error:
        return export_run(video, output, None, assume_cfr, roi_method, landmark_model, str(error))
    return export_run(video, output, decoded, assume_cfr, roi_method, landmark_model)


def select_method(captured, method):
    times, results, nominal, backend = captured
    return (times, [observed[method] for observed, _ in results], nominal, backend,
            [diagnostic[method] for _, diagnostic in results] if method != "rectangular" else [])


def export_run(video, output, decoded, assume_cfr, roi_method, landmark_model, error=None):
    names = [f"{region}_{method}" for region in REGIONS for method in ("green", "pos")]
    grid = np.array([])
    raw = {name: np.array([]) for name in names}
    processed = raw.copy()
    frames = {"frame_index": [], "source_timestamp_s": [], "analysis_timestamp_s": [], "face_detected": []}
    box_fields = tuple(
        f"{prefix}_{field}_px"
        for prefix in ("face", "sampling")
        for field in ("x", "y", "width", "height")
    )
    frames.update((field, []) for field in box_fields)
    for region in REGIONS:
        for field in ("valid_pixel_fraction", "red", "green", "blue"):
            frames[f"{region}_{field}"] = []
    metadata = {
        "input_video": str(video),
        "roi_method": roi_method,
        "fps": None,
        "nominal_fps": None,
        "duration_seconds": None,
        "timestamp_source": "unavailable",
        "decoded_frames": 0,
        "face_frames": 0,
        "decoder_backend": None,
        "signal_quality": {},
        "settings": {
            "bandpass_hz": BAND_HZ,
            "pos_window_seconds": POS_WINDOW_SECONDS,
            "minimum_segment_seconds": MIN_SEGMENT_SECONDS,
            "maximum_interpolation_gap_seconds": MAX_GAP_SECONDS,
            "minimum_spectral_concentration": MIN_QUALITY,
            "minimum_processed_coverage": 0.5,
            "roi_box_update_weight": BOX_UPDATE_WEIGHT,
            "regions": REGIONS,
        },
    }
    if roi_method in METHODS[1:]:
        metadata["settings"].update(
            roi_box_update_weight=BOX_UPDATE_WEIGHT if roi_method == "hybrid" else None,
            regions=SKIN_POLYGONS if roi_method == "landmark-legacy" else RELIABLE_POLYGONS,
            landmark_model=str(landmark_model),
            landmark_crop_padding_fraction=CROP_PADDING,
        )
        if roi_method != "landmark-legacy":
            metadata["settings"].update(
                landmark_update_weight=LANDMARK_UPDATE_WEIGHT,
                maximum_normalized_shape_step=MAX_NORMALIZED_STEP,
            )
        if roi_method == "hybrid":
            metadata["settings"]["fallback_neighborhood_fraction"] = FALLBACK_MARGIN_FRACTION
    diagnostics = []
    if decoded is None:
        metadata.update(status="input_error", detail=error)
    else:
        source_times, observations, nominal, backend, *extras = decoded
        if extras:
            diagnostics = extras[0]
        timing = choose_timing(source_times, nominal, assume_cfr)
        metadata.update(
            fps=timing.fps,
            nominal_fps=nominal if np.isfinite(nominal) and nominal > 0 else None,
            duration_seconds=float(timing.timestamps[-1]) if len(source_times) and timing.fps else None,
            timestamp_source=timing.source,
            decoded_frames=len(source_times),
            decoder_backend=backend,
        )
        if observations:
            samples, coverage, detected, boxes = zip(*observations, strict=True)
            rgb = np.asarray(samples)
            frames.update(
                frame_index=np.arange(len(source_times)),
                source_timestamp_s=source_times,
                analysis_timestamp_s=timing.timestamps,
                face_detected=np.asarray(detected, dtype=int),
            )
            metadata["face_frames"] = sum(detected)
            frames.update(zip(box_fields, np.asarray(boxes).T, strict=True))
            for i, region in enumerate(REGIONS):
                frames[f"{region}_valid_pixel_fraction"] = np.asarray(coverage)[:, i]
                for j, channel in enumerate(("red", "green", "blue")):
                    frames[f"{region}_{channel}"] = rgb[:, i, j]

        if len(source_times) < 2:
            metadata["status"] = "too_short" if len(source_times) else "no_frames"
        elif timing.fps is None or timing.fps <= 2 * BAND_HZ[1]:
            metadata["status"] = "invalid_timing"
        else:
            groups = np.asarray([row[2] for row in diagnostics]) if roi_method == "hybrid" else None
            grid, raw, processed, quality = extract_waveforms(timing.timestamps, rgb, timing.fps, groups)
            metadata["signal_quality"] = quality
            if not metadata["face_frames"]:
                metadata["status"] = "no_face"
            elif len(grid) < np.ceil(MIN_SEGMENT_SECONDS * timing.fps):
                metadata["status"] = "too_short"
            elif any(metrics["passes_quality_threshold"] for metrics in quality.values()):
                metadata["status"] = "ok"
            else:
                metadata["status"] = "low_quality"

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "frame_traces.csv", frames)
    if diagnostics:
        write_roi_diagnostics(output / "roi_diagnostics.csv", diagnostics)
    write_csv(output / "raw_waveform.csv", {"timestamp_s": grid, **raw})
    write_csv(output / "processed_waveform.csv", {"timestamp_s": grid, **processed})
    write_plot(output / "waveforms.png", grid, processed, metadata["status"])
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def write_roi_diagnostics(path, diagnostics):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_index", "region", "face_status", "region_status", "source",
                         "mask_pixels", "valid_pixels", "center_x_px", "center_y_px", "nose_offset"))
        for frame_index, (status, measurement, sources, nose_offset) in enumerate(diagnostics):
            failures = measurement.failures(status)
            for i, region in enumerate(REGIONS):
                x, y = measurement.centers[i]
                writer.writerow((
                    frame_index, region, status, failures[i], ("missing", "landmark", "fallback")[sources[i]],
                    measurement.area[i], measurement.valid_area[i], x if np.isfinite(x) else "",
                    y if np.isfinite(y) else "", nose_offset if np.isfinite(nose_offset) else "",
                ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input facial video")
    parser.add_argument("output", type=Path, help="Directory for CSV, PNG and JSON outputs")
    parser.add_argument("--roi-method", choices=METHODS, default="rectangular")
    parser.add_argument("--landmark-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--assume-cfr", action="store_true",
        help="Allow nominal FPS timing if decoder timestamps fail and the video is known to be CFR",
    )
    args = parser.parse_args()
    metadata = run(args.video, args.output, args.assume_cfr, args.roi_method, args.landmark_model)
    detail = metadata.get("detail")
    print(f"{metadata['status']} | {args.output}" + (f" | {detail}" if detail else ""))
    return 0 if metadata["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

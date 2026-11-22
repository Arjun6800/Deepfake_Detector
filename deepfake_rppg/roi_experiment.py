"""Run and compare all ROI methods on a single decode of a video."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy import signal

from .__main__ import export_run, extract_waveforms, select_method, write_csv
from .face import REGIONS
from .landmarks import DEFAULT_MODEL, mask_crop, skin_masks
from .reliability import METHODS, ReliabilitySampler, reliable_masks
from .signal import BAND_HZ, finite_runs
from .video import choose_timing, read_video


LABELS = ("Rectangular", "Legacy landmarks", "Stabilized landmarks", "Hybrid")


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def peak_spread(values, fps):
    """Median and IQR of in-band peaks across fully supported 8 s windows."""
    bins = []
    size = round(8 * fps)
    hop = round(4 * fps)
    for run in finite_runs(np.isfinite(values)):
        for start in range(run.start, run.stop - size + 1, hop):
            f, power = signal.periodogram(values[start:start + size], fs=fps,
                                          window="hann", detrend="linear")
            in_band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
            bins.append(f[in_band][np.argmax(power[in_band])])
    if not bins:
        return 0, None, None
    q1, median, q3 = np.percentile(bins, (25, 50, 75))
    return len(bins), median, q3 - q1


def analyze_capture(video, model, output, captured):
    output.mkdir(parents=True, exist_ok=True)
    positions, results, nominal, _ = captured
    timing = choose_timing(positions, nominal)
    if timing.fps is None or timing.fps <= 2 * BAND_HZ[1]:
        raise ValueError("Comparison requires usable video timestamps and analysis FPS")
    times, fps = timing.timestamps, timing.fps
    decoded = {method: select_method(captured, method) for method in METHODS}
    colors = {method: np.asarray([row[0] for row in data[1]]) for method, data in decoded.items()}
    valid = {method: np.isfinite(rgb).all(axis=2) for method, rgb in colors.items()}
    common = np.logical_and.reduce(list(valid.values()))
    hybrid_groups = np.asarray([row[2] for row in decoded["hybrid"][4]])
    native_rows, trace_rows = [], []
    full_processed, matched_processed = {}, {}
    diagnostics = {method: data[4] for method, data in decoded.items() if data[4]}
    for method, data in decoded.items():
        export_run(video, output / method, data, False, method, model)
        groups = hybrid_groups if method == "hybrid" else None
        grid, full_raw, full_processed[method], full = extract_waveforms(times, colors[method], fps, groups)
        # Start from the same native observations and prevent interpolation over any missing frame.
        matched_rgb = np.where(common[:, :, None], colors[method], np.nan)
        _, _, matched_processed[method], matched = extract_waveforms(
            times, matched_rgb, fps, hybrid_groups, max_gap=min(1 / fps, 0.25),
        )
        for i, region in enumerate(REGIONS):
            reasons = Counter()
            fallback = 0
            if method in diagnostics:
                for face_status, measurement, sources, _ in diagnostics[method]:
                    reasons[measurement.failures(face_status)[i]] += 1
                    fallback += int(sources[i] == 2)
            else:
                reasons["face_tracking"] = sum(not row[2] for row in data[1])
                reasons["rectangular_pixel_failure"] = int((~valid[method][:, i]).sum()) - reasons["face_tracking"]
            native_rows.append({
                "method": method, "region": region, "decoded_frames": len(results),
                "valid_rgb_frames": int(valid[method][:, i].sum()),
                "missing_rgb_frames": int((~valid[method][:, i]).sum()),
                "fallback_frames": fallback, "shared_rgb_frames": int(common[:, i].sum()),
                **{key: reasons[key] for key in (
                    "face_tracking", "landmark_inference", "landmark_association", "empty_geometry",
                    "insufficient_mask_pixels", "insufficient_exposure_valid_pixels", "rectangular_pixel_failure",
                )},
            })
            for algorithm in ("green", "pos"):
                key = f"{region}_{algorithm}"
                raw_count = int(np.isfinite(full_raw[key]).sum())
                processed_count = int(np.isfinite(full_processed[method][key]).sum())
                n_windows, peak, spread = peak_spread(matched_processed[method][key], fps)
                trace_rows.append({
                    "method": method, "trace": key, "analysis_samples": len(grid),
                    "resampled_rgb_samples": int(np.isfinite(full_raw[f"{region}_green"]).sum()),
                    "raw_samples": raw_count, "processed_samples": processed_count,
                    "lost_after_raw_processing": raw_count - processed_count,
                    "full_concentration": full[key]["spectral_concentration"],
                    "full_pass": full[key]["passes_quality_threshold"],
                    "matched_processed_samples": int(np.isfinite(matched_processed[method][key]).sum()),
                    "matched_concentration": matched[key]["spectral_concentration"],
                    "matched_8s_windows": n_windows, "matched_peak_median_hz": peak,
                    "matched_peak_iqr_hz": spread,
                })
    write_table(output / "coverage_by_region.csv", native_rows)
    write_table(output / "signal_comparison.csv", trace_rows)
    write_table(output / "stability_by_region.csv", stability_rows(decoded, colors, common))
    common_csv = {"timestamp_s": times}
    common_csv.update((f"{region}_shared_rgb", common[:, i].astype(int)) for i, region in enumerate(REGIONS))
    write_csv(output / "shared_timestamps.csv", common_csv)
    write_csv(output / "matched_waveforms.csv", {
        "timestamp_s": grid,
        **{f"{method}_{key}": trace for method, traces in matched_processed.items() for key, trace in traces.items()},
    })
    plot_coverage(output, native_rows, trace_rows)
    plot_support(output, times, diagnostics)
    plot_matched(output, grid, matched_processed)


def stability_rows(decoded, colors, common):
    rows = []
    for method in ("landmark-legacy", "landmark"):
        data = decoded[method]
        boxes = np.asarray([row[3][:4] for row in data[1]])
        measurements = [row[1] for row in data[4]]
        centers = np.asarray([item.centers for item in measurements])
        relative = (centers - boxes[:, None, :2]) / boxes[:, None, 2:]
        motion = np.linalg.norm(np.diff(relative, axis=0), axis=2)
        green_change = np.abs(np.diff(colors[method][:, :, 1], axis=0))
        for i, region in enumerate(REGIONS):
            pairs = common[1:, i] & common[:-1, i]
            areas = [item.area[i] for keep, item in zip(common[:, i], measurements, strict=True) if keep]
            rows.append({
                "method": method, "region": region, "adjacent_shared_pairs": int(pairs.sum()),
                "median_shared_mask_pixels": np.median(areas) if areas else None,
                "p95_box_relative_centroid_step": np.percentile(motion[pairs, i], 95) if pairs.any() else None,
                "p95_green_level_step": np.percentile(green_change[pairs, i], 95) if pairs.any() else None,
            })
    return rows


def plot_coverage(output, native_rows, trace_rows):
    figure = Figure(figsize=(13, 6), layout="constrained")
    axes = figure.subplots(1, 3, sharey=True)
    for axis, region in zip(axes, REGIONS, strict=True):
        rows = [row for row in native_rows if row["region"] == region]
        traces = [row for row in trace_rows if row["trace"] == f"{region}_pos"]
        x = np.arange(len(rows))
        stages = (
            ("Decoded RGB", [r["valid_rgb_frames"] / r["decoded_frames"] for r in rows]),
            ("Raw POS", [r["raw_samples"] / r["analysis_samples"] for r in traces]),
            ("Processed POS", [r["processed_samples"] / r["analysis_samples"] for r in traces]),
            ("Matched processed POS", [r["matched_processed_samples"] / r["analysis_samples"] for r in traces]),
        )
        for i, (label, fractions) in enumerate(stages):
            axis.bar(x + (i - 1.5) * 0.19, 100 * np.asarray(fractions), width=0.19, label=label)
        axis.set_xticks(x, LABELS, rotation=30, ha="right")
        axis.set_title(region.replace("_", " "))
        axis.set_ylim(0, 104)
    axes[0].set_ylabel("Coverage (%)")
    axes[0].legend(fontsize=8, loc="lower left")
    FigureCanvasAgg(figure).print_png(output / "coverage_comparison.png")


def plot_support(output, times, diagnostics):
    figure = Figure(figsize=(12, 8), layout="constrained")
    axes = figure.subplots(4, sharex=True)
    for i, region in enumerate(REGIONS):
        for method, color in (("landmark-legacy", "tab:orange"), ("landmark", "tab:blue")):
            data = diagnostics[method]
            area = np.where([row[0] == "accepted" for row in data],
                            [row[1].area[i] for row in data], np.nan)
            axes[i].plot(times, area, lw=0.8, label=method, color=color)
        axes[i].axhline(16, color="black", linestyle=":", label="minimum pixel count")
        fallback = np.array([row[2][i] == 2 for row in diagnostics["hybrid"]])
        axes[i].scatter(times[fallback], np.full(fallback.sum(), 16), marker="|", s=35,
                        color="tab:green", label="fallback used")
        axes[i].set_ylabel(region.replace("_", " ") + "\nmask pixels")
        axes[i].set_yscale("symlog", linthresh=16)
    axes[0].legend(ncols=2, fontsize=8)
    axes[-1].plot(times, [row[3] for row in diagnostics["landmark"]], color="tab:purple", lw=1)
    axes[-1].set_ylabel("Nose offset\n(rotation proxy)")
    axes[-1].set_xlabel("Time (s)")
    FigureCanvasAgg(figure).print_png(output / "mask_support.png")


def plot_matched(output, grid, processed):
    figure = Figure(figsize=(12, 8), layout="constrained")
    axes = figure.subplots(3, 2, sharex=True)
    for axis, trace in zip(axes.flat, processed["rectangular"], strict=True):
        for label, data in zip(LABELS, processed.values(), strict=True):
            if label != "Hybrid":  # Hybrid and stabilized landmarks have identical shared RGB by design.
                axis.plot(grid, data[trace], linewidth=0.8, label=label)
        axis.set_title(trace.replace("_", " "))
    axes[0, 0].legend(fontsize=8)
    figure.supxlabel("Time (s)")
    figure.supylabel("Processed waveform (z-score)")
    FigureCanvasAgg(figure).print_png(output / "matched_waveforms.png")


def plot_masks(video, model, output):
    """Replay short lead-ins so the displayed landmarks have temporal context."""
    indices = (24, 144, 504, 984)
    figure = Figure(figsize=(10, 6), layout="constrained")
    capture = cv2.VideoCapture(str(video))
    try:
        with ReliabilitySampler(model) as sampler:
            for col, index in enumerate(indices):
                axes = (figure.add_subplot(2, len(indices), col + 1),
                        figure.add_subplot(2, len(indices), col + len(indices) + 1))
                sampler.target.previous = sampler.target.sampling_box = None
                sampler.previous = None
                sampler.stabilizer.reset()
                capture.set(cv2.CAP_PROP_POS_FRAMES, index - 8)
                for _ in range(9):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Could not decode around frame {index}")
                    face = sampler.select_face(frame)
                    if face is None:
                        sampler.stabilizer.reset()
                    else:
                        points, box = face
                        stable = sampler.stabilizer.update(points, box)
                for axis, label in zip(axes, ("Legacy", "Stabilized"), strict=True):
                    axis.set_title(f"{label} | frame {index}")
                    axis.axis("off")
                if face is None:
                    continue
                shown, origin = mask_crop(frame, box + [-8, -8, 16, 16])
                shape = shown.shape[:2]
                legacy = skin_masks(points - origin, shape)
                updated, _ = reliable_masks(points - origin, stable - origin, shape, box[2])
                for axis, masks in zip(axes, (legacy, updated), strict=True):
                    rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
                    for mask, color in zip(masks, ([255, 196, 43], [45, 215, 235], [230, 110, 230]), strict=True):
                        rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.asarray(color)
                    axis.imshow(rgb)
    finally:
        capture.release()
    FigureCanvasAgg(figure).print_png(output / "roi_mask_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--landmark-model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    with ReliabilitySampler(args.landmark_model) as sampler:
        captured = read_video(args.video, sampler.sample_methods)
    analyze_capture(args.video, args.landmark_model, args.output, captured)
    plot_masks(args.video, args.landmark_model, args.output)

"""Plot test2 rectangular and test3 landmark extraction diagnostics."""

import argparse
import json
from pathlib import Path

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy import signal

from .face import REGIONS
from .landmarks import FACE_OVAL, LandmarkSampler, skin_masks
from .signal import finite_runs, process_signal


def load_csv(directory, filename):
    return np.genfromtxt(directory / filename, delimiter=",", names=True)


def comparison_plots(baseline, output):
    styles = ("test2 rectangular", "test3 landmark")
    directories = (baseline, output)
    metadata = [json.loads((path / "metadata.json").read_text()) for path in directories]
    processed = [load_csv(path, "processed_waveform.csv") for path in directories]
    raw = [load_csv(path, "raw_waveform.csv") for path in directories]
    keys = [f"{region}_{method}" for region in REGIONS for method in ("green", "pos")]
    for spectral in (False, True):
        figure = Figure(figsize=(12, 8), layout="constrained")
        axes = figure.subplots(3, 2, sharex=True)
        for axis, key in zip(axes.flat, keys, strict=True):
            axis.set_title(key.replace("_", " "))
            for i, label in enumerate(styles):
                if spectral:
                    values = raw[i][key]
                    run = max(finite_runs(np.isfinite(values)),
                              key=lambda part: part.stop - part.start, default=None)
                    if run is None or run.stop - run.start < 2:
                        continue
                    fps = metadata[i]["fps"]
                    section = signal.detrend(values[run])
                    f, power = signal.welch(section, fs=fps, nperseg=min(len(section), round(8 * fps)))
                    total = power[1:].sum()
                    if total > 0:
                        axis.plot(f[1:], power[1:] / total, label=label)
                else:
                    axis.plot(processed[i]["timestamp_s"], processed[i][key], label=label, lw=0.9)
            if spectral:
                axis.axvspan(*metadata[1]["settings"]["bandpass_hz"], color="gray", alpha=0.15)
            if axis.lines:
                axis.legend(fontsize=8)
        figure.supxlabel("Frequency (Hz)" if spectral else "Time (s)")
        figure.supylabel("Fraction of non-DC power per bin" if spectral else "Processed waveform (z-score)")
        filename = "psd_comparison.png" if spectral else "waveform_comparison.png"
        FigureCanvasAgg(figure).print_png(output / filename)

    if not np.array_equal(raw[0]["timestamp_s"], raw[1]["timestamp_s"]):
        raise ValueError("Shared-support comparison requires matching timestamp grids")
    figure = Figure(figsize=(11, 5), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True, sharex=True)
    positions = np.arange(len(keys))
    for axis, shared_only in zip(axes, (False, True), strict=True):
        for i, label in enumerate(styles):
            scores = []
            for key in keys:
                if shared_only:
                    shared = np.isfinite(raw[0][key]) & np.isfinite(raw[1][key])
                    _, metrics = process_signal(np.where(shared, raw[i][key], np.nan), metadata[i]["fps"])
                else:
                    metrics = metadata[i]["signal_quality"][key]
                score = metrics["spectral_concentration"]
                scores.append(np.nan if score is None else score)
            axis.barh(positions + (i - 0.5) * 0.35, scores, height=0.35, label=label)
        threshold = metadata[1]["settings"]["minimum_spectral_concentration"]
        axis.axvline(threshold, color="black", linestyle=":", label="quality threshold")
        axis.set_title("Shared time support" if shared_only else "Full run")
        axis.set_xlabel("Spectral concentration")
    axes[0].set_yticks(positions, [key.replace("_", " ") for key in keys])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    FigureCanvasAgg(figure).print_png(output / "concentration_comparison.png")


def mask_samples(video, model, baseline, output):
    frames = load_csv(baseline, "frame_traces.csv")
    indices = [24, 144, 264, 384, 504, 624, 744, 864, 984]
    colors = np.array([[250, 190, 50], [55, 210, 235], [220, 110, 245]])
    figure = Figure(figsize=(10, 9), layout="constrained")
    capture = cv2.VideoCapture(str(video))
    try:
        with LandmarkSampler(model) as sampler:
            for index, axis in zip(indices, figure.subplots(3, 3).flat, strict=True):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Could not read sample frame {index}")
                faces = sampler.detect(frame)
                x, y, width, height = (frames[f"sampling_{name}_px"][index]
                                       for name in ("x", "y", "width", "height"))
                if len(faces) != 1:
                    axis.set_title(f"Frame {index} | landmark unavailable")
                    axis.axis("off")
                    continue
                points = faces[0]
                oval = points[list(FACE_OVAL)]
                x0, y0 = np.maximum(0, np.floor(oval.min(axis=0)).astype(int) - 24)
                x1, y1 = np.ceil(oval.max(axis=0)).astype(int) + 25
                crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                for mask, color in zip(skin_masks(points - [x0, y0], crop.shape[:2]), colors, strict=True):
                    crop[mask] = 0.55 * crop[mask] + 0.45 * color
                if np.isfinite([x, y, width, height]).all():
                    for left, top, right, bottom in REGIONS.values():
                        cv2.rectangle(crop, (round(x + left * width - x0), round(y + top * height - y0)),
                                      (round(x + right * width - x0), round(y + bottom * height - y0)),
                                      (255, 255, 255), 1)
                axis.imshow(crop)
                axis.set_title(f"Frame {index} | {frames['analysis_timestamp_s'][index]:g} s")
                axis.axis("off")
    finally:
        capture.release()
    figure.suptitle("Landmark masks (color) and test2 rectangles (white)")
    FigureCanvasAgg(figure).print_png(output / "roi_mask_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "model", "baseline", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    comparison_plots(args.baseline, args.output)
    mask_samples(args.video, args.model, args.baseline, args.output)

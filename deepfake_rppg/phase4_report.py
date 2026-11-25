"""Summarize Phase 4 audit records and pilot measurements."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .batch import FEATURE_COUNTS
from .dataset import DFD, METHODS, save_json
from .signal import MIN_QUALITY
from .window_features import MIN_PROCESSED_COVERAGE, MIN_PULSE_SECONDS


def quantiles(values):
    return dict(zip(("min", "p25", "median", "p75", "max"),
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist(), strict=True)) if values else {}


def summarize(manifest, batch):
    results = batch["results"]
    complete = [row for row in results if row["status"] == "complete"]
    windows = sum(row["windows"] for row in complete)
    times = [row["seconds"] for row in results if "seconds" in row]
    coverage = [row["expression_frame_coverage"] for row in complete]
    total_frames = sum(row["decoded_frames"] for row in complete)
    quality_checks = sum(row["quality_checks"] for row in complete)
    quality_passes = sum(row["quality_passes"] for row in complete)
    concentrations = [value for row in complete for value in row["spectral_concentration"]]
    selected = {row["video_id"] for row in results}
    attempted = [row for row in manifest if row["video_id"] in selected]
    eligible = [row for row in manifest if row["validation_status"] in ("audited", "metadata_warning")
                and row["partition"] in ("train", "val", "test") and row["manipulation_method"] != DFD]
    pilot_frames = sum(row["frame_count"] or 0 for row in attempted)
    bytes_used = sum(row.get("storage_bytes", 0) for row in results)
    projections = {}
    for name, population in (("eligible_ffpp", eligible), ("all_primary_scenario", manifest)):
        frame_ratio = sum(row["frame_count"] or 0 for row in population) / pilot_frames if pilot_frames else None
        projections[name] = {
            "videos": len(population),
            "worker_hours_by_container_frames": sum(times) * frame_ratio / 3600 if frame_ratio else None,
            "wall_hours_by_container_frames": (batch["wall_seconds"] * frame_ratio / 3600
                                               if frame_ratio and not any(row.get("resumed") for row in results)
                                               else None),
            "storage_bytes_by_container_frames": round(bytes_used * frame_ratio) if frame_ratio else None,
        }
    return {
        "attempted": len(results), "status": Counter(row["status"] for row in results),
        "success_percent": 100 * len(complete) / len(results), "windows": windows,
        "wall_seconds": batch["wall_seconds"], "workers": batch["workers"],
        "seconds_per_video": {**quantiles(times), "mean": float(np.mean(times)) if times else None},
        "pilot_checkpoint_bytes": bytes_used,
        "expression_frame_coverage_per_video": quantiles(coverage),
        "expression_frame_coverage_weighted": (sum(row["decoded_frames"] * row["expression_frame_coverage"]
                                                    for row in complete) / total_frames if total_frames else None),
        "quality_gates": {"min_concentration": MIN_QUALITY, "min_processed_coverage": MIN_PROCESSED_COVERAGE,
                          "min_pulse_seconds": MIN_PULSE_SECONDS},
        "rppg_quality_checks": quality_checks, "rppg_quality_passes": quality_passes,
        "rppg_concentration_unavailable": quality_checks - len(concentrations),
        "rppg_concentration_quantiles": quantiles(concentrations),
        "missing_feature_percent": {
            name: 100 * sum(row["missing_features"][name] for row in complete) / (windows * size) if windows else None
            for name, size in FEATURE_COUNTS.items()
        },
        "projections": projections,
    }


def report(directory):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    audit = read("audit.json")
    manifest = read("manifest.json")
    extra = read("excluded_neuraltextures.json")
    encoder_hints = read("encoder_hints.json")
    batch = read("pilot/batch_summary.json")
    summary = summarize(manifest, batch)
    save_json(directory / "pilot_metrics.json", summary)
    primary_ids = {row["video_id"] for row in manifest}
    primary_duplicates = [items for items in audit["duplicate_hashes"].values()
                          if len(primary_ids.intersection(items)) > 1]
    metadata_duplicates = sum(len(item.get("duplicate_entries", {})) for item in audit["metadata"].values())
    flags = Counter(flag for row in manifest for flag in row["flags"])
    diff = Counter(key for row in manifest for key in row["metadata_mismatches"])
    lines = [
        "# Phase 4 dataset audit and pilot",
        "",
        "## Inventory and provenance",
        "",
        f"The primary inventory contains {len(manifest):,} files. The initial scan listed "
        f"{len(extra):,} clips from the excluded folder.",
        "",
        "| Subset | Files | Label | Review | Resolution variants | Most common resolution | Codec tags |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for method in ("original", *METHODS, DFD):
        method_audit = audit["by_method"][method]
        resolutions = Counter(method_audit["resolution"])
        leading = resolutions.most_common(1)
        common = f"{leading[0][0]} ({leading[0][1]})" if leading else "unknown"
        reviewed = sum(row["validation_status"] == "review" for row in manifest if row["manipulation_method"] == method)
        lines.append(f"| {method} | {audit['counts'].get(method, 0)} | {'REAL' if method == 'original' else 'FAKE'} "
                     f"| {reviewed} | {len(resolutions)} | {common} | {', '.join(method_audit['codec'])} |")
    lines.extend([
        "",
        f"Both roots were enumerated independently of the CSVs. The audit found "
        f"{len(audit['missing_metadata_paths'])} metadata paths absent from disk, "
        f"{len(audit['unexpected_paths'])} primary video paths absent from metadata, "
        f"{metadata_duplicates} distinct within-file duplicate metadata keys, "
        f"{len(audit['other_files'])} other files and {len(audit['missing_directories'])} missing directories. "
        f"Repeated references across per-method, combined and shuffled CSVs are expected. "
        f"Probe errors affected {flags['probe_error']} files and metadata label conflicts affected "
        f"{flags['metadata_label_conflict']} files.",
        "",
        f"Observed mismatches by metadata field, counted once per affected video, were {dict(diff)}. "
        "The audit compares the rounded file size in MiB, reads stream properties from the container, "
        "and estimates duration from frame count divided by reported FPS. The pilot tests sequential frame decoding.",
        "",
        f"SHA-256 found {len(primary_duplicates)} byte-identical sets with multiple primary videos. "
        f"Among the excluded clips, {sum(bool(row['same_name']) for row in extra)} share a primary NeuralTextures filename, "
        f"{sum(row['same_name_identical'] for row in extra)} match that namesake byte for byte and "
        f"{sum(bool(row['identical']) for row in extra)} match any primary file byte for byte. "
        "Hashes identify exact file duplicates. Re-encoded or visually near-identical copies remain unresolved.",
        f"Hashes were available for {sum(bool(row['sha256']) for row in extra)}/{len(extra)} excluded clips.",
    ])
    unavailable = sum(row["sha256"] is None for row in extra)
    if unavailable:
        lines.append(f"The identities of {unavailable} excluded clips remain unresolved because their files "
                     "were unavailable for hashing. The comparison retains their enumerated filenames.")
    lines.extend([
        "",
        "The audit searched the first 64 KiB of each file for embedded x264 encoder settings. "
        "The observed CRF and encoder core counts below provide evidence about the local encoding. "
        "Exact release provenance would require trusted upstream hashes. DeepFakeDetection is a Google dataset "
        "extension. The local 1,000 clips represent a subset of the documented release. Genuine counterparts "
        "and verified source links were unavailable.",
        "See the [FaceForensics++ dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md).",
        "",
        "| Subset | Embedded CRF counts | x264 core counts |",
        "| --- | --- | --- |",
    ])
    for name, excluded in [*((method, False) for method in ("original", *METHODS, DFD)),
                           ("NeuralTextures", True)]:
        records = [row for row in encoder_hints if row["method"] == name and row["excluded"] == excluded]
        crfs = Counter(str(row["crf"]) if row["crf"] is not None else "unknown" for row in records)
        cores = Counter(str(row["x264_core"]) if row["x264_core"] is not None else "unknown" for row in records)
        label = f"{name} (excluded)" if excluded else name
        lines.append(f"| {label} | {dict(crfs)} | {dict(cores)} |")
    lines.extend([
        "",
        "## Partition integrity",
        "",
        f"The [official split definitions](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) "
        f"contain {audit['official_source_ids']} source IDs by split and "
        f"{audit['official_directed_pairs']} directed pairs. The manifest partition counts are {audit['partitions']}. "
        f"The audit flagged {flags['cross_split_component']} primary videos in components touching multiple official splits, "
        f"{flags['unresolved_source']} unresolved primary source links outside DeepFakeDetection, and "
        f"{flags['pair_not_in_official_splits']} manipulated filenames outside official pairs.",
        "",
        f"The official conversion map links {audit['original_upload_source_count']} source clips to "
        f"{audit['original_upload_count']} YouTube uploads. "
        f"{len(audit['original_upload_cross_splits'])} uploads appear across official partitions. "
        "Components connect original IDs, both IDs from a manipulated filename, shared uploads in the "
        "official conversion map, and identical file hashes. "
        "Components touching several official partitions or DeepFakeDetection stay in holdout. "
        "All overlapping windows inherit the video's partition and group ID. The feature CSVs contain only "
        "the modality features. Join labels and partitions through the separate window and manifest records. "
        "Subject identity across distinct uploads and copies with different encodings remain unverified. "
        "Fit any future learned preprocessing on the training partition.",
        "",
        "## Pilot",
        "",
        f"The pilot selected {summary['attempted']} videos using the saved selection and default seed 42. "
        "It used five training source pairs, both originals per pair and four fakes per method. "
        "Each clip was decoded once with shared landmark and blendshape inference. "
        f"The existing {summary['quality_gates']['min_concentration']} spectral concentration threshold, "
        f"{summary['quality_gates']['min_processed_coverage']} processed coverage threshold and "
        f"{summary['quality_gates']['min_pulse_seconds']} s minimum pulse support were retained. "
        "Quality flags describe measurement support and carry no authenticity interpretation.",
        "",
        f"Extraction statuses were {dict(summary['status'])}. "
        f"The successful extraction rate was {summary['success_percent']:.1f}% with {summary['windows']} synchronized windows.",
        "",
        f"Wall time was {summary['wall_seconds']:.1f} s with {summary['workers']} workers. "
        f"Per-video elapsed seconds were {summary['seconds_per_video']}.",
        "",
        f"The rPPG quality gate passed {summary['rppg_quality_passes']}/{summary['rppg_quality_checks']} "
        f"window traces. Spectral concentration was unavailable for {summary['rppg_concentration_unavailable']} "
        f"traces. Available concentration quantiles were {summary['rppg_concentration_quantiles']}.",
        "",
        f"Complete selected-channel expression coverage by video was {summary['expression_frame_coverage_per_video']}. "
        f"Frame-weighted coverage was {summary['expression_frame_coverage_weighted']}. "
        f"Missing feature percentages were {summary['missing_feature_percent']}. "
        "Missing measurements remain empty in CSV and counts include successful windows only.",
        "",
        "| Method | Pilot videos | Complete | Windows | Mean processing seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    selected_methods = {row["video_id"]: row["manipulation_method"] for row in read("pilot/selection.json")}
    for method in ("original", *METHODS):
        items = [row for row in batch["results"] if selected_methods[row["video_id"]] == method]
        times = [row["seconds"] for row in items if "seconds" in row]
        mean = f"{np.mean(times):.2f}" if times else "unavailable"
        lines.append(f"| {method} | {len(items)} | {sum(row['status'] == 'complete' for row in items)} "
                     f"| {sum(row['windows'] for row in items)} | {mean} |")
    lines.extend([
        "",
        f"Pilot checkpoint archives occupied {summary['pilot_checkpoint_bytes']:,} bytes. The projections "
        "scale elapsed time and compressed archive size by container frame counts. Resolution, video duration, "
        "I/O and unsuccessful extraction can shift the actual throughput. The all-primary scenario includes "
        "unpiloted DeepFakeDetection and is a rough capacity scenario.",
        "",
        "| Scenario | Videos | Worker hours | Wall hours | Checkpoint MiB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, estimate in summary["projections"].items():
        wall = estimate["wall_hours_by_container_frames"]
        worker = estimate["worker_hours_by_container_frames"]
        size = estimate["storage_bytes_by_container_frames"]
        vals = [f"{value:.2f}" if value is not None else "unavailable"
                for value in (worker, wall, size / 2**20 if size is not None else None)]
        lines.append(f"| {name} | {estimate['videos']} | {' | '.join(vals)} |")
    failures = [row for row in batch["results"] if row["status"] != "complete"]
    lines.extend(["", "## Readiness", ""])
    if failures:
        lines.extend(f"- `{row['video_id']}` ended with `{row['status']}`. {row.get('error', '')}" for row in failures)
        lines.append("")
    lines.extend([
        "The batch pipeline can process eligible FaceForensics++ rows by official partition after pilot review. "
        "Review flagged manifest rows and keep the DeepFakeDetection and extra NeuralTextures files outside the "
        "training-ready selection. Phase 4 ends with the pilot.",
        "",
        "Per-video paths, metadata mismatches, flags, processing times, coverage and failures are available "
        "in the manifest, audit, excluded-folder comparison and pilot summary alongside this report.",
        "",
    ])
    (directory / "PHASE4_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)


if __name__ == "__main__":
    main()

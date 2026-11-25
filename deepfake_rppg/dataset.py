"""Read-only FaceForensics++ inventory, metadata reconciliation and split assignment."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path, PurePosixPath
import re
from concurrent.futures import ThreadPoolExecutor

from .fusion import digest

METHODS = ("Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures")
DFD = "DeepFakeDetection"
SPLITS = ("train", "val", "test")
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
GROUP_FLAGS = {"pair_not_in_official_splits", "unresolved_source", "cross_split_component"}
BLOCKED_FLAGS = GROUP_FLAGS | {
    "probe_error", "ambiguous_path", "metadata_label_conflict", "duplicate_provenance_conflict",
    "duplicate_redundant", "unexpected_path",
}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_splits(directory):
    owners, pairs = {}, set()
    for split in SPLITS:
        for pair in json.loads((directory / f"{split}.json").read_text(encoding="utf-8")):
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(f"Invalid official pair: {pair}")
            for source in pair:
                if not re.fullmatch(r"[0-9]{3}", source) or source in owners:
                    raise ValueError(f"Invalid or repeated official source ID: {source}")
                owners[source] = split
            pairs.update((tuple(pair), tuple(reversed(pair))))
    return owners, pairs


def load_origins(path):
    if not path.is_file():
        return {}
    origins = {}
    for source, value in json.loads(path.read_text(encoding="utf-8")).items():
        upload, segment = value.split()
        if not re.fullmatch(r"[0-9]{3}", source) or not re.fullmatch(r"[\w-]{11}", upload) or not segment.isdigit():
            raise ValueError(f"Invalid official conversion mapping for {source}")
        origins[source] = upload
    return origins


def read_metadata(directory):
    entries, reports = defaultdict(list), {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if "File Path" not in (reader.fieldnames or []):
            reports[path.name] = {"summary_rows": rows}
            continue
        keys = Counter()
        for row in rows:
            key = PurePosixPath(row["File Path"].replace(chr(92), "/")).as_posix().casefold()
            keys[key] += 1
            entries[key].append((path.name, row))
        reports[path.name] = {"rows": len(rows), "duplicate_entries": {k: n for k, n in keys.items() if n > 1}}
    return entries, reports


def inspect(item):
    import cv2

    root_id, root, path, method, excluded = item
    relative = path.relative_to(root).as_posix()
    record = dict(video_id=f"{root_id}:{relative}", absolute_path=str(path), relative_path=relative,
                  root_id=root_id, label=int(method != "original"), manipulation_method=method,
                  dataset_source="Google DeepFakeDetection" if method == DFD else "FaceForensics++ (nominal C23)",
                  excluded=excluded, flags=[], frame_count=None, fps=None, duration=None,
                  width=None, height=None, resolution=None, codec_fourcc=None,
                  file_size=None, modified_ns=None, sha256=None)
    try:
        stat = path.stat()
        record.update(file_size=stat.st_size, modified_ns=stat.st_mtime_ns, sha256=digest(path))
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError("Container could not be opened")
            count, fps, width, height, codec = (cap.get(prop) for prop in (
                cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FOURCC,
            ))
            if not all(math.isfinite(x) and x > 0 for x in (count, fps, width, height)):
                raise ValueError("Invalid container video properties")
            fourcc = int(codec).to_bytes(4, "little").decode("latin-1").strip("\0")
            record.update(frame_count=int(count), fps=fps, duration=count / fps,
                          width=int(width), height=int(height), resolution=f"{int(width)}x{int(height)}",
                          codec_fourcc=fourcc)
        finally:
            cap.release()
    except (OSError, ValueError, OverflowError, cv2.error) as error:
        record["flags"].append("probe_error")
        record["probe_error"] = str(error)
    return record


def assign_groups(rows, owners, pairs, origins=None):
    parent = {}
    origins = origins or {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(left, right):
        a, b = sorted((find(left), find(right)))
        parent[b] = a

    originals = {Path(row["relative_path"]).stem for row in rows
                 if row["manipulation_method"] == "original" and not row["excluded"]}
    for first, second in sorted(pairs):
        join(f"source:{first}", f"source:{second}")
    for source, upload in origins.items():
        join(f"source:{source}", f"upload:{upload}")
    for row in rows:
        row["flags"] = [flag for flag in row["flags"] if flag not in GROUP_FLAGS]
        node = f"video:{row['video_id']}"
        find(node)
        stem = Path(row["relative_path"]).stem
        method = row["manipulation_method"]
        ids = stem.split("_")
        recognized = (method == "original" and re.fullmatch(r"[0-9]{3}", stem)
                      or method in METHODS and re.fullmatch(r"[0-9]{3}_[0-9]{3}", stem))
        row["source_video_ids"] = ids if recognized and set(ids) <= originals else []
        row["original_upload_ids"] = sorted({origins[source] for source in row["source_video_ids"]
                                              if source in origins})
        row["source_relationship"] = ("official_pair" if recognized and tuple(ids) in pairs
                                      else "matching_original_filenames" if row["source_video_ids"]
                                      else "unresolved")
        # Group even unverified numeric references conservatively.
        if recognized:
            for source in ids:
                join(node, f"source:{source}")
        if row["sha256"]:
            join(node, f"sha256:{row['sha256']}")
        if method in METHODS and pairs and tuple(ids) not in pairs:
            row["flags"].append("pair_not_in_official_splits")
        if method != DFD and not row["source_video_ids"]:
            row["flags"].append("unresolved_source")

    partitions, holds = defaultdict(set), set()
    for source, split in owners.items():
        partitions[find(f"source:{source}")].add(split)
    for row in rows:
        group = find(f"video:{row['video_id']}")
        row["group_id"] = group
        if row["manipulation_method"] == DFD and not row["excluded"]:
            holds.add(group)
    for row in rows:
        group = row["group_id"]
        choices = partitions[group]
        if len(choices) > 1:
            row["flags"].append("cross_split_component")
        row["partition"] = next(iter(choices)) if len(choices) == 1 and group not in holds else "holdout"
        row["validation_status"] = ("excluded" if row["excluded"] else
                                    "review" if BLOCKED_FLAGS.intersection(row["flags"]) else
                                    "metadata_warning" if row["flags"] else "audited")


def reconcile(rows, entries):
    paths = defaultdict(list)
    hashes = defaultdict(list)
    for row in rows:
        if not row["excluded"]:
            paths[row["relative_path"].casefold()].append(row)
        if row["sha256"]:
            hashes[row["sha256"]].append(row)
    for path, matches in paths.items():
        for row in matches:
            metadata = entries.get(path, [])
            row["metadata_files"] = [name for name, _ in metadata]
            if not metadata:
                row["flags"].append("unexpected_path")
            if len(matches) > 1:
                row["flags"].append("ambiguous_path")
            values = {"Label": "FAKE" if row["label"] else "REAL", "Frame Count": row["frame_count"],
                      "Width": row["width"], "Height": row["height"],
                      "Codec": "h264" if row["codec_fourcc"] in ("h264", "avc1", "H264") else row["codec_fourcc"]}
            mismatches = defaultdict(list)
            for name, entry in metadata:
                for field, observed in values.items():
                    if observed is not None and entry.get(field, "").strip().casefold() != str(observed).casefold():
                        mismatches[field].append(name)
                if row["file_size"] is not None:
                    try:
                        difference = abs(float(entry["File Size(MB)"]) - row["file_size"] / 2**20)
                    except (KeyError, ValueError):
                        difference = math.inf
                    if difference > 0.005001:
                        mismatches["File Size(MB)"].append(name)
            row["metadata_mismatches"] = dict(mismatches)
            if mismatches:
                row["flags"].append("metadata_mismatch")
            if "Label" in mismatches:
                row["flags"].append("metadata_label_conflict")
    duplicate_sets = {}
    for checksum, members in hashes.items():
        if len(members) < 2:
            continue
        duplicate_sets[checksum] = [row["video_id"] for row in members]
        primary = [row for row in members if not row["excluded"]]
        if len(primary) < 2:
            continue
        conflicting = len({(row["label"], row["manipulation_method"]) for row in primary}) > 1
        for index, row in enumerate(primary):
            if conflicting:
                row["flags"].append("duplicate_provenance_conflict")
            elif index:
                row["flags"].append("duplicate_redundant")
    return paths, hashes, duplicate_sets


def scan(config, output):
    roots = {key: Path(value).resolve(strict=True) for key, value in config["roots"].items()}
    output = output.resolve()
    if any(output.is_relative_to(root) for root in roots.values()):
        raise ValueError("Audit output must be outside dataset roots")
    output.mkdir(parents=True, exist_ok=True)
    inventory, nonvideo, missing_dirs = [], [], []
    directories = {}
    for subset in config["subsets"]:
        root_id, folder, method = (subset[key] for key in ("root", "folder", "method"))
        directories[(root_id, folder.casefold())] = (method, subset.get("exclude", False))
        if not (roots[root_id] / folder).is_dir():
            missing_dirs.append(str(roots[root_id] / folder))
    ignored = Path(config["metadata"]).resolve()
    for root_id, root in roots.items():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_relative_to(ignored):
                continue
            relative = path.relative_to(root)
            match = next((directories[(root_id, parent.as_posix().casefold())]
                          for parent in relative.parents if (root_id, parent.as_posix().casefold()) in directories), None)
            if path.suffix.casefold() in EXTENSIONS and match:
                inventory.append((root_id, root, path, *match))
            else:
                nonvideo.append(str(path))
    return inventory, nonvideo, missing_dirs


def audit(config, output, workers):
    inventory, nonvideo, missing_dirs = scan(config, output)
    entries, metadata = read_metadata(Path(config["metadata"]))
    splits = Path(config["splits"])
    owners, pairs = load_splits(splits) if all((splits / f"{name}.json").is_file() for name in SPLITS) else ({}, set())
    origin_path = splits / "conversion_dict.json"
    origins = load_origins(origin_path)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(inspect, inventory), 1):
            rows.append(row)
            if i % 500 == 0:
                print(f"Audited {i}/{len(inventory)} containers and SHA-256 hashes", flush=True)
    paths, hashes, duplicates = reconcile(rows, entries)
    assign_groups(rows, owners, pairs, origins)
    extra = []
    for row in rows:
        if not row["excluded"]:
            continue
        name = Path(row["relative_path"]).name
        names = paths.get(f"neuraltextures/{name}".casefold(), [])
        identical = [other["video_id"] for other in hashes.get(row["sha256"], [])
                     if not other["excluded"]]
        extra.append({"video_id": row["video_id"], "sha256": row["sha256"],
                      "absolute_path": row["absolute_path"], "probe_error": row.get("probe_error"),
                      "hash_status": "hashed" if row["sha256"] else "unavailable",
                      "same_name": [other["video_id"] for other in names], "identical": identical,
                      "same_name_identical": any(other["video_id"] in identical for other in names)})
    primary = [row for row in rows if not row["excluded"]]
    write_manifest(output, primary)
    save_json(output / "excluded_neuraltextures.json", extra)
    save_json(output / "audit.json", {
        "counts": Counter(row["manipulation_method"] for row in primary),
        "labels": Counter("FAKE" if row["label"] else "REAL" for row in primary),
        "by_method": {method: {
            "resolution": Counter(row["resolution"] or "unknown" for row in primary if row["manipulation_method"] == method),
            "codec": Counter(row["codec_fourcc"] or "unknown" for row in primary if row["manipulation_method"] == method),
            "flags": Counter(flag for row in primary if row["manipulation_method"] == method for flag in row["flags"]),
        } for method in ("original", *METHODS, DFD)},
        "metadata": metadata, "missing_metadata_paths": {key: sorted({name for name, _ in entries[key]})
                                                         for key in sorted(entries.keys() - paths.keys())},
        "unexpected_paths": [row["absolute_path"] for row in primary if "unexpected_path" in row["flags"]],
        "other_files": nonvideo, "missing_directories": missing_dirs, "duplicate_hashes": duplicates,
        "official_source_ids": Counter(owners.values()), "official_directed_pairs": len(pairs),
        "official_split_sha256": {name: digest(splits / f"{name}.json") for name in SPLITS} if owners else {},
        "excluded_count": len(extra),
        **group_summary(primary, owners, origins, origin_path),
    })
    return primary


def write_manifest(output, primary):
    save_json(output / "manifest.json", primary)
    if primary:
        fields = list(dict.fromkeys(key for row in primary for key in row))
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                              for key, value in row.items()} for row in primary)


def group_summary(primary, owners, origins, origin_path):
    upload_splits = defaultdict(set)
    for source, upload in origins.items():
        if source in owners:
            upload_splits[upload].add(owners[source])
    return {
        "partitions": Counter(row["partition"] for row in primary),
        "status": Counter(row["validation_status"] for row in primary),
        "original_upload_count": len(set(origins.values())),
        "original_upload_source_count": len(origins),
        "original_upload_cross_splits": {key: sorted(values) for key, values in upload_splits.items() if len(values) > 1},
        "conversion_sha256": digest(origin_path) if origins else None,
    }


def encoder_hint(item):
    root_id, root, path, method, excluded = item
    result = {"video_id": f"{root_id}:{path.relative_to(root).as_posix()}",
              "method": method, "excluded": excluded, "x264_core": None, "crf": None}
    try:
        with path.open("rb") as stream:
            prefix = stream.read(65536)
        hint = re.search(rb"x264 - core ([0-9]+)[^\x00]{0,2000}", prefix)
        if hint:
            result["x264_core"] = int(hint[1])
            crf = re.search(rb"\bcrf=([0-9]+(?:\.[0-9]+)?) ", hint[0])
            if crf:
                result["crf"] = float(crf[1])
    except OSError as error:
        result["error"] = str(error)
    return result


def audit_encoder_hints(config, output, workers):
    inventory, _, _ = scan(config, output)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        save_json(output / "encoder_hints.json", list(pool.map(encoder_hint, inventory)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--encoder-hints-only", action="store_true")
    args = parser.parse_args()
    action = audit_encoder_hints if args.encoder_hints_only else audit
    action(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.workers)


if __name__ == "__main__":
    main()

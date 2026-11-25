"""Verify and join Phase 4.6 rPPG with aligned 187-feature expression rows."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .phase47_capture import CACHE, KEYS, OUTPUT, PHASE46, load_inputs
from .quality_experiment import write_table
from .expressions import SELECTED
from .window_features import expression_features

CORE_KEYS = ("clip_id", "split_group", "partition", "window_index", "window_start_s", "window_end_s_exclusive")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV feature names in {path}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"Unexpected CSV row width in {path}")
        return rows


def index_rows(rows):
    indexed = {}
    for row in rows:
        if row["clip_id"] in indexed:
            raise ValueError("Duplicate selected video observation")
        indexed[row["clip_id"]] = row
    return indexed


def match(row, window, keys):
    if any(row[key] != window[key] for key in keys):
        raise ValueError(f"Window identity mismatch for {window['clip_id']}")


def numeric(row, fields):
    values = np.array([float(row[key]) if row[key] not in (None, "") else np.nan for key in fields])
    if np.isinf(values).any():
        raise ValueError("Infinite classifier input")
    return values


def verify_families(items):
    owners = {}
    for item in items:
        for source_type in ("source_video_ids", "original_upload_ids", "sha256"):
            values = item.get(source_type, [])
            if isinstance(values, str):
                values = [values]
            for value in values or []:
                key = source_type, value
                if owners.setdefault(key, item["group_id"]) != item["group_id"]:
                    raise ValueError(f"Source crosses source-family folds: {key}")


def integrity(names, matrix):
    """Describe global redundancy without using these findings to fit any fold."""
    details = []
    for i, name in enumerate(names):
        column = matrix[:, i]
        finite = np.isfinite(column)
        previous = next((names[j] for j in range(i)
                         if np.array_equal(column, matrix[:, j], equal_nan=True)), "")
        details.append(dict(feature=name, available=int(finite.sum()), missing=int((~finite).sum()),
                            constant_when_observed=int(finite.any() and np.ptp(column[finite]) == 0),
                            all_missing=int(not finite.any()), exact_duplicate_of=previous))
    return details


def load_fusion():
    items, windows, _, revision = load_inputs()
    verify_families(items)
    schema = json.loads((PHASE46 / "feature_schema.json").read_text(encoding="utf-8"))
    groups = schema["features"]
    rppg = [key for fields in groups.values() for key in fields]
    expanded = read_csv(PHASE46 / "rppg_consistency_v1.csv")
    selected = {method: index_rows(row for row in expanded
                                  if row["method"] == method and row["variant"] == "reference")
                for method in ("POS", "GREEN")}
    original = index_rows(read_csv(PHASE46 / "baseline_rppg69.csv"))
    legacy = [key for key in next(iter(original.values())) if key.startswith("rppg_") and key not in KEYS]
    expression = list(expression_features(np.arange(3) / 30, np.full((3, len(SELECTED)), np.nan), 30))
    if (len(rppg), len(expression), len(legacy), len(set(rppg + expression + legacy))) != (89, 187, 69, 345):
        raise ValueError("Expected unique 89, 187 and 69 feature names")
    for table in (*selected.values(), original):
        if table.keys() != windows.keys():
            raise ValueError("An ablation has a different video inventory")
    rows, legacy_rows, green_rows, alignment = [], [], [], []
    for item in items:
        clip = item["video_id"]
        window = windows[clip]
        pos, green, old = selected["POS"][clip], selected["GREEN"][clip], original[clip]
        for row in (pos, green):
            match(row, window, CORE_KEYS)
            if row["schema"] != schema["version"] or row["manipulation"] != item["manipulation_method"]:
                raise ValueError("Expanded schema or manifest metadata mismatch")
        match(old, window, KEYS)
        token = sha256(clip.encode()).hexdigest()
        checkpoint = json.loads((CACHE / f"{token}.json").read_text(encoding="utf-8"))
        if checkpoint["revision"] != revision or checkpoint["source_sha256"] != item["sha256"]:
            raise ValueError("Expression checkpoint provenance mismatch")
        match(checkpoint["keys"], window, KEYS)
        au = checkpoint["features"]
        if set(au) != set(expression) or checkpoint["aligned_frames"] != int(window["decoded_frames"]):
            raise ValueError("Invalid expression schema or frame support")
        identity = dict(**{key: window[key] for key in KEYS}, label=item["label"],
                        manipulation=item["manipulation_method"])
        rows.append(dict(**identity, **dict(zip(rppg, numeric(pos, rppg), strict=True)),
                         **dict(zip(expression, numeric(au, expression), strict=True))))
        legacy_rows.append(dict(**identity, **dict(zip(legacy, numeric(old, legacy), strict=True)),
                                **dict(zip(expression, numeric(au, expression), strict=True))))
        green_rows.append(dict(**identity, **dict(zip(rppg, numeric(green, rppg), strict=True))))
        alignment.append(dict(**identity, source_sha256=item["sha256"], frames_verified=checkpoint["aligned_frames"],
                              expression_coverage=rows[-1]["expr_all_selected_coverage"],
                              capture_seconds=checkpoint["seconds"]))
    signal = [key for key in rppg if key not in groups["reliability"]]
    definitions = {
        "rppg69": (legacy_rows, legacy), "rppg_signal63": (rows, signal),
        "reliability26": (rows, groups["reliability"]), "expression187": (rows, expression),
        "fusion256": (legacy_rows, legacy + expression), "fusion276": (rows, rppg + expression),
        "fusion250": (rows, signal + expression), "green89": (green_rows, rppg),
    }
    matrices = {arm: np.array([[row[key] for key in fields] for row in source])
                for arm, (source, fields) in definitions.items()}
    write_table(OUTPUT, "fusion276", rows)
    write_table(OUTPUT, "fusion256_reference", legacy_rows)
    write_table(OUTPUT, "alignment", alignment)
    schema_document = dict(version="fusion276_v1", rppg_version=schema["version"], expanded_count=276,
                           rppg=groups, expression=expression, baseline=legacy,
                           ablations={arm: fields for arm, (_, fields) in definitions.items()},
                           metadata_fields=(*KEYS, "label", "manipulation"))
    (OUTPUT / "feature_schema.json").write_text(json.dumps(schema_document, indent=2) + "\n", encoding="utf-8")
    audits, availability = [], []
    methods = np.array([row["manipulation"] for row in rows])
    for arm, (_, fields) in definitions.items():
        matrix = matrices[arm]
        audits.extend(dict(arm=arm, **item) for item in integrity(fields, matrix))
        for method in sorted(set(methods)):
            values = matrix[methods == method]
            availability.extend(dict(arm=arm, manipulation=method, feature=field, videos=len(values),
                                     available=int(np.isfinite(values[:, j]).sum()))
                                for j, field in enumerate(fields))
    write_table(OUTPUT, "feature_integrity", audits)
    write_table(OUTPUT, "feature_availability", availability)
    return rows, matrices, schema_document

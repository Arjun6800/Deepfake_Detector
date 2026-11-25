"""Prepare the larger training cohort without opening any video for decoding."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np

from .dataset import METHODS
from .fusion import digest
from .phase47_capture import load_inputs
from .phase47_fusion import read_csv, verify_families

OUTPUT = Path("outputs/test11")
AUDITED = Path("outputs/test7/manifest.json")
RGB = Path("outputs/test8/training")
PHASE47 = Path("outputs/test10")
SEED = 480
FAMILIES = 50
CLASSES = ("original", *METHODS)


def eligible_families(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["partition"] == "train":
            grouped[row["group_id"]].append(row)
    result = {}
    for family, members in grouped.items():
        if len(members) != 12 or Counter(row["manipulation_method"] for row in members) != dict.fromkeys(CLASSES, 2):
            continue
        if any(row["excluded"] or row["validation_status"] not in ("audited", "metadata_warning")
               or row["dataset_source"] != "FaceForensics++ (nominal C23)" or (row["duration"] or 0) < 8
               for row in members):
            continue
        originals = [row for row in members if row["manipulation_method"] == "original"]
        original_ids = {Path(row["relative_path"]).stem for row in originals}
        if len(original_ids) != 2 or any(row["label"] != 0 or set(row["source_video_ids"]) !=
                                         {Path(row["relative_path"]).stem} for row in originals):
            continue
        first, second = sorted(original_ids)
        directions = {f"{first}_{second}", f"{second}_{first}"}
        if any({Path(row["relative_path"]).stem for row in members
                if row["manipulation_method"] == method and row["label"] == 1
                and set(row["source_video_ids"]) == original_ids} != directions for method in METHODS):
            continue
        result[family] = sorted(members, key=lambda row: (CLASSES.index(row["manipulation_method"]), row["video_id"]))
    return result


def select(rows, previous, families=FAMILIES, seed=SEED):
    eligible = eligible_families(rows)
    old_by_id = {row["video_id"]: row for row in previous}
    retained = sorted({row["group_id"] for row in previous})
    if any(family not in eligible for family in retained):
        raise ValueError("An existing family no longer meets training-only eligibility")
    if set(old_by_id) != {row["video_id"] for family in retained for row in eligible[family]}:
        raise ValueError("Existing cohort must contain complete retained families")
    for family in retained:
        for row in eligible[family]:
            old = old_by_id[row["video_id"]]
            if any(row[key] != old[key] for key in ("partition", "group_id", "sha256")):
                raise ValueError("Existing cohort provenance changed")
    pool = sorted(set(eligible) - set(retained))
    if not len(retained) <= families <= len(eligible):
        raise ValueError("Insufficient eligible complete families")
    sampled = sorted(random.Random(seed).sample(pool, families - len(retained)))
    selected = [dict(row, phase48_cohort="retained" if family in retained else "new")
                for family in (*retained, *sampled) for row in eligible[family]]
    if len({row["video_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate selected video")
    verify_families(selected)
    return selected, len(eligible)


def inspect_checkpoints(previous, windows, settings, revision):
    results = []
    for item in previous:
        clip = item["video_id"]
        token = sha256(clip.encode()).hexdigest()
        window = windows[clip]
        expected = sha256((item["sha256"] + settings["revision"]).encode()).hexdigest()
        rgb_path = RGB / "observations" / f"{token}.npz"
        au_path = PHASE47 / "expression_checkpoints" / f"{token}.json"
        with np.load(rgb_path, allow_pickle=False) as rgb:
            rgb_ok = (str(rgb["key"]) == expected and len(rgb["times"]) == item["frame_count"]
                      and rgb["baseline"].shape == (item["frame_count"], 3, 3))
        waveform_path = Path("outputs/test9") / window["archive"]
        with np.load(waveform_path, allow_pickle=False) as waveform:
            grid = waveform["time_s"]
            wave_ok = (len(grid) == int(window["rppg_grid_samples"]) and grid[0] == float(window["window_start_s"])
                       and waveform["POS__reference__wave"].shape == (len(grid), 4))
        au = json.loads(au_path.read_text(encoding="utf-8"))
        au_ok = (au["revision"] == revision and au["source_sha256"] == item["sha256"]
                 and all(window.get(key) == value for key, value in au["keys"].items())
                 and au["aligned_frames"] == int(window["decoded_frames"]) and len(au["features"]) == 187)
        results.append(dict(video_id=clip, rgb_reusable=bool(rgb_ok), centered_rppg_reusable=bool(rgb_ok and wave_ok),
                            centered_expression_reusable=bool(au_ok), rgb_bytes=rgb_path.stat().st_size,
                            waveform_bytes=waveform_path.stat().st_size, expression_bytes=au_path.stat().st_size))
    return results


def window_budget(frames, fps):
    """Estimate the regular four-second-hop windows plus the archived center."""
    width, step = round(8 * fps), round(4 * fps)
    if width < 1 or step < 1 or frames < width:
        return 0, 0
    last = frames - width
    regular = len(range(0, last + 1, step))
    if regular < 2:
        return regular, 1
    return regular, regular + int((last // 2) % step != 0)


def plan():
    previous, windows, settings, revision = load_inputs()
    rows = json.loads(AUDITED.read_text(encoding="utf-8"))
    selected, eligible = select(rows, previous)
    reuse = inspect_checkpoints(previous, windows, settings, revision)
    fusion = read_csv(PHASE47 / "fusion276.csv")
    if {row["clip_id"] for row in fusion} != windows.keys() or len(fusion) != len(windows):
        raise ValueError("Phase 4.7 fused inventory does not match the retained cohort")
    available = {row["video_id"]: Path(row["absolute_path"]).is_file() for row in selected}
    capture = json.loads((RGB / "capture_summary.json").read_text(encoding="utf-8"))
    expression = json.loads((PHASE47 / "expression_capture_summary.json").read_text(encoding="utf-8"))
    new = [row for row in selected if row["phase48_cohort"] == "new"]
    rgb_frames = sum(row["frame_count"] for row in previous)
    au_frames = sum(int(row["last_frame_index"]) + 1 for row in windows.values())
    new_frames = sum(row["frame_count"] for row in new)
    old_frames = sum(row["frame_count"] for row in previous)
    all_frames = old_frames + new_frames
    stage_rgb_seconds = capture["seconds"] * new_frames / rgb_frames
    stage_au_seconds = expression["seconds"] * all_frames / au_frames
    budgets = [window_budget(row["frame_count"], row["fps"]) for row in selected]
    estimates = dict(eligible_complete_families=eligible, videos=len(selected), families=FAMILIES,
                     seed=SEED, retained_videos=len(previous), new_videos=len(new), paths_visible=sum(available.values()),
                     prior_rgb_frames=rgb_frames, prior_au_inference_frames=au_frames, new_frames=new_frames,
                     all_frames=all_frames, prior_rgb_wall_s=capture["seconds"], prior_au_wall_s=expression["seconds"],
                     rgb_reference_equivalent_s=stage_rgb_seconds, au_reference_equivalent_s=stage_au_seconds,
                     projected_combined_stage_hours=(stage_rgb_seconds + stage_au_seconds) / 3600,
                     estimated_new_rgb_bytes=round(sum(row["rgb_bytes"] for row in reuse) * new_frames / rgb_frames),
                     predicted_regular_windows=sum(regular for regular, _ in budgets),
                     predicted_windows_including_center=sum(total for _, total in budgets),
                     multiwindow_eligible_videos=sum(regular >= 2 for regular, _ in budgets),
                     manifest_sha256=digest(AUDITED))
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "selection.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "planning_estimates.json").write_text(json.dumps(estimates, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "checkpoint_reuse_audit.json").write_text(json.dumps(reuse, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(estimates, indent=2))
    print("Checkpoint reuse:", {key: sum(row[key] for row in reuse) for key in
                               ("rgb_reusable", "centered_rppg_reusable", "centered_expression_reusable")})
    print("Invisible sources:", [clip for clip, visible in available.items() if not visible])


if __name__ == "__main__":
    plan()

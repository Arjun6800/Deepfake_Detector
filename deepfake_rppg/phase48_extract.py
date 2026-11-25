"""Offline, resumable extraction for the approved Phase 4.8 cohort."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from hashlib import sha256
import json
from pathlib import Path
from shutil import disk_usage
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import BadZipFile

import numpy as np

from .batch import roi_runs
from .consistency_experiment import extract
from .expressions import ExpressionSampler, SELECTED, blendshape_scores
from .fusion import digest
from .landmarks import DEFAULT_MODEL
from .phase47_fusion import index_rows, read_csv, verify_families
from .phase48_plan import AUDITED, OUTPUT, PHASE47, RGB, eligible_families
from .quality_capture import ReducedDetector, ResearchSampler
from .reliability import ReliabilitySampler
from .video import choose_timing, read_video
from .window_features import expression_features, rppg_features, snap_rgb

CHECKPOINTS = OUTPUT / "checkpoints"
MIN_FREE = 2 * 1024**3
BUFFER = 64 * 1024**2
WORKERS = 5


class DiskPause(RuntimeError):
    pass


def ensure_space():
    if disk_usage(OUTPUT).free < MIN_FREE + BUFFER:
        raise DiskPause("Free disk space reached the 2 GiB reserve plus the 64 MiB write buffer")


def atomic(path, writer, reserve=True):
    if reserve:
        ensure_space()
    with NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        staged = Path(stream.name)
        try:
            writer(stream)
            stream.close()
            staged.replace(path)
        finally:
            stream.close()
            staged.unlink(missing_ok=True)


def save(path, data, reserve=True):
    atomic(path, lambda stream: stream.write(json.dumps(data, allow_nan=False).encode("utf-8")), reserve)


def token(clip):
    return sha256(clip.encode()).hexdigest()


def approved_selection():
    selected = json.loads((OUTPUT / "selection.json").read_text(encoding="utf-8"))
    if len(selected) != 600 or any(row["partition"] != "train" for row in selected):
        raise ValueError("Expected exactly the approved 600 training videos")
    training = [row for row in json.loads(AUDITED.read_text(encoding="utf-8")) if row["partition"] == "train"]
    eligible = eligible_families(training)
    groups = {row["group_id"] for row in selected}
    if len(groups) != 50 or not groups <= eligible.keys():
        raise ValueError("Expected 50 complete audited training families")
    expected = {row["video_id"]: row for group in groups for row in eligible[group]}
    previous = {row["clip_id"] for row in read_csv(PHASE47 / "alignment.csv")}
    if len(previous) != 96 or len({row["video_id"] for row in selected}) != 600:
        raise ValueError("Invalid retained or selected cohort")
    for row in selected:
        clip = row["video_id"]
        cohort = "retained" if clip in previous else "new"
        if clip not in expected or row != dict(expected[clip], phase48_cohort=cohort):
            raise ValueError(f"Selection differs from audited manifest for {clip}")
    if not previous <= expected.keys():
        raise ValueError("The approved selection must retain all 96 videos")
    verify_families(selected)
    return selected


def signature():
    sources = ("phase48_extract.py", "quality_capture.py", "video.py", "landmarks.py", "face.py",
               "reliability.py", "expressions.py", "window_features.py", "signal.py", "batch.py",
               "quality_algorithms.py", "consistency_experiment.py", "consistency.py")
    metadata = [OUTPUT / "selection.json", RGB / "capture_settings.json", DEFAULT_MODEL]
    metadata += [PHASE47 / name for name in ("feature_schema.json", "fusion276.csv", "fusion256_reference.csv")]
    metadata += [Path("outputs/test9/rppg_consistency_v1.csv")]
    revision = "".join(digest(path) for path in metadata + [Path(__file__).with_name(name) for name in sources])
    return token(revision)


def windows(times, fps):
    """Use the Phase 4.6 center and deduplicate complete windows on the native grid."""
    grid = np.arange(int(np.floor(times[-1] * fps + 1e-8)) + 1) / fps
    width, hop = round(8 * fps), round(4 * fps)
    last = len(grid) - width
    if last < 0:
        raise ValueError("Decoded clip cannot support an eight-second window")
    center = last // 2
    regular = list(range(0, last + 1, hop))
    offsets = sorted(set([*regular, center])) if len(regular) >= 2 else [center]
    return [(offset == center, grid[offset:offset + width]) for offset in offsets]


class JointSampler(ResearchSampler):
    def __init__(self, detector_width):
        ReliabilitySampler.__init__(self, DEFAULT_MODEL, output_blendshapes=True)
        self.target = ReducedDetector(detector_width)
        self.template = None

    def sample(self, frame):
        observation = self.observe(frame)
        rgb, source, geometry, light = (observation[i] for i in (0, 3, 6, 8))
        scores = blendshape_scores(self.blendshapes) if observation[-1] == "accepted" else {}
        return rgb, source, geometry, light, [scores.get(key, np.nan) for key in SELECTED]


def decode(item, settings):
    retained = item["phase48_cohort"] == "retained"
    detector_width = settings["detector_width"]
    source = Path(item["absolute_path"])
    frames = 0
    original = {}
    if retained:
        with np.load(RGB / "observations" / f"{token(item['video_id'])}.npz", allow_pickle=False) as archive:
            if str(archive["key"]) != token(item["sha256"] + settings["revision"]):
                raise ValueError("Retained RGB provenance mismatch")
            original = {key: archive[key] for key in ("baseline", "base_codes", "positions", "times",
                                                       "fps", "geometry", "light")}
        if len(windows(original["times"], round(float(original["fps"]), 12))) == 1:
            original["scores"] = np.empty((0, len(SELECTED)))
            return original
        sampler = ExpressionSampler(DEFAULT_MODEL)
        sampler.target = ReducedDetector(detector_width)
    else:
        sampler = JointSampler(detector_width)
    with sampler:
        def observe(frame):
            nonlocal frames
            if frames % 32 == 0:
                ensure_space()
            frames += 1
            if retained:
                _, scores = sampler.sample(frame)
                return [scores.get(key, np.nan) for key in SELECTED]
            return sampler.sample(frame)

        positions, observations, nominal, _ = read_video(source, observe)
    if retained:
        if not np.array_equal(positions, original["positions"]):
            raise ValueError("Expression timestamps changed from retained RGB archive")
        return dict(original, scores=np.asarray(observations, dtype=float))
    timing = choose_timing(positions, nominal)
    if timing.fps is None or timing.fps <= 6 or len(positions) != item["frame_count"]:
        raise ValueError("Incomplete decode or unsupported video timebase")
    fields = ("baseline", "base_codes", "geometry", "light", "scores")
    return dict(zip(fields, map(np.asarray, zip(*observations, strict=True)), strict=True),
                positions=positions, times=timing.timestamps, fps=timing.fps)


def clean(values):
    if any(value is not None and np.isinf(value) for value in values.values()):
        raise ValueError("Infinite extracted feature")
    return {key: float(value) if value is not None and np.isfinite(value) else None for key, value in values.items()}


def feature_rows(item, data, schema, retained_rows=None):
    times, fps = data["times"], round(float(data["fps"]), 12)
    rgb, groups = data["baseline"], roi_runs(data["base_codes"])
    all_rows = []
    names = schema["ablations"]
    for index, (center, grid) in enumerate(windows(times, fps)):
        ensure_space()
        start, end = float(grid[0]), float(grid[0] + 8)
        frames = (times >= start - 1e-8) & (times < grid[-1] + 1 / fps - 1e-8)
        frame_ids = np.flatnonzero(frames)
        if not len(frame_ids):
            raise ValueError("Window contains no decoded observations")
        if center and retained_rows is not None:
            pos_row, legacy_row, green_row = retained_rows
            if (float(pos_row["window_start_s"]) != start or float(pos_row["window_end_s_exclusive"]) != end
                    or int(pos_row["decoded_frames"]) != len(frame_ids)
                    or int(pos_row["first_frame_index"]) != frame_ids[0]
                    or int(pos_row["last_frame_index"]) != frame_ids[-1]
                    or int(pos_row["rppg_grid_samples"]) != len(grid)):
                raise ValueError("Retained center does not match window or observation support")
            def select(row, fields):
                return clean({key: float(row[key]) if row[key] not in (None, "") else np.nan for key in fields})

            pos = select(pos_row, names["fusion276"])
            green = select(green_row, names["green89"])
            legacy = select(legacy_row, names["rppg69"])
        else:
            snapped, ids = snap_rgb(times[frames], rgb[frames], groups[frames], grid, fps)
            pos, green = [clean(extract(method, snapped, ids, fps, None, None)[0]) for method in ("POS", "GREEN")]
            pos.update(clean(expression_features(times[frames], data["scores"][frames], fps)))
            legacy = clean(rppg_features(rgb[frames], snapped, ids, fps))
        if (set(pos) != set(names["fusion276"]) or set(green) != set(names["green89"])
                or set(legacy) != set(names["rppg69"])):
            raise ValueError("Feature names differ from the frozen Phase 4.7 schema")
        geometry = data["geometry"][frames]
        widths = geometry[:-1, 2]
        displacement = np.linalg.norm(np.diff(geometry[:, :2] + geometry[:, 2:] / 2, axis=0), axis=1)
        valid = (widths > 0) & np.isfinite(displacement)
        light = data["light"][frames]
        light = light[np.isfinite(light)]
        audit = clean(dict(motion=float(np.median(displacement[valid] / widths[valid]) * fps) if valid.any() else np.nan,
                           light_cv=float(light.std() / light.mean()) if len(light) and light.mean() > 0 else np.nan,
                           bits_per_pixel=8 * item["file_size"] / (item["width"] * item["height"] * len(times))))
        all_rows.append(dict(window=index, center=center, start_s=start, end_s=end, first_frame=int(frame_ids[0]),
                             last_frame=int(frame_ids[-1]), frames=len(frame_ids), grid_samples=len(grid),
                             pos=pos, green=green, legacy=legacy, audit=audit))
    return all_rows


def read_checkpoint(path, key, schema):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        windows_ = row["windows"]
        if (row["key"] == key and windows_ and sum(w["center"] for w in windows_) == 1
                and all(set(w["pos"]) == set(schema["ablations"]["fusion276"])
                        and set(w["green"]) == set(schema["ablations"]["green89"])
                        and set(w["legacy"]) == set(schema["ablations"]["rppg69"]) for w in windows_)):
            return row
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def process(task):
    import cv2

    item, settings, key, schema, retained_rows = task
    cv2.setNumThreads(1)
    clip = item["video_id"]
    started = perf_counter()
    path = CHECKPOINTS / f"{token(clip)}.json"
    try:
        if read_checkpoint(path, key, schema) is not None:
            return dict(clip_id=clip, status="cached", seconds=0)
        ensure_space()
        if digest(Path(item["absolute_path"])) != item["sha256"]:
            raise ValueError("Source bytes differ from audited SHA-256")
        observation_path = CHECKPOINTS / f"{token(clip)}.npz"
        data = None
        if observation_path.exists():
            try:
                with np.load(observation_path, allow_pickle=False) as archive:
                    if str(archive["key"]) == key:
                        data = dict(archive)
            except (OSError, ValueError, KeyError, BadZipFile, EOFError):
                pass
        if data is None:
            data = decode(item, settings)
            atomic(observation_path, lambda stream: np.savez_compressed(stream, key=key, **data))
        records = feature_rows(item, data, schema, retained_rows)
        elapsed = perf_counter() - started
        save(path, dict(key=key, clip_id=clip, cohort=item["phase48_cohort"], windows=records, seconds=elapsed))
        return dict(clip_id=clip, status="complete", windows=len(records), seconds=elapsed)
    except DiskPause as error:
        return dict(clip_id=clip, status="paused", error=str(error), seconds=perf_counter() - started)
    except Exception as error:
        return dict(clip_id=clip, status="failed", error=f"{type(error).__name__}: {error}",
                    seconds=perf_counter() - started)


def run():
    items = approved_selection()
    revision = signature()
    schema = json.loads((PHASE47 / "feature_schema.json").read_text(encoding="utf-8"))
    settings = json.loads((RGB / "capture_settings.json").read_text(encoding="utf-8"))
    if digest(DEFAULT_MODEL) != settings["model_sha256"]:
        raise ValueError("Landmark model differs from the retained capture")
    pos = index_rows(read_csv(PHASE47 / "fusion276.csv"))
    legacy = index_rows(read_csv(PHASE47 / "fusion256_reference.csv"))
    green = index_rows(row for row in read_csv(Path("outputs/test9/rppg_consistency_v1.csv"))
                       if row["method"] == "GREEN" and row["variant"] == "reference")
    CHECKPOINTS.mkdir(exist_ok=True)
    tasks = []
    for row in items:
        clip = row["video_id"]
        retained_rows = None
        if row["phase48_cohort"] == "retained":
            retained_rows = (pos[clip], legacy[clip], green[clip])
            for feature in retained_rows:
                if feature["split_group"] != row["group_id"] or feature["partition"] != "train":
                    raise ValueError("Retained features have a conflicting family or partition")
        tasks.append((row, settings, token(revision + row["sha256"]), schema, retained_rows))
    started = perf_counter()
    done, successes, failures, paused = 0, 0, 0, False
    durations = []
    pending = iter(tasks)
    live = {}
    with (OUTPUT / "extraction_progress.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        with (OUTPUT / "extraction_failures.jsonl").open("a", encoding="utf-8", buffering=1) as errors:
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                while True:
                    while not paused and len(live) < WORKERS:
                        task = next(pending, None)
                        if task is None:
                            break
                        try:
                            ensure_space()
                        except DiskPause:
                            paused = True
                            break
                        live[pool.submit(process, task)] = task[0]["video_id"]
                    if not live:
                        break
                    finished, _ = wait(live, return_when=FIRST_COMPLETED)
                    for future in finished:
                        clip = live.pop(future)
                        try:
                            row = future.result()
                        except Exception as error:
                            row = dict(clip_id=clip, status="failed", error=str(error), seconds=0)
                            paused = True
                        done += 1
                        successes += row["status"] in ("cached", "complete")
                        failures += row["status"] == "failed"
                        paused |= row["status"] == "paused"
                        if row["status"] == "complete":
                            durations.append(row["seconds"])
                        eta = round(np.mean(durations) * (len(tasks) - done) / WORKERS) if durations else None
                        row.update(done=done, total=len(tasks), eta_seconds=eta)
                        print(json.dumps(row), file=log)
                        if row["status"] in ("failed", "paused"):
                            print(json.dumps(row), file=errors)
                        print(f"{done}/{len(tasks)} {clip} {row['status']} ETA {eta if eta is not None else '?'} s",
                              flush=True)
    save(OUTPUT / "extraction_summary.json", dict(completed=successes, total=len(tasks), failures=failures,
                                                 paused=paused, seconds=perf_counter() - started,
                                                 revision=revision), reserve=False)
    return 0 if successes == len(tasks) else 2


def main():
    # The Windows file lock releases automatically if the process exits or is interrupted.
    import msvcrt

    with (OUTPUT / "extraction.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return run()


if __name__ == "__main__":
    raise SystemExit(main())

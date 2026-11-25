import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from deepfake_rppg.batch import FEATURE_COUNTS, commit, extract, restore, table_csv
from deepfake_rppg.dataset import assign_groups, audit, load_splits


def row(path, method, checksum):
    return dict(video_id=path, relative_path=path, manipulation_method=method,
                excluded=False, sha256=checksum, flags=[])


class DatasetTests(unittest.TestCase):
    def test_roots_metadata_missing_and_excluded_duplicate(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            roots = {"a": base / "a", "b": base / "b"}
            for relative in ("a/Deepfakes/000_003.mp4", "b/original/000.mp4", "b/original/003.mp4",
                             "b/NeuralTextures/000_003.mp4", "b/original/NeuralTextures/000_003.mp4"):
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            metadata = base / "metadata"
            metadata.mkdir()
            (metadata / "all.csv").write_text("File Path,Label\noriginal/000.mp4,REAL\n"
                                               "original/000.mp4,REAL\noriginal/004.mp4,REAL\n"
                                               "Deepfakes/000_003.mp4,FAKE\n"
                                               "NeuralTextures/000_003.mp4,FAKE\n")
            config = {"roots": roots, "metadata": metadata, "splits": base / "splits",
                      "subsets": [{"root": root, "folder": folder, "method": method, "exclude": excluded}
                                  for root, folder, method, excluded in (
                                      ("a", "Deepfakes", "Deepfakes", False),
                                      ("b", "original", "original", False),
                                      ("b", "NeuralTextures", "NeuralTextures", False),
                                      ("b", "original/NeuralTextures", "NeuralTextures", True),
                                  )]}

            def probe(item):
                root_id, root, path, method, excluded = item
                relative = path.relative_to(root).as_posix()
                result = row(relative, method, path.name if method != "Deepfakes" else "fake")
                return dict(result, video_id=f"{root_id}:{relative}", absolute_path=str(path),
                            excluded=excluded, label=int(method != "original"), frame_count=None,
                            width=None, height=None, codec_fourcc=None, file_size=None, resolution=None)

            output = base / "out"
            with patch("deepfake_rppg.dataset.inspect", side_effect=probe):
                manifest = audit(config, output, 2)
            self.assertEqual(len(manifest), 4)
            self.assertEqual(len({item["video_id"] for item in manifest}), 4)
            self.assertEqual({Path(item["absolute_path"]).parts[-3] for item in manifest}, {"a", "b"})
            report = json.loads((output / "audit.json").read_text())
            self.assertIn("original/004.mp4", report["missing_metadata_paths"])
            self.assertEqual(report["metadata"]["all.csv"]["duplicate_entries"], {"original/000.mp4": 2})
            extra, = json.loads((output / "excluded_neuraltextures.json").read_text())
            self.assertTrue(extra["same_name_identical"])

    def test_two_source_ids_and_transitive_hash_conflict(self):
        rows = [row("original/000.mp4", "original", "x"),
                row("original/003.mp4", "original", "y"),
                row("Deepfakes/000_003.mp4", "Deepfakes", "z"),
                row("original/101.mp4", "original", "z")]
        owners = {"000": "train", "003": "train", "101": "test"}
        assign_groups(rows, owners, {("000", "003")})
        self.assertEqual(rows[2]["source_video_ids"], ["000", "003"])
        self.assertEqual(len({item["group_id"] for item in rows}), 1)
        self.assertEqual({item["partition"] for item in rows}, {"holdout"})

    def test_dfd_never_inherits_a_train_partition(self):
        rows = [row("original/000.mp4", "original", "x"),
                row("DeepFakeDetection/000_003.mp4", "DeepFakeDetection", "x")]
        assign_groups(rows, {"000": "train"}, set())
        self.assertEqual(rows[1]["source_video_ids"], [])
        self.assertEqual({item["partition"] for item in rows}, {"holdout"})

    def test_shared_original_upload_is_held_out_across_official_partitions(self):
        rows = [row("original/000.mp4", "original", "x"),
                row("original/003.mp4", "original", "y"),
                row("Deepfakes/000_003.mp4", "Deepfakes", "z"),
                row("original/101.mp4", "original", "w")]
        assign_groups(rows, {"000": "train", "003": "train", "101": "val"}, {("000", "003")},
                      {"003": "some_upload", "101": "some_upload"})
        self.assertEqual(len({item["group_id"] for item in rows}), 1)
        self.assertEqual({item["partition"] for item in rows}, {"holdout"})
        self.assertEqual(rows[2]["original_upload_ids"], ["some_upload"])

    def test_repeated_official_source_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, data in (("train", [["001", "002"]]), ("val", [["001", "003"]]), ("test", [])):
                (folder / f"{name}.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_splits(folder)


class CheckpointTests(unittest.TestCase):
    def test_complete_window_keys_and_features_serialize_to_checkpoint(self):
        from zipfile import ZipFile

        observations = [(np.full((3, 3), np.nan), np.zeros(3, dtype=int), np.full(30, np.nan))] * 240
        settings = dict(model="unused", roi="hybrid", assume_cfr=False, length=8, overlap=4)
        item = dict(absolute_path="unused", sha256="content", frame_count=240,
                    video_id="id", group_id="group")
        with patch("deepfake_rppg.batch.digest", return_value="content"), \
             patch("deepfake_rppg.batch.ReliabilitySampler"), \
             patch("deepfake_rppg.batch.read_video", return_value=(np.arange(240) / 30, observations, 30, "test")):
            metrics, files = extract(item, settings)
        self.assertEqual(metrics["status"], "complete")
        self.assertEqual(metrics["windows"], 1)
        self.assertEqual(json.loads(files["windows.json"])[0]["last_frame_index"], 239)
        self.assertEqual({name: len(fields) for name, fields in metrics["feature_columns"].items()}, FEATURE_COUNTS)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "video.zip"
            commit(output, metrics, files)
            with ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())

    def test_resume_invalidated_and_corrupt_archive(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "video.zip"
            complete = {"checkpoint_key": "current", "status": "complete"}
            files = {name: "" for name in ("windows.json", "rppg.csv", "expression.csv", "fused.csv")}
            commit(checkpoint, complete, files)
            self.assertEqual(restore(checkpoint, "current", False), complete)
            self.assertIsNone(restore(checkpoint, "changed", False))
            checkpoint.write_bytes(b"interrupted")
            self.assertIsNone(restore(checkpoint, "current", False))
            commit(checkpoint, complete, {})
            self.assertIsNone(restore(checkpoint, "current", False))
            failure = {**complete, "status": "error"}
            commit(checkpoint, failure, {})
            self.assertEqual(restore(checkpoint, "current", False), failure)
            self.assertIsNone(restore(checkpoint, "current", True))

    def test_feature_output_preserves_missing_without_identifiers(self):
        rows = [{"video_id": "source.mp4", "rppg_a": float("nan"), "expr_b": 0.0}]
        self.assertEqual(table_csv(rows, ["rppg_a", "expr_b"]), "rppg_a,expr_b\r\n,0.0\r\n")


if __name__ == "__main__":
    unittest.main()

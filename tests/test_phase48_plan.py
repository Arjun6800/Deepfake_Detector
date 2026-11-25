"""Verify selection and window budgeting without any video I/O."""
import unittest

from deepfake_rppg.phase48_plan import CLASSES, eligible_families, select, window_budget


def family(group, partition="train"):
    first, second = f"{group}a", f"{group}b"
    rows = []
    for method in CLASSES:
        names = [first, second] if method == "original" else [f"{first}_{second}", f"{second}_{first}"]
        for name in names:
            path = f"{method}/{name}.mp4"
            rows.append(dict(video_id=path, relative_path=path, group_id=group, partition=partition,
                             manipulation_method=method, label=int(method != "original"), duration=16,
                             excluded=False, validation_status="audited", sha256=path,
                             dataset_source="FaceForensics++ (nominal C23)",
                             source_video_ids=[name] if method == "original" else [first, second]))
    return rows


class Phase48PlanTests(unittest.TestCase):
    def test_retains_previous_complete_families_and_samples_training_only(self):
        rows = family("a") + family("b") + family("c") + family("d", "val") + family("e", "test")
        selected, eligible = select(rows, family("a"), families=2, seed=480)
        self.assertEqual(eligible, 3)
        self.assertEqual(len(selected), 24)
        self.assertEqual({row["partition"] for row in selected}, {"train"})
        self.assertEqual({row["group_id"] for row in selected if row["phase48_cohort"] == "retained"}, {"a"})
        self.assertEqual(selected, select(list(reversed(rows)), family("a"), families=2, seed=480)[0])

    def test_missing_direction_and_short_video_disqualify_whole_family(self):
        bad = family("a")
        bad[-1]["relative_path"] = bad[-2]["relative_path"]
        self.assertEqual(eligible_families(bad), {})
        short = family("b")
        short[0]["duration"] = 7.99
        self.assertEqual(eligible_families(short), {})
        with self.assertRaisesRegex(ValueError, "complete"):
            select(family("c"), family("c")[:-1], families=1)

    def test_center_window_estimate_counts_duplicate_only_once(self):
        self.assertEqual(window_budget(210, 30), (0, 0))
        self.assertEqual(window_budget(350, 30), (1, 1))
        self.assertEqual(window_budget(480, 30), (3, 3))
        self.assertEqual(window_budget(460, 30), (2, 3))

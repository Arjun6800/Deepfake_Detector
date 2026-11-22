"""Phase 1 fixtures use synthetic signals and mocked video observations."""

import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deepfake_rppg.__main__ import run
from deepfake_rppg.face import FaceSampler
from deepfake_rppg.signal import pos, process_signal, resample_rgb
from deepfake_rppg.video import choose_timing, read_video


FPS = 30
TIMES = np.arange(12 * FPS) / FPS
PULSE_HZ = 1.25
PULSE = np.sin(2 * np.pi * PULSE_HZ * TIMES)


def synthetic_rgb():
    drift = 1 + 0.04 * np.sin(2 * np.pi * 0.2 * TIMES)
    return drift[:, None] * (np.array([140, 100, 80]) + PULSE[:, None] * [0.3, 1.3, 0.5])


class SignalTests(unittest.TestCase):
    def test_filter_removes_drift_and_high_frequency(self):
        raw = PULSE + 3 * TIMES + 2 * np.sin(2 * np.pi * 7 * TIMES)
        output, quality = process_signal(raw, FPS)
        self.assertAlmostEqual(float(output.mean()), 0, places=10)
        self.assertAlmostEqual(float(output.std()), 1, places=10)
        self.assertGreater(np.corrcoef(output[FPS:-FPS], PULSE[FPS:-FPS])[0, 1], 0.99)
        self.assertLess(quality["spectral_concentration"], 0.35)

    def test_missing_samples_split_processing(self):
        raw = PULSE.copy()
        raw[150:210] = np.inf
        raw[160] = np.nan
        output, quality = process_signal(raw, FPS)
        self.assertTrue(np.isnan(output[150:210]).all())
        self.assertTrue(np.isfinite(output[:150]).all())
        self.assertTrue(np.isfinite(output[210:]).all())
        self.assertAlmostEqual(quality["processed_coverage"], 300 / 360)

    def test_empty_short_flat_and_invalid_rate(self):
        for input_signal in ([], [1, 2, 3], np.full(len(TIMES), 100)):
            output, quality = process_signal(input_signal, FPS)
            self.assertTrue(np.isnan(output).all())
            self.assertIsNone(quality["spectral_concentration"])
        for invalid_fps in (0, 6, np.inf, np.nan):
            with self.assertRaises(ValueError):
                process_signal(PULSE, invalid_fps)

    def test_resampling_bridges_only_short_gaps(self):
        rgb = synthetic_rgb()[:, None, :]
        rgb[90:92] = np.nan
        rgb[120:155] = np.nan
        jittered = TIMES + 0.002 * np.sin(TIMES * 3)
        grid, colors = resample_rgb(jittered, rgb, FPS)
        self.assertTrue(np.isfinite(colors[90:92]).all())
        self.assertTrue(np.isnan(colors[121:154]).all())
        np.testing.assert_allclose(np.diff(grid), 1 / FPS)

    def test_resampling_includes_exact_gap_limit_at_24_fps(self):
        times = np.arange(30) / 24
        rgb = np.broadcast_to(times[:, None, None], (30, 1, 3)).copy()
        rgb[9:14] = np.nan
        rgb[21:27] = np.nan
        grid, colors = resample_rgb(times, rgb, 24)
        np.testing.assert_allclose(colors[9:14, 0, 0], grid[9:14])
        self.assertTrue(np.isnan(colors[21:27]).all())


class PosTests(unittest.TestCase):
    def test_projection_matches_normalized_overlap_add(self):
        rgb = synthetic_rgb()[:80]
        size = 48
        projection = np.array([[0, 1, -1], [-2, 1, 1]])
        total = np.zeros(len(rgb))
        counts = np.zeros(len(rgb))
        for start in range(len(rgb) - size + 1):
            section = slice(start, start + size)
            window = rgb[section].T
            first, second = projection @ (window / window.mean(axis=1, keepdims=True))
            h = first + first.std() / second.std() * second
            total[section] += h - h.mean()
            counts[section] += 1
        np.testing.assert_allclose(pos(rgb, FPS), total / counts, atol=1e-14)

    def test_recovers_injected_pulse(self):
        output, quality = process_signal(pos(synthetic_rgb(), FPS), FPS)
        frequencies = np.fft.rfftfreq(len(output), d=1 / FPS)
        peak = frequencies[np.argmax(abs(np.fft.rfft(output)))]
        self.assertAlmostEqual(peak, PULSE_HZ)
        self.assertGreater(quality["spectral_concentration"], 0.9)
        self.assertGreater(abs(np.corrcoef(output, PULSE)[0, 1]), 0.95)

    def test_constant_black_short_and_missing(self):
        self.assertTrue(np.isnan(pos(np.zeros((100, 3)), FPS)).all())
        self.assertTrue(np.isnan(pos(np.ones((10, 3)), FPS)).all())
        np.testing.assert_array_equal(pos(np.ones((100, 3)), FPS), 0)
        rgb = synthetic_rgb()
        rgb[150:200] = np.nan
        rgb[155, 0] = np.inf
        output = pos(rgb, FPS)
        self.assertTrue(np.isnan(output[150:200]).all())
        self.assertTrue(np.isfinite(output[:150]).all())


class TimingTests(unittest.TestCase):
    def test_preserves_irregular_times_with_missing_nominal_fps(self):
        timing = choose_timing([5.0, 5.03, 5.08, 5.10], np.nan)
        np.testing.assert_allclose(timing.timestamps, [0, 0.03, 0.08, 0.10])
        self.assertAlmostEqual(timing.fps, 1 / 0.03)
        self.assertEqual(timing.source, "opencv_pos_msec")

    def test_constant_rate_fallback_is_explicit(self):
        for positions in ([0, 0, 0], [0, 0.05, 0.04], [0, np.nan, 0.08]):
            self.assertIsNone(choose_timing(positions, 25).fps)
            timing = choose_timing(positions, 25, assume_cfr=True)
            np.testing.assert_allclose(timing.timestamps, [0, 0.04, 0.08])
            self.assertEqual(timing.source, "assumed_cfr")
        self.assertIsNone(choose_timing([0, 0], 0, assume_cfr=True).fps)


class VideoAdapterTests(unittest.TestCase):
    def test_reads_frame_positions_after_decoding_and_releases_capture(self):
        capture = Mock()
        capture.read.side_effect = [(True, "first"), (True, "second"), (False, None)]
        capture.get.side_effect = [25, 500, 548]
        capture.getBackendName.return_value = "mock-opencv"
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=capture), CAP_PROP_FPS=5, CAP_PROP_POS_MSEC=0)
        with patch.dict(sys.modules, {"cv2": cv2}):
            positions, observations, fps, backend = read_video(Path("mock.avi"), str.upper)
        np.testing.assert_allclose(positions, [0.5, 0.548])
        self.assertEqual(observations, ["FIRST", "SECOND"])
        self.assertEqual((fps, backend), (25, "mock-opencv"))
        capture.release.assert_called_once()

    def test_face_sampler_returns_rgb_and_rejects_ambiguous_track(self):
        frame = np.full((128, 128, 3), [70, 110, 150], dtype=np.uint8)
        detector = Mock()
        detector.empty.return_value = False
        box = (16, 16, 96, 96)
        detector.detectMultiScale.side_effect = [[box], [box, (17, 17, 96, 96)], []]
        cv2 = SimpleNamespace(
            data=SimpleNamespace(haarcascades="mock"),
            CascadeClassifier=Mock(return_value=detector),
            cvtColor=Mock(),
            COLOR_BGR2GRAY=6,
        )
        with patch.dict(sys.modules, {"cv2": cv2}):
            sampler = FaceSampler()
        rgb, coverage, detected, geometry = sampler.sample_with_geometry(frame)
        self.assertTrue(detected)
        np.testing.assert_array_equal(geometry, box * 2)
        np.testing.assert_array_equal(rgb, np.tile([150, 110, 70], (3, 1)))
        np.testing.assert_array_equal(coverage, 1)
        for _ in range(2):
            rgb, coverage, detected, geometry = sampler.sample_with_geometry(frame)
            self.assertFalse(detected)
            self.assertTrue(np.isnan(geometry).all())
            self.assertTrue(np.isnan(rgb).all())
            np.testing.assert_array_equal(coverage, 0)

    def test_sampling_box_smooths_consecutive_detections_and_resets_on_miss(self):
        sampler = FaceSampler()
        frame = np.full((128, 128, 3), 100, dtype=np.uint8)
        with patch.object(sampler, "detector") as detector:
            detector.detectMultiScale.side_effect = [
                [(16, 16, 96, 96)], [(20, 16, 96, 96)], [], [(24, 16, 96, 96)],
            ]
            sampler.sample_with_geometry(frame)
            _, _, _, moving = sampler.sample_with_geometry(frame)
            _, _, _, missing = sampler.sample_with_geometry(frame)
            _, _, _, resumed = sampler.sample_with_geometry(frame)
        self.assertEqual(moving[0], 20)
        self.assertAlmostEqual(moving[4], 17.4)
        self.assertTrue(np.isnan(missing).all())
        np.testing.assert_array_equal(resumed[:4], resumed[4:])


class ExportTests(unittest.TestCase):
    def test_synthetic_observations_export_with_missing_face_and_full_quality(self):
        observations = [
            (np.tile(rgb, (3, 1)), np.ones(3), True, [16, 16, 96, 96] * 2)
            for rgb in synthetic_rgb()
        ]
        observations[140] = (np.full((3, 3), np.nan), np.zeros(3), False, np.full(8, np.nan))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video, output = root / "mock-input.avi", root / "outputs"
            video.touch()
            with patch("deepfake_rppg.__main__.FaceSampler"), patch(
                "deepfake_rppg.__main__.read_video",
                return_value=(TIMES, observations, FPS, "test-decoder"),
            ):
                run(video, output)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "ok")
            self.assertEqual(metadata["face_frames"], len(TIMES) - 1)
            self.assertEqual(len(metadata["signal_quality"]), 6)
            self.assertTrue((output / "waveforms.png").read_bytes().startswith(b"\x89PNG"))
            with (output / "frame_traces.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[140]["face_detected"], "0")
            self.assertEqual(rows[140]["forehead_green"], "")
            self.assertEqual(rows[140]["face_x_px"], "")
            self.assertEqual(rows[139]["face_width_px"], "96.0")
            for filename in ("raw_waveform.csv", "processed_waveform.csv"):
                with (output / filename).open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), len(TIMES))
                self.assertTrue(np.isfinite(float(rows[140]["forehead_pos"])))

    def test_unusable_clip_statuses(self):
        good = (np.full((3, 3), 100.0), np.ones(3), True, [16, 16, 96, 96] * 2)
        missing = (np.full((3, 3), np.nan), np.zeros(3), False, np.full(8, np.nan))
        cases = {
            "no_frames": ([], []),
            "invalid_timing": (np.zeros(2), [good] * 2),
            "too_short": (TIMES[:15], [good] * 15),
            "no_face": (TIMES, [missing] * len(TIMES)),
            "low_quality": (TIMES, [good] * len(TIMES)),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "mock-input.avi"
            video.touch()
            for status, (times, observations) in cases.items():
                with self.subTest(status=status), patch(
                    "deepfake_rppg.__main__.FaceSampler"
                ), patch(
                    "deepfake_rppg.__main__.read_video",
                    return_value=(np.asarray(times), observations, FPS, "test-decoder"),
                ):
                    self.assertEqual(run(video, root / "outputs")["status"], status)

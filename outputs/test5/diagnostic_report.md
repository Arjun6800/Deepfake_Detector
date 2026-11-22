# Phase 2 facial expression and AU-proxy features

MediaPipe returned all **52 blendshape channels** on **1,029 of 1,044 frames (98.56%)** in `01_02__meeting_serious__YVGY8LOK.mp4`. All 30 selected channels were present on those frames. The missing observations were frames **0–12 and 47–48**, using zero-based indices. All 15 were Haar tracking misses. There were no further landmark-association or blendshape failures on this clip.

## Extraction and timing

The extractor enables blendshape output on the existing Face Landmarker crop and target association. It uses MediaPipe's synchronous image mode on each accepted crop and decodes the video in order through OpenCV/FFmpeg. The 24 FPS analysis timebase spans 43.458 seconds from the first to the last frame. The exported frame indices, decoder timestamps, and analysis timestamps were identical to `outputs/test4/landmark/frame_traces.csv` on all 1,044 rows. The 1,029 frames with scores also had an accepted face in that rPPG run. rPPG waveform availability still depends on ROI support and signal processing. For frame-level joins, use the same clip and `frame_index`; use `analysis_timestamp_s` when relating those frames to an rPPG time grid. [MediaPipe Face Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python)

| Step | Wall time |
| --- | ---: |
| Model and detector setup | 1.00 s |
| Decode, Haar tracking, and MediaPipe inference | 112.62 s |
| Temporal features, export, plots, and alignment check | 1.55 s |
| Total | **115.17 s** |

The measured run averaged **9.07 decoded frames per second** end to end on this Windows workspace with MediaPipe 0.10.32 and OpenCV 5.1.0. The extraction time includes tracking and decoding, so it cannot isolate the cost of the blendshape head.

## Selected channels and AU-related interpretations

The model returns coefficients for driving facial animation. The associations below describe similar visible movements and have no validated FACS interpretation in this pipeline. The left and right suffixes refer to the subject's sides, while the rPPG cheek labels use image sides. Paired channels are kept separate. The AU descriptions are based on published FACS action descriptions, and the proposed correspondences are based on movement resemblance. [MediaPipe output and side convention](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python) · [Human FACS action descriptions](https://pmc.ncbi.nlm.nih.gov/articles/PMC11653099/)

| Region | MediaPipe channels | Related movement |
| --- | --- | --- |
| Eyebrow | `browInnerUp` | AU1-like inner brow raise |
| Eyebrow | `browOuterUpLeft/Right` | AU2-like outer brow raise |
| Eyebrow | `browDownLeft/Right` | AU4-like brow lowering |
| Eye | `eyeBlinkLeft/Right` | AU45-like blink movement. A score alone cannot classify a blink event. |
| Eye | `eyeWideLeft/Right` | AU5-like upper-lid raise |
| Eye | `eyeSquintLeft/Right` | AU7-like lid tightening |
| Cheek | `cheekSquintLeft/Right` | AU6-like cheek raise. It may overlap with eyelid movement. |
| Cheek | `cheekPuff` | Cheek inflation; descriptive channel |
| Jaw | `jawOpen` | AU26-like jaw drop |
| Mouth | `mouthClose` | Lip closure; descriptive channel |
| Mouth | `mouthSmileLeft/Right` | AU12-like lip-corner pull |
| Mouth | `mouthFrownLeft/Right` | AU15-like lip-corner depression |
| Mouth | `mouthStretchLeft/Right` | AU20-like lip stretch |
| Mouth | `mouthPressLeft/Right` | AU24-like lip press |
| Mouth | `mouthPucker` | AU18-like lip pucker |
| Mouth | `mouthFunnel` | AU22-like lip funnel |
| Mouth | `mouthUpperUpLeft/Right` | AU10-like upper-lip raise |
| Mouth | `mouthLowerDownLeft/Right` | AU16-like lower-lip depression |

## Feature exports and observed behavior

- `blendshapes.csv` contains every emitted coefficient plus a status and availability flag on every decoded row. Missing scores are blank. A numeric zero remains an observed score. [Per-frame blendshapes](blendshapes.csv)
- `temporal_features.csv` contains the trailing 0.5 s mean and population standard deviation for each selected channel, using at least three contiguous observations. Signed velocity divides the adjacent score difference by the actual timestamp interval. A missing frame or a time step greater than 1.5 frame periods restarts the calculation. No values are interpolated or carried forward. Each channel had **1,025** rolling values and **1,027** velocity values on this clip. [Temporal features](temporal_features.csv)
- `clip_summary.csv` reports means, population standard deviations, mean absolute velocity, median rolling standard deviation, and support counts per selected channel. `browInnerUp` varied from 0.015 to 0.954 and `jawOpen` from 0.0002 to 0.333. `cheekSquintLeft` stayed below 0.000008 despite full score coverage, so coverage alone tells little about channel sensitivity. These are model outputs on one clip. [Clip statistics](clip_summary.csv) · [Expression scores](expression_scores.png) · [Expression dynamics](expression_dynamics.png)

## Limitations and verification

The expression extractor inherits Haar track misses and can be affected by head pose, occlusion, lighting, and model jitter. Speech can drive many of the selected mouth channels, and velocity can amplify frame-to-frame score noise. The model outputs were evaluated on one clip without expert FACS annotations. The rPPG quality gate and ROI behavior were preserved. A real-frame smoke test confirmed rPPG landmark sampling still works with blendshapes disabled, and all **25 unit tests passed**.

```powershell
.\.venv\Scripts\python.exe -m deepfake_rppg.expressions 01_02__meeting_serious__YVGY8LOK.mp4 outputs/test5 --landmark-model models/face_landmarker.task --alignment-reference outputs/test4/landmark/frame_traces.csv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

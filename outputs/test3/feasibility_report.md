# Landmark ROI feasibility and test2 baseline

Landmark extraction is blocked in the current Windows x64 Python 3.13.7 environment. The project virtual environment has OpenCV 4.11.0 from `opencv-python-headless`, including `FaceDetectorYN` and `dnn`. It has no `cv2.face`, MediaPipe, dlib, or ONNX Runtime. The workspace scan found no landmark model files. The bundled Haar cascades return detection boxes. Optical flow can follow initialized points, but the current pipeline has no semantic landmark coordinates to initialize it with.

## Detector options

| Option | Fit for this experiment |
| --- | --- |
| MediaPipe Face Landmarker | Preferred. The Apache-2.0 runtime has a Windows x64 wheel listed with Python 3.13 support. It estimates 478 mesh points, with temporal smoothing in single-face video mode. It needs approval for the runtime and the separate Face Landmarker task bundle. Verify the exact bundle's terms and checksum and smoke-test the combination locally. [Runtime license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) · [Windows wheel](https://pypi.org/project/mediapipe/1.2.3/) · [Face Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python) |
| OpenCV Facemark LBF | Requires a contrib-enabled OpenCV build and a separate trained model. The 68-point layout supplies eyes, eyebrows, lips, and the jawline. The upper forehead boundary would need to be estimated. Verify the model's terms separately from the OpenCV code. [OpenCV Facemark tutorial](https://docs.opencv.org/4.12.0/d7/dec/tutorial_facemark_usage.html) |
| OpenCV YuNet | Can use the installed `FaceDetectorYN` API with an approved compatible ONNX asset. Its MIT-licensed model directory is independent of PRISM. Its five anchors cover the eyes, nose tip, and mouth corners, which can support alignment. They are too sparse to define the feature exclusion boundaries needed here. [YuNet license and models](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [OpenCV output layout](https://docs.opencv.org/4.10.0/javadoc/org/opencv/objdetect/FaceDetectorYN.html) |

An existing licensed dense landmark model, or timestamped dense landmark tracks supplied for this exact video with a documented topology, would also let the ROI experiment proceed. With a verified model, keep the rectangular strategy available and associate the landmark mesh with the same target track. Use fixed mesh indices for inset forehead and cheek polygons and exclude dilated eye, eyebrow, and lip polygons. Mask out regions with unreliable geometry, self-occlusion, or insufficient visible area. External occlusions such as hair or hands still require validation because a face mesh may estimate points behind them.

## Measured test2 baseline

The baseline has 1,029 detected face frames out of 1,044 (98.56%). Each region supplied RGB on all 1,029 detected frames. Mean valid-pixel fractions on detected frames round to 100.00% for all three regions. That metric reflects crop support and exposure screening, so a high fraction can include background. All six traces had 98.75% processed coverage.

| Trace | Spectral concentration | Clip in-band peak (Hz) | In-band peak range across windows (Hz) |
| --- | ---: | ---: | ---: |
| Forehead green | 0.0607 | 0.750 | 0.750–1.625 |
| Forehead POS | 0.1567 | 0.750 | 0.750–1.625 |
| Image-left cheek green | 0.0301 | 1.000 | 0.750–1.125 |
| Image-left cheek POS | 0.1339 | 0.750 | 0.750–1.375 |
| Image-right cheek green | 0.0247 | 0.750 | 0.750–1.250 |
| Image-right cheek POS | 0.1964 | 0.750 | 0.750–1.375 |

Peaks are from the detrended pre-filter spectrum in the 0.7–3.0 Hz band. The clip PSD uses 8-second Welch segments. Window ranges use eight fully observed 8-second windows on a 4-second hop. The test2 95th-percentile sampling ROI center steps were 3.50, 3.45, and 3.43 px per frame for the forehead, image-left cheek, and image-right cheek respectively. These are geometry and spectral measurements. [Baseline timing, motion, and peak plot](../test2/roi_timing_peaks.png) · [Baseline ROI images](../test2/baseline_roi_samples.png)

## Verification status

All 15 existing tests passed. There are no extraction code changes or new ROI tests. The model and dependency approvals are needed before generating a landmark extraction, test2 versus test3 comparison, or comparison plots. This directory contains this report only. The 0.35 threshold remains in effect, and the clip has no synchronized reference pulse for physiological validation.

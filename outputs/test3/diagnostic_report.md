# Phase 1.3 landmark ROI diagnostic

Test2 and test3 both returned **low_quality** on `01_02__meeting_serious__YVGY8LOK.mp4`. The best full-run score rose from **0.1964 to 0.2436** on image-right cheek POS. Its processed coverage fell from **98.75% to 56.80%**. On shared time support, the rectangular baseline scored **0.2595** and the landmark method scored **0.2436** for that trace. The full-run increase therefore does not establish a signal-quality gain from the mask.

## Signal comparison

Both runs decoded 1,044 frames at 24 FPS using OpenCV/FFmpeg position timestamps. The first-to-last frame span was 43.458 s. Both accepted the target on 1,029 frames (98.56%). The missed frame indices were 0–12 and 47–48 in each run.

| Trace | Test2 concentration | Test3 concentration | Change | Test3 raw coverage | Test3 processed coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forehead green | 0.0607 | 0.0418 | −0.0189 | 98.75% | 98.75% |
| Forehead POS | 0.1567 | 0.1003 | −0.0564 | 98.75% | 98.75% |
| Image-left cheek green | 0.0301 | 0.0386 | +0.0085 | 90.80% | 90.80% |
| Image-left cheek POS | 0.1339 | 0.1710 | +0.0371 | 90.80% | 90.80% |
| Image-right cheek green | 0.0247 | 0.0488 | +0.0241 | 69.92% | 56.80% |
| Image-right cheek POS | 0.1964 | 0.2436 | +0.0472 | 64.37% | 56.80% |

All test2 traces had 98.75% raw and processed coverage. Zero of six traces passed in either run at the existing thresholds of 0.35 concentration and 50% processed coverage. Concentration measures power around the dominant in-band peak relative to pre-filter non-DC broadband power. The PSD plot shows the detrended, unfiltered spectrum of the longest finite run for each trace and uses Welch windows of up to eight seconds. The test3 POS peaks in that plot were approximately 0.875 Hz on the forehead, 0.750 Hz on the image-left cheek, and 1.397 Hz on the image-right cheek. Those spectra use different time spans where samples are missing. [Waveform comparison](waveform_comparison.png) · [PSD comparison](psd_comparison.png)

### Shared time support

For each trace, the exported raw signals were re-scored with the existing pipeline after setting timestamps missing from either run to missing in both. The rectangular baseline scored higher on all six traces on this shared support. POS windows had already been constructed in each run before this check, so their original window support can still differ near gaps. [Concentration comparison](concentration_comparison.png)

| Trace | Test2 shared concentration | Test3 shared concentration |
| --- | ---: | ---: |
| Forehead green | 0.0607 | 0.0418 |
| Forehead POS | 0.1567 | 0.1003 |
| Image-left cheek green | 0.0430 | 0.0386 |
| Image-left cheek POS | 0.2116 | 0.1710 |
| Image-right cheek green | 0.0548 | 0.0488 |
| Image-right cheek POS | 0.2595 | 0.2436 |

## Inference and mask behavior

- Verified MediaPipe 0.10.32 and OpenCV 5.1.0. The local model SHA-256 was `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. The initial full-frame MediaPipe pass accepted zero detections on this clip. Cropping to the existing Haar track with 25% padding on each side yielded one set of 478 landmarks at frame 24 and accepted landmarks on all 1,029 frames where the baseline found the target. [MediaPipe Vision API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision)
- The sampling polygons follow the landmarks, stay inside an inset face oval, and subtract buffered eye, eyebrow, and lip polygons. The nine sample overlays show areas where the test2 cheek rectangles extend into the background during turns and the far-side landmark mask shrinks or disappears. [ROI overlays](roi_mask_samples.png)
- Before interpolation, the forehead, image-left cheek, and image-right cheek yielded RGB observations on 1,029, 945, and 728 frames respectively. The exposure-valid fraction has a median of 100% for each ROI among accepted face frames. That fraction uses the remaining polygon pixels as its denominator, so a very small polygon can have full exposure-valid coverage and still fail the 16-pixel minimum. [Frame traces](frame_traces.csv)

## Remaining limitations

The landmark method inherits misses and target association from the Haar track. Large turns can collapse a far-side cheek polygon, and the geometric masks can include occlusions such as hair or eyewear. Test2 uses smoothed boxes and test3 uses per-frame landmark geometry, so this comparison includes the effects of geometry and temporal stabilization. This single clip has no synchronized reference pulse, leaving physiological accuracy unverified.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m deepfake_rppg 01_02__meeting_serious__YVGY8LOK.mp4 outputs/test3 --roi-method landmark --landmark-model models/face_landmarker.task
.\.venv\Scripts\python.exe -m deepfake_rppg.compare 01_02__meeting_serious__YVGY8LOK.mp4 models/face_landmarker.task outputs/test2 outputs/test3
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The extraction CLI exited 1 with the documented `low_quality` status and wrote the CSV, PNG, and JSON outputs. All 17 unit tests passed.

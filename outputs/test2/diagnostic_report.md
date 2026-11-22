# Phase 1 rPPG diagnostic

Both runs processed `01_02__meeting_serious__YVGY8LOK.mp4` and returned **low_quality**. The best concentration increased from **0.1801 to 0.1964** at the existing **0.35** threshold. Both runs decoded 1,044 frames and detected the target on 1,029 frames (98.56%). Raw and processed coverage was 1,031 / 1,044 (98.75%) for each trace after short-gap interpolation.

| Trace | test1 | test2 | Change |
| --- | ---: | ---: | ---: |
| Forehead green | 0.0596 | 0.0607 | +0.0010 |
| Forehead POS | 0.1013 | 0.1567 | +0.0554 |
| Image-left cheek green | 0.0421 | 0.0301 | −0.0120 |
| Image-left cheek POS | 0.0963 | 0.1339 | +0.0376 |
| Image-right cheek green | 0.0274 | 0.0247 | −0.0027 |
| Image-right cheek POS | 0.1801 | 0.1964 | +0.0163 |

## What the clip showed

- **Timing and coverage.** OpenCV/FFmpeg reported strictly increasing timestamps at 41.667 ms intervals. All timestamps agreed with frame index / 24 within 7.2 femtoseconds, and the first-to-last frame span was 43.458 s. Missing observations were frames 0–12 and 47–48, using zero-based indices. Only the internal two-frame gap was interpolated. The floating-point boundary fix for the 0.25 s interpolation limit had no effect on this clip.
- **Waveforms and spectra.** The baseline processed traces show intermittent bursts, including around 2 s and 17 s, with changing amplitude and limited agreement across regions. All three baseline green traces peaked at the lowest non-DC Welch bin (0.125 Hz), with only 5.5–11.8% of non-DC power in 0.7–3 Hz. For raw POS, the in-band fraction increased from 39.8% to 51.0% on the forehead, 34.1% to 44.5% on the image-left cheek, and 43.3% to 52.0% on the image-right cheek. Baseline broadband POS peaks differed across ROIs (12 Hz forehead, 0.375 Hz image-left cheek, 0.75 Hz image-right cheek). The strongest in-band POS bins varied by ROI and across 8 s windows with a 4 s hop, including after stabilization. [Waveforms](waveform_comparison.png) · [Spectra](psd_comparison.png) · [Window peaks](roi_timing_peaks.png)
- **ROI stability.** Baseline face box width ranged from 137 to 173 px, with a median of 157 px. The 95th-percentile ROI center step fell from 5.07 to 3.50 px for the forehead, 5.02 to 3.45 px for the image-left cheek, and 4.97 to 3.43 px for the image-right cheek. These are changes in sampling geometry, which include subject motion and detector jitter. The image samples show turning and speaking, with some cheek boxes including background despite nearly full usable-pixel fractions. That fraction screens exposure extremes and measures crop support. [Box motion and timestamps](roi_timing_peaks.png) · [Baseline ROI samples at 1, 6, …, 41 s](baseline_roi_samples.png)

## Verification and changes

- **POS.** The RGB mean normalization, projection, standard-deviation ratio, and centering agree with Wang et al. The implementation divides the overlap-add result by the number of contributing windows to normalize edge gain. A literal overlap-add sum changed concentration by less than 0.0004 across baseline and stabilized traces. The synthetic test recovered its injected 1.25 Hz component, and an independent matrix-based test confirmed the projection and overlap behavior. [Wang et al., Algorithm 1](https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf)
- **Filtering.** The pipeline linearly detrends each valid run, applies a third-order Butterworth bandpass in SOS form forward and backward, and z-scores the result. At 24 FPS the theoretical forward-backward response is approximately 0 dB at 1.25 Hz, −6.02 dB at the band edges, −77.5 dB at 0.2 Hz, and −71.6 dB at 7 Hz. The concentration score uses the detrended **pre-filter** spectrum, dividing peak-neighborhood power in the analysis band by broadband non-DC power. The PSD plots use 8 s Welch segments from the longest valid run and normalize each spectrum by total non-DC power.
- **Extraction.** The sampling box now blends 35% of the current detection with 65% of the previous sampling box on consecutive detections and resets on a miss. Face matching still uses the detector boxes, and the frame CSV records both sets of coordinates. The resampler now includes gaps exactly at the existing limit despite timestamp rounding. All 15 unit tests passed.

Box smoothing can lag rapid head motion and cannot keep rectangular ROIs on the same skin through turns or facial deformation. The mixed green results and inconsistent POS peaks leave pulse recoverability unresolved. This clip has no synchronized reference pulse for physiological validation.

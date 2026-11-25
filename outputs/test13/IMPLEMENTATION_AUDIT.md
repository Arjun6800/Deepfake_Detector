# Phase 4.10 implementation audit

## Historical measurements verified

The winning Phase 4.9 row is stage 1, arm `temporal_spectral`, mean aggregation,
Logistic Regression. It uses 187 expression, 12 POS temporal and 28 POS spectral
fields. The matched training cohort has 484 videos, 47 families, 84 originals and
400 manipulated videos. The saved configuration used C=0.1 in all five outer
folds, with fold-specific thresholds selected from inner OOF predictions.
Full-feature and late-fusion arms did not exceed its pooled balanced accuracy on
this matched cohort. The winning row has 66 TN, 18 FP, 69 FN and 331 TP.

| Configuration | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expression only, mean LR | .764463 | .730536 | .920588 | .782500 | .845946 | .788795 | .932426 |
| Selective temporal and spectral, mean LR | .805785 | .764940 | .929775 | .827500 | .875661 | .828943 | .950073 |

Mean held-out permutation AUC drops were .243977 for expression, .048059 for
POS temporal and .044942 for POS spectral. These are conditional on the fitted
model and correlated feature groups. The run checks the historical winner and
matches each eligible video to its saved outer fold.
[Selective CSV](../test12/selective_fusion_comparison.csv) |
[Baseline CSV](../test12/baseline_comparison.csv) |
[Late fusion CSV](../test12/late_fusion_comparison.csv) |
[Importance CSV](../test12/feature_importance_summary.csv) |
[Configuration](../test12/best_model_config.json)

## Checkpoint compatibility

| Input | Audited contents | Use in 4.10 |
| --- | --- | --- |
| test8 | ROI-average RGB, source codes, timestamps, geometry, nose and brightness for the earlier training subset. Alternate skin and rigid ROI RGB and mask statistics exist for that subset. Prior algorithm, compression, and timing controls are available. | Historical audit. Alternate ROI or nose regression is feasible on the cached subset only. |
| test9 | 96 centered waveform NPZs indexed by video, window and frame support. Includes projections, adjusted/detrended waveforms, run IDs and PSD. Reference POS/GREEN/CHROM/ICA/LGI, nuisance regression, clipping, adaptive mask choice and coverage/exposure weighting were studied. | Preserve existing waveform archives. They cover one window per video. Reconstruct POS reference from matching test11 observations for multi-window plots and identical pilot support. |
| test10 | Center-window expression checkpoints, 187 expression proxies, 69 legacy rPPG fields, 89 expanded signal fields and aligned fusion tables. Classification used grouped families. | Existing expression and legacy rPPG feature definitions carry forward through test11. |
| test11 | 600 training feature JSONs and raw observation NPZs. The matched 484 have multiple eight-second windows with frame-aligned regional RGB, ROI source codes, face-box geometry, brightness and blendshape scores. | Reuse baseline feature rows and original folds. Read RGB, geometry and scores together from one observation archive per video. No source frames are read. |
| test12 | Original selective schema, reusable nested evaluation routines and historical OOF predictions. | Verify historical comparison and replay mean LR baselines with the same folds, inner search and threshold rule. |

The Phase 4.9 checkpoint audit verified all 600 Phase 4.8 feature records, archive
keys, window times, source-family metadata and fold assignments. Phase 4.10
also checks alignment and hashes each observation archive used in extraction.
It uses the documented nominal or selected timebase stored with the matching
observations. RGB and scores use the same one-observation-per-grid-slot rule,
with a quarter-frame tolerance. Colliding and off-grid samples are missing.
Windows are eight seconds, with four-second hops and a deduplicated center.
Finite runs stop at missing samples or ROI provenance changes.

## Coverage of proposed enhancements

- **Signal processing.** Reuses POS, CHROM, GREEN, linear run detrending,
  MAD clipping, ridge nuisance reduction and signed regional aggregation.
  The six pilot variants include reference POS, GREEN, CHROM, robust POS,
  motion-adjusted POS and combined POS. The new variants use a slow robust
  common multiplicative brightness estimate, length-dependent broadband
  filtering and a coverage/discontinuity weight. The face-box and light
  regressors derive from test11. The filter passband is fixed in physical
  frequency, with Nyquist protection and a minimum run length. No synthesis,
  interpolation or spectral-concentration gate is used in enhanced paths.
- **Temporal.** Existing fields cover half-second autocorrelation, short-block
  RMS variation, amplitude RMS/IQR, derivative RMS and regional correlations.
  Added ACF peak/lag and decay, ordinal entropy, zero-crossing rate/spacing,
  Hjorth statistics, absolute derivative quantiles, short-to-long scale ratio,
  regional correlation summary and inter-window variation of ACF and regional
  activity. Hjorth activity re-expresses variance in regional projection units.
  Regional intensity units vary across projection methods.
- **Spectral.** Existing fields cover spectral entropy, relative band energies,
  concentration, dominant frequency and cross-region coherence. The original
  selective baseline reuses those fields. Enhanced blocks add flatness,
  centroid, bandwidth, dominant-peak prominence, second-harmonic power,
  within-window peak variation, inter-window centroid variation and Haar
  detail-band entropy. The enhanced regional coherence summary wraps the
  existing coherence implementation. The new groups are evaluated as
  alternatives to the old groups alongside the common expression baseline.
- **Cross-modal.** Computes amplitude correlation with mean blendshape
  activity, absolute waveform derivative correlation with mean absolute
  blendshape velocity, maximum-absolute bounded lagged correlation and its
  signed lag, activity-quartile amplitude variability, a high-vs-low motion
  dominant-peak shift and across-window correlation variation. Cross-modal
  statistics require a contiguous co-observed run of at least three seconds,
  and motion peak shift requires at least two usable two-second blocks. Separate
  short blocks can have quantized peak locations. The AU proxies are
  MediaPipe-derived, and these relationships are experimental features.
- **Controls and missingness.** Features retain missing values whenever
  contiguous support or variability is insufficient. Temporal features use
  at least two seconds, spectral features and pairwise coherence at least
  four seconds. Synthetic controls compare clean pulse, mixed contamination
  and artifact-only conditions using a known sinusoidal input. Synthetic
  components do not reproduce the range of signals in compressed videos.
  Motion alone, recorded nuisance variables and paired coverage are explicit
  model controls. Unsupported videos remain in the cohort.
- **Frame-dependent work.** Dense optical flow, image registration and
  re-sampling of tracked pixels, detailed landmark trajectories for the new
  families, pixel-level illumination/color correction, new ROI masks and
  pixel-derived exposure weights over the matched cohort require frame
  decoding. The test8 alternatives already captured for its smaller subset
  can be reanalyzed from cache. The existing blendshape scores and frame
  timebase support the planned Phase 4.10 cross-modal tests.


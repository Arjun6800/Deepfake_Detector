# Phase 4.6 rPPG consistency for deepfake detection

Phase 4.6 evaluated 96 existing FaceForensics++ C23 training videos from eight source families. The cohort comprises 16 originals and 16 clips from each of the five manipulation methods. Eleven method and preprocessing combinations produced 1,056 experimental rows on the same centered eight-second windows. The expanded reference POS path retained **281 regional waveforms that did not pass the existing 0.35 gate**. The study measured usable signal and feature coverage, consistency, artifact behavior and cost. It did not estimate physiological accuracy or train a classifier.

## Gate audit and preserved baselines

The 69-feature rPPG baseline contains three observed-RGB coverage fields, 54 regional GREEN and POS fields, and 12 POS region-pair fields. The gate directly controls only the **six pulse BPM fields**. These also require at least six contiguous seconds and a passing longest run. The six quality flags record the gate decision. Waveform processing, prefilter variance, concentration, dominant frequency, band power ratio and region-pair statistics already have their own support checks and are independent of the 0.35 decision. Treating a failed gate as a reason to discard the entire row would therefore discard valid data.

The expanded path has no BPM field. It keeps projections and detrended broadband waveforms, alongside the legacy gate and processed-coverage diagnostics. The 69-feature code path and 256-feature fusion baseline remain available without schema changes. The reproduced 69-feature table uses identical windows and cached tracking inputs to the new comparison for a controlled input match. Phase 4 archived outputs and the 187-feature action-unit path remain intact. Phase 4.5 captures contain no action units for these windows, so Phase 4.6 reports rPPG results and specifies the future matched fusion comparison. [Matched baseline table](baseline_rppg69.csv)

## Extraction architecture and feature schema

1. Use the audited cached C23 observations. Face detection ran on frames reduced to 384 pixels wide where needed, while landmark masks and RGB were sampled on native pixels. Snap decoder observations to the existing analysis grid. Preserve missing slots and provenance changes.
2. Keep the forehead and both cheeks. Evaluate the existing stabilized hybrid masks, skin-masked candidates and rigid landmark-template candidates. For each region, allow a single mask selection per window. The alternate must retain nearly the same observed support and reduce the 90th-percentile adjacent RGB change by at least 5% on shared observations. Skin also requires at least 70% area retention. Choices use color and support only, and can favor a smoother signal even if some genuine modulation is lost.
3. Run GREEN, POS, CHROM, ICA and LGI separately on the reference masks. Around POS, test adaptive skin, adaptive tracking, nuisance regression, robust preprocessing, and reliability-weighted aggregation separately and together. Fit nuisance regression to observed face position, size, nose offset and background luminance within valid runs. Use ridge shrinkage and partial subtraction. Apply linear detrending per run, with optional six-scaled-MAD clipping. Keep the original projection alongside the processed waveform.
4. Aggregate signed, standardized traces only when at least two ROIs contribute. The weighted variant uses observed waveform coverage and baseline exposure support, independent of the gate. Changes in contributing regions or ROI source break the aggregate analysis run. Keep individual regional waveforms, so opposite phases can be inspected even when they cancel in the aggregate.

| Feature group | Count | Measurements |
| --- | ---: | --- |
| rPPG signal | 8 | RMS and amplitude IQR for three ROIs and the aggregate |
| Spatial consistency | 15 | Signed cross-ROI correlation, magnitude-squared coherence, phase-locking consistency, spectral disagreement and relative RMS disagreement for three ROI pairs |
| Temporal consistency | 12 | Half-second autocorrelation, one-second block RMS coefficient of variation and derivative RMS |
| Spectral | 28 | Normalized spectral entropy, five relative PSD band powers between 0.25 and 5 Hz, and legacy spectral concentration |
| Reliability and missingness | 26 | Raw and waveform coverage, contiguous support, legacy processed coverage and gate, plus pairwise joint coverage and contiguous support |

The **89-feature schema** is called **rppg_consistency_v1**. The arrays retain timestamps, raw projections, nuisance-adjusted projections, waveforms, run IDs and frequency-resolved PSD. Amplitude remains in method-specific projection units, except for the standardized aggregate. PSD and paired spectral statistics require four seconds of contiguous support. Coherence and phase locking use overlapping two-second STFT blocks on the longest joint run with at least three supported blocks. Stable phase differences, including opposite phase, can yield high phase locking. Shared artifacts can yield high coherence and agreement. Absent observations stay missing. [Feature manifest](feature_schema.json) · [Expanded table](rppg_consistency_v1.csv) · [Waveform and PSD archive index](waveform_index.csv)

## Training cohort results

| Reference method | Regional waveform coverage, real / manipulated | Feature availability, real / manipulated | Mean extraction ms per window, real / manipulated |
| --- | --- | --- | --- |
| GREEN | 99.47% / 98.87% | 100% / 99.34% | 11.5 / 11.4 |
| POS | 99.47% / 98.34% | 100% / 99.34% | 34.7 / 35.5 |
| CHROM | 96.27% / 96.73% | 100% / 99.12% | 23.6 / 23.7 |
| ICA | 71.27% / 71.85% | 79.49% / 78.89% | 26.4 / 29.2 |
| LGI | 96.27% / 96.73% | 100% / 99.12% | 11.8 / 11.7 |

Runtimes cover the projections, aggregation and feature extraction on cached RGB. Video decode, tracking and archive writes are excluded. Feature availability counts finite values across all 89 fields, including explicit reliability indicators. The reference POS gate passed 2 of 48 real and 5 of 240 manipulated regional traces. All 281 remaining POS traces retained some broadband waveform support. Legacy POS processed coverage was 96.27% and 96.73%. The new support measure also includes broadband runs as short as one second, so the coverage difference measures what is retained under the new representation.

Adaptive skin selected 14 of 288 ROI-windows and adaptive rigid tracking selected 44. Both left coverage unchanged at the reported precision. Weighted aggregation also left coverage essentially unchanged because exposure support was high. Combined POS kept the same feature availability as reference POS and changed mean pairwise correlation from **0.175 to 0.204** on real clips and **0.247 to 0.269** on manipulated clips. Coherence fell from 0.175 to 0.169 and 0.201 to 0.195, respectively. Phase consistency increased slightly. Combined extraction averaged 35.9 and 36.4 ms per window. Higher correlation alone does not establish artifact removal or detection value.

In twelve seeded synthetic trials per condition on projected signals, regression with the known nuisance proxy improved waveform correlation with the known pulse from **0.441 to 0.766** when motion had a distinct frequency. Mean estimated pulse-component amplitude was unchanged at 0.999 relative to the unit input. In the artifact-only case, residual RMS fell to **41.1%** of the original and **all 12 traces still passed** the legacy gate. When motion and pulse were in phase at the same frequency, correlation remained high despite the ambiguity. The fitted subtraction cannot identify which component was physiological. These are controlled nuisance-regression checks with a known proxy. Real tracking and illumination proxies are noisier. [Cohort comparisons](cohort_comparison.csv) · [Synthetic measurements](synthetic_artifact_check.csv)

## Recommendations and future ablation

Use reference POS as the primary expanded representation and GREEN as a low-cost control. Keep the combined POS path as a comparison arm until grouped experiments show stable gains. Preserve the legacy concentration flag as a reliability covariate. Keep signal, spatial, temporal and spectral measurements available when they have sufficient support. ICA has lower observed coverage in this cohort and should remain a challenger rather than the default extractor.

EfficientPhys remains the first neural feasibility candidate from the earlier inventory. No pretrained checkpoint is present locally and the current environment lacks the documented inference runtimes. Neural inference and CPU cost are therefore unmeasured. If later authorized, compare the frozen model with classical projections on the same timeline and report coverage, consistency, artifact behavior and computational cost.

For the future four-arm ablation, generate action-unit features using the existing expression pipeline on the exact archived time bounds. Join on clip ID, source family and window bounds after checking alignment. The rPPG-only arm uses the signal, spatial, temporal and spectral groups and excludes explicit reliability flags. The reliability-only arm uses the 26 reliability fields. The expression-only arm uses the existing 187 expression fields. The fused arm combines rPPG and expression. Compare a prespecified fused-plus-reliability sensitivity analysis and the existing 256-feature fusion baseline. Use identical source-family-grouped folds within training, fit imputation and scaling within each training fold, and keep explicit missingness indicators confined to reliability-enabled arms. Report performance and confidence intervals by held-out family, manipulation method and coverage stratum. Keep the official validation and test partitions untouched in this study.

The training cohort is small and contains related clips. Periodic lighting and motion can survive correction and create apparent cross-region agreement. Neither poor signal quality nor a high consistency score determines authenticity. The new schema is ready for alignment and a grouped ablation. Full extraction and classifier training remain deferred.

Run the scoped study with the existing virtual environment using **python -m deepfake_rppg.consistency_experiment**. The complete suite passed **54 tests**. [Test log](test_results.txt)

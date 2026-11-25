# Phase 4.5 rPPG quality investigation

**Decision.** Keep the eight-second Phase 4 baseline and its 0.35 concentration threshold. Pause full dataset extraction and classifier training until synchronized contact PPG validates pulse recovery on representative compressed footage.

## Why the pilot failed

The replay reproduced all 540 gate decisions. Of these, 412 traces had sufficient processed coverage and low concentration, 122 were unscorable, and six passed. Four of the passing traces also met the pulse-release conditions. The audit found correct RGB ordering and POS projection algebra. Short gaps can fragment usable observations, but the main direct cause was low pre-bandpass concentration. Compression and nuisance variation can explain some failures. FF++ cannot establish how much genuine pulse was present. [Root-cause audit](ROOT_CAUSE.md) · [Pilot replay](phase4_pilot_gate_audit.csv)

## Experiment design

Seed 451 selects eight training source families with both originals and both directions of each of Deepfakes, Face2Face, FaceShifter, FaceSwap and NeuralTextures. One of the eight families was reserved using audited duration metadata so 30-second analysis would be possible. Selection used no quality or classifier outcomes. The comparison uses one centered window per eligible video and duration. The existing Phase 4 used overlapping eight-second windows and is reported separately.

The main capture uses 384-pixel-wide target detection on larger frames. Landmarks and RGB pixels still come from the original frames. This acquisition choice reduced the cost of the full-frame cascade and can change tracking success, mask position and observed RGB. Full-resolution overlap clips measure that sensitivity. The eight-second training comparison applies the same analysis thresholds and POS/GREEN scoring as Phase 4. [Selection and hashes](training/selection.json) · [Capture results](training/capture_summary.json) · [Detector sensitivity](detector_sensitivity.csv)

The preprocessing comparisons change one component at a time around POS. Skin segmentation uses a broad YCrCb mask inside geometric regions. Rigid alignment follows a landmark template using a 2D similarity fit. Both use pair-specific common frame support and separate all-observed sensitivity rows. Adaptive ROI chooses the most-observed region once per window and compares it with a fixed forehead. Illumination normalization divides the ROI colors by a background luminance proxy. Aggregation averages RGB across all three ROIs on joint support before projection. Motion-aware scoring adds an exploratory face-speed cutoff of 0.5 face widths per second after the baseline gate. Bounded interpolation fills up to 0.25 s and preserves ROI provenance boundaries.

Each pass rate uses all attempted trace-windows in its row. Mean concentration and power ratios use available values, with denominators in the tables. Missing feature percentage counts absent concentration, dominant frequency, pre-filter variance, in-band/out-of-band power ratio and gated pulse estimate. The FF++ power ratio is descriptive. Synthetic SNR uses power within ±0.25 Hz of the declared pulse frequency divided by the remaining non-DC power. Cross-ROI agreement uses the longest joint processed run. Runtime per trace includes projection and scoring and is recorded separately from whole-video decode and capture.

## Training results

All 96 selected videos completed, including 16 originals and 16 per manipulation method. The experiment evaluated 7,416 trace-windows across methods, durations and ablations. The decode produced 46,640 frames, with every frame count and nominal frame rate matching the audited manifests to the recorded precision. The six-clip PTS probe found agreement with decoder millisecond positions. In the six primary detector-overlap clips, observed RGB coverage was 97.38% at native resolution and 97.08% at 384 pixels. Both detectors yielded two passes among the 36 common-support GREEN and POS comparisons per detector. [Video inventory](video_inventory.csv) · [Timing probe](timing_probe.csv) · [Trace metrics](trace_metrics.csv)

| Eight-second method | Passes real / manipulated | Mean concentration real / manipulated | Processed coverage real / manipulated | Mean ms per trace real / manipulated |
| --- | --- | --- | --- | --- |
| GREEN | 0/48 · 0/240 | 0.096 · 0.097 | 96.3% · 96.7% | 5 · 5 |
| POS | 2/48 · 5/240 | 0.181 · 0.152 | 96.3% · 96.7% | 26 · 27 |
| CHROM | 3/48 · 0/240 | 0.161 · 0.143 | 96.3% · 96.7% | 17 · 17 |
| ICA | 1/48 · 1/240 | 0.179 · 0.163 | 71.3% · 71.9% | 20 · 22 |
| LGI | 0/48 · 1/240 | 0.107 · 0.090 | 96.3% · 96.7% | 6 · 6 |

POS passed seven of 288 traces and released five pulse estimates. Mean cross-ROI POS peak disagreement was 0.409 Hz for real clips and 0.374 Hz for manipulated clips. ICA had the lowest conditional peak disagreement, 0.208 and 0.269 Hz, with substantially less processed support. ICA also had the highest missing-feature percentages, 39.6% and 41.3%, compared with 19.6% and 20.4% for POS. Five-worker capture took 17.5 minutes, with a median 52.4 seconds of wall time per video under concurrent capture. [Algorithm comparison](algorithm_comparison.csv) · [Breakdown by manipulation method](method_family_comparison.csv) · [Runtime](runtime_comparison.csv)

Scoring CHROM after bandpass raised passes to 26/48 and 121/240. This attenuated out-of-band energy before computing concentration, so those scores are on a different basis. The table above uses the pre-bandpass counterpart and retains a common denominator definition. The measured pass rates describe the selected clips and cannot establish pulse accuracy or distinguish real from manipulated video. [Real waveforms and PSD](real_waveforms_psd.png) · [Manipulated waveforms and PSD](manipulated_waveforms_psd.png)

### ROI and preprocessing

- Bounded interpolation improved processed POS coverage from 96.3% to 99.8% for real and from 96.7% to 98.6% for manipulated videos. It lost one pass in each group, with no gained passes. Treat this as a coverage improvement.
- On matched frame support, the skin mask lost one manipulated pass. Rigid alignment gained one real pass and lost one manipulated pass. Both had small average concentration changes. Background luminance normalization left the pass counts unchanged.
- The coverage-based selector chose the forehead in all 96 clips. The adaptive rule therefore yielded the same output as the fixed forehead in this cohort. Three-ROI averaging on joint support yielded 0/16 real and 2/80 manipulated passes. The average concentration was lower than the corresponding individual ROI controls in both groups.
- The motion veto retained all seven baseline POS passes. Descriptive per-video rank correlations of POS concentration with motion, illumination and encoded bitrate proxy ranged from −0.044 to 0.212 across the two groups. The proxies and small number of source families limit inference about individual causes.

[Paired ablations](paired_ablation_comparison.csv) · [ROI comparison](roi_comparison.csv) · [Real quality and ROI plot](real_quality_duration_roi.png) · [Manipulated quality and ROI plot](manipulated_quality_duration_roi.png) · [Nuisance strata](nuisance_sensitivity.csv) · [Per-video associations](nuisance_rank_associations.csv)

### Duration versus coverage

| Seconds | Real videos | Manipulated videos | Share of selected videos | POS passes real / manipulated |
| --- | --- | --- | --- | --- |
| 8 | 16 | 80 | 100.0% | 2/48 · 5/240 |
| 12 | 13 | 59 | 75.0% | 1/39 · 2/177 |
| 16 | 9 | 39 | 50.0% | 0/27 · 0/117 |
| 20 | 7 | 29 | 37.5% | 0/21 · 0/87 |
| 30 | 2 | 10 | 12.5% | 0/6 · 0/30 |

On nested windows from the same 12 videos eligible for 30 seconds, all five methods registered zero passes at every duration. Mean POS concentration went from 0.164 to 0.168 on the two real clips and from 0.126 to 0.133 on the ten manipulated clips between eight and 30 seconds. Pairwise comparisons restricted to at least 95% processed coverage at both durations also showed no POS pass-rate increase over eight seconds in either group. The 30-second result represents one source family and cannot establish a general duration effect. [All eligible videos](duration_comparison.csv) · [Common 30-second subset](duration_matched_30.csv) · [Matched support](duration_paired_support.csv) · [Coverage and runtime plot](coverage_runtime.png)

## Synthetic and compression checks

All five methods recovered the clean input frequencies across 15 trials each. POS and CHROM recovered the simulated pulse within 5 bpm in all mixed pulse-and-motion trials. ICA did so in eight of 15. LGI accepted 15 of 15 and tracked the nuisance peak in every trial. In the artifact-only condition, GREEN, CHROM, ICA and LGI accepted the nuisance in all repeats, while POS rejected that specific RGB mixture. The no-pulse cases repeat five independent noise seeds across the three frequency slots. When noise overwhelmed the weak pulse, GREEN and ICA each issued one estimate outside the 5 bpm tolerance and the other methods withheld their estimates. These controlled inputs test algorithm behavior and gate specificity.

In the textured 75 bpm phantom, the lossless FFV1 round trip preserved each ROI mean. The Windows Media Foundation H.264 encode reduced POS concentration from 0.970 to 0.052 at 0.35 RGB counts of green modulation. At 1.2 counts, POS concentration fell from 0.990 to 0.798 and the gated estimate remained 75 bpm. The default MPEG-4 Part 2 encode also produced a passing 60 bpm POS estimate at the lower amplitude. These are encoder- and scene-specific observations. [Synthetic measurements](synthetic.csv) · [Synthetic waveforms and PSD](synthetic_waveforms_psd.png) · [Compression measurements](compression_phantom.csv)

## Recommendations

Retain GREEN as a low-cost control and POS as the primary classical candidate. Keep CHROM as a benchmark challenger and retain the pre-bandpass scoring denominator. Preserve acquisition coverage, contiguous support, peak frequency and motion/illumination diagnostics as separate measurements. Verify gap filling and any ROI or duration changes against matched contact PPG before adopting them. Estimate coverage and pulse error together and stratify by subject, lighting, motion and controlled codec settings. Poor trace quality is insufficient evidence of manipulation.

The physiological validation is insufficient to proceed. COHFACE offers compressed recordings with synchronized contact BVP. PURE and UBFC-rPPG provide complementary motion and lighting tests. EfficientPhys-C is the first neural throughput candidate if a suitable checkpoint and environment are approved. Local neural inference speed and accuracy are unmeasured. [Benchmarks and model feasibility](BENCHMARKS_AND_MODELS.md)

## Reproduction

Use the project virtual environment and the existing audited manifest. The capture hashes source files before reading frames. The analysis only uses training selections and writes to this directory.

    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:OPENCV_FFMPEG_THREADS = '1'
    $env:MPLBACKEND = 'Agg'
    .venv/Scripts/python.exe -m deepfake_rppg.quality_capture --workers 5 --detector-width 384 --include-long --output outputs/test8/training
    .venv/Scripts/python.exe -m deepfake_rppg.quality_experiment --require-complete
    .venv/Scripts/python.exe -m unittest discover -s tests -v

Earlier capture attempts ran into Windows paging and process errors. The initial retry artifacts are retained for diagnosis. The training subdirectory contains the run used for the comparisons. Source videos, the Phase 4 artifacts and the reference code were preserved. [Test results](test_results.txt)

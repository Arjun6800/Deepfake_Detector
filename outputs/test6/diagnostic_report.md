# Phase 3 window features and multimodal fusion

The stabilized landmark rPPG and MediaPipe expression exports for `01_02__meeting_serious__YVGY8LOK.mp4` matched on all **1,044 decoded frames** at 24 FPS. The run produced **nine 8-second windows with 4-second overlap**, from [0, 8) through [32, 40). Each window has 192 decoded frames and 192 analysis-grid samples. The last 3.5 seconds of grid support fell outside the last complete window.

| Table | Window rows | Feature columns | Key and support columns | Blank feature cells |
| --- | ---: | ---: | ---: | ---: |
| rPPG-only | 9 | 69 | 9 | 74 of 621 |
| Expression-only | 9 | 187 | 9 | 0 of 1,683 |
| Early fusion | 9 | 256 | 9 | 74 of 2,304 |

The fused table concatenates the two sets of features on the same clip and window key. Each table retains window bounds, decoded frame bounds and counts, and the source split group. Blank feature cells indicate unavailable values. A zero in a coverage or pass column records the measured lack of support. [rPPG features](rppg_features.csv) · [Expression features](expression_features.csv) · [Fused features](fused_features.csv) · [Feature schema and provenance](metadata.json)

## rPPG results

| ROI | Native RGB coverage across windows | Processed coverage across windows | Windows with processed data |
| --- | ---: | ---: | ---: |
| Forehead | 92.19–100% | 74.48–100% | 9 / 9 |
| Image-left cheek | 92.19–100% | 74.48–100% | 9 / 9 |
| Image-right cheek | 65.63–100% | 0–100% | 7 / 9 |

Green and POS had the same processed coverage within each ROI on this run. The right cheek in [0, 8) and [12, 20) had observations but no contiguous run long enough for filtering, so the spectral and pairwise metrics that required it are blank. Of **54 trace-window combinations**, 50 had spectral concentration values and **zero passed** the 0.35 concentration and 50% processed-coverage gate. The highest concentration was **0.2858**, for forehead green in [12, 20). All 54 pulse estimates are blank. Of the 74 blank rPPG feature cells, 54 are pulse estimates and 20 are unsupported spectral or inter-region measurements. [rPPG coverage and quality plot](rppg_window_quality.png)

Inter-region POS correlation was defined in all nine forehead–left cheek windows and seven of nine windows for each pair involving the right cheek. Observed zero-lag correlations ranged from −0.189 to 0.372. They describe waveform agreement and can also reflect shared motion or lighting. The quality flags describe measurement support. This clip supplies no physiological reference for the spectral peaks.

## Feature definitions and missingness

- **Synchronization.** Frame indices, source timestamps, and analysis timestamps must match between the native exports. The raw and processed rPPG files also supply an analysis-grid check. The feature builder recomputes POS, filtering, and expression dynamics inside each half-open window from native RGB and blendshapes. Native RGB is assigned once to a unique grid slot within a quarter frame period. Off-grid or colliding frames and missing observations remain unsupported. Hybrid source changes split rPPG runs.
- **rPPG.** Each region supplies native RGB coverage and green/POS features for raw and processed support, longest run duration, pre-filter variance, concentration, strongest in-band frequency, in-band versus non-DC out-of-band power in dB, quality-pass flag, and gated pulse estimate. The power ratio is a noise indicator and depends on what falls inside the band. Concentration uses the existing score over eligible runs. Frequency and power ratio use the longest eligible run. A pulse estimate additionally requires six contiguous seconds and the peak's own run to pass the concentration gate. Pairwise POS features measure zero-lag correlation and absolute spectral peak separation on the longest common run, with four seconds required.
- **Expression.** Each of 30 selected channels contributes coverage, mean, population standard deviation, mean trailing 0.5-second standard deviation, and mean absolute velocity in score units per second. Dynamics restart at gaps. Twelve left–right channel pairs contribute simultaneous coverage and signed and absolute differences. All selected scores were present on 92.19% of the first window's frames and 100% thereafter. Every expression feature had sufficient support. These MediaPipe coefficients have AU-related interpretations as documented in Phase 2. [Expression window plot](expression_windows.png) · [Phase 2 mapping](../test5/diagnostic_report.md)

## Readiness for dataset processing

The pipeline is ready for a dataset **pilot** with paired Phase 1 and 2 exports. It validates timing, records file SHA-256 fingerprints and settings, and keeps all windows from a clip in the same default split group. For related encodes, manipulated derivatives, or repeated subjects, assign a shared `--split-group` from the dataset manifest before partitioning. Split by that group, keep overlapping windows together, and exclude clip, group, and window keys from model inputs. Window-local processing keeps waveform and expression statistics within their bounds. The upstream face track can still depend on earlier frames of the same clip.

The sample establishes alignment and feature generation on one clip. A pilot should measure coverage across frame rates, motion, and compression and check how often variable-rate timestamps fall outside the grid tolerance. Source and subject relationships need dataset-level provenance. Missing pulse estimates and low-quality traces should be tracked as measurement coverage.

All **31 tests passed**. A secondary smoke check using hybrid rPPG and six-second non-overlapping windows produced seven synchronized windows. The primary run is reproducible with the following commands.

```powershell
.\.venv\Scripts\python.exe -m deepfake_rppg.fusion 01_02__meeting_serious__YVGY8LOK.mp4 outputs/test4/landmark outputs/test5 outputs/test6 --window-seconds 8 --overlap-seconds 4
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

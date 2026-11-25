# Phase 4.9 implementation and resume

The checkpoint audit and implementation are complete. Full Phase 4.9 classification and evaluation are pending. The classification runner uses only existing Phase 4.8 training artifacts. [Checkpoint audit](checkpoint_audit.json) · [Test results](test_results.txt)

## Checkpoint compatibility

All 600 approved training feature JSONs and the timestamp metadata of their 600 observation NPZs passed the audit. The audit verified eight-second window bounds, frame and grid support, feature schemas, video IDs, labels, manipulation labels and the five Phase 4.8 source-family folds. All 32,480 saved Phase 4.8 predictions from 58 configurations aligned with the appropriate cohort. The regression tests recomputed balanced accuracy and ROC-AUC from the underlying predictions for both published Phase 4.8 comparison tables.

The temporal cohort contains 484 videos across 47 source families, including 412 videos from 39 of the 42 newly studied families. A separate center-window check uses all 600 videos to evaluate all 42 newly studied families. Phase 4.8 informed the Phase 4.9 design, so these are development results.

The archived window-level POS, GREEN, expression and legacy features support every planned arm. Adaptive fusion uses the mean of four saved POS waveform coverage features across a video's windows. Raw observations are used for the timestamp audit. The models consume the saved features directly, so video decoding and waveform reconstruction are unnecessary. No physiological reference measurements are part of this study.

## Experiment stages

| Stage | Experiments | Cohort |
| --- | ---: | --- |
| 1 | 27 | 484 videos. Five baselines with four aggregation methods and seven selective mean-window Logistic Regression arms. |
| 2 | 13 | 484 videos. One strongly regularized Logistic Regression arm, three alternative classifiers on each of three configurations, and three late fusion methods. |
| 3 | 6 | 600 videos. Center-window Logistic Regression on the five baseline arms and expression plus POS temporal features. |
| Total | 46 | 230 outer-fold checkpoints and 22,960 out-of-fold video predictions when complete. |

The POS and GREEN method arms use matching schema-defined temporal, spectral and spatial blocks. The automatic arm ranks the 63 non-reliability POS fields using training-fold class separation and retains up to 12 alongside expression. The regularized arm searches two stronger Logistic Regression penalties than the baseline. The alternative classifier settings cover Random Forest depth, RBF SVM C, and Histogram Gradient Boosting leaf count. Window scores use weights that give each video equal total weight within its class. Thresholds maximize balanced accuracy on four grouped inner-fold OOF scores, with ROC-AUC used to break parameter-selection ties. The fixed-setting late base learners are fitted separately inside each inner split. Weighted and adaptive fusion include zero rPPG weight, and stacking can fall back to its expression-only score.

The compatibility audit found 4.52 GiB available on the workspace drive and selected one worker under the RAM policy at implementation time. On launch, Windows allows two workers when at least 3 GiB of physical memory is available. The command requests a maximum of two. Each worker uses one native thread. Writes check a 2 GiB free-space reserve plus a 64 MiB buffer.

Allow about **15–50 minutes with one worker** or **8–30 minutes with two**, plus roughly a minute of evaluation overhead. These are planning estimates. Phase 4.8 logged 5,179 seconds summed across fits and predictions, including about 506 seconds for one mean-window Histogram Gradient Boosting configuration. Phase 4.9 has three such configurations with narrower binning, and most other experiments use cheaper Logistic Regression. Early ETA will move as the slower stage starts.

## PowerShell

From the project root, run stage 1 first. The console and log report completed experiments and folds, elapsed time and ETA. Repeat a classification command to resume. Completed compatible folds are skipped. Interrupted folds are refitted. Run only one classifier instance at a time.

~~~powershell
Set-Location -LiteralPath 'D:\Research\Deepfake_using_rPPG'
$python = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

& $python -u -B -m deepfake_rppg.phase49_classify --stage 1 --workers 2 | Tee-Object -FilePath outputs/test12/stage1.log -Append
& $python -B -m deepfake_rppg.phase49_evaluate --stage 1
& $python -u -B -m deepfake_rppg.phase49_classify --stage all --workers 2 | Tee-Object -FilePath outputs/test12/classify.log -Append
& $python -B -m deepfake_rppg.phase49_evaluate
~~~

To start the full classification independently of the PowerShell window, use this alternative after the setup above. Run evaluation after the process exits successfully and writes the completion marker.

~~~powershell
$repo = (Get-Location).Path
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Start-Process -FilePath $python -ArgumentList @('-u','-B','-m','deepfake_rppg.phase49_classify','--stage','all','--workers','2') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $repo "outputs/test12/classify-$stamp.out.log") -RedirectStandardError (Join-Path $repo "outputs/test12/classify-$stamp.err.log") -PassThru
Get-Content -LiteralPath "outputs/test12/classify-$stamp.out.log" -Tail 12
~~~

For a read-only fit-free input audit and the regression suite, use these commands. The audit writes its summary under test12.

~~~powershell
& $python -B -m deepfake_rppg.phase49_classify --audit
& $python -B -m unittest discover -s tests -v
~~~

## Pending outputs

Stage 1 evaluation generates interim tables and plots. Full evaluation writes the final report and CSV comparisons, predictions, paired family intervals, manipulation results, confusion matrices, permutation importance, plots and configuration and runtime summaries. The final model configuration is a post hoc description of the highest observed temporal score by balanced accuracy and ROC-AUC. Use the paired family results to assess whether fusion improved on expression-only. Overlapping training folds and experiment selection limit the interpretation of the descriptive intervals.

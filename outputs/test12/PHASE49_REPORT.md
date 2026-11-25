# Phase 4.9 selective rPPG and facial expression fusion

Evaluation scope **all**. The completed run contains 46 experiments and 22,960 out-of-fold video predictions.
Across the shared 484-video temporal cohort, the top observed arm was **temporal_spectral** with **logistic** and **mean** aggregation. Balanced accuracy was **0.765** and ROC-AUC was **0.829**. On the 412 eligible videos from 39 newly studied families, these were **0.761** and **0.825**.
The top configuration is a post hoc development ranking. Its scores can reflect selection across experiments.

The outer splits match Phase 4.8. Four inner source-family folds provide out-of-fold scores for threshold and parameter selection. Late fusion uses fixed base learner settings and learns its fusion rule from inner out-of-fold scores. The adaptive gate uses waveform coverage as a signal-availability proxy.
PR-AUC is average precision. Manipulation comparisons pair each manipulation with available originals from its source families. Family intervals are descriptive percentile bootstrap intervals over families containing both classes. Group permutation importance is the average held-out ROC-AUC drop across folds. Dependent and redundant features can affect this measure.
Phase 4.9 was informed by Phase 4.8. The newly studied families are a development robustness subset. All videos come from the approved training partition. Physiological accuracy was outside the evaluation.

## Paired balanced accuracy against expression-only

| Cohort | Mean change (points) | 95% descriptive interval (points) | Families |
| --- | ---: | ---: | ---: |
| all | +3.26 | -1.60 to +8.19 | 47 |
| new | +3.08 | -2.35 to +8.59 | 39 |

## Center-window development check

All 42 newly studied families contribute 504 videos to this check.

| Arm | Balanced accuracy | ROC-AUC |
| --- | ---: | ---: |
| expression187 | 0.712 | 0.781 |
| rppg69 | 0.536 | 0.532 |
| fusion256 | 0.651 | 0.765 |
| fusion276 | 0.677 | 0.740 |
| fusion250 | 0.698 | 0.761 |
| temporal | 0.724 | 0.801 |

[Baselines](baseline_comparison.csv) · [Selective fusion](selective_fusion_comparison.csv) · [Late fusion](late_fusion_comparison.csv) · [Paired families](paired_family_comparison.csv) · [Runtime](runtime_summary.json) · [ROC](roc_curves.png) · [Precision–recall](precision_recall_curves.png)

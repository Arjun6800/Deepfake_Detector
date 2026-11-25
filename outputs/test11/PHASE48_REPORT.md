# Phase 4.8 training-cohort replication results

The study evaluated 600 videos using five outer source-family folds and four inner grouped folds.
The temporal comparison included 484 videos with at least two complete windows.

## Paired Logistic Regression results

The table reports mean held-out family balanced-accuracy differences against expression-only.
The percentile intervals resample families. Overlapping training folds limit their inferential interpretation.

| Comparison | Cohort | Mean change (points) | 95% descriptive interval (points) | Families |
| --- | --- | ---: | ---: | ---: |
| fusion256 | all | -3.00 | -7.10 to +0.90 | 50 |
| fusion256 | new | -2.98 | -7.62 to +1.55 | 42 |
| fusion276 | all | -4.30 | -8.40 to -0.40 | 50 |
| fusion276 | new | -5.00 | -9.40 to -0.71 | 42 |
| fusion250 | all | -3.70 | -7.40 to -0.30 | 50 |
| fusion250 | new | -4.29 | -8.10 to -0.71 | 42 |

## Interpretation

Use the new-family comparison as the replication check. Read paired intervals alongside pooled video metrics and the reliability, missingness and nuisance probes. The evaluation supports no claim about physiological accuracy. Select a final architecture only after reviewing the paired results and feature-group evidence.

[Ablations](ablation_comparison.csv) · [Temporal results](temporal_comparison.csv) · [Predictions](video_predictions.csv) · [Importance](feature_importance_summary.csv) · [ROC](roc_curves.png) · [Precision and recall](precision_recall_curves.png)

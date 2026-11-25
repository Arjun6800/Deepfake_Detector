# Phase 4.11A targeted hybrid feature optimization

These are exploratory development estimates. Earlier experiments informed the feature choices.
The run used 484 matched training videos, the five established source-family outer folds and
four inner grouped folds. Official validation and test partitions were untouched.
Configurations were ranked after evaluation by pooled balanced accuracy, then ROC-AUC.

| Configuration | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase49_historical / mean / logistic_l2 | 0.80579 | 0.76494 | 0.92978 | 0.82750 | 0.87566 | 0.82894 | 0.95007 |
| phase49_selective / mean / logistic_l2 | 0.80579 | 0.76494 | 0.92978 | 0.82750 | 0.87566 | 0.82894 | 0.95007 |
| hybrid / mean_std / logistic_l2 | 0.77479 | 0.78851 | 0.95046 | 0.76750 | 0.84924 | 0.82719 | 0.94624 |
| expression_only / mean / logistic_l2 | 0.76446 | 0.73054 | 0.92059 | 0.78250 | 0.84595 | 0.78879 | 0.93243 |
| original_rppg_only / mean / logistic_l2 | 0.55372 | 0.56542 | 0.86220 | 0.54750 | 0.66972 | 0.63095 | 0.87314 |

The top configuration changed balanced accuracy by +0.02357
and ROC-AUC by -0.00176 relative to the historical Phase 4.9 predictions.
The replay row uses the original Phase 4.9 fitting recipe. AP denotes average precision.

## Aspirational targets

- Accuracy 77.48% against 90% target
- Balanced Accuracy 78.85% against 85% target
- Precision 95.05% against 95% target
- Recall 76.75% against 90% target
- F1 84.92% against 92% target
- Roc Auc 82.72% against 90% target

## Paired full-hybrid comparisons

Deltas use equally weighted source families with both classes. Bootstrap intervals describe
the observed family deltas and do not adjust for searching across configurations.

- phase49_historical, balanced accuracy +0.01986 (95% family bootstrap interval -0.03156 to +0.07057, 47 families)
- phase49_historical, roc auc -0.00319 (95% family bootstrap interval -0.04753 to +0.04043, 47 families)
- expression_only, balanced accuracy +0.05248 (95% family bootstrap interval +0.00993 to +0.09645, 47 families)
- expression_only, roc auc +0.00957 (95% family bootstrap interval -0.04150 to +0.06064, 47 families)
- original_rppg_only, balanced accuracy +0.22801 (95% family bootstrap interval +0.18369 to +0.27624, 47 families)
- original_rppg_only, roc auc +0.20585 (95% family bootstrap interval +0.13422 to +0.28280, 47 families)

## Interpretation

Permutation importance measures held-fold ROC-AUC changes after jointly shuffling each block.
Window models permute videos within window-count strata to preserve within-video structure.
Correlated blocks can share predictive signal. Importance and ablation results are exploratory
and do not identify independent feature effects. Window models omit the two video-only enhanced
spread features. The classifier shortlist was fixed before fitting Phase 4.11A.

See the comparison CSVs for all configurations, the confusion matrix CSV for class counts,
and the PNG figures for ROC, precision–recall, confusion matrices and benchmark comparisons.

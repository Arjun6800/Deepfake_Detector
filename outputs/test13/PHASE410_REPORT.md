# Phase 4.10 rPPG and expression enhancement

Completed nested evaluation of 18 configurations on 484 matched training videos.
Pilot selection was exploratory and used overlapping source families. Final development estimates are
conditional on that screening. The official validation and test partitions were untouched.

| Arm | Waveform | Classifier | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expression187 | original | logistic | 0.7645 | 0.7305 | 0.9206 | 0.7825 | 0.8459 | 0.7888 | 0.9324 |
| rppg69 | original | logistic | 0.5537 | 0.5654 | 0.8622 | 0.5475 | 0.6697 | 0.6310 | 0.8731 |
| temporal_spectral | original | logistic | 0.8058 | 0.7649 | 0.9298 | 0.8275 | 0.8757 | 0.8289 | 0.9501 |
| enhanced_rppg | pos_reference | logistic | 0.6095 | 0.5757 | 0.8625 | 0.6275 | 0.7265 | 0.5907 | 0.8656 |
| expression_temporal | pos_reference | logistic | 0.7665 | 0.7506 | 0.9309 | 0.7750 | 0.8458 | 0.8178 | 0.9360 |
| expression_spectral | pos_reference | logistic | 0.7066 | 0.7143 | 0.9243 | 0.7025 | 0.7983 | 0.7890 | 0.9344 |
| expression_enhanced | pos_reference | logistic | 0.7190 | 0.7265 | 0.9286 | 0.7150 | 0.8079 | 0.8142 | 0.9418 |
| expression_cross | pos_reference | logistic | 0.7397 | 0.7155 | 0.9177 | 0.7525 | 0.8269 | 0.7768 | 0.9307 |
| expression_enhanced_cross | pos_reference | logistic | 0.7004 | 0.7294 | 0.9352 | 0.6850 | 0.7908 | 0.8138 | 0.9438 |
| nuisance | pos_reference | logistic | 0.4607 | 0.4857 | 0.8174 | 0.4475 | 0.5784 | 0.5045 | 0.8303 |
| motion_only | pos_reference | logistic | 0.4855 | 0.4865 | 0.8186 | 0.4850 | 0.6091 | 0.4988 | 0.8316 |
| missingness | pos_reference | logistic | 0.2479 | 0.5027 | 0.8333 | 0.1125 | 0.1982 | 0.5098 | 0.8305 |
| expression_enhanced | pos_reference | random_forest | 0.6694 | 0.6871 | 0.9167 | 0.6600 | 0.7674 | 0.7235 | 0.9180 |
| expression_enhanced | pos_reference | svm_rbf | 0.6632 | 0.6975 | 0.9247 | 0.6450 | 0.7599 | 0.7864 | 0.9393 |
| expression_enhanced | pos_reference | hist_gradient_boosting | 0.6591 | 0.6856 | 0.9181 | 0.6450 | 0.7577 | 0.7640 | 0.9289 |
| expression_enhanced_cross | pos_reference | random_forest | 0.6302 | 0.6681 | 0.9139 | 0.6100 | 0.7316 | 0.7222 | 0.9173 |
| expression_enhanced_cross | pos_reference | svm_rbf | 0.6963 | 0.6893 | 0.9121 | 0.7000 | 0.7921 | 0.7762 | 0.9343 |
| expression_enhanced_cross | pos_reference | hist_gradient_boosting | 0.7169 | 0.7065 | 0.9175 | 0.7225 | 0.8084 | 0.7699 | 0.9341 |

Top observed configuration was temporal_spectral with original and logistic.
Historical Phase 4.9 winner achieved 0.7649 balanced accuracy and 0.8289 ROC-AUC.

Thresholds and model settings came from four inner grouped folds. Family intervals are descriptive
bootstrap intervals over complete label pairs. AU proxies come from MediaPipe blendshapes.
The waveform and cross-modal features are experimental discriminators. Reference pulse measurements
were unavailable.

[Pilot](signal_enhancement_comparison.csv) | [Paired families](paired_family_comparison.csv) | [Importance](feature_importance_summary.csv) | [Runtime](runtime_summary.json) | [ROC](roc_curves.png) | [Precision-recall](precision_recall_curves.png)

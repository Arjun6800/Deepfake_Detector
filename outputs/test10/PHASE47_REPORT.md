# Phase 4.7 expanded fusion and preliminary classification

Expanded rPPG supplied a small measurable increment in some comparisons, but **did not show a reliable improvement over facial expressions alone**. With Logistic Regression, the fused representation excluding explicit reliability reached **0.650 balanced accuracy versus 0.638 for expressions**, a **1.25-point** mean gain across eight held-out source families. Three families improved, three degraded and two tied. The paired family-resampling interval was **−8.13 to +11.25 points**. Signal-only rPPG showed more discrimination than explicit reliability alone, with pooled ROC-AUC **0.661 versus 0.487**.

## Fusion and evaluation design

The expanded representation joins 89 reference POS features to the existing 187 expression features for **276 inputs**. The original 69-feature rPPG and 256-feature fusion schemas were preserved and evaluated on the same windows. The 96 training videos each contributed one centered eight-second window. Clip IDs, source families, window boundaries and the decoded frame timeline were verified before joining. Alternative feature variants contribute only to their own comparison arms. [Fusion verification](FUSION_VERIFICATION.md) · [Expanded schema](feature_schema.json)

The eight independently assigned source families determined the feasible split count. Leave-one-family-out cross-validation held out all 12 related clips at once, leaving 84 training clips in each fold. Every ablation used the same folds and videos. Imputation, duplicate and constant removal, scaling and inverse-frequency class weights were fitted using training rows. Models were Logistic Regression, Random Forest, RBF SVM and HistGradientBoosting with fixed settings and no search over held-out scores. [Fold assignment](fold_assignment.csv) · [Leave-one-group-out reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)

Video score was defined as the mean of its window scores and classified as manipulated at 0.5 or above. Each video has one window, so no averaging across time was needed. SVM uses a sigmoid of the decision score for this common threshold. Scores are model-specific and uncalibrated. The primary comparison uses family-mean balanced accuracy with Logistic Regression. Pooled video ROC-AUC and average precision are secondary, and fold-level metrics expose differences in score ranking across families. Uncertainty intervals resample eight held-out family differences and are descriptive because model training sets overlap. [Video predictions](video_predictions.csv) · [Fold metrics](fold_metrics.csv) · [Paired comparisons](paired_family_deltas.csv)

## Video-level results

| Logistic Regression arm | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original rPPG69 | .729 | .563 | .855 | .813 | .833 | .702 | .931 |
| Expanded rPPG signal63 | .760 | .606 | .870 | .838 | .854 | .661 | .910 |
| Reliability26 | .750 | .500 | .833 | .875 | .854 | .487 | .837 |
| Expression187 | .771 | .638 | .882 | .838 | .859 | .734 | .936 |
| Original fusion256 | .750 | .600 | .868 | .825 | .846 | .752 | .940 |
| Expanded fusion276 | .740 | .644 | .887 | .788 | .834 | .727 | .930 |
| Expanded fusion250 without explicit reliability | .750 | .650 | .889 | .800 | .842 | .734 | .933 |
| GREEN89 | .760 | .581 | .861 | .850 | .855 | .541 | .849 |
| Missingness-mask probe | .250 | .500 | .833 | .125 | .217 | .500 | .833 |
| Majority baseline | .833 | .500 | .833 | 1.000 | .909 | .500 | .833 |

Average precision is the reported precision-recall summary. The cohort has five manipulated clips per original, explaining the majority rule’s high accuracy, F1 and average precision. Balanced accuracy and the confusion counts expose the trade-off. Logistic Regression expression-only had **7 TN, 9 FP, 13 FN and 67 TP**. Fusion250 had **8 TN, 8 FP, 16 FN and 64 TP**. Fold balanced-accuracy SD was **0.119** for expressions and **0.158** for fusion250. [All classifiers and ablations](ablation_comparison.csv) · [Confusion counts](confusion_matrices.csv) · [Logistic confusion matrices](confusion_matrices_logistic.png) · [ROC curves](roc_curves.png) · [Precision-recall curves](precision_recall_curves.png) · [Average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

For fusion250 versus expression-only, pooled ROC-AUC was effectively unchanged at 0.734, while mean within-family ROC-AUC rose from **0.794 to 0.831**. The paired change was **+0.0375** with a descriptive interval of **−0.094 to +0.156**. Random Forest and SVM also showed modest pooled balanced-accuracy gains for fusion250 over expressions, from 0.569 to 0.600 and 0.519 to 0.563. HistGradientBoosting fell from 0.575 to 0.519. Original fusion256 had the strongest pooled Logistic Regression ROC-AUC among the compared fusion representations, **0.752**. These mixed results leave incremental value unresolved.

For Logistic Regression, fusion250 improved manipulated recall over expression-only on Face2Face (87.5% to 100%) and NeuralTextures (68.8% to 75.0%). Recall fell on Deepfakes, FaceShifter and FaceSwap. Each estimate uses 16 manipulated videos of that method. Per-method ROC and other two-class metrics use the same 16 original clips for comparison. [Per-manipulation results](per_manipulation.csv)

## Features and runtime

Explicit reliability alone produced balanced accuracy near chance across the four classifiers. The fused missingness-mask probe gave ROC-AUC and balanced accuracy of 0.500. Expanded Logistic Regression did slightly better without the explicit reliability block. All 47 expanded missing cells were concentrated in FaceSwap rows, so this small cohort cannot rule out missingness shortcuts on a broader set.

Held-out feature-block permutation for Logistic Regression reduced mean within-family ROC-AUC in fusion250 by **0.201 for expression**, **0.051 for rPPG spectral**, **0.032 for spatial**, **0.030 for temporal** and **0.011 for amplitude features**. In fusion276 the mean drops for expression and reliability were **0.179 and −0.009**. These changes measure model reliance under permutation, and correlated blocks can substitute for one another. The largest mean absolute standardized fusion250 Logistic Regression coefficients included left-cheek 2–3 Hz relative power, forehead–left-cheek correlation and forehead 3–5 Hz relative power. A large coefficient in a correlated feature set does not establish an independent effect. [Held-out feature-block importance](feature_group_importance.csv) · [Training-fold coefficients](logistic_feature_coefficients.csv)

Aligned expression re-decoding and extraction took **382.4 seconds** across five workers. Across eight folds, fusion250 fitting and preprocessing took about **0.06 s** for Logistic Regression, **1.19 s** for Random Forest, **0.03 s** for SVM and **1.93 s** for HistGradientBoosting. These exclude expression and rPPG extraction. Prediction costs and fold-wise timings are in the result tables. The scoped classifiers used scikit-learn 1.9.1 in the existing project virtual environment. [Capture summary](expression_capture_summary.json) · [Evaluation settings](evaluation.json)

## Extraction decision

Run a larger, prespecified training-family replication before committing to full extraction. Retain expression-only, original fusion256 and expanded fusion250 as primary comparisons. Keep fusion276 and reliability26 as diagnostics, and include GREEN for extraction-cost comparisons. Report family-level uncertainty and label-blind coverage audit results, especially for FaceSwap. Evaluate whether gains persist across different source families and manipulation types before treating expanded rPPG as a routine addition. These results establish neither physiological accuracy nor a final detector performance estimate.

All **61 tests passed**. The experiment used only the original 96 training videos and the local face-landmark model, with source files opened for decoding. There was no full extraction or final model fit. [Test log](test_results.txt)

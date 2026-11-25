# Research asset index

All experimental metrics are from the recorded project outputs. The matched classifier figures use 484 approved training videos (84 original, 400 manipulated) from 47 source families, with the same video-to-fold assignments in Phases 4.9, 4.10 and 4.11A. These are exploratory outer out-of-fold estimates. The official validation and test partitions were untouched. Phase 4.10’s top recorded row is an exact Phase 4.9 replay.

PNG files are 300 DPI and fit a 179 mm-wide A4 content area. SVG editions retain selectable text. Metric tables store unit-interval scores unless their column names indicate otherwise. Manipulated is the positive class. Average precision is the reported PR-AUC. Separate rows in the workbook preserve cohort, phase and source provenance.

The Phase 4.10 implementation audit lists different TN and FP counts in prose. The original prediction and comparison CSVs agree on 59 TN and 25 FP for the historical replay. The tables and figures use those verified counts.

## Tables

[Excel workbook](tables/research_tables.xlsx) with one worksheet per CSV.

- [Dataset overview](tables/dataset_overview.csv). Datasets and splits. Unique scored clips by phase, eligible cohort, true class and manipulation. Subset rows overlap all-cohort totals.
- [Feature extraction comparison](tables/feature_extraction_comparison.csv). Methods, feature engineering. Requested, schema and retained dimensions have distinct definitions.
- [Model comparison](tables/model_comparison.csv). Results, classifier evaluation. All unique configurations and cohort summaries with the original pooled metrics and available family statistics.
- [Phase comparison](tables/phase_comparison.csv). Results, matched phase comparison. Includes the Phase 4.10 replay and saved winning fold settings.
- [Rppg signal quality](tables/rppg_signal_quality.csv). Results, signal quality. Phase 4.5 observed method/ROI summaries and Phase 4.10 observed pilot coverage/runtime. Ratios and gate passes describe signal quality proxies.
- [Feature importance](tables/feature_importance.csv). Results, model interpretation. Includes group permutation drops and separately identified Phase 4.7 fitted coefficient rows.
- [Ablation study](tables/ablation_study.csv). Results, ablation. Recorded arms and available paired family comparisons. Phase and source columns identify the design.
- [Actual vs targets](tables/actual_vs_targets.csv). Discussion, performance targets. Aspirational targets and actual-minus-target differences in percentage points.
- [Fold model configurations](tables/fold_model_configurations.csv). Methods appendix, model fitting. Saved run metadata, fold-specific selections where recorded, retained feature counts and timings.

## Graphs

- Accuracy comparison · [PNG](graphs/accuracy_comparison.png) · [SVG](graphs/accuracy_comparison.svg). Results, matched phase comparison. The Phase 4.10 point repeats Phase 4.9.
- Balanced accuracy · [PNG](graphs/balanced_accuracy.png) · [SVG](graphs/balanced_accuracy.svg). Results, matched phase comparison. The Phase 4.10 point repeats Phase 4.9.
- Precision recall f1 · [PNG](graphs/precision_recall_f1.png) · [SVG](graphs/precision_recall_f1.svg). Results, matched phase comparison. The Phase 4.10 point repeats Phase 4.9.
- Roc auc comparison · [PNG](graphs/roc_auc_comparison.png) · [SVG](graphs/roc_auc_comparison.svg). Results, matched phase comparison. The Phase 4.10 point repeats Phase 4.9.
- Phase wise performance · [PNG](graphs/phase_wise_performance.png) · [SVG](graphs/phase_wise_performance.svg). Results, phase comparison. Seven pooled metrics on identical clips and folds.
- Roc curves · [PNG](graphs/roc_curves.png) · [SVG](graphs/roc_curves.svg). Results, threshold-independent discrimination. Curves use saved video scores. Legend lists ROC-AUC.
- Precision recall curves · [PNG](graphs/precision_recall_curves.png) · [SVG](graphs/precision_recall_curves.svg). Results, manipulated-class performance. Legend lists average precision.
- Confusion matrices · [PNG](graphs/confusion_matrices.png) · [SVG](graphs/confusion_matrices.svg). Results, operating-point errors. Counts and actual-class row percentages use each configuration’s saved fold thresholds.
- Feature importance · [PNG](graphs/feature_importance.png) · [SVG](graphs/feature_importance.svg). Results, interpretation of the selected hybrid. Permuted blocks can share predictive signal.
- Ablation study · [PNG](graphs/ablation_study.png) · [SVG](graphs/ablation_study.svg). Results, feature ablation. Fixed mean aggregation and logistic L2. Historical and optimized arms may use different preprocessing.
- Dataset class distribution · [PNG](graphs/dataset_class_distribution.png) · [SVG](graphs/dataset_class_distribution.svg). Datasets and splits. Cohort counts by original and manipulation type.
- Current vs targets · [PNG](graphs/current_vs_targets.png) · [SVG](graphs/current_vs_targets.svg). Discussion, remaining performance gaps. Values and exact differences are in the targets table.
- Pos chrom green · [PNG](graphs/pos_chrom_green.png) · [SVG](graphs/pos_chrom_green.svg). Results, signal quality. Mean spectral concentration for the actual Phase 4.5 native-snap traces, pooled over the three facial regions.
- Facial region signal quality · [PNG](graphs/facial_region_signal_quality.png) · [SVG](graphs/facial_region_signal_quality.svg). Results, signal quality by anatomical region. Native-snap Phase 4.5 trace means.
- Rppg waveform comparison · [PNG](graphs/rppg_waveform_comparison.png) · [SVG](graphs/rppg_waveform_comparison.svg). Results, qualitative signal examples. Real cached RGB observations, Phase 4.10 variants on centered windows. Example selection uses prediction order and label.
- Frequency domain psd · [PNG](graphs/frequency_domain_psd.png) · [SVG](graphs/frequency_domain_psd.svg). Results, signal spectra. Same two observed clips, using project PSD code and the longest contiguous provenance-consistent run per variant.

## Diagrams

- System architecture · [PNG](diagrams/system_architecture.png) · [SVG](diagrams/system_architecture.svg). Methods overview. Connects video ingestion, the two feature branches, fusion and development evaluation. Sources deepfake_rppg/dataset.py, phase48_extract.py and phase411a.py.
- Rppg extraction pipeline · [PNG](diagrams/rppg_extraction_pipeline.png) · [SVG](diagrams/rppg_extraction_pipeline.svg). Methods, physiological feature extraction. Sources deepfake_rppg/phase410_features.py, quality_algorithms.py and consistency.py.
- Facial expression pipeline · [PNG](diagrams/facial_expression_pipeline.png) · [SVG](diagrams/facial_expression_pipeline.svg). Methods, facial expression feature engineering. Sources deepfake_rppg/expressions.py and outputs/test14/feature_schema.json.
- Multimodal fusion · [PNG](diagrams/multimodal_fusion.png) · [SVG](diagrams/multimodal_fusion.svg). Methods, fusion and aggregation. Sources deepfake_rppg/phase411a.py and outputs/test14/feature_schema.json.
- Sliding window mechanism · [PNG](diagrams/sliding_window_mechanism.png) · [SVG](diagrams/sliding_window_mechanism.svg). Methods, time aggregation. Schematic with illustrative times. The project rounds width and hop on the native frame grid and deduplicates offsets. Source deepfake_rppg/phase48_extract.py.
- Nested cross validation · [PNG](diagrams/nested_cross_validation.png) · [SVG](diagrams/nested_cross_validation.svg). Methods, evaluation protocol and leakage controls. Applies to the matched development evaluation. Sources deepfake_rppg/phase48_classify.py, phase49_classify.py and phase411a.py.
- Development timeline · [PNG](diagrams/development_timeline.png) · [SVG](diagrams/development_timeline.svg). Methods overview or background, experiment lineage. Sequence follows the Phase reports in outputs/test10 through outputs/test14.
- Current vs proposed architecture · [PNG](diagrams/current_vs_proposed_architecture.png) · [SVG](diagrams/current_vs_proposed_architecture.svg). Discussion, proposed work. The right side is a proposed evaluation design drawn from the open validation questions in Phase 4.5 and Phase 4.11A. It has no performance results.

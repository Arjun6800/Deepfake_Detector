# Phase 4.11A implementation audit

## Checkpoint and cohort lineage

- Phase 4.8 supplies original POS and legacy feature windows and synchronized numeric
  observations. The audit checks training membership, checkpoint keys and window support.
- The Phase 4.9 historical selective predictions supply the exact matched clip IDs,
  labels, families, folds and benchmark metrics. The Phase 4.9 selective recipe is also
  refit as a replay control.
- Phase 4.10 supplies the recorded POS-reference shortlist and provenance-keyed
  enhanced temporal summaries. The preparation stage reconstructs window temporal
  inputs from numeric observations, checks grid alignment and verifies all 15 summary
  fields against the stored checkpoint. The two spread features retain their original
  video-level definitions and enter feature-aggregation models only.
- Handcrafted cross-modal, enhanced spectral and nuisance controls are excluded from
  the feature matrix. The reused Phase 4.10 extractor may compute those values while
  reconstructing the time-domain features. The prepared cache stores only temporal values.

## Model and evaluation design

- Ten feature arms, including three historical-recipe controls. The seven new arms
  each use mean, mean–standard-deviation and mean window-score aggregation.
- The historical selective and full hybrid mean representations form the fixed
  classifier shortlist. Paired comparisons between shortlist classifiers use the
  same feature representation and preprocessing within each arm.
- The grid has 34 configurations and 170 outer-fold results. Two settings per model
  are tested except for modality regularization, which uses six settings. Selection
  ranks inner video-level OOF balanced accuracy, then inner ROC-AUC. The cutoff
  maximizing inner balanced accuracy is applied to the corresponding outer fold.
- Optimized arms summarize finite observations and retain per-feature missing
  fractions in feature-aggregation models. Standard deviations require at least two
  finite windows. Training medians serve only as model inputs. A missingness flag
  is learned when fitting rows expose both states. Historical controls retain
  Phase 4.9 imputation and scaling for replay comparability, so comparisons to
  those controls can include a preprocessing effect.
- Optimized preprocessing removes training-constant and near-constant values
  using numerical spread and 99.5% modal dominance. The redundancy arm greedily
  removes absolute Pearson correlations of .95 or larger after fold-local imputation
  and scaling, preferring greater observed coverage and then schema order.
- Modality regularization scales standardized columns by the inverse square root
  of modality width times penalty. The feature scaling induces separate L2
  coefficient penalties for expression and rPPG. The expression-to-rPPG
  penalty multiplier and global C are fit using only the inner training protocol.
- Every video contributes equal total fitting weight within its class, including
  window models. Paired bootstrap intervals resample equally weighted source
  families with both classes. Permutation importance shuffles held-out video
  blocks, preserving window counts and within-video ordering.
- The final ranking is post hoc. Previous development runs informed the feature
  choices. Results cannot establish independent modality contributions or official
  validation/test performance. Count strata can limit window permutation coverage.

## Verification and operational coverage

Regression tests cover alignment, preprocessing isolation, missing values,
redundancy, modality scaling, deterministic evaluation, checkpoint invalidation,
shortlisted estimator fitting and synthetic report generation. The two-video real
checkpoint smoke verifies summary reproduction and cache reload. The complete
experiment is pending standalone execution. No Phase 4.11B code is included.

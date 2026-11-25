# Phase 4.8 cohort and extraction approval brief

**Status: approved and implemented.** The study uses **600 FaceForensics++ C23 training videos from 50 independent source families**. The leakage-controlled manifest supplies the assignments. The selection retains all eight Phase 4.7 families and adds 42 complete families. No full extraction or classification was launched in Codex. [Standalone commands and handoff](RESUME.md)

## Cohort

| Video category | Retained | New | Total |
| --- | ---: | ---: | ---: |
| Original | 16 | 84 | 100 |
| Deepfakes | 16 | 84 | 100 |
| Face2Face | 16 | 84 | 100 |
| FaceShifter | 16 | 84 | 100 |
| FaceSwap | 16 | 84 | 100 |
| NeuralTextures | 16 | 84 | 100 |
| **Total** | **96** | **504** | **600** |

The planner found 328 eligible complete training families. Each selected family has two originals, both directions of all five manipulations, and at least eight seconds of nominal duration for every recording. It sampled new families with seed 480 from sorted eligible family IDs, without considering feature values or prior classifier outcomes. Source and upload identifiers and content hashes do not cross selected family boundaries. All 600 source paths were visible during planning. Source hashes will be checked before extraction. [Selection manifest](selection.json) · [Planning estimates](planning_estimates.json)

## Runtime, storage and reuse

| Planning item | Estimate or finding |
| --- | --- |
| New workload | 504 recordings and 250,276 audited frames. Total selected cohort has 296,916 frames. |
| Extraction time | **About 2.5–3.5 hours** on the current Ryzen 5 7530U. Scaling the Phase 4.5 full RGB pass and the Phase 4.7 AU pass to the planned frame counts gives **2.50 hours** before contingency. The estimate assumes comparable throughput. It excludes nested model evaluation. |
| Workers | Start with **five** checkpointed processes and one OpenCV and decoder thread per worker. Verify memory and throughput on one complete new family after approval. Use a smaller worker pool if Windows paging or decoder contention appears. |
| Storage | Estimated **74 MB** for new RGB observation archives at the Phase 4.5 compression ratio. A full float32 cache of the 30 selected expression scores for all audited frames is about **36 MB** before compression. Reserve **1–2 GB** for additional checkpoints, atomic staging, feature tables, optional waveforms and diagnostics. The workspace had about **5.0 GB free** during planning. |
| Existing checkpoints | Verified **96/96** compatible full-length RGB archives, **96/96** centered rPPG waveform archives and **96/96** centered expression checkpoints. The 96 centered fused rows also match the retained inventory. |

For the retained cohort, reuse the full RGB archives for other windows and the existing matched feature rows for the center. The expression checkpoints contain centered-window summaries, so additional expression windows require an aligned decoding pass on retained videos that contribute to the multi-window study. For new videos, capture RGB and expression scores in a shared decode. The 2.50-hour projection adds separate RGB and AU reference timings and therefore allows for duplicated work that a shared pass could avoid. Per-video checkpoint keys will record source hash, model, extractor revision and window protocol. Resume will validate complete checkpoints before skipping work. [Checkpoint reuse audit](checkpoint_reuse_audit.json)

## Evaluation protocol after approval

- Preserve the existing 69-, 89-, 187-, 256- and 276-feature definitions. Use reference POS for the main expanded representation and reference GREEN for the low-cost control. Use one centered eight-second window for the full 600-video replication.
- For temporal comparisons, retain the center and add complete eight-second windows at four-second hops when at least two regular windows are available. Deduplicate bounds. Metadata predicts **484** eligible multi-window videos and **2,296** windows including centers across the proposed cohort. Recheck decoded support and use the same eligible subset for all five temporal comparisons. Compare center, mean, mean plus standard deviation, median and mean of window scores. Give each video the same total training weight before class balancing in the window-model arm. Score at the video level.
- Use five outer grouped folds and four grouped inner folds for bounded hyperparameter search. Complete families have equal class composition, so the proposed outer folds each hold out ten families. Distribute retained families across folds. Fit every transformation and any model settings on training folds. Report pooled out-of-fold video results and results restricted to the 42 newly added families.
- Run the nine requested ablations on the same eligible videos and folds. Compare expression-only and fused scores using paired source-family resampling. Audit reliability, missingness, motion and encoding proxies without feeding dataset metadata into classifiers. Keep conclusions about fusion conditional on the observed family-level uncertainty.

The plan and selection can be regenerated with **python -m deepfake_rppg.phase48_plan**. This command reads manifests and feature archives only. The complete implementation regression suite passed **72 tests**. [Test log](test_results.txt)

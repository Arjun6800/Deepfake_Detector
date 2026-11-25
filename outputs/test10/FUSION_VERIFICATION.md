# Phase 4.7 fusion verification

The expanded fusion table contains **96 rows and 276 feature columns**, with 89 reference POS rPPG features and 187 facial expression features. The comparison table contains the corresponding 69-feature legacy rPPG block and the same expression block, for 256 features. Both use the centered eight-second windows from Phase 4.6. The original Phase 4 baseline artifacts are separate.

## Identity and timing

- Each record passed checks for clip ID, source family, training partition, window index and exact archived bounds. The legacy and expression records also matched first frame, last frame, decoded frame count and rPPG grid length.
- The expression pass re-decoded each original video after confirming its SHA-256. Every decoder timestamp matched the cached rPPG positions, and the frame indices within each half-open window matched the archived support. Waveform grid start, sample count and cadence were checked against the Phase 4.6 archive. Expression statistics used the original observation timestamps on that window.
- The expression pass used the same MediaPipe landmark model hash and reduced-detector policy. Facial scores were extracted through the existing expression sampler and summarized with the existing temporal feature functions.
- Only reference POS contributes to expanded fusion. Reference GREEN is a separate ablation on the same rows. The other Phase 4.6 experimental variants were excluded as observations.
- The 96 clips comprise 16 originals and 16 per manipulation method across eight families. Repeated source-video IDs, original upload IDs and content hashes cannot cross family assignments. Each held-out family contains 12 videos, including two originals and two clips per manipulation method.

## Values and classifier inputs

The feature manifest explicitly enumerates every input. Labels, clip IDs, source-family IDs, manipulation names, window timestamps and indices, file hashes and capture metadata are outside classifier inputs. The 26 expanded reliability features remain individually identifiable. The 63 expanded signal and consistency features omit those 26 fields while retaining spectral concentration as a spectral measurement. Expression-only retains the existing expression coverage features.

| Table | Features | All missing | Constant among finite observations | Exact duplicate of an earlier column | Missing cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expanded fusion | 276 | 0 | 0 | 58 | 47 |
| Legacy fusion | 256 | 3 | 5 | 66 | 595 |
| Expression only | 187 | 0 | 0 | 42 | 0 |
| Expanded signal and consistency | 63 | 0 | 0 | 0 | 47 |
| Expanded reliability | 26 | 0 | 0 | 16 | 0 |

The counts above describe the raw matrices and can overlap. All 47 expanded missing cells occurred in FaceSwap rows. Mean proportion of frames with all selected expression scores ranged from 98.78% for FaceSwap to 100% for Face2Face. The 187 window-level expression fields were present for every video. The join rejects duplicate headers, invalid input, infinite values and duplicate rows for a selected method.

Missing values stay missing in the saved feature tables. Within each training fold, the classifier drops all-missing columns, learns medians, drops constants and exact post-imputation duplicate columns, and standardizes the retained features. The held-out fold uses those fitted transformations. For expanded fusion, 217–218 of the 276 columns were fitted per fold. Full-cohort diagnostics did not select columns or tune models.

[Expanded schema](feature_schema.json) · [Alignment checks](alignment.csv) · [Feature integrity](feature_integrity.csv) · [Availability by method and feature](feature_availability.csv) · [Expanded features](fusion276.csv) · [Legacy comparison features](fusion256_reference.csv) · [Fold assignments](fold_assignment.csv)

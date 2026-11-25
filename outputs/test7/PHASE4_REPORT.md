# Phase 4 dataset audit and pilot

## Inventory and provenance

The primary inventory contains 7,000 files. The initial scan listed 404 clips from the excluded folder.

| Subset | Files | Label | Review | Resolution variants | Most common resolution | Codec tags |
| --- | ---: | --- | ---: | ---: | --- | --- |
| original | 1000 | REAL | 30 | 29 | 1280x720 (325) | h264 |
| Deepfakes | 1000 | FAKE | 30 | 29 | 1280x720 (325) | h264 |
| Face2Face | 1000 | FAKE | 30 | 14 | 640x480 (348) | h264 |
| FaceShifter | 1000 | FAKE | 30 | 29 | 1280x720 (325) | h264 |
| FaceSwap | 1000 | FAKE | 30 | 29 | 1280x720 (325) | h264 |
| NeuralTextures | 1000 | FAKE | 30 | 14 | 640x480 (348) | h264 |
| DeepFakeDetection | 1000 | FAKE | 0 | 1 | 1920x1080 (1000) | h264 |

Both roots were enumerated independently of the CSVs. The audit found 0 metadata paths absent from disk, 0 primary video paths absent from metadata, 0 distinct within-file duplicate metadata keys, 0 other files and 1 missing directories. Repeated references across per-method, combined and shuffled CSVs are expected. Probe errors affected 0 files and metadata label conflicts affected 0 files.

Observed mismatches by metadata field, counted once per affected video, were {}. The audit compares the rounded file size in MiB, reads stream properties from the container, and estimates duration from frame count divided by reported FPS. The pilot tests sequential frame decoding.

SHA-256 found 0 byte-identical sets with multiple primary videos. Among the excluded clips, 404 share a primary NeuralTextures filename, 0 match that namesake byte for byte and 0 match any primary file byte for byte. Hashes identify exact file duplicates. Re-encoded or visually near-identical copies remain unresolved.
Hashes were available for 0/404 excluded clips.
The identities of 404 excluded clips remain unresolved because their files were unavailable for hashing. The comparison retains their enumerated filenames.

The audit searched the first 64 KiB of each file for embedded x264 encoder settings. The observed CRF and encoder core counts below provide evidence about the local encoding. Exact release provenance would require trusted upstream hashes. DeepFakeDetection is a Google dataset extension. The local 1,000 clips represent a subset of the documented release. Genuine counterparts and verified source links were unavailable.
See the [FaceForensics++ dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md).

| Subset | Embedded CRF counts | x264 core counts |
| --- | --- | --- |
| original | {'23.0': 1000} | {'148': 749, '157': 2, '152': 249} |
| Deepfakes | {'23.0': 1000} | {'148': 998, '157': 2} |
| Face2Face | {'23.0': 1000} | {'148': 996, '157': 4} |
| FaceShifter | {'23.0': 1000} | {'148': 469, '160': 2, '152': 529} |
| FaceSwap | {'23.0': 1000} | {'148': 997, '157': 3} |
| NeuralTextures | {'23.0': 1000} | {'148': 992, '157': 8} |
| DeepFakeDetection | {'23.0': 1000} | {'157': 1000} |
| NeuralTextures (excluded) | {} | {} |

### Comparison with the named target originals

The first ID in each official manipulated filename identifies the target original. The local frame counts and dimensions show method-specific differences even when the CSV agrees.

| Manipulation | Different resolution | Different frame count |
| --- | ---: | ---: |
| Deepfakes | 0 | 0 |
| Face2Face | 282 | 992 |
| FaceShifter | 0 | 0 |
| FaceSwap | 0 | 496 |
| NeuralTextures | 282 | 497 |

Among partitioned files, 33 container durations are below eight seconds. The batch records these as `no_windows` whenever the decoded timestamps also fall short.

## Partition integrity

The [official split definitions](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) contain {'train': 720, 'val': 140, 'test': 140} source IDs by split and 1000 directed pairs. The manifest partition counts are {'holdout': 1180, 'test': 816, 'train': 4236, 'val': 768}. The audit flagged 180 primary videos in components touching multiple official splits, 0 unresolved primary source links outside DeepFakeDetection, and 0 manipulated filenames outside official pairs.

The official conversion map links 1000 source clips to 977 YouTube uploads. 7 uploads appear across official partitions. Components connect original IDs, both IDs from a manipulated filename, shared uploads in the official conversion map, and identical file hashes. Components touching several official partitions or DeepFakeDetection stay in holdout. All overlapping windows inherit the video's partition and group ID. The feature CSVs contain only the modality features. Join labels and partitions through the separate window and manifest records. Subject identity across distinct uploads and copies with different encodings remain unverified. Fit any future learned preprocessing on the training partition.

## Pilot

The initial run exposed JSON serialization of NumPy frame indices. The corrected run below passed archive verification and is the basis for the estimates.

The pilot selected 30 videos using the saved selection and default seed 42. It used five training source pairs, both originals per pair and four fakes per method. Each clip was decoded once with shared landmark and blendshape inference. The existing 0.35 spectral concentration threshold, 0.5 processed coverage threshold and 6.0 s minimum pulse support were retained. Quality flags describe measurement support and carry no authenticity interpretation.

Extraction statuses were {'complete': 30}. The successful extraction rate was 100.0% with 90 synchronized windows.

Wall time was 487.0 s with 4 workers. Per-video elapsed seconds were {'min': 26.735603699999047, 'p25': 34.27538577500309, 'median': 47.08007610000277, 'p75': 69.37998597500155, 'max': 122.68314190000092, 'mean': 59.08520590333331}.

The rPPG quality gate passed 6/540 window traces. Spectral concentration was unavailable for 122 traces. Available concentration quantiles were {'min': 0.008409855793063392, 'p25': 0.07278272837698549, 'median': 0.1185152282126317, 'p75': 0.18728815579511104, 'max': 0.4264736540564899}.

Complete selected-channel expression coverage by video was {'min': 0.0, 'p25': 0.9906055900621118, 'median': 1.0, 'p75': 1.0, 'max': 1.0}. Frame-weighted coverage was 0.8457304066607032. Missing feature percentages were {'rppg': 18.502415458937197, 'expression': 15.401069518716577, 'fused': 16.236979166666668}. Missing measurements remain empty in CSV and counts include successful windows only.

| Method | Pilot videos | Complete | Windows | Mean processing seconds |
| --- | ---: | ---: | ---: | ---: |
| original | 10 | 10 | 31 | 60.73 |
| Deepfakes | 4 | 4 | 17 | 60.60 |
| Face2Face | 4 | 4 | 9 | 38.78 |
| FaceShifter | 4 | 4 | 7 | 67.21 |
| FaceSwap | 4 | 4 | 8 | 63.23 |
| NeuralTextures | 4 | 4 | 18 | 61.49 |

Pilot checkpoint archives occupied 418,833 bytes. The projections scale elapsed time and compressed archive size by container frame counts. Resolution, video duration, I/O and unsuccessful extraction can shift the actual throughput. The all-primary scenario includes unpiloted DeepFakeDetection and is a rough capacity scenario.

| Scenario | Videos | Worker hours | Wall hours | Checkpoint MiB |
| --- | ---: | ---: | ---: | ---: |
| eligible_ffpp | 5820 | 93.68 | 25.74 | 76.00 |
| all_primary_scenario | 7000 | 121.38 | 33.35 | 98.46 |

## Readiness

The batch pipeline can process eligible FaceForensics++ rows by official partition after pilot review. Review flagged manifest rows and keep the DeepFakeDetection and extra NeuralTextures files outside the training-ready selection. Phase 4 ends with the pilot.

Per-video paths, metadata mismatches, flags, processing times, coverage and failures are available in the manifest, audit, excluded-folder comparison and pilot summary alongside this report.

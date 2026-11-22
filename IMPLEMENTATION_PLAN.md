# Deepfake Detection Using rPPG and Facial Action Unit Fusion

## Workspace audit

The project contains a read-only reference checkout at `reference_code/prism-rppg/`, audited at commit `76ae9456d460153c80ed3cab8b15bc5c3f7d3fb3`. See [CODE_REUSE_MAP.md](CODE_REUSE_MAP.md) for exact symbols, integration defects, weight paths, and licensing evidence.

| Component | PRISM capability | Project decision |
| --- | --- | --- |
| Face detection | MediaPipe boxes for router features and ONNX BlazeFace boxes and six keypoints for neural inference | Use a verified detector or landmark package directly. Build shared identity tracking and forehead and cheek ROIs. |
| rPPG | Neural BVP inference and waveform postprocessing | Start with classical green and POS. Evaluate a fixed neural estimator after verifying its assets and waveform contract. |
| Filtering and quality | Butterworth bandpass, sparse detrending, autocorrelation SQI, and optical quality proxies | Use SciPy directly. Implement guarded, mask-aware quality features per region and window. |
| AU extraction | No AU inference or weights | Add a verified AU backend and schema. |
| Temporal alignment | Single-stream timestamps and frame fill/drop | Build a shared decoder clock, track association, uniform rPPG grid, and modality masks. |
| Fusion and classification | PRISM predicts heart-rate estimator error and selects a specialist | Build rPPG-only, AU-only, early-fusion, and late-fusion real/fake experiments. |

PRISM's README leaves license terms pending and flags possible institutional restrictions. Obtain permission for PRISM-specific code or assets before copying or integrating them. For upstream Open-rPPG code, use the original licensed source and verify model-specific terms independently.
## Dependencies and weights

The environment observations below are from the prior workspace audit. Recheck compatibility and availability before installation.

| Item | Current evidence | Decision or gate |
| --- | --- | --- |
| Python | 3.13.7 at C:/Program Files/Python313/python.exe | Verify decoder and AU backend compatibility before choosing an isolated environment and pinning versions. |
| Scientific stack | Imports verified for NumPy 2.3.3, SciPy 1.18.1, pandas 2.3.2, scikit-learn 1.9.1, and PyArrow 25.0.1 | Reuse for signal processing, features, classical classification, and optional Parquet output. |
| Video decoding | No importable PyAV or OpenCV. FFmpeg and FFprobe did not resolve on PATH. | Prefer PyAV if its timestamp handling and Windows build pass a smoke test. Use decoder presentation timestamps and verify variable-frame-rate behavior. |
| Face tracking and AU inference | No importable MediaPipe, Py-Feat, PyTorch, torchvision, or ONNX Runtime. OpenFace FeatureExtraction.exe did not resolve on PATH. | Evaluate OpenFace as the initial AU backend and as a potential source of landmarks. Evaluate Py-Feat if OpenFace is unsuitable. Approve and install only the selected stack. The scikit-learn baseline needs no PyTorch training stack. |
| Pretrained artifacts | The PRISM checkout lacks ONNX detector, HDF5 specialist, and router pickle assets. Historical Git LFS pointers and upstream file listings identify candidates. External payloads and model loading are unverified. | Classical POS needs no weights. Track and AU inference require the chosen backend's assets. A deepfake classifier can be trained locally once labeled data and features are available. |
| Dataset and reference pulse | No videos, labels, dataset access configuration, or reference pulse recordings found. | Establish authorized video access and provenance. Use synchronized reference pulse where available to validate physiological accuracy. |

Check the upstream package or executable, Windows/Python requirements, code and weight licenses, and intended research usage. Request approval before installing dependencies. Record the exact backend version, asset source, expected files, and SHA-256 hashes. Verify the model loads and produces the expected schema on an authorized clip before treating weights as available. Configure asset paths explicitly and surface missing assets at startup. [OpenFace AU documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) · [Py-Feat model documentation](https://py-feat.org/pages/models/)

## Implementation sequence

### 1. Working local rPPG extraction pipeline

- Build a decoder that emits presentation time in seconds, frame index, and RGB pixels. Record the timebase and any missing or nonmonotonic timestamps. Explicitly flag nominal-FPS fallback for confirmed constant-rate clips.
- Define one target face track per clip for the first experiment. Preserve identity through short detection gaps and mark ambiguous or lost tracks. Associate both modalities with this same track.

- Use tracked landmarks to define forehead and left/right cheek regions. Exclude eyes, lips, occlusions, and invalid pixels. Use original decoded pixels for color traces. Record per-frame region coverage, pose or motion indicators, and mean RGB values.
- Resample valid RGB traces onto a uniform grid for POS and filtering. Avoid bridging long gaps. Implement POS with guarded normalization and overlap-add. Keep a green-channel baseline and version the filter band, sample rate, and region definitions. Validate the filter band against Nyquist and exclude segments too short for reliable filtering.
- Store per-region waveforms, masks, and quality measures. Derive window features such as spectral concentration, dominant frequency, and inter-region coherence. Preserve usable coverage and missingness separately from pulse amplitude.

**Acceptance** — On one authorized clip, produce timestamped per-region BVP, coverage masks, quality, and diagnostic plots. Exercise variable frame rate, short clips, and multiple faces. Recover injected pulse frequencies from synthetic RGB traces, handle constant and missing traces without invalid numerical output, and compare to synchronized reference pulse if obtained. A synthetic test establishes algorithm behavior. Claims about physiological accuracy require reference data. [Wang et al., Algorithmic Principles of Remote PPG](https://research.tue.nl/en/publications/algorithmic-principles-of-remote-ppg/)

### 2. Dataset manifest and splits

- Create a manifest with video ID, path, real/fake label, original source or derivative group, known identity IDs, manipulation method, and compression level when available. Record excluded and unreadable clips with reasons.
- Split by original source lineage and group by known identities where feasible. Keep original and manipulated variants together. Freeze train, validation, and test assignments before constructing windows or fitting any transform.
**Acceptance** — A labeled fixture manifest validates, unreadable clips have recorded reasons, and no known source group crosses a split.

### 3. AU extraction

- Implement an adapter to emit video ID, face track ID, frame or presentation time, backend version, AU names and values, and available quality fields. Begin with intensity outputs when supported. Store presence separately and version the selected AU set.
- For a video-level backend, verify how frame numbers and timestamps map to the decoder, especially on variable-frame-rate clips. Confirm face identity through tracking or controlled input crops. Preserve a way to map every inference row back to the original frame.
- Mask unsuccessful detections, low-confidence rows, and unavailable AUs. Keep observed zero activation distinct from a missing value. Validate schema, value semantics, and row-to-frame mapping on clips with clear facial movement.

**Acceptance** — A verified, locally loadable backend returns correctly mapped AU rows for a smoke-test clip and reports a missing model path clearly. Alignment development can proceed with synthetic AU records while model assets are pending.

### 4. Temporal alignment and feature storage

- Join by video, face track, and presentation time. Compute rPPG on its uniform grid and summarize AU observations within the same half-open time windows. For sequence experiments, resample continuous AU values only across short gaps and retain observation masks. Use nearest-frame matching with a bounded tolerance for discrete presence.
- Start with configurable 8-second windows and 4-second hop as provisional settings. Review window coverage and spectral resolution on training clips before freezing settings. Define short-clip handling and per-modality minimum-valid-coverage rules. Record windows rejected or scored with one available modality.
- Store one row per aligned window keyed by video ID, track ID, start, and end. Include split, label reference, features, masks, and quality fields. Save configuration, schema, extractor versions, and model hashes alongside cached features.

**Acceptance** — Fixtures verify clock offsets, irregular frame times, dropped detections, window boundaries, and track switches. Long gaps remain masked and windows never span separate face tracks.

### 5. Fusion, classification, and evaluation

- Train rPPG-only and AU-only regularized logistic regression baselines using training-fitted preprocessing. Aggregate AU features over each window using statistics such as intensity mean and variance, activation fraction where presence is available, and changes over time.
- Train early fusion using concatenated branch features with fold-fitted imputation and scaling. Compare against late fusion from calibrated unimodal scores, using held-out validation or out-of-fold training predictions for calibration and weighting. Specify how single-branch predictions and full abstentions are handled.
- Fit at window level with weighting to limit dominance by long clips, then aggregate held-out scores to video level using a fixed rule selected on validation. Compare modalities on identical eligible videos, and report coverage plus a separate missing-modality analysis. Run a quality-only baseline to detect reliance on extraction failures or compression.
- Report video-level ROC-AUC, PR-AUC, balanced accuracy, sensitivity and specificity at a validation-selected threshold, plus confidence intervals resampled by source group where feasible. Break down performance by available manipulation, compression, and acquisition conditions. Evaluate on a held-out dataset if access permits.
- Save split manifests, feature schemas, seeds, preprocessing, model parameters, thresholds, and video-level predictions. Consider learned temporal fusion only after baseline comparison and a review of dataset size and compute requirements.

**Acceptance** — A small CPU run produces reproducible video-level predictions and ablation results. The test split remains held out from preprocessing, calibration, and model or threshold selection.

## Proposed code layout

| Path | Responsibility |
| --- | --- |
| src/deepfake_rppg/video.py and tracking.py | Timestamped decoding and face track selection |
| src/deepfake_rppg/rppg.py and au.py | Region signal extraction and backend AU adapter |
| src/deepfake_rppg/alignment.py and features.py | Masked alignment, shared windows, and feature generation |
| src/deepfake_rppg/dataset.py | Manifest validation and source-grouped splits |
| src/deepfake_rppg/train.py and evaluate.py | Unimodal and fused models, video scoring, and evaluation |
| tests/ | Synthetic signal, timestamp, missing-modality, and split-integrity fixtures |

After the classical pipeline works, optionally compare one fixed, licensed neural rPPG estimator on the same clips and time windows. Add PRISM routing only if the router artifacts, provenance, and compatible feature schema are verified and routing serves an experiment.


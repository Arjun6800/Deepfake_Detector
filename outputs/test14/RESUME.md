# Phase 4.11A execution and resume

Run from the project root in PowerShell using the existing project environment.

```powershell
.\.venv\Scripts\python.exe -B -m deepfake_rppg.phase411a prepare
.\.venv\Scripts\python.exe -B -m deepfake_rppg.phase411a classify
.\.venv\Scripts\python.exe -B -m deepfake_rppg.phase411a evaluate
.\.venv\Scripts\python.exe -B -m deepfake_rppg.phase411a report
```

Run each command after the preceding command succeeds. Repeat an interrupted command
to resume. Preparation commits a checkpoint per video. Classification commits one
checkpoint per configuration and outer fold. Both use atomic writes and revision
checks. Evaluation and report stages can be rerun from completed checkpoints.
Run one Phase 4.11A command at a time. The Windows lock is released when the process exits.

Preparation validates Phase 4.8 window support and saved Phase 4.9 cohort/fold metadata.
It also checks each enhanced temporal result against its Phase 4.10 checkpoint.
Use the environment that produced the checkpoints if the audit reports a version mismatch.

Allow roughly **5–30 minutes** for all four stages on a Ryzen 5 7530U. This is a planning
estimate from the short checkpoint smoke and bounded workload. The complete run has
not been timed. Models and numerical libraries use one worker/thread. Classification
progress counts both fitted and reused folds. Its ETA reflects the observed rate and
can change as more expensive classifiers start.

Budget **100 MiB** for Phase 4.11A files and working headroom beyond the **2 GiB**
reserve. Writes require at least 64 MiB of headroom above the reserve, or more
for a larger payload. The existing Phase 4.8–4.10 checkpoints stay in place.
No video decoding, downloads, datasets or pretrained weights are required.

Optional verification commands:

```powershell
.\.venv\Scripts\python.exe -B -m deepfake_rppg.phase411a smoke
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_phase411a.py' -v
```

The smoke writes its two-video cache under `outputs/test14/smoke/`.
The standalone stages write production artifacts under `outputs/test14/`.

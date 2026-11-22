# MediaPipe acquisition and compatibility

The project virtual environment uses Windows x64 CPython 3.13.7 and OpenCV 4.11.0 (`opencv-python-headless` 4.11.0.86). PyPI publishes a `py3-none-win_amd64` MediaPipe 0.10.32 wheel, which matches this interpreter and platform. The intended integration passes OpenCV-decoded frames to MediaPipe as RGB arrays. Runtime compatibility, including imports, model loading, and inference alongside the existing OpenCV build, remains unverified. [MediaPipe wheel](https://pypi.org/project/mediapipe/0.10.32/) · [Python API](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python)

| Model provenance | Value |
| --- | --- |
| Model | MediaPipe Face Landmarker FP16 task bundle, Google Storage version 1, with FaceMesh V2 |
| Versioned upstream URL | [Google-hosted version 1 bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |
| Official documentation download | [LiteRT Community bundle](https://huggingface.co/litert-community/MediaPipe-FaceLandmarker/resolve/main/face_landmarker.task) |
| License | Apache-2.0 per the [model card](https://huggingface.co/litert-community/MediaPipe-FaceLandmarker) |
| Publisher SHA-256 for the LiteRT bundle | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| Local SHA-256 | Unavailable. No bundle bytes reached the workspace. The publisher hash has yet to be checked against a local file or the versioned Google Storage object. |

The shell's PyPI request failed with Windows socket error 10013. Direct connections to the wheel host, Google Storage, and Hugging Face also failed. The session approval policy rejected the escalated network request. The workspace and pip cache contained no usable MediaPipe wheel or Face Landmarker bundle.

All **15 existing unit tests passed**. The extraction code and virtual environment have no changes from this attempt. Landmark geometry and masking tests, test3 extraction, and comparison plots require a runnable detector. The rectangular baseline remains available at [test2](../test2/diagnostic_report.md).

An offline handoff can use a workspace folder containing the matching MediaPipe wheel, its required dependency wheels, and the official `face_landmarker.task` bundle. The package can then be installed inside `.venv` with pip's offline mode, and the bundle hash and OpenCV interoperability can be checked before implementation and extraction.

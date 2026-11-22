# Phase 1.4 ROI reliability and signal stability

On `01_02__meeting_serious__YVGY8LOK.mp4`, stabilized landmarks recovered 136 processed image-right cheek samples compared with the Phase 1.3 method. Coverage rose from **56.80% to 69.83%**. Image-left cheek coverage rose from **90.80% to 98.75%**. The clipped rectangular fallback recovered 105 additional right-cheek RGB observations, but source boundaries reduced continuity and left hybrid processed coverage at **69.44%**. All four methods returned `low_quality` at the existing 0.35 concentration threshold and 50% processed-coverage requirement.

## Where the observations were lost

The four methods used one decode, one Haar face track, and one landmark inference per accepted track frame. The video had 1,044 frames at 24 FPS and 43.458 seconds between the first and last frames. All modes lost the same 15 frames to tracking. Landmark inference or association lost zero additional frames on this clip. The following counts describe missing RGB before interpolation.

| Method | Forehead missing | Image-left cheek missing | Image-right cheek missing |
| --- | ---: | ---: | ---: |
| Rectangular | 15 | 15 | 15 |
| Phase 1.3 landmarks | 15 | 99 | 316 |
| Stabilized landmarks | 15 | 15 | 178 |
| Hybrid | 15 | 15 | 73 |

Beyond the tracking misses, Phase 1.3 had **55 empty masks and 29 masks under 16 pixels** on the image-left cheek and **249 empty and 52 under 16 pixels** on the image-right cheek. The revised method eliminated those left-cheek failures and reduced the right-cheek failures to **111 empty and 52 under 16 pixels**. Exposure screening caused zero additional RGB losses. The 16-pixel and 50% exposure-valid requirements were retained.

Nose position relative to the face box serves as a rotation proxy. In the 496 accepted frames with normalized nose offset at least +0.2, valid image-right cheek RGB increased from **197 to 335** with stabilized landmarks and to **440** with fallback. At offsets below −0.2, valid image-left cheek RGB increased from **312 to 396 out of 396**. This proxy reports image geometry and has no angle calibration. [Per-region counts](coverage_by_region.csv) · [Mask area and rotation proxy](mask_support.png) · [ROI overlays](roi_mask_samples.png)

## What changed in signal support

The revised cheek polygons extend across more of the cheek while following the mesh. Smoothing uses face-relative coordinates, tracks current translation and scale, and resets after a miss or large shape step. The current inset face outline and buffered eye, brow, and lip exclusions constrain the sampled pixels. Hybrid uses the rectangle only within the permitted neighborhood of the current region and retains its source label. Interpolation, POS windows, and filtering do not cross source changes. The production interpolation limit remains 0.25 seconds.

| Method | Forehead processed | Image-left cheek processed | Image-right cheek processed | Image-right cheek RGB → raw POS → processed |
| --- | ---: | ---: | ---: | --- |
| Rectangular | 98.75% | 98.75% | 98.75% | 1,029 → 1,031 → 1,031 |
| Phase 1.3 landmarks | 98.75% | 90.80% | 56.80% | 728 → 672 → 593 |
| Stabilized landmarks | 98.75% | 98.75% | 69.83% | 866 → 850 → 729 |
| Hybrid | 98.75% | 98.75% | 69.44% | 971 → 806 → 725 |

Green and POS had identical processed coverage within each mode. For the image-right cheek, the bounded resampler produced 730, 881, and 976 RGB grid samples for legacy, stabilized, and hybrid respectively. The corresponding POS stages lost 58, 31, and 170 samples because the finite source runs did not support a complete POS window. Filtering then excluded another 79, 121, and 81 POS samples. For hybrid, the 105 fallback observations increased native RGB coverage from 82.95% to 93.01%, but the shorter uninterrupted runs limited the usable waveform. [Coverage plot](coverage_comparison.png) · [Signal statistics](signal_comparison.csv)

## Comparison on identical timestamps

The comparison starts from the intersection of valid **decoded RGB frames** across the four methods for each region. It rebuilds POS and filtering on that intersection and leaves every missing frame unfilled. The regions shared 1,029 forehead, 945 image-left cheek, and 728 image-right cheek RGB frames. All four methods had identical processed timestamps for each trace, with 995, 910, and 583 processed samples respectively. Hybrid equals stabilized landmarks on this intersection because fallback only contributes where stabilized RGB is missing. Its effects are visible in the full-run coverage and scores.

| Trace | Rectangular | Legacy landmarks | Stabilized landmarks | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Forehead green | 0.0656 | 0.0457 | 0.0467 | 0.0467 |
| Forehead POS | 0.1729 | 0.1079 | 0.1045 | 0.1045 |
| Image-left cheek green | 0.0421 | 0.0281 | 0.0402 | 0.0402 |
| Image-left cheek POS | 0.1760 | 0.1768 | 0.1621 | 0.1621 |
| Image-right cheek green | 0.0578 | 0.0473 | 0.0706 | 0.0706 |
| Image-right cheek POS | 0.2396 | 0.2479 | 0.1982 | 0.1982 |

On adjacent shared observations, the 95th-percentile box-relative centroid step decreased from 2.03% to 1.66% on the left cheek and 2.08% to 1.87% on the right cheek. The corresponding green-channel steps decreased from 2.61 to 2.22 and from 2.43 to 1.72 intensity levels. These measures combine the effects of polygon enlargement and smoothing. Spectral stability was mixed. The image-left cheek POS peak IQR widened from 0.25 to 0.75 Hz across five fully supported 8-second windows, and the matched image-right cheek had no complete 8-second window. [Matched waveforms](matched_waveforms.png) · [Shared timestamps](shared_timestamps.csv) · [Spatial and RGB stability](stability_by_region.csv)

## Decision and remaining limitations

**Retain stabilized landmarks as a research mode and keep rectangles as the default.** The revised geometry improves skin localization during turns and recovers useful segment length relative to Phase 1.3. On this clip it still yields less uninterrupted right-cheek signal than rectangles, and its matched right-cheek POS concentration is lower. Keep hybrid optional for further evaluation because its extra color samples did not increase processed coverage over pure landmarks.

The masks are geometric and can still contain hair, shadows, or other occlusions. Strong foreshortening can leave too little visible cheek, and the dependence on Haar leaves the common tracking gaps. The geometry was evaluated on one clip. There is no synchronized reference pulse, so neither higher concentration nor improved spatial stability establishes physiological accuracy.

All 22 unit tests passed. The report can be reproduced from the project root with the following commands.

```powershell
.\.venv\Scripts\python.exe -m deepfake_rppg.roi_experiment 01_02__meeting_serious__YVGY8LOK.mp4 outputs/test4 --landmark-model models/face_landmarker.task
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

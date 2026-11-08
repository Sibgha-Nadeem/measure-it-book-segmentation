# Definition of Done

The project is complete when each requirement below is implemented, tested and documented. A box is ticked only with evidence (a number, a file or a commit link) beside it. No results are fabricated.

## 1. Camera calibration
- [x] At least 20 checkerboard images captured. (36 used: `calibration/images/`, commit link to be added)
- [x] Camera matrix and distortion coefficients saved. (`calibration/camera_params.npz`, commit link to be added)
- [x] Reprojection error calculated and reported. (0.968 px over 36 images: `calibration/calibration_log.txt`, table in `docs/Objectives.md`)
- [x] Undistortion demonstrated on sample images. (`calibration/undistort_demo.py`, `calibration/undistort_demo.jpg`; visually inspected by project author)
- [x] Numerical straightness comparison completed. (36/36 images improved; mean line-fit RMS: 0.671 px raw, 0.446 px undistorted; `calibration/straightness_check.py`, `calibration/straightness_log.txt`)

Note: The straightness test uses the same 36 images used for calibration, so it is not an independent validation. The checkerboard print had ink streaks and was not mounted on a rigid backing.

## 2. Dataset
- [ ] At least 70 book images captured (target: 100).
- [ ] Images undistorted.
- [ ] Segmentation masks created and checked.
- [x] Fixed 70/20/10 split documented and committed before training. (`dataset/splits/split.txt`, commit link to be added)
- [ ] Test images kept separate from training and validation.

## 3. Segmentation model
- [ ] A segmentation model trained with PyTorch. (planned: U-Net; reason to be documented)
- [ ] Model weights saved, with SHA-256 hash in the README.
- [ ] IoU reported.
- [ ] mAP@0.5 reported.
- [ ] mAP@0.5:0.95 reported.
- [ ] Precision and recall reported.

## 4. Measurement pipeline
- [ ] A single command runs the image-to-measurement pipeline.
- [ ] Input image is undistorted.
- [ ] Book segmentation is performed.
- [ ] Known-size reference object is detected.
- [ ] Width and height are output in millimetres.
- [ ] Output includes an overlay and confidence information.
- [ ] Missing book or reference detections are handled safely.
- [ ] Low-confidence measurements are refused.

## 5. Accuracy testing
- [ ] At least 10 instances tested.
- [ ] Predictions compared with hand measurements (ruler or calliper).
- [ ] Per-instance table included, with raw outputs saved.
- [ ] Mean absolute error, mean percentage error, median and worst case calculated.

## 6. Additional experiments
- [ ] Raw versus undistorted measurements compared, with an explanation using my own numbers.
- [ ] MO-1 and MO-2 targets evaluated: met, or missed with the reason written down.
- [ ] At least one failure condition tested, and why I chose it.
- [ ] Confidence-based refusal demonstrated, with the threshold and the number of refused images.

## 7. Documentation and submission
- [ ] README: setup, how to run each stage, hardware, dataset summary (object, image count, split, labelling tool).
- [ ] Plan (with decision record), Objectives and Definition of Done are in `docs/`.
- [ ] Random seeds, library versions and training config pinned.
- [ ] Results and limitations documented honestly.
- [ ] Everything I did not finish is listed here, with the reason: [fill in at the end]
- [ ] Public GitHub repository with incremental commits on `dev-test01`, and a pull request into `main`.
- [ ] Loom demonstration (5 minutes maximum) recorded.
- [ ] Four assessment questions (K1 to K4) answered in the demonstration.

**Current physical measurements (book):**
- Height: 7.8 inches = 198.12 mm (three repeated measurements gave the same reading).
- Width: 5.1 inches = 129.54 mm (three repeated measurements gave the same reading).
- Thickness: 1 inch = 25.40 mm (three repeated measurements gave the same reading).

**Checkerboard:** 10 × 8 squares, 9 × 7 internal corners, previously reported dimensions 200 × 160 mm, nominal square pitch 20 × 20 mm. Actual physical pitch still needs to be verified before final metric measurement.
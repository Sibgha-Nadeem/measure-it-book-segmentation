# Definition of Done

The project is complete when each requirement below is implemented, tested and documented. A box is ticked only with evidence (a number, a file or a commit link) beside it. No results are fabricated.

## 1. Camera calibration
- [x] At least 20 checkerboard images captured. (36 used: `calibration/images/`, see git history)
- [x] Camera matrix and distortion coefficients saved. (`calibration/camera_params.npz`, see git history)
- [x] Reprojection error calculated and reported. (0.968 px over 36 images: `calibration/calibration_log.txt`, table in `docs/Objectives.md`)
- [x] Undistortion demonstrated on sample images. (`calibration/undistort_demo.py`, `calibration/undistort_demo.jpg`; visually inspected by project author, change was subtle)
- [x] Numerical straightness comparison completed. (36/36 improved; mean line-fit RMS 0.671 px raw, 0.446 px undistorted; `calibration/straightness_check.py`, `calibration/straightness_log.txt`)

Note: the straightness test uses the calibration images, so it is not an independent validation. The checkerboard print had ink streaks and was not on a rigid backing.

## 2. Dataset
- [x] At least 70 book images captured (target: 100). (100 kept from 105 taken, `dataset/splits/name_map.csv`)
- [x] Images undistorted. (`dataset/prepare_images.py`, using `calibration/camera_params.npz`)
- [x] Segmentation masks created and checked. (100 masks from `dataset/annotate.py`; the 8 smallest masks were viewed and no problems were found. Other masks were not all inspected individually.)
- [x] Fixed 70/20/10 split documented and committed before training. (`dataset/splits/split.txt`, see git history)
- [x] Test images kept separate from training and validation. (`book_091-100` were never loaded by `models/train.py`. They were first used by `models/evaluate.py`, run once, and afterwards for the MO-1 accuracy test. Nothing was tuned on them.)

## 3. Segmentation model
- [x] A segmentation model trained with PyTorch. (U-Net, ResNet34 encoder, `models/train.py`; reason in `docs/Objectives.md`)
- [x] Model weights saved, with SHA-256 hash in the README. (`models/best.pt`, hash `1e353efc...e5779d5`, GitHub release linked in the README)
- [x] IoU reported. (0.9434 on the 10 test images, `models/test_metrics.json`)
- [x] mAP@0.5 reported. (1.0, same file; trivially high with one book per image, see Objectives)
- [x] mAP@0.5:0.95 reported. (0.9233)
- [x] Precision and recall reported. (1.0 and 1.0 at IoU 0.5)

## 4. Measurement pipeline
- [x] A single command runs the image-to-measurement pipeline. (`python inference/measure.py <image>`)
- [x] Input image is undistorted. (`cv2.undistort` in `main`, skipped only with `--no-undistort`)
- [x] Book segmentation is performed. (U-Net, largest connected component)
- [x] Known-size reference object is detected. (checkerboard, `findChessboardCornersSB`, 9 x 7)
- [x] Width and height are output in millimetres. (homography from the board plane, e.g. `book_091`: 128.76 x 200.57 mm)
- [x] Output includes an overlay and confidence information. (`inference/outputs/<name>_overlay.jpg`, confidence printed)
- [x] Book not found handled: `python inference/measure.py calibration/images/IMG_20261108_175201.jpg` (checkerboard, no book) prints "BOOK NOT FOUND: no measurement reported."
- [ ] Reference not found handled: message exists in `inference/measure.py` but was not run on a failing image.
- [ ] Low-confidence measurements are refused. (code refuses below 0.9, a placeholder threshold; all test images scored about 0.99, so it never triggered and was not evaluated)

## 5. Accuracy testing
- [x] At least 10 instances tested. (10 raw photos, `book_091-100`)
- [x] Predictions compared with hand measurements (ruler or calliper). (flexible tape, three repeated readings; the 10 instances are photos of the same book, so one true size is used. Resolution roughly 1 to 2 mm.)
- [x] Per-instance table included, with raw outputs saved. (table in `docs/Objectives.md`, `measurement/accuracy_log.txt`, `measurement/accuracy_results.csv`)
- [x] Mean absolute error, mean percentage error, median and worst case calculated. (undistorted: MAE 5.91 mm, mean % error 3.58, median 4.17 mm, worst 16.09 mm)

## 6. Additional experiments
- [x] Raw versus undistorted measurements compared, with an explanation using my own numbers. (MAE 9.58 raw vs 5.91 undistorted; 15 of 20 errors improved; caveat that the model was trained on undistorted images, see Objectives)
- [x] MO-1 and MO-2 targets evaluated. (MO-1 MISSED: MAE 5.91 mm vs 2.0 mm target, worst case 16.09 mm vs 4.0 mm, cause unknown. MO-2 MET: median 1.80 s vs 5 s, `measurement/timing_log.txt`)
- [ ] At least one failure condition tested, and why I chose it. (NOT REACHED)
- [ ] Confidence-based refusal demonstrated, with the threshold and the number of refused images. (NOT REACHED: see section 4)

## 7. Documentation and submission
- [x] README: setup, how to run each stage, hardware, dataset summary. (`README.md`; weights release linked there with the SHA-256)
- [x] Plan (with decision record), Objectives and Definition of Done are in `docs/`. (`docs/Plan.md`, `docs/Objectives.md`, this file)
- [x] Random seeds, library versions and training config pinned. (seed 42 and config in `models/train.py` and `docs/Objectives.md`; `requirements.txt` re-frozen after installing torch and segmentation-models-pytorch)
- [x] Results and limitations documented honestly. (`docs/Objectives.md`, Limitations section)
- [x] Everything I did not finish is listed here, with the reason. (below)
- [x] Public GitHub repository with incremental commits on `dev-test01`, and a pull request into `main`. (PR: https://github.com/Sibgha-Nadeem/measure-it-book-segmentation/pull/1; `main` was created from the first commit so the PR shows the full history)
- [ ] Loom demonstration (5 minutes maximum) recorded. (pending)
- [ ] Four assessment questions (K1 to K4) answered in the demonstration. (pending)

## Not reached or partly done, and why

- **Item 8, failure condition (tilt, distance, lighting or frame edge): not done.** Dataset capture and masking took most of the 8 hours.
- **Item 9, refusal: partly done.** The code refuses below a confidence of 0.9 (a placeholder), but it was never triggered, the threshold was not tuned, and the number of refused images was not measured.
- **Reference-not-found handling: written, not tested** on a failing image. (Book-not-found was tested, see section 4.)
- **MO-1 missed.** The height bias of about +6.5 mm was not investigated.
- **Checkerboard pitch not measured** (20 mm used, unverified).
- **Ground truth** has a resolution of roughly 1 to 2 mm, set by a flexible tape read in inches.
- **Accuracy numbers need the raw photos**, which are not in the repository.

## Physical measurements used as ground truth

- Book height: 7.8 inches = 198.12 mm (three repeated readings gave the same value).
- Book width: 5.1 inches = 129.54 mm (three repeated readings gave the same value).
- Book thickness: 1 inch = 25.40 mm (three repeated readings gave the same value).
- Checkerboard: 10 x 8 squares, 9 x 7 inner corners, design size 200 x 160 mm, nominal pitch 20 mm. 
# Project Objectives

## Main objective
Build a computer vision system that measures the physical width and height of a book cover in millimetres from a smartphone photo.

## Specific objectives

1. **Camera calibration**
   - Capture at least 20 checkerboard images with the same camera, resolution and focus.
   - Use OpenCV to calculate the camera matrix and distortion coefficients.
   - Record the reprojection error and demonstrate undistortion.
2. **Dataset creation**
   - Capture 100 images of the book, undistort them and create segmentation masks.
   - Keep a fixed 70/20/10 train/validation/test split, committed before training.
3. **Segmentation model**
   - Train a PyTorch segmentation model (not Ultralytics YOLO, not Roboflow).
   - Evaluate on the test split with IoU, mAP@0.5, mAP@0.5:0.95, precision and recall.
4. **Measurement pipeline**
   - One command: load image, undistort, segment the book, detect the known-size reference in the same plane, convert to mm, output an overlay with width, height and confidence.
   - Handle "book not found" and "reference not found" cleanly.
5. **Measurement accuracy**
   - Compare at least 10 instances against hand measurements.
   - Report a per-instance table, MAE, mean percentage error, median and worst case.
6. **Robustness and evaluation**
   - Compare raw versus undistorted measurements.
   - Test one failure condition and measure how error changes. (NOT REACHED)
   - Refuse to report a measurement when it cannot be trusted. (PARTIAL: see Limitations)
7. **Documentation**
   - Document method, setup, results, limitations and usage. Keep incremental commits on `dev-test01`.

## Measurable objectives

| ID | Field | Entry |
|---|---|---|
| MO-1 | What is measured | Absolute error of book width and height, in mm, for each test instance. |
| | How | `measurement/accuracy.py` runs the pipeline on the 10 test photos and compares each output with the hand-measured book size (constants in the script). Every instance is listed, then mean, median and worst case. |
| | Target | Mean absolute error of 2.0 mm or less, and no single error above 4.0 mm, over at least 10 instances. |
| | Conditions | Vivo Y19s, main camera, 1836 x 4080 portrait, top-down photos, book and checkerboard in the same plane (checkerboard raised to the cover height on two stacked books), 10 different background and lighting setups. |
| | Not included | Images the system refuses to measure, and images where the book or reference is partly outside the frame. |
| MO-2 | What is measured | Time for the pipeline to measure one image on CPU: undistortion (when enabled), segmentation, checkerboard detection and measurement. |
| | How | Run the command 5 times (5 separate processes) on the same image. Report the median and the spread (min and max). The timer starts after the model is loaded and stops before the overlay is written, so model loading, image reading and overlay writing are not included. |
| | Target | Median of 5 seconds or less. |
| | Conditions | Local laptop, Intel Core i5-1340P, 16 GB RAM, Windows 11 Home, CPU only (no GPU), input 1836 x 4080. |
| | Not included | Model loading time, image reading, overlay writing. |

The calibration reprojection error is reported as a result, not as a pre-set target. The MO-1 and MO-2 targets were written before the measurement evaluation was run.

## Results summary

| Objective | Target | Result | Status |
|---|---|---|---|
| MO-1 | MAE 2.0 mm or less, worst case 4.0 mm or less | MAE 5.91 mm, worst case 16.09 mm (undistorted, 10 test photos) | MISSED |
| MO-2 | Median 5 s or less | Median 1.80 s (min 1.20, max 2.49) | MET |

## Dataset

- Object: one book (front cover). Cover 198.12 x 129.54 mm, thickness 25.40 mm.
- 105 portrait photos were taken (1836 x 4080). The 5 weakest were set aside (the 3 lowest sharpness scores, plus one of each of 2 near-duplicate pairs), leaving 100. They cover 10 background and lighting setups, with 10 layout patterns each.
- In every photo the checkerboard lies beside the book, raised to the cover height by two stacked books of the same thickness, so both are in the same plane.
- Images were renamed `book_001` to `book_100` (mapping in `dataset/splits/name_map.csv`) and undistorted with `calibration/camera_params.npz` (`dataset/prepare_images.py`).
- Masks: made with `dataset/annotate.py` (click the 4 cover corners, the tool fills the polygon). 100 masks exist. The 8 smallest masks (area under 0.6 times the median) were viewed and no problems were found.
- Split: groups of 10 photos in capture order, with 7 train, 2 validation and 1 test per group chosen at random (seed 42). `book_001-070` train, `book_071-090` validation, `book_091-100` test. `dataset/splits/split.txt` was committed before any images existed. The mapping to the final images was fixed before training.
- Labelling tool: own OpenCV script (`dataset/annotate.py`).
- Full-resolution originals and the undistorted images are not in the repository (size, gitignored). Masks and the resized 512 x 1152 training copy are.

## Calibration results

Camera: Vivo Y19s, main rear camera, 1836 x 4080 (portrait), no zoom, JPEG.

Board: 9 x 7 inner corners (10 x 8 squares). Nominal square pitch 20 mm (design size 200 x 160 mm). **The printed pitch was not independently measured. 20 mm is used and is unverified.**

Detector: `cv2.findChessboardCornersSB`. Raw output: `calibration/calibration_log.txt`.

| Run | Images | k3 | RMS (px) | fx | fy | cx | cy | k1 | k2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22 | free | 0.9217 | 3381 | 3396 | 932 | 2060 | 0.221 | -0.969 |
| 2 | 22 | fixed to 0 | 0.9244 | 3385 | 3400 | 932 | 2061 | 0.197 | -0.631 |
| 3 (final) | 36 | fixed to 0 | 0.9680 | 3385 | 3399 | 936 | 2059 | 0.176 | -0.499 |

Why run 3 is final: the first 22 photos were mostly face-on and centred. I added 14 portrait photos with the board near the frame edges and with stronger tilt. RMS rose slightly, which I read as the new photos testing the lens where it was previously untested. fx changed by about 0.01%, which suggests the camera matrix is stable. k3 was fixed to 0 because freeing it changed RMS by only 0.003 px and gave a large, unstable coefficient (+1.29).

### Undistortion straightness test

Run on all 36 calibration images, using the detected checkerboard corners (`calibration/straightness_check.py`, output `calibration/straightness_log.txt`).

| Measurement | Raw | Undistorted |
|---|---:|---:|
| Mean line-fit RMS (px) | 0.671 | 0.446 |
| Mean worst deviation (px) | 1.919 | 1.471 |

All 36 images improved. This test uses the same images as the calibration, so it is not an independent validation. The visual demo (`calibration/undistort_demo.jpg`) showed only a subtle change.

## Segmentation model

- Model: U-Net with a ResNet34 ImageNet encoder (`segmentation-models-pytorch`), trained on my own images. Reason: a U-Net is simple to explain and train, works well on small datasets with a pretrained encoder, and a single book per image needs only a binary mask.
- Training: BCE + Dice loss, seed 42, 40 epochs, batch 4, learning rate 3e-4, input 512 x 1152. Trained on `book_001-070`. The checkpoint was chosen by validation IoU on `book_071-090` (best 0.9324 at epoch 25). The test photos were never loaded during training.
- Training hardware: Kaggle, Tesla T4. Versions: torch 2.11.0+cu130, segmentation-models-pytorch 0.6.0, opencv 4.14.0, numpy 2.1.3, timm 1.0.30.
- Weights: `models/best.pt`, SHA-256 `1e353efcea9cc0785b50441c707888d0c23dbe46dac0ec1bf3174eb50e5779d5` (verified on my laptop after download).

### Test results (10 images, `book_091-100`, evaluated once)

| Metric | Value |
|---|---:|
| Mean mask IoU | 0.9434 |
| mAP@0.5 | 1.0 |
| mAP@0.5:0.95 | 0.9233 |
| Precision@0.5 | 1.0 |
| Recall@0.5 | 1.0 |

Per-image IoU ranged from 0.9015 (`book_091`) to 0.9726 (`book_094`). Raw output: `models/test_metrics.json`.

Method: the largest connected component of the thresholded mask is the single instance, the mean mask probability inside it is its confidence, and IoU is thresholded from 0.5 to 0.95. With one book per image, mAP@0.5, precision and recall are 1.0 whenever every mask has IoU above 0.5, so they say little with only 10 images. All confidences were about 0.99, so they do not rank masks by quality. One book only, so no claim is made about other books.

## Measurement accuracy (MO-1): MISSED

Ground truth: book cover 198.12 mm high and 129.54 mm wide, read with a flexible tape (three repeated readings gave the same value). The reading was taken in inches and converted, so the true size could differ by roughly 1 to 2 mm (my estimate). Errors of that size cannot be resolved by this test.

Method: `measurement/accuracy.py` measured the raw phone photos of `book_091-100` (never used for training or checkpoint selection), once with undistortion and once without. Raw outputs: `measurement/accuracy_log.txt`, `measurement/accuracy_results.csv`. The 10 instances are 10 photos of the same book in different setups.

Error = measured minus true (mm).

| Book | Undistorted width (error) | Undistorted height (error) | Raw width (error) | Raw height (error) |
|---|---|---|---|---|
| 091 | 128.85 (-0.69) | 200.49 (+2.37) | 129.58 (+0.04) | 207.60 (+9.48) |
| 092 | 134.43 (+4.89) | 203.67 (+5.55) | 140.75 (+11.21) | 207.27 (+9.15) |
| 093 | 140.80 (+11.26) | 204.73 (+6.61) | 147.26 (+17.72) | 210.40 (+12.28) |
| 094 | 130.41 (+0.87) | 214.21 (+16.09) | 133.12 (+3.58) | 224.01 (+25.89) |
| 095 | 132.30 (+2.76) | 208.87 (+10.75) | 135.62 (+6.08) | 215.32 (+17.20) |
| 096 | 126.08 (-3.46) | 211.81 (+13.69) | 129.20 (-0.34) | 222.29 (+24.17) |
| 097 | 127.35 (-2.19) | 209.14 (+11.02) | 131.77 (+2.23) | 218.46 (+20.34) |
| 098 | 128.26 (-1.28) | 196.36 (-1.76) | 133.57 (+4.03) | 201.27 (+3.15) |
| 099 | 121.22 (-8.32) | 195.45 (-2.67) | 122.31 (-7.23) | 199.76 (+1.64) |
| 100 | 120.50 (-9.04) | 201.06 (+2.94) | 122.56 (-6.98) | 206.92 (+8.80) |

| Mode | MAE (mm) | Median abs. error (mm) | Worst (mm) | Mean % error |
|---|---:|---:|---:|---:|
| Undistorted | 5.91 | 4.17 | 16.09 | 3.58 |
| Raw | 9.58 | 8.01 | 25.89 | 5.63 |

Mean % error is the mean over all 20 width and height errors. With undistortion, 10 of the 20 individual errors exceed 4.0 mm. Both parts of MO-1 were missed.

Observations: with undistortion, heights are on average about 6.5 mm too large and widths about 0.5 mm too small, so a single wrong scale does not explain the error (a wrong checkerboard pitch would scale both). The cause was not investigated. Candidates that were not tested: the unverified 20 mm pitch, an uneven print or board, corner placement from the mask, and the board not being exactly level with the cover.

### Raw versus undistorted
Undistortion reduced MAE from 9.58 to 5.91 mm. 15 of the 20 width and height errors got smaller (9 of 10 heights, 6 of 10 widths). Caveat: the model was trained on undistorted images, so part of the raw error may come from worse segmentation on raw photos, not from lens distortion alone.

## Speed (MO-2): MET

5 runs on `book_091.jpg` with `--no-undistort`: 1.20, 1.26, 1.80, 2.40, 2.49 s. Median 1.80 s, min 1.20 s, max 2.49 s (target: median 5 s or less). Raw output: `measurement/timing_log.txt`. The output was identical on every run. The test image was already undistorted, so a raw photo takes somewhat longer. In the accuracy runs, per-image times were 1.3 to 3.0 s with undistortion and 2.6 to 2.7 s without.

## Limitations

- MO-1 was missed, and the cause of the height bias is unknown.
- The checkerboard pitch (20 mm) is unverified, and the print had ink streaks and was not on a rigid backing.
- One book and 10 test photos, so no generalisation claim.
- Ground truth is limited by a flexible tape read in inches (roughly 1 to 2 mm).
- Confidence-based refusal exists in the code (threshold 0.9, a placeholder chosen without tuning), but every test image scored about 0.99, so it never triggered and its cost in refused images was not measured. The "book not found" and "reference not found" messages exist in the code but were not tested on a failing image (see Definition of Done).
- No failure-condition experiment (tilt, distance, lighting or frame edge) was run.
- Reprojection error in pixels (0.968) is not a direct measure of accuracy in mm.
- The accuracy script needs the raw originals in `dataset/raw`, which are not in the repository, so the accuracy numbers cannot be reproduced without them.

## Hardware

- Local (measurement, timing): 13th Gen Intel Core i5-1340P (1.90 GHz), 16 GB RAM (16,016 MB reported by Windows), Windows 11 Home 10.0.26200, no CUDA-capable GPU.
- Training: Kaggle, Tesla T4.
- Camera: Vivo Y19s, main rear camera, 1836 x 4080.
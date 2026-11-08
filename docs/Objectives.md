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
   - Test one failure condition and measure how error changes.
   - Refuse to report a measurement when it cannot be trusted.

7. **Documentation**
   - Document method, setup, results, limitations and usage. Keep incremental commits on `dev-test01`.

## Measurable objectives

| ID | Field | Entry |
|---|---|---|
| MO-1 | What is measured | Absolute error of book width and height, in mm, for each test instance. |
| | How | A script compares the system's output with hand-measured values stored in a CSV file. Every instance is listed, then mean, median and worst case. |
| | Target | Mean absolute error of 2.0 mm or less, and no single error above 4.0 mm, over at least 10 instances. |
| | Conditions | Local laptop, Intel Core i5-1340P, 16 GB RAM, Windows 11 Home, CPU only (no GPU), input 1836 × 4080. |
| | Not included | Images the system refuses to measure, and images where the book or reference is partly outside the frame. |
| MO-2 | What is measured | Time to run the full command on one image (undistort, segment, measure, write overlay), on CPU. |
| | How | Run the command 5 times on the same image. Report the median and the spread (min and max). The first run is reported separately because it includes loading the model. |
| | Target | Median of 5 seconds or less. |
| | Conditions | Local laptop, Intel Core i5-1340P, 16 GB RAM, Windows 11 Home, no GPU, input 1836 × 4080. |
| | Not included | The first run after starting Python, and the time to save the file to disk if reported separately. |

The calibration reprojection error is reported as a result, not as a pre-set target. MO-1 and MO-2 targets were set before measurement evaluation.

## Calibration results

Camera: Vivo Y19s, main rear camera, 1836 × 4080 (portrait), no zoom, JPEG.

Board: 9 × 7 inner corners (10 × 8 squares). Nominal square pitch 20 mm. Previously reported full board dimensions: 200 × 160 mm. Actual printed pitch still needs to be verified before final metric measurements.

Detector: `cv2.findChessboardCornersSB`. Raw output: `calibration/calibration_log.txt`.

| Run | Images | k3 | RMS (px) | fx | fy | cx | cy | k1 | k2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22 | free | 0.9217 | 3381 | 3396 | 932 | 2060 | 0.221 | -0.969 |
| 2 | 22 | fixed to 0 | 0.9244 | 3385 | 3400 | 932 | 2061 | 0.197 | -0.631 |
| 3 (final) | 36 | fixed to 0 | 0.9680 | 3385 | 3399 | 936 | 2059 | 0.176 | -0.499 |

Why run 3 is final: the first 22 photos were mostly face-on and centred. I added 14 portrait photos with the board near the frame edges and with stronger tilt. RMS rose slightly, which I read as the new photos testing the lens where it was previously untested. fx changed by approximately 0.01%, which suggests the camera matrix is stable. k3 was fixed to 0 because freeing it changed RMS by only 0.003 px and gave a large, unstable coefficient (+1.29).

### Undistortion straightness test

The straightness test was run on all 36 calibration images using the detected checkerboard corners.

| Measurement | Raw | Undistorted |
|---|---:|---:|
| Mean line-fit RMS (px) | 0.671 | 0.446 |
| Mean worst deviation (px) | 1.919 | 1.471 |

All 36 images improved after undistortion. Mean line-fit RMS decreased by approximately 33.5%, and mean worst deviation decreased by approximately 23.3%.

The straightness test uses the same images used for calibration, so it is not an independent validation. Visual inspection of the undistortion demo also showed straight lines after correction, although the visual difference was subtle.

### Limitations

The checkerboard print had ink streaks and was not mounted on a rigid backing. The previously reported board dimensions and nominal square pitch have not yet been independently verified for final metric measurement. Reprojection error in pixels is not a direct measurement of book accuracy in millimetres. Nine landscape images were excluded because calibration uses a single image size.

## Hardware

CPU: 13th Gen Intel Core i5-1340P (1.90 GHz), used for local measurement and timing runs.

GPU: Local computer has no CUDA-capable NVIDIA GPU. Training hardware is to be confirmed after choosing the training machine.

RAM: 16 GB (16,016 MB reported by Windows). OS: Windows 11 Home, version 10.0.26200.

Camera: Vivo Y19s, main rear camera, 1836 × 4080.
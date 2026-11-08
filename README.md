# Measure It: book cover measurement in millimetres

Measures the width and height of a book cover from one phone photo, using a camera calibration, a U-Net segmentation model and a checkerboard of known size lying in the same plane as the cover.

Camera: Vivo [Y19s], main rear camera, 1836 x 4080 portrait. Python 3.11.

## Setup

Run in cmd from the project folder:

    python -m venv .venv
    .venv\Scripts\activate.bat
    pip install -r requirements.txt

Download `best.pt` from the release below and put it in `models/`.

- Weights: [GitHub release link]
- SHA-256: `1e353efcea9cc0785b50441c707888d0c23dbe46dac0ec1bf3174eb50e5779d5`

Check the hash on Windows with `certutil -hashfile models\best.pt SHA256`.

## Run (image to millimetres)

    python inference\measure.py path\to\photo.jpg

It prints `width_mm`, `height_mm` and `confidence`, and saves an overlay in `inference/outputs/`. Add `--no-undistort` for an image that is already undistorted.

The photo must be taken top-down with the Vivo camera at 1836 x 4080, with the book and the printed 9 x 7 checkerboard both in view and at the same height (I raised the board with two stacked books of the same thickness as the book). If no book is found it prints `BOOK NOT FOUND`, and if no checkerboard is found it prints `REFERENCE NOT FOUND`, and in both cases it reports no measurement.

## Stages

| Stage | Where | How to run |
|---|---|---|
| Calibration | `calibration/` | `python calibration\calibrate.py` (log in `calibration/calibration_log.txt`) |
| Undistortion demo | `calibration/` | `python calibration\undistort_demo.py <image>` |
| Dataset | `dataset/` | scripts: `prepare_images.py`, `annotate.py`, `make_kaggle_zip.py` |
| Training | `models/train.py` | run on Kaggle (Tesla T4), seed 42 |
| Test evaluation | `models/evaluate.py` | `python models\evaluate.py` (output in `models/test_metrics.json`) |
| Accuracy test | `measurement/accuracy.py` | needs the raw photos in `dataset/raw` (not in the repo) |

## Dataset summary

- Object: one book (front cover), 198.12 x 129.54 mm, 25.40 mm thick.
- Images: 100 portrait photos (1836 x 4080), taken by me. 105 were taken and the 5 weakest were set aside. They cover 10 background and lighting setups with 10 layouts each.
- Labelling: my own OpenCV script (`dataset/annotate.py`): click the 4 cover corners and it fills the mask.
- Split: fixed in `dataset/splits/split.txt`, committed before any image existed. `book_001-070` train, `book_071-090` validation, `book_091-100` test. The test photos were used only once, by `models/evaluate.py`.
- The original and undistorted photos are not in the repository (size). Masks and the resized training copy are.

## Results

Full tables and the method are in `docs/Objectives.md`.

- Calibration: 36 images, RMS reprojection error 0.968 px.
- Segmentation on 10 test images: mean IoU 0.9434, mAP@0.5 1.0, mAP@0.5:0.95 0.9233, precision 1.0, recall 1.0. With one book per image and 10 test photos, the 1.0 values say little.
- Measurement accuracy (MO-1): **target missed.** On the 10 test photos with undistortion, MAE 5.91 mm, median 4.17 mm, worst case 16.09 mm, mean error 3.58%. Without undistortion, MAE 9.58 mm.
- Speed (MO-2): met. Median 1.80 s over 5 runs (1.20 to 2.49 s) on CPU, target 5 s.

## Hardware

- Measurement and timing: Intel Core i5-1340P (1.90 GHz), 16 GB RAM, Windows 11 Home, no GPU.
- Training: Kaggle, Tesla T4.

## Limitations and what was not done

- MO-1 was missed, and I did not find the cause of the height bias (about +6.5 mm on average).
- The printed checkerboard pitch (20 mm) was not measured, so it is unverified.
- One book only, so there is no claim that it works on other books.
- The confidence threshold of 0.9 is a placeholder, and it never triggered on the test photos.
- No failure-condition experiment was run, and "reference not found" was not tested on a real failing image.
- The accuracy script needs the raw photos, which are not in the repository.

Details: `docs/Definition_of_Done.md`.
# Measure It — Project Plan

## Project goal
Build a computer vision system that measures the front-cover width and height of a physical book in millimetres using camera calibration, image segmentation and a known-size reference object.

## Selected object
A physical book. (Novel: No! I dont need reading glasses!)

Approximate ground-truth dimensions:
- Height: 198.12 mm
- Width: 129.54 mm

These measurements will be verified with a physical measuring instrument.

## Technical approach

1. Camera calibration using a printed checkerboard and OpenCV.
2. Capture at least 20 calibration images using a smartphone.
3. Undistort all object images using saved camera parameters.
4. Capture at least 100 original images of the book.
5. Annotate book masks using a segmentation annotation tool.
6. Create a fixed 70/20/10 train/validation/test split.
7. Train a U-Net segmentation model using PyTorch.
8. Evaluate segmentation on the untouched test set.
9. Develop an image-to-millimetres inference pipeline.
10. Compare system measurements against physical measurements.
11. Evaluate undistortion and one selected failure condition.
12. Document limitations and confidence-based refusal.

## Proposed time allocation

- Setup and planning: 45 minutes
- Camera calibration: 60 minutes
- Dataset collection and annotation: 90 minutes
- Model training and evaluation: 120 minutes
- Measurement pipeline: 90 minutes
- Accuracy experiments: 60 minutes
- Documentation and submission: 15 minutes

## Tools
- Python
- OpenCV
- PyTorch
- NumPy
- CVAT or Label Studio
- Git and GitHub
- Kaggle GPU

## Known limitations and planned skips
The project prioritises requirements 1–4 before advanced experiments. Any unfinished requirements will be clearly reported. No experimental results will be fabricated.

## Current status
Planning and equipment preparation completed. Implementation has not yet started.
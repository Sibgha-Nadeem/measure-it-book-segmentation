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

## Decision record

**Approach taken:** a U-Net (ResNet34 encoder, PyTorch) trained on my own 70 training images to find the book cover, then a homography from the checkerboard to millimetres. The checkerboard lies beside the book on two stacked books of the same thickness, so it is level with the cover. This is what makes the homography valid for the cover.

**Approach rejected:** a single pixels-per-mm scale from the reference. It is only right if the camera is exactly square-on and the reference is at the same height as the object. Neither is guaranteed with hand-held photos.

**What rejecting it cost:** more code (corner detection and a homography), a strict capture setup (board and cover must be coplanar), and a pipeline that fails if the board is not found. The result was still not accurate enough: MO-1 was missed (MAE 5.91 mm against a 2.0 mm target) and the cause of the height bias was not investigated.

**What changed from the plan:** items 8 (failure condition) and 9 (refusal threshold) were not reached, because dataset capture and masking took most of the 8 hours.
# Project Objectives

## Main Objective
Develop a computer vision system that measures the physical dimensions of a book in millimetres from a smartphone image.

## Specific Objectives

1. **Camera Calibration**
   - Capture at least 20 checkerboard calibration images.
   - Use OpenCV to calculate the camera matrix and distortion coefficients.
   - Record the calibration reprojection error.

2. **Dataset Creation**
   - Capture 100 images of the selected book.
   - Undistort the images using the camera calibration parameters.
   - Create segmentation masks for the book.
   - Maintain a fixed 70/20/10 training, validation and test split.

3. **Segmentation Model**
   - Train a segmentation model using PyTorch.
   - Do not use Ultralytics YOLO or a Roboflow-trained model.
   - Evaluate using IoU, mAP@0.5, mAP@0.5:0.95, precision and recall.

4. **Measurement Pipeline**
   - Create a single-command image-to-measurement pipeline.
   - Undistort the input image.
   - Segment the book.
   - Detect a known-size reference object in the same plane.
   - Calculate book width and height in millimetres.
   - Display an overlay with measurements and confidence.
   - Handle missing book or reference-object detections.

5. **Measurement Accuracy**
   - Compare predictions against physical measurements.
   - Test at least 10 measurement instances.
   - Calculate mean absolute error (MAE) and mean percentage error.

6. **Robustness and Evaluation**
   - Compare measurements from raw and undistorted images.
   - Test at least one failure condition.
   - Implement confidence-based refusal when measurements are unreliable.

7. **Documentation**
   - Document the methodology, setup, results, limitations and usage instructions.
   - Maintain a public GitHub repository with incremental commits.
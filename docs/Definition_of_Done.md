# Definition of Done

The project will be considered complete when the following requirements have been implemented, tested and documented.

## 1. Camera Calibration
- [ ] At least 20 checkerboard images captured.
- [ ] Camera matrix and distortion coefficients saved.
- [ ] Reprojection error calculated and reported.
- [ ] Undistortion demonstrated on sample images.

## 2. Dataset
- [ ] At least 70 book images captured (target: 100).
- [ ] Images undistorted.
- [ ] Segmentation masks created and checked.
- [ ] Fixed 70/20/10 split documented and committed before training.
- [ ] Test images kept separate from training and validation.

## 3. Segmentation Model
- [ ] A segmentation model trained using PyTorch.
- [ ] Model weights saved.
- [ ] IoU reported.
- [ ] mAP@0.5 reported.
- [ ] mAP@0.5:0.95 reported.
- [ ] Precision and recall reported.

## 4. Measurement Pipeline
- [ ] A single command runs the image-to-measurement pipeline.
- [ ] Input image is undistorted.
- [ ] Book segmentation is performed.
- [ ] Known-size reference object is detected.
- [ ] Width and height are output in millimetres.
- [ ] Output includes an image overlay and confidence information.
- [ ] Missing book or reference detections are handled safely.

## 5. Accuracy Testing
- [ ] At least 10 measurement instances tested.
- [ ] Predicted measurements compared with physical measurements.
- [ ] Per-instance results table included.
- [ ] Mean absolute error calculated.
- [ ] Mean percentage error calculated.

## 6. Additional Experiments
- [ ] Raw versus undistorted image measurements compared.
- [ ] At least one speed or accuracy target selected and evaluated.
- [ ] At least one failure condition tested.
- [ ] Confidence-based refusal demonstrated.

## 7. Documentation and Submission
- [ ] README includes installation and usage instructions.
- [ ] Plan, Objectives and Definition of Done are included.
- [ ] Results and limitations documented honestly.
- [ ] Public GitHub repository contains incremental commits on `dev-test01`.
- [ ] Short Loom demonstration (maximum 5 minutes) recorded.
- [ ] Four assessment questions addressed in the demonstration.

**Important:** A checkbox should only be marked complete after the requirement has actually been implemented and verified. No results should be fabricated.